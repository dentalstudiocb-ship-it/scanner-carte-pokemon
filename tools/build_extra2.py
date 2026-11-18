#!/usr/bin/env python3
"""Extend the supplementary visual DB (data/extra-en.json + data/fp2-extra-en.bin), second pass.

Keeps the records written by build_extra.py (same order, prices refreshed for `ptcg:` records that had none)
and appends every English TCGdex card (live /v2/en/cards list, TCG Pocket excluded) that is still in neither
data/ids-en.json nor the extra DB, using images from:
  * images.scrydex.com (CORS *) - card lists scraped from the scrydex.com expansion pages
    (trainer kits tk3a..tk10b, McDonald's mcd14..mcd24, mep/mee/sve/svp/swshp/hsp/det1/miscp/wb1/ecard2 ...)
  * MANUAL sources below for the few cards scrydex does not have (flagged "nocors": true when the host
    sends no Access-Control-Allow-Origin header).
Matching: same set mapping + normalised number + compatible name (names are checked, never number alone).

Fingerprints: decode + to_rgb + first Pillow resize to 630x880 on CPU exactly as build_fp2.card630(); the
numeric stages (8x7 box means, luma grid, high-pass, z-norm, chroma, quantisation) run batched in torch
float64 on CUDA when available (falls back to build_fp2.desc_from_base on CPU).  --verify N checks the GPU
path against data/fp2-en.bin for N cached main-DB images (max abs byte diff must be <= 1).

Usage:  <cuda python> tools/build_extra2.py [--verify 200] [--no-prices]
"""
import argparse
import hashlib
import html
import io
import json
import os
import re
import sys
import time
import difflib
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_fp2 as fp  # noqa: E402

DATA = fp.DATA
SCRATCH = r"C:\Users\Fabio\AppData\Local\Temp\claude\d395c8141171058b\be555625-db4a-4cfa-8cf3-32e7aeb800e3\scratchpad"
CACHE = os.path.join(SCRATCH, "extracache2")
SRC = os.path.join(CACHE, "_src")
TD = "https://api.tcgdex.net/v2/en"
PTCG = "https://api.pokemontcg.io/v2"
SCRY_IMG = "https://images.scrydex.com/pokemon/{}/{}"
SCRY_PLACEHOLDER = "f294d6fb"           # md5 prefix of the "no image" PNG scrydex serves for unknown ids
POCKET = re.compile(r"^(A\d|B\d|P-A)")
NCPU = max(1, (os.cpu_count() or 2) // 2)
NET = 8                                 # max concurrent requests

# tcgdex set id -> scrydex expansion id
SETMAP = {
    "tk-dp-m": "tk3a", "tk-dp-l": "tk3b", "tk-hs-r": "tk4a", "tk-hs-g": "tk4b",
    "tk-bw-z": "tk5a", "tk-bw-e": "tk5b", "tk-xy-sy": "tk6a", "tk-xy-n": "tk6b",
    "tk-xy-w": "tk7a", "tk-xy-b": "tk7b", "tk-xy-latio": "tk8a", "tk-xy-latia": "tk8b",
    "tk-xy-su": "tk9a", "tk-xy-p": "tk9b", "tk-sm-l": "tk10a", "tk-sm-r": "tk10b",
    "2014xy": "mcd14", "2015xy": "mcd15", "2017sm": "mcd17", "2018sm": "mcd18",
    "2023sv": "mcd23", "2024sv": "mcd24",
    "mep": "mep", "mee": "mee", "sve": "sve", "svp": "svp", "swshp": "swshp", "det1": "det1",
    "miscp": "miscp", "ex5.5": "wb1", "ecard2": "ecard2", "hgssp": "hsp",
}
# per-card overrides: tcgdex id -> scrydex card id (number differs on scrydex)
CARDMAP = {"mep-Museum": "mep-1000"}
# cards scrydex lacks.  "bulba:<File>" = Bulbagarden Archives file (300px thumb, resolved through the
# MediaWiki API; recorded with "nocors": true, see below), "scry:<id>" = scrydex card.
MFB_POKEMON = ("Bulbasaur Ivysaur Oddish Gloom Exeggcute Exeggutor Scyther Charmander Charmeleon Vulpix Ninetales "
               "Growlithe Arcanine Magmar Pikachu Raichu Magnemite Magneton Voltorb Electrode Electabuzz Squirtle "
               "Wartortle Poliwag Poliwhirl Magikarp Gyarados Lapras").split()
MANUAL = {
    "xya-24a": "bulba:MManectricEXPhantomForces24a.jpg",
    "xya-28a": "bulba:JolteonEXGenerations28a.jpg",
    "xya-54a": "bulba:ZygardeEXFatesCollide54a.jpg",
    "xya-55a": "bulba:MLucarioEXFuriousFists55a.jpg",
    "xya-92a": "bulba:TrainerMailRoaringSkies92a.jpg",
    "xya-107a": "bulba:ProfessorSycamoreBREAKpoint102a.jpg",   # file misnamed on Bulbapedia; card reads 107a/122
    "svp-500": "bulba:TerapagosFriendsSVPPromo.jpg",
}
# My First Battle Trainers/Energy are reprints of the SVI / SVE cards (Bulbapedia deck lists use those scans):
# same artwork, so the reprint picture is used and the record carries "proxy": <card the image shows>.
MFB_PROXY = {"Potion": "sv1-188", "Switch": "sv1-194", "Grass Energy": "sve-1", "Fire Energy": "sve-2",
             "Water Energy": "sve-3", "Lightning Energy": "sve-4"}
BULBA_API = "https://archives.bulbagarden.net/w/api.php"


def bulba_thumbs(files):
    """Bulbagarden Archives file names -> 300px thumbnail URL (None when the file does not exist)."""
    out = {}
    files = sorted(set(files))
    for i in range(0, len(files), 40):
        q = "|".join("File:" + f for f in files[i:i + 40])
        raw = get(f"{BULBA_API}?action=query&format=json&prop=imageinfo&iiprop=url&iiurlwidth=300&titles="
                  + urllib.parse.quote(q))
        for p in json.loads(raw)["query"]["pages"].values():
            ii = p.get("imageinfo")
            out[p["title"][5:].replace(" ", "_")] = ii[0]["thumburl"] if ii else None
    return {f: out.get(f.replace(" ", "_")) for f in files}

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (card-db-builder)"


def get(url, timeout=60, tries=6, ok404=False):
    for i in range(tries):
        try:
            r = S.get(url, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                wait = int(r.headers.get("Retry-After", 0) or 0) or min(60, 2 ** i * 3)
                time.sleep(wait)
                continue
            if r.status_code == 404 and ok404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(min(60, 2 ** i * 2))
    raise RuntimeError(f"giving up on {url}")


def cached(name, url, **kw):
    p = os.path.join(SRC, name)
    if not os.path.exists(p) or os.path.getsize(p) < 2:
        data = get(url, **kw)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
    with open(p, "rb") as f:
        return f.read()


def nn(x):
    x = urllib.parse.unquote(str(x))
    m = re.match(r"^([A-Za-z]*)0*(\d+)(.*)$", x)
    return (m.group(1).upper() + m.group(2) + m.group(3).lower()) if m else x.upper()


def nm(s):
    s = s.lower().replace("’", "'").replace("☆", "star").replace("★", "star").replace("δ", "delta").replace("&", "and")
    s = re.sub(r"^basic ", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def name_ok(a, b):
    a, b = nm(a), nm(b)
    return a == b or a.startswith(b) or b.startswith(a) or difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


# ---------------------------------------------------------------- sources
def missing_cards():
    with open(os.path.join(DATA, "ids-en.json"), encoding="utf-8") as f:
        have = set(json.load(f))
    with open(os.path.join(DATA, "extra-en.json"), encoding="utf-8") as f:
        have |= {r["id"] for r in json.load(f)}
    lst = json.loads(get(f"{TD}/cards", timeout=180))
    miss = [c for c in lst if c["id"] not in have and not POCKET.match(c["id"])]
    print(f"live TCGdex list: {len(lst)} cards; not covered (excl. Pocket): {len(miss)}", flush=True)

    def detail(c):
        q = urllib.parse.quote(c["id"], safe="")
        return json.loads(cached(f"tdcards/{q}.json", f"{TD}/cards/{q}"))
    with ThreadPoolExecutor(NET) as ex:
        det = list(ex.map(detail, miss))
    det = [d for d in det if d.get("set", {}).get("id") != "tcgp" and "Pocket" not in d.get("set", {}).get("name", "")]
    sets = {}
    for sid in sorted({d["set"]["id"] for d in det}):
        sets[sid] = json.loads(cached(f"tdsets/{urllib.parse.quote(sid, safe='')}.json",
                                      f"{TD}/sets/{urllib.parse.quote(sid, safe='')}"))
    return det, sets


def scrydex_expansions(wanted):
    idx = html.unescape(cached("scrydex_expansions.html", "https://scrydex.com/pokemon/expansions").decode("utf-8"))
    links = {m.group(2): m.group(1) for m in re.finditer(r'href="/pokemon/expansions/([^"/]+)/([^"/]+)"', idx)}
    out = {}

    def one(e):
        t = html.unescape(cached(f"scrydex/{e}.html", f"https://scrydex.com/pokemon/expansions/{links[e]}/{e}").decode("utf-8"))
        cards, seen = [], set()
        for m in re.finditer(r'href="/pokemon/cards/[^/"]+/(' + re.escape(e) + r'-([^"?]+))\?variant=', t):
            if m.group(1) in seen:
                continue
            seen.add(m.group(1))
            n = re.search(r"<span[^>]*>([^<]+)</span>", t[m.end():m.end() + 3000])
            name = re.sub(r"\s+#\S+$", "", n.group(1).strip()) if n else ""
            cards.append({"id": m.group(1), "number": m.group(2), "name": name})
        return e, cards
    with ThreadPoolExecutor(4) as ex:
        for e, c in ex.map(one, [w for w in wanted if w in links]):
            out[e] = c
    return out


def scry_probe(cid):
    """True when scrydex serves a real picture for this card id (not its placeholder)."""
    data = get(SCRY_IMG.format(cid, "small"))
    return hashlib.md5(data).hexdigest()[:8] != SCRY_PLACEHOLDER


# ---------------------------------------------------------------- prices
def refresh_prices(recs):
    todo = [r for r in recs if r["id"].startswith("ptcg:") and not r.get("cardmarket")]
    print(f"retrying cardmarket prices for {len(todo)} ptcg records ...", flush=True)

    def one(r):
        pid = r["id"][5:]
        try:
            raw = get(f"{PTCG}/cards/{urllib.parse.quote(pid)}?select=id,cardmarket", timeout=120, tries=7, ok404=True)
        except Exception as e:  # noqa: BLE001
            return r, f"error {type(e).__name__}"
        if raw is None:
            return r, "404"
        cm = json.loads(raw).get("data", {}).get("cardmarket")
        if not cm:
            return r, "no cardmarket entry"
        p = cm.get("prices", {})
        r["cardmarket"] = {"trend": p.get("trendPrice"), "avg30": p.get("avg30"), "avg7": p.get("avg7"),
                           "avg1": p.get("avg1"), "low": p.get("lowPrice"), "avg": p.get("averageSellPrice"),
                           "reverseTrend": p.get("reverseHoloTrend"), "url": cm.get("url"),
                           "updated": cm.get("updatedAt")}
        return r, "ok"
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, todo))
    stat = {}
    for r, s in res:
        stat.setdefault(s, []).append(r["id"])
    for s, ids in stat.items():
        print(f"  prices {s}: {len(ids)} {ids if s != 'ok' else ''}", flush=True)
    return stat


# ---------------------------------------------------------------- images + fingerprints
def img_file(url):
    return os.path.join(CACHE, re.sub(r"[^A-Za-z0-9._-]", "_", url.split("://", 1)[1]))


def download(url):
    p = img_file(url)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    try:
        data = get(url)
    except Exception as e:  # noqa: BLE001
        print(f"  download failed {url}: {e}", flush=True)
        return None
    with open(p, "wb") as f:
        f.write(data)
    return p


def load630(path):
    """CPU stage, identical to build_fp2.card630: decode, flatten alpha, first Pillow BILINEAR resize."""
    return np.asarray(fp.card630(Image.open(path)), dtype=np.uint8)


class GpuFP:
    def __init__(self):
        import torch
        self.t = torch
        torch.set_num_threads(NCPU)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"fingerprint numeric stage on: {self.dev} "
              f"({torch.cuda.get_device_name(0) if self.dev.type == 'cuda' else 'cpu'})", flush=True)

    def desc(self, imgs):
        """imgs: (B, 880, 630, 3) uint8 -> (B, 396) int8, same maths as build_fp2.desc_from_base(base_of())."""
        t = self.t
        a = t.from_numpy(imgs).to(self.dev, non_blocking=True).to(t.float64)
        B = a.shape[0]
        b = a.reshape(B, fp.BH, 8, fp.BW, 7, 3).mean(dim=(2, 4))                      # (B,110,90,3)
        y = b[..., 0] * 0.299 + b[..., 1] * 0.587 + b[..., 2] * 0.114
        g = y.reshape(B, fp.GH, fp.BH // fp.GH, fp.GW, fp.BW // fp.GW).mean(dim=(2, 4))  # (B,22,15)
        p = t.nn.functional.pad(g[:, None], (2, 2, 2, 2), mode="replicate")[:, 0]
        acc = t.zeros_like(g)
        for dy in range(5):
            for dx in range(5):
                acc += p[:, dy:dy + fp.GH, dx:dx + fp.GW]
        g = (g - acc / 25.0).reshape(B, fp.NG)
        g = g - g.mean(-1, keepdim=True)
        g = g / (t.sqrt((g * g).mean(-1, keepdim=True)) + 1e-6)
        c = b.reshape(B, fp.CH, fp.BH // fp.CH, fp.CW, fp.BW // fp.CW, 3).mean(dim=(2, 4))
        s = c.sum(-1) + 3.0
        o = t.stack([(c[..., 0] - c[..., 1]) / s, (c[..., 0] + c[..., 1] - 2 * c[..., 2]) / (2 * s)], -1)

        def q(v, sc):
            return t.clamp(t.floor(v * sc + 0.5), -127, 127).to(t.int8)
        out = t.cat([q(g, fp.G_SCALE), q(o.reshape(B, fp.NC), fp.C_SCALE)], -1)
        return out.cpu().numpy()

    def run(self, paths, batch=64):
        out = [None] * len(paths)
        with ThreadPoolExecutor(NCPU) as ex:
            for i in range(0, len(paths), batch):
                chunk = list(enumerate(paths[i:i + batch], i))
                arrs = list(ex.map(lambda kp: _safe630(kp[1]), chunk))
                good = [(k, a) for (k, _), a in zip(chunk, arrs) if a is not None]
                if good:
                    d = self.desc(np.stack([a for _, a in good]))
                    for (k, _), row in zip(good, d):
                        out[k] = row
        return out


def _safe630(p):
    try:
        return load630(p)
    except Exception as e:  # noqa: BLE001
        print(f"  bad image {p}: {e}", flush=True)
        return None


def verify(gpu, n):
    with open(os.path.join(DATA, "ids-en.json"), encoding="utf-8") as f:
        ids = json.load(f)
    db = np.fromfile(os.path.join(DATA, "fp2-en.bin"), dtype=np.int8).reshape(-1, fp.RECORD_BYTES)
    rng = np.random.default_rng(1)
    pick = [int(i) for i in rng.permutation(len(ids)) if os.path.exists(fp.img_path("en", ids[i]))][:n]
    paths = [fp.img_path("en", ids[i]) for i in pick]
    gpu.run(paths[:8])                                   # warm-up (CUDA init)
    t0 = time.time(); g = gpu.run(paths); tg = time.time() - t0
    t0 = time.time(); c = [fp.fingerprint(Image.open(p)) for p in paths]; tc = time.time() - t0
    arrs = [load630(p) for p in paths]
    t0 = time.time(); [fp.desc_from_base(fp.base_of(a)) for a in arrs]; tcn = time.time() - t0
    t0 = time.time()
    for i in range(0, len(arrs), 64):
        gpu.desc(np.stack(arrs[i:i + 64]))
    if gpu.dev.type == "cuda":
        gpu.t.cuda.synchronize()
    tgn = time.time() - t0
    g = np.stack(g).astype(np.int16)
    dmax_db = int(np.abs(g - db[pick].astype(np.int16)).max())
    dmax_cpu = int(np.abs(g - np.stack(c).astype(np.int16)).max())
    neq = int((g != np.stack(c).astype(np.int16)).any(1).sum())
    print(f"VERIFY {len(paths)} images: max|gpu-fp2-en.bin|={dmax_db}, max|gpu-build_fp2|={dmax_cpu}, "
          f"records not byte-identical={neq}", flush=True)
    print(f"TIMING full pipeline: CPU build_fp2 1 thread {tc:.2f}s, GPU path ({NCPU} CPU threads for decode/resize) "
          f"{tg:.2f}s | numeric stage only: CPU numpy {tcn:.2f}s, torch {gpu.dev.type} {tgn:.3f}s", flush=True)
    return dmax_db <= 1 and dmax_cpu <= 1


# ---------------------------------------------------------------- main
def new_counts(recs):
    out = {}
    for r in recs:
        out.setdefault(r["setId"], []).append(r["id"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", type=int, default=200)
    ap.add_argument("--no-prices", action="store_true")
    ap.add_argument("--dry", action="store_true", help="only match sources and print what would be added")
    args = ap.parse_args()
    os.makedirs(SRC, exist_ok=True)
    if args.dry:
        args.verify, args.no_prices = 0, True
    gpu = None if args.dry else GpuFP()
    if args.verify and not verify(gpu, args.verify):
        sys.exit("GPU fingerprints differ from build_fp2 by more than 1 - aborting")

    with open(os.path.join(DATA, "extra-en.json"), encoding="utf-8") as f:
        old = json.load(f)
    oldbin = open(os.path.join(DATA, "fp2-extra-en.bin"), "rb").read()
    assert len(oldbin) == len(old) * fp.RECORD_BYTES
    if not args.no_prices:
        refresh_prices(old)

    det, sets = missing_cards()
    scry = scrydex_expansions(sorted(set(SETMAP.values())))
    manual = dict(MANUAL)
    for d in det:
        if d["set"]["id"] == "mfb":
            if d["name"] in MFB_POKEMON:
                manual[d["id"]] = f"bulba:{d['name']}MyFirstBattle.jpg"
            elif d["name"] in MFB_PROXY:
                manual[d["id"]] = "scry:" + MFB_PROXY[d["name"]]
    thumbs = bulba_thumbs([v[6:] for v in manual.values() if v.startswith("bulba:")])
    new, unresolved = [], {}
    used = set()
    for d in det:
        sid, num, name = d["set"]["id"], d["localId"], d["name"]
        url, src, nocors, proxy = None, None, False, None
        if d["id"] in manual:
            kind, ref = manual[d["id"]].split(":", 1)
            if kind == "bulba":
                # the host sends ACAO *, but Cloudflare answers browser fetch() (mode cors) with 403 while
                # <img> loads work -> the app cannot re-download it for verification
                url, src, nocors = thumbs.get(ref), "bulbagarden:" + ref, True
            else:
                url, src = SCRY_IMG.format(ref, "small"), ref
                if sid == "mfb":
                    proxy = ref
        elif sid in SETMAP:
            e = SETMAP[sid]
            want = CARDMAP.get(d["id"])
            cands = [c for c in scry.get(e, []) if c["id"] not in used]
            hit = next((c for c in cands if (c["id"] == want if want else nn(c["number"]) == nn(num))
                        and (want or name_ok(c["name"], name))), None)
            if hit:
                used.add(hit["id"]); src = hit["id"]
            elif not want and not any(nn(c["number"]) == nn(num) for c in scry.get(e, [])):
                # expansion page lists only part of the set (e.g. swshp is truncated): probe the CDN directly
                cid = f"{e}-{num.lstrip('0') if num.isdigit() else num}"
                if scry_probe(cid):
                    src = cid
            if src:
                url = SCRY_IMG.format(src, "small")
        if not url:
            unresolved.setdefault(sid, []).append(f"{d['id']} {name}")
            continue
        rec = {"id": d["id"], "name": name, "number": num,
               "printedTotal": sets[sid]["cardCount"].get("official"), "setName": d["set"]["name"], "setId": sid,
               "rarity": d.get("rarity"), "image": url, "cardmarket": None, "source": src}
        if nocors:
            rec["nocors"] = True
        if proxy:
            rec["proxy"] = proxy
        new.append((sets[sid].get("releaseDate", ""), rec))
    new.sort(key=lambda x: (x[0], x[1]["setId"]), reverse=True)     # newest sets first, like build_extra.py
    new = [r for _, r in new]
    if args.dry:
        print(f"DRY RUN: {len(new)} matched;", {k: len(v) for k, v in new_counts(new).items()})
        print("unresolved:", json.dumps(unresolved, ensure_ascii=False, indent=1))
        return

    print(f"downloading {len(new)} images ...", flush=True)
    with ThreadPoolExecutor(NET) as ex:
        paths = list(ex.map(download, [r["image"] for r in new]))
    for k, (r, p) in enumerate(zip(new, paths)):   # scrydex 'small' is sometimes 244 px wide: use 'medium'
        if p and r["image"].startswith("https://images.scrydex.com/") and Image.open(p).size[0] < 245:
            r["image"] = SCRY_IMG.format(r["source"], "medium")
            paths[k] = download(r["image"])
    t0 = time.time()
    fps = gpu.run([p for p in paths if p])
    print(f"fingerprinted {len(fps)} new images in {time.time() - t0:.1f}s", flush=True)
    it = iter(fps)
    out_recs, blobs = list(old), [oldbin]
    for r, p in zip(new, paths):
        f = next(it) if p else None
        if f is None:
            unresolved.setdefault(r["setId"], []).append(f"{r['id']} {r['name']} (download/decode failed)")
            continue
        out_recs.append(r)
        blobs.append(f.tobytes())
    with open(os.path.join(DATA, "extra-en.json"), "w", encoding="utf-8") as f:
        json.dump(out_recs, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(DATA, "fp2-extra-en.bin"), "wb") as f:
        f.write(b"".join(blobs))

    cnt = {}
    for r in out_recs[len(old):]:
        cnt[r["setId"]] = cnt.get(r["setId"], 0) + 1
    print(f"wrote {len(out_recs)} records ({len(old)} kept + {len(out_recs) - len(old)} new)")
    for k, v in sorted(cnt.items()):
        print(f"  + {k}: {v}")
    print("STILL WITHOUT IMAGE:", json.dumps(unresolved, ensure_ascii=False, indent=1))
    print("priced ptcg records:", sum(1 for r in out_recs if r["id"].startswith("ptcg:") and r.get("cardmarket")),
          "/", sum(1 for r in out_recs if r["id"].startswith("ptcg:")))


if __name__ == "__main__":
    main()
