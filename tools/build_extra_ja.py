#!/usr/bin/env python3
"""Build the supplementary JAPANESE visual DB (data/extra-ja.json + data/fp2-extra-ja.bin).

Records (same order in .json and .bin, 396 B each, fingerprint identical to build_fp2.fingerprint):
  1. TCGdex ja cards that have no image (absent from fp2-ja.bin): id = TCGdex ja id, image from
     the Scrydex CDN (images.scrydex.com/pokemon/<exp>_ja-<n>/medium; CORS: Access-Control-Allow-Origin: *).
  2. Japanese printings Scrydex lists but TCGdex does not have: id = "jsrc:<scrydex id>".
     cardmarket = null (no public source exposes Cardmarket prices for them).

Source mapping: TCGdex set id -> Scrydex Japanese expansion code (lower-case id, S* -> swsh*, plus
SET_MAP overrides); card matched by set + collector number; listing taken from the public Scrydex
expansion pages (scrydex.com/pokemon/expansions/x/<code>_ja?page=N), a missing-image placeholder
(md5 14bc00fb.../4a46ee65...) is rejected.

Fingerprints: decode + RGB + Pillow BILINEAR resize to 630x880 on CPU (worker pool, half the cores),
then the numeric stages (8x7 box means, luma 5x6 means, 5x5 local-mean high-pass, z-norm, chroma,
quantisation) as batched float64 torch ops on CUDA when available (--cpu forces build_fp2 on CPU).
--verify N compares N GPU records against build_fp2.fingerprint.

Run with the CUDA venv:  %LOCALAPPDATA%\\claude-gpu-venv\\Scripts\\python.exe tools/build_extra_ja.py
"""
import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_fp2 import (fingerprint, card630, RECORD_BYTES, DATA, BH, BW, GH, GW, CH, CW, NG, NC,  # noqa: E402
                       G_SCALE, C_SCALE)

SCRATCH = r"C:\Users\Fabio\AppData\Local\Temp\claude\d395c8141171058b\be555625-db4a-4cfa-8cf3-32e7aeb800e3\scratchpad"
SRC = os.path.join(SCRATCH, "jasrc")      # cached JSON / HTML
IMG = os.path.join(SCRATCH, "jacache")    # cached images
TD = "https://api.tcgdex.net/v2/ja"
SCRY = "https://scrydex.com"
CDN = "https://images.scrydex.com/pokemon"
SIZE = "medium"
PLACEHOLDER_MD5 = {"14bc00fbe48a20c4e018cc54b4d53b5b", "4a46ee6546fda3f1c66605b96701f7a1"}  # large, medium
CONC = 8
THREADS = max(1, (os.cpu_count() or 2) // 2)

SET_MAP = {"PMCG1": "base1", "PMCG2": "base2", "PMCG3": "base3", "PMCG4": "base4", "PMCG5": "gym1",
           "PMCG6": "gym2", "E1": "ecard1", "E2": "ecard2", "E3": "ecard3", "E4": "ecard4", "E5": "ecard5",
           "L1a": "l1hg", "L1b": "l1ss", "LL": "ll1", "BW1a": "bw1b", "BW1b": "bw1w", "XY8b": "xy8r", "BW3a": "bw3p", "BW3b": "bw3h",
           "DS": "ds1", "BW5a": "bw5d", "BW5b": "bw5s", "BW6a": "bw6f", "BW6b": "bw6c", "BW8a": "bw8s",
           "BW8b": "bw8t", "SC": "sc1", "EBB": "ebb1", "XY1a": "xy1x", "XY1b": "xy1y", "XY5a": "xy5g",
           "XY5b": "xy5t", "XY8a": "xy8b", "XY11a": "xy11f", "XY11b": "xy11c", "M-P": "mp", "SV-P": "svp"}

_lock = threading.Lock()
_pause = [0.0]


def get(url, timeout=60, tries=6):
    """GET with retries, exponential backoff and Retry-After on 429/503 (shared pause across threads)."""
    for i in range(tries):
        wait = _pause[0] - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (card-db-builder)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            d = float(e.headers.get("Retry-After") or 0) if e.code in (429, 503) else 0
            d = max(d, 2 ** i)
            if e.code in (429, 503):
                with _lock:
                    _pause[0] = max(_pause[0], time.time() + d)
            if i == tries - 1:
                raise
            time.sleep(d)
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def cached(path, url, binary=False):
    if not os.path.exists(path) or os.path.getsize(path) < 2:
        data = get(url)
        if data is None:
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    if binary:
        return open(path, "rb").read()
    with open(path, encoding="utf-8") as f:
        return f.read()


def fname(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def nnum(x):
    x = urllib.parse.unquote(str(x)).strip()
    m = re.match(r"^([A-Za-z]*)0*(\d+)([A-Za-z]*)$", x)
    return (m.group(1).upper() + m.group(2) + m.group(3).upper()) if m else x.upper()


# ---------------------------------------------------------------- sources
def load_tcgdex():
    cards = json.loads(cached(os.path.join(SRC, "cards.json"), f"{TD}/cards"))
    sets = json.loads(cached(os.path.join(SRC, "sets.json"), f"{TD}/sets"))
    with ThreadPoolExecutor(CONC) as ex:
        sdet = dict(ex.map(lambda s: (s["id"], json.loads(cached(os.path.join(SRC, "tdsets", fname(s["id"]) + ".json"),
                                                                 f"{TD}/sets/{urllib.parse.quote(s['id'])}"))), sets))
    return cards, sets, sdet


def card_details(ids):
    def one(cid):
        t = cached(os.path.join(SRC, "tdcards", fname(cid) + ".json"), f"{TD}/cards/{urllib.parse.quote(cid)}")
        return cid, (json.loads(t) if t else {})
    with ThreadPoolExecutor(CONC) as ex:
        return dict(ex.map(one, ids))


CARD_RE = re.compile(r'href="/pokemon/cards/([^/"]+)/([a-z0-9]+)_ja-([^"?]+)\?[^"]*"><div class="flex flex-col justify-center'
                     r'[^>]*data-id="[^"]+">.*?<span class="text-body-12 text-white text-center">([^<]*)</span>', re.S)


def scrydex_listing():
    page = cached(os.path.join(SRC, "jpexp.html"), f"{SCRY}/pokemon/jp/expansions")
    exps = []
    for m in re.finditer(r'href="/pokemon/expansions/([^"/]+)/([a-z0-9]+)_ja"', page):
        if m.group(2) not in [e[0] for e in exps]:
            exps.append((m.group(2), m.group(1)))

    def crawl(e):
        code, slug = e
        out, p = {}, 1
        while True:
            t = cached(os.path.join(SRC, "scry", f"{code}_p{p}.html"),
                       f"{SCRY}/pokemon/expansions/{slug}/{code}_ja" + (f"?page={p}" if p > 1 else ""))
            if not t:
                break
            n0 = len(out)
            for m in CARD_RE.finditer(t):
                num = m.group(3)
                nm = html.unescape(m.group(4))
                nm = re.sub(r"\s+#\S+$", "", nm)
                out.setdefault(num, nm)
            if len(out) == n0 or f"page={p + 1}" not in t:
                break
            p += 1
        p1 = os.path.join(SRC, "scry", f"{code}_p1.html")
        title = re.search(r"<title>([^<]*)</title>", open(p1, encoding="utf-8").read()) if os.path.exists(p1) else None
        return code, slug, out, html.unescape(title.group(1)).split("|")[0].strip() if title else slug

    with ThreadPoolExecutor(CONC) as ex:
        return {c: (s, o, t) for c, s, o, t in ex.map(crawl, exps)}


def download(url):
    p = os.path.join(IMG, fname(url.split("://", 1)[1]))
    if os.path.exists(p) and os.path.getsize(p) > 0:
        with open(p, "rb") as f:
            return None if hashlib.md5(f.read()).hexdigest() in PLACEHOLDER_MD5 else p
    try:
        data = get(url)
    except Exception as e:  # noqa: BLE001
        print(f"  download failed {url}: {e}", flush=True)
        return None
    if not data or hashlib.md5(data).hexdigest() in PLACEHOLDER_MD5:
        return None
    with open(p, "wb") as f:
        f.write(data)
    return p


# ---------------------------------------------------------------- fingerprints
def _load630(path):
    try:
        return np.asarray(card630(Image.open(path)), dtype=np.uint8)
    except Exception:  # noqa: BLE001
        return None


def gpu_desc(batch_u8, torch, dev):
    """(N,880,630,3) uint8 -> (N,396) int8, mirroring build_fp2.desc_from_base(base_of(.)) in float64."""
    a = torch.from_numpy(batch_u8).to(dev).to(torch.float64)
    n = a.shape[0]
    b = a.reshape(n, BH, 8, BW, 7, 3).mean(dim=(2, 4))                                  # (n,110,90,3)
    y = b[..., 0] * 0.299 + b[..., 1] * 0.587 + b[..., 2] * 0.114
    g = y.reshape(n, GH, BH // GH, GW, BW // GW).mean(dim=(2, 4))                        # (n,22,15)
    p = torch.nn.functional.pad(g.unsqueeze(1), (2, 2, 2, 2), mode="replicate").squeeze(1)
    acc = torch.zeros_like(g)
    for dy in range(5):
        for dx in range(5):
            acc += p[:, dy:dy + GH, dx:dx + GW]
    g = (g - acc / 25.0).reshape(n, NG)
    g = g - g.mean(-1, keepdim=True)
    g = g / (torch.sqrt((g * g).mean(-1, keepdim=True)) + 1e-6)
    c = b.reshape(n, CH, BH // CH, CW, BW // CW, 3).mean(dim=(2, 4))                     # (n,11,3,3)
    s = c.sum(-1) + 3.0
    o = torch.stack([(c[..., 0] - c[..., 1]) / s, (c[..., 0] + c[..., 1] - 2 * c[..., 2]) / (2 * s)], -1)
    q = lambda v, k: torch.clamp(torch.floor(v * k + 0.5), -127, 127).to(torch.int8)  # noqa: E731
    return torch.cat([q(g, G_SCALE), q(o.reshape(n, NC), C_SCALE)], -1).cpu().numpy()


def smi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi unavailable: {e}"


def fingerprints(paths, use_gpu=True, batch=64):
    out = [None] * len(paths)
    torch = dev = None
    if use_gpu:
        try:
            import torch  # noqa: F811
            if torch.cuda.is_available():
                dev = torch.device("cuda")
                torch.set_num_threads(THREADS)
        except ImportError:
            pass
    t0 = time.time()
    with Pool(THREADS) as pool:
        if dev is None:
            print("fingerprints on CPU (build_fp2)", flush=True)
            for i, a in enumerate(pool.imap(_load630, paths, chunksize=8)):
                out[i] = None if a is None else fingerprint(Image.fromarray(a)).tobytes()
        else:
            print(f"fingerprints on GPU {torch.cuda.get_device_name(0)}, batch {batch}, {THREADS} CPU workers", flush=True)
            idx, buf, nb = [], [], 0
            it = pool.imap(_load630, paths, chunksize=4)
            for i, a in enumerate(it):
                if a is not None:
                    idx.append(i); buf.append(a)
                if len(buf) == batch or (i == len(paths) - 1 and buf):
                    d = gpu_desc(np.stack(buf), torch, dev)
                    for j, r in zip(idx, d):
                        out[j] = r.tobytes()
                    idx, buf = [], []
                    nb += 1
                    if nb % 25 == 1:
                        print(f"  [{i + 1}/{len(paths)}] {time.time() - t0:.0f}s  nvidia-smi: {smi()}", flush=True)
    print(f"fingerprinted {len(paths)} images in {time.time() - t0:.1f}s", flush=True)
    return out


def verify(paths, n=200):
    paths = paths[:n]
    t0 = time.time()
    cpu = [fingerprint(Image.open(p)) for p in paths]
    tc = time.time() - t0
    t0 = time.time()
    gpu = fingerprints(paths, use_gpu=True)
    tg = time.time() - t0
    diffs = [int(np.abs(np.frombuffer(g, np.int8).astype(int) - c.astype(int)).max()) for g, c in zip(gpu, cpu)]
    neq = sum(d > 0 for d in diffs)
    print(f"VERIFY {len(paths)} images: max abs byte diff {max(diffs)}, records differing {neq}; "
          f"CPU single-thread build_fp2 {tc:.1f}s, GPU pipeline {tg:.1f}s", flush=True)
    return max(diffs)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--verify", type=int, default=200)
    ap.add_argument("--no-jsrc", action="store_true")
    args = ap.parse_args()
    os.makedirs(IMG, exist_ok=True)
    print(smi(), flush=True)

    cards, sets, sdet = load_tcgdex()
    with open(os.path.join(DATA, "ids-ja.json"), encoding="utf-8") as f:
        have = set(json.load(f))
    scry = scrydex_listing()
    print(f"TCGdex ja: {len(cards)} cards, {len(sets)} sets; Scrydex ja: {len(scry)} expansions, "
          f"{sum(len(v[1]) for v in scry.values())} cards", flush=True)

    smap = {}
    for s in sets:
        i = s["id"]
        c = SET_MAP.get(i) or (i.lower() if i.lower() in scry else
                              ("swsh" + i[1:].lower() if i[:1] in "Ss" and "swsh" + i[1:].lower() in scry else None))
        if c in scry:
            smap[i] = c
    used_codes = set(smap.values())

    # 1. TCGdex ja cards without image
    todo = []
    td_nums = defaultdict(set)
    for c in cards:
        sid = c["id"].rsplit("-", 1)[0] if c["id"].rsplit("-", 1)[0] in sdet else None
        if sid is None:
            sid = next((s for s in sdet if c["id"].startswith(s + "-")), None)
        td_nums[sid].add(nnum(c["localId"]))
        if c.get("image") or c["id"] in have:
            continue
        code = smap.get(sid)
        if not code:
            continue
        listing = scry[code][1]
        lk = {nnum(k): k for k in listing}
        k = lk.get(nnum(c["localId"]))
        if k is None:
            continue
        todo.append((c, sid, f"{code}_ja-{k}"))
    print(f"TCGdex ja cards lacking image: {sum(1 for c in cards if not c.get('image'))}; matched on Scrydex: {len(todo)}",
          flush=True)

    # 2. Scrydex Japanese cards not in TCGdex
    jsrc = []
    if not args.no_jsrc:
        inv = defaultdict(list)
        for s, c in smap.items():
            inv[c].append(s)
        for code, (slug, listing, title) in scry.items():
            for k, nm in listing.items():
                if code in used_codes and any(nnum(k) in td_nums[s] for s in inv[code]):
                    continue
                jsrc.append((code, k, nm, title))
    print(f"Scrydex-only Japanese cards (jsrc): {len(jsrc)}", flush=True)

    # downloads + TCGdex details (rarity) in parallel, <= CONC requests each
    urls = [f"{CDN}/{sx}/{SIZE}" for _, _, sx in todo] + [f"{CDN}/{c}_ja-{k}/{SIZE}" for c, k, _, _ in jsrc]
    t0 = time.time()
    with ThreadPoolExecutor(2) as outer:
        fd = outer.submit(card_details, [c["id"] for c, _, _ in todo])
        with ThreadPoolExecutor(CONC) as ex:
            paths = []
            for i, p in enumerate(ex.map(download, urls)):
                paths.append(p)
                if (i + 1) % 1000 == 0:
                    print(f"  downloaded {i + 1}/{len(urls)} ({time.time() - t0:.0f}s)", flush=True)
        details = fd.result()
    print(f"images: {sum(p is not None for p in paths)}/{len(urls)} in {time.time() - t0:.0f}s", flush=True)

    ok = [p for p in paths if p]
    if not args.cpu and args.verify:
        if verify(ok, args.verify) > 1:
            print("GPU mismatch > 1 -> falling back to CPU", flush=True)
            args.cpu = True
    fps = fingerprints([p or "" for p in paths], use_gpu=not args.cpu)

    recs, bins = [], []
    n_td = len(todo)
    for i, (url, p, fp) in enumerate(zip(urls, paths, fps)):
        if p is None or fp is None:
            continue
        if i < n_td:
            c, sid, sx = todo[i]
            d = details.get(c["id"]) or {}
            sd = sdet[sid]
            recs.append({"id": c["id"], "name": d.get("name") or c["name"], "number": c["localId"],
                         "printedTotal": (sd.get("cardCount") or {}).get("official"), "setName": sd.get("name"),
                         "setId": sid, "rarity": d.get("rarity"), "image": url, "cardmarket": None, "source": sx})
        else:
            code, k, nm, title = jsrc[i - n_td]
            recs.append({"id": f"jsrc:{code}_ja-{k}", "name": nm, "number": k, "printedTotal": None,
                         "setName": title, "setId": f"{code}_ja", "rarity": None, "image": url, "cardmarket": None,
                         "source": f"{code}_ja-{k}", "lang": "ja"})
        bins.append(fp)
    with open(os.path.join(DATA, "extra-ja.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(DATA, "fp2-extra-ja.bin"), "wb") as f:
        f.write(b"".join(bins))
    per = defaultdict(int)
    for r in recs:
        per[r["setId"] if not r["id"].startswith("jsrc:") else "jsrc"] += 1
    print(f"wrote {len(recs)} records ({sum(1 for r in recs if not r['id'].startswith('jsrc:'))} TCGdex, "
          f"{per['jsrc']} jsrc)", flush=True)
    print(json.dumps(dict(sorted(per.items())), ensure_ascii=False), flush=True)
    print(smi(), flush=True)


if __name__ == "__main__":
    main()
