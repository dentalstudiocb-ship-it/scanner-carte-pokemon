#!/usr/bin/env python3
"""Build the supplementary visual DB (data/extra-en.json + data/fp2-extra-en.bin).

Two kinds of records (same order in .json and .bin, 396 B per record, fingerprint from build_fp2):
  1. TCGdex cards that exist but have no image (so they are absent from fp2-en.bin):
     id = original TCGdex id; image taken from the PokemonTCG data (images.pokemontcg.io / scrydex CDN).
  2. Printings that TCGdex does not have at all: id = "ptcg:<pokemontcg id>".

Sources: TCGdex API (sets + card lists), PokemonTCG/pokemon-tcg-data GitHub JSON (cards + images),
optionally api.pokemontcg.io for cardmarket prices (flaky; prices recorded only when it answers).
Usage: python tools/build_extra.py
"""
import json
import os
import re
import sys
import time
import difflib
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_fp2 import fingerprint, RECORD_BYTES, DATA, NG, C_WEIGHT  # noqa: E402

SCRATCH = r"C:\Users\Fabio\AppData\Local\Temp\claude\d395c8141171058b\be555625-db4a-4cfa-8cf3-32e7aeb800e3\scratchpad"
SRC = os.path.join(SCRATCH, "extra")          # cached source JSON
IMG = os.path.join(SCRATCH, "extracache")     # cached images
GH = "https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master"
TD = "https://api.tcgdex.net/v2/en"
PTCG = "https://api.pokemontcg.io/v2"
POCKET = re.compile(r"^(A\d|B\d|P-A)")
PRIORITY = ["30th", "30th-c"]

# gh set id -> tcgdex set id where names differ
SET_OVERRIDE = {"base1": "base1", "hgss1": "hgss1", "hgss2": "hgss2", "hgss3": "hgss3", "hgss4": "hgss4",
                "fut20": "fut2020", "svp": "svp", "sve": "sve", "me55c": "30th-c", "cel25c": "cel25cc"}
EXTRA_SOURCES = {"exu": ["ex10"]}             # tcgdex set -> additional gh sets to search


def get(url, timeout=60, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "card-db-builder"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def cached_json(path, url):
    if not os.path.exists(path) or os.path.getsize(path) < 2:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(get(url))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def nn(x):
    x = urllib.parse.unquote(x)
    m = re.match(r"^([A-Za-z]*)0*(\d+)(.*)$", x)
    return (m.group(1).upper() + m.group(2) + m.group(3).lower()) if m else x.upper()


def nm(s):
    s = s.lower().replace("☆", "star").replace("★", "star").replace("δ", "delta").replace("&", "and")
    return re.sub(r"[^a-z0-9]", "", s)


def name_ok(a, b):
    a, b = nm(a), nm(b)
    return a == b or a.startswith(b) or b.startswith(a) or difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def load_sources():
    gh_sets = cached_json(os.path.join(SRC, "gh_sets.json"), f"{GH}/sets/en.json")
    td_list = cached_json(os.path.join(SRC, "tcgdex_sets.json"), f"{TD}/sets")
    with ThreadPoolExecutor(16) as ex:
        td = dict(ex.map(lambda s: (s["id"], cached_json(os.path.join(SRC, "tdsets", s["id"] + ".json"),
                                                         f"{TD}/sets/{urllib.parse.quote(s['id'])}")), td_list))
        ghc = dict(ex.map(lambda s: (s["id"], cached_json(os.path.join(SRC, "ghcards", s["id"] + ".json"),
                                                          f"{GH}/cards/en/{s['id']}.json")), gh_sets))
    return gh_sets, td, ghc


def map_sets(gh_sets, td):
    byname = {}
    for d in td.values():
        byname.setdefault(nm(d["name"]), []).append(d)
    out = {}
    for s in gh_sets:
        if s["id"] in SET_OVERRIDE:
            out[s["id"]] = SET_OVERRIDE[s["id"]]
            continue
        c = byname.get(nm(s["name"]), [])
        c = [d for d in c if d.get("releaseDate", "")[:7] == s["releaseDate"].replace("/", "-")[:7]] or c
        out[s["id"]] = c[0]["id"] if len(c) == 1 else None
    return out


def fetch_prices(set_ids):
    """cardmarket prices from api.pokemontcg.io; {} for sets where the API does not answer."""
    prices = {}
    for sid in set_ids:
        try:
            raw = get(f"{PTCG}/cards?q=set.id:{sid}&pageSize=250&select=id,cardmarket", tries=3)
            for c in json.loads(raw)["data"]:
                cm = c.get("cardmarket")
                if cm:
                    p = cm.get("prices", {})
                    prices[c["id"]] = {"trend": p.get("trendPrice"), "avg30": p.get("avg30"), "avg7": p.get("avg7"),
                                       "avg1": p.get("avg1"), "low": p.get("lowPrice"), "avg": p.get("averageSellPrice"),
                                       "reverseTrend": p.get("reverseHoloTrend"), "url": cm.get("url"),
                                       "updated": cm.get("updatedAt")}
        except Exception as e:  # noqa: BLE001
            print(f"  prices for {sid}: API unavailable ({e})", flush=True)
    return prices


def img_file(url):
    return os.path.join(IMG, re.sub(r"[^A-Za-z0-9._-]", "_", url.split("://", 1)[1]))


def download(url):
    p = img_file(url)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    try:
        data = get(url)
        with open(p, "wb") as f:
            f.write(data)
        return p
    except Exception as e:  # noqa: BLE001
        print(f"  download failed {url}: {e}", flush=True)
        return None


def main():
    os.makedirs(IMG, exist_ok=True)
    with open(os.path.join(DATA, "ids-en.json"), encoding="utf-8") as f:
        have = set(json.load(f))
    gh_sets, td, ghc = load_sources()
    gh_meta = {s["id"]: s for s in gh_sets}
    smap = map_sets(gh_sets, td)
    td2gh = {}
    for g, t in smap.items():
        if t:
            td2gh.setdefault(t, []).append(g)
    for t, gs in EXTRA_SOURCES.items():
        td2gh.setdefault(t, []).extend(gs)

    used = set()          # gh card ids matched to some tcgdex card
    records = []          # (sortkey, rec, imgurl)
    unresolved = {}
    for tid, d in td.items():
        if POCKET.match(tid):
            continue
        cands = [(g, c) for g in td2gh.get(tid, []) for c in ghc[g]]
        cards = d.get("cards", [])
        match = {}
        for c in cards:      # pass 1: same number + compatible name
            for g, gc in cands:
                if gc["id"] not in used and nn(gc["number"]) == nn(c["localId"]) and name_ok(gc["name"], c["name"]):
                    match[c["id"]] = (g, gc); used.add(gc["id"]); break
        for c in cards:      # pass 2: name only (subsets renumbered by TCGdex), first unused in source order
            if c["id"] in match:
                continue
            for g, gc in cands:
                if gc["id"] not in used and nm(gc["name"]) == nm(c["name"]):
                    match[c["id"]] = (g, gc); used.add(gc["id"]); break
        for c in cards:      # pass 3: loose name
            if c["id"] in match:
                continue
            for g, gc in cands:
                if gc["id"] not in used and name_ok(gc["name"], c["name"]):
                    match[c["id"]] = (g, gc); used.add(gc["id"]); break
        for c in cards:
            if c["id"] in have:
                continue
            if c["id"] not in match:
                unresolved.setdefault(tid, []).append(c["id"])
                continue
            g, gc = match[c["id"]]
            pr = 0 if tid in PRIORITY else 1
            rec = {"id": c["id"], "name": c["name"], "number": c["localId"],
                   "printedTotal": d["cardCount"].get("official"), "setName": d["name"], "setId": tid,
                   "rarity": gc.get("rarity"), "image": gc["images"]["small"], "_large": gc["images"].get("large"), "cardmarket": None,
                   "source": gc["id"]}
            records.append(((pr, 0, "~" + d.get("releaseDate", ""), tid), rec))

    # printings TCGdex does not have at all
    missing = [(g, gc) for g in ghc for gc in ghc[g] if gc["id"] not in used]
    prices = fetch_prices(sorted({g for g, _ in missing}))
    for g, gc in missing:
        s = gh_meta[g]
        rec = {"id": "ptcg:" + gc["id"], "name": gc["name"], "number": gc["number"],
               "printedTotal": s.get("printedTotal"), "setName": s["name"], "setId": g,
               "rarity": gc.get("rarity"), "image": gc["images"]["small"], "_large": gc["images"].get("large"),
               "cardmarket": prices.get(gc["id"])}
        pr = 0 if g in ("cel25c",) else 2
        records.append(((pr, 1, "~" + s["releaseDate"], g), rec))
    # newest sets first inside each priority group (release date inverted)
    records.sort(key=lambda r: (r[0][0], r[0][1], "".join(chr(255 - ord(ch)) for ch in r[0][2]), r[0][3]))
    records = [r for _, r in records]

    print(f"downloading {len(records)} images ...", flush=True)
    with ThreadPoolExecutor(16) as ex:
        paths = list(ex.map(download, [r["image"] for r in records]))
    out_recs, blobs = [], []
    for r, p in zip(records, paths):
        if not p:
            continue
        try:
            img = Image.open(p)
            if img.size[0] < 245 and r.get("_large"):   # some old small scans are 240 px: use the large one
                p = download(r["_large"])
                if not p:
                    continue
                img = Image.open(p)
                r["image"] = r["_large"]
            if img.size[0] < 245:
                print(f"  too small {r['id']} {img.size}"); continue
            blobs.append(fingerprint(img).tobytes())
        except Exception as e:  # noqa: BLE001
            print(f"  bad image {r['id']}: {e}"); continue
        r.pop("_large", None)
        out_recs.append(r)
    with open(os.path.join(DATA, "extra-en.json"), "w", encoding="utf-8") as f:
        json.dump(out_recs, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(DATA, "fp2-extra-en.bin"), "wb") as f:
        f.write(b"".join(blobs))

    # report
    cnt = {}
    for r in out_recs:
        k = ("ptcg " if r["id"].startswith("ptcg:") else "tcgdex ") + r["setId"]
        cnt[k] = cnt.get(k, 0) + 1
    print(f"wrote {len(out_recs)} records")
    for k, v in sorted(cnt.items()):
        print(f"  {k}: {v}")
    print("unresolved (tcgdex no-image cards without a source image):",
          {k: len(v) for k, v in sorted(unresolved.items())})
    print("priced ptcg records:", sum(1 for r in out_recs if r["cardmarket"]))
    sanity(out_recs, blobs)


def sanity(recs, blobs):
    ids = json.load(open(os.path.join(DATA, "ids-en.json"), encoding="utf-8"))
    db = np.fromfile(os.path.join(DATA, "fp2-en.bin"), dtype=np.int8).reshape(-1, RECORD_BYTES).astype(np.int16)
    for want in ("30th-c-018", "cel25cc-CC016"):
        i = next((k for k, r in enumerate(recs) if r["id"] == want), None)
        if i is None:
            print(f"SANITY: {want} NOT present"); continue
        q = np.frombuffer(blobs[i], np.int8).astype(np.int16)
        d = np.abs(db[:, :NG] - q[:NG]).sum(1) + C_WEIGHT * np.abs(db[:, NG:] - q[NG:]).sum(1)
        o = np.argsort(d)[:3]
        print(f"SANITY {want} ({recs[i]['name']}, {recs[i]['source']}): nearest "
              + ", ".join(f"{ids[k]}={d[k]:.0f}" for k in o)
              + (f"; hgss4-94={d[ids.index('hgss4-94')]:.0f}" if "hgss4-94" in ids else ""))


if __name__ == "__main__":
    main()
