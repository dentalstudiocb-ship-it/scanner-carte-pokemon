#!/usr/bin/env python3
"""Build the visual-fingerprint database of Pokemon TCG card images (TCGdex).

Usage:
    python tools/build_hashes.py              # build en + ja (data/hashes-*.bin, data/ids-*.json, tools/hash_ref.json)
    python tools/build_hashes.py --lang en    # one language only
    python tools/build_hashes.py --verify 30  # distortion sanity check against data/hashes-en.bin

Image cache: set IMGCACHE env var to override the default cache directory.

Fingerprint (60 bytes per card) = 24-byte dHash of the art region + 36 bytes of 3x4 grid mean colours.
The JS side reimplements fingerprint() exactly; keep both in sync.
"""
import argparse
import io
import json
import os
import random
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter

API = "https://api.tcgdex.net/v2"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DEFAULT_CACHE = r"C:\Users\Fabio\AppData\Local\Temp\claude\d395c8141171058b\be555625-db4a-4cfa-8cf3-32e7aeb800e3\scratchpad\imgcache"
CACHE = os.environ.get("IMGCACHE", DEFAULT_CACHE)
WORKERS = 16
REF_IDS = [("en", "sm4-57"), ("en", "sm4-115"), ("en", "sv03.5-199"), ("ja", "SV2a-006")]

W, H = 192, 264
COL_X = [0, 16, 32, 48]
COL_Y = [0, 16, 33, 49, 66]
ART_X0, ART_X1, ART_Y0, ART_Y1, BLK = 11, 181, 24, 144, 10
REC = 60

_session = requests.Session()
_session.headers["User-Agent"] = "tcg-fingerprint-builder/1.0"


# ---------------------------------------------------------------- fingerprint
def to_rgb(img):
    """Convert to RGB, flattening any alpha onto white."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        return bg.convert("RGB")
    return img.convert("RGB")


def _round_u8(a):
    # round half up (matches JS Math.round for non-negative values), clamp 0..255
    return np.clip(np.floor(a + 0.5), 0, 255).astype(np.uint8)


def fingerprint(img):
    img = to_rgb(img).resize((W, H), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float64)  # (264, 192, 3)

    # dHash on the art region
    gray = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    art = gray[ART_Y0:ART_Y1, ART_X0:ART_X1]
    G = art.reshape(12, BLK, 17, BLK).mean(axis=(1, 3))  # (12, 17)
    bits = (G[:, :16] > G[:, 1:]).astype(np.uint8).reshape(-1)  # 192 bits row-major
    dhash = np.packbits(bits)  # MSB first, 24 bytes

    # colour grid on the 48x66 box-averaged image
    S = a.reshape(66, 4, 48, 4, 3).mean(axis=(1, 3))  # (66, 48, 3)
    col = []
    for r in range(4):
        for c in range(3):
            cell = S[COL_Y[r]:COL_Y[r + 1], COL_X[c]:COL_X[c + 1]]
            col.extend(cell.reshape(-1, 3).mean(axis=0))
    color = _round_u8(np.array(col))
    return dhash.tobytes() + color.tobytes()


# ---------------------------------------------------------------- fetching
def get_json(url):
    # API metadata calls (a handful per run): be patient with transient 5xx
    for attempt in range(6):
        try:
            r = _session.get(url, timeout=90)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))


def card_list(lang):
    cards = get_json(f"{API}/{lang}/cards") or []
    tcgp = get_json(f"{API}/{lang}/series/tcgp")
    pocket = {s["id"] for s in (tcgp or {}).get("sets", [])}
    out = [c for c in cards if c.get("image") and c["id"].rsplit("-", 1)[0] not in pocket]
    return out, len(cards), len(pocket)


def cache_path(lang, cid):
    return os.path.join(CACHE, lang, urllib.parse.quote(cid, safe="") + ".png")


def download(url, retries=2):
    """GET bytes; retry 404/5xx/network errors at most `retries` times."""
    last = None
    for attempt in range(retries + 1):
        try:
            r = _session.get(url, timeout=60)
            if r.status_code == 200 and r.content:
                return r.content
            last = f"HTTP {r.status_code}"
            if not (r.status_code == 404 or r.status_code >= 500):
                break
        except requests.RequestException as e:
            last = type(e).__name__
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def fetch_card(lang, card):
    p = cache_path(lang, card["id"])
    if os.path.exists(p) and os.path.getsize(p) > 0:
        with open(p, "rb") as f:
            data = f.read()
    else:
        data = download(card["image"] + "/low.png")
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
    return fingerprint(Image.open(io.BytesIO(data)))


# ---------------------------------------------------------------- build
def build(lang):
    t0 = time.time()
    os.makedirs(os.path.join(CACHE, lang), exist_ok=True)
    cards, total, npocket = card_list(lang)
    print(f"[{lang}] {total} cards listed, {npocket} Pocket sets excluded, {len(cards)} with image", flush=True)
    results = {}
    failures = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch_card, lang, c): c["id"] for c in cards}
        for i, fut in enumerate(as_completed(futs), 1):
            cid = futs[fut]
            try:
                results[cid] = fut.result()
            except Exception as e:  # noqa: BLE001
                failures.append((cid, str(e)))
            if i % 1000 == 0:
                print(f"[{lang}] {i}/{len(cards)} ({time.time() - t0:.0f}s, {len(failures)} failed)", flush=True)
    ids = [c["id"] for c in cards if c["id"] in results]
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, f"hashes-{lang}.bin"), "wb") as f:
        for cid in ids:
            f.write(results[cid])
    with open(os.path.join(DATA, f"ids-{lang}.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[{lang}] wrote {len(ids)} records in {time.time() - t0:.0f}s; {len(failures)} failures", flush=True)
    for cid, err in failures:
        print(f"  FAIL {cid}: {err}", flush=True)
    return ids, results, failures


def write_ref(all_results):
    ref = {}
    for lang, cid in REF_IDS:
        rec = all_results.get(lang, {}).get(cid)
        if rec is None:  # language not rebuilt this run: compute from cache / network
            rec = fetch_card(lang, {"id": cid, "image": f"https://assets.tcgdex.net/{lang}/" + _image_path(lang, cid)})
        ref[cid] = {"hex": rec.hex()}
    with open(os.path.join(ROOT, "tools", "hash_ref.json"), "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=2)
    print("wrote tools/hash_ref.json", flush=True)


def _image_path(lang, cid):
    c = get_json(f"{API}/{lang}/cards/{urllib.parse.quote(cid, safe='')}")
    return c["image"].split(f"/{lang}/", 1)[1]


# ---------------------------------------------------------------- verify
def load_db(lang):
    with open(os.path.join(DATA, f"ids-{lang}.json"), encoding="utf-8") as f:
        ids = json.load(f)
    raw = np.fromfile(os.path.join(DATA, f"hashes-{lang}.bin"), dtype=np.uint8).reshape(-1, REC)
    assert len(ids) == len(raw)
    return ids, raw


def distances(rec, raw):
    q = np.frombuffer(rec, dtype=np.uint8)
    ham = np.unpackbits(raw[:, :24] ^ q[:24], axis=1).sum(axis=1)
    cd = np.abs(raw[:, 24:].astype(np.int32) - q[24:].astype(np.int32)).sum(axis=1)
    return 3 * ham + cd / 8.0


def distort(img):
    img = to_rgb(img)
    img = img.rotate(2, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    img = ImageEnhance.Brightness(img).enhance(0.9)
    w, h = img.size
    img = img.resize((round(w * 0.7), round(h * 0.7)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return img


def verify(n, seed=1234):
    ids, raw = load_db("en")
    cards, _, _ = card_list("en")
    by_id = {c["id"]: c for c in cards}
    rng = random.Random(seed)
    sample = rng.sample(ids, n)
    top1 = top5 = 0
    for cid in sample:
        data = download(by_id[cid]["image"] + "/low.png")  # fresh download
        rec = fingerprint(distort(Image.open(io.BytesIO(data))))
        d = distances(rec, raw)
        order = np.argsort(d, kind="stable")[:5]
        hits = [ids[i] for i in order]
        ok1, ok5 = hits[0] == cid, cid in hits
        top1 += ok1
        top5 += ok5
        print(f"{'OK ' if ok1 else ('top5' if ok5 else 'MISS')} {cid:>16}  -> {hits[0]} (d={d[order[0]]:.1f}, true d={d[ids.index(cid)]:.1f})", flush=True)
    print(f"top-1: {top1}/{n} = {100 * top1 / n:.1f}%   top-5: {top5}/{n} = {100 * top5 / n:.1f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "ja"], action="append")
    ap.add_argument("--verify", type=int, metavar="N")
    args = ap.parse_args()
    t0 = time.time()
    if args.verify:
        verify(args.verify)
    else:
        allres = {}
        for lang in args.lang or ["en", "ja"]:
            _, res, _ = build(lang)
            allres[lang] = res
        write_ref(allres)
    print(f"total runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
