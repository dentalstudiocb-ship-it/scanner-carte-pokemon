#!/usr/bin/env python3
"""Build the v2 visual-fingerprint database (data/fp2-{en,ja}.bin) and benchmark it.

Usage:
    python tools/build_fp2.py               # build en + ja from the cached low.png images
    python tools/build_fp2.py --lang en
    python tools/build_fp2.py --bench       # distortion benchmark (v2 vs v1) on data/*-en.bin

Records are in the SAME order as data/ids-{lang}.json (built by build_hashes.py).
A card whose cached image is missing gets an all-zero record (reported on stdout).

Fingerprint v2 (RECORD_BYTES = 396, int8 stored as raw bytes):
  1. image -> RGB (alpha flattened on white) -> resize to exactly 630x880 (PIL BILINEAR;
     fp2.js reimplements Pillow's fixed-point bilinear resample so both sides agree bit-exactly)
  2. "base" = 110x90 RGB grid of 8x7-pixel box means (float)
  3. luma Y = .299R + .587G + .114B on the base; 22x15 grid of 5x6 box means of Y
  4. high-pass: subtract the 5x5 neighbourhood mean (edge-clamped) on the 22x15 grid
  5. z-normalise (zero mean, unit population std over the 330 cells), q = clamp(round(v*24))
     -> bytes [0, 330)
  6. chroma: 11x3 grid of 10x30 box means of the RGB base; s = R+G+B+3;
     o1 = (R-G)/s, o2 = (R+G-2B)/(2s); q = clamp(round(o*127)), interleaved (o1,o2) per cell
     -> bytes [330, 396)
  distance(a, b) = sum|ga-gb| + 0.5*sum|ca-cb|
At query time fp2.js additionally evaluates the query under 225 small geometric variants
(shift +-4% step 2%, scale 0.96/1/1.04, rotation -2/0/+2 deg) and takes the minimum;
query_variants() below mirrors that for the benchmark. fp2.js mirrors fingerprint() exactly.
"""
import argparse
import io
import json
import os
import random
import sys
import time
import urllib.parse
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DEFAULT_CACHE = r"C:\Users\Fabio\AppData\Local\Temp\claude\d395c8141171058b\be555625-db4a-4cfa-8cf3-32e7aeb800e3\scratchpad\imgcache"
CACHE = os.environ.get("IMGCACHE", DEFAULT_CACHE)

W, H = 630, 880
BH, BW = 110, 90          # base grid (8x7 px blocks)
GH, GW = 22, 15           # luma grid (5x6 base blocks)
CH, CW = 11, 3            # chroma grid (10x30 base blocks)
NG = GH * GW              # 330
NC = CH * CW * 2          # 66
RECORD_BYTES = NG + NC    # 396
G_SCALE, C_SCALE, C_WEIGHT = 24.0, 127.0, 0.5

GROUP = ("sm4-57 sm4-104 sm4-115 sv03-125 sv03-223 sv03-228 sv03.5-006 sv03.5-183 "
         "sv03.5-199 swsh1-211 sv01-256 sm4-120").split()


# ---------------------------------------------------------------- fingerprint
def to_rgb(img):
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        return bg.convert("RGB")
    return img.convert("RGB")


def card630(img):
    img = to_rgb(img)
    if img.size != (W, H):
        img = img.resize((W, H), Image.BILINEAR)
    return img


def base_of(img630):
    a = np.asarray(img630, dtype=np.float64)
    return a.reshape(BH, 8, BW, 7, 3).mean(axis=(1, 3))  # (110, 90, 3)


def _q(v, scale):
    return np.clip(np.floor(v * scale + 0.5), -127, 127).astype(np.int8)


def desc_from_base(b):
    """b: (..., 110, 90, 3) float base(s) -> (..., 396) int8."""
    lead = b.shape[:-3]
    y = b[..., 0] * 0.299 + b[..., 1] * 0.587 + b[..., 2] * 0.114
    g = y.reshape(*lead, GH, BH // GH, GW, BW // GW).mean(axis=(-3, -1))       # (...,22,15)
    p = np.pad(g, [(0, 0)] * len(lead) + [(2, 2), (2, 2)], mode="edge")
    acc = np.zeros_like(g)
    for dy in range(5):
        for dx in range(5):
            acc += p[..., dy:dy + GH, dx:dx + GW]
    g = (g - acc / 25.0).reshape(*lead, NG)
    g = g - g.mean(-1, keepdims=True)
    g = g / (np.sqrt((g * g).mean(-1, keepdims=True)) + 1e-6)
    c = b.reshape(*lead, CH, BH // CH, CW, BW // CW, 3).mean(axis=(-4, -2))     # (...,11,3,3)
    s = c.sum(-1) + 3.0
    o = np.stack([(c[..., 0] - c[..., 1]) / s, (c[..., 0] + c[..., 1] - 2 * c[..., 2]) / (2 * s)], -1)
    return np.concatenate([_q(g, G_SCALE), _q(o.reshape(*lead, NC), C_SCALE)], -1)


def fingerprint(img):
    return desc_from_base(base_of(card630(img)))


# ---------------------------------------------------------------- query variants (mirror of fp2.js)
def variant_params():
    sh = (0.0, -0.04, 0.04, -0.02, 0.02)
    first = [(dx, dy, 1.0, 0.0) for dx in (0.0, -0.04, 0.04) for dy in (0.0, -0.04, 0.04)]  # stage-1 set (9)
    rest = [(dx, dy, s, r) for dx in sh for dy in sh for s in (1.0, 0.96, 1.04) for r in (0.0, -2.0, 2.0)
            if (dx, dy, s, r) not in first]
    return first + rest  # 225, identity first


def _bilinear(img, X, Y):
    h, w = img.shape[:2]
    X = np.clip(X, 0, w - 1); Y = np.clip(Y, 0, h - 1)
    x0 = np.floor(X).astype(int); y0 = np.floor(Y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    fx = (X - x0)[..., None]; fy = (Y - y0)[..., None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def query_variants(img):
    b = base_of(card630(img))
    yy, xx = np.mgrid[0:BH, 0:BW]
    X = (xx + 0.5) * 7 - W / 2; Y = (yy + 0.5) * 8 - H / 2
    out = []
    for dx, dy, s, r in variant_params():
        if (dx, dy, s, r) == (0.0, 0.0, 1.0, 0.0):
            out.append(b); continue
        c, sn = np.cos(np.radians(r)), np.sin(np.radians(r))
        Xq = s * (c * X - sn * Y) + W / 2 + dx * W
        Yq = s * (sn * X + c * Y) + H / 2 + dy * H
        out.append(_bilinear(b, Xq / 7 - 0.5, Yq / 8 - 0.5))
    return desc_from_base(np.stack(out))  # (225, 396)


def dist_many(qv, db):
    """qv (V,396) int8, db (N,396) int8 -> (N,) min over variants."""
    db16 = db.astype(np.int16)
    best = None
    for v in qv.astype(np.int16):
        d = np.abs(db16[:, :NG] - v[:NG]).sum(1) + C_WEIGHT * np.abs(db16[:, NG:] - v[NG:]).sum(1)
        best = d if best is None else np.minimum(best, d)
    return best


def search(qv, db, k=5, n1=100):
    d1 = dist_many(qv[:9], db)
    cand = np.argpartition(d1, n1)[:n1]
    d2 = dist_many(qv, db[cand])
    o = np.argsort(d2, kind="stable")[:k]
    return cand[o], d2[o]


# ---------------------------------------------------------------- build
def img_path(lang, cid):
    return os.path.join(CACHE, lang, urllib.parse.quote(cid, safe="") + ".png")


def _fp_job(args):
    lang, cid = args
    p = img_path(lang, cid)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    return fingerprint(Image.open(p)).tobytes()


def build(lang):
    t0 = time.time()
    with open(os.path.join(DATA, f"ids-{lang}.json"), encoding="utf-8") as f:
        ids = json.load(f)
    with Pool(os.cpu_count()) as pool:
        recs = pool.map(_fp_job, [(lang, c) for c in ids], chunksize=32)
    missing = [c for c, r in zip(ids, recs) if r is None]
    out = os.path.join(DATA, f"fp2-{lang}.bin")
    with open(out, "wb") as f:
        for r in recs:
            f.write(r if r is not None else bytes(RECORD_BYTES))
    print(f"[{lang}] {len(ids)} records x {RECORD_BYTES} B -> {out} ({os.path.getsize(out)} bytes) "
          f"in {time.time() - t0:.0f}s; missing images (zero records): {len(missing)} {missing[:20]}", flush=True)


def load_db(lang):
    with open(os.path.join(DATA, f"ids-{lang}.json"), encoding="utf-8") as f:
        ids = json.load(f)
    raw = np.fromfile(os.path.join(DATA, f"fp2-{lang}.bin"), dtype=np.int8).reshape(-1, RECORD_BYTES)
    assert len(raw) == len(ids)
    return ids, raw


# ---------------------------------------------------------------- benchmark
def distort(img, rng):
    """Residual error after card detection + warp: geometry, blur, lighting, cast, glare, noise, JPEG."""
    s = rng.uniform(0.96, 1.04); ang = rng.uniform(-3, 3)
    dx = rng.uniform(-0.04, 0.04) * W; dy = rng.uniform(-0.04, 0.04) * H
    bg = tuple(int(rng.uniform(20, 200)) for _ in range(3))
    img = img.rotate(ang, resample=Image.BILINEAR, center=(W / 2, H / 2), translate=(dx, dy), fillcolor=bg)
    nw, nh = round(W * s), round(H * s)
    canvas = Image.new("RGB", (W, H), bg)
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((W - nw) // 2, (H - nh) // 2))
    img = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.5)))
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = a ** rng.uniform(0.85, 1.15)                                         # gamma
    a = a * rng.uniform(0.75, 1.15)                                          # brightness
    a = a * np.array([rng.uniform(0.9, 1.1) for _ in range(3)], np.float32)  # colour cast / white balance
    cx, cy = rng.uniform(0, W), rng.uniform(0, H); r = rng.uniform(50, 160); al = rng.uniform(0.3, 0.8)
    yy, xx = np.mgrid[0:H, 0:W]
    gl = al * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))      # glare blob
    a = a * (1 - gl[..., None]) + gl[..., None]
    a = a + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.02, a.shape).astype(np.float32)
    img = Image.fromarray(np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=60); buf.seek(0)
    return Image.open(buf).convert("RGB")


def _bench_job(args):
    cid, seed = args
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import build_hashes as bh
    img = distort(card630(Image.open(img_path("en", cid))), random.Random(seed))
    return query_variants(img), np.frombuffer(bh.fingerprint(img), np.uint8)


def bench(n_random=400, reps=5, seed=7):
    ids, db = load_db("en")
    idx = {c: i for i, c in enumerate(ids)}
    v1 = np.fromfile(os.path.join(DATA, "hashes-en.bin"), np.uint8).reshape(-1, 60)
    popc = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], 1).sum(1)
    rng = random.Random(seed)
    qs = [(c, 1000 + i) for i, c in enumerate(rng.sample(ids, n_random))]
    qs += [(c, 50000 + 10 * j + k) for j, c in enumerate(GROUP) for k in range(reps)]
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        res = pool.map(_bench_job, qs, chunksize=2)
    print(f"distorted {len(qs)} queries in {time.time() - t0:.0f}s", flush=True)
    stats = {}
    for (cid, sd), (qv, q1) in zip(qs, res):
        t = idx[cid]
        top2, _ = search(qv, db, k=5)
        d1 = 3 * popc[v1[:, :24] ^ q1[:24]].sum(1) + np.abs(v1[:, 24:].astype(int) - q1[24:]).sum(1) / 8
        top1v1 = np.argsort(d1, kind="stable")[:5]
        key = "group" if cid in GROUP and sd >= 50000 else "random"
        s = stats.setdefault(key, np.zeros(5))
        s += [1, top2[0] == t, t in top2, top1v1[0] == t, t in top1v1]
        if top2[0] != t:  # clean-record distance between truth and prediction: small = reprint with same art
            dd = dist_many(db[t:t + 1], db[top2[0]:top2[0] + 1])[0]
            print(f"  v2 miss {cid} -> {ids[top2[0]]}  (clean record distance {dd:.0f})", flush=True)
    for key, s in stats.items():
        n = s[0]
        print(f"{key:6s} n={int(n)}  v2 top1 {100*s[1]/n:.1f}% top5 {100*s[2]/n:.1f}%  |  "
              f"v1 top1 {100*s[3]/n:.1f}% top5 {100*s[4]/n:.1f}%", flush=True)
    s = sum(stats.values()); n = s[0]
    print(f"ALL    n={int(n)}  v2 top1 {100*s[1]/n:.1f}% top5 {100*s[2]/n:.1f}%  |  "
          f"v1 top1 {100*s[3]/n:.1f}% top5 {100*s[4]/n:.1f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "ja"], action="append")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if args.bench:
        bench()
    else:
        for lang in args.lang or ["en", "ja"]:
            build(lang)


if __name__ == "__main__":
    sys.exit(main())
