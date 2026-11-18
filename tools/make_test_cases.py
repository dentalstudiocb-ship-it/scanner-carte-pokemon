"""Phone photos of cards inside a TOPLOADER or a SLEEVE (plastic haze, rim, glare streaks, card resting low).
Output: tests/photos-case/*.jpg + manifest.json.  Japanese + English cards.
Usage: python tools/make_test_cases.py
"""
import json, math, os, random, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_test_photos as M

ROOT = M.ROOT; FW, FH = M.FW, M.FH
CACHE = os.path.join(tempfile.gettempdir(), "pkm_hicache")


def case_layer(card_png, kind, rng):
    """RGBA image of the case with the card inside (scale: 10 px per mm)."""
    if kind == "toploader":
        W, H = 760, 1020
    else:
        W, H = 668, 925
    card = Image.open(card_png).convert("RGB").resize((630, 880), Image.LANCZOS)
    cmask = M.rounded_mask(630, 880, 28)
    L = np.zeros((H, W, 4), np.float32)
    # plastic haze
    L[..., :3] = 235; L[..., 3] = 0.10 if kind == "toploader" else 0.07
    # rim
    rim = 14 if kind == "toploader" else 4
    yy, xx = np.mgrid[0:H, 0:W]
    edge = np.minimum.reduce([xx, yy, W - 1 - xx, H - 1 - yy])
    L[..., 3] = np.where(edge < rim, 0.45 + 0.25 * (1 - edge / rim), L[..., 3])
    if kind == "toploader":  # opening at the top: brighter lip
        L[..., 3] = np.where((yy < 30) & (edge >= rim), 0.25, L[..., 3])
    # card position: resting at the bottom (toploader) / anywhere (sleeve)
    cx = (W - 630) // 2 + int(rng.uniform(-1, 1) * (W - 630) * 0.35)
    if kind == "toploader":
        cy = H - 880 - rim - int(rng.uniform(0, 30)) if rng.random() < 0.7 else int(rng.uniform(rim, H - 880 - rim))
    else:
        cy = (H - 880) // 2 + int(rng.uniform(-15, 15))
    a = np.asarray(cmask, np.float32) / 255
    L[cy:cy + 880, cx:cx + 630, :3] = L[cy:cy + 880, cx:cx + 630, :3] * (1 - a[..., None]) + np.asarray(card, np.float32) * a[..., None]
    L[cy:cy + 880, cx:cx + 630, 3] = np.maximum(L[cy:cy + 880, cx:cx + 630, 3], a)
    # haze over the card (plastic in front)
    haze = 0.06 + rng.uniform(0, 0.08)
    L[..., :3] = L[..., :3] * (1 - haze) + 240 * haze
    # glare streaks
    for _ in range(int(rng.integers(1, 3))):
        ang = rng.uniform(-0.9, 0.9); off = rng.uniform(-0.4, 0.4) * W; wdt = rng.uniform(20, 90)
        d = (xx - W / 2) * math.cos(ang) + (yy - H / 2) * math.sin(ang) - off
        g = np.exp(-(d / wdt) ** 2) * rng.uniform(0.25, 0.6)
        L[..., :3] = L[..., :3] + (255 - L[..., :3]) * g[..., None]
        L[..., 3] = np.maximum(L[..., 3], g * 0.8)
    if kind == "sleeve":  # wrinkle lines
        for _ in range(3):
            y0 = rng.uniform(0, H); d = np.abs(yy - y0 - (xx - W / 2) * rng.uniform(-0.2, 0.2))
            g = np.exp(-(d / 3) ** 2) * 0.3
            L[..., :3] = L[..., :3] + (255 - L[..., :3]) * g[..., None]
    L[..., 3] = np.clip(L[..., 3], 0, 1) * 255
    return Image.fromarray(np.clip(L, 0, 255).astype(np.uint8), "RGBA")


def photo(card_png, kind, rng):
    layer = case_layer(card_png, kind, rng)
    bgname = str(rng.choice(list(M.BGS)))
    bg = M.BGS[bgname](rng)
    sw, sh = layer.size
    scale = rng.uniform(0.6, 0.85)
    cw = FW * scale; ch = cw * sh / sw
    if ch > FH * 0.93: ch = FH * 0.93; cw = ch * sw / sh
    rot = math.radians(rng.uniform(-6, 6)); t = rng.uniform(0, 0.05)
    half = np.array([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]])
    half = half + rng.uniform(-1, 1, (4, 2)) * t * 0.5 * np.array([cw, ch])
    R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    pts = half @ R.T + [FW / 2 + rng.uniform(-0.06, 0.06) * FW, FH / 2 + rng.uniform(-0.05, 0.05) * FH]
    m = 8
    pts = pts + [max(0, m - pts[:, 0].min()) - max(0, pts[:, 0].max() - (FW - m)), max(0, m - pts[:, 1].min()) - max(0, pts[:, 1].max() - (FH - m))]
    co = M.perspective_coeffs([tuple(p) for p in pts], [(0, 0), (sw, 0), (sw, sh), (0, sh)])
    w = np.asarray(layer.transform((FW, FH), Image.PERSPECTIVE, co, Image.BICUBIC), np.float32)
    a = w[..., 3:] / 255
    img = bg * (1 - a) + w[..., :3] * a
    img = img * rng.uniform(0.8, 1.1)
    tt = rng.uniform(-0.8, 0.8)
    img = img * np.array([1 + 0.10 * tt, 1 + 0.02 * tt, 1 - 0.12 * tt], np.float32)
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.3)))
    a2 = np.asarray(out, np.float32) + rng.normal(0, rng.uniform(2, 7), (FH, FW, 1))
    return Image.fromarray(np.clip(a2, 0, 255).astype(np.uint8)), bgname


def main():
    rng = np.random.default_rng(5)
    out = os.path.join(ROOT, "tests", "photos-case"); os.makedirs(out, exist_ok=True)
    ja = json.load(open(os.path.join(ROOT, "tests", "photos-ja", "manifest.json"), encoding="utf-8"))
    en = json.load(open(os.path.join(ROOT, "tests", "photos", "manifest.json"), encoding="utf-8"))
    def src(m):
        fn = os.path.join(CACHE, f"{m['lang']}_{m['id'].replace(':', '_')}.png")
        return fn if os.path.exists(fn) else None
    seen = set(); cards = []
    for m in ja + en:
        if m["id"] in seen or not src(m): continue
        seen.add(m["id"]); cards.append(m)
    jac = [c for c in cards if c["lang"] == "ja"][:24]; enc = [c for c in cards if c["lang"] != "ja"][:12]
    man = []
    for c in jac + enc:
        for kind in ("toploader", "sleeve"):
            im, bgname = photo(src(c), kind, rng)
            name = f"{kind}_{c['lang']}_{c['id'].replace(':', '_')}.jpg"
            im.save(os.path.join(out, name), "JPEG", quality=int(rng.uniform(60, 80)))
            man.append({"file": name, "id": c["id"], "lang": c["lang"], "difficulty": kind, "params": {"bg": bgname}})
    json.dump(man, open(os.path.join(out, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("total", len(man))


if __name__ == "__main__":
    main()
