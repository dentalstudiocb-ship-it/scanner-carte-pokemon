"""Generate realistic phone-camera test photos of Pokemon cards.

Output: tests/photos/*.jpg (1080x1440, portrait 3:4) + tests/photos/manifest.json
        [{file, id, lang, difficulty, params}]

Usage:  python tools/make_test_photos.py [--seed 1234] [--per-card 3] [--cache DIR]

Card images: high.png from TCGdex (cached on disk; --cache, default %TEMP%/pkm_hicache).
"""
import argparse, io, json, math, os, sys, tempfile, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FW, FH = 1080, 1440
CARD_AR = 88 / 63  # height / width

# 16 extra cards: gold/rainbow vs normal pairs, commons, vintage.
EXTRA = [
    ("sv01-081", "en", "sv/sv01/081"), ("sv01-253", "en", "sv/sv01/253"),      # Miraidon ex / gold
    ("swsh7-111", "en", "swsh/swsh7/111"), ("swsh7-217", "en", "swsh/swsh7/217"),  # Rayquaza VMAX / rainbow
    ("swsh4-44", "en", "swsh/swsh4/44"), ("swsh4-188", "en", "swsh/swsh4/188"),    # Pikachu VMAX / rainbow
    ("swsh9-018", "en", "swsh/swsh9/018"), ("swsh9-174", "en", "swsh/swsh9/174"),  # Charizard VSTAR / rainbow
    ("sm1-89", "en", "sm/sm1/89"), ("sm1-143", "en", "sm/sm1/143"),                # Solgaleo GX / full art
    ("sv03.5-025", "en", "sv/sv03.5/025"), ("sv03.5-173", "en", "sv/sv03.5/173"),  # Pikachu common / IR
    ("sv01-001", "en", "sv/sv01/001"), ("sv02-001", "en", "sv/sv02/001"),          # commons
    ("base1-58", "en", "base/base1/58"), ("base1-2", "en", "base/base1/2"),         # vintage
]


def load_cases():
    with open(os.path.join(ROOT, "tests", "cases.json"), encoding="utf-8") as f:
        cases = json.load(f)
    out = [{"id": c["id"], "lang": c["lang"], "image": c["image"]} for c in cases]
    for cid, lang, path in EXTRA:
        out.append({"id": cid, "lang": lang, "image": f"https://assets.tcgdex.net/{lang}/{path}/high.png"})
    return out


def fetch(case, cache):
    fn = os.path.join(cache, f"{case['lang']}_{case['id']}.png")
    if not os.path.exists(fn) or os.path.getsize(fn) < 1000:
        req = urllib.request.Request(case["image"], headers={"User-Agent": "test-photo-generator"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        with open(fn, "wb") as f:
            f.write(data)
    return fn


# ---------------------------------------------------------------- backgrounds
def smooth_noise(rng, h, w, scale):
    sh, sw = max(2, h // scale), max(2, w // scale)
    n = rng.random((sh, sw)).astype(np.float32)
    return np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 255


def bg_wood(rng):
    y, x = np.mgrid[0:FH, 0:FW].astype(np.float32)
    warp = smooth_noise(rng, FH, FW, 180) * 40 + smooth_noise(rng, FH, FW, 40) * 6
    ang = rng.uniform(-0.3, 0.3)
    u = x * math.cos(ang) + y * math.sin(ang) + warp
    freq = rng.uniform(0.04, 0.09)
    grain = 0.5 + 0.5 * np.sin(u * freq + smooth_noise(rng, FH, FW, 300) * 8)
    grain = grain ** 2 * 0.6 + smooth_noise(rng, FH, FW, 6) * 0.25 + 0.15
    base = np.array(rng.choice([[150, 100, 60], [120, 80, 45], [185, 140, 95], [90, 60, 40]]), np.float32)
    img = base[None, None, :] * (0.65 + 0.55 * grain[..., None])
    return img


def bg_cloth(rng):
    y, x = np.mgrid[0:FH, 0:FW].astype(np.float32)
    weave = (np.sin(x * 1.3) * np.sin(y * 1.3)) * 0.5 + 0.5
    n = rng.random((FH, FW)).astype(np.float32)
    v = 0.6 + 0.15 * weave + 0.25 * n + smooth_noise(rng, FH, FW, 250) * 0.4
    base = np.array(rng.choice([[30, 30, 34], [22, 28, 45], [40, 25, 25], [25, 38, 28]]), np.float32)
    return base[None, None, :] * v[..., None]


def bg_table(rng):
    y, x = np.mgrid[0:FH, 0:FW].astype(np.float32)
    grad = 1 - 0.12 * ((x - rng.uniform(0, FW)) ** 2 + (y - rng.uniform(0, FH)) ** 2) / (FW * FW + FH * FH)
    v = grad * (0.97 + 0.03 * smooth_noise(rng, FH, FW, 120)) + rng.normal(0, 0.008, (FH, FW))
    base = np.array(rng.choice([[238, 236, 232], [225, 228, 230], [245, 242, 235]]), np.float32)
    return base[None, None, :] * v[..., None]


def bg_pattern(rng):
    y, x = np.mgrid[0:FH, 0:FW].astype(np.float32)
    kind = rng.integers(0, 3)
    p = rng.uniform(40, 110)
    c1 = np.array(rng.integers(30, 230, 3), np.float32)
    c2 = np.array(rng.integers(30, 230, 3), np.float32)
    ang = rng.uniform(0, math.pi)
    u = x * math.cos(ang) + y * math.sin(ang)
    v = -x * math.sin(ang) + y * math.cos(ang)
    if kind == 0:  # checks / gingham
        m = ((np.floor(u / p) + np.floor(v / p)) % 2).astype(np.float32)
    elif kind == 1:  # stripes
        m = (np.sin(u / p * math.pi) > 0).astype(np.float32)
    else:  # polka dots
        m = ((((u % p) - p / 2) ** 2 + ((v % p) - p / 2) ** 2) < (p * 0.25) ** 2).astype(np.float32)
    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
    img = c1 * (1 - m[..., None]) + c2 * m[..., None]
    return img * (0.9 + 0.1 * smooth_noise(rng, FH, FW, 200))[..., None]


BGS = {"wood": bg_wood, "cloth": bg_cloth, "table": bg_table, "pattern": bg_pattern}


# ---------------------------------------------------------------- geometry
def perspective_coeffs(dst, src):
    """Coefficients for Image.transform(PERSPECTIVE): maps output (dst) points -> input (src) points."""
    A, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    return np.linalg.solve(np.array(A, np.float64), np.array(b, np.float64)).tolist()


def rounded_mask(w, h, r, ss=4):
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=r * ss, fill=255)
    return m.resize((w, h), Image.LANCZOS)


# ---------------------------------------------------------------- one photo
def make_photo(card_png, rng, sev):
    """sev in [0,1): severity bias; parameters are random but drift harder with sev."""
    def pick(lo, hi, harder_is_high=True, spread=0.45):
        t = np.clip(sev + rng.uniform(-spread, spread), 0, 1)
        if not harder_is_high:
            t = 1 - t
        return lo + (hi - lo) * t

    P = {}
    P["bg"] = str(rng.choice(list(BGS)))
    P["scale"] = round(float(pick(0.45, 0.80, harder_is_high=False)), 3)
    P["rot"] = round(float(rng.choice([-1, 1]) * pick(0, 7)), 2)
    P["tilt"] = round(float(pick(0, 0.06)), 4)
    P["bright"] = round(float(1.0 + rng.choice([-1, 1]) * pick(0, 0.2) if rng.random() < 0.5 else rng.uniform(0.75, 1.15)), 3)
    P["bright"] = float(np.clip(P["bright"], 0.75, 1.15))
    P["temp"] = round(float(rng.uniform(-1, 1) * (0.4 + 0.6 * sev)), 3)  # -1 cool .. +1 warm
    P["glare"] = bool(rng.random() < 0.5)
    P["glare_strength"] = round(float(pick(0.25, 0.75)), 3) if P["glare"] else 0.0
    P["blur"] = round(float(pick(0.5, 1.5)), 3)
    P["noise"] = round(float(pick(2.0, 9.0)), 2)
    P["quality"] = int(round(pick(75, 55, harder_is_high=True)))

    bg = BGS[P["bg"]](rng)

    # card raster at the target size (slightly larger, for clean resampling)
    cw = int(FW * P["scale"])
    ch = int(round(cw * CARD_AR))
    card = Image.open(card_png).convert("RGB")
    src_w, src_h = card.size
    card_mask = rounded_mask(src_w, src_h, int(src_w * 0.045))

    # destination quad: rect centred, in-plane rotation, perspective corner jitter
    half = np.array([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]])
    a = math.radians(P["rot"])
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    # perspective (in card-local coords, before rotation): keystone pitch/yaw + corner jitter.
    # corners TL, TR, BR, BL; each corner moves at most ~tilt * card size.
    t = P["tilt"]
    pitch, yaw = rng.uniform(-1, 1, 2) * t * 0.5
    px, qy = pitch * cw, yaw * ch
    keystone = np.array([      # pitch: top narrows / bottom widens; yaw: left shrinks / right grows
        [ px,  qy],   # TL
        [-px, -qy],   # TR
        [ px,  qy],   # BR
        [-px, -qy],   # BL
    ])
    jitter = rng.uniform(-1, 1, (4, 2)) * t * 0.5 * np.array([cw, ch])
    quad = (half + keystone + jitter) @ R.T

    # position: centre within +-12% of frame centre, then nudge to stay inside frame
    cx = FW / 2 + rng.uniform(-0.12, 0.12) * FW
    cy = FH / 2 + rng.uniform(-0.12, 0.12) * FH
    pts = quad + [cx, cy]
    margin = 12
    dx = max(0, margin - pts[:, 0].min()) - max(0, pts[:, 0].max() - (FW - margin))
    dy = max(0, margin - pts[:, 1].min()) - max(0, pts[:, 1].max() - (FH - margin))
    pts = pts + [dx, dy]
    P["center"] = [round(float(pts[:, 0].mean() / FW), 3), round(float(pts[:, 1].mean() / FH), 3)]
    P["quad"] = [[round(float(x), 1), round(float(y), 1)] for x, y in pts]

    src = [(0, 0), (src_w, 0), (src_w, src_h), (0, src_h)]
    coeffs = perspective_coeffs([tuple(p) for p in pts], src)
    warped = card.transform((FW, FH), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    wmask = card_mask.transform((FW, FH), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=0)
    wmask = wmask.filter(ImageFilter.GaussianBlur(0.6))
    m = np.asarray(wmask, np.float32)[..., None] / 255
    cimg = np.asarray(warped, np.float32)

    # soft drop shadow on the background
    sh = wmask.filter(ImageFilter.GaussianBlur(14))
    sh = np.roll(np.asarray(sh, np.float32) / 255, (int(rng.uniform(4, 14)), int(rng.uniform(-8, 8))), (0, 1))
    bg = bg * (1 - 0.35 * sh[..., None])

    # glare blob on the card (additive, follows card mask)
    if P["glare"]:
        gx, gy = pts[:, 0].min() + rng.uniform(0.2, 0.8) * np.ptp(pts[:, 0]), pts[:, 1].min() + rng.uniform(0.15, 0.85) * np.ptp(pts[:, 1])
        sx, sy = cw * rng.uniform(0.12, 0.3), cw * rng.uniform(0.08, 0.25)
        yy, xx = np.mgrid[0:FH, 0:FW].astype(np.float32)
        g = np.exp(-(((xx - gx) / sx) ** 2 + ((yy - gy) / sy) ** 2) / 2)
        cimg = cimg + (255 - cimg) * (g * P["glare_strength"])[..., None]

    img = bg * (1 - m) + cimg * m

    # lighting: brightness, colour temperature, vignette
    img = img * P["bright"]
    tt = P["temp"]
    img = img * np.array([1 + 0.10 * tt, 1 + 0.02 * tt, 1 - 0.12 * tt], np.float32)
    yy, xx = np.mgrid[0:FH, 0:FW].astype(np.float32)
    r2 = ((xx - FW / 2) / (FW / 2)) ** 2 + ((yy - FH / 2) / (FH / 2)) ** 2
    img = img * (1 - rng.uniform(0.12, 0.3) * np.clip(r2 / 2, 0, 1))[..., None]

    # camera: blur, sensor noise (luma + chroma), jpeg
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(P["blur"]))
    a = np.asarray(out, np.float32)
    a = a + rng.normal(0, P["noise"], (FH, FW, 1)) + rng.normal(0, P["noise"] * 0.4, (FH, FW, 3))
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    # difficulty score in ~[0,1]
    s = (
        0.30 * (0.80 - P["scale"]) / 0.35
        + 0.12 * abs(P["rot"]) / 7
        + 0.13 * P["tilt"] / 0.06
        + 0.15 * (P["glare_strength"] / 0.75)
        + 0.10 * (P["blur"] - 0.5)
        + 0.08 * abs(P["bright"] - 1) / 0.25
        + 0.05 * abs(P["temp"])
        + 0.07 * (75 - P["quality"]) / 20
    )
    P["score"] = round(float(s), 3)
    diff = "easy" if s < 0.35 else ("normal" if s < 0.58 else "hard")
    return out, P, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--per-card", type=int, default=3)
    ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "pkm_hicache"))
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    outdir = os.path.join(ROOT, "tests", "photos")
    os.makedirs(outdir, exist_ok=True)

    cases = load_cases()
    print(f"{len(cases)} cards, downloading/caching images ...")
    with ThreadPoolExecutor(8) as ex:
        files = list(ex.map(lambda c: fetch(c, args.cache), cases))

    rng = np.random.default_rng(args.seed)
    manifest = []
    for c, fn in zip(cases, files):
        for k in range(args.per_card):
            sev = (k + rng.random()) / args.per_card
            img, P, diff = make_photo(fn, rng, sev)
            name = f"{c['lang']}_{c['id']}_{k + 1}.jpg"
            img.save(os.path.join(outdir, name), "JPEG", quality=P["quality"])
            manifest.append({"file": name, "id": c["id"], "lang": c["lang"], "difficulty": diff, "params": P})
            print(f"  {name:32s} {diff:6s} score={P['score']:.2f} scale={P['scale']:.2f} bg={P['bg']}")
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    from collections import Counter
    print("total", len(manifest), dict(Counter(m["difficulty"] for m in manifest)))


if __name__ == "__main__":
    sys.exit(main())
