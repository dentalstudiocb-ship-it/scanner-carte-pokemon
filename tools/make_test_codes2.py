"""Code-strip test images (what the "Codice" box sees): bottom-left of the card, camera blur/noise.
Output tests/codes/*.jpg + manifest.json"""
import json, os, tempfile
import numpy as np
from PIL import Image, ImageFilter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(tempfile.gettempdir(), "pkm_hicache")
out = os.path.join(ROOT, "tests", "codes2"); os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(3)
man = []
seen = set()
for mf in ["photos-ja", "photos"]:
    for m in json.load(open(os.path.join(ROOT, "tests", mf, "manifest.json"), encoding="utf-8")):
        if m["id"] in seen: continue
        fn = os.path.join(CACHE, f"{m['lang']}_{m['id'].replace(':', '_')}.png")
        if not os.path.exists(fn): continue
        seen.add(m["id"])
        im = Image.open(fn).convert("RGB"); W, H = im.size
        x0, x1 = 0.0, W * rng.uniform(0.36, 0.46)
        h = (x1 - x0) / 3.75; y1 = H * rng.uniform(0.975, 0.995); y0 = max(0, y1 - h)
        s = im.crop((int(x0), int(y0), int(x1), int(y1)))
        tw = int(rng.uniform(800, 1150)); s = s.resize((tw, int(tw * s.height / s.width)), Image.BICUBIC)
        s = s.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.6)))
        a = np.asarray(s, np.float32) * rng.uniform(0.8, 1.1) + rng.normal(0, rng.uniform(3, 9), (s.height, s.width, 1))
        s = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        name = f"{m['lang']}_{m['id'].replace(':', '_')}.jpg"
        s.save(os.path.join(out, name), "JPEG", quality=int(rng.uniform(60, 85)))
        man.append({"file": name, "id": m["id"], "lang": m["lang"]})
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(len(man))
