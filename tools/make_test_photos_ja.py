"""Realistic phone photos of JAPANESE cards -> tests/photos-ja/*.jpg + manifest.json
Reuses the renderer of make_test_photos.py.  Usage: python tools/make_test_photos_ja.py [--per-card 2]
"""
import argparse, json, os, random, sys, tempfile, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_test_photos as M

ROOT = M.ROOT
API = "https://api.tcgdex.net/v2/ja/cards/"
POPULAR = ["SV2a-201", "SV2a-006", "SV2a-025", "SV4a-349", "SV4a-237", "S8b-280", "SM12a-100", "SV3-139",
           "SV5K-087", "SV1V-099", "S12a-262", "SV6-125", "SV8-133", "SV7-120", "S11a-085"]


def tcgdex_img(cid):
    try:
        with urllib.request.urlopen(urllib.request.Request(API + cid, headers={"User-Agent": "test"}), timeout=30) as r:
            d = json.load(r)
        return d.get("image") and d["image"] + "/high.png"
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--per-card", type=int, default=2)
    ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "pkm_hicache"))
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    out = os.path.join(ROOT, "tests", "photos-ja"); os.makedirs(out, exist_ok=True)
    random.seed(a.seed)
    ids = json.load(open(os.path.join(ROOT, "data", "ids-ja.json"), encoding="utf-8"))
    extra = json.load(open(os.path.join(ROOT, "data", "extra-ja.json"), encoding="utf-8"))
    ex_tcg = [r for r in extra if not r["id"].startswith("jsrc:")]
    ex_src = [r for r in extra if r["id"].startswith("jsrc:")]
    cases = []
    for cid in POPULAR + random.sample(ids, 12):
        cases.append({"id": cid, "lang": "ja", "image": None})
    with ThreadPoolExecutor(8) as ex:
        imgs = list(ex.map(lambda c: tcgdex_img(c["id"]), cases))
    exmap = {r["id"]: r for r in extra}
    for c, u in zip(cases, imgs):
        c["image"] = u or (exmap.get(c["id"]) or {}).get("image")
    cases = [c for c in cases if c["image"]]
    for r in random.sample(ex_tcg, 10) + random.sample(ex_src, 5):
        cases.append({"id": r["id"], "lang": "ja", "image": r["image"].replace("/medium", "/large")})
    print(len(cases), "cards")
    def fetch(c):
        safe = c["id"].replace(":", "_")
        fn = os.path.join(a.cache, f"ja_{safe}.png")
        if not os.path.exists(fn) or os.path.getsize(fn) < 1000:
            with urllib.request.urlopen(urllib.request.Request(c["image"], headers={"User-Agent": "test"}), timeout=60) as r:
                open(fn, "wb").write(r.read())
        return fn
    with ThreadPoolExecutor(8) as ex:
        files = list(ex.map(lambda c: (lambda: fetch(c))() if True else None, cases))
    rng = np.random.default_rng(a.seed)
    man = []
    for c, fn in zip(cases, files):
        for k in range(a.per_card):
            img, P, diff = M.make_photo(fn, rng, (k + rng.random()) / a.per_card)
            name = f"ja_{c['id'].replace(':', '_')}_{k + 1}.jpg"
            img.save(os.path.join(out, name), "JPEG", quality=P["quality"])
            man.append({"file": name, "id": c["id"], "lang": "ja", "difficulty": diff, "params": P})
    json.dump(man, open(os.path.join(out, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("total", len(man))


if __name__ == "__main__":
    main()
