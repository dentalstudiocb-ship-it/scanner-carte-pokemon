"""Photos of Delta Reign base vs full-art pairs -> tests/photos-me06"""
import json, os, sys, urllib.request, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_test_photos as M
out = os.path.join(M.ROOT, "tests", "photos-me06"); os.makedirs(out, exist_ok=True)
cache = os.path.join(tempfile.gettempdir(), "pkm_hicache")
ids = ["me06-080", "me06-115", "me06-011", "me06-118", "me06-135", "me06-003", "me06-104"]
rng = np.random.default_rng(9); man = []
for cid in ids:
    fn = os.path.join(cache, f"en_{cid}.png")
    if not os.path.exists(fn):
        n = cid.split("-")[1]
        open(fn, "wb").write(urllib.request.urlopen(urllib.request.Request(f"https://assets.tcgdex.net/en/me/me06/{n}/high.png", headers={"User-Agent": "t"}), timeout=60).read())
    for k in range(2):
        img, P, d = M.make_photo(fn, rng, (k + rng.random()) / 2)
        name = f"en_{cid}_{k+1}.jpg"; img.save(os.path.join(out, name), "JPEG", quality=P["quality"])
        man.append({"file": name, "id": cid, "lang": "en", "difficulty": d})
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1); print(len(man))
