"""data/abbr-en.json: printed set abbreviation (SVI, PAF, OBF…) -> TCGdex set id(s)."""
import json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
get = lambda u: json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "abbr"}), timeout=60))
sets = get("https://api.tcgdex.net/v2/en/sets")
def one(s):
    try: return s["id"], (get("https://api.tcgdex.net/v2/en/sets/" + s["id"]).get("abbreviation") or {}).get("official")
    except Exception: return s["id"], None
with ThreadPoolExecutor(6) as ex: res = list(ex.map(one, sets))
m = {}
for sid, ab in res:
    if ab: m.setdefault(ab.upper(), []).append(sid)
json.dump(m, open(os.path.join(ROOT, "data", "abbr-en.json"), "w"), separators=(",", ":"))
print(len(sets), "sets,", len(m), "abbreviations", list(m.items())[-8:])
