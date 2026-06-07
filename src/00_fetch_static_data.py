"""
00_fetch_static_data.py
=======================
Fetch versioned static game data (Data Dragon) for the patch/static-context
encoder. For each requested patch (major.minor, e.g. "16.5") we resolve the
matching Data Dragon version (e.g. "16.5.1") and download:
  championFull.json  - champion base stats + per-level growth, tags, info, text
  item.json          - item gold, stat-mods, tags, text
  runesReforged.json - rune trees / slots / text
  summoner.json      - summoner spells

Data Dragon is free, no API key, no rate limit. Output: data/raw/static/<ver>/.

NOT covered by Data Dragon (TODO, separate source): neutral monsters (baron,
dragons, herald, void grubs, jungle camps), minions, turrets/objectives — these
are game constants; use CommunityDragon or a small curated per-patch table.

Usage:
  conda run -n lol_shap_env python src/00_fetch_static_data.py            # our data patches
  conda run -n lol_shap_env python src/00_fetch_static_data.py --patches 16.11,16.5
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "data" / "raw" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

DDRAGON = "https://ddragon.leagueoflegends.com"
FILES = ["championFull", "item", "runesReforged", "summoner"]
# Patches present in our harvested data (extend as the harvest grows).
DEFAULT_PATCHES = ["15.24", "16.1", "16.2", "16.3", "16.4", "16.5",
                   "16.6", "16.7", "16.8", "16.9", "16.10", "16.11"]


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=str, default=",".join(DEFAULT_PATCHES),
                    help="comma list of major.minor patches, or 'latest'")
    args = ap.parse_args()

    versions = json.loads(get(f"{DDRAGON}/api/versions.json"))
    print(f"Data Dragon has {len(versions)} versions; latest {versions[0]}")

    if args.patches.strip().lower() == "latest":
        want = [versions[0].rsplit(".", 1)[0]]
    else:
        want = [p.strip() for p in args.patches.split(",") if p.strip()]

    for mm in want:
        # resolve major.minor -> first matching ddragon version (e.g. 16.5 -> 16.5.1)
        match = next((v for v in versions if v.startswith(mm + ".") or v == mm), None)
        if not match:
            print(f"  [{mm}] no Data Dragon version found — skipping")
            continue
        out = STATIC_DIR / match
        out.mkdir(exist_ok=True)
        sizes = []
        for f in FILES:
            try:
                data = get(f"{DDRAGON}/cdn/{match}/data/en_US/{f}.json")
                (out / f"{f}.json").write_bytes(data)
                sizes.append(f"{f}={len(data)//1024}KB")
            except Exception as e:
                sizes.append(f"{f}=ERR({e})")
        print(f"  [{mm}] -> {match}: " + ", ".join(sizes))

    (STATIC_DIR / "LATEST.txt").write_text(versions[0])
    print(f"Done. Static data in {STATIC_DIR}/")


if __name__ == "__main__":
    main()
