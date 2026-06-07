"""
03d_build_game_meta.py
======================
Build a per-game metadata sidecar: patch (from gameVersion) + source tier (from
the harvester's game_source_tier.csv). This unblocks two things without a full
features.parquet reprocess:
  * PATCH-indexing the static-context encoder (champion stats for the game's patch)
  * RANK-conditioning (approx game elo as a feature / replacement-baseline bucket)

Output: data/processed/game_meta.parquet  (game_id, game_version, patch, source_tier)
Untagged games (the pre-multi-elo apex harvest) get source_tier='APEX_UNTAGGED'
(they are Challenger/GM).

Usage:
  conda run -n lol_shap_env python src/03d_build_game_meta.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATCH_DIR    = PROJECT_ROOT / "data" / "raw" / "matches"
TIER_FILE    = PROJECT_ROOT / "data" / "raw" / "game_source_tier.csv"
OUT          = PROJECT_ROOT / "data" / "processed" / "game_meta.parquet"
N_WORKERS    = max(1, (os.cpu_count() or 4) // 2)


def _read_version(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            info = json.load(f).get("info", {})
        gid = Path(path).stem.replace("match_", "")
        gv = info.get("gameVersion", "")
        return gid, gv
    except Exception:
        return None


def main():
    files = [str(p) for p in MATCH_DIR.glob("match_*.json")]
    print(f"Scanning {len(files)} match files for gameVersion ({N_WORKERS} workers)...")
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, r in enumerate(ex.map(_read_version, files, chunksize=200)):
            if r and r[1]:
                gid, gv = r
                rows.append({"game_id": gid, "game_version": gv,
                             "patch": ".".join(gv.split(".")[:2])})
            if (i + 1) % 20000 == 0:
                print(f"  {i+1}/{len(files)}")
    df = pd.DataFrame(rows)

    # merge source tier (approx game elo) from the harvester tag file
    tier = {}
    if TIER_FILE.exists():
        t = pd.read_csv(TIER_FILE, header=None, names=["game_id", "tier"])
        tier = dict(zip(t["game_id"], t["tier"]))
    df["source_tier"] = df["game_id"].map(lambda g: tier.get(g, "APEX_UNTAGGED"))

    df.to_parquet(OUT, index=False)
    print(f"\nSaved {OUT}  ({len(df)} games)")
    print("Patch distribution:\n", df["patch"].value_counts().sort_index().to_string())
    print("\nSource-tier distribution:\n", df["source_tier"].value_counts().to_string())


if __name__ == "__main__":
    main()
