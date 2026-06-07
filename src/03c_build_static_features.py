"""
03c_build_static_features.py
============================
Build a champion STATIC-FEATURE table from Data Dragon (Phase 1 of the
patch/static-context encoder). For each champion: base stats + per-level growth,
the info ratings, class tags (multi-hot), and resource type (one-hot). Numerics
are z-scored across champions; tags/partype left as 0/1 indicators.

Output: data/processed/champion_static.parquet  (champion_key + feature columns)
keyed by `champion_key` (the numeric id == our `*_champion_id`).

v1 uses the LATEST downloaded patch (champ stats barely move across 16.x, and our
data spans only 16.x+15.24). Patch-indexing (per-game patch lookup) is the small
follow-up once `patch` is a feature — see reports/static_context_plan.md.

Usage:
  conda run -n lol_shap_env python src/03c_build_static_features.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR   = PROJECT_ROOT / "data" / "raw" / "static"
OUT          = PROJECT_ROOT / "data" / "processed" / "champion_static.parquet"

STAT_KEYS = [
    "hp", "hpperlevel", "mp", "mpperlevel", "movespeed", "armor", "armorperlevel",
    "spellblock", "spellblockperlevel", "attackrange", "hpregen", "hpregenperlevel",
    "mpregen", "mpregenperlevel", "crit", "critperlevel", "attackdamage",
    "attackdamageperlevel", "attackspeedperlevel", "attackspeed",
]
INFO_KEYS = ["attack", "defense", "magic", "difficulty"]


def main():
    ver = (STATIC_DIR / "LATEST.txt").read_text().strip()
    data = json.load(open(STATIC_DIR / ver / "championFull.json"))["data"]
    print(f"Champion static features from patch {ver}: {len(data)} champions")

    rows = []
    all_tags, all_partypes = set(), set()
    for c in data.values():
        all_tags.update(c.get("tags", []))
        all_partypes.add(c.get("partype", "None") or "None")
    tags = sorted(all_tags); partypes = sorted(all_partypes)

    for c in data.values():
        row = {"champion_key": int(c["key"]), "champion_name": c["id"]}
        for k in STAT_KEYS:
            row[f"stat_{k}"] = float(c["stats"].get(k, 0.0))
        for k in INFO_KEYS:
            row[f"info_{k}"] = float(c.get("info", {}).get(k, 0.0))
        for t in tags:
            row[f"tag_{t}"] = 1.0 if t in c.get("tags", []) else 0.0
        pt = c.get("partype", "None") or "None"
        for p in partypes:
            row[f"partype_{p}"] = 1.0 if p == pt else 0.0
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("champion_key").reset_index(drop=True)

    # z-score the numeric (stat_/info_) columns across champions; leave indicators.
    num_cols = [c for c in df.columns if c.startswith(("stat_", "info_"))]
    df[num_cols] = (df[num_cols] - df[num_cols].mean()) / (df[num_cols].std() + 1e-6)

    df.to_parquet(OUT, index=False)
    feat_cols = [c for c in df.columns if c not in ("champion_key", "champion_name")]
    print(f"Saved {OUT}  ({len(df)} champions x {len(feat_cols)} features)")
    print(f"  numeric: {len(num_cols)} | tags: {len(tags)} {tags} | partype: {len(partypes)} {partypes}")


if __name__ == "__main__":
    main()
