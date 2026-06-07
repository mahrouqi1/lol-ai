"""
03c_build_static_features.py
============================
Build a champion STATIC-FEATURE table from Data Dragon (Phase 1 of the
patch/static-context encoder). For each champion: base stats + per-level growth,
the info ratings, class tags (multi-hot), and resource type (one-hot). Numerics
are z-scored across champions; tags/partype left as 0/1 indicators.

PATCH-INDEXED (Phase 1.5): one row per (patch, champion), built from EVERY
downloaded `data/raw/static/<ver>/championFull.json`. The `patch` column is the
two-component patch ("16.2") derived from the version dir ("16.2.1"). 04f --static
looks up each game's patch via game_meta.parquet and feeds the patch-correct
champion stats per node, so cross-patch stat drift (buffs/nerfs) becomes signal
rather than being averaged away.

Numerics are z-scored with a SINGLE globally-pooled (mean, std) over all
(patch, champion) rows — NOT per-patch — so a champion buffed in a later patch
moves relative to its own earlier self (cross-patch drift preserved) AND
within-patch champion differences are kept. Per-patch z-scoring would erase the
cross-patch signal this whole step exists to capture.

Output: data/processed/champion_static.parquet
  columns: patch, champion_key, champion_name, <feature cols>
  keyed by (patch, champion_key); champion_key is the numeric id == *_champion_id.

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


def patch_of(ver: str) -> str:
    """Version dir '16.2.1' -> patch '16.2'."""
    return ".".join(ver.split(".")[:2])


def main():
    ver_dirs = sorted([p.name for p in STATIC_DIR.iterdir() if p.is_dir()])
    if not ver_dirs:
        raise SystemExit(f"No static patch dirs under {STATIC_DIR} (run 00_fetch_static_data.py)")
    print(f"Building per-patch champion static features from {len(ver_dirs)} patches: {ver_dirs}")

    # First pass: union tags / partypes across ALL patches (champ roster grows).
    all_tags, all_partypes = set(), set()
    per_patch_data = {}
    for ver in ver_dirs:
        jf = STATIC_DIR / ver / "championFull.json"
        if not jf.exists():
            print(f"  skip {ver}: no championFull.json")
            continue
        data = json.load(open(jf))["data"]
        per_patch_data[ver] = data
        for c in data.values():
            all_tags.update(c.get("tags", []))
            all_partypes.add(c.get("partype", "None") or "None")
    tags = sorted(all_tags); partypes = sorted(all_partypes)

    rows = []
    for ver, data in per_patch_data.items():
        patch = patch_of(ver)
        for c in data.values():
            row = {"patch": patch, "champion_key": int(c["key"]), "champion_name": c["id"]}
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

    df = pd.DataFrame(rows).sort_values(["patch", "champion_key"]).reset_index(drop=True)

    # z-score numeric (stat_/info_) cols with ONE globally-pooled mean/std across
    # all (patch, champion) rows so cross-patch drift survives normalization.
    num_cols = [c for c in df.columns if c.startswith(("stat_", "info_"))]
    df[num_cols] = (df[num_cols] - df[num_cols].mean()) / (df[num_cols].std() + 1e-6)

    df.to_parquet(OUT, index=False)
    feat_cols = [c for c in df.columns if c not in ("patch", "champion_key", "champion_name")]
    n_patch = df["patch"].nunique()
    print(f"Saved {OUT}  ({len(df)} rows = {n_patch} patches x ~{len(df)//n_patch} champs, "
          f"{len(feat_cols)} features)")
    print(f"  patches: {sorted(df['patch'].unique())}")
    print(f"  numeric: {len(num_cols)} | tags: {len(tags)} {tags} | partype: {len(partypes)} {partypes}")


if __name__ == "__main__":
    main()
