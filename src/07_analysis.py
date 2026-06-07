"""
07_analysis.py
==============
Extended SHAP analysis: role-level, champion-level, and champion-normalised contributions.

Reads shap_contributions.parquet (produced by 06_shap_explain.py) and joins
champion IDs back from features.parquet to enable champion/role breakdowns.

Sign convention
---------------
  SHAP values are in log-odds space for blue_win=1.
  For fair champion comparison we flip the sign for red-side players so that
  positive always means "helped your own team win".

Analyses
--------
1. Role importance          — mean |SHAP| per role (top/jungle/mid/bot/utility)
2. Champion contribution    — mean signed SHAP per (champion, role)
3. Champion-normalised z-score — deviation from champion-role baseline per player-game

Outputs
-------
  data/processed/shap_long.parquet             - long-format (one row per game-minute-slot)
  data/processed/champion_role_stats.parquet   - per (champion_id, role) mean/std/count
  data/processed/player_zscores.parquet        - per (game_id, minute, slot) z-score
  reports/analysis_role_importance.png
  reports/analysis_champion_contribution.png   - top-N champs per role (5 subplots)
  reports/analysis_zscore_distribution.png     - z-score sanity histogram

Usage
-----
    conda activate lol_shap_env
    python src/07_analysis.py
    python src/07_analysis.py --top-champs 15   # show top 15 champs per role plot
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
FEATURES_PATH  = PROJECT_ROOT / "data" / "processed" / "features.parquet"
CONTRIB_PATH   = PROJECT_ROOT / "data" / "processed" / "shap_contributions.parquet"
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR    = PROJECT_ROOT / "reports"
LOG_DIR        = PROJECT_ROOT / "logs"

for _d in (PROCESSED_DIR, REPORTS_DIR, LOG_DIR):
    _d.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "07_analysis.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SLOTS = [
    "blue_top", "blue_jungle", "blue_middle", "blue_bottom", "blue_utility",
    "red_top",  "red_jungle",  "red_middle",  "red_bottom",  "red_utility",
]
ROLES = ["top", "jungle", "middle", "bottom", "utility"]

# For champion names: try DDragon, fall back to numeric IDs
DDRAGON_VERSION = "15.1.1"   # recent stable; names rarely change between patches
DDRAGON_URL     = f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VERSION}/data/en_US/champion.json"

MIN_GAMES_PER_CHAMP = 5   # require at least this many appearances for z-score baseline

# Region labels: these must match REGION_INT_MAP in 03_process_features.py
REGION_LABELS: dict[int, str] = {0: "NA1", 1: "EUW1", 2: "KR", -1: "Unknown"}

ROLE_COLORS = {
    "top":     "#E53935",
    "jungle":  "#43A047",
    "middle":  "#1E88E5",
    "bottom":  "#FB8C00",
    "utility": "#8E24AA",
}

# ── Champion name lookup ───────────────────────────────────────────────────────

def load_champion_names() -> dict[int, str]:
    """
    Fetch champion id -> name mapping from DDragon.
    Falls back to empty dict (IDs used as labels) if network unavailable.
    """
    try:
        resp = requests.get(DDRAGON_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"]
        # DDragon key is name string, value has "key" (numeric id as string)
        mapping = {int(v["key"]): v["name"] for v in data.values()}
        log.info("Loaded %d champion names from DDragon", len(mapping))
        return mapping
    except Exception as exc:
        log.warning("Could not load champion names (%s); using IDs instead.", exc)
        return {}


# ── Data preparation ──────────────────────────────────────────────────────────

def build_long_format(contrib_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt wide-format contributions into one row per (game_id, minute, slot).

    Added columns:
      role            — top/jungle/middle/bottom/utility
      team            — blue/red
      team_sign       — +1 for blue, -1 for red
      champion_id     — integer champion ID for that slot
      signed_contrib  — contribution * team_sign  (positive = helped own team)
      region          — region string extracted from game_id prefix (NA1/EUW1/KR/Unknown)
    """
    # Keep only champion_id columns from features (game_id, minute, + 10 champ cols)
    champ_cols = [f"{slot}_champion_id" for slot in SLOTS]
    feat_slim  = features_df[["game_id", "minute"] + champ_cols].copy()

    # Join contributions with champion IDs
    merged = contrib_df.merge(feat_slim, on=["game_id", "minute"], how="inner")

    rows = []
    for slot in SLOTS:
        team      = slot.split("_")[0]          # blue / red
        role      = "_".join(slot.split("_")[1:])  # top / jungle / middle / bottom / utility
        team_sign = 1 if team == "blue" else -1

        sub = merged[["game_id", "minute", "blue_win", slot, f"{slot}_champion_id"]].copy()
        sub = sub.rename(columns={slot: "contribution", f"{slot}_champion_id": "champion_id"})
        sub["slot"]           = slot
        sub["team"]           = team
        sub["role"]           = role
        sub["team_sign"]      = team_sign
        sub["signed_contrib"] = sub["contribution"] * team_sign
        rows.append(sub)

    long_df = pd.concat(rows, ignore_index=True)

    # Extract region from game_id prefix (e.g. "NA1_XXXXX" -> "NA1")
    # Works whether region came from features.parquet or is derived here at analysis time.
    long_df["region"] = (
        long_df["game_id"].str.split("_").str[0].str.upper()
        .map(lambda r: r if r in {"NA1", "EUW1", "KR"} else "Unknown")
    )

    log.info("Long format: %d rows", len(long_df))
    region_counts = long_df.drop_duplicates("game_id")["region"].value_counts()
    log.info("Games per region:\n%s", region_counts.to_string())
    return long_df


# ── Analysis 1: Role importance ───────────────────────────────────────────────

def analyse_roles(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per role (averaged across both teams)."""
    role_stats = (
        long_df.groupby("role")["contribution"]
        .agg(mean_abs_shap=lambda x: x.abs().mean(), n=len)
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
    )
    log.info("Role importance:\n%s", role_stats.to_string(index=False))
    return role_stats


def plot_role_importance(role_stats: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: mean |SHAP| per role
    rs = role_stats.sort_values("mean_abs_shap", ascending=True)
    colors = [ROLE_COLORS[r] for r in rs["role"]]
    bars = axes[0].barh(rs["role"], rs["mean_abs_shap"], color=colors, edgecolor="white")
    axes[0].bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    axes[0].set_xlabel("Mean |SHAP| (log-odds)")
    axes[0].set_title("Role Importance — Mean |SHAP| Contribution")

    # Right: signed mean SHAP per role x team (blue team vs red team)
    # positive = pushes toward blue win, negative = pushes toward red win
    # For a balanced dataset both teams should be roughly symmetric
    role_team = (
        long_df_global.groupby(["role", "team"])["contribution"]
        .mean()
        .reset_index()
        .rename(columns={"contribution": "mean_shap"})
    )
    roles_ordered = role_stats.sort_values("mean_abs_shap", ascending=False)["role"].tolist()
    x = np.arange(len(roles_ordered))
    width = 0.35

    blue_vals = [role_team.loc[(role_team["role"] == r) & (role_team["team"] == "blue"), "mean_shap"].values[0]
                 if len(role_team.loc[(role_team["role"] == r) & (role_team["team"] == "blue")]) > 0 else 0
                 for r in roles_ordered]
    red_vals  = [role_team.loc[(role_team["role"] == r) & (role_team["team"] == "red"), "mean_shap"].values[0]
                 if len(role_team.loc[(role_team["role"] == r) & (role_team["team"] == "red")]) > 0 else 0
                 for r in roles_ordered]

    axes[1].bar(x - width/2, blue_vals, width, label="Blue side", color="#1565C0", edgecolor="white")
    axes[1].bar(x + width/2, red_vals,  width, label="Red side",  color="#B71C1C", edgecolor="white")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(roles_ordered)
    axes[1].set_ylabel("Mean SHAP (log-odds)")
    axes[1].set_title("Mean SHAP by Role x Team\n(positive = pushes toward blue win)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


# ── Analysis 1b: Role importance by region ───────────────────────────────────

def analyse_roles_by_region(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per (region, role). Only includes regions with data."""
    region_role_stats = (
        long_df.groupby(["region", "role"])["contribution"]
        .agg(mean_abs_shap=lambda x: x.abs().mean(), n=len)
        .reset_index()
        .sort_values(["region", "mean_abs_shap"], ascending=[True, False])
    )
    log.info("Role importance by region:\n%s", region_role_stats.to_string(index=False))
    return region_role_stats


def plot_role_importance_by_region(
    region_role_stats: pd.DataFrame,
    path: Path,
) -> None:
    """
    One subplot per region showing mean |SHAP| per role.
    Regions with < 10 games are skipped.
    """
    regions = [r for r in region_role_stats["region"].unique()
               if r != "Unknown"]
    if not regions:
        log.warning("No recognised regions found — skipping regional plot.")
        return

    fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 4), sharey=False)
    if len(regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, sorted(regions)):
        sub = region_role_stats[region_role_stats["region"] == region].copy()
        sub = sub.sort_values("mean_abs_shap", ascending=True)
        colors = [ROLE_COLORS.get(r, "steelblue") for r in sub["role"]]
        bars = ax.barh(sub["role"], sub["mean_abs_shap"], color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.set_xlabel("Mean |SHAP| (log-odds)")
        ax.set_title(f"{region}", fontsize=12, fontweight="bold")

    fig.suptitle("Role Importance by Region — Mean |SHAP| Contribution", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


# ── Analysis 2: Champion contribution ────────────────────────────────────────

def analyse_champions(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (champion_id, role) mean and std of signed_contrib.
    signed_contrib > 0  means the champion helped their team win on average.
    """
    champ_stats = (
        long_df.groupby(["champion_id", "role"])["signed_contrib"]
        .agg(
            mean_contrib="mean",
            std_contrib="std",
            n_appearances="count",
        )
        .reset_index()
    )
    # Only keep champs with enough appearances for reliable estimates
    champ_stats = champ_stats[champ_stats["n_appearances"] >= MIN_GAMES_PER_CHAMP].copy()
    log.info("Champion-role pairs with >= %d appearances: %d",
             MIN_GAMES_PER_CHAMP, len(champ_stats))
    return champ_stats


def plot_champion_contribution(
    champ_stats: pd.DataFrame,
    champ_names: dict[int, str],
    path: Path,
    top_n: int = 10,
) -> None:
    """5-subplot grid: top-N and bottom-N champions by mean signed contribution per role."""
    fig, axes = plt.subplots(1, len(ROLES), figsize=(4 * len(ROLES), 6), sharey=False)

    for ax, role in zip(axes, ROLES):
        sub = champ_stats[champ_stats["role"] == role].copy()
        if sub.empty:
            ax.set_title(role)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Top-N by mean contribution (most helpful champions for their team)
        sub = sub.nlargest(top_n, "mean_contrib")
        labels = [champ_names.get(cid, str(cid)) for cid in sub["champion_id"]]
        colors = ["#1565C0" if v >= 0 else "#B71C1C" for v in sub["mean_contrib"]]

        bars = ax.barh(labels[::-1], sub["mean_contrib"].values[::-1],
                       xerr=sub["std_contrib"].values[::-1] / np.sqrt(sub["n_appearances"].values[::-1]),
                       color=colors[::-1], edgecolor="white", capsize=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(role.capitalize(), fontsize=11)
        ax.set_xlabel("Mean signed SHAP")
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(f"Top-{top_n} Champions by Mean Contribution per Role\n"
                 f"(positive = helped own team win; error bars = SEM)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


# ── Analysis 3: Champion-normalised z-score ───────────────────────────────────

def compute_zscores(long_df: pd.DataFrame, champ_stats: pd.DataFrame) -> pd.DataFrame:
    """
    For each (game_id, minute, slot) row, compute:
      z_score = (signed_contrib - champ_role_mean) / champ_role_std

    Rows whose champion-role pair has < MIN_GAMES_PER_CHAMP appearances get NaN.
    """
    baseline = champ_stats[["champion_id", "role", "mean_contrib", "std_contrib"]].rename(
        columns={"mean_contrib": "baseline_mean", "std_contrib": "baseline_std"}
    )
    merged = long_df.merge(baseline, on=["champion_id", "role"], how="left")
    merged["z_score"] = (
        (merged["signed_contrib"] - merged["baseline_mean"])
        / (merged["baseline_std"] + 1e-9)
    )
    # Set NaN for champs without a reliable baseline
    no_baseline = merged["baseline_mean"].isna()
    merged.loc[no_baseline, "z_score"] = np.nan
    log.info("Z-score coverage: %.1f%% of rows have a baseline",
             100 * (~no_baseline).mean())
    return merged


def plot_zscore_distribution(zscores_df: pd.DataFrame, path: Path) -> None:
    """
    Sanity check: z-scores should be roughly N(0,1).
    Also facet by role to see if certain roles have more variance.
    """
    fig, axes = plt.subplots(1, len(ROLES) + 1, figsize=(3 * (len(ROLES) + 1), 4))

    # Overall
    valid = zscores_df["z_score"].dropna()
    axes[0].hist(valid, bins=60, color="steelblue", edgecolor="none", density=True, alpha=0.8)
    axes[0].set_title(f"All roles\nN={len(valid):,}", fontsize=9)
    axes[0].set_xlabel("z-score")
    axes[0].axvline(0, color="red", linewidth=0.8)

    # Per role
    for ax, role in zip(axes[1:], ROLES):
        sub = zscores_df.loc[zscores_df["role"] == role, "z_score"].dropna()
        ax.hist(sub, bins=40, color=ROLE_COLORS[role], edgecolor="none", density=True, alpha=0.8)
        ax.set_title(f"{role.capitalize()}\nN={len(sub):,}", fontsize=9)
        ax.set_xlabel("z-score")
        ax.axvline(0, color="black", linewidth=0.8)

    fig.suptitle("Champion-Normalised Z-Score Distribution\n"
                 "(0 = average for that champion-role; positive = above average)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

# Module-level reference used in plot_role_importance (needs long_df)
long_df_global: pd.DataFrame = pd.DataFrame()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Role/champion SHAP analysis")
    p.add_argument("--top-champs", type=int, default=10,
                   help="Top-N champions to show per role in champion plot (default: 10)")
    return p.parse_args()


def main() -> None:
    global long_df_global
    args = parse_args()
    t0   = time.time()

    # ── Load SHAP contributions ───────────────────────────────────────────────
    if not CONTRIB_PATH.exists():
        log.error("shap_contributions.parquet not found — run 06_shap_explain.py first.")
        sys.exit(1)
    contrib_df = pd.read_parquet(CONTRIB_PATH)
    log.info("Loaded contributions: %d rows, %d games",
             len(contrib_df), contrib_df["game_id"].nunique())

    # ── Load feature champion IDs ─────────────────────────────────────────────
    champ_cols = [f"{slot}_champion_id" for slot in SLOTS]
    log.info("Loading champion IDs from features.parquet ...")
    features_df = pd.read_parquet(
        FEATURES_PATH,
        columns=["game_id", "minute"] + champ_cols,
    )

    # ── Build long format ─────────────────────────────────────────────────────
    long_df = build_long_format(contrib_df, features_df)
    long_df_global = long_df   # needed by plot helper

    # Save long format for downstream use
    long_path = PROCESSED_DIR / "shap_long.parquet"
    long_df.to_parquet(long_path, index=False)
    log.info("Saved: %s  (%d rows)", long_path, len(long_df))

    # ── Champion name lookup ──────────────────────────────────────────────────
    champ_names = load_champion_names()

    # ── Analysis 1: Role importance ───────────────────────────────────────────
    log.info("--- Analysis 1: Role importance ---")
    role_stats = analyse_roles(long_df)
    plot_role_importance(role_stats, REPORTS_DIR / "analysis_role_importance.png")

    # ── Analysis 1b: Role importance by region ────────────────────────────────
    log.info("--- Analysis 1b: Role importance by region ---")
    region_role_stats = analyse_roles_by_region(long_df)
    region_role_stats.to_parquet(PROCESSED_DIR / "region_role_stats.parquet", index=False)
    log.info("Saved: region_role_stats.parquet  (%d rows)", len(region_role_stats))
    plot_role_importance_by_region(
        region_role_stats, REPORTS_DIR / "analysis_role_importance_by_region.png"
    )

    # ── Analysis 2: Champion contribution ────────────────────────────────────
    log.info("--- Analysis 2: Champion contribution ---")
    champ_stats = analyse_champions(long_df)
    champ_stats.to_parquet(PROCESSED_DIR / "champion_role_stats.parquet", index=False)
    log.info("Saved: champion_role_stats.parquet  (%d rows)", len(champ_stats))
    plot_champion_contribution(
        champ_stats, champ_names,
        REPORTS_DIR / "analysis_champion_contribution.png",
        top_n=args.top_champs,
    )

    # ── Analysis 3: Champion-normalised z-score ───────────────────────────────
    log.info("--- Analysis 3: Champion-normalised z-scores ---")
    zscores_df = compute_zscores(long_df, champ_stats)
    zscore_out = zscores_df[["game_id", "minute", "slot", "role", "team",
                              "champion_id", "signed_contrib", "z_score"]]
    zscore_out.to_parquet(PROCESSED_DIR / "player_zscores.parquet", index=False)
    log.info("Saved: player_zscores.parquet  (%d rows)", len(zscore_out))
    plot_zscore_distribution(zscores_df, REPORTS_DIR / "analysis_zscore_distribution.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info("Done in %.1f s", elapsed)

    print("\n" + "=" * 60)
    print("ROLE IMPORTANCE (mean |SHAP|, descending):")
    for _, row in role_stats.sort_values("mean_abs_shap", ascending=False).iterrows():
        print(f"  {row['role']:<10}: {row['mean_abs_shap']:.5f}  (n={int(row['n']):,} slot-minutes)")

    print("\nTOP 5 CHAMPIONS OVERALL (by mean signed contribution):")
    overall_top = (
        champ_stats.groupby("champion_id")[["mean_contrib", "n_appearances"]]
        .apply(lambda g: pd.Series({
            "mean_contrib":    np.average(g["mean_contrib"], weights=g["n_appearances"]),
            "n_appearances":   g["n_appearances"].sum(),
        }))
        .reset_index()
        .nlargest(5, "mean_contrib")
    )
    for _, row in overall_top.iterrows():
        name = champ_names.get(int(row["champion_id"]), f"ID={int(row['champion_id'])}")
        print(f"  {name:<20}: {row['mean_contrib']:+.5f}  (n={int(row['n_appearances'])} appearances)")

    print("\nZ-SCORE SUMMARY (champion-normalised performance):")
    valid_z = zscores_df["z_score"].dropna()
    print(f"  Coverage : {len(valid_z):,} / {len(zscores_df):,} rows  "
          f"({100*len(valid_z)/len(zscores_df):.1f}%)")
    print(f"  Mean     : {valid_z.mean():.4f}  (should be ~0)")
    print(f"  Std      : {valid_z.std():.4f}  (should be ~1)")

    print("\nROLE IMPORTANCE BY REGION (mean |SHAP|):")
    for region in sorted(region_role_stats["region"].unique()):
        if region == "Unknown":
            continue
        sub = region_role_stats[region_role_stats["region"] == region].sort_values(
            "mean_abs_shap", ascending=False
        )
        print(f"  [{region}]")
        for _, row in sub.iterrows():
            print(f"    {row['role']:<10}: {row['mean_abs_shap']:.5f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
