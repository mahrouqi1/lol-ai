"""
06_shap_explain.py
==================
SHAP player attribution using the trained LightGBM snapshot model.

For each (game_id, minute) row, sums SHAP values for all {slot}_* features
per player to get that player's contribution to the win probability prediction.

SHAP values are in log-odds space (before sigmoid).
  Positive value  -> pushes prediction toward blue win
  Negative value  -> pushes prediction toward red win

Attribution scope
-----------------
  Per-slot   : sum of SHAP values for all {slot}_* columns (e.g. blue_top_*)
  Differential: sum of SHAP values for diff_* columns (team matchup context)
  The 10 slot contributions + differential SHAP + base_value ~ model log-odds output.

Outputs
-------
  data/processed/shap_contributions.parquet  - (game_id, minute, <10 slots>, diff_shap)
  data/processed/shap_game_summary.parquet   - per-game mean contribution per slot
  reports/shap_beeswarm.png                  - top-30 feature SHAP beeswarm
  reports/shap_slot_importance.png           - mean |SHAP| per slot
  reports/shap_contribution_timeline.png     - per-slot SHAP over time for sample games
  reports/shap_game_bar.png                  - single game, single minute, bar chart

Usage
-----
    conda activate lol_shap_env
    python src/06_shap_explain.py
    python src/06_shap_explain.py --n-games 200   # faster test
    python src/06_shap_explain.py --n-games 0     # use ALL games (slow)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODEL_PATH    = PROJECT_ROOT / "models" / "lgbm_snapshot.txt"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR   = PROJECT_ROOT / "reports"
LOG_DIR       = PROJECT_ROOT / "logs"

for _d in (PROCESSED_DIR, REPORTS_DIR, LOG_DIR):
    _d.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "06_shap_explain.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SLOTS = [
    "blue_top", "blue_jungle", "blue_middle", "blue_bottom", "blue_utility",
    "red_top",  "red_jungle",  "red_middle",  "red_bottom",  "red_utility",
]
CAT_SUFFIXES  = ["champion_id", "summoner1_id", "summoner2_id", "keystone", "primary_tree"]
METADATA_COLS = {"game_id", "minute", "blue_win", "game_duration_min"}

# Colours: blue team = blues, red team = reds
SLOT_COLORS = {
    "blue_top":     "#1565C0",
    "blue_jungle":  "#1976D2",
    "blue_middle":  "#1E88E5",
    "blue_bottom":  "#42A5F5",
    "blue_utility": "#90CAF9",
    "red_top":      "#B71C1C",
    "red_jungle":   "#C62828",
    "red_middle":   "#E53935",
    "red_bottom":   "#EF5350",
    "red_utility":  "#EF9A9A",
}

MINUTE_BUCKETS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 999)]
BUCKET_LABELS  = ["0-5", "5-10", "10-15", "15-20", "20-25", "25+"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_cat_cols(feature_cols: list[str]) -> list[str]:
    return [
        f"{slot}_{suf}"
        for slot in SLOTS
        for suf in CAT_SUFFIXES
        if f"{slot}_{suf}" in feature_cols
    ]


def build_slot_index_map(feature_cols: list[str]) -> dict[str, list[int]]:
    """Map each slot to the column indices of its features."""
    return {
        slot: [i for i, c in enumerate(feature_cols) if c.startswith(f"{slot}_")]
        for slot in SLOTS
    }


def diff_indices(feature_cols: list[str]) -> list[int]:
    """Indices of diff_* feature columns."""
    return [i for i, c in enumerate(feature_cols) if c.startswith("diff_")]


def compute_contributions(
    shap_matrix: np.ndarray,
    slot_index_map: dict[str, list[int]],
    diff_idx: list[int],
) -> pd.DataFrame:
    """
    Returns a DataFrame with one column per slot + 'diff_shap'.
    Each value is the sum of SHAP values for that group of features.
    """
    rows: dict[str, np.ndarray] = {}
    for slot, indices in slot_index_map.items():
        rows[slot] = shap_matrix[:, indices].sum(axis=1)
    rows["diff_shap"] = shap_matrix[:, diff_idx].sum(axis=1) if diff_idx else np.zeros(len(shap_matrix))
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_beeswarm(shap_matrix: np.ndarray, X_sample: pd.DataFrame, path: Path, top_n: int = 30) -> None:
    """Standard SHAP beeswarm for top-N features by mean |SHAP|."""
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    shap_top = shap_matrix[:, top_idx]
    X_top    = X_sample.iloc[:, top_idx]
    names    = [X_sample.columns[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, top_n * 0.35 + 1))
    # Manual beeswarm-style dot plot
    for row_i, (name, shap_col, feat_col) in enumerate(
        zip(names[::-1], shap_top.T[::-1], X_top.T.values[::-1])
    ):
        # Colour by feature value (normalised 0-1)
        vmin, vmax = np.percentile(feat_col, [5, 95])
        normed = np.clip((feat_col - vmin) / (vmax - vmin + 1e-9), 0, 1)
        scatter = ax.scatter(
            shap_col, np.full_like(shap_col, row_i) + np.random.uniform(-0.3, 0.3, len(shap_col)),
            c=normed, cmap="coolwarm", alpha=0.4, s=8, linewidths=0,
        )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1], fontsize=7)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("SHAP value (log-odds)")
    ax.set_title(f"Top {top_n} Features — SHAP Beeswarm (LightGBM Snapshot)")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label("Feature value (low → high)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_slot_importance(contrib_df: pd.DataFrame, path: Path) -> None:
    """Horizontal bar chart: mean |SHAP| per slot."""
    slot_importance = contrib_df[SLOTS].abs().mean().sort_values(ascending=True)

    colors = [SLOT_COLORS[s] for s in slot_importance.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(slot_importance.index, slot_importance.values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Mean |SHAP| contribution (log-odds)")
    ax.set_title("Per-Slot Win Contribution — Mean |SHAP| (all sampled rows)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_contribution_timeline(
    contrib_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    game_ids: list,
    path: Path,
) -> None:
    """
    For each sample game, plot per-slot SHAP contribution over time.
    Blue slots above zero line, red slots below (by convention of sign).
    """
    n = len(game_ids)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)

    for ax, gid in zip(axes[:, 0], game_ids):
        mask  = meta_df["game_id"] == gid
        mins  = meta_df.loc[mask, "minute"].values
        order = np.argsort(mins)
        mins  = mins[order]
        gc    = contrib_df.loc[mask].iloc[order]
        outcome = "Blue Win" if meta_df.loc[mask, "blue_win"].iloc[0] == 1 else "Red Win"

        for slot in SLOTS:
            ax.plot(mins, gc[slot].values, color=SLOT_COLORS[slot],
                    linewidth=1.4, label=slot)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Minute")
        ax.set_ylabel("SHAP contribution (log-odds)")
        ax.set_title(f"Game {gid} — {outcome}")
        ax.legend(fontsize=7, ncol=5, loc="upper left")

    fig.suptitle("Per-Slot SHAP Contribution Over Time", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_game_bar(
    contrib_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    game_id,
    minute: int,
    path: Path,
) -> None:
    """Single-game, single-minute horizontal bar chart of per-slot SHAP."""
    mask = (meta_df["game_id"] == game_id) & (meta_df["minute"] == minute)
    if not mask.any():
        # Fallback: closest available minute
        game_mask = meta_df["game_id"] == game_id
        available = meta_df.loc[game_mask, "minute"].values
        minute    = available[np.argmin(np.abs(available - minute))]
        mask      = (meta_df["game_id"] == game_id) & (meta_df["minute"] == minute)

    row     = contrib_df.loc[mask].iloc[0]
    outcome = "Blue Win" if meta_df.loc[mask, "blue_win"].iloc[0] == 1 else "Red Win"

    slots  = SLOTS
    values = [row[s] for s in slots]
    colors = [SLOT_COLORS[s] for s in slots]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(slots, values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution (log-odds)")
    ax.set_title(f"Game {game_id} @ Minute {minute} — {outcome}\nPer-Slot Win Contribution")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP player attribution")
    p.add_argument("--n-games", type=int, default=500,
                   help="Number of games to sample for SHAP (0 = all, default: 500)")
    p.add_argument("--timeline-games", type=int, default=4,
                   help="Number of games to show in timeline plot (default: 4)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0   = time.time()

    # ── Load model ────────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        log.error("Model not found: %s — run 04a_train_snapshot.py first.", MODEL_PATH)
        sys.exit(1)
    log.info("Loading model: %s", MODEL_PATH)
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    # ── Load features ─────────────────────────────────────────────────────────
    log.info("Loading features: %s", FEATURES_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    log.info("Loaded: %d rows x %d cols", len(df), df.shape[1])

    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    cat_cols     = get_cat_cols(feature_cols)
    for col in cat_cols:
        df[col] = df[col].astype("int32")

    # ── Sample games ──────────────────────────────────────────────────────────
    all_game_ids = df["game_id"].unique()
    if args.n_games > 0 and args.n_games < len(all_game_ids):
        rng         = np.random.default_rng(42)
        sampled_ids = rng.choice(all_game_ids, size=args.n_games, replace=False)
        df_sample   = df[df["game_id"].isin(sampled_ids)].copy()
        log.info("Sampled %d games -> %d rows", args.n_games, len(df_sample))
    else:
        df_sample = df.copy()
        log.info("Using all %d games -> %d rows", len(all_game_ids), len(df_sample))

    X_sample = df_sample[feature_cols].copy()
    meta     = df_sample[["game_id", "minute", "blue_win"]].copy().reset_index(drop=True)

    # ── SHAP values ───────────────────────────────────────────────────────────
    log.info("Computing SHAP values for %d rows ...", len(X_sample))
    explainer  = shap.TreeExplainer(booster)
    shap_out   = explainer.shap_values(X_sample)

    # Handle both SHAP API versions:
    # - Older: returns [neg_class_array, pos_class_array]
    # - Newer: returns single array (for positive class)
    if isinstance(shap_out, list):
        shap_matrix = shap_out[1]   # positive class = blue_win=1
        base_val    = float(explainer.expected_value[1])
    else:
        shap_matrix = shap_out
        bv = explainer.expected_value
        base_val = float(bv[1]) if hasattr(bv, "__len__") else float(bv)

    log.info("SHAP matrix shape: %s  base_value: %.4f", shap_matrix.shape, base_val)

    # ── Per-slot contributions ─────────────────────────────────────────────────
    slot_index_map = build_slot_index_map(feature_cols)
    d_idx          = diff_indices(feature_cols)
    contrib_df     = compute_contributions(shap_matrix, slot_index_map, d_idx)
    contrib_df     = pd.concat([meta, contrib_df], axis=1)

    # Sanity check: slot sums + diff_shap should approximate total SHAP per row
    slot_sum = contrib_df[SLOTS].sum(axis=1) + contrib_df["diff_shap"]
    total_shap = shap_matrix.sum(axis=1)
    residual = (slot_sum - total_shap).abs().mean()
    log.info("Attribution residual (unaccounted SHAP): mean |diff| = %.6f", residual)

    # ── Save contributions ─────────────────────────────────────────────────────
    contrib_path = PROCESSED_DIR / "shap_contributions.parquet"
    contrib_df.to_parquet(contrib_path, index=False)
    log.info("Saved: %s  (%d rows)", contrib_path, len(contrib_df))

    # Per-game summary (mean contribution per slot over all minutes)
    game_summary = (
        contrib_df.groupby("game_id")[SLOTS + ["diff_shap", "blue_win"]]
        .mean()
        .reset_index()
    )
    summary_path = PROCESSED_DIR / "shap_game_summary.parquet"
    game_summary.to_parquet(summary_path, index=False)
    log.info("Saved: %s  (%d games)", summary_path, len(game_summary))

    # ── Plots ─────────────────────────────────────────────────────────────────
    # 1. Beeswarm (subsample rows for speed if large)
    n_beeswarm = min(2000, len(X_sample))
    rng2 = np.random.default_rng(0)
    bee_idx = rng2.choice(len(X_sample), size=n_beeswarm, replace=False)
    plot_beeswarm(
        shap_matrix[bee_idx],
        X_sample.iloc[bee_idx].reset_index(drop=True),
        REPORTS_DIR / "shap_beeswarm.png",
    )

    # 2. Slot importance
    plot_slot_importance(contrib_df, REPORTS_DIR / "shap_slot_importance.png")

    # 3. Contribution timeline (pick games with >= 20 minutes for interesting plots)
    long_games = (
        contrib_df.groupby("game_id")["minute"].max()
        .loc[lambda s: s >= 20]
        .index.tolist()
    )
    n_tl = min(args.timeline_games, len(long_games))
    timeline_ids = list(np.random.default_rng(7).choice(long_games, size=n_tl, replace=False))
    plot_contribution_timeline(
        contrib_df.set_index(contrib_df.index),  # keep integer index
        meta,
        timeline_ids,
        REPORTS_DIR / "shap_contribution_timeline.png",
    )

    # 4. Single-game bar chart (pick game with a clear outcome near minute 15)
    bar_game_id = timeline_ids[0]
    plot_game_bar(
        contrib_df,
        meta,
        bar_game_id,
        minute=15,
        path=REPORTS_DIR / "shap_game_bar.png",
    )

    # ── Summary stats ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info("Done in %.1f s (%.1f min)", elapsed, elapsed / 60)

    print("\n" + "=" * 60)
    print(f"  Games analysed      : {contrib_df['game_id'].nunique()}")
    print(f"  Rows (game-minutes) : {len(contrib_df):,}")
    print(f"  SHAP base value     : {base_val:.4f}  (= {1/(1+np.exp(-base_val))*100:.1f}% blue win prior)")
    print(f"  Attribution residual: {residual:.6f} (should be ~0)")
    print("=" * 60)
    print("\nMean |SHAP| per slot (all sampled rows):")
    slot_imp = contrib_df[SLOTS].abs().mean().sort_values(ascending=False)
    for slot, val in slot_imp.items():
        team = "BLUE" if slot.startswith("blue") else " RED"
        print(f"  {team}  {slot:<20} : {val:.5f}")


if __name__ == "__main__":
    main()
