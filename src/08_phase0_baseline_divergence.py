"""
08_phase0_baseline_divergence.py
================================
PHASE 0 — the one-figure motivator for the whole project.

Thesis being demonstrated: in counterfactual attribution, **the baseline IS the
modeling decision**. We hold the trained LightGBM win-probability model FIXED and
vary only the replacement baseline, then show that the per-player contribution
attributions disagree materially. Large disagreement = the empirical hook that
justifies the population-conditional replacement baseline (Phases 1-2).

METHOD. We attribute at the PLAYER level (not the game-state-feature level) using
**exact per-team interventional Shapley over the 5 slots of a team** (2^5 = 32
coalitions per team, the other team held at its real values), via direct
`booster.predict` with background-swapping. This (a) is exactly the Phase-2
estimator previewed here, (b) handles LightGBM categorical splits natively
(shap's interventional TreeExplainer cannot — it chokes on categorical thresholds),
and (c) varies only the BACKGROUND so the comparison is clean.

Baselines compared (same model, same games, ONLY the replacement background differs):
  default_treepath : legacy approach — per-feature TreeSHAP (tree_path_dependent /
                     conditional), summed per slot. What 06_shap_explain.py does today.
  mean_bg          : group-Shapley, replacement = single mean-vector (off-manifold;
                     the "mean champion" is not a real player — the report's critique).
  pop_bg           : group-Shapley, replacement = K real rows from the whole
                     population (on-manifold, context-agnostic).
  cond_bg          : group-Shapley, replacement = K real rows matched to each row's
                     context (region x minute-bucket). On-manifold + conditional —
                     the closest Phase-0 preview of the population-conditional baseline.

Disagreement metrics (per row, over the 10 slot contributions):
  - top-contributor flip rate (argmax slot differs between two baselines)
  - Spearman rank correlation of the 10-slot ordering (mean over rows)
  - mean L1 distance between the 10-slot contribution vectors

Outputs:
  data/processed/phase0_contributions.parquet  - per row x slot x baseline
  reports/phase0_baseline_divergence.png        - the money figure
  reports/phase0_pairwise_disagreement.png      - pairwise heatmaps

Usage:
  conda run -n lol_shap_env python src/08_phase0_baseline_divergence.py --n-games 8 --background-size 16   # smoke
  conda run -n lol_shap_env python src/08_phase0_baseline_divergence.py --n-games 200                       # workstation dev
  # full sweep -> OSC (see slurm/).  Cost ~ n_rows x 4 x 64 x K predictions.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from itertools import chain, combinations
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODEL_PATH    = PROJECT_ROOT / "models" / "lgbm_snapshot.txt"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR   = PROJECT_ROOT / "reports"
LOG_DIR       = PROJECT_ROOT / "logs"
for _d in (PROCESSED_DIR, REPORTS_DIR, LOG_DIR):
    _d.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "08_phase0_baseline_divergence.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SLOTS = [
    "blue_top", "blue_jungle", "blue_middle", "blue_bottom", "blue_utility",
    "red_top",  "red_jungle",  "red_middle",  "red_bottom",  "red_utility",
]
BLUE, RED   = SLOTS[:5], SLOTS[5:]
CAT_SUFFIXES  = ["champion_id", "summoner1_id", "summoner2_id", "keystone", "primary_tree"]
METADATA_COLS = {"game_id", "minute", "blue_win", "game_duration_min"}
MINUTE_EDGES  = [5, 10, 15, 20, 25]  # np.digitize internal edges -> buckets 0..5
BASELINES     = ["default_treepath", "mean_bg", "pop_bg", "cond_bg"]

# Exact Shapley over m=5 players: subsets of the other 4 and their weights.
_M = 5
_SUBSETS5 = [frozenset(s) for r in range(_M + 1)
             for s in combinations(range(_M), r)]
_SHAP_W = [math.factorial(s) * math.factorial(_M - s - 1) / math.factorial(_M)
           for s in range(_M)]  # weight for a coalition of size s when adding player i

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_cat_cols(feature_cols: list[str]) -> list[str]:
    cats = [f"{s}_{suf}" for s in SLOTS for suf in CAT_SUFFIXES
            if f"{s}_{suf}" in feature_cols]
    if "region" in feature_cols:
        cats.append("region")
    return cats


def build_slot_index_map(feature_cols: list[str]) -> dict[str, list[int]]:
    return {s: [i for i, c in enumerate(feature_cols) if c.startswith(f"{s}_")]
            for s in SLOTS}


def minute_bucket(minutes: np.ndarray) -> np.ndarray:
    return np.digitize(minutes, MINUTE_EDGES)  # 0..5


# ── Baseline A: legacy per-feature TreeSHAP (tree_path_dependent), summed per slot ──

def slot_contribs_treepath(booster, X: pd.DataFrame, slot_index_map) -> np.ndarray:
    expl = shap.TreeExplainer(booster)  # default tree_path_dependent (conditional)
    out = expl.shap_values(X)
    mat = out[1] if isinstance(out, list) else out
    return np.column_stack([mat[:, slot_index_map[s]].sum(axis=1) for s in SLOTS])


# ── Baselines B/C/D: exact per-team interventional group-Shapley ─────────────────

def _predict_logodds(booster, M: np.ndarray) -> np.ndarray:
    return booster.predict(M, raw_score=True)


def _team_phi(booster, row: np.ndarray, bg: np.ndarray, team_cols: list[list[int]]) -> np.ndarray:
    """Exact Shapley for one team's 5 slots. `row` (F,) is the full real game with
    the OTHER team already real; `bg` (K,F) supplies replacement values; only this
    team's non-coalition slot columns are swapped to bg. Returns (5,) log-odds."""
    K = bg.shape[0]
    blocks = np.empty((len(_SUBSETS5), K, row.shape[0]), dtype=np.float64)
    for bi, S in enumerate(_SUBSETS5):
        M = np.tile(row, (K, 1))
        for j in range(_M):
            if j not in S:                       # slot j replaced by background
                M[:, team_cols[j]] = bg[:, team_cols[j]]
        blocks[bi] = M
    preds = _predict_logodds(booster, blocks.reshape(-1, row.shape[0])).reshape(len(_SUBSETS5), K)
    vS = {S: float(preds[bi].mean()) for bi, S in enumerate(_SUBSETS5)}
    phi = np.zeros(_M)
    for i in range(_M):
        for S in _SUBSETS5:
            if i in S:
                continue
            phi[i] += _SHAP_W[len(S)] * (vS[S | {i}] - vS[S])
    return phi


def group_shapley(
    booster, X: np.ndarray, feature_cols, slot_index_map,
    background_fn, rng,
) -> np.ndarray:
    """(n,10) per-slot contributions via exact per-team Shapley. `background_fn(i)`
    returns the (K,F) replacement background to use for explained row i."""
    blue_cols = [slot_index_map[s] for s in BLUE]
    red_cols  = [slot_index_map[s] for s in RED]
    out = np.zeros((len(X), 10), dtype=np.float64)
    for i in range(len(X)):
        bg = background_fn(i)
        out[i, :5] = _team_phi(booster, X[i], bg, blue_cols)
        out[i, 5:] = _team_phi(booster, X[i], bg, red_cols)
        if (i + 1) % 200 == 0:
            log.info("    ... %d/%d rows", i + 1, len(X))
    return out


# ── Disagreement metrics ────────────────────────────────────────────────────────

def _decisiveness(C: np.ndarray) -> np.ndarray:
    """Per-row top1-minus-top2 margin: how clearly one player 'wins' the credit.
    Near-zero for ambiguous (e.g. early-game) rows, so it down-weights noise."""
    s = np.sort(C, axis=1)
    return s[:, -1] - s[:, -2]


def pairwise_disagreement(contribs: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for a, b in combinations(contribs, 2):
        ca, cb = contribs[a], contribs[b]
        flips = (ca.argmax(1) != cb.argmax(1))
        flip = float(flips.mean())
        # Decisiveness-weighted flip: only "real" disagreements (rows where at
        # least one baseline names a clear top contributor) count for much.
        w = np.maximum(_decisiveness(ca), _decisiveness(cb))
        flip_w = float((flips * w).sum() / w.sum()) if w.sum() > 0 else float("nan")
        rhos = [spearmanr(ca[i], cb[i]).statistic for i in range(len(ca))]
        rhos = [r for r in rhos if not np.isnan(r)]
        l1 = float(np.abs(ca - cb).sum(1).mean())
        rows.append({"pair": f"{a} vs {b}", "a": a, "b": b,
                     "flip_rate": flip, "flip_rate_weighted": flip_w,
                     "mean_spearman": float(np.mean(rhos)) if rhos else float("nan"),
                     "mean_l1": l1})
    return pd.DataFrame(rows)


def aggregate_per_game(contribs: dict[str, np.ndarray], meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """Integrate per-minute contributions into one (n_games, 10) per baseline by
    summing each slot's contribution over the game's minutes. More stable than
    per-minute argmax, and closer to 'who contributed to THIS game'."""
    gids = meta["game_id"].to_numpy()
    order = np.unique(gids)
    out = {}
    for name, C in contribs.items():
        agg = np.zeros((len(order), 10))
        for gi, g in enumerate(order):
            agg[gi] = C[gids == g].sum(axis=0)
        out[name] = agg
    return out


# ── Plots ───────────────────────────────────────────────────────────────────────

def plot_money_figure(contribs, disagree, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    a, b = "mean_bg", "cond_bg"
    if a in contribs and b in contribs:
        xa, yb = contribs[a].ravel(), contribs[b].ravel()
        n = min(20000, len(xa))
        sel = np.random.default_rng(0).choice(len(xa), size=n, replace=False)
        ax = axes[0]
        ax.scatter(xa[sel], yb[sel], s=5, alpha=0.18, color="#C62828", linewidths=0)
        lim = float(np.percentile(np.abs(np.concatenate([xa[sel], yb[sel]])), 99)) or 1.0
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="agree (y=x)")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("per-player contribution — mean background (off-manifold)")
        ax.set_ylabel("per-player contribution — conditional background (on-manifold)")
        ax.set_title("Same model, same game:\nthe replacement baseline changes player credit")
        ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    d = disagree.sort_values("flip_rate")
    bars = ax.barh(d["pair"], d["flip_rate"], color="#1565C0", edgecolor="white")
    ax.bar_label(bars, labels=[f"{v*100:.1f}%" for v in d["flip_rate"]], padding=3, fontsize=9)
    ax.set_xlim(0, max(0.6, float(d["flip_rate"].max()) * 1.3))
    ax.set_xlabel("fraction of game-minutes where the TOP contributor changes")
    ax.set_title("How often the identified 'best player' flips\nwith the baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_pairwise_heatmaps(disagree, path) -> None:
    order = BASELINES
    idx = {b: i for i, b in enumerate(order)}
    metrics = [("mean_spearman", "Mean Spearman (slot ordering)\nhigher = agreement", "viridis"),
               ("flip_rate",     "Top-contributor flip rate\nhigher = disagreement", "magma"),
               ("mean_l1",       "Mean L1 distance (log-odds)\nhigher = disagreement", "magma")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (metric, title, cmap) in zip(axes, metrics):
        M = np.full((len(order), len(order)), np.nan)
        for _, r in disagree.iterrows():
            i, j = idx[r["a"]], idx[r["b"]]
            M[i, j] = M[j, i] = r[metric]
        if metric == "mean_spearman":
            np.fill_diagonal(M, 1.0)
        im = ax.imshow(M, cmap=cmap)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
        for i in range(len(order)):
            for j in range(len(order)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 0: replacement-baseline divergence")
    p.add_argument("--n-games", type=int, default=120, help="games to explain (0=all). Default 120.")
    p.add_argument("--background-size", type=int, default=32, help="K replacement rows. Default 32.")
    p.add_argument("--pool-games", type=int, default=3000, help="games for the background pool. Default 3000.")
    p.add_argument("--max-minute", type=int, default=30, help="cap explained minutes for speed. Default 30.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    if not MODEL_PATH.exists():
        log.error("Model not found: %s — run 04a_train_snapshot.py first.", MODEL_PATH)
        sys.exit(1)
    log.info("Loading model: %s", MODEL_PATH)
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    feature_cols = list(booster.feature_name())  # 401 trained features, in order

    log.info("Loading features: %s", FEATURES_PATH)
    df = pd.read_parquet(FEATURES_PATH, columns=feature_cols + ["game_id", "minute", "blue_win"])
    for col in get_cat_cols(feature_cols):
        df[col] = df[col].astype("int32")
    if args.max_minute:
        df = df[df["minute"] <= args.max_minute]
    log.info("Loaded %d rows (%d model features)", len(df), len(feature_cols))

    slot_index_map = build_slot_index_map(feature_cols)
    all_ids = df["game_id"].unique()

    # Background pool.
    pool_ids = rng.choice(all_ids, size=min(args.pool_games, len(all_ids)), replace=False)
    pool_df  = df[df["game_id"].isin(pool_ids)].reset_index(drop=True)
    pool_X   = pool_df[feature_cols].to_numpy(dtype=np.float64)
    ctx_pool = pool_df["region"].to_numpy() * 100 + minute_bucket(pool_df["minute"].to_numpy())
    log.info("Background pool: %d games -> %d rows", len(pool_ids), len(pool_X))

    # Explained set (disjoint from pool when possible).
    rest = np.setdiff1d(all_ids, pool_ids)
    pick_from = rest if len(rest) >= (args.n_games or 1) else all_ids
    expl_ids = (rng.choice(pick_from, size=args.n_games, replace=False)
                if args.n_games and args.n_games < len(pick_from) else pick_from)
    edf = df[df["game_id"].isin(expl_ids)].reset_index(drop=True)
    X = edf[feature_cols].to_numpy(dtype=np.float64)
    meta = edf[["game_id", "minute", "blue_win"]].copy()
    ctx_X = edf["region"].to_numpy() * 100 + minute_bucket(edf["minute"].to_numpy())
    log.info("Explained set: %d games -> %d rows", edf["game_id"].nunique(), len(X))

    K = args.background_size
    mean_row = pool_X.mean(axis=0, keepdims=True)          # (1,F) off-manifold point
    pop_idx  = rng.choice(len(pool_X), size=min(K, len(pool_X)), replace=False)
    pop_bg   = pool_X[pop_idx]                              # (K,F) fixed population sample

    def bg_mean(i): return mean_row
    def bg_pop(i):  return pop_bg
    def bg_cond(i):
        idx = np.where(ctx_pool == ctx_X[i])[0]
        if len(idx) == 0:
            idx = np.arange(len(pool_X))
        take = rng.choice(idx, size=min(K, len(idx)), replace=False)
        return pool_X[take]

    contribs: dict[str, np.ndarray] = {}
    log.info("[1/4] default_treepath (legacy per-feature conditional SHAP) ...")
    contribs["default_treepath"] = slot_contribs_treepath(booster, edf[feature_cols], slot_index_map)
    log.info("[2/4] mean_bg (off-manifold mean replacement) ...")
    contribs["mean_bg"] = group_shapley(booster, X, feature_cols, slot_index_map, bg_mean, rng)
    log.info("[3/4] pop_bg (on-manifold population replacement) ...")
    contribs["pop_bg"] = group_shapley(booster, X, feature_cols, slot_index_map, bg_pop, rng)
    log.info("[4/4] cond_bg (on-manifold conditional replacement) ...")
    contribs["cond_bg"] = group_shapley(booster, X, feature_cols, slot_index_map, bg_cond, rng)

    # Persist.
    frames = []
    for name, C in contribs.items():
        f = pd.DataFrame(C, columns=SLOTS)
        f.insert(0, "baseline", name)
        frames.append(pd.concat([meta.reset_index(drop=True), f], axis=1))
    out_df = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED_DIR / "phase0_contributions.parquet"
    out_df.to_parquet(out_path, index=False)
    log.info("Saved: %s (%d rows)", out_path, len(out_df))

    disagree = pairwise_disagreement(contribs)              # per-minute
    pergame = aggregate_per_game(contribs, meta)
    disagree_game = pairwise_disagreement(pergame)          # per-game integrated
    plot_money_figure(contribs, disagree, REPORTS_DIR / "phase0_baseline_divergence.png")
    plot_pairwise_heatmaps(disagree, REPORTS_DIR / "phase0_pairwise_disagreement.png")

    # Minute-bucket breakdown for the headline pair (noise concentrates early).
    mb = minute_bucket(meta["minute"].to_numpy())
    labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25+"]
    ca, cb = contribs["mean_bg"], contribs["cond_bg"]
    bucket_rows = []
    for k, lab in enumerate(labels):
        m = mb == k
        if m.sum() == 0:
            continue
        bucket_rows.append({"minute_bucket": lab, "n": int(m.sum()),
                            "flip_rate": float((ca[m].argmax(1) != cb[m].argmax(1)).mean())})
    bucket_df = pd.DataFrame(bucket_rows)

    elapsed = time.time() - t0
    print("\n" + "=" * 78)
    print("  PHASE 0 — replacement-baseline divergence")
    print(f"  explained rows : {len(X):,}  ({edf['game_id'].nunique()} games, minute<= {args.max_minute})")
    print(f"  replacement K  : {K}   pool rows: {len(pool_X):,}")
    print("=" * 78)
    print("PER-MINUTE disagreement:")
    print(disagree.to_string(index=False))
    print("\nPER-GAME (integrated) disagreement:")
    print(disagree_game.to_string(index=False))
    print("\nmean_bg vs cond_bg flip rate BY MINUTE BUCKET:")
    print(bucket_df.to_string(index=False))
    print("=" * 78)
    kr = disagree[(disagree.a == "mean_bg") & (disagree.b == "cond_bg")].iloc[0]
    kg = disagree_game[(disagree_game.a == "mean_bg") & (disagree_game.b == "cond_bg")].iloc[0]
    print(f"  HEADLINE (mean-bg vs conditional-bg):")
    print(f"    per-minute : top contributor flips {kr.flip_rate*100:.1f}% raw, "
          f"{kr.flip_rate_weighted*100:.1f}% decisiveness-weighted; Spearman {kr.mean_spearman:.2f}.")
    print(f"    per-game   : flips {kg.flip_rate*100:.1f}% of games; Spearman {kg.mean_spearman:.2f}.")
    print(f"  done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
