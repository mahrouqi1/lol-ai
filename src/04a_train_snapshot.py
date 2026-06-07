"""
04a_train_snapshot.py
=====================
LightGBM snapshot model: predict blue_win from the game state at a single
(game_id, minute) row. Each row is treated independently.

Cross-validation
----------------
  GroupKFold(n_splits=5) grouped by game_id — all minutes of a game always
  land in the same fold, preventing any game-state leakage.

Feature exclusions
------------------
  Metadata: game_id, minute, blue_win, game_duration_min (leaks game length)

Categorical features (passed to LightGBM natively as integers)
--------------------------------------------------------------
  *_champion_id, *_summoner1_id, *_summoner2_id, *_keystone, *_primary_tree

Outputs
-------
  models/lgbm_snapshot.txt          — trained model (all folds, retrained on full data)
  models/lgbm_snapshot_cv.pkl       — fold OOF predictions + metrics
  reports/snapshot_auc_by_minute.png
  reports/snapshot_calibration.png
  reports/snapshot_feature_importance.png

Usage
-----
    conda activate lol_shap_env
    python src/04a_train_snapshot.py
    python src/04a_train_snapshot.py --folds 3   # faster test run
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = Path(os.environ.get("LOL_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
FEATURES_PATH = DATA_DIR / "features.parquet"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"
LOG_DIR       = PROJECT_ROOT / "logs"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "04a_train_snapshot.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SLOTS = [
    "blue_top", "blue_jungle", "blue_middle", "blue_bottom", "blue_utility",
    "red_top",  "red_jungle",  "red_middle",  "red_bottom",  "red_utility",
]

CAT_SUFFIXES = ["champion_id", "summoner1_id", "summoner2_id", "keystone", "primary_tree"]

METADATA_COLS = {"game_id", "minute", "blue_win", "game_duration_min"}

MINUTE_BUCKETS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 999)]
BUCKET_LABELS  = ["0-5", "5-10", "10-15", "15-20", "20-25", "25+"]

LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "num_leaves":       127,
    "learning_rate":    0.05,
    "n_estimators":     1000,
    "subsample":        0.8,
    "subsample_freq":   1,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "n_jobs":           max(1, (os.cpu_count() or 4) // 2),
    "random_state":     42,
    "verbose":          -1,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_cat_cols(all_cols: list[str]) -> list[str]:
    """Return column names that should be treated as LightGBM categoricals."""
    cats = []
    for slot in SLOTS:
        for suf in CAT_SUFFIXES:
            col = f"{slot}_{suf}"
            if col in all_cols:
                cats.append(col)
    # region is a small integer label (0=NA1, 1=EUW1, 2=KR) — treat as categorical
    if "region" in all_cols:
        cats.append("region")
    return cats


def minute_bucket_label(minute: int) -> str:
    for (lo, hi), label in zip(MINUTE_BUCKETS, BUCKET_LABELS):
        if lo <= minute < hi:
            return label
    return "25+"


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_auc_by_minute(oof_df: pd.DataFrame, path: Path) -> None:
    """Bar chart: mean OOF AUC per minute bucket."""
    rows = []
    for (lo, hi), label in zip(MINUTE_BUCKETS, BUCKET_LABELS):
        mask = (oof_df["minute"] >= lo) & (oof_df["minute"] < hi)
        sub = oof_df[mask]
        if len(sub) < 100:
            continue
        auc = roc_auc_score(sub["blue_win"], sub["oof_pred"])
        rows.append({"bucket": label, "auc": auc, "n_rows": len(sub)})

    df_plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df_plot["bucket"], df_plot["auc"], color="#4C72B0", edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random")
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Game minute bucket")
    ax.set_ylabel("OOF AUC")
    ax.set_title("LightGBM Snapshot — OOF AUC by Minute Bucket")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_calibration(oof_df: pd.DataFrame, path: Path) -> None:
    """Reliability diagram: predicted probability vs. actual win rate."""
    prob_true, prob_pred = calibration_curve(
        oof_df["blue_win"], oof_df["oof_pred"], n_bins=20, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, "o-", label="LightGBM snapshot", color="#4C72B0")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration — LightGBM Snapshot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_feature_importance(model: lgb.LGBMClassifier, path: Path, top_n: int = 40) -> None:
    """Horizontal bar chart of top-N features by gain importance."""
    fi = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=model.booster_.feature_name(),
    ).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, top_n * 0.3 + 1))
    fi[::-1].plot.barh(ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Gain importance")
    ax.set_title(f"Top {top_n} Features — LightGBM Snapshot (full-data model)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM snapshot model")
    p.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    p.add_argument("--no-retrain", action="store_true",
                   help="Skip full-data retraining after CV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading %s …", FEATURES_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    log.info("Loaded: %d rows × %d cols", len(df), df.shape[1])

    # ── Split metadata / features / label ─────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    cat_cols     = get_cat_cols(feature_cols)
    log.info("Feature columns : %d", len(feature_cols))
    log.info("Categorical cols: %d — %s …", len(cat_cols), cat_cols[:4])

    X      = df[feature_cols].copy()
    y      = df["blue_win"].values
    groups = df["game_id"].values
    mins   = df["minute"].values

    # LightGBM requires categoricals to be int32 >= 0
    for col in cat_cols:
        X[col] = X[col].astype("int32")

    # ── Cross-validation ──────────────────────────────────────────────────────
    log.info("Starting GroupKFold CV with %d folds …", args.folds)
    gkf = GroupKFold(n_splits=args.folds)

    oof_preds  = np.zeros(len(X), dtype=np.float32)
    fold_aucs  = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx],      y[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            categorical_feature=cat_cols,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        preds = model.predict_proba(X_val)[:, 1].astype(np.float32)
        oof_preds[val_idx] = preds

        fold_auc = roc_auc_score(y_val, preds)
        fold_aucs.append(fold_auc)
        log.info("Fold %d/%d — best iter: %4d — val AUC: %.4f",
                 fold, args.folds, model.best_iteration_, fold_auc)

    overall_auc = roc_auc_score(y, oof_preds)
    log.info("Overall OOF AUC: %.4f  (mean fold AUC: %.4f ± %.4f)",
             overall_auc, np.mean(fold_aucs), np.std(fold_aucs))

    # ── Save OOF predictions ──────────────────────────────────────────────────
    oof_df = pd.DataFrame({
        "game_id":  groups,
        "minute":   mins,
        "blue_win": y,
        "oof_pred": oof_preds,
    })
    cv_path = MODELS_DIR / "lgbm_snapshot_cv.pkl"
    with open(cv_path, "wb") as f:
        pickle.dump({"oof_df": oof_df, "fold_aucs": fold_aucs, "overall_auc": overall_auc}, f)
    log.info("Saved OOF results: %s", cv_path)

    # ── Plots (CV) ────────────────────────────────────────────────────────────
    plot_auc_by_minute(oof_df, REPORTS_DIR / "snapshot_auc_by_minute.png")
    plot_calibration(oof_df,   REPORTS_DIR / "snapshot_calibration.png")

    # ── Retrain on full data ──────────────────────────────────────────────────
    if not args.no_retrain:
        log.info("Retraining on full dataset …")
        # Use best_iteration from last fold as a reasonable estimate;
        # add a small buffer since more data → more iterations needed.
        n_iter = int(model.best_iteration_ * 1.1)
        log.info("Using n_estimators=%d (last-fold best_iter × 1.1)", n_iter)

        full_params = {**LGBM_PARAMS, "n_estimators": n_iter}
        full_model = lgb.LGBMClassifier(**full_params)
        full_model.fit(X, y, categorical_feature=cat_cols)

        model_path = MODELS_DIR / "lgbm_snapshot.txt"
        full_model.booster_.save_model(str(model_path))
        log.info("Saved full model: %s", model_path)

        plot_feature_importance(full_model, REPORTS_DIR / "snapshot_feature_importance.png")
    else:
        log.info("--no-retrain flag set; skipping full-data training.")

    elapsed = time.time() - t0
    log.info("Done in %.1f s (%.1f min)", elapsed, elapsed / 60)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Overall OOF AUC : {overall_auc:.4f}")
    print(f"  Mean fold AUC   : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"  Fold AUCs       : {[f'{a:.4f}' for a in fold_aucs]}")
    print("=" * 60)

    # Per-bucket AUC summary to console
    print("\nAUC by minute bucket:")
    for (lo, hi), label in zip(MINUTE_BUCKETS, BUCKET_LABELS):
        mask = (oof_df["minute"] >= lo) & (oof_df["minute"] < hi)
        sub  = oof_df[mask]
        if len(sub) < 100:
            continue
        auc = roc_auc_score(sub["blue_win"], sub["oof_pred"])
        print(f"  {label:>6} min : AUC={auc:.4f}  ({len(sub):>7,} rows)")


if __name__ == "__main__":
    main()
