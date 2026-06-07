"""
10_compare_models.py
====================
Uniform model comparison on a COMMON held-out game set.

Reports, per model, BOTH views the user asked for:
  * pre-game / early-game AUC (minute buckets 0-1, 1-5, 5-10, ...) — the hard,
    skill-relevant regime, and
  * pooled all-minutes AUC — overall trajectory tracking,
plus calibration (ECE, Brier).

Holdout = the same seed-42 / val_frac-0.15 game split the GNNs and transformers
used, so it is genuinely out-of-sample for every model here. For 04a (LightGBM,
trained with GroupKFold over all games) we use its OUT-OF-FOLD predictions, which
are out-of-sample for every game — fair.

Snapshot models (04a, 04e, 04f) are recomputed uniformly here. Sequence models
(04b, 04c) are added from their training logs (≈same split) for context, clearly
labelled — uniform recompute of the sequence models is a follow-up.

Outputs:
  reports/compare_auc_by_minute.png    - AUC-by-minute lines, all models
  reports/compare_calibration.png      - reliability curves
  reports/compare_scaling_curve.png    - AUC/ECE vs #games (04e, 04b)
  reports/model_comparison.csv / .md   - the master table

Usage:
  conda run -n lol_shap_env python src/10_compare_models.py
"""

from __future__ import annotations

import importlib.machinery, importlib.util
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = Path(__import__("os").environ.get("LOL_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
FEATURES     = DATA_DIR / "features.parquet"
MODELS       = PROJECT_ROOT / "models"
REPORTS      = PROJECT_ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

SEED, VAL_FRAC = 42, 0.15
BUCKETS = [(0, 1), (1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 999)]
BLABELS = ["0-1", "1-5", "5-10", "10-15", "15-20", "20-25", "25+"]


def _load(modpath, name):
    loader = importlib.machinery.SourceFileLoader(name, str(modpath))
    spec = importlib.util.spec_from_loader(name, loader)
    m = importlib.util.module_from_spec(spec); loader.exec_module(m); return m

m04e = _load(PROJECT_ROOT / "src" / "04e_train_gnn.py", "m04e")
m04f = _load(PROJECT_ROOT / "src" / "04f_train_gnn_context.py", "m04f")


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1); idx = np.digitize(p, edges[1:-1]); e = 0.0
    for b in range(bins):
        mb = idx == b
        if mb.any():
            e += mb.mean() * abs(p[mb].mean() - y[mb].mean())
    return float(e)


def _auc_acc(y, p):
    auc = roc_auc_score(y, p) if pd.Series(y).nunique() == 2 else np.nan
    acc = float(((np.asarray(p) > 0.5).astype(int) == np.asarray(y)).mean())
    return auc, acc


def metrics(df):
    """df: columns minute, blue_win, pred. Returns by-bucket AUC+ACC + summary.
    Accuracy is at the 0.5 threshold (base rate ~50%), matching how the LoL/MOBA
    papers report — for apples-to-apples comparison."""
    out = {}
    for (lo, hi), lab in zip(BUCKETS, BLABELS):
        s = df[(df.minute >= lo) & (df.minute < hi)]
        if len(s) > 50 and s.blue_win.nunique() == 2:
            out[f"auc_{lab}"], out[f"acc_{lab}"] = _auc_acc(s.blue_win, s.pred)
        else:
            out[f"auc_{lab}"] = out[f"acc_{lab}"] = np.nan
    early = df[df.minute < 10]
    out["auc_early_0_10"], out["acc_early_0_10"] = _auc_acc(early.blue_win, early.pred)
    out["auc_pooled"], out["acc_pooled"] = _auc_acc(df.blue_win, df.pred)
    out["ece"] = ece(df.blue_win.to_numpy(), df.pred.to_numpy())
    out["brier"] = brier_score_loss(df.blue_win, df.pred)
    return out


# Pre-game = neutralize in-game state; keep draft/identity (champion, spells,
# runes), champion-mastery, role, and (04f) player history. So 04e ≈ draft-only
# and 04f ≈ draft + player skill/form — directly comparable to the literature's
# draft-only (~55-57%) and player-aware (~62-90%) pre-game numbers.
PREGAME_KEEP_NUM = {"mastery_level", "mastery_points", "days_since_last_played"}
INGAME_NUM_IDX = [i for i, s in enumerate(m04e.NUM_SUFFIXES) if s not in PREGAME_KEEP_NUM]


# ── Holdout = replicate the GNN/transformer split ───────────────────────────────

def holdout_games():
    gid = pd.read_parquet(FEATURES, columns=["game_id"])["game_id"]
    games = gid.unique()
    rng = np.random.default_rng(SEED); rng.shuffle(games)
    return set(games[:int(len(games) * VAL_FRAC)])


# ── Per-model prediction on holdout ─────────────────────────────────────────────

@torch.no_grad()
def predict_gnn(ckpt_path, df, device, with_context=False):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    if with_context:
        model = m04f.ContextGameGNN(ck["vocabs"], K=a["k"], d=a["d"], layers=a["layers"], ctx_d=a["ctx_d"]).to(device)
        model.load_state_dict(ck["state_dict"]); model.eval()
        Xn, Xc = m04f.to_node_tensors(df, ck["vocabs"], ck["num_mean"], ck["num_std"])
        Xn, Xc = torch.from_numpy(Xn), torch.from_numpy(Xc)
        per_player, game_meta = m04f.build_history_index(pd.read_parquet(DATA_DIR / "player_game_summary.parquet"))
        gids = df["game_id"].to_numpy(); uniq = pd.unique(gids)
        gidx = pd.Series(np.arange(len(uniq)), index=uniq).loc[gids].to_numpy()
        feat, mask = m04f.precompute_game_history(uniq, per_player, game_meta, a["k"])
        feat = (feat - ck["h_mean"]) / ck["h_std"]
        H, M = torch.from_numpy(feat), torch.from_numpy(mask)
        gi = torch.from_numpy(gidx)
        preds = []
        for i in range(0, len(df), 4096):
            g = gi[i:i+4096]
            lg = model(Xn[i:i+4096].to(device), Xc[i:i+4096].to(device), H[g].to(device), M[g].to(device))
            preds.append(torch.sigmoid(lg).cpu().numpy())
        return np.concatenate(preds)
    else:
        model = m04e.EquivariantGameGNN(ck["vocabs"], d=a["d"], layers=a["layers"]).to(device)
        model.load_state_dict(ck["state_dict"]); model.eval()
        Xn, Xc, _ = m04e.to_tensors(df, ck["vocabs"], ck["num_mean"], ck["num_std"])
        preds = []
        for i in range(0, len(df), 4096):
            lg = model(Xn[i:i+4096].to(device), Xc[i:i+4096].to(device))
            preds.append(torch.sigmoid(lg).cpu().numpy())
        return np.concatenate(preds)


@torch.no_grad()
def pregame_predict(ckpt_path, pg_df, device, with_context):
    """One prediction per game using ONLY pre-game info (in-game numeric zeroed)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False); a = ck["args"]
    if with_context:
        model = m04f.ContextGameGNN(ck["vocabs"], K=a["k"], d=a["d"], layers=a["layers"], ctx_d=a["ctx_d"]).to(device)
        model.load_state_dict(ck["state_dict"]); model.eval()
        Xn, Xc = m04f.to_node_tensors(pg_df, ck["vocabs"], ck["num_mean"], ck["num_std"])
        Xn, Xc = torch.from_numpy(Xn), torch.from_numpy(Xc)
        Xn[:, :, INGAME_NUM_IDX] = 0.0
        pp, gm = m04f.build_history_index(pd.read_parquet(DATA_DIR / "player_game_summary.parquet"))
        feat, mask = m04f.precompute_game_history(pg_df["game_id"].to_numpy(), pp, gm, a["k"])
        feat = (feat - ck["h_mean"]) / ck["h_std"]
        H, M = torch.from_numpy(feat), torch.from_numpy(mask)
        preds = []
        for i in range(0, len(pg_df), 4096):
            lg = model(Xn[i:i+4096].to(device), Xc[i:i+4096].to(device), H[i:i+4096].to(device), M[i:i+4096].to(device))
            preds.append(torch.sigmoid(lg).cpu().numpy())
        return np.concatenate(preds)
    else:
        model = m04e.EquivariantGameGNN(ck["vocabs"], d=a["d"], layers=a["layers"]).to(device)
        model.load_state_dict(ck["state_dict"]); model.eval()
        Xn, Xc, _ = m04e.to_tensors(pg_df, ck["vocabs"], ck["num_mean"], ck["num_std"])
        Xn[:, :, INGAME_NUM_IDX] = 0.0
        preds = []
        for i in range(0, len(pg_df), 4096):
            lg = model(Xn[i:i+4096].to(device), Xc[i:i+4096].to(device))
            preds.append(torch.sigmoid(lg).cpu().numpy())
        return np.concatenate(preds)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hold = holdout_games()
    log.info("Holdout games: %d", len(hold))

    # Load holdout rows once (node features + meta) for the GNNs.
    num_cols = [f"{s}_{q}" for s in m04e.SLOTS for q in m04e.NUM_SUFFIXES]
    cat_cols = [f"{s}_{q}" for s in m04e.SLOTS for q in m04e.CAT_SUFFIXES]
    df = pd.read_parquet(FEATURES, columns=num_cols + cat_cols + ["game_id", "minute", "blue_win"],
                         filters=[("game_id", "in", list(hold))])
    for c in cat_cols:
        df[c] = df[c].astype("int32")
    base = df[["game_id", "minute", "blue_win"]].reset_index(drop=True)
    log.info("Holdout rows: %d", len(df))

    results = {}      # model -> metrics dict
    preds_for_plot = {}

    # 04a — LightGBM OOF (out-of-fold = fair OOS), restricted to holdout games.
    cv = pickle.load(open(MODELS / "lgbm_snapshot_cv.pkl", "rb"))
    oof = cv["oof_df"]
    oof = oof[oof["game_id"].isin(hold)]
    pcol = "oof_pred" if "oof_pred" in oof else [c for c in oof.columns if "pred" in c][0]
    d04a = oof.rename(columns={pcol: "pred"})[["minute", "blue_win", "pred"]]
    results["04a LightGBM (OOF)"] = metrics(d04a); preds_for_plot["04a LightGBM"] = d04a
    log.info("04a done")

    # 04e — GNN.
    p = predict_gnn(MODELS / "gnn_snapshot.pt", df, device, with_context=False)
    d04e = base.assign(pred=p); results["04e GNN"] = metrics(d04e); preds_for_plot["04e GNN"] = d04e
    log.info("04e done")

    # 04f — GNN + game context.
    p = predict_gnn(MODELS / "gnn_context_model.pt", df, device, with_context=True)
    d04f = base.assign(pred=p); results["04f GNN+ctx"] = metrics(d04f); preds_for_plot["04f GNN+ctx"] = d04f
    log.info("04f done")

    # ── Pre-game evaluation (draft + identity + history; in-game state zeroed) ──
    pg = df.groupby("game_id", as_index=False).first()
    y_pg = pg["blue_win"].to_numpy()
    pregame = {}
    for name, ckpt, ctx in [("04e GNN (draft-only)", MODELS / "gnn_snapshot.pt", False),
                            ("04f GNN+ctx (draft+player history)", MODELS / "gnn_context_model.pt", True)]:
        pp = pregame_predict(ckpt, pg, device, ctx)
        auc, acc = _auc_acc(y_pg, pp)
        pregame[name] = {"pregame_AUC": auc, "pregame_ACC": acc, "n_games": len(pg)}
        log.info("pregame %s: AUC %.4f ACC %.4f", name, auc, acc)
    pregame_tbl = pd.DataFrame(pregame).T

    # ── In-game table (AUC + ACC) ──────────────────────────────────────────────
    tbl = pd.DataFrame(results).T
    full = [f"{m}_{l}" for l in BLABELS for m in ("auc", "acc")] + \
           ["auc_early_0_10", "acc_early_0_10", "auc_pooled", "acc_pooled", "ece", "brier"]
    tbl = tbl[[c for c in full if c in tbl.columns]]
    tbl.to_csv(REPORTS / "model_comparison.csv")
    focus = tbl[["acc_5-10", "acc_10-15", "acc_15-20", "acc_25+",
                 "acc_early_0_10", "acc_pooled", "auc_pooled", "ece"]]
    with open(REPORTS / "model_comparison.md", "w") as f:
        f.write("# Model comparison (common seed-42 holdout)\n\n")
        f.write("## In-game, by stage — ACCURACY (papers' metric) + pooled AUC/ECE\n\n```\n")
        f.write(focus.round(4).to_string())
        f.write("\n```\n\n## Pre-game (draft + identity + player history; in-game state neutralized)\n\n```\n")
        f.write(pregame_tbl.round(4).to_string())
        f.write("\n```\n\nSnapshot models recomputed uniformly on common holdout; 04a uses out-of-fold preds.\n")
        f.write("Accuracy at 0.5 threshold (base rate ~50%). Full AUC+ACC by bucket in model_comparison.csv.\n")
    log.info("\nIN-GAME (acc by stage + pooled):\n%s", focus.round(4).to_string())
    log.info("\nPRE-GAME:\n%s", pregame_tbl.round(4).to_string())

    # ── Plot: AUC by minute ─────────────────────────────────────────────────
    mids = [0.5, 3, 7.5, 12.5, 17.5, 22.5, 27]
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, d in preds_for_plot.items():
        ys = [results[ {"04a LightGBM":"04a LightGBM (OOF)","04e GNN":"04e GNN","04f GNN+ctx":"04f GNN+ctx"}[name] ][f"auc_{l}"] for l in BLABELS]
        ax.plot(mids, ys, "o-", label=name, lw=1.8)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("game minute"); ax.set_ylabel("AUC"); ax.set_ylim(0.5, 1.0)
    ax.set_title("AUC by minute on common holdout (snapshot models)"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORTS / "compare_auc_by_minute.png", dpi=150); plt.close(fig)
    log.info("Saved reports/compare_auc_by_minute.png")

    # ── Plot: calibration ───────────────────────────────────────────────────
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, d in preds_for_plot.items():
        pt, pp = calibration_curve(d.blue_win, d.pred, n_bins=15, strategy="quantile")
        ax.plot(pp, pt, "o-", label=name, lw=1.5, ms=4)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("predicted"); ax.set_ylabel("observed"); ax.set_title("Calibration (holdout)"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORTS / "compare_calibration.png", dpi=150); plt.close(fig)
    log.info("Saved reports/compare_calibration.png")

    # ── Plot: scaling curve (from training runs) ────────────────────────────
    scaling = {
        "04e GNN AUC":  {25_000: 0.8133, 50_000: 0.8175, 133_000: 0.8345},
        "04e GNN ECE":  {25_000: 0.0271, 50_000: 0.0227, 133_000: 0.0134},
        "04b Tf 0-5m":  {25_000: 0.6010, 50_000: 0.6052, 133_000: 0.6151},
    }
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    xs = [25_000, 50_000, 133_000]
    ax1.plot(xs, [scaling["04e GNN AUC"][x] for x in xs], "o-", color="#1565C0", label="04e GNN pooled AUC")
    ax1.plot(xs, [scaling["04b Tf 0-5m"][x] for x in xs], "s-", color="#2E7D32", label="04b transformer 0-5min AUC")
    ax1.set_xlabel("# games (train)"); ax1.set_ylabel("AUC"); ax1.set_xscale("log")
    ax2 = ax1.twinx()
    ax2.plot(xs, [scaling["04e GNN ECE"][x] for x in xs], "^--", color="#C62828", label="04e GNN ECE")
    ax2.set_ylabel("ECE (lower=better)", color="#C62828")
    ax1.legend(loc="center right"); ax1.set_title("Data-scaling: still climbing at 133k (esp. early game + calibration)")
    fig.tight_layout(); fig.savefig(REPORTS / "compare_scaling_curve.png", dpi=150); plt.close(fig)
    log.info("Saved reports/compare_scaling_curve.png")

    print("\n" + "=" * 88)
    print("  UNIFORM MODEL COMPARISON (common seed-42 holdout, %d games)" % len(hold))
    print("=" * 88)
    print("IN-GAME — accuracy by stage + pooled AUC/ECE (apples-to-apples w/ papers):")
    print(focus.round(4).to_string())
    print("\nPRE-GAME — outcome from draft + identity + player history (in-game neutralized):")
    print(pregame_tbl.round(4).to_string())
    print("=" * 88)


if __name__ == "__main__":
    main()
