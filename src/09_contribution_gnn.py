"""
09_contribution_gnn.py
======================
PHASE 2 — exact counterfactual player contribution on the trained equivariant GNN.

This is the paper payoff on a real, well-calibrated model. We hold the trained
04e GNN fixed and compute each player's contribution to P(blue win) at each minute
via EXACT per-team Shapley over the 5 team slots (2^5 = 32 coalitions/team). The
"removal" of a player = swapping that node's features to an on-manifold,
ROLE-CONDITIONED real replacement (a real player at the same role, sampled from
the population); interactions are carried by the GNN's message passing (a removed
node changes everyone's embedding). The other team is held at its real values.

This is the native version of the 08 Phase-0 engine: same exact-Shapley math, but
on the equivariant graph predictor and a proper replacement baseline, in
win-PROBABILITY space.

Outputs:
  data/processed/gnn_contributions.parquet   - per (game,minute,slot) contribution
  reports/gnn_contribution_example.png        - a sample game's per-player timeline
                                                + mean |contribution| by role

Usage:
  conda run -n lol_shap_env python src/09_contribution_gnn.py --n-games 30 --k 16   # workstation
  # full sweep -> OSC (GPU). Cost ~ rows x 2 x 32 x K forwards.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import logging
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = Path(__import__("os").environ.get("LOL_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
FEATURES     = DATA_DIR / "features.parquet"
MODEL_PATH   = PROJECT_ROOT / "models" / "gnn_snapshot.pt"
REPORTS_DIR  = PROJECT_ROOT / "reports"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
LOG_DIR      = PROJECT_ROOT / "logs"
for _d in (REPORTS_DIR, PROC_DIR, LOG_DIR):
    _d.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOG_DIR / "09_contribution_gnn.log", mode="w")])
log = logging.getLogger(__name__)

# Reuse the 04e model + helpers (numeric filename -> load via importlib).
_loader = importlib.machinery.SourceFileLoader("m04e", str(PROJECT_ROOT / "src" / "04e_train_gnn.py"))
_spec = importlib.util.spec_from_loader("m04e", _loader)
m04e = importlib.util.module_from_spec(_spec); _loader.exec_module(m04e)
SLOTS, ROLES, NUM_SUFFIXES, CAT_SUFFIXES = m04e.SLOTS, m04e.ROLES, m04e.NUM_SUFFIXES, m04e.CAT_SUFFIXES

# Exact Shapley over 5 players.
_M = 5
_SUBSETS = [frozenset(s) for r in range(_M + 1) for s in combinations(range(_M), r)]
_W = [math.factorial(s) * math.factorial(_M - s - 1) / math.factorial(_M) for s in range(_M)]


def load_model(device):
    ck = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = m04e.EquivariantGameGNN(ck["vocabs"], d=ck["args"]["d"], layers=ck["args"]["layers"]).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck


@torch.no_grad()
def prob(model, Xn, Xc):
    return torch.sigmoid(model(Xn, Xc))


@torch.no_grad()
def team_contrib(model, Xn_row, Xc_row, team_nodes, pool_by_role, K, device, rng):
    """Exact Shapley (prob space) for one team's 5 nodes of ONE game-minute.
    Xn_row (10,32), Xc_row (10,5) are the real game. team_nodes = node indices (5).
    Replacements for slot j are sampled from pool_by_role[ROLES[node]]."""
    # Pre-sample K replacements per team node: rep_n (5,K,32), rep_c (5,K,5).
    rep_n = np.zeros((5, K, Xn_row.shape[1]), dtype=np.float32)
    rep_c = np.zeros((5, K, Xc_row.shape[1]), dtype=np.int64)
    for j, node in enumerate(team_nodes):
        pn, pc = pool_by_role[ROLES[node]]
        idx = rng.integers(0, len(pn), size=K)
        rep_n[j] = pn[idx]; rep_c[j] = pc[idx]
    rep_n = torch.from_numpy(rep_n).to(device); rep_c = torch.from_numpy(rep_c).to(device)

    base_n = Xn_row.unsqueeze(0).expand(K, -1, -1).contiguous()   # (K,10,32)
    base_c = Xc_row.unsqueeze(0).expand(K, -1, -1).contiguous()
    vS = {}
    for S in _SUBSETS:                              # S = team-local indices kept REAL
        Xn = base_n.clone(); Xc = base_c.clone()
        for j, node in enumerate(team_nodes):
            if j not in S:                          # replaced by sampled replacements
                Xn[:, node, :] = rep_n[j]
                Xc[:, node, :] = rep_c[j]
        vS[S] = float(prob(model, Xn, Xc).mean())   # avg over K replacements
    phi = np.zeros(5)
    for i in range(5):
        for S in _SUBSETS:
            if i in S:
                continue
            phi[i] = phi[i] + _W[len(S)] * (vS[S | {i}] - vS[S])
    full = vS[frozenset(range(5))]; empty = vS[frozenset()]
    return phi, full, empty


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: GNN player contribution")
    p.add_argument("--n-games", type=int, default=30)
    p.add_argument("--pool-games", type=int, default=1500)
    p.add_argument("--k", type=int, default=16, help="replacement samples")
    p.add_argument("--max-minute", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args(); t0 = time.time()
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    model, ck = load_model(device)
    vocabs, num_mean, num_std = ck["vocabs"], ck["num_mean"], ck["num_std"]
    log.info("Loaded GNN (val AUC %.3f, ECE %.3f)", ck["val_metrics"]["auc"], ck["val_metrics"]["ece"])

    num_cols = [f"{s}_{q}" for s in SLOTS for q in NUM_SUFFIXES]
    cat_cols = [f"{s}_{q}" for s in SLOTS for q in CAT_SUFFIXES]
    df = pd.read_parquet(FEATURES, columns=num_cols + cat_cols + ["game_id", "minute", "blue_win"])
    df = df[df["minute"] <= args.max_minute]
    ids = df["game_id"].unique()

    # Replacement pool: real node vectors grouped by role (0..4).
    pool_ids = rng.choice(ids, size=min(args.pool_games, len(ids)), replace=False)
    pdf = df[df["game_id"].isin(pool_ids)]
    Xn_pool, Xc_pool, _ = m04e.to_tensors(pdf, vocabs, num_mean, num_std)   # (P,10,32),(P,10,5)
    pool_by_role = {}
    for r in range(5):
        nodes = [n for n in range(10) if ROLES[n] == r]
        pn = np.concatenate([Xn_pool[:, n, :].numpy() for n in nodes], 0)
        pc = np.concatenate([Xc_pool[:, n, :].numpy() for n in nodes], 0)
        pool_by_role[r] = (pn, pc)
    log.info("Replacement pool: %d rows/role from %d games", len(pool_by_role[0][0]), len(pool_ids))

    # Explained set.
    rest = np.setdiff1d(ids, pool_ids)
    ex_ids = rng.choice(rest if len(rest) >= args.n_games else ids, size=args.n_games, replace=False)
    edf = df[df["game_id"].isin(ex_ids)].reset_index(drop=True)
    Xn, Xc, _ = m04e.to_tensors(edf, vocabs, num_mean, num_std)
    Xn, Xc = Xn.to(device), Xc.to(device)
    log.info("Explained: %d games -> %d rows", len(ex_ids), len(edf))

    blue_nodes, red_nodes = list(range(5)), list(range(5, 10))
    contribs = np.zeros((len(edf), 10), dtype=np.float32)
    eff_err = []
    for i in range(len(edf)):
        pb, fb, eb = team_contrib(model, Xn[i], Xc[i], blue_nodes, pool_by_role, args.k, device, rng)
        pr, fr, er = team_contrib(model, Xn[i], Xc[i], red_nodes, pool_by_role, args.k, device, rng)
        contribs[i, :5] = pb; contribs[i, 5:] = pr
        eff_err.append(abs(pb.sum() - (fb - eb)) + abs(pr.sum() - (fr - er)))
        if (i + 1) % 200 == 0:
            log.info("  ... %d/%d rows", i + 1, len(edf))

    out = edf[["game_id", "minute", "blue_win"]].copy()
    for s, name in enumerate(SLOTS):
        out[name] = contribs[:, s]
    out_path = PROC_DIR / "gnn_contributions.parquet"
    out.to_parquet(out_path, index=False)
    log.info("Saved %s (%d rows)", out_path, len(out))

    # Efficiency check: per-team Shapley sums to v(team real) - v(team replaced).
    log.info("Shapley efficiency residual (per team, mean): %.2e", float(np.mean(eff_err)))

    # Figure: a sample game's per-player contribution timeline + mean |contrib| by role.
    gid = edf["game_id"].iloc[0]
    g = out[out["game_id"] == gid].sort_values("minute")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for s, name in enumerate(SLOTS):
        ax[0].plot(g["minute"], g[name], color=colors[s], label=name, lw=1.6)
    ax[0].axhline(0, color="k", lw=0.7, ls="--")
    ax[0].set_title(f"Per-player contribution to P(blue win) — game {gid}\n(blue win={int(g['blue_win'].iloc[0])})")
    ax[0].set_xlabel("minute"); ax[0].set_ylabel("Shapley contribution (prob)")
    ax[0].legend(fontsize=7, ncol=2)
    role_names = ["top", "jungle", "middle", "bottom", "utility"]
    mean_abs = [np.abs(out[[SLOTS[n] for n in range(10) if ROLES[n] == r]].to_numpy()).mean() for r in range(5)]
    ax[1].bar(role_names, mean_abs, color="#1565C0")
    ax[1].set_title("Mean |contribution| by role (all explained rows)")
    ax[1].set_ylabel("mean |Shapley| (prob)")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "gnn_contribution_example.png", dpi=150)
    plt.close(fig)
    log.info("Saved reports/gnn_contribution_example.png")

    print("\n" + "=" * 64)
    print("  PHASE 2 — GNN player contribution (exact per-team Shapley)")
    print(f"  explained rows : {len(edf):,} ({len(ex_ids)} games)  K={args.k}")
    print(f"  efficiency residual (mean per team): {np.mean(eff_err):.2e}  (should be ~0)")
    print(f"  mean |contribution| by role: " + ", ".join(f"{r}={v:.4f}" for r, v in zip(role_names, mean_abs)))
    print(f"  done in {time.time()-t0:.1f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
