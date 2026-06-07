"""
04e_train_gnn.py
================
PHASE 1 — equivariant per-minute graph predictor (the "real" model family).

Each (game, minute) is a 10-node graph: 5 blue + 5 red players. Node features are
that player's 37 per-slot features (32 numeric + 5 categorical embeddings) plus a
role embedding (top/jungle/mid/bottom/utility). The model predicts P(blue win) at
that minute, with TWO hard symmetry constraints baked in:

  * Within-team permutation invariance — message passing is set-based (masked mean
    aggregation), role is a *feature* not a position, and the readout sums over a
    team. Relabeling players within a team cannot change the output.
  * Team-swap antisymmetry  f(A,B) = 1 - f(B,A) — all weights are team-RELATIVE
    (a node only ever sees "teammate" vs "opponent", never absolute blue/red), and
    the logit is  Σ_blue node_score − Σ_red node_score. Swapping teams negates the
    logit, so sigmoid gives exactly 1 - p.

Why this shape: it composes with the contribution engine (08_phase0). Removing a
player = swapping a node's features to a replacement; exact per-team Shapley over
5 nodes = 32 coalitions. The additive logit (sum of node scores) makes per-player
credit natural, while interactions live in the message passing (a removed node
changes everyone's embedding).

Training optimizes calibration-aware metrics (Brier, ECE) alongside AUC — per the
plan, accuracy is NOT the objective; calibration + intervention deltas are.

Usage:
  conda run -n lol_shap_env python src/04e_train_gnn.py --limit 300 --epochs 3   # workstation smoke
  # full training -> OSC (slurm/train_gpu.slurm with TRAIN_SCRIPT=src/04e_train_gnn.py)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss, roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = Path(__import__("os").environ.get("LOL_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
FEATURES_PATH = DATA_DIR / "features.parquet"
MODELS_DIR    = Path(__import__("os").environ.get("LOL_MODELS_DIR", PROJECT_ROOT / "models"))
REPORTS_DIR   = PROJECT_ROOT / "reports"
LOG_DIR       = PROJECT_ROOT / "logs"
for _d in (MODELS_DIR, REPORTS_DIR, LOG_DIR):
    _d.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_DIR / "04e_train_gnn.log", mode="w")],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SLOTS = [
    "blue_top", "blue_jungle", "blue_middle", "blue_bottom", "blue_utility",
    "red_top",  "red_jungle",  "red_middle",  "red_bottom",  "red_utility",
]
ROLES = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]          # role index per node
TEAM  = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]          # 0=blue, 1=red
CAT_SUFFIXES = ["champion_id", "summoner1_id", "summoner2_id", "keystone", "primary_tree"]
NUM_SUFFIXES = [
    "ability_haste", "ability_power", "armor", "armor_pen_pct", "attack_damage",
    "attack_speed", "cs_total", "current_gold", "days_since_last_played", "dmg_taken",
    "dmg_to_champs", "gold_per_second", "health", "health_max", "hp_ratio",
    "jungle_minions_killed", "level", "magic_dmg_to_champs", "magic_pen", "magic_resist",
    "mastery_level", "mastery_points", "minions_killed", "movement_speed", "omnivamp",
    "phys_dmg_to_champs", "pos_x", "pos_y", "time_enemy_cc", "total_gold",
    "true_dmg_to_champs", "xp",
]

# ── Data ────────────────────────────────────────────────────────────────────────

def load_frame(limit_games: int, seed: int) -> pd.DataFrame:
    num_cols = [f"{s}_{suf}" for s in SLOTS for suf in NUM_SUFFIXES]
    cat_cols = [f"{s}_{suf}" for s in SLOTS for suf in CAT_SUFFIXES]
    cols = num_cols + cat_cols + ["game_id", "minute", "blue_win"]
    log.info("Loading %d feature columns from %s", len(cols), FEATURES_PATH)
    df = pd.read_parquet(FEATURES_PATH, columns=cols)
    if limit_games:
        rng = np.random.default_rng(seed)
        keep = rng.choice(df["game_id"].unique(),
                          size=min(limit_games, df["game_id"].nunique()), replace=False)
        df = df[df["game_id"].isin(keep)].reset_index(drop=True)
    log.info("Frame: %d rows, %d games", len(df), df["game_id"].nunique())
    return df


def build_cat_vocabs(df: pd.DataFrame) -> dict[str, dict[int, int]]:
    """Map raw category ids -> contiguous indices (0 reserved for unknown/pad)."""
    vocabs = {}
    for suf in CAT_SUFFIXES:
        vals = pd.unique(pd.concat([df[f"{s}_{suf}"] for s in SLOTS], ignore_index=True))
        vocabs[suf] = {int(v): i + 1 for i, v in enumerate(sorted(int(x) for x in vals))}
    return vocabs


def to_tensors(df: pd.DataFrame, vocabs, num_mean, num_std):
    """Return (X_num (N,10,32), X_cat (N,10,5) long, y (N,))."""
    N = len(df)
    X_num = np.empty((N, 10, len(NUM_SUFFIXES)), dtype=np.float32)
    for si, s in enumerate(SLOTS):
        block = df[[f"{s}_{suf}" for suf in NUM_SUFFIXES]].to_numpy(np.float32)
        X_num[:, si, :] = (block - num_mean) / num_std
    X_cat = np.zeros((N, 10, len(CAT_SUFFIXES)), dtype=np.int64)
    for si, s in enumerate(SLOTS):
        for ci, suf in enumerate(CAT_SUFFIXES):
            m = vocabs[suf]
            X_cat[:, si, ci] = df[f"{s}_{suf}"].map(lambda v: m.get(int(v), 0)).to_numpy()
    y = df["blue_win"].to_numpy(np.float32)
    return (torch.from_numpy(X_num), torch.from_numpy(X_cat), torch.from_numpy(y))


# ── Model ───────────────────────────────────────────────────────────────────────

class EquivariantGameGNN(nn.Module):
    """Team-relative, permutation-invariant, team-swap-antisymmetric GNN."""

    def __init__(self, vocabs, n_num=32, d=96, layers=3, cat_emb=12, role_emb=12):
        super().__init__()
        self.cat_emb = nn.ModuleDict({
            suf: nn.Embedding(len(vocabs[suf]) + 1, cat_emb) for suf in CAT_SUFFIXES
        })
        self.role_emb = nn.Embedding(5, role_emb)
        in_dim = n_num + cat_emb * len(CAT_SUFFIXES) + role_emb
        self.in_proj = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.LayerNorm(d))

        # Per-layer team-relative message transforms: teammate, opponent, lane-opp.
        self.msg_team = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.msg_opp  = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.msg_lane = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.upd      = nn.ModuleList(
            nn.Sequential(nn.Linear(d * 4, d), nn.ReLU(), nn.LayerNorm(d)) for _ in range(layers))

        self.node_score = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

        # Static masks (10x10): teammate (same team, no self), opponent, lane (same role, diff team).
        role = torch.tensor(ROLES); team = torch.tensor(TEAM)
        same_team = (team[:, None] == team[None, :])
        eye = torch.eye(10, dtype=torch.bool)
        self.register_buffer("m_team", (same_team & ~eye).float())
        self.register_buffer("m_opp",  (~same_team).float())
        self.register_buffer("m_lane", ((role[:, None] == role[None, :]) & ~same_team).float())
        self.register_buffer("role_idx", role)
        self.register_buffer("team_idx", team)

    def _masked_mean(self, H, mask):
        # H (B,10,d), mask (10,10) -> (B,10,d) mean of neighbours per node
        deg = mask.sum(1, keepdim=True).clamp(min=1.0)              # (10,1)
        return torch.einsum("ij,bjd->bid", mask, H) / deg

    def node_embeddings(self, X_num, X_cat):
        B = X_num.shape[0]
        cats = [self.cat_emb[suf](X_cat[:, :, ci]) for ci, suf in enumerate(CAT_SUFFIXES)]
        role = self.role_emb(self.role_idx).unsqueeze(0).expand(B, -1, -1)
        H = self.in_proj(torch.cat([X_num] + cats + [role], dim=-1))
        for mt, mo, ml, up in zip(self.msg_team, self.msg_opp, self.msg_lane, self.upd):
            t = self._masked_mean(mt(H), self.m_team)
            o = self._masked_mean(mo(H), self.m_opp)
            l = self._masked_mean(ml(H), self.m_lane)
            H = up(torch.cat([H, t, o, l], dim=-1))
        return H

    def forward(self, X_num, X_cat):
        H = self.node_embeddings(X_num, X_cat)
        score = self.node_score(H).squeeze(-1)                      # (B,10) per-player
        blue = score[:, self.team_idx == 0].sum(1)
        red  = score[:, self.team_idx == 1].sum(1)
        return blue - red                                          # logit; antisymmetric


# ── Metrics ─────────────────────────────────────────────────────────────────────

def expected_calibration_error(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


@torch.no_grad()
def evaluate(model, Xn, Xc, y, device, bs=4096):
    model.eval()
    ps = []
    for i in range(0, len(y), bs):
        logit = model(Xn[i:i+bs].to(device), Xc[i:i+bs].to(device))
        ps.append(torch.sigmoid(logit).cpu().numpy())
    p = np.concatenate(ps); yt = y.numpy()
    return {"auc": roc_auc_score(yt, p), "brier": brier_score_loss(yt, p),
            "ece": expected_calibration_error(yt, p)}


@torch.no_grad()
def antisymmetry_check(model, Xn, Xc, device, n=256):
    """Swap blue<->red node blocks; logit should negate."""
    model.eval()
    xn, xc = Xn[:n].to(device), Xc[:n].to(device)
    swap = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    l1 = model(xn, xc)
    l2 = model(xn[:, swap], xc[:, swap])
    return float((l1 + l2).abs().max())


# ── Train ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 equivariant per-minute GNN")
    p.add_argument("--limit", type=int, default=0, help="limit #games (0=all)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--d", type=int, default=96)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    df = load_frame(args.limit, args.seed)

    # Game-level split (never split a game's minutes across train/val).
    rng = np.random.default_rng(args.seed)
    games = df["game_id"].unique()
    rng.shuffle(games)
    n_val = int(len(games) * args.val_frac)
    val_games = set(games[:n_val])
    is_val = df["game_id"].isin(val_games).to_numpy()
    tr, va = df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
    log.info("Train rows %d (%d games) | Val rows %d (%d games)",
             len(tr), tr["game_id"].nunique(), len(va), va["game_id"].nunique())

    vocabs = build_cat_vocabs(tr)
    num_cols_flat = np.concatenate(
        [tr[[f"{s}_{suf}" for suf in NUM_SUFFIXES]].to_numpy(np.float32) for s in SLOTS], axis=0)
    num_mean = num_cols_flat.mean(0); num_std = num_cols_flat.std(0) + 1e-6

    Xn_tr, Xc_tr, y_tr = to_tensors(tr, vocabs, num_mean, num_std)
    Xn_va, Xc_va, y_va = to_tensors(va, vocabs, num_mean, num_std)

    model = EquivariantGameGNN(vocabs, d=args.d, layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model params: %d", n_params)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(y_tr)
    best = {"auc": 0.0}
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            b = perm[i:i+args.batch_size]
            logit = model(Xn_tr[b].to(device), Xc_tr[b].to(device))
            loss = loss_fn(logit, y_tr[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        m = evaluate(model, Xn_va, Xc_va, y_va, device)
        log.info("ep %2d | loss %.4f | val AUC %.4f Brier %.4f ECE %.4f",
                 ep, tot / n, m["auc"], m["brier"], m["ece"])
        if m["auc"] > best["auc"]:
            best = {**m, "epoch": ep}
            torch.save({"state_dict": model.state_dict(), "vocabs": vocabs,
                        "num_mean": num_mean, "num_std": num_std,
                        "args": vars(args), "val_metrics": m},
                       MODELS_DIR / "gnn_snapshot.pt")

    anti = antisymmetry_check(model, Xn_va, Xc_va, device)
    log.info("Antisymmetry max|f(A,B)+f(B,A)| (logits, should be ~0): %.2e", anti)

    print("\n" + "=" * 64)
    print("  PHASE 1 — equivariant per-minute GNN")
    print(f"  params {n_params:,} | device {device} | {time.time()-t0:.1f}s")
    print(f"  best val: AUC {best['auc']:.4f}  Brier {best['brier']:.4f}  "
          f"ECE {best['ece']:.4f}  (epoch {best.get('epoch','-')})")
    print(f"  antisymmetry residual (logit): {anti:.2e}  (hard constraint -> ~0)")
    print(f"  saved: models/gnn_snapshot.pt")
    print("=" * 64)


if __name__ == "__main__":
    main()
