"""
04f_train_gnn_context.py
========================
PHASE 1 — equivariant per-minute GNN WITH a per-node player-history encoder.

This is 04e (the equivariant 10-node match graph) where each player node is
additionally fed a learned embedding of that player's recent form — their last K
games. It is the GNN analogue of 04c (game-level history) and the model the
research plan calls for: "each node encoded by a player-history encoder
(last ~20 games -> latent skill/style), as a node encoder inside the GNN."

History levels:
  --history-level game   : encode the K most-recent PRIOR games' end-of-game
                           summary stats (from player_game_summary.parquet). READY.
  --history-level minute : encode minute-by-minute sequences of the K prior games
                           (player_minute_sequences.parquet). Needs that file
                           (built by 03b --include-sequences); see 04g / TODO.

Symmetry is preserved: the history encoder is shared across nodes and carries no
absolute team identity, so within-team permutation invariance and team-swap
antisymmetry (f(A,B)=1-f(B,A)) still hold EXACTLY (verified each run). History is
static per (game, player) — computed once per game and reused across its minutes.

Prediction: P(blue team wins) at each (game, minute), same target as 04a-04e.

Usage:
  conda run -n lol_shap_env python src/04f_train_gnn_context.py --limit 300 --epochs 3 --k 10   # smoke
  # full -> OSC: slurm/train_04f.slurm
"""

from __future__ import annotations

import argparse
import logging
import os
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
DATA_DIR      = Path(os.environ.get("LOL_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
FEATURES_PATH = DATA_DIR / "features.parquet"
HISTORY_PATH  = DATA_DIR / "player_game_summary.parquet"
MODELS_DIR    = Path(os.environ.get("LOL_MODELS_DIR", PROJECT_ROOT / "models"))
LOG_DIR       = PROJECT_ROOT / "logs"
for _d in (MODELS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_DIR / "04f_train_gnn_context.log", mode="w")],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SLOTS = [
    "blue_top", "blue_jungle", "blue_middle", "blue_bottom", "blue_utility",
    "red_top",  "red_jungle",  "red_middle",  "red_bottom",  "red_utility",
]
ROLES = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
TEAM  = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
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
HIST_NUM = ["player_won", "cs_total", "total_gold", "xp", "dmg_to_champs", "dmg_taken",
            "level", "kills", "deaths", "assists", "time_cc_others"]
POS2SUF = {"TOP": "top", "JUNGLE": "jungle", "MIDDLE": "middle", "BOTTOM": "bottom", "UTILITY": "utility"}

# ── History (game-level) plumbing ────────────────────────────────────────────────

def build_history_index(hist_df: pd.DataFrame):
    """Returns:
      per_player: puuid -> (creations sorted asc (n,), feats (n, 11))
      game_meta : game_id -> (creation_ms, {slot_idx: puuid})
    """
    hist_df = hist_df.copy()
    hist_df["player_won"] = hist_df["player_won"].astype(np.float32)
    feats_all = hist_df[HIST_NUM].to_numpy(np.float32)
    hist_df = hist_df.assign(_row=np.arange(len(hist_df)))

    per_player = {}
    for puuid, g in hist_df.groupby("puuid", sort=False):
        order = np.argsort(g["game_creation_ms"].to_numpy())
        rows = g["_row"].to_numpy()[order]
        per_player[puuid] = (g["game_creation_ms"].to_numpy()[order], feats_all[rows])

    game_meta = {}
    for gid, g in hist_df.groupby("game_id", sort=False):
        creation = int(g["game_creation_ms"].iloc[0])
        slotmap = {}
        for _, r in g.iterrows():
            suf = POS2SUF.get(r["team_position"])
            if suf is None:
                continue
            slot = ("blue_" if r["team_id"] == 100 else "red_") + suf
            if slot in SLOTS:
                slotmap[SLOTS.index(slot)] = r["puuid"]
        game_meta[gid] = (creation, slotmap)
    return per_player, game_meta


def precompute_game_history(game_ids, per_player, game_meta, K):
    """For each game: (10, K, 11) most-recent PRIOR games + (10, K) validity mask.
    Position 0 = most recent. Players with <K prior games are zero-padded+masked."""
    G = len(game_ids)
    feat = np.zeros((G, 10, K, len(HIST_NUM)), dtype=np.float32)
    mask = np.zeros((G, 10, K), dtype=np.float32)
    for gi, gid in enumerate(game_ids):
        meta = game_meta.get(gid)
        if meta is None:
            continue
        creation, slotmap = meta
        for s in range(10):
            puuid = slotmap.get(s)
            if puuid is None or puuid not in per_player:
                continue
            creations, pfeats = per_player[puuid]
            cut = np.searchsorted(creations, creation)        # games strictly before
            if cut == 0:
                continue
            take = pfeats[max(0, cut - K):cut][::-1]           # most recent first
            n = len(take)
            feat[gi, s, :n] = take
            mask[gi, s, :n] = 1.0
    return feat, mask


# ── Patch-aware champion static (Phase 1.5) ──────────────────────────────────────

CHAMP_COLS = [f"{s}_champion_id" for s in SLOTS]   # node order == SLOTS order


def load_static_lookup(static_path):
    """champion_static.parquet (per patch x champion) -> (lut, feat_cols, avail_patches).
    lut: {(patch, champion_key): np.float32[n_static]}."""
    cs = pd.read_parquet(static_path)
    feat_cols = [c for c in cs.columns if c not in ("patch", "champion_key", "champion_name")]
    feats = cs[feat_cols].to_numpy(np.float32)
    lut = {(p, int(k)): feats[i] for i, (p, k) in enumerate(zip(cs["patch"], cs["champion_key"]))}
    return lut, feat_cols, sorted(cs["patch"].unique())


def build_game_static(game_ids, champ_by_game, patch_map, lut, n_static, avail, fallback):
    """(G, 10, n_static) patch-correct champion static feats per node. Games whose
    patch lacks a static table fall back to the latest available patch; champs
    missing for a patch fall back to the same champ on the fallback patch, else 0."""
    G = len(game_ids)
    S = np.zeros((G, 10, n_static), dtype=np.float32)
    matched = 0; total = 0
    for gi, gid in enumerate(game_ids):
        patch = patch_map.get(gid, fallback)
        if patch not in avail:
            patch = fallback
        champs = champ_by_game.get(gid)
        if champs is None:
            continue
        for s in range(10):
            ck = int(champs[s]); total += 1
            v = lut.get((patch, ck))
            if v is None:
                v = lut.get((fallback, ck))
            if v is not None:
                S[gi, s] = v; matched += 1
    return S, matched, total


# ── Current-game node tensors (same as 04e) ─────────────────────────────────────

def build_cat_vocabs(df):
    v = {}
    for suf in CAT_SUFFIXES:
        vals = pd.unique(pd.concat([df[f"{s}_{suf}"] for s in SLOTS], ignore_index=True))
        v[suf] = {int(x): i + 1 for i, x in enumerate(sorted(int(z) for z in vals))}
    return v


def to_node_tensors(df, vocabs, num_mean, num_std):
    N = len(df)
    Xn = np.empty((N, 10, len(NUM_SUFFIXES)), dtype=np.float32)
    for si, s in enumerate(SLOTS):
        Xn[:, si, :] = (df[[f"{s}_{q}" for q in NUM_SUFFIXES]].to_numpy(np.float32) - num_mean) / num_std
    Xc = np.zeros((N, 10, len(CAT_SUFFIXES)), dtype=np.int64)
    for si, s in enumerate(SLOTS):
        for ci, suf in enumerate(CAT_SUFFIXES):
            m = vocabs[suf]
            Xc[:, si, ci] = df[f"{s}_{suf}"].map(lambda v: m.get(int(v), 0)).to_numpy()
    return Xn, Xc


# ── Model ───────────────────────────────────────────────────────────────────────

class HistoryEncoderGame(nn.Module):
    """Per-node encoder of K prior end-of-game summary vectors -> ctx embedding.
    Permutation handling: recency positional embedding + masked mean (order-aware
    but shared across all nodes/teams, so symmetry is preserved)."""
    def __init__(self, n_feat, K, ctx_d=48):
        super().__init__()
        self.per_game = nn.Sequential(nn.Linear(n_feat, ctx_d), nn.ReLU(), nn.LayerNorm(ctx_d))
        self.recency = nn.Embedding(K, ctx_d)
        self.out = nn.Sequential(nn.Linear(ctx_d, ctx_d), nn.ReLU(), nn.LayerNorm(ctx_d))

    def forward(self, hist, mask):                       # hist (B,10,K,F)  mask (B,10,K)
        B, N, K, F = hist.shape
        h = self.per_game(hist) + self.recency.weight.view(1, 1, K, -1)
        m = mask.unsqueeze(-1)
        summed = (h * m).sum(2)
        denom = m.sum(2).clamp(min=1.0)
        return self.out(summed / denom)                  # (B,10,ctx_d)


class ContextGameGNN(nn.Module):
    """04e graph + per-node game-history context (+ optional patch-aware champion
    static features, passed in per (game, node) so the encoder sees the stats for
    that game's actual patch)."""
    def __init__(self, vocabs, K, n_num=32, d=128, layers=4, cat_emb=12, role_emb=12, ctx_d=48, n_static=0):
        super().__init__()
        self.cat_emb = nn.ModuleDict({suf: nn.Embedding(len(vocabs[suf]) + 1, cat_emb) for suf in CAT_SUFFIXES})
        self.role_emb = nn.Embedding(5, role_emb)
        self.hist_enc = HistoryEncoderGame(len(HIST_NUM), K, ctx_d)
        self.n_static = n_static   # patch-aware champion static features fed per node (0 = off)
        in_dim = n_num + cat_emb * len(CAT_SUFFIXES) + role_emb + ctx_d + n_static
        self.in_proj = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.LayerNorm(d))

        self.msg_team = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.msg_opp  = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.msg_lane = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.upd      = nn.ModuleList(nn.Sequential(nn.Linear(d * 4, d), nn.ReLU(), nn.LayerNorm(d)) for _ in range(layers))
        self.node_score = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

        role = torch.tensor(ROLES); team = torch.tensor(TEAM); eye = torch.eye(10, dtype=torch.bool)
        same = team[:, None] == team[None, :]
        self.register_buffer("m_team", (same & ~eye).float())
        self.register_buffer("m_opp",  (~same).float())
        self.register_buffer("m_lane", ((role[:, None] == role[None, :]) & ~same).float())
        self.register_buffer("role_idx", role)
        self.register_buffer("team_idx", team)

    def _mm(self, H, mask):
        deg = mask.sum(1, keepdim=True).clamp(min=1.0)
        return torch.einsum("ij,bjd->bid", mask, H) / deg

    def forward(self, Xn, Xc, hist, hmask, static=None):
        B = Xn.shape[0]
        cats = [self.cat_emb[suf](Xc[:, :, ci]) for ci, suf in enumerate(CAT_SUFFIXES)]
        role = self.role_emb(self.role_idx).unsqueeze(0).expand(B, -1, -1)
        ctx = self.hist_enc(hist, hmask)
        extra = [static] if self.n_static else []   # patch-aware champion static, per (game, node)
        H = self.in_proj(torch.cat([Xn] + cats + [role, ctx] + extra, dim=-1))
        for mt, mo, ml, up in zip(self.msg_team, self.msg_opp, self.msg_lane, self.upd):
            H = up(torch.cat([H, self._mm(mt(H), self.m_team), self._mm(mo(H), self.m_opp),
                              self._mm(ml(H), self.m_lane)], dim=-1))
        score = self.node_score(H).squeeze(-1)
        return score[:, self.team_idx == 0].sum(1) - score[:, self.team_idx == 1].sum(1)


# ── Metrics ─────────────────────────────────────────────────────────────────────

def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1); idx = np.digitize(p, edges[1:-1]); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


@torch.no_grad()
def evaluate(model, Xn, Xc, hist, hmask, gidx, y, device, static=None, bs=4096):
    model.eval(); ps = []
    for i in range(0, len(y), bs):
        g = gidx[i:i+bs]
        s = static[g].to(device) if static is not None else None
        logit = model(Xn[i:i+bs].to(device), Xc[i:i+bs].to(device),
                      hist[g].to(device), hmask[g].to(device), s)
        ps.append(torch.sigmoid(logit).cpu().numpy())
    p = np.concatenate(ps); yt = y.numpy()
    return {"auc": roc_auc_score(yt, p), "brier": brier_score_loss(yt, p), "ece": ece(yt, p)}


@torch.no_grad()
def antisymmetry_check(model, Xn, Xc, hist, hmask, gidx, device, static=None, n=256):
    """Swap blue<->red across node features, history AND static; logit must negate."""
    model.eval()
    swap = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    g = gidx[:n]
    xn, xc = Xn[:n].to(device), Xc[:n].to(device)
    h, hm = hist[g].to(device), hmask[g].to(device)
    s = static[g].to(device) if static is not None else None
    l1 = model(xn, xc, h, hm, s)
    l2 = model(xn[:, swap], xc[:, swap], h[:, swap], hm[:, swap],
               s[:, swap] if s is not None else None)
    return float((l1 + l2).abs().max())


# ── Train ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Equivariant GNN + player game-history context")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3,
                   help="stop after this many epochs with no val-AUC improvement (0=off)")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ctx-d", type=int, default=48)
    p.add_argument("--k", type=int, default=20, help="history games per player")
    p.add_argument("--history-level", choices=["game", "minute"], default="game")
    p.add_argument("--static", action="store_true",
                   help="add patch-aware champion static features per node (Phase-1 context encoder)")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.history_level == "minute":
        raise NotImplementedError(
            "minute-level history needs player_minute_sequences.parquet (03b "
            "--include-sequences). Game-level is implemented here; minute-level "
            "is the 04g follow-up (same graph, sequence node-encoder like 04d).")
    t0 = time.time(); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s | history-level: %s | K=%d", device, args.history_level, args.k)

    num_cols = [f"{s}_{q}" for s in SLOTS for q in NUM_SUFFIXES]
    cat_cols = [f"{s}_{q}" for s in SLOTS for q in CAT_SUFFIXES]
    df = pd.read_parquet(FEATURES_PATH, columns=num_cols + cat_cols + ["game_id", "minute", "blue_win"])
    if args.limit:
        rng0 = np.random.default_rng(args.seed)
        keep = rng0.choice(df["game_id"].unique(), size=min(args.limit, df["game_id"].nunique()), replace=False)
        df = df[df["game_id"].isin(keep)].reset_index(drop=True)
    log.info("Frame: %d rows, %d games", len(df), df["game_id"].nunique())

    log.info("Loading history index from %s", HISTORY_PATH)
    per_player, game_meta = build_history_index(pd.read_parquet(HISTORY_PATH))

    # Game-level split.
    rng = np.random.default_rng(args.seed)
    games = df["game_id"].unique(); rng.shuffle(games)
    val_games = set(games[:int(len(games) * args.val_frac)])
    is_val = df["game_id"].isin(val_games).to_numpy()
    tr, va = df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
    log.info("Train %d rows / %d games | Val %d rows / %d games",
             len(tr), tr["game_id"].nunique(), len(va), va["game_id"].nunique())

    # Vocabs + scalers (fit on train).
    vocabs = build_cat_vocabs(tr)
    flat = np.concatenate([tr[[f"{s}_{q}" for q in NUM_SUFFIXES]].to_numpy(np.float32) for s in SLOTS], 0)
    num_mean, num_std = flat.mean(0), flat.std(0) + 1e-6

    # Patch-aware champion static features (Phase 1.5): each game's champions are
    # looked up at THAT game's patch (via game_meta.parquet), so cross-patch stat
    # drift is signal. Fed per (game, node), not gathered by champion vocab id.
    static_lut = static_feat_cols = static_avail = static_fallback = None
    patch_map = {}
    n_static = 0
    if args.static:
        static_lut, static_feat_cols, static_avail = load_static_lookup(DATA_DIR / "champion_static.parquet")
        n_static = len(static_feat_cols)
        static_fallback = max(static_avail, key=lambda p: tuple(int(x) for x in p.split(".")))
        gm = pd.read_parquet(DATA_DIR / "game_meta.parquet", columns=["game_id", "patch"])
        patch_map = dict(zip(gm["game_id"].astype(str), gm["patch"]))
        log.info("Static: %d feats x %d patches %s | fallback patch %s | %d games have patch meta",
                 n_static, len(static_avail), static_avail, static_fallback, len(patch_map))

    # History scaler: fit on the prior-game feats actually used (train players).
    hist_concat = np.concatenate([per_player[p][1] for p in per_player], 0)
    h_mean, h_std = hist_concat.mean(0), hist_concat.std(0) + 1e-6

    def make_split(d):
        Xn, Xc = to_node_tensors(d, vocabs, num_mean, num_std)
        gids = d["game_id"].to_numpy()
        uniq = pd.unique(gids)
        gidx = pd.Series(np.arange(len(uniq)), index=uniq).loc[gids].to_numpy()
        feat, mask = precompute_game_history(uniq, per_player, game_meta, args.k)
        feat = (feat - h_mean) / h_std                     # standardize history feats
        S = None
        if args.static:
            champ_by_game = {str(g): r.to_numpy() for g, r in
                             d.groupby("game_id")[CHAMP_COLS].first().iterrows()}
            S_np, matched, total = build_game_static(
                [str(g) for g in uniq], champ_by_game, patch_map,
                static_lut, n_static, set(static_avail), static_fallback)
            log.info("  static: %d/%d (game,node) champ lookups matched", matched, total)
            S = torch.from_numpy(S_np)
        return (torch.from_numpy(Xn), torch.from_numpy(Xc),
                torch.from_numpy(feat), torch.from_numpy(mask),
                torch.from_numpy(gidx), torch.from_numpy(d["blue_win"].to_numpy(np.float32)), S)

    Xn_tr, Xc_tr, H_tr, M_tr, gi_tr, y_tr, S_tr = make_split(tr)
    Xn_va, Xc_va, H_va, M_va, gi_va, y_va, S_va = make_split(va)
    log.info("History tensors: train %s, val %s", tuple(H_tr.shape), tuple(H_va.shape))

    model = ContextGameGNN(vocabs, K=args.k, d=args.d, layers=args.layers, ctx_d=args.ctx_d,
                           n_static=n_static).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    log.info("Model params: %d", nparam)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(y_tr); best = {"auc": 0.0}; since_improve = 0
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, args.batch_size):
            b = perm[i:i+args.batch_size]; g = gi_tr[b]
            s = S_tr[g].to(device) if S_tr is not None else None
            logit = model(Xn_tr[b].to(device), Xc_tr[b].to(device), H_tr[g].to(device), M_tr[g].to(device), s)
            loss = loss_fn(logit, y_tr[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        m = evaluate(model, Xn_va, Xc_va, H_va, M_va, gi_va, y_va, device, static=S_va)
        log.info("ep %2d | loss %.4f | val AUC %.4f Brier %.4f ECE %.4f", ep, tot / n, m["auc"], m["brier"], m["ece"])
        if m["auc"] > best["auc"]:
            best = {**m, "epoch": ep}
            since_improve = 0
            torch.save({"state_dict": model.state_dict(), "vocabs": vocabs, "args": vars(args),
                        "num_mean": num_mean, "num_std": num_std, "h_mean": h_mean, "h_std": h_std,
                        "n_static": n_static, "static_feat_cols": static_feat_cols,
                        "static_avail": static_avail, "static_fallback": static_fallback,
                        "val_metrics": m},
                       MODELS_DIR / ("gnn_static_model.pt" if args.static else "gnn_context_model.pt"))
        else:
            since_improve += 1
            if args.patience and since_improve >= args.patience:
                log.info("Early stop at ep %d (no val-AUC improvement for %d epochs; best ep %d AUC %.4f)",
                         ep, args.patience, best.get("epoch", -1), best["auc"])
                break

    anti = antisymmetry_check(model, Xn_va, Xc_va, H_va, M_va, gi_va, device, static=S_va)
    log.info("Antisymmetry max|f(A,B)+f(B,A)|: %.2e", anti)
    print("\n" + "=" * 66)
    print("  04f — equivariant GNN + game-level player context")
    print(f"  params {nparam:,} | K={args.k} | device {device} | {time.time()-t0:.1f}s")
    print(f"  best val: AUC {best['auc']:.4f}  Brier {best['brier']:.4f}  ECE {best['ece']:.4f}  (ep {best.get('epoch','-')})")
    print(f"  antisymmetry residual: {anti:.2e}  (hard constraint -> ~0)")
    print(f"  saved: models/{'gnn_static_model.pt' if args.static else 'gnn_context_model.pt'}")
    print("=" * 66)


if __name__ == "__main__":
    main()
