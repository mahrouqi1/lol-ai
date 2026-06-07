"""
04c_train_player_context.py
============================
Transformer model that injects per-player game-history context into the
per-minute game sequence, then predicts win probability.

Architecture
------------

  Per-player history encoder  (one per player in the current game)
  ──────────────────────────
    Input : last K games for this player (end-of-game summary stats),
            ordered chronologically, from *before* the current game.
    Model : small non-causal Transformer  (all history is already past)
            → mean-pool over K tokens
            → player_embedding  (dim = PLAYER_D)

  Game-level predictor
  ─────────────────────
    Input : [player_token_0, ..., player_token_9,  # 10 context tokens
              game_frame_0, ..., game_frame_T-1]   # causal minute tokens

    Mask  : hybrid causal mask
              - player tokens: see all other player tokens (fully visible)
              - game  tokens : see all player tokens + causally past game tokens

    Output: win_prob at each game-frame timestep (same as 04b)

Data requirements
-----------------
  data/processed/features.parquet          — per-minute game features (from 03)
  data/processed/player_game_summary.parquet — player history index (from 03b)

Outputs
-------
  models/player_context_model.pt           — best checkpoint
  models/player_context_artifacts.pkl      — scalers, encoders, config
  logs/04c_train_player_context.log

Usage
-----
    # Quick test  (~5 min on CPU, 10K games, K=10)
    python src/04c_train_player_context.py --limit 10000 --epochs 5 --k 10

    # Full run
    python src/04c_train_player_context.py

    # GPU workstation (fast)
    python src/04c_train_player_context.py --epochs 30
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = Path(os.environ.get("LOL_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
FEATURES_PATH  = DATA_DIR / "features.parquet"
HISTORY_PATH   = DATA_DIR / "player_game_summary.parquet"
MODELS_DIR     = PROJECT_ROOT / "models"
LOG_DIR        = PROJECT_ROOT / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT     = MODELS_DIR / "player_context_model.pt"
ARTIFACT_OUT  = MODELS_DIR / "player_context_artifacts.pkl"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "04c_train_player_context.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SLOTS = [
    "blue_top", "blue_jungle", "blue_middle", "blue_bottom", "blue_utility",
    "red_top",  "red_jungle",  "red_middle",  "red_bottom",  "red_utility",
]

# End-of-game features used in player history encoder
# These are the per-player columns in player_game_summary.parquet
HISTORY_CONT_COLS = [
    "total_gold", "xp", "cs_total", "dmg_to_champs", "dmg_taken",
    "kills", "deaths", "assists", "time_cc_others",
]
HISTORY_FEAT_DIM = len(HISTORY_CONT_COLS) + 2  # + level (int) + player_won (float)

METADATA_COLS = {"game_id", "minute", "blue_win", "game_duration_min"}
CAT_SUFFIXES  = ["champion_id", "summoner1_id", "summoner2_id", "keystone", "primary_tree"]

N_THREADS = max(1, (os.cpu_count() or 4) // 2)


# ── Model ─────────────────────────────────────────────────────────────────────

class MinutePositionalEncoding(nn.Module):
    """Sinusoidal encoding indexed by actual game minute (not sequence position)."""

    def __init__(self, d_model: int, max_minute: int = 120):
        super().__init__()
        pe  = torch.zeros(max_minute + 1, d_model)
        pos = torch.arange(0, max_minute + 1, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, minutes: torch.Tensor) -> torch.Tensor:
        minutes = minutes.clamp(0, self.pe.shape[0] - 1)
        return x + self.pe[minutes]


class PlayerHistoryEncoder(nn.Module):
    """
    Encodes a player's K most recent games into a single embedding vector.

    Input : (batch, K, history_feat_dim)
    Output: (batch, player_d)
    """

    def __init__(
        self,
        input_dim:  int,
        player_d:   int = 64,
        nhead:      int = 2,
        num_layers: int = 2,
        ffn_dim:    int = 128,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, player_d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=player_d, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(player_d)

    def forward(
        self,
        hist: torch.Tensor,           # (B, K, input_dim) — padded to K
        hist_mask: torch.Tensor,      # (B, K) — True where padding
    ) -> torch.Tensor:                # (B, player_d)
        # Ensure at least one unmasked position per row so softmax never sees all-inf.
        # Rows that are fully masked have no real history; their pooled output = 0.
        all_masked = hist_mask.all(dim=1)          # (B,)
        safe_mask  = hist_mask.clone()
        safe_mask[all_masked, 0] = False           # expose dummy first slot

        x = self.input_proj(hist)                  # (B, K, player_d)
        x = self.transformer(x, src_key_padding_mask=safe_mask)
        # Mean-pool using the ORIGINAL mask (fully-masked rows -> zero embedding)
        real  = (~hist_mask).float().unsqueeze(-1)   # (B, K, 1)
        denom = real.sum(dim=1).clamp(min=1.0)
        x     = (x * real).sum(dim=1) / denom       # (B, player_d)
        return self.norm(x)


class PlayerContextTransformer(nn.Module):
    """
    Two-stage model:
      1. PlayerHistoryEncoder  (one per player, weights shared across players)
         → 10 player embeddings → 10 player tokens
      2. Game Transformer (causal) with 10 player tokens prepended
         → win prob at each game minute

    Hybrid causal mask:
      - player tokens (positions 0-9) are fully visible to each other
      - game tokens (positions 10+) can see all player tokens + causally past
        game tokens, but NOT future game tokens or padding
    """

    def __init__(
        self,
        game_feat_dim:  int,
        history_feat_dim: int,
        n_players:      int   = 10,
        player_d:       int   = 64,
        game_d:         int   = 128,
        nhead:          int   = 4,
        num_layers:     int   = 4,
        ffn_dim:        int   = 512,
        dropout:        float = 0.1,
        max_minute:     int   = 120,
        cat_vocab:      dict  | None = None,  # {suffix: vocab_size}
        cat_embed_dim:  int   = 8,
    ):
        super().__init__()
        self.n_players     = n_players
        self.player_d      = player_d
        self.game_d        = game_d

        # ── Player history encoder (shared weights) ────────────────────────────
        self.history_encoder = PlayerHistoryEncoder(
            input_dim=history_feat_dim,
            player_d=player_d,
            nhead=2,
            num_layers=2,
            ffn_dim=player_d * 2,
            dropout=dropout,
        )
        # Project player embedding to game_d for injection
        self.player_proj = nn.Linear(player_d, game_d)

        # ── Categorical embeddings for game features ──────────────────────────
        cat_vocab = cat_vocab or {}
        self.cat_embeds = nn.ModuleDict({
            suf: nn.Embedding(vocab + 1, cat_embed_dim, padding_idx=0)
            for suf, vocab in cat_vocab.items()
        })
        # game_feat_dim = len(cont_cols) (continuous features only)
        # categorical slots: n_cat_types * n_players * cat_embed_dim
        n_cat_types    = len(cat_vocab)
        game_input_dim = game_feat_dim + n_cat_types * n_players * cat_embed_dim

        # ── Game feature projection ───────────────────────────────────────────
        self.game_proj = nn.Linear(game_input_dim, game_d)
        self.pos_enc   = MinutePositionalEncoding(game_d, max_minute)

        # ── Unified Transformer ────────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=game_d, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # ── Output head ───────────────────────────────────────────────────────
        self.output_head = nn.Linear(game_d, 1)

    @staticmethod
    def _hybrid_mask(
        n_player: int,
        n_game:   int,
        device:   torch.device,
    ) -> torch.Tensor:
        """
        Build the (n_player + n_game) x (n_player + n_game) attention mask.
        -inf = blocked, 0 = allowed.

        player tokens (rows 0..n_player-1):
          - can attend to all player tokens (columns 0..n_player-1)
          - cannot attend to game tokens (columns n_player..)
        game tokens (rows n_player..):
          - can attend to all player tokens (columns 0..n_player-1)
          - can attend causally to game tokens (column <= own row-offset)
          - blocked from future game tokens
        """
        total = n_player + n_game
        mask  = torch.zeros(total, total, device=device)

        # Block player rows from attending to game columns
        mask[:n_player, n_player:] = float("-inf")

        # Block game rows from attending to future game columns (causal)
        for i in range(n_game):
            mask[n_player + i, n_player + i + 1:] = float("-inf")

        return mask

    def _embed_game_features(
        self,
        x_cont: torch.Tensor,    # (B, T, cont_cols)
        x_cat:  dict[str, torch.Tensor],  # {suf: (B, T, n_players)}
    ) -> torch.Tensor:           # (B, T, game_input_dim)
        parts = [x_cont]
        for suf, embed_layer in self.cat_embeds.items():
            ids  = x_cat[suf]                        # (B, T, n_players)
            emb  = embed_layer(ids)                  # (B, T, n_players, embed_dim)
            B, T, NP, D = emb.shape
            parts.append(emb.reshape(B, T, NP * D))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x_cont:    torch.Tensor,  # (B, T, n_cont)
        x_cat:     dict[str, torch.Tensor],  # {suf: (B, T, n_players)}
        minutes:   torch.Tensor,  # (B, T) int
        lengths:   torch.Tensor,  # (B,)  int — real game frames (no padding)
        hist_feats: torch.Tensor, # (B, n_players, K, history_feat_dim)
        hist_mask:  torch.Tensor, # (B, n_players, K) — True=padding
    ) -> torch.Tensor:            # (B, T) win prob per timestep
        B, T, _  = x_cont.shape
        device   = x_cont.device

        # ── Encode player histories ────────────────────────────────────────────
        # Flatten batch × player for encoder
        hf = hist_feats.reshape(B * self.n_players, hist_feats.shape[2], -1)
        hm = hist_mask.reshape(B * self.n_players, hist_mask.shape[2])
        player_emb = self.history_encoder(hf, hm)             # (B*NP, player_d)
        player_emb = player_emb.reshape(B, self.n_players, -1)  # (B, NP, player_d)
        player_tok = self.player_proj(player_emb)              # (B, NP, game_d)

        # ── Embed game features ────────────────────────────────────────────────
        game_emb = self._embed_game_features(x_cont, x_cat)  # (B, T, game_input_dim)
        game_tok = self.game_proj(game_emb)                   # (B, T, game_d)
        game_tok = self.pos_enc(game_tok, minutes)

        # ── Concatenate: [player_tokens | game_tokens] ────────────────────────
        seq = torch.cat([player_tok, game_tok], dim=1)        # (B, NP+T, game_d)

        # ── Hybrid attention mask ─────────────────────────────────────────────
        attn_mask = self._hybrid_mask(self.n_players, T, device)

        x = self.transformer(seq, mask=attn_mask)             # (B, NP+T, game_d)

        # Take only game-frame outputs
        x_game = x[:, self.n_players:, :]                    # (B, T, game_d)

        # Mask out padding positions
        real = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        probs = torch.sigmoid(self.output_head(x_game).squeeze(-1))
        return probs * real.float()


# ── Data loading & preparation ────────────────────────────────────────────────

def load_and_split(features_path: Path, limit: int) -> tuple[pd.DataFrame, list, list, list]:
    log.info("Loading %s ...", features_path)
    df = pd.read_parquet(features_path)
    log.info("Loaded: %d rows x %d cols", len(df), df.shape[1])

    all_games = df["game_id"].unique()
    if limit:
        rng = np.random.default_rng(42)
        all_games = rng.choice(all_games, size=min(limit, len(all_games)), replace=False)
        df = df[df["game_id"].isin(all_games)]
        log.info("Limited to %d games -> %d rows", len(all_games), len(df))

    rng = np.random.default_rng(42)
    rng.shuffle(all_games)
    n = len(all_games)
    n_tr  = int(n * 0.80)
    n_val = int(n * 0.10)
    train_games = set(all_games[:n_tr])
    val_games   = set(all_games[n_tr: n_tr + n_val])
    test_games  = set(all_games[n_tr + n_val:])

    return df, train_games, val_games, test_games


def build_cat_vocab(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, int]:
    """Return {suffix: max_value} for each categorical suffix."""
    vocab: dict[str, int] = {}
    for suf in CAT_SUFFIXES:
        cols = [c for c in feature_cols if c.endswith(f"_{suf}")]
        if not cols:
            continue
        mx = 0
        for col in cols:
            mx = max(mx, int(df[col].max()))
        vocab[suf] = mx
    return vocab


def split_cont_cat(
    feature_cols: list[str],
    cat_vocab: dict[str, int],
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Separate feature_cols into:
      cont_cols  — float columns passed directly
      cat_groups — {suffix: [col_name, ...]} for categorical columns
    """
    cat_set = set()
    cat_groups: dict[str, list[str]] = {}
    for suf in cat_vocab:
        grp = [c for c in feature_cols if c.endswith(f"_{suf}")]
        if grp:
            cat_groups[suf] = grp
            cat_set.update(grp)
    cont_cols = [c for c in feature_cols if c not in cat_set]
    return cont_cols, cat_groups


def build_player_lookup(
    history_df: pd.DataFrame,
    hist_cols: list[str],
) -> dict[str, tuple]:
    """
    Build a dict: puuid -> (gc_ms_array, feat_array)
      gc_ms_array : int64 array of game_creation_ms, sorted ascending
      feat_array  : float32 array shape (n_games, len(hist_cols))

    Pre-extracting numpy arrays avoids slow DataFrame.loc lookups in the hot path.
    """
    # Sort by puuid then time — groupby preserves order within group
    sorted_df = history_df.sort_values(["puuid", "game_creation_ms"])
    feats  = sorted_df[hist_cols].values.astype(np.float32)
    gc_arr = sorted_df["game_creation_ms"].values.astype(np.int64)
    puuids = sorted_df["puuid"].values

    lookup: dict[str, tuple] = {}
    starts: dict[str, int]   = {}
    prev   = None
    for i, p in enumerate(puuids):
        if p != prev:
            if prev is not None:
                end = i
                lookup[prev] = (
                    gc_arr[starts[prev]: end],
                    feats[starts[prev]: end],
                )
            starts[p] = i
            prev = p
    if prev is not None:
        lookup[prev] = (gc_arr[starts[prev]:], feats[starts[prev]:])

    return lookup


def get_player_history(
    puuid: str,
    game_creation_ms: int,
    lookup: dict[str, tuple],
    history_df,  # unused — kept for API compatibility
    hist_cols: list[str],
    K: int,
) -> np.ndarray:
    """
    Return the last K games for `puuid` that occurred BEFORE `game_creation_ms`.
    Shape: (K, len(hist_cols)) — zero-padded at the beginning if < K games available.
    """
    import bisect as _bisect

    entry = lookup.get(puuid)
    result = np.zeros((K, len(hist_cols)), dtype=np.float32)
    if entry is None:
        return result

    gc_ms_arr, feat_arr = entry
    lo = int(np.searchsorted(gc_ms_arr, game_creation_ms, side="left"))
    if lo == 0:
        return result

    start   = max(0, lo - K)
    n_real  = lo - start
    offset  = K - n_real
    result[offset:] = feat_arr[start: lo]
    return result


# ── Dataset ───────────────────────────────────────────────────────────────────

class GameDataset(torch.utils.data.Dataset):
    """
    One sample = one game (variable-length sequence of minutes).
    Returns:
        x_cont    (T, n_cont)
        x_cat     {suf: (T, n_players)}
        minutes   (T,) int
        length    int
        label     (T,) float  (same blue_win value for all timesteps)
        hist_feats (n_players, K, hist_dim)
        hist_mask  (n_players, K) bool  True=padding
    """

    def __init__(
        self,
        game_ids:    list,
        df:          pd.DataFrame,
        cont_cols:   list[str],
        cat_groups:  dict[str, list[str]],
        cont_scaler: StandardScaler,
        cat_vocab:   dict[str, int],
        hist_lookup: dict[str, tuple],  # puuid -> (gc_ms_arr, feat_arr)
        hist_cols:   list[str],
        K:           int,
        slot_puuids: dict,   # game_id -> {slot: puuid}
        game_gc_ms:  dict,   # game_id -> game_creation_ms (int)
    ):
        self.game_ids    = list(game_ids)
        self.df          = df.set_index("game_id") if "game_id" in df.columns else df
        self.cont_cols   = cont_cols
        self.cat_groups  = cat_groups
        self.cont_scaler = cont_scaler
        self.cat_vocab   = cat_vocab
        self.hist_lookup = hist_lookup
        self.hist_cols   = hist_cols
        self.K           = K
        self.slot_puuids = slot_puuids
        self.game_gc_ms  = game_gc_ms
        self.n_players   = len(SLOTS)

    def __len__(self) -> int:
        return len(self.game_ids)

    def __getitem__(self, idx: int) -> dict:
        gid = self.game_ids[idx]
        rows = self.df.loc[gid]
        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T
        rows = rows.sort_values("minute")

        T      = len(rows)
        label  = float(rows["blue_win"].iloc[0])
        mins   = rows["minute"].values.astype(np.int32)

        # Continuous features
        x_cont = self.cont_scaler.transform(rows[self.cont_cols].values.astype(np.float32))

        # Categorical features: {suf: (T, n_players)} as int32
        x_cat: dict[str, np.ndarray] = {}
        for suf, cols in self.cat_groups.items():
            arr = rows[cols].values.astype(np.int32)  # (T, n_players)
            # Clip to vocab range
            arr = np.clip(arr, 0, self.cat_vocab[suf])
            x_cat[suf] = arr

        # ── Player history ────────────────────────────────────────────────────
        gc_ms = self.game_gc_ms.get(gid, 0)

        puuids_for_game = self.slot_puuids.get(gid, {})

        hist_feats = np.zeros(
            (self.n_players, self.K, len(self.hist_cols)), dtype=np.float32
        )
        hist_mask  = np.ones((self.n_players, self.K), dtype=bool)  # True = padding

        for si, slot in enumerate(SLOTS):
            puuid = puuids_for_game.get(slot, "")
            if puuid:
                hf = get_player_history(
                    puuid, gc_ms, self.hist_lookup,
                    None, self.hist_cols, self.K,
                )
                hist_feats[si] = hf
                # Mask is True where we have zeros (no history) at the front
                n_real = int(np.any(hf != 0, axis=1).sum())
                if n_real > 0:
                    hist_mask[si, : self.K - n_real] = True
                    hist_mask[si, self.K - n_real:]  = False

        return {
            "x_cont":     torch.from_numpy(x_cont),
            "x_cat":      {s: torch.from_numpy(v) for s, v in x_cat.items()},
            "minutes":    torch.from_numpy(mins),
            "length":     T,
            "label":      label,
            "hist_feats": torch.from_numpy(hist_feats),
            "hist_mask":  torch.from_numpy(hist_mask),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length sequences to the longest in the batch."""
    max_T = max(b["length"] for b in batch)
    B = len(batch)
    n_cont = batch[0]["x_cont"].shape[1]
    n_players = batch[0]["hist_feats"].shape[0]
    K        = batch[0]["hist_feats"].shape[1]
    hist_dim = batch[0]["hist_feats"].shape[2]

    x_cont_pad    = torch.zeros(B, max_T, n_cont)
    minutes_pad   = torch.zeros(B, max_T, dtype=torch.long)
    lengths       = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    labels        = torch.zeros(B, max_T)
    hist_feats_b  = torch.zeros(B, n_players, K, hist_dim)
    hist_mask_b   = torch.ones(B, n_players, K, dtype=torch.bool)

    cat_suf = list(batch[0]["x_cat"].keys())
    n_pl_cat = {s: batch[0]["x_cat"][s].shape[1] for s in cat_suf}
    x_cat_pad = {s: torch.zeros(B, max_T, n_pl_cat[s], dtype=torch.long) for s in cat_suf}

    for i, b in enumerate(batch):
        T = b["length"]
        x_cont_pad[i, :T]  = b["x_cont"]
        minutes_pad[i, :T] = b["minutes"]
        labels[i, :T]      = b["label"]
        hist_feats_b[i]    = b["hist_feats"]
        hist_mask_b[i]     = b["hist_mask"]
        for s in cat_suf:
            x_cat_pad[s][i, :T] = b["x_cat"][s]

    return {
        "x_cont":     x_cont_pad,
        "x_cat":      x_cat_pad,
        "minutes":    minutes_pad,
        "lengths":    lengths,
        "labels":     labels,
        "hist_feats": hist_feats_b,
        "hist_mask":  hist_mask_b,
    }


# ── Training helpers ──────────────────────────────────────────────────────────

def eval_auc(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    minute_cutoffs: dict[str, int],  # {"5m": 5, "10m": 10, "end": 9999}
) -> dict[str, float]:
    model.eval()
    all_preds: dict[str, list] = {k: [] for k in minute_cutoffs}
    all_labels: dict[str, list] = {k: [] for k in minute_cutoffs}

    with torch.no_grad():
        for batch in loader:
            x_cont    = batch["x_cont"].to(device)
            x_cat     = {s: v.to(device) for s, v in batch["x_cat"].items()}
            minutes   = batch["minutes"].to(device)
            lengths   = batch["lengths"].to(device)
            labels    = batch["labels"].to(device)
            hist_feats = batch["hist_feats"].to(device)
            hist_mask  = batch["hist_mask"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                probs = model(x_cont, x_cat, minutes, lengths, hist_feats, hist_mask)
            probs = probs.float().nan_to_num(nan=0.5)

            B, T = probs.shape
            for name, cutoff in minute_cutoffs.items():
                for b in range(B):
                    L = lengths[b].item()
                    mins_b = minutes[b, :L].cpu().numpy()
                    probs_b = probs[b, :L].cpu().float().numpy()
                    lab = labels[b, 0].item()

                    if cutoff >= 9999:
                        # last real timestep
                        all_preds[name].append(float(probs_b[-1]))
                    else:
                        # first timestep at or before cutoff
                        idxs = np.where(mins_b <= cutoff)[0]
                        if len(idxs) == 0:
                            continue
                        all_preds[name].append(float(probs_b[idxs[-1]]))
                    all_labels[name].append(float(lab))

    aucs = {}
    for name in minute_cutoffs:
        if len(set(all_labels[name])) < 2:
            aucs[name] = float("nan")
        else:
            aucs[name] = roc_auc_score(all_labels[name], all_preds[name])
    return aucs


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--limit",       type=int,   default=0,    help="Max games (0=all)")
    p.add_argument("--epochs",      type=int,   default=20,   help="Max training epochs")
    p.add_argument("--patience",    type=int,   default=5,    help="Early stop patience")
    p.add_argument("--batch",       type=int,   default=32,   help="Batch size (games)")
    p.add_argument("--lr",          type=float, default=3e-4, help="Learning rate")
    p.add_argument("--k",           type=int,   default=20,   help="History games per player")
    p.add_argument("--player-d",    type=int,   default=64,   help="Player embedding dim")
    p.add_argument("--game-d",      type=int,   default=128,  help="Game Transformer d_model")
    p.add_argument("--nhead",       type=int,   default=4,    help="Attention heads (game Transformer)")
    p.add_argument("--layers",      type=int,   default=4,    help="Game Transformer layers")
    p.add_argument("--ffn",         type=int,   default=512,  help="FFN dim")
    p.add_argument("--dropout",     type=float, default=0.1,  help="Dropout")
    p.add_argument("--label-smooth",type=float, default=0.05, help="Label smoothing")
    p.add_argument("--workers",     type=int,   default=min(4, N_THREADS), help="DataLoader workers")
    p.add_argument("--no-compile",  action="store_true", help="Disable torch.compile()")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.set_num_threads(N_THREADS)
    t0 = time.time()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    log.info("Device: %s | AMP: %s", device, use_amp)

    # ── Load features ─────────────────────────────────────────────────────────
    df, train_games, val_games, test_games = load_and_split(FEATURES_PATH, args.limit)

    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    cat_vocab    = build_cat_vocab(df, feature_cols)
    cont_cols, cat_groups = split_cont_cat(feature_cols, cat_vocab)

    log.info("Feature cols: %d cont + %d cat groups", len(cont_cols), len(cat_groups))

    # Fit scaler on training data
    log.info("Fitting StandardScaler on train rows ...")
    train_mask   = df["game_id"].isin(train_games)
    cont_scaler  = StandardScaler()
    cont_scaler.fit(df.loc[train_mask, cont_cols].values.astype(np.float32))

    # ── Load player history index ─────────────────────────────────────────────
    if not HISTORY_PATH.exists():
        log.error("Player index not found: %s — run 03b_build_player_index.py first", HISTORY_PATH)
        sys.exit(1)

    log.info("Loading player history index from %s ...", HISTORY_PATH)
    history_df = pd.read_parquet(HISTORY_PATH)
    log.info("History rows: %d  unique players: %d", len(history_df), history_df["puuid"].nunique())

    # History columns: cont features + player_won + level
    hist_cols = HISTORY_CONT_COLS + ["level", "player_won"]

    # Build PUUID lookup
    log.info("Building player lookup index ...")
    hist_lookup = build_player_lookup(history_df, hist_cols)

    # Build game_id -> {slot: puuid} mapping
    log.info("Building game->slot->puuid map ...")
    # We need the raw match data for slot ordering; use features.parquet column names
    # to figure out which PUUID belongs to which slot by cross-referencing history_df
    # (history_df has team_position + team_id)
    slot_puuids: dict[str, dict[str, str]] = {}

    # Build from history_df: for each game, map (team_position, team_id) -> puuid
    _pos_map = {
        # team 100 (blue) positions
        (100, "TOP"):     "blue_top",
        (100, "JUNGLE"):  "blue_jungle",
        (100, "MIDDLE"):  "blue_middle",
        (100, "BOTTOM"):  "blue_bottom",
        (100, "UTILITY"): "blue_utility",
        # team 200 (red) positions
        (200, "TOP"):     "red_top",
        (200, "JUNGLE"):  "red_jungle",
        (200, "MIDDLE"):  "red_middle",
        (200, "BOTTOM"):  "red_bottom",
        (200, "UTILITY"): "red_utility",
    }
    # Build in one vectorized pass over history_df
    game_gc_ms: dict[str, int] = {}
    for row in history_df.itertuples(index=False):
        gid  = row.game_id
        slot = _pos_map.get((int(row.team_id), str(row.team_position).upper()), None)
        if gid not in game_gc_ms:
            game_gc_ms[gid] = int(row.game_creation_ms)
        if slot is None:
            continue
        if gid not in slot_puuids:
            slot_puuids[gid] = {}
        slot_puuids[gid][slot] = row.puuid

    # ── Build datasets ────────────────────────────────────────────────────────
    common_kw = dict(
        df=df,
        cont_cols=cont_cols,
        cat_groups=cat_groups,
        cont_scaler=cont_scaler,
        cat_vocab=cat_vocab,
        hist_lookup=hist_lookup,
        hist_cols=hist_cols,
        K=args.k,
        slot_puuids=slot_puuids,
        game_gc_ms=game_gc_ms,
    )

    log.info("Building datasets: train=%d val=%d test=%d games",
             len(train_games), len(val_games), len(test_games))

    train_ds = GameDataset(game_ids=list(train_games), **common_kw)
    val_ds   = GameDataset(game_ids=list(val_games),   **common_kw)

    dl_kw = dict(
        batch_size=args.batch, collate_fn=collate_fn,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True,  **dl_kw)
    val_dl   = torch.utils.data.DataLoader(val_ds,   shuffle=False, **dl_kw)

    # ── Instantiate model ─────────────────────────────────────────────────────
    # Compute game_feat_dim as seen by the model
    n_cat_total = sum(len(cols) for cols in cat_groups.values())
    game_feat_dim_equiv = len(cont_cols) + n_cat_total  # raw cols before embed

    model = PlayerContextTransformer(
        game_feat_dim    = len(cont_cols),   # cont only (cats handled via embed)
        history_feat_dim = len(hist_cols),
        n_players        = len(SLOTS),
        player_d         = args.player_d,
        game_d           = args.game_d,
        nhead            = args.nhead,
        num_layers       = args.layers,
        ffn_dim          = args.ffn,
        dropout          = args.dropout,
        cat_vocab        = cat_vocab,
        cat_embed_dim    = 8,
    ).to(device)

    if device.type == "cuda" and not args.no_compile:
        try:
            model = torch.compile(model)
            log.info("torch.compile() applied.")
        except Exception:
            pass

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model params: %s  player_d=%d game_d=%d layers=%d",
             f"{n_params:,}", args.player_d, args.game_d, args.layers)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    minute_cutoffs = {"5m": 5, "10m": 10, "end": 9999}

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info("Starting training: max %d epochs, patience %d", args.epochs, args.patience)
    log.info("Log: loss=train/val | AUC@5m=tr/val  AUC@10m=tr/val  AUC@end=tr/val")

    best_val_auc = 0.0
    best_state   = None
    patience_cnt = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_t = time.time()
        total_loss = 0.0
        n_batches  = 0

        for batch in train_dl:
            x_cont     = batch["x_cont"].to(device)
            x_cat      = {s: v.to(device) for s, v in batch["x_cat"].items()}
            minutes    = batch["minutes"].to(device)
            lengths    = batch["lengths"].to(device)
            labels     = batch["labels"].to(device)
            hist_feats = batch["hist_feats"].to(device)
            hist_mask  = batch["hist_mask"].to(device)

            B, T = labels.shape
            real_mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)

            # Label smoothing
            smooth = labels.clone()
            smooth[real_mask & (labels == 1)] = 1.0 - args.label_smooth
            smooth[real_mask & (labels == 0)] = args.label_smooth

            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                probs = model(x_cont, x_cat, minutes, lengths, hist_feats, hist_mask)
            probs = probs.float().nan_to_num(nan=0.5)
            loss  = F.binary_cross_entropy(
                probs[real_mask].clamp(1e-6, 1 - 1e-6), smooth[real_mask].float()
            )

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        train_loss = total_loss / max(n_batches, 1)

        # Validation
        val_aucs  = eval_auc(model, val_dl, device, use_amp, minute_cutoffs)
        # Train AUC quick estimate (re-use last batch is not ideal; do full pass)
        train_aucs = eval_auc(model, train_dl, device, use_amp, minute_cutoffs)

        ep_s = time.time() - ep_t
        log.info(
            "Epoch %02d/%02d | loss=%.4f | "
            "AUC@5m=%.3f/%.3f  AUC@10m=%.3f/%.3f  AUC@end=%.3f/%.3f | %.1fs",
            epoch, args.epochs, train_loss,
            train_aucs["5m"], val_aucs["5m"],
            train_aucs["10m"], val_aucs["10m"],
            train_aucs["end"], val_aucs["end"],
            ep_s,
        )

        # Track best by val AUC@end
        if val_aucs["end"] > best_val_auc:
            best_val_auc = val_aucs["end"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                log.info("Early stopping at epoch %d.", epoch)
                break

    log.info("Restoring best weights (val AUC@end=%.4f).", best_val_auc)
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_OUT)
    log.info("Saved model: %s", MODEL_OUT)

    artifacts = {
        "cont_scaler":  cont_scaler,
        "cat_vocab":    cat_vocab,
        "cont_cols":    cont_cols,
        "cat_groups":   cat_groups,
        "hist_cols":    hist_cols,
        "K":            args.k,
        "player_d":     args.player_d,
        "game_d":       args.game_d,
        "nhead":        args.nhead,
        "layers":       args.layers,
        "ffn":          args.ffn,
        "dropout":      args.dropout,
        "best_val_auc": best_val_auc,
    }
    with open(ARTIFACT_OUT, "wb") as f:
        pickle.dump(artifacts, f)
    log.info("Saved artifacts: %s", ARTIFACT_OUT)

    elapsed = time.time() - t0
    log.info("Done in %.1f s (%.1f min)", elapsed, elapsed / 60)

    print("\n" + "=" * 55)
    print(f"  Best val AUC@end: {best_val_auc:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
