"""
04b_train_timeseries.py
=======================
Causal Transformer time-series model: predict blue_win at every minute T
using the full sequence of per-minute snapshots from minute 1 to T.

Architecture
------------
  Shared embedding tables per categorical type (champion, summoner spells, runes)
  → concat with standardised continuous features
  → linear projection to d_model (with LayerNorm + Dropout)
  → sinusoidal positional encoding indexed by actual game minute
  → N-layer causal TransformerEncoder (pre-norm, batch_first)
  → linear output head → sigmoid per timestep

Causality is guaranteed by an upper-triangular attention mask: prediction at
minute T only uses minutes 1..T.  No padding tokens can leak into real tokens
because our sequences are left-aligned and the causal mask prevents look-ahead.

GPU / CPU
---------
  Device is selected automatically: CUDA > CPU.
  Mixed-precision (AMP) is enabled automatically on CUDA for ~2× speed-up.
  DataLoader uses pin_memory on CUDA.
  torch.compile() is applied on CUDA when PyTorch >= 2.0 (speeds up ~20-40%).

  Recommended GPU config  : d_model=256, nhead=8, num_layers=6, ffn_dim=1024
  Default (CPU-safe) config: d_model=128, nhead=4, num_layers=4, ffn_dim=512

Split
-----
  80 / 10 / 10  train / val / test at the GAME level.
  All minutes of a game are always in the same split.

Outputs
-------
  models/transformer_timeseries.pt      — model weights (state_dict)
  models/transformer_artifacts.pkl      — encoders, scaler, feature indices, config
  reports/transformer_auc_by_minute.png
  reports/transformer_calibration.png
  reports/transformer_train_history.png
  reports/transformer_vs_snapshot_auc.png  (if snapshot CV results exist)
  reports/transformer_win_probability_trajectory.png

SHAP note
---------
  TreeSHAP does not apply here.  For attribution use shap.GradientExplainer
  or shap.DeepExplainer on this PyTorch model.  The LightGBM snapshot model
  (04a) remains the primary SHAP vehicle.

Usage
-----
    conda activate lol_shap_env
    python src/04b_train_timeseries.py
    python src/04b_train_timeseries.py --epochs 5 --limit 2000     # quick CPU test
    python src/04b_train_timeseries.py --d-model 256 --num-layers 6 --nhead 8  # GPU
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = Path(os.environ.get("LOL_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
FEATURES_PATH = DATA_DIR / "features.parquet"
MODELS_DIR    = Path(os.environ.get("LOL_MODELS_DIR", PROJECT_ROOT / "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR   = PROJECT_ROOT / "reports"
LOG_DIR       = PROJECT_ROOT / "logs"
SNAPSHOT_CV   = MODELS_DIR / "lgbm_snapshot_cv.pkl"

for _d in (MODELS_DIR, REPORTS_DIR, LOG_DIR):
    _d.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "04b_train_timeseries.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SLOTS = [
    "blue_top", "blue_jungle", "blue_middle", "blue_bottom", "blue_utility",
    "red_top",  "red_jungle",  "red_middle",  "red_bottom",  "red_utility",
]
CAT_SUFFIXES = ["champion_id", "summoner1_id", "summoner2_id", "keystone", "primary_tree"]

# Shared embedding dim per categorical type (10 slots share one table per type)
EMBED_DIM: dict[str, int] = {
    "champion_id":  16,
    "summoner1_id":  8,
    "summoner2_id":  8,
    "keystone":      8,
    "primary_tree":  4,
}

METADATA_COLS = {"game_id", "minute", "blue_win", "game_duration_min"}

# Default model hyperparameters (CPU-safe).
# For GPU runs override via CLI: --d-model 256 --nhead 8 --num-layers 6
D_MODEL    = 128
NHEAD      = 4
NUM_LAYERS = 4
FFN_DIM    = 512
DROPOUT    = 0.1

LABEL_SMOOTHING = 0.05

BATCH_SIZE   = 32
MAX_EPOCHS   = 50
PATIENCE     = 7
LR           = 1e-3
WEIGHT_DECAY = 1e-4

# Leave 2 cores free so the machine stays responsive
N_THREADS = max(1, (os.cpu_count() or 4) // 2)

MINUTE_BUCKETS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 999)]
BUCKET_LABELS  = ["0-5", "5-10", "10-15", "15-20", "20-25", "25+"]

# ── Dataset ───────────────────────────────────────────────────────────────────

class GameSequenceDataset(Dataset):
    """One item = one game (variable-length sequence)."""

    def __init__(self, games: list[dict]):
        self.games = games

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, idx: int) -> dict:
        return self.games[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Pad sequences to the longest game in the batch."""
    seq_lens = [g["seq_len"] for g in batch]
    max_len  = max(seq_lens)
    n_feats  = batch[0]["features"].shape[1]
    bsz      = len(batch)

    features = torch.zeros(bsz, max_len, n_feats, dtype=torch.float32)
    minutes  = torch.zeros(bsz, max_len, dtype=torch.long)
    labels   = torch.tensor([g["label"] for g in batch], dtype=torch.float32)
    lengths  = torch.tensor(seq_lens, dtype=torch.long)

    for i, g in enumerate(batch):
        sl = g["seq_len"]
        features[i, :sl] = torch.from_numpy(g["features"])
        minutes[i, :sl]  = torch.from_numpy(g["minutes"])

    return {
        "features": features,
        "minutes":  minutes,
        "labels":   labels,
        "lengths":  lengths,
        "game_ids": [g["game_id"] for g in batch],
    }


# ── Positional encoding ───────────────────────────────────────────────────────

class MinutePositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding indexed by actual game minute (not sequence
    position).  Minute 5 and minute 30 are always encoded the same way regardless
    of where they appear in the batch tensor — unlike standard sequence-position PE.

    The encoding table is pre-computed up to max_minute and stored as a buffer
    so it moves to the correct device automatically with the model.
    """

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
        self.register_buffer("pe", pe)           # (max_minute+1, d_model)

    def forward(self, x: torch.Tensor, minutes: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, T, d_model)
        minutes : (B, T)  integer game minutes  (values 0 .. max_minute)
        returns : (B, T, d_model)
        """
        minutes = minutes.clamp(0, self.pe.shape[0] - 1)   # safety clamp
        return x + self.pe[minutes]                         # broadcast lookup


# ── Model ─────────────────────────────────────────────────────────────────────

class LoLTransformer(nn.Module):
    """
    Causal Transformer that predicts win probability at every game minute.

    Input flow:
      categorical cols → shared embedding tables (one per type, all 10 slots)
      continuous cols  → standardised floats
      concat → linear projection → LayerNorm → Dropout
      → add minute-indexed sinusoidal PE
      → N-layer causal TransformerEncoder (pre-norm, batch_first=True)
      → linear output → sigmoid  →  (B, T) win probabilities

    Causality: an upper-triangular additive mask (-inf above diagonal) prevents
    any timestep from attending to future timesteps.  Padding positions (after
    real sequence end) are left-aligned and cannot be attended to by earlier
    real positions under the causal mask, so no key_padding_mask is needed.
    """

    def __init__(
        self,
        cat_col_indices:  dict[str, list[int]],
        cont_col_indices: list[int],
        cat_n_unique:     dict[str, int],
        d_model:    int = D_MODEL,
        nhead:      int = NHEAD,
        num_layers: int = NUM_LAYERS,
        ffn_dim:    int = FFN_DIM,
        dropout:    float = DROPOUT,
    ):
        super().__init__()
        self.cat_col_indices  = cat_col_indices
        self.cont_col_indices = cont_col_indices
        self.d_model = d_model

        # One shared embedding table per categorical type across all 10 slots.
        # Index 0 is the padding sentinel (padding_idx fixes its gradient to 0).
        self.embeddings = nn.ModuleDict({
            suffix: nn.Embedding(
                cat_n_unique[suffix] + 1,   # +1 for padding index 0
                EMBED_DIM[suffix],
                padding_idx=0,
            )
            for suffix in CAT_SUFFIXES
        })

        embed_total = 10 * sum(EMBED_DIM.values())   # 10 slots × 44 = 440
        n_cont      = len(cont_col_indices)
        input_size  = embed_total + n_cont

        # Project raw input → d_model with normalisation
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.pos_enc = MinutePositionalEncoding(d_model)

        # Pre-norm TransformerEncoderLayer (norm_first=True, PyTorch >= 1.11)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward = ffn_dim,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # pre-norm: more stable gradient flow
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
            enable_nested_tensor = False,   # avoids a known warning with masks
        )

        self.output_head = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    @staticmethod
    def _causal_mask(max_len: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular additive mask: 0.0 on/below diagonal, -inf above.
        Shape (max_len, max_len).  Position i cannot attend to j > i.
        """
        return torch.triu(
            torch.full((max_len, max_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(
        self,
        features: torch.Tensor,   # (B, T, F)  float32
        lengths:  torch.Tensor,   # (B,)        long
        minutes:  torch.Tensor,   # (B, T)      long — actual game minutes
    ) -> torch.Tensor:            # (B, T)      win probabilities in [0, 1]
        B, T, _ = features.shape
        device   = features.device

        # ── Categorical embeddings ─────────────────────────────────────────────
        cat_parts: list[torch.Tensor] = []
        for suffix in CAT_SUFFIXES:
            col_idx  = self.cat_col_indices[suffix]        # list of 10 col indices
            cat_vals = features[:, :, col_idx].long()      # (B, T, 10)
            emb      = self.embeddings[suffix](cat_vals)   # (B, T, 10, embed_dim)
            cat_parts.append(emb.view(B, T, -1))           # (B, T, 10*embed_dim)

        # ── Continuous features ────────────────────────────────────────────────
        cont = features[:, :, self.cont_col_indices]       # (B, T, n_cont)

        x = torch.cat(cat_parts + [cont], dim=-1)          # (B, T, input_size)

        # ── Project + positional encoding ──────────────────────────────────────
        x = self.input_proj(x)                             # (B, T, d_model)
        x = self.pos_enc(x, minutes)                       # (B, T, d_model)

        # ── Causal Transformer ─────────────────────────────────────────────────
        # Causal mask prevents attending to future timesteps.
        # No key_padding_mask: left-aligned sequences mean padded positions only
        # appear AFTER real ones and are unreachable under the causal mask.
        mask = self._causal_mask(T, device)
        x    = self.transformer(x, mask=mask)              # (B, T, d_model)

        # ── Output head ────────────────────────────────────────────────────────
        logits = self.output_head(x).squeeze(-1)           # (B, T)
        probs  = torch.sigmoid(logits)

        # Zero out padding positions (they contain junk — belt and suspenders)
        real = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        return probs * real.float()


# ── Inference helper ──────────────────────────────────────────────────────────

@torch.no_grad()
def predict_dataset(
    model: LoLTransformer,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> pd.DataFrame:
    """Run inference; return DataFrame with (game_id, minute, blue_win, pred)."""
    model.eval()
    rows: list[dict] = []

    for batch in loader:
        feats   = batch["features"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device,  non_blocking=True)
        minutes = batch["minutes"].to(device,  non_blocking=True)
        labels  = batch["labels"]           # (B,)  CPU
        min_cpu = batch["minutes"]          # (B, T) CPU

        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            probs = model(feats, lengths, minutes)   # (B, T)

        for i, (gid, sl, lbl) in enumerate(zip(batch["game_ids"], batch["lengths"], labels)):
            sl_i  = sl.item()
            p_i   = probs[i, :sl_i].cpu().numpy()
            m_i   = min_cpu[i, :sl_i].numpy()
            lbl_v = int(lbl.item())
            for t in range(sl_i):
                rows.append({
                    "game_id":  gid,
                    "minute":   int(m_i[t]),
                    "blue_win": lbl_v,
                    "pred":     float(p_i[t]),
                })

    return pd.DataFrame(rows)


# ── Metric helpers ─────────────────────────────────────────────────────────────

def auc_by_bucket(pred_df: pd.DataFrame) -> list[tuple[str, float, int]]:
    results = []
    for (lo, hi), label in zip(MINUTE_BUCKETS, BUCKET_LABELS):
        sub = pred_df[(pred_df["minute"] >= lo) & (pred_df["minute"] < hi)]
        if len(sub) < 100:
            continue
        auc = roc_auc_score(sub["blue_win"], sub["pred"])
        results.append((label, auc, len(sub)))
    return results


def _auc_at_minute(pred_df: pd.DataFrame, minute: int) -> float:
    """AUC using only predictions at a specific game minute."""
    sub = pred_df[pred_df["minute"] == minute]
    if len(sub) < 50:
        return float("nan")
    return roc_auc_score(sub["blue_win"], sub["pred"])


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_train_history(
    train_losses:    list[float],
    val_losses:      list[float],
    train_aucs_end:  list[float],
    val_aucs_end:    list[float],
    val_aucs_m5:     list[float],
    val_aucs_m10:    list[float],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="train loss")
    axes[0].plot(epochs, val_losses,   label="val loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Smoothed BCE")
    axes[0].set_title("Training Loss"); axes[0].legend()

    axes[1].plot(epochs, train_aucs_end, label="train AUC@end", linestyle="--", alpha=0.7)
    axes[1].plot(epochs, val_aucs_end,   label="val AUC@end",   color="C1")
    axes[1].plot(epochs, val_aucs_m10,   label="val AUC@min10", color="C2")
    axes[1].plot(epochs, val_aucs_m5,    label="val AUC@min5",  color="C3")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC over Epochs"); axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_auc_by_minute(pred_df: pd.DataFrame, path: Path, title: str = "Transformer") -> None:
    bucket_results = auc_by_bucket(pred_df)
    labels = [r[0] for r in bucket_results]
    aucs   = [r[1] for r in bucket_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, aucs, color="#55A868", edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random")
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Game minute bucket"); ax.set_ylabel("AUC")
    ax.set_title(f"{title} — Val AUC by Minute Bucket"); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_calibration(pred_df: pd.DataFrame, path: Path) -> None:
    prob_true, prob_pred = calibration_curve(
        pred_df["blue_win"], pred_df["pred"], n_bins=20, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, "o-", label="Transformer", color="#55A868")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction positives")
    ax.set_title("Calibration — Transformer"); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_vs_snapshot(transformer_df: pd.DataFrame, path: Path) -> None:
    """Bar chart comparing Transformer val AUC vs LightGBM OOF AUC by minute."""
    if not SNAPSHOT_CV.exists():
        log.info("Snapshot CV not found; skipping comparison plot.")
        return

    with open(SNAPSHOT_CV, "rb") as f:
        snap = pickle.load(f)
    snap_df = snap["oof_df"]

    snap_aucs, tfm_aucs, used_labels = [], [], []
    for (lo, hi), label in zip(MINUTE_BUCKETS, BUCKET_LABELS):
        s_sub = snap_df[(snap_df["minute"] >= lo) & (snap_df["minute"] < hi)]
        t_sub = transformer_df[(transformer_df["minute"] >= lo) & (transformer_df["minute"] < hi)]
        if len(s_sub) < 100 or len(t_sub) < 100:
            continue
        snap_aucs.append(roc_auc_score(s_sub["blue_win"], s_sub["oof_pred"]))
        tfm_aucs.append( roc_auc_score(t_sub["blue_win"], t_sub["pred"]))
        used_labels.append(label)

    xi    = np.arange(len(used_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(xi - width / 2, snap_aucs, width, label="LightGBM (snapshot)", color="#4C72B0", edgecolor="white")
    b2 = ax.bar(xi + width / 2, tfm_aucs,  width, label="Transformer (causal)",  color="#55A868", edgecolor="white")
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(xi); ax.set_xticklabels(used_labels)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Game minute bucket"); ax.set_ylabel("AUC")
    ax.set_title("LightGBM vs Transformer — AUC by Minute Bucket"); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_win_probability_trajectories(
    model: LoLTransformer,
    test_games: list[dict],
    device: torch.device,
    path: Path,
    use_amp: bool = False,
    n_samples: int = 6,
) -> None:
    model.eval()
    rng     = np.random.default_rng(0)
    indices = rng.choice(len(test_games), size=min(n_samples, len(test_games)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            g   = test_games[idx]
            sl  = g["seq_len"]
            x   = torch.from_numpy(g["features"]).unsqueeze(0).to(device)  # (1, T, F)
            ln  = torch.tensor([sl], dtype=torch.long).to(device)
            mn  = torch.from_numpy(g["minutes"]).unsqueeze(0).to(device)   # (1, T)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                p = model(x, ln, mn)[0, :sl].cpu().numpy()

            color   = "#4C72B0" if g["label"] == 1 else "#DD8452"
            outcome = "Blue Win" if g["label"] == 1 else "Red Win"
            ax.plot(g["minutes"].tolist(), p, color=color, linewidth=1.5)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Minute"); ax.set_ylabel("P(blue win)")
            ax.set_title(f"Game {g['game_id']} — {outcome}", fontsize=9)

    fig.suptitle("Transformer Win Probability Trajectories (test set)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


# ── Preprocessing ──────────────────────────────────────────────────────────────

def build_cat_encoders(df: pd.DataFrame) -> tuple[dict[str, LabelEncoder], dict[str, int]]:
    """Fit one LabelEncoder per categorical type over all 10 slots."""
    encoders: dict[str, LabelEncoder] = {}
    n_unique: dict[str, int]          = {}
    for suffix in CAT_SUFFIXES:
        cols     = [f"{slot}_{suffix}" for slot in SLOTS]
        all_vals = df[cols].values.ravel()
        le       = LabelEncoder().fit(all_vals)
        encoders[suffix] = le
        n_unique[suffix] = len(le.classes_)
    return encoders, n_unique


def apply_cat_encoders(df: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    """Label-encode categoricals and shift by +1 (0 reserved for padding)."""
    df = df.copy()
    for suffix, le in encoders.items():
        for slot in SLOTS:
            col     = f"{slot}_{suffix}"
            df[col] = le.transform(df[col].values).astype(np.int32) + 1
    return df


def build_game_list(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "blue_win",
) -> list[dict]:
    """
    Group rows by game_id; return one dict per game:
        features  : float32 array (seq_len, n_features)
        minutes   : int32 array   (seq_len,)
        label     : float
        seq_len   : int
        game_id   : str | int
    """
    games: list[dict] = []
    for gid, group in df.groupby("game_id", sort=False):
        group = group.sort_values("minute")
        feats = group[feature_cols].values.astype(np.float32)
        mins  = group["minute"].values.astype(np.int32)
        label = float(group[label_col].iloc[0])
        games.append({
            "game_id":  gid,
            "features": feats,
            "minutes":  mins,
            "label":    label,
            "seq_len":  len(feats),
        })
    return games


# ── AUC sampling helpers (used during training loop) ─────────────────────────

def _sample_auc(
    model: LoLTransformer,
    games: list[dict],
    device: torch.device,
    use_amp: bool,
    minute_targets: tuple[int, int],
    sample_n: int = 1000,
) -> tuple[list[float], list[float], list[float], list[int]]:
    """
    Quick AUC sampling from a random subset of games during training.
    Returns (preds@min_a, preds@min_b, preds@end, labels).
    """
    model.eval()
    rng     = np.random.default_rng(0)
    indices = rng.choice(len(games), size=min(sample_n, len(games)), replace=False)

    pa, pb, pe, lbls = [], [], [], []
    m_a, m_b = minute_targets

    with torch.no_grad():
        for idx in indices:
            g   = games[idx]
            sl  = g["seq_len"]
            x   = torch.from_numpy(g["features"]).unsqueeze(0).to(device)
            ln  = torch.tensor([sl], dtype=torch.long).to(device)
            mn  = torch.from_numpy(g["minutes"]).unsqueeze(0).to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                probs = model(x, ln, mn)[0, :sl].cpu().numpy()

            mins_i = g["minutes"].tolist()

            def _idx_at(target: int) -> int:
                cands = [j for j, m in enumerate(mins_i) if m <= target]
                return cands[-1] if cands else 0

            pa.append(float(probs[_idx_at(m_a)]))
            pb.append(float(probs[_idx_at(m_b)]))
            pe.append(float(probs[-1]))
            lbls.append(int(g["label"]))

    model.train()
    return pa, pb, pe, lbls


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train causal Transformer time-series model")
    p.add_argument("--epochs",     type=int,   default=MAX_EPOCHS)
    p.add_argument("--limit",      type=int,   default=None,
                   help="Use only first N games (quick testing)")
    p.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--d-model",    type=int,   default=D_MODEL,
                   help="Transformer d_model. CPU default=128; GPU recommend 256")
    p.add_argument("--nhead",      type=int,   default=NHEAD,
                   help="Number of attention heads (must divide d_model evenly)")
    p.add_argument("--num-layers", type=int,   default=NUM_LAYERS,
                   help="Number of TransformerEncoder layers")
    p.add_argument("--ffn-dim",    type=int,   default=FFN_DIM,
                   help="Feed-forward network hidden dim")
    p.add_argument("--dropout",    type=float, default=DROPOUT)
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile() even when CUDA is available")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0   = time.time()

    # ── Device selection ──────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    log.info("Device: %s  |  AMP: %s", device, use_amp)

    torch.set_num_threads(N_THREADS)
    if device.type == "cpu":
        log.info("PyTorch CPU threads: %d  (cores: %d)", N_THREADS, os.cpu_count() or 0)
    else:
        log.info("CUDA device: %s  (%s)",
                 torch.cuda.get_device_name(0),
                 f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        log.info("PyTorch CPU threads: %d  (cores: %d)", N_THREADS, os.cpu_count() or 0)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading %s …", FEATURES_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    log.info("Loaded: %d rows × %d cols", len(df), df.shape[1])

    if args.limit:
        ids    = df["game_id"].unique()[: args.limit]
        df     = df[df["game_id"].isin(ids)].copy()
        log.info("Limiting to %d games -> %d rows", args.limit, len(df))

    # ── Feature columns ───────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    all_cat_cols = {f"{slot}_{suf}" for slot in SLOTS for suf in CAT_SUFFIXES}
    cont_cols    = [c for c in feature_cols if c not in all_cat_cols]

    cat_col_indices: dict[str, list[int]] = {
        suffix: [feature_cols.index(f"{slot}_{suffix}") for slot in SLOTS]
        for suffix in CAT_SUFFIXES
    }
    cont_col_indices = [feature_cols.index(c) for c in cont_cols]

    log.info("Feature cols: %d  (cat: %d, cont: %d)",
             len(feature_cols), len(all_cat_cols), len(cont_cols))

    # ── Categorical encoding ──────────────────────────────────────────────────
    log.info("Fitting categorical encoders …")
    cat_encoders, cat_n_unique = build_cat_encoders(df)
    df = apply_cat_encoders(df, cat_encoders)
    log.info("Categorical n_unique: %s", cat_n_unique)

    # ── Game-level 80 / 10 / 10 split ────────────────────────────────────────
    all_ids                   = df["game_id"].unique()
    train_ids, valtest_ids    = train_test_split(all_ids,    test_size=0.2,  random_state=42)
    val_ids,   test_ids       = train_test_split(valtest_ids, test_size=0.5, random_state=42)
    log.info("Games — train: %d  val: %d  test: %d",
             len(train_ids), len(val_ids), len(test_ids))

    train_df = df[df["game_id"].isin(train_ids)]
    val_df   = df[df["game_id"].isin(val_ids)]
    test_df  = df[df["game_id"].isin(test_ids)]

    # ── StandardScaler on continuous features (fit on train only) ─────────────
    log.info("Fitting StandardScaler on %d train rows …", len(train_df))
    scaler   = StandardScaler()
    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()
    train_df[cont_cols] = scaler.fit_transform(train_df[cont_cols].values.astype(np.float32))
    val_df[cont_cols]   = scaler.transform(val_df[cont_cols].values.astype(np.float32))
    test_df[cont_cols]  = scaler.transform(test_df[cont_cols].values.astype(np.float32))

    # ── Build per-game sequence lists ─────────────────────────────────────────
    log.info("Building game sequence lists …")
    train_games = build_game_list(train_df, feature_cols)
    val_games   = build_game_list(val_df,   feature_cols)
    test_games  = build_game_list(test_df,  feature_cols)
    log.info("Sequences — train: %d  val: %d  test: %d",
             len(train_games), len(val_games), len(test_games))

    del df, train_df, val_df, test_df   # free memory before model allocation

    # ── DataLoaders ───────────────────────────────────────────────────────────
    pin = device.type == "cuda"
    # num_workers > 0 requires care on Windows; keep 0 for portability.
    # On Linux + GPU, set num_workers=4 for faster data loading.
    nw  = 0
    train_loader = DataLoader(
        GameSequenceDataset(train_games), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=nw, pin_memory=pin,
    )
    val_loader = DataLoader(
        GameSequenceDataset(val_games), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=nw, pin_memory=pin,
    )
    test_loader = DataLoader(
        GameSequenceDataset(test_games), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=nw, pin_memory=pin,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LoLTransformer(
        cat_col_indices  = cat_col_indices,
        cont_col_indices = cont_col_indices,
        cat_n_unique     = cat_n_unique,
        d_model          = args.d_model,
        nhead            = args.nhead,
        num_layers       = args.num_layers,
        ffn_dim          = args.ffn_dim,
        dropout          = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model params: %s  |  d_model=%d  nhead=%d  layers=%d  ffn=%d",
             f"{n_params:,}", args.d_model, args.nhead, args.num_layers, args.ffn_dim)

    # torch.compile: significant GPU speed-up (~20-40%); skip on CPU (overhead)
    if device.type == "cuda" and not args.no_compile and hasattr(torch, "compile"):
        log.info("Applying torch.compile() for GPU acceleration …")
        model = torch.compile(model)   # type: ignore[assignment]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    # Gradient scaler for AMP on GPU; no-op on CPU
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auc  = 0.0
    best_state:   dict | None = None
    no_improve    = 0

    train_loss_hist:  list[float] = []
    val_loss_hist:    list[float] = []
    train_auc_hist:   list[float] = []
    val_auc_end_hist: list[float] = []
    val_auc_m5_hist:  list[float] = []
    val_auc_m10_hist: list[float] = []

    log.info("Starting training — max %d epochs, patience %d …", args.epochs, PATIENCE)
    log.info("Log format: loss=train/val | AUC@5m=tr/val  AUC@10m=tr/val  AUC@end=tr/val")

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss_sum   = 0.0
        train_total_toks = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch:02d} train", leave=False):
            feats   = batch["features"].to(device, non_blocking=True)   # (B, T, F)
            lengths = batch["lengths"].to(device,  non_blocking=True)   # (B,)
            minutes = batch["minutes"].to(device,  non_blocking=True)   # (B, T)
            labels  = batch["labels"].to(device,   non_blocking=True)   # (B,)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                probs = model(feats, lengths, minutes)   # (B, T)

            probs = probs.float().nan_to_num(nan=0.5)
            B, T = probs.shape
            mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)

            # Label smoothing: broadcast game label to every real timestep
            labels_exp = labels.unsqueeze(1).expand(B, T)
            smooth     = labels_exp[mask] * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / 2
            loss       = F.binary_cross_entropy(probs[mask].clamp(1e-6, 1 - 1e-6), smooth.float())

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            optimizer.zero_grad(set_to_none=True)

            n_toks         = mask.sum().item()
            train_loss_sum += loss.item() * n_toks
            train_total_toks += n_toks

        train_loss = train_loss_sum / max(train_total_toks, 1)

        # ── Validate (full pass over val set) ─────────────────────────────────
        model.eval()
        val_loss_sum   = 0.0
        val_total_toks = 0

        with torch.no_grad():
            for batch in val_loader:
                feats   = batch["features"].to(device, non_blocking=True)
                lengths = batch["lengths"].to(device,  non_blocking=True)
                minutes = batch["minutes"].to(device,  non_blocking=True)
                labels  = batch["labels"].to(device,   non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                    probs = model(feats, lengths, minutes)
                probs = probs.float().nan_to_num(nan=0.5)
                B, T  = probs.shape
                mask  = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
                lexp  = labels.unsqueeze(1).expand(B, T)
                loss  = F.binary_cross_entropy(probs[mask].clamp(1e-6, 1 - 1e-6), lexp[mask].float())

                n_toks        = mask.sum().item()
                val_loss_sum += loss.item() * n_toks
                val_total_toks += n_toks

        val_loss = val_loss_sum / max(val_total_toks, 1)

        # ── AUC sampling (subset to keep CPU training fast) ────────────────────
        sample_n = min(2000, len(train_games))
        tr_pa, tr_pb, tr_pe, tr_lbl = _sample_auc(model, train_games, device, use_amp, (5, 10), sample_n)
        va_pa, va_pb, va_pe, va_lbl = _sample_auc(model, val_games,   device, use_amp, (5, 10))

        try:
            tr_auc_end  = roc_auc_score(tr_lbl, tr_pe)
            val_auc_end = roc_auc_score(va_lbl, va_pe)
            val_auc_m5  = roc_auc_score(va_lbl, va_pa)
            val_auc_m10 = roc_auc_score(va_lbl, va_pb)
        except Exception:
            tr_auc_end = val_auc_end = val_auc_m5 = val_auc_m10 = float("nan")

        scheduler.step()

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_auc_hist.append(tr_auc_end)
        val_auc_end_hist.append(val_auc_end)
        val_auc_m5_hist.append(val_auc_m5)
        val_auc_m10_hist.append(val_auc_m10)

        elapsed = time.time() - ep_t0
        log.info(
            "Epoch %02d/%02d | loss=%.4f/%.4f | "
            "AUC@5m=%.3f/%.3f  AUC@10m=%.3f/%.3f  AUC@end=%.3f/%.3f | %.1fs",
            epoch, args.epochs, train_loss, val_loss,
            roc_auc_score(tr_lbl, tr_pa) if not math.isnan(tr_auc_end) else float("nan"),
            val_auc_m5,
            roc_auc_score(tr_lbl, tr_pb) if not math.isnan(tr_auc_end) else float("nan"),
            val_auc_m10,
            tr_auc_end, val_auc_end,
            elapsed,
        )

        # ── Early stopping ──────────────────────────────────────────────────────
        if not math.isnan(val_auc_end) and val_auc_end > best_val_auc + 1e-4:
            best_val_auc = val_auc_end
            best_state   = {k: v.clone() for k, v in (
                model._orig_mod if hasattr(model, "_orig_mod") else model
            ).state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                log.info("Early stopping at epoch %d (patience=%d).", epoch, PATIENCE)
                break

    # ── Restore best weights ──────────────────────────────────────────────────
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if best_state is not None:
        raw_model.load_state_dict(best_state)
        log.info("Restored best weights (val AUC@end=%.4f).", best_val_auc)

    # ── Final evaluation on val set (all timesteps) ───────────────────────────
    log.info("Computing full per-minute AUC on val set …")
    val_pred_df     = predict_dataset(raw_model, val_loader, device, use_amp)
    overall_val_auc = roc_auc_score(val_pred_df["blue_win"], val_pred_df["pred"])
    log.info("Val AUC (all timesteps): %.4f", overall_val_auc)

    print("\n" + "=" * 60)
    print(f"  Best val AUC (last timestep)     : {best_val_auc:.4f}")
    print(f"  Val AUC (all timesteps combined) : {overall_val_auc:.4f}")
    print("=" * 60)
    print("\nAUC by minute bucket (val set, all timesteps):")
    for label, auc, n in auc_by_bucket(val_pred_df):
        print(f"  {label:>6} min : AUC={auc:.4f}  ({n:>7,} rows)")

    # ── Save model + artifacts ────────────────────────────────────────────────
    model_path = MODELS_DIR / "transformer_timeseries.pt"
    torch.save(raw_model.state_dict(), model_path)
    log.info("Saved model: %s", model_path)

    artifacts = {
        "cat_encoders":     cat_encoders,
        "cat_n_unique":     cat_n_unique,
        "scaler":           scaler,
        "feature_cols":     feature_cols,
        "cont_cols":        cont_cols,
        "cat_col_indices":  cat_col_indices,
        "cont_col_indices": cont_col_indices,
        "model_config": {
            "cat_col_indices":  cat_col_indices,
            "cont_col_indices": cont_col_indices,
            "cat_n_unique":     cat_n_unique,
            "d_model":          args.d_model,
            "nhead":            args.nhead,
            "num_layers":       args.num_layers,
            "ffn_dim":          args.ffn_dim,
            "dropout":          args.dropout,
        },
        "train_history": {
            "train_loss":     train_loss_hist,
            "val_loss":       val_loss_hist,
            "train_auc_end":  train_auc_hist,
            "val_auc_end":    val_auc_end_hist,
            "val_auc_m5":     val_auc_m5_hist,
            "val_auc_m10":    val_auc_m10_hist,
        },
    }
    art_path = MODELS_DIR / "transformer_artifacts.pkl"
    with open(art_path, "wb") as f:
        pickle.dump(artifacts, f)
    log.info("Saved artifacts: %s", art_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_train_history(
        train_loss_hist, val_loss_hist,
        train_auc_hist,  val_auc_end_hist,
        val_auc_m5_hist, val_auc_m10_hist,
        REPORTS_DIR / "transformer_train_history.png",
    )
    plot_auc_by_minute(val_pred_df,  REPORTS_DIR / "transformer_auc_by_minute.png")
    plot_calibration(val_pred_df,    REPORTS_DIR / "transformer_calibration.png")
    plot_vs_snapshot(val_pred_df,    REPORTS_DIR / "transformer_vs_snapshot_auc.png")
    plot_win_probability_trajectories(
        raw_model, test_games, device,
        REPORTS_DIR / "transformer_win_probability_trajectory.png",
        use_amp=use_amp,
    )

    total = time.time() - t0
    log.info("Done in %.1f s (%.1f min)", total, total / 60)


if __name__ == "__main__":
    main()
