# Agent Handoff — LoL Win-Contribution Pipeline
*Written 2026-03-17. This document is the primary context brief for the GPU workstation agent.*

---

## 1. Project Overview

We are building a Python ML pipeline to quantify the **win contribution** of each of the 10 players in a League of Legends match.

- **Core idea:** Train a model that predicts win probability at every minute of a game, then use **SHAP** to attribute the model's output to individual players. Summing a player's SHAP values across all their features gives their total contribution to the win/loss.
- **Output goal:** A per-player contribution score (eventually a tier rating S/A/B/C) for any given match.
- **Why accuracy matters:** SHAP attributions are only meaningful if the underlying model is genuinely predictive. Low AUC = noisy, unreliable SHAP values.

**Project root:** wherever the codebase was copied to on the workstation (was `d:\AntiGravity_codes\LoL_AI\` on the source machine).

---

## 2. FIRST THINGS TO DO — Environment Setup

### Step 1: Check for an existing conda environment

```bash
conda env list
```

Look for an environment named **`lol_shap_env`**. If it exists, check whether PyTorch is GPU-enabled:

```bash
conda activate lol_shap_env
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

**If CUDA shows False:** PyTorch was installed as CPU-only (the original environment.yml installs the CPU wheel). You must reinstall PyTorch with CUDA support (see Step 2b).

**If CUDA shows True:** Environment is ready — skip to Step 3.

### Step 2a: Create the environment from scratch (if lol_shap_env does not exist)

```bash
conda create -n lol_shap_env python=3.10 -y
conda activate lol_shap_env
conda install -c conda-forge pandas numpy pyarrow lightgbm xgboost scikit-learn shap matplotlib seaborn jupyter tqdm requests -y
pip install riotwatcher python-dotenv
```

Then install PyTorch with CUDA (see Step 2b).

### Step 2b: Install GPU-enabled PyTorch

Check the CUDA version on the workstation first:

```bash
nvidia-smi
```

Then install the matching PyTorch wheel. For CUDA 12.x (RTX 3090/4090):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify after install:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

### Step 3: Verify all data files are present

```bash
ls data/processed/
```

Expected files:
| File | Size | Description |
|------|------|-------------|
| `data/processed/features.parquet` | ~750 MB | 3.59M rows × 482 cols, 133K games, per-minute game features |
| `data/processed/player_game_summary.parquet` | ~200 MB | 1.33M rows, 133K games × 10 players, end-of-game stats per player |

```bash
python -c "
import pandas as pd
f = pd.read_parquet('data/processed/features.parquet')
p = pd.read_parquet('data/processed/player_game_summary.parquet')
print('features:', f.shape)
print('player_summary:', p.shape)
print('games in features:', f.game_id.nunique())
print('players in summary:', p.puuid.nunique())
"
```

Expected:
- features: ~(3,590,000, 482)
- player_summary: ~(1,330,000, ~25 cols)
- ~133,347 unique game_ids
- ~43,447 unique players

### Step 4: Test Riot API connectivity

The `.env` file at the project root must contain a valid Riot API key:
```
RIOT_API_KEY=RGAPI-xxxx-...
```

**Important:** Development API keys expire every 24 hours. You will need to generate a fresh key at https://developer.riotgames.com/ before any API calls.

Run this connectivity test to confirm the key works and the API is reachable. It fetches the Challenger ladder for NA1, picks the first player, then fetches their last 3 match IDs, one full match JSON, and one timeline JSON — a complete end-to-end test of the entire harvesting chain:

```python
# Save as test_api.py and run: python test_api.py
import os, json
from dotenv import load_dotenv
from riotwatcher import LolWatcher, ApiError

load_dotenv()
key = os.getenv("RIOT_API_KEY")
assert key, "RIOT_API_KEY not found in .env"
watcher = LolWatcher(key)

# 1. Pull Challenger ladder
print("Fetching Challenger ladder (NA1)...")
ladder = watcher.league.challenger_by_queue("na1", "RANKED_SOLO_5x5")
entry = ladder.entries[0]
summoner_name = entry["summonerId"]
print(f"  Top player summonerId: {summoner_name}")

# 2. Get PUUID
summoner = watcher.summoner.by_id("na1", summoner_name)
puuid = summoner["puuid"]
print(f"  PUUID: {puuid[:20]}...")

# 3. Fetch last 3 match IDs
match_ids = watcher.match.matchlist_by_puuid("americas", puuid, count=3, queue=420)
print(f"  Match IDs: {match_ids}")

# 4. Fetch one full match
match = watcher.match.by_id("americas", match_ids[0])
print(f"  Match fetched: {match['metadata']['matchId']} — {len(match['info']['participants'])} participants")

# 5. Fetch timeline
timeline = watcher.match.timeline_by_match("americas", match_ids[0])
n_frames = len(timeline["info"]["frames"])
print(f"  Timeline fetched: {n_frames} frames (minutes)")

print("\nAPI connectivity OK.")
```

```bash
python test_api.py
```

If you get a 403 error, the API key is expired — generate a new one and update `.env`.
If you get a connection error, check network/firewall settings.

### Step 5: Smoke-test all key scripts

Run each of these quick tests in order. They should all complete without errors.

```bash
# 4a — LightGBM snapshot model (quick test, 3 folds)
python src/04a_train_snapshot.py --limit 5000 --n-folds 3

# 4b — Causal Transformer, per-minute time series (5 epochs, 5K games)
python src/04b_train_timeseries.py --limit 5000 --epochs 5

# 4c — Player context Transformer, game-summary history (5 epochs, 5K games, K=5)
python src/04c_train_player_context.py --limit 5000 --epochs 5 --k 5

# 4d — Full minute-level context architecture, benchmark only
python src/04d_train_player_context_minute.py --benchmark --limit 500 --no-sequences
```

All logs go to `logs/`. If a script fails, read the log file for details.

---

## 3. Full Model Vision — What We Are Building

This section explains the overall goal and the final model architecture at a high level.

### The Problem

A LoL match has 10 players on 2 teams. **Win/loss is a team outcome** — there is no ground-truth "contribution" label per player. We solve this by:
1. Training a model to predict win probability at every minute from the game state
2. Using SHAP to decompose the model's output into per-player contributions
3. Summing each player's SHAP values to get their total contribution score

For SHAP to be meaningful, the underlying model must be highly predictive — especially in the early/mid game, where individual skill matters most.

### Why Player History Context Matters

Without context, the model only sees what's happening in the current game. It cannot distinguish:
- A player who is already 1000 gold ahead because they are individually skilled
- A player who got a lucky early lead that any Challenger-level player would have

By encoding each player's **historical performance across past games**, the model can condition its predictions on individual skill level, champion mastery, and playstyle patterns. This improves both AUC and SHAP attribution quality.

### The Final Architecture (04c / 04d)

```
For each of 10 players in the current game:
  ┌─────────────────────────────────────────────────────────────┐
  │  Player History Encoder                                     │
  │                                                             │
  │  [Game K-1] [Game K-2] ... [Game 1]  ← past K games        │
  │      ↓           ↓             ↓                            │
  │  [end-of-game summary stats per game]   (04c)               │
  │  OR [per-minute sequence → causal Transformer → embedding]  │   (04d)
  │      ↓           ↓             ↓                            │
  │  Cross-game Transformer (non-causal, all history is past)   │
  │      ↓                                                      │
  │  Mean-pool → player_embedding  (dim=64 or 128)              │
  └─────────────────────────────────────────────────────────────┘
              ↓ (×10 players, one embedding per player)

  ┌─────────────────────────────────────────────────────────────┐
  │  Game-Level Predictor (Transformer)                         │
  │                                                             │
  │  Input sequence:                                            │
  │  [P0][P1][P2][P3][P4][P5][P6][P7][P8][P9]  ← player tokens │
  │  [min1][min2][min3]...[minT]                ← game tokens   │
  │                                                             │
  │  Hybrid causal mask:                                        │
  │  - Player tokens: attend to all other player tokens         │
  │  - Game tokens: attend to all player tokens                 │
  │                 + causally past game tokens (not future)    │
  │                                                             │
  │  Output head: win_prob at each game token position          │
  └─────────────────────────────────────────────────────────────┘
              ↓
  win_probability(t) for t = 1..T_game

              ↓  (via SHAP)
  per-player contribution score for each of the 10 players
```

### Model Variants (in order of complexity)

| Script | Player context | Game encoding | Notes |
|--------|---------------|---------------|-------|
| `04a` | None | LightGBM snapshot | Fastest; best for SHAP baseline |
| `04b` | None | Causal Transformer | Sequence-aware, no player context |
| `04c` | End-of-game stats → Transformer | Causal Transformer | Recommended first full run |
| `04d` | Per-minute history → 3-level Transformer | Causal Transformer | Most powerful; needs sequences parquet |

The target model is **04d** on a 1M+ game dataset on the GPU workstation. **04c** is the practical intermediate that can be trained well with 133K games.

---

## 4. Codebase Overview

### Directory structure

```
LoL_AI/
├── environment.yml                      # conda spec (has CPU torch — see setup above)
├── project_summary.md                   # detailed project doc (may be outdated vs. this file)
├── agent_handoff.md                     # THIS FILE
├── .env                                 # Riot API key (RIOT_API_KEY=...)
├── src/
│   ├── utils.py                         # API key loader, RiotWatcher helpers
│   ├── 01_explore_data.py               # data exploration (DONE, informational)
│   ├── 02_bulk_harvest.py               # Riot API data harvesting (see §5)
│   ├── 03a_schema_audit.py              # schema audit (DONE, informational)
│   ├── 03_process_features.py           # raw JSON → features.parquet
│   ├── 03b_build_player_index.py        # features.parquet → player_game_summary + sequences
│   ├── 04a_train_snapshot.py            # LightGBM per-minute snapshot model
│   ├── 04b_train_timeseries.py          # Causal Transformer, per-minute sequence
│   ├── 04c_train_player_context.py      # Transformer + game-summary player context
│   ├── 04d_train_player_context_minute.py # Full 3-level minute context + benchmark
│   ├── 06_shap_explain.py               # SHAP TreeExplainer on LightGBM (04a)
│   └── 07_analysis.py                   # role importance, champion contribution
├── data/
│   ├── raw/
│   │   ├── matches/                     # 133K+ match JSON files (Riot V5)
│   │   ├── timelines/                   # 133K+ timeline JSON files
│   │   ├── players/                     # player mastery JSON files
│   │   ├── context_matches/             # (future) context game match JSONs
│   │   └── context_timelines/           # (future) context game timeline JSONs
│   └── processed/
│       ├── features.parquet             # DONE — 3.59M rows × 482 cols
│       ├── player_game_summary.parquet  # DONE — 1.33M rows
│       └── player_minute_sequences.parquet  # NOT YET — run 03b --include-sequences
├── models/                              # trained model checkpoints (may be empty)
├── reports/
│   ├── schema_audit.txt
│   └── event_audit.txt
└── logs/                                # per-script log files
```

### Script dependency order

```
02_bulk_harvest.py          → data/raw/matches/ + timelines/
    ↓
03_process_features.py      → data/processed/features.parquet
    ↓
03b_build_player_index.py   → data/processed/player_game_summary.parquet
                              data/processed/player_minute_sequences.parquet (--include-sequences)
    ↓
04a_train_snapshot.py       → models/lgbm_snapshot_cv.pkl
04b_train_timeseries.py     → models/transformer_timeseries.pt
04c_train_player_context.py → models/player_context_model.pt        (needs player_game_summary)
04d_train_player_context_minute.py → models/player_context_minute_model.pt  (needs sequences)
    ↓
06_shap_explain.py          → reports/shap_values.parquet  (needs 04a)
07_analysis.py              → role/champion contribution plots (needs 06)
```

---

## 5. What Has Been Done (Before This Handoff)

| Phase | Status | Notes |
|-------|--------|-------|
| Raw data harvesting | DONE | 133K+ games, NA1/EUW1/KR, Challenger/GM, Ranked Solo |
| Feature engineering (03) | DONE | features.parquet — 482 cols including delta/rate features |
| Player index (03b) | DONE | player_game_summary.parquet built (1.33M rows, ~2 min) |
| player_minute_sequences | NOT DONE | Requires `--include-sequences` flag on 03b — large file (~35M rows) |
| 04a LightGBM (30K games) | Done (old run) | OOF AUC 0.8163 |
| 04b Causal Transformer | TESTED | val AUC 0.8144 on 10K game quick test |
| 04c Player context model | WRITTEN + TESTED | Quick smoke test passed; NO full training yet |
| 04d Minute-level context | WRITTEN + TESTED | Benchmark + --no-sequences smoke test passed; NO full training yet |
| 06/07 SHAP analysis | WRITTEN | Not yet run at full scale |

---

## 6. Full Training — Run Order on GPU Workstation

This is the primary task. Run in order, monitoring GPU utilization and logs.

### 5a. LightGBM snapshot model (full, 133K games)

```bash
python src/04a_train_snapshot.py
```

- GroupKFold(5) CV — ~30–60 min on CPU, fast on CPU (LightGBM doesn't benefit much from GPU)
- Output: `models/lgbm_snapshot_cv.pkl`, `models/lgbm_snapshot_final.txt`
- Expected OOF AUC: ~0.82+
- Logs: `logs/04a_train_snapshot.log`

### 5b. Causal Transformer (full, 133K games)

```bash
python src/04b_train_timeseries.py --d-model 256 --nhead 8 --num-layers 6 --epochs 30
```

- Auto-detects CUDA, enables AMP, applies torch.compile
- Output: `models/transformer_timeseries.pt`, `models/transformer_artifacts.pkl`
- Logs: `logs/04b_train_timeseries.log`
- Log format: `loss=train/val | AUC@5m=tr/val  AUC@10m=tr/val  AUC@end=tr/val`

### 5c. Player context model — game-summary (full, 133K games, K=20)

```bash
python src/04c_train_player_context.py --epochs 30 --k 20
```

- Uses `player_game_summary.parquet` (must exist)
- Injects 10 player history embeddings as context tokens into the game Transformer
- Output: `models/player_context_model.pt`, `models/player_context_artifacts.pkl`
- Logs: `logs/04c_train_player_context.log`

### 5d. Generate per-minute sequences (needed for 04d)

```bash
python src/03b_build_player_index.py --include-sequences
```

- Outputs `data/processed/player_minute_sequences.parquet` (~35M rows, may take 15–30 min, ~8 GB RAM)
- Only needed if running 04d without `--no-sequences`

### 5e. Full minute-level player context model

```bash
# Benchmark first to estimate training time
python src/04d_train_player_context_minute.py --benchmark --k 50

# Then full training
python src/04d_train_player_context_minute.py --epochs 30 --k 50
```

- This is the most powerful and most expensive model (3-level Transformer hierarchy)
- Output: `models/player_context_minute_model.pt`
- Logs: `logs/04d_train_player_context_minute.log`

### 5f. SHAP analysis (after 04a trained)

```bash
python src/06_shap_explain.py
python src/07_analysis.py
```

---

## 7. Model Architecture Summary

### 04a — LightGBM Snapshot
- Input: single (game, minute) feature vector (~482 features)
- CV: GroupKFold(5) by game_id (critical — never split rows of the same game)
- Label: `blue_win` (1 = blue team wins)
- SHAP: TreeSHAP, sum per-slot features for per-player contribution

### 04b — Causal Transformer (per-minute sequence)
- Input: sequence of feature vectors per game, shape (T, 482)
- Positional encoding: **minute-indexed sinusoidal** (encodes actual game minute, not position index)
- Mask: upper-triangular causal mask (minute T cannot see T+1)
- Output: win probability at every minute
- GPU flags: AMP (mixed precision), torch.compile, `--d-model 256 --nhead 8 --num-layers 6` for GPU

### 04c — Player Context Transformer (game-summary history)
- **PlayerHistoryEncoder:** for each of 10 players, look up their K most recent past games (end-of-game summary stats from `player_game_summary.parquet`) → non-causal Transformer → mean-pool → player_embedding
- **Game predictor:** [10 player_embedding tokens | per-minute game frame tokens] → Transformer with **hybrid causal mask**
  - Player tokens: see all other player tokens (not game tokens)
  - Game tokens: see all player tokens + causally past game tokens
- Output: win probability at every game minute

### 04d — Full Minute-Level Player Context (3 levels)
- **Level 1 — Within-game encoder** (shared weights): one past game's per-minute sequence → causal Transformer → last real token = game_embedding (dim=64)
- **Level 2 — Cross-game encoder** (shared weights): K game_embeddings → non-causal Transformer → mean-pool = player_embedding (dim=64)
- **Level 3 — Game predictor:** same as 04c but player_embeddings come from minute-level encoding
- `--no-sequences`: runs with zero-padded player embeddings to test architecture without the sequence parquet
- `--benchmark`: reports forward pass ms/batch, VRAM usage, estimated training time

### Key design decisions (apply to all models)
1. **GroupKFold by game_id** — never split rows of the same game into different train/val sets
2. **`game_duration_min` is metadata only** — NOT a model feature (would leak game length)
3. **Left-aligned padding** with causal mask — no `src_key_padding_mask` needed (real tokens always left-aligned, padding is on the right, causal mask prevents attending forward into padding)
4. **NaN guard** in player context: when all K history slots are padding, expose first slot to prevent softmax NaN; zero mean-pool gives zero embedding
5. **SHAP attribution convention**: signed_contrib = shap_value × team_sign (+1 blue, −1 red)
6. **Slot names**: `blue_top, blue_jungle, blue_middle, blue_bottom, blue_utility, red_top, red_jungle, red_middle, red_bottom, red_utility`

---

## 8. Data Facts

| Stat | Value |
|------|-------|
| Total games | 133,347 |
| Regions | NA1, EUW1, KR |
| ELO | Challenger + Grandmaster only |
| Queue | Ranked Solo/Duo (queue 420) |
| Feature rows | 3.59M (one per game × minute) |
| Feature columns | 482 (including delta/rate features) |
| Unique players | 43,447 |
| Avg games per player | 30.7 (in player_game_summary) |
| Label | `blue_win` (binary: 1 = blue wins) |

### Delta / rate features (added in last session)
Per player slot (×10): `gold_delta_1m`, `gold_delta_3m`, `xp_delta_1m`, `cs_delta_1m`, `dmg_delta_1m`, `gold_per_min`, `cs_per_min`
Team diffs: `diff_{role}_gold_delta_1m` (×5), `gold_diff_delta_1m`, `gold_diff_delta_3m`
Total ~77 new columns added by `03_process_features.py` → now 482 cols total.

---

## 9. Known Benchmarks

| Model | Dataset | AUC |
|-------|---------|-----|
| 04a LightGBM | 30K games (old run) | OOF AUC 0.8163 |
| 04b Causal Transformer | 10K games (quick test, 5 epochs) | val AUC 0.8144 |
| Early-game AUC weakness | min 0-5 | ~0.63 (player context will help most here) |

---

## 10. Riot API / Data Harvesting

The `.env` file at the project root contains the Riot API key:
```
RIOT_API_KEY=RGAPI-xxxx-...
```

**Note:** Riot API keys expire every 24 hours for development keys. If harvesting more data, you'll need a fresh key from https://developer.riotgames.com/

### Harvesting more data

```bash
# Harvest new games (resumes from where it left off)
python src/02_bulk_harvest.py

# Back-fill player context games (fetch K prior games per player)
python src/02_bulk_harvest.py --context-pass --context-depth 20
```

Context games are stored in:
- `data/raw/context_matches/` — match JSONs
- `data/raw/context_timelines/` — timeline JSONs
- Sentinel files: `ctx_done_{puuid}.flag` (prevents re-fetching on restart)

After harvesting more data, re-run the processing pipeline:
```bash
python src/03_process_features.py      # regenerates features.parquet
python src/03b_build_player_index.py   # regenerates player_game_summary.parquet
python src/03b_build_player_index.py --include-sequences  # (optional, large)
```

---

## 11. GPU Training Tips

### Recommended CLI flags for GPU workstation

```bash
# 04b — large Transformer
python src/04b_train_timeseries.py --d-model 256 --nhead 8 --num-layers 6 --epochs 30 --batch-size 64

# 04c — player context
python src/04c_train_player_context.py --epochs 30 --k 20 --batch-size 32

# 04d — full minute-level (benchmark first)
python src/04d_train_player_context_minute.py --benchmark --k 50
python src/04d_train_player_context_minute.py --epochs 30 --k 50 --batch-size 16
```

### What happens automatically on CUDA
- AMP (mixed precision) enabled via `torch.amp.autocast`
- `torch.compile()` applied to model (PyTorch 2.0+, ~2× speedup)
- GradScaler enabled for fp16 gradient stability
- GPU memory reported in benchmark mode

### Monitor GPU during training
```bash
watch -n 1 nvidia-smi
```

---

## 12. Next Steps After Full Training

1. **Run SHAP analysis** on the best model: `python src/06_shap_explain.py`
2. **Analyse results**: `python src/07_analysis.py` — role-level importance, champion contribution plots
3. **Compare models** — which of 04a/04b/04c/04d gives the best early-game AUC? (Early game is where player skill matters most and where SHAP contributions are most discriminating)
4. **Scale data** — harvest more games (target: 1M+ games for GPU training), re-run pipeline
5. **Harvest context games** — `--context-pass --context-depth 20` to give 04c/04d deeper player history
6. **Build inference API** — given a live game_id, fetch timeline, build features, run model, compute SHAP per player, return contribution scores

---

## 13. Common Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA: False` after installing torch | Reinstall with correct CUDA wheel (see §2b) |
| `UnicodeEncodeError` in logs | Windows terminal encoding artifact — safe to ignore, or pipe output to a file |
| OOM during 04d training | Reduce `--batch-size` or `--k` or `--max-seq-len` |
| Missing `player_minute_sequences.parquet` | Run `python src/03b_build_player_index.py --include-sequences` |
| Riot API 429 rate limit | `02_bulk_harvest.py` already has retry logic with backoff |
| Script says "file exists, overwrite?" | Type `y` to confirm overwrite of existing processed files |
| `torch.compile` error on older PyTorch | Upgrade to PyTorch ≥ 2.0, or the code will fall back gracefully |

---

## 14. Starter Prompt for the GPU Workstation Agent

You are picking up a machine learning project that has been transferred to this GPU workstation
from a development laptop. The full context is documented in agent_handoff.md at the root of
the project directory. Please read that file first before doing anything else.

Here is your onboarding checklist — work through it in order:

1. READ agent_handoff.md in full. It contains the project goal, all architecture decisions,
   what has been done, and what needs to be done next.

2. ENVIRONMENT CHECK:
   a. Run `conda env list` — look for lol_shap_env.
   b. If it exists, test whether PyTorch has CUDA support (command in §2 of the handoff).
   c. If CUDA is False or the env is missing, follow §2a/2b of the handoff to create/fix it.
   d. Confirm GPU is detected: nvidia-smi should show the RTX 3090 or 4090.

3. DATA CHECK:
   Run the data verification command from §2 Step 3 of the handoff to confirm both parquet
   files are intact and have the expected number of rows and columns.

4. API TEST:
   Create and run the test_api.py script from §2 Step 4 of the handoff to confirm the Riot
   API key in .env is valid and the full harvesting chain works (ladder → PUUID → match →
   timeline). If the key is expired, a new one must be generated at
   https://developer.riotgames.com/ and updated in .env.

5. SMOKE TESTS:
   Run each of the quick smoke-test commands from §2 Step 5 of the handoff (04a, 04b, 04c,
   04d --benchmark). All should complete without errors. Check logs/ if any fail.

6. FULL TRAINING:
   Once everything checks out, proceed with the full training run order from §6 of the
   handoff. Start with 04a (LightGBM) and 04b (Transformer), then 04c (player context,
   game-summary). Use the GPU-appropriate CLI flags listed in §11.

   Before running 04d (minute-level player context), first generate the per-minute sequence
   file: python src/03b_build_player_index.py --include-sequences

7. Report back with:
   - Which GPU was detected and how much VRAM is available
   - Whether the API key was valid or needed renewal
   - Results of the smoke tests (pass/fail)
   - Any issues found, with the relevant log output
   - Estimated training time from the 04d --benchmark run

