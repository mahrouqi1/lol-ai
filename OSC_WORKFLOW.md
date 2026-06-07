# OSC Workflow — LoL_AI

Project-specific quickstart for running LoL win-contribution experiments on OSC.
For the general OSC reference (storage, modules, pricing, troubleshooting) see
[.framework/_docs/osc-access.md](.framework/_docs/osc-access.md) (framework-level, lab-wide).

- **Account:** `PAS1457` (research, PI shafieezadeh.1).
- **OSC user:** `mahrouqi1`
- **Project space (code):** `/fs/ess/PAS1457/mahrouqi1/LoL_AI/`
- **Scratch (big data):** `/fs/scratch/PAS1457/mahrouqi1/LoL_AI/data/processed/`
- **User-pip overlay:** `/fs/ess/PAS1457/mahrouqi1/envs/lol_user/`
- **Default cluster:** Ascend (A100). Cardinal (H100) for the single heaviest
  training job; Pitzer (CPU) for LightGBM CV, SHAP, and data processing.

## When to use OSC vs the workstation

The workstation has **2× RTX 4090 (24 GB each)** and the conda env `lol_shap_env`.
Prefer it for: Phase 0, prototyping, LightGBM, and current-scale (133k-game)
transformer training — all fit comfortably. Reach for OSC when:

- **The equivariant temporal GNN** grows past what fits / what's fast on a 4090.
- **The exact 32-coalition Shapley sweep** (~`32 × N × T × 10` forward passes
  per game) needs to run over many games — embarrassingly parallel, ideal for
  several single-GPU OSC jobs in parallel (or Pitzer CPU if the model is small).
- **Data scale-up** to ~1M games (heavier feature processing — Pitzer hugemem).
- You want to sweep hyperparameters concurrently instead of serially.

## One-time bootstrap

```bash
ssh mahrouqi1@ascend.osc.edu
mkdir -p /fs/ess/PAS1457/mahrouqi1/LoL_AI/logs
mkdir -p /fs/scratch/PAS1457/mahrouqi1/LoL_AI/data/processed
exit
# from local repo root:
bash slurm/sync_to_osc.sh
# push the processed parquets to scratch (NOT re-downloadable on OSC):
rsync -avz --progress data/processed/ \
  mahrouqi1@sftp.osc.edu:/fs/scratch/PAS1457/mahrouqi1/LoL_AI/data/processed/
# back on OSC, build the env overlay:
ssh mahrouqi1@ascend.osc.edu 'bash /fs/ess/PAS1457/mahrouqi1/LoL_AI/slurm/setup_env.sh'
```

## Daily loop

```bash
# 1. Push code changes
bash slurm/sync_to_osc.sh

# 2. Dry-run the job first (REQUIRED — parses #SBATCH, estimates cost):
#    /osc-submit-dryrun slurm/train_gpu.slurm ascend

# 3. Submit (blessed path; never type `sbatch` directly — the guard hook
#    blocks it and the deny rule in settings.json forbids it):
bash osc_submit.sh slurm/smoke_test.slurm ascend
TRAIN_SCRIPT=src/04c_train_player_context.py TRAIN_ARGS="--epochs 30 --k 20" \
  bash osc_submit.sh slurm/train_gpu.slurm ascend

# 4. Monitor (JOBID printed by step 3):
ssh mahrouqi1@ascend.osc.edu 'squeue -u mahrouqi1 --cluster=ascend'
ssh mahrouqi1@ascend.osc.edu 'tail -F /fs/ess/PAS1457/mahrouqi1/LoL_AI/logs/*_<JOBID>.out'

# 5. Pull results back
bash slurm/sync_from_osc.sh
```

## Cost guardrails

- Lab credit is **$1,000/yr, shared, no rollover.** Check before anything big:
  `ssh mahrouqi1@ascend.osc.edu 'OSCusage -P PAS1457'`.
- Right-size: `--gpus-per-node=1` unless the script truly does DDP. Never
  `--exclusive`. You are billed for what you **request**, not what you use.
- Rough rates: GPU-hour $0.09, CPU-core-hour $0.003. The provided
  `smoke_test.slurm` ≈ $0.03; `train_gpu.slurm` (12 h, 1 GPU) ≈ $1.08.
- Always confirm the plan + estimated cost with Mazin before submitting
  (see CLAUDE.md "Always confirm before doing").
