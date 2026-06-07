#!/bin/bash
#SBATCH --account=PAS1457
#SBATCH --job-name=lol_04f
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/fs/ess/PAS1457/%u/LoL_AI/logs/%x_%j.out
#SBATCH --error=/fs/ess/PAS1457/%u/LoL_AI/logs/%x_%j.err
#
# 04f equivariant GNN + game-level player-history context, full data. 1xA100.
# Needs features.parquet + player_game_summary.parquet on scratch (both present).
# Small model, data-bound; est <=$0.40.
# Submit:  bash osc_submit.sh slurm/train_04f.slurm ascend

set -euo pipefail
module load pytorch/2.8.0
module load cuda/12.8.1
export PYTHONUSERBASE=/fs/ess/PAS1457/$USER/envs/lol_user
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONUNBUFFERED=1
cd /fs/ess/PAS1457/$USER/LoL_AI
export LOL_DATA_DIR=/fs/scratch/PAS1457/$USER/LoL_AI/data/processed
echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date)"; nvidia-smi || true
python src/04f_train_gnn_context.py --epochs 30 --k 20 --d 128 --layers 4 --batch-size 4096 --ctx-d 48
echo "==> Done at $(date)"
