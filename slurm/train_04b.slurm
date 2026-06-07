#!/bin/bash
#SBATCH --account=PAS1457
#SBATCH --job-name=lol_04b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/fs/ess/PAS1457/%u/LoL_AI/logs/%x_%j.out
#SBATCH --error=/fs/ess/PAS1457/%u/LoL_AI/logs/%x_%j.err
#
# 04b causal Transformer, full data. 1xA100. Est <=$1.08 (billed for actual).
# Submit:  bash osc_submit.sh slurm/train_04b.slurm ascend

set -euo pipefail
module load pytorch/2.8.0
module load cuda/12.8.1
export PYTHONUSERBASE=/fs/ess/PAS1457/$USER/envs/lol_user
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONUNBUFFERED=1
cd /fs/ess/PAS1457/$USER/LoL_AI
export LOL_DATA_DIR=/fs/scratch/PAS1457/$USER/LoL_AI/data/processed
echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date)"; nvidia-smi || true
python src/04b_train_timeseries.py --d-model 256 --nhead 8 --num-layers 6 --epochs 20 --batch-size 64
echo "==> Done at $(date)"
