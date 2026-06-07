#!/bin/bash
#SBATCH --account=PAS1457
#SBATCH --job-name=lol_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/fs/ess/PAS1457/%u/LoL_AI/logs/%x_%j.out
#SBATCH --error=/fs/ess/PAS1457/%u/LoL_AI/logs/%x_%j.err
#
# Single-A100 full training. Est cost: 1 GPU x 12 h x $0.09 ~= $1.08 (less if it
# exits early). Right-sized: 1 GPU, 8 cores. Do NOT raise --gpus-per-node unless
# the script actually does DDP. Run /osc-submit-dryrun first.
#
# Pick the model to train via TRAIN_SCRIPT / TRAIN_ARGS (forwarded by osc_submit
# through --export, or just edit below). Examples:
#   04b causal transformer:  src/04b_train_timeseries.py --d-model 256 --nhead 8 --num-layers 6 --epochs 30
#   04c player-context:      src/04c_train_player_context.py --epochs 30 --k 20
#   04d minute-context:      src/04d_train_player_context_minute.py --epochs 30 --k 50 --batch-size 16
#
# Submit (blessed path):  bash osc_submit.sh slurm/train_gpu.slurm ascend
# Cardinal (H100) for the heaviest single job:  ... slurm/train_gpu.slurm cardinal

set -euo pipefail

module load pytorch/2.8.0
module load cuda/12.8.1
export PYTHONUSERBASE=/fs/ess/PAS1457/$USER/envs/lol_user
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONUNBUFFERED=1

cd /fs/ess/PAS1457/$USER/LoL_AI
export LOL_DATA_DIR=/fs/scratch/PAS1457/$USER/LoL_AI/data/processed

echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi || true

TRAIN_SCRIPT="${TRAIN_SCRIPT:-src/04e_train_gnn.py}"
TRAIN_ARGS="${TRAIN_ARGS:---epochs 30 --batch-size 4096 --d 128 --layers 4}"
echo "==> python $TRAIN_SCRIPT $TRAIN_ARGS"
python $TRAIN_SCRIPT $TRAIN_ARGS

echo "==> Training complete at $(date)"
