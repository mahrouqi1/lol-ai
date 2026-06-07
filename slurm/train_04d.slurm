#!/bin/bash
#SBATCH --account=PAS1457
#SBATCH --job-name=lol_04d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gpus-per-node=1
#SBATCH --time=16:00:00
#SBATCH --output=/fs/ess/PAS1457/%u/LoL_AI/logs/%x_%j.out
#SBATCH --error=/fs/ess/PAS1457/%u/LoL_AI/logs/%x_%j.err
#
# 04d minute-level player-context (3-level Transformer hierarchy), full data.
# Most expensive model. Needs player_minute_sequences.parquet on scratch.
# 1xA100, early-stopping (patience 5). Est <=$1.44 (billed for actual).
# Submit:  bash osc_submit.sh slurm/train_04d.slurm ascend

set -euo pipefail
module load pytorch/2.8.0
module load cuda/12.8.1
export PYTHONUSERBASE=/fs/ess/PAS1457/$USER/envs/lol_user
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONUNBUFFERED=1
cd /fs/ess/PAS1457/$USER/LoL_AI
export LOL_DATA_DIR=/fs/scratch/PAS1457/$USER/LoL_AI/data/processed
echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date)"; nvidia-smi || true
python src/04d_train_player_context_minute.py --epochs 15 --k 20 --batch 16 --max-seq-len 27
echo "==> Done at $(date)"
