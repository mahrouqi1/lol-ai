#!/bin/bash
#SBATCH --account=PAS1457
#SBATCH --job-name=lol_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --output=/fs/ess/PAS1457/%u/LoL_AI/logs/%x_%j.out
#SBATCH --error=/fs/ess/PAS1457/%u/LoL_AI/logs/%x_%j.err
#
# CPU job (Pitzer is cheapest for CPU). Default: LightGBM snapshot CV (04a) on the
# full current 478-feature data -> a fresh, consistent snapshot model.
# Est cost: 40 cores x ~1.5h x $0.003 ~= $0.18 (charged for actual use).
# Submit (blessed path):  bash osc_submit.sh slurm/train_cpu.slurm pitzer
#
# Override via TRAIN_SCRIPT/TRAIN_ARGS, e.g. Phase-0 sweep:
#   TRAIN_SCRIPT=src/08_phase0_baseline_divergence.py TRAIN_ARGS="--n-games 0" \
#     bash osc_submit.sh slurm/train_cpu.slurm pitzer

set -euo pipefail

module load pytorch/2.8.0
export PYTHONUSERBASE=/fs/ess/PAS1457/$USER/envs/lol_user
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /fs/ess/PAS1457/$USER/LoL_AI
export LOL_DATA_DIR=/fs/scratch/PAS1457/$USER/LoL_AI/data/processed
echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date)"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-src/04a_train_snapshot.py}"
TRAIN_ARGS="${TRAIN_ARGS:---folds 5}"
echo "==> python $TRAIN_SCRIPT $TRAIN_ARGS"
python $TRAIN_SCRIPT $TRAIN_ARGS
echo "==> Done at $(date)"
