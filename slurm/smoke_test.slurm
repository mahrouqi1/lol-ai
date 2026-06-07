#!/bin/bash
#SBATCH --account=PAS1457
#SBATCH --job-name=lol_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/fs/ess/PAS1457/%u/LoL_AI/logs/%x_%j.out
#SBATCH --error=/fs/ess/PAS1457/%u/LoL_AI/logs/%x_%j.err
#
# ~20 min smallest allocation that proves env + GPU + data path + logging work.
# Est cost: 1 GPU x ~0.33 h x $0.09 ~= $0.03.
# Submit (blessed path):  bash osc_submit.sh slurm/smoke_test.slurm ascend

set -euo pipefail

module load pytorch/2.8.0
module load cuda/12.8.1
export PYTHONUSERBASE=/fs/ess/PAS1457/$USER/envs/lol_user
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONUNBUFFERED=1

cd /fs/ess/PAS1457/$USER/LoL_AI
echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi || true
python -c "import torch; print('cuda available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import lightgbm, shap, pyarrow; print('extras OK')"

# Point the scripts at the scratch copy of the processed data.
export LOL_DATA_DIR=/fs/scratch/PAS1457/$USER/LoL_AI/data/processed
echo "==> data dir: $LOL_DATA_DIR"
ls -lh "$LOL_DATA_DIR" || echo "WARNING: processed data not yet pushed to scratch (see slurm/sync_to_osc.sh)"

# Tiny end-to-end: equivariant GNN on a small slice (LOL_DATA_DIR-aware).
python src/04e_train_gnn.py --limit 2000 --epochs 3
echo "==> Smoke complete at $(date)"
