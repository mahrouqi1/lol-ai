#!/bin/bash
# One-time environment setup on OSC for LoL_AI.
#
# Strategy: use OSC's pytorch/2.8.0 module (torch 2.8.0+cu128 + numpy, scipy,
# sklearn, matplotlib, pandas, tqdm) and layer the LoL-specific extras on top
# via a user-pip overlay at /fs/ess/PAS1457/mahrouqi1/envs/lol_user.
# This reuses OSC's CUDA-tuned torch and avoids a multi-GB conda build.
#
# Extras NOT in the pytorch module: lightgbm, xgboost, shap, seaborn, pyarrow,
# riotwatcher, python-dotenv.
#
# Run on an OSC login node (whichever cluster you'll train on):
#   ssh mahrouqi1@ascend.osc.edu
#   bash /fs/ess/PAS1457/mahrouqi1/LoL_AI/slurm/setup_env.sh

set -euo pipefail

OVERLAY=/fs/ess/PAS1457/mahrouqi1/envs/lol_user
PROJECT_DIR=/fs/ess/PAS1457/mahrouqi1/LoL_AI
SCRATCH_DIR=/fs/scratch/PAS1457/mahrouqi1/LoL_AI

mkdir -p "$OVERLAY"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$SCRATCH_DIR/data/processed"

module load pytorch/2.8.0
module load cuda/12.8.1

export PYTHONUSERBASE=$OVERLAY

echo "==> torch sanity check"
python -c "import torch; print('torch', torch.__version__, 'cuda built:', torch.version.cuda)"

echo "==> Installing LoL extras into $OVERLAY"
pip install --user --no-build-isolation \
  lightgbm xgboost shap seaborn pyarrow riotwatcher python-dotenv

echo "==> extras sanity check"
python -c "import lightgbm, xgboost, shap, pyarrow, seaborn; print('extras OK')"

echo "==> Done. Add these two lines to any SLURM script after 'module load':"
echo "    export PYTHONUSERBASE=$OVERLAY"
echo "    export PATH=\$PYTHONUSERBASE/bin:\$PATH"
