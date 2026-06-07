#!/bin/bash
# Push local LoL_AI repo state to OSC project space.
# Run from the local repo root:
#   bash slurm/sync_to_osc.sh
#
# Excludes git, caches, secrets, and large blobs. NOTE: data/ is excluded by
# default because features.parquet is 2.2 GB. The processed parquets are NOT
# re-downloadable on OSC (unlike public benchmark datasets), so the FIRST time
# you run on OSC you must push the processed data explicitly — see the
# one-liner at the bottom of this script. Raw JSONs (130k+ files) should stay
# on the workstation; only push processed parquets when a job needs them.

set -euo pipefail

OSC_USER=mahrouqi1
PROJECT=PAS1457
REMOTE=$OSC_USER@sftp.osc.edu:/fs/ess/$PROJECT/$OSC_USER/LoL_AI/

LOCAL_ROOT=$(cd "$(dirname "$0")/.." && pwd)

echo "==> rsync $LOCAL_ROOT/ -> $REMOTE"
rsync -avz --progress \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='.env' \
  --exclude='*.png' \
  --exclude='*.gif' \
  --exclude='*.pdf' \
  --exclude='data/' \
  --exclude='models/' \
  --exclude='reports/' \
  --exclude='logs/' \
  --exclude='.smoke/' \
  "$LOCAL_ROOT/" "$REMOTE"

echo "==> Code sync complete."
echo
echo "First-time / when a job needs the processed data, push it explicitly:"
echo "  rsync -avz --progress data/processed/ \\"
echo "    $OSC_USER@sftp.osc.edu:/fs/scratch/$PROJECT/$OSC_USER/LoL_AI/data/processed/"
echo "(scratch, not ess: 2.2 GB features.parquet belongs on high-I/O scratch.)"
