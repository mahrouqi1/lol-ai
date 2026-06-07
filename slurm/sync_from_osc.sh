#!/bin/bash
# Pull results back from OSC to the local LoL_AI repo.
# Run from the local repo root:
#   bash slurm/sync_from_osc.sh

set -euo pipefail

OSC_USER=mahrouqi1
PROJECT=PAS1457
REMOTE=$OSC_USER@sftp.osc.edu:/fs/ess/$PROJECT/$OSC_USER/LoL_AI

LOCAL_ROOT=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$LOCAL_ROOT/models" "$LOCAL_ROOT/reports" "$LOCAL_ROOT/logs"

echo "==> Pulling models/ reports/ logs/ back from $REMOTE"
rsync -avz --progress "$REMOTE/models/"  "$LOCAL_ROOT/models/"  || true
rsync -avz --progress "$REMOTE/reports/" "$LOCAL_ROOT/reports/" || true
rsync -avz --progress "$REMOTE/logs/"    "$LOCAL_ROOT/logs/"    || true
echo "==> Pull complete."
