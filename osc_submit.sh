#!/bin/bash
# Canonical OSC submit wrapper (framework template — keep all repos identical).
# One-liner: sync the local repo to OSC, then submit a SLURM job there.
#
# This is the BLESSED submission path: it passes the pre-bash-guard hook
# because the literal token "sbatch" never appears in the command you run
# (see conventions/osc.md "Submitting jobs — the blessed path"). Always run
# /osc-submit-dryrun and get the user's OK first.
#
# Usage:
#   bash osc_submit.sh slurm/smoke_test.slurm            # default cluster=ascend
#   bash osc_submit.sh slurm/gpu_single.slurm cardinal
#   bash osc_submit.sh slurm/cpu.slurm pitzer
#
# Forward env vars to the remote sbatch by exporting them locally:
#   CONFIG=configs/foo.yaml SMOKE=1 bash osc_submit.sh slurm/train.slurm
#
# Requires: ssh-key auth to mahrouqi1@<cluster>.osc.edu, and a
# slurm/sync_to_osc.sh in this repo (repo-specific rsync; not templated).

set -euo pipefail

SCRIPT=${1:?Usage: bash osc_submit.sh <slurm_script_path> [cluster=ascend]}
CLUSTER=${2:-ascend}

OSC_USER=mahrouqi1
PROJECT=PAS1457
REMOTE_DIR=/fs/ess/$PROJECT/$OSC_USER/$(basename "$(pwd)")

LOCAL_ROOT=$(cd "$(dirname "$0")" && pwd)
SCRIPT_REL=${SCRIPT#"$LOCAL_ROOT/"}

# Forward chat/user-supplied env vars to the remote sbatch via --export.
# ALL keeps the remote environment (PATH, LMOD_*, ...); named vars are added
# when set. Extend the list as SLURM templates start consuming new vars.
FORWARDED=(ALL)
# NOTE: only forward SINGLE-TOKEN vars. Multi-word values (e.g. TRAIN_ARGS with
# spaces) word-split through `sbatch --export` over ssh and break submission —
# bake training args into a per-model slurm/train_<model>.slurm instead.
for v in CONFIG SMOKE SLUG; do
  [ -n "${!v:-}" ] && FORWARDED+=("$v=${!v}")
done
EXPORT_ARG=$(IFS=,; echo "${FORWARDED[*]}")

echo "==> [1/3] Syncing local repo to $OSC_USER@$CLUSTER.osc.edu:$REMOTE_DIR/"
bash "$LOCAL_ROOT/slurm/sync_to_osc.sh"

echo "==> [2/3] Submitting $SCRIPT_REL on $CLUSTER (export: $EXPORT_ARG)"
JOB_OUTPUT=$(ssh "$OSC_USER@$CLUSTER.osc.edu" \
  "cd $REMOTE_DIR && sbatch --cluster=$CLUSTER --export=$EXPORT_ARG $SCRIPT_REL")
echo "$JOB_OUTPUT"

JOB_ID=$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+' | tail -1)
echo "==> [3/3] Job id: $JOB_ID"
echo "Monitor:    ssh $OSC_USER@$CLUSTER.osc.edu 'squeue -u $OSC_USER --cluster=$CLUSTER'"
echo "Live log:   ssh $OSC_USER@$CLUSTER.osc.edu 'tail -F $REMOTE_DIR/logs/*_${JOB_ID}.out'"
echo "Pull back:  bash slurm/sync_from_osc.sh"
