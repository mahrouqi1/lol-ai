#!/usr/bin/env bash
# .claude/hooks/session-start.sh
#
# Logs session start to today's journal AND injects host/dir context
# into the chat as additionalContext (so the chat knows which machine
# it is on; revision 4 from framework v0.3 design).
#
# Hook contract:
#   stdin: minimal JSON (event metadata)
#   exit 0 with JSON output adds context to the model's first turn

set -euo pipefail

proj="$(basename "${CLAUDE_PROJECT_DIR:-$(pwd)}")"
host="$(hostname -s 2>/dev/null || echo unknown)"
day="$(date +%F)"
journal_dir="${RESEARCH_ROOT:-$HOME/research}/_journal"
journal="$journal_dir/${day}.md"

mkdir -p "$journal_dir" 2>/dev/null || true
echo "[$(date '+%F %T')] session-start in $proj on $host" >> "$journal" 2>/dev/null || true

# Detect kernel and user — useful when running over Remote-SSH between
# Windows PC and Linux workstation.
kernel="$(uname -s 2>/dev/null || echo unknown)"

cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "Running on host: $host (kernel: $kernel). Project dir: ${CLAUDE_PROJECT_DIR:-$(pwd)}. Today: $day. If host is unexpected (e.g. user is on the PC but expected the workstation), pause and confirm before doing anything that touches the file system."
  }
}
EOF
exit 0
