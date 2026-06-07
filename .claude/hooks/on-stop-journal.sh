#!/usr/bin/env bash
# .claude/hooks/on-stop-journal.sh
#
# Append a session-stop entry to today's journal. Best-effort; never
# blocks the chat.

set -euo pipefail

proj="$(basename "${CLAUDE_PROJECT_DIR:-$(pwd)}")"
day="$(date +%F)"
journal_dir="${RESEARCH_ROOT:-$HOME/research}/_journal"
journal="$journal_dir/${day}.md"
mkdir -p "$journal_dir" 2>/dev/null || exit 0

{
  echo ""
  echo "## $(date +%H:%M) $proj STOP"
  echo "- session ended in ${CLAUDE_PROJECT_DIR:-$(pwd)}"
  if [ -f "${CLAUDE_PROJECT_DIR:-$(pwd)}/HANDOFF.md" ]; then
    echo "- HANDOFF.md head:"
    head -20 "${CLAUDE_PROJECT_DIR:-$(pwd)}/HANDOFF.md" | sed 's/^/    /'
  fi
} >> "$journal" 2>/dev/null || true

exit 0
