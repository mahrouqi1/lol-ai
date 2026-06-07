#!/usr/bin/env bash
# .claude/hooks/post-edit-format.sh
#
# Post-tool-use auto-formatter. Fires on Edit/Write/MultiEdit. Silent
# on failure (formatters may not be installed in every project's env).
#
# Hook contract:
#   stdin:  {"tool_name":"Edit"|"Write"|"MultiEdit",
#            "tool_input":{"file_path":"..."}, ...}
#   exit 0: continue (format-or-skip is best-effort)

set -euo pipefail

_format_one() {
  local f="$1"
  [ -f "$f" ] || return 0

  case "$f" in
    *.py)
      command -v ruff >/dev/null 2>&1 && ruff format "$f" 2>/dev/null || true
      command -v ruff >/dev/null 2>&1 && ruff check --fix "$f" 2>/dev/null || true
      ;;
    *.tex)
      command -v latexindent >/dev/null 2>&1 && latexindent -w "$f" 2>/dev/null || true
      ;;
    *.json)
      # Validate but don't reformat (preserves intentional layout).
      command -v jq >/dev/null 2>&1 && jq empty "$f" 2>/dev/null || true
      ;;
    *.sh)
      command -v shellcheck >/dev/null 2>&1 && shellcheck "$f" 2>/dev/null || true
      ;;
  esac
}

input="$(cat)"
file="$(echo "$input" | jq -r '.tool_input.file_path // .tool_input.path // empty')"

if [ -n "$file" ]; then
  _format_one "$file"
fi

# Some edit tools provide an array of file_paths instead of a single path.
files="$(echo "$input" | jq -r '.tool_input.edits[]?.file_path // empty' 2>/dev/null || true)"
if [ -n "$files" ]; then
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    _format_one "$f"
  done <<< "$files"
fi

exit 0
