#!/usr/bin/env bash
# .claude/hooks/pre-bash-guard.sh
#
# Pre-tool-use guard for Bash commands. Reads tool-call JSON on stdin
# and decides allow / pass-through / block.
#
# Hook contract (Claude Code v2.x):
#   stdin:  {"tool_name":"Bash","tool_input":{"command":"...", "description":"..."}, ...}
#   exit 0 with no JSON output: pass-through (let normal permission rules decide)
#   exit 0 with JSON {"hookSpecificOutput":{"hookEventName":"PreToolUse",
#                     "permissionDecision":"allow"|"deny",
#                     "permissionDecisionReason":"..."}}: explicit decision
#   exit 2: block; stderr is injected back into Claude's context
#
# This hook fires even under --dangerously-skip-permissions; hook denies
# cannot be bypassed. That is by design.

set -euo pipefail

input="$(cat)"
cmd="$(echo "$input" | jq -r '.tool_input.command // empty')"

# Empty command: pass-through (shouldn't happen, but be safe).
if [ -z "$cmd" ]; then
  exit 0
fi

# ---------- never-allow patterns (block at exit 2) ----------
# These are catastrophic-or-nearly-so. Hook denies cannot be bypassed,
# so they're the last line of defense even in aggressive mode.
deny_patterns=(
  '\brm -rf /(?!tmp/)'              # rm -rf / ... (allow rm -rf /tmp/foo)
  '\brm -rf ~'                       # rm -rf ~ or ~/whatever
  '\brm -rf \$HOME'
  '\bdd if=/dev/'                    # disk dd
  '\bmkfs\.'                         # filesystem create
  '\bsudo\b'                         # any sudo
  '\bcurl [^|]+\| ?(ba)?sh'          # curl ... | bash
  '\bwget [^|]+\| ?(ba)?sh'          # wget ... | bash
  '\bgit push --force'
  '\bgit push -f\b'
  '\bgit reset --hard\b'
  '\bgit clean -fx?d?\b'             # git clean -f, -fd, -fx, -fxd
  '(^|[;&|]|env [A-Z_]+=[^ ]* )\s*sbatch\b'  # sbatch INVOCATION (cmd position) — not a token mentioned inside an echo string
  '\bssh\b[^|]*sbatch\b'             # sbatch over ssh (incl. quoted remote cmd: ssh osc.edu "sbatch ...")
  '(^|[;&|]|env [A-Z_]+=[^ ]* )\s*find [^|]* -delete'  # find -delete INVOCATION (cmd position) — not a token mentioned in prose
  'python[0-9.]*\b[^|]*-c\b[^|]*shutil\.rmtree'        # python -c rmtree INVOCATION (rm-rf evasion) — not a prose mention
  'python[0-9.]*\b[^|]*-c\b[^|]*os\.system\(.*[\x27"]rm '   # python -c os.system rm evasion
  'python[0-9.]*\b[^|]*-c\b[^|]*subprocess\.[a-z]+\(.*[\x27"]rm '  # python -c subprocess rm evasion
)

for pat in "${deny_patterns[@]}"; do
  if echo "$cmd" | grep -Pq "$pat"; then
    echo "pre-bash-guard.sh: blocked command matching /$pat/." >&2
    echo "If intentional, run it manually after review." >&2
    exit 2
  fi
done

# ---------- safe positive list (auto-allow over the "ask" tier) ----------
# These are common dev-loop commands we never want to prompt on. Match
# the start of the command (after possible env-var prefix).
# Smoke entry points handle two forms: dotted module (python -m pkg.smoke)
# and module+arg (python -m pkg smoke).
auto_ok='^(env [A-Z_=][^ ]* )?(pytest|ruff|mypy|black|isort|pre-commit|latexmk|pdflatex|bibtex|biber|tectonic|nvidia-smi|htop|nvtop|free|df|ls|cat|head|tail|grep|find|wc)\b'
auto_ok_smoke='^(env [A-Z_=][^ ]* )?python -m [a-zA-Z_.]+(\.smoke| smoke)\b'

if echo "$cmd" | grep -Pq "$auto_ok" || echo "$cmd" | grep -Pq "$auto_ok_smoke"; then
  cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow",
    "permissionDecisionReason": "auto-approved by pre-bash-guard (test/lint/build/render/inspect)"
  }
}
EOF
  exit 0
fi

# Pass-through: let normal allow/ask/deny rules handle this.
exit 0
