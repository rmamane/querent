#!/usr/bin/env bash
# Zero-secrets gate for a PUBLIC repo: grep tracked + staged files for
# credential-shaped strings. Run before every push (also wired into Makefile).
set -uo pipefail

cd "$(dirname "$0")/.."

PATTERNS=(
  'WANDB_API_KEY[[:space:]]*[=:][[:space:]]*[A-Za-z0-9]'   # assigned, not just referenced
  'api[_-]?key[[:space:]]*[=:][[:space:]]*["'\'']?[A-Za-z0-9_\-]\{16,\}'
  'Authorization:[[:space:]]*Bearer[[:space:]]+[A-Za-z0-9]'
  '-----BEGIN (RSA|EC|OPENSSH|DSA) PRIVATE KEY-----'
  'ghp_[A-Za-z0-9]\{20,\}'      # GitHub PAT
  'sk-[A-Za-z0-9]\{20,\}'       # generic secret-key shape
  'AKIA[0-9A-Z]\{16\}'          # AWS access key id
)

FILES=$(git ls-files; git diff --cached --name-only)
STATUS=0
for pat in "${PATTERNS[@]}"; do
  HITS=$(echo "$FILES" | sort -u | xargs grep -InE "$pat" 2>/dev/null | grep -v "check_no_secrets.sh" || true)
  if [ -n "$HITS" ]; then
    echo "POTENTIAL SECRET (pattern: $pat):"
    echo "$HITS"
    STATUS=1
  fi
done

if [ $STATUS -eq 0 ]; then
  echo "secrets check: clean"
else
  echo "secrets check: FAILED — remove the above before pushing (public repo!)"
fi
exit $STATUS
