#!/usr/bin/env bash
# Fixture tests for scripts/verify_claude_review.sh — the enforcement logic
# behind the Claude Review Gate. Run: tests/gate/test_verify_claude_review.sh
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
verify="$repo_root/scripts/verify_claude_review.sh"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
reviews="$work/reviews"
mkdir -p "$reviews"

pass=0
fail=0

# run <expected-exit> <pr> <label> [reviews-dir]
run() {
  local want="$1" pr="$2" label="$3" dir="${4:-$reviews}"
  "$verify" "$pr" "$dir" >/dev/null 2>&1
  local got=$?
  if [[ "$got" == "$want" ]]; then
    echo "ok   - $label (exit $got)"
    pass=$((pass + 1))
  else
    echo "FAIL - $label (want exit $want, got $got)"
    fail=$((fail + 1))
  fi
}

# A well-formed artifact with a substantive body.
good_artifact() { # <pr> <pull-url-number>
  local pr="$1" urlnum="$2"
  {
    echo "# Claude Review: PR #${pr}"
    echo
    echo "- PR: https://github.com/KestrelSovereignAI/kestrel-llms/pull/${urlnum}"
    echo "- Title: something"
    echo "- Reviewed: 2026-07-06T00:00:00Z"
    echo
    echo "## Review"
    for i in 1 2 3 4 5 6 7 8 9 10; do echo "Finding line ${i}: a real observation."; done
  } > "$reviews/claude-pr-${pr}.md"
}

# 1. valid → 0
good_artifact 20 20
run 0 20 "valid artifact passes"

# 2. missing file → 1
run 1 999 "missing artifact fails"

# 3. wrong header PR number → 1
good_artifact 21 21
sed -i.bak 's/# Claude Review: PR #21/# Claude Review: PR #999/' "$reviews/claude-pr-21.md" && rm -f "$reviews/claude-pr-21.md.bak"
run 1 21 "mismatched header fails"

# 4. missing PR url → 1
good_artifact 22 22
grep -v '/pull/' "$reviews/claude-pr-22.md" > "$reviews/claude-pr-22.md.tmp" && mv "$reviews/claude-pr-22.md.tmp" "$reviews/claude-pr-22.md"
run 1 22 "missing PR url fails"

# 5. empty body (header + metadata only) → 1, AND prints the friendly diagnostic
#    (not a bare set -e abort).
{
  echo "# Claude Review: PR #23"
  echo
  echo "- PR: https://github.com/KestrelSovereignAI/kestrel-llms/pull/23"
  echo "- Title: stub"
  echo
} > "$reviews/claude-pr-23.md"
run 1 23 "empty-body stub fails"
out23="$("$verify" 23 "$reviews" 2>&1 || true)"
if grep -q "body line(s)" <<<"$out23"; then
  echo "ok   - empty-body stub prints the actionable diagnostic"
  pass=$((pass + 1))
else
  echo "FAIL - empty-body stub aborted before its diagnostic: ${out23}"
  fail=$((fail + 1))
fi

# 6. url boundary: PR #1 must not be satisfied by a /pull/13 link → 1
{
  echo "# Claude Review: PR #1"
  echo
  echo "- PR: https://github.com/KestrelSovereignAI/kestrel-llms/pull/13"
  echo
  echo "## Review"
  for i in 1 2 3 4 5 6 7 8 9 10; do echo "body ${i}"; done
} > "$reviews/claude-pr-1.md"
run 1 1 "url boundary rejects /pull/13 for PR #1"

# 7. non-numeric PR arg → 2
run 2 abc "non-numeric pr arg errors"

echo
echo "passed=$pass failed=$fail"
[[ "$fail" == 0 ]]
