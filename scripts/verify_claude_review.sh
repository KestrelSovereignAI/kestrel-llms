#!/usr/bin/env bash
# Verify that a committed Claude review artifact exists for a PR and carries a
# substantive review body. Reads one markdown file; executes nothing from it.
# Used by .github/workflows/claude-review-gate.yml and covered by
# tests/gate/test_verify_claude_review.sh.
#
# Usage: scripts/verify_claude_review.sh <pr-number> [reviews-dir]
#   reviews-dir defaults to docs/code_reviews
set -euo pipefail

# Minimum non-metadata, non-blank body lines a real review must contain. Blocks
# an empty stub (header-only) or a hand-written 3-line file from passing. It does
# NOT prove a genuine review occurred (see CLAUDE_PR_REVIEW.md) — CI cannot,
# since it deliberately never runs Claude — but it forces real content to exist.
MIN_BODY_LINES="${MIN_BODY_LINES:-8}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: scripts/verify_claude_review.sh <pr-number> [reviews-dir]" >&2
  exit 2
fi

pr="$1"
reviews_dir="${2:-docs/code_reviews}"

if ! [[ "$pr" =~ ^[0-9]+$ ]]; then
  echo "::error::PR number must be a positive integer, got: ${pr}" >&2
  exit 2
fi

file="${reviews_dir}/claude-pr-${pr}.md"

if [[ ! -f "$file" ]]; then
  echo "::error::Missing Claude review artifact: ${file}"
  echo "Run 'scripts/claude_pr_review.sh ${pr}' locally, then commit ${file}."
  echo "See docs/code_reviews/CLAUDE_PR_REVIEW.md for the review policy."
  exit 1
fi

# Header must name this exact PR (guards against copying a stale review).
# Tolerate trailing whitespace / a CR (CRLF-committed artifact) so the failure
# mode is "wrong PR", not a baffling exact-match miss on an invisible \r.
if ! grep -qE "^# Claude Review: PR #${pr}[[:space:]]*$" "$file"; then
  echo "::error::${file} header must be '# Claude Review: PR #${pr}'"
  exit 1
fi

# Body must link back to this PR's canonical URL. The trailing boundary stops
# /pull/13 from matching /pull/130.
if ! grep -qE "/pull/${pr}([^0-9]|$)" "$file"; then
  echo "::error::${file} must reference this PR's URL (.../pull/${pr})"
  exit 1
fi

# Substance check: strip the header line, the '- Key:' metadata block, and blank
# lines, then require at least MIN_BODY_LINES of real content. A failed local
# review that left only the metadata header will have zero and fail here.
# `|| true`: when grep -v selects zero lines it exits 1; without this the
# command substitution would trip `set -e` and abort before the friendly
# diagnostic below can print.
body_lines="$(
  { grep -vE '^# Claude Review: PR #|^- (PR|Title|Author|Base|Head|Reviewed):|^[[:space:]]*$' "$file" \
    | wc -l | tr -d '[:space:]'; } || true
)"
if (( body_lines < MIN_BODY_LINES )); then
  echo "::error::${file} has only ${body_lines} body line(s); need >= ${MIN_BODY_LINES}."
  echo "The artifact looks empty or truncated — re-run scripts/claude_pr_review.sh ${pr}."
  exit 1
fi

echo "OK: Claude review artifact present with ${body_lines} body line(s) for PR #${pr}: ${file}"
