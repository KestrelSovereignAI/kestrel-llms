#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/claude_pr_review.sh <pr-number-or-url>" >&2
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required" >&2
  exit 2
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "Claude CLI is required" >&2
  exit 2
fi

pr="$1"
mkdir -p docs/code_reviews

number="$(gh pr view "$pr" --json number --jq '.number')"
title="$(gh pr view "$pr" --json title --jq '.title')"
author="$(gh pr view "$pr" --json author --jq '.author.login')"
base="$(gh pr view "$pr" --json baseRefName --jq '.baseRefName')"
head="$(gh pr view "$pr" --json headRefName --jq '.headRefName')"
url="$(gh pr view "$pr" --json url --jq '.url')"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

gh pr diff "$pr" > "$tmpdir/diff.patch"
gh pr view "$pr" --json body --jq '.body // ""' > "$tmpdir/body.md"

out="docs/code_reviews/claude-pr-${number}.md"

{
  echo "# Claude Review: PR #${number}"
  echo
  echo "- PR: ${url}"
  echo "- Title: ${title}"
  echo "- Author: ${author}"
  echo "- Base: ${base}"
  echo "- Head: ${head}"
  echo "- Reviewed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
} > "$out"

{
  cat <<PROMPT
You are reviewing a Kestrel Sovereign LLM provider PR.

Review stance:
- Lead with findings, ordered by severity.
- Prioritize bugs, regressions, security/auth problems, provider-contract violations, packaging/release hazards, and missing tests.
- For every finding, cite the relevant file/path from the diff.
- If there are no blocking findings, say so clearly and list residual risks.

Provider-specific checklist:
- Entry-point group is kestrel_sovereign.llm_providers.
- Provider packages depend on kestrel-sovereign-sdk, not kestrel_sovereign.
- Route names, vendor names, and local/cloud flags are stable.
- OpenAI-compatible clients use max_retries=0.
- Auth env vars and missing-key errors are clear.
- list_models uses the configured client.
- Tool calls and malformed JSON are handled safely.
- Tests cover provider factory and registry discovery.

PR body:
PROMPT
  cat "$tmpdir/body.md"
  cat <<PROMPT

Diff:
PROMPT
  cat "$tmpdir/diff.patch"
} | claude -p \
  --permission-mode dontAsk \
  --allowedTools "" \
  --max-budget-usd 2 >> "$out"

echo "Claude review written to $out"
