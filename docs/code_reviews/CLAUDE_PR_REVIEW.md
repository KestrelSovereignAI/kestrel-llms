# Claude CLI PR Review Protocol

All PRs for the `kestrel-llms` provider monorepo must receive a Claude CLI
review before merge.

## Required Command

```bash
scripts/claude_pr_review.sh <pr-number-or-url>
```

The script:

1. Reads PR metadata with `gh`.
2. Fetches the PR diff.
3. Runs `claude -p` in code-review mode.
4. Writes the review to `docs/code_reviews/claude-pr-<number>.md`.

## Review Standard

Claude should lead with bugs, regressions, security issues, provider-contract
breakage, missing tests, packaging mistakes, and release hazards. Style notes
are secondary.

Provider review must specifically check:

- Entry-point group and package metadata.
- SDK-only dependency boundary.
- Route name, vendor, and local/cloud flags.
- Auth env vars and missing-key errors.
- `max_retries=0` for OpenAI-compatible clients.
- Model discovery against the configured client.
- Tool-call and malformed-JSON handling.
- Provider-specific parameters passed through intentionally.

## Merge Rule

Do not merge until the committed Claude review artifact
(`docs/code_reviews/claude-pr-<number>.md`) is present for the PR. If Claude
flags a blocking issue, either fix it or document in that artifact why the
finding is not applicable.

## Enforcement

Enforcement is a **required status check** (`verify-claude-review`) backed by
branch protection on `main` — *not* a CI job that runs Claude.

### Why CI does not run Claude

Running `claude -p` in a workflow would require `ANTHROPIC_API_KEY` as a repo
secret. This repository is public, so a fork PR can propose changes to workflows
and tooling. Exposing that key to untrusted PR code — via `pull_request_target`
or by executing checked-out code — is a credential-exfiltration vector. The
review therefore runs **locally**, on a maintainer's machine, via
`scripts/claude_pr_review.sh`, and CI only verifies that the resulting evidence
artifact was committed.

### What CI verifies

`.github/workflows/claude-review-gate.yml` runs on `pull_request` with
`permissions: contents: read`. It checks out the PR head and asserts that
`docs/code_reviews/claude-pr-<PR>.md` exists, has the exact
`# Claude Review: PR #<PR>` header, and links this PR's URL. It reads one
markdown file — it never executes checked-out code and needs no secrets, so it
runs safely on fork PRs. The check is red until the artifact is pushed; it going
green *is* the signal that the review was performed.

Review loop for a PR:

1. Open the PR (the script needs the PR number).
2. Run `scripts/claude_pr_review.sh <PR>` locally.
3. Commit and push `docs/code_reviews/claude-pr-<PR>.md`.
4. `verify-claude-review` turns green; the PR becomes mergeable.

### Making the check required

A repository admin enables branch protection on `main` requiring the
`verify-claude-review` status. Branch protection on **public** repositories is
available on GitHub's Free plan:

```bash
gh api -X PUT repos/KestrelSovereignAI/kestrel-llms/branches/main/protection \
  -H "Accept: application/vnd.github+json" \
  --input - <<'JSON'
{
  "required_status_checks": {
    "strict": false,
    "checks": [{"context": "verify-claude-review"}]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null
}
JSON
```

Set `"enforce_admins": true` to hold maintainers to the same gate. Leave it
`false` to keep an admin escape hatch for emergency fixes.

