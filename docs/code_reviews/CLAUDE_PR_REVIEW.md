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

Do not merge until the PR description links or includes the Claude review.
If Claude flags a blocking issue, either fix it or document why the finding is
not applicable.

