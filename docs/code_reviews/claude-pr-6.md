# Claude Review: PR #6

- PR: https://github.com/KestrelSovereignAI/kestrel-llms/pull/6
- Title: feat: declare provider capabilities
- Author: UncleSaurus
- Base: main
- Head: codex/provider-capabilities
- Reviewed: 2026-05-26T16:45:16Z

`Iterable` is still used by `passthrough_keys` parameters on lines 109 and 176 — **not dead**. The prior review's finding #2 was incorrect. No action needed.

**2. Self-review doc in the diff** — `docs/code_reviews/claude-pr-6.md`

This file is the review document for _this very PR_, committed as part of the PR. If this is an intentional convention (review-as-docs), fine. If not, it should be dropped before merge. It also contains a now-incorrect finding about `Iterable` being dead.

---

### Checklist

| Check | Status |
|---|---|
| Entry-point group `kestrel_sovereign.llm_providers` | N/A (no entry-point changes) |
| Providers depend on SDK, not `kestrel_sovereign` | **OK** — `pyproject.toml` deps are `kestrel-sovereign-sdk`, test asserts no `kestrel_sovereign` dep |
| Stable route/vendor names | N/A |
| `max_retries=0` | N/A (no client changes) |
| Auth env vars | N/A |
| `list_models` uses configured client | N/A |
| Tool calls / malformed JSON | N/A |
| Tests cover factory + registry | **OK** — parametrized test covers all three adapters + negative-path test + SDK contract test |
| Version bumps consistent | **OK** — compat 0.1.7, plugins 0.1.8, meta 0.1.6, SDK floor 0.17.0 all aligned |
| `openai_compatible_capabilities` returns `ProviderCapabilities` | **OK** — returns SDK dataclass, not a raw dict |

---

### Residual risks

- **Convention-coupled string values**: `StructuredOutputMode.JSON_SCHEMA`, `ToolStreamingMode.NATIVE_DELTA`, `VisionInputMode.OPENAI_IMAGE_URL` are now SDK enums (good), but `model_dependent` entries (`"vision"`, `"structured_output"`) are still free-form strings. Tests lock the values, but there's no SDK-level validation that these strings are meaningful.
- **No base-class enforcement of `provider_capabilities()`**: A future provider that omits the method will only fail at call-time. Low risk today with three providers and full test coverage.
- **SDK 0.17.0 not yet published**: As noted in the PR body, local tests use an editable install of the SDK. The merge is safe but the packages are **not installable from PyPI** until SDK PR #24 lands. Coordinate the release order.

**Verdict: Approve.** The only concrete action item is deciding whether `docs/code_reviews/claude-pr-6.md` should ship — and correcting its claim that `Iterable` is dead if it does.
