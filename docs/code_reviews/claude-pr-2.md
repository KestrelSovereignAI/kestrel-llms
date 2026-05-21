# Claude Review: PR #2

- PR: https://github.com/KestrelSovereignAI/kestrel-llms/pull/2
- Title: fix: preserve provider reasoning for tool roundtrips
- Author: UncleSaurus
- Base: main
- Head: codex/deepseek-reasoning-roundtrip
- Reviewed: 2026-05-21T15:11:24Z

## PR Review: DeepSeek Reasoning Roundtrip

**No blocking findings.** The diff is clean, consistent, and well-tested. Details below.

---

### Checklist pass/fail

| Check | Status | Notes |
|---|---|---|
| Entry-point group `kestrel_sovereign.llm_providers` | N/A | No entry-point changes in this diff |
| Providers depend on `kestrel-sovereign-sdk`, not `kestrel_sovereign` | Pass | All pyproject.toml files use `kestrel-sovereign-sdk>=0.14.1,<1` |
| `max_retries=0` | N/A | No client construction changes |
| Version bumps consistent | Pass | compat 0.1.4→0.1.5, providers 0.1.5→0.1.6, meta 0.1.3→0.1.4; all floors updated |
| Tests cover the new behavior | Pass | Reasoning chunk added to stream test, `final.raw` assertion added |
| Contract tests updated | Pass | Version floors bumped in `test_provider_pyprojects_keep_plugin_contract` and `test_meta_package_extras_track_first_wave_packages` |
| Lock file consistent | Pass | All version bumps reflected in `uv.lock` |

---

### Residual risks (non-blocking)

1. **`kestrel-llms/__init__.py` version jump 0.1.1 → 0.1.4** — skips 0.1.2 and 0.1.3. The pyproject.toml says 0.1.4, so the `__init__` is now correct, but the gap suggests the `__init__` was missed in prior releases. No functional impact since the pyproject version is authoritative for packaging.

2. **`reasoning_content` accumulation assumes string-only deltas** — the guard `isinstance(delta_reasoning, str) and delta_reasoning` at `__init__.py:168-169` is correct, but providers that return reasoning as a list or structured object would silently drop it. Given that all three target providers (DeepSeek, Kimi, xAI) use plain strings, this is fine today. Worth a comment if other providers get added.

3. **`raw` field contract** — previously `raw=None` was always set on the terminal `LLMResponse`. Now it can be `{"reasoning_content": "..."}`. Any downstream code doing `if response.raw is not None` to detect non-streaming responses (or similar) would change behavior. Low risk since `raw` is documented as opaque/optional, but worth confirming no agent-loop code branches on `raw` truthiness.

4. **No test for empty reasoning** — the test covers the happy path (reasoning present). A chunk with `reasoning_content=""` or `reasoning_content=None` would correctly be skipped by the guard, but there's no explicit test asserting `final.raw is None` when no reasoning arrives. Minor coverage gap.

---

**Verdict: Approve.** The implementation is minimal, the version matrix is consistent, and the test coverage meaningfully exercises the new streaming path.
