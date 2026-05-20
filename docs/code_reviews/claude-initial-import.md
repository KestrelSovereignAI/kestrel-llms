I can't write the review file in the current permission mode. Here's the review inline:

---

# Claude Review: Commit 53f4413 — Seed kestrel-llms monorepo

**No blocking issues.** Structure is clean, entry points correct, `max_retries=0` set everywhere, auth handling consistent. Six findings below, ordered by severity.

## Findings

### 1. Duplicated `_completion_kwargs` / `_to_llm_response` helpers (Medium)
The OpenAI-compatible providers copy-paste identical helper functions. Extract into a shared internal module.

### 2. Meta-package pins exact versions (Medium)
`providers/kestrel-llms/pyproject.toml` — extras pin `==0.1.0`. This breaks the moment any provider ships 0.1.1. Use `~=0.1.0` or `>=0.1.0,<0.2`.

### 3. `LlamaCppAdapter` missing `key_env_var()` and `deliberation_style()` (Low)
Resolved by removing the duplicate llama.cpp package from the monorepo; core owns llama.cpp support.

### 4. Code formatting: extreme line lengths (Low)
Kimi and xAI providers have 200+ char lines (method signatures, comprehensions) while DeepSeek uses clean multi-line formatting. Apply ruff/black consistently.

### 5. Test config route mismatch (Low)
Resolved by removing the duplicate llama.cpp package from the monorepo; core owns llama.cpp support.

### 6. Tests import kestrel-sovereign internals (Info)
Tests import `ProviderRegistry` from `kestrel_sovereign.llm.provider_registry` rather than through the SDK. Fine for now since the workspace depends on core, but creates coupling.

## Provider Checklist

| Check | Status |
|---|---|
| Entry-point group `kestrel_sovereign.llm_providers` | Pass |
| Depends on `kestrel-sovereign-sdk`, not core | Pass |
| Route/vendor/cloud flags stable | Pass |
| `max_retries=0` on all OpenAI clients | Pass |
| Auth env vars and missing-key errors clear | Pass |
| `list_models` uses configured client | Pass |
| Malformed JSON tool-call handling | Pass (`_raw` fallback) |
| Test coverage for factory + registry | Pass |

## Residual Risks
- No linting/formatting CI step — long lines will accumulate
- SDK version range `>=0.14,<1` is broad; a breaking 0.15 change could silently affect providers
- No integration tests (expected at seed stage)
