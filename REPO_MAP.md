# kestrel-llms — Repo Map

Auto-generated file-tree + per-file purpose index. Do **not** edit by hand —
regenerate via `python scripts/generate_repo_map.py` (refreshed nightly by
`.github/workflows/repo-map.yml`). No timestamp on purpose: the nightly job
commits only when the tree actually changes; `git log REPO_MAP.md` has the date.

**Scope:** 36 tracked files (9 `.py`, 12 `.md`, 15 other). Excludes caches, lockfiles, and build artifacts.

**Format per file:** `path — one-line purpose` plus the public top-level Python symbols on the next line
(classes and functions; private `_name` skipped).

---
## Top-level files

Repo entry points and standard project files.

- **.gitignore** — —
- **AGENTS.md** — kestrel-llms — Agent Instructions — See [README.md](README.md) for the monorepo overview and package table.
- **LICENSE** — —
- **README.md** — Kestrel LLM Provider Monorepo — This directory is the seed for the future `KestrelSovereignAI/kestrel-llms` monorepo: one repository, many independently published provider packages.
- **REPO_MAP.md** — kestrel-llms — Repo Map — Auto-generated file-tree + per-file purpose index.
- **pyproject.toml** — (configuration)

## `.github/`

- **.github/PULL_REQUEST_TEMPLATE.md** — ## Summary
- **.github/workflows/llm-provider-packages.yml** — (configuration)
- **.github/workflows/publish.yml** — (configuration)
- **.github/workflows/repo-map.yml** — (configuration)

## `docs/`

- **docs/architecture/LLM_PROVIDER_PLUGIN_EPIC.md** — Epic: Extracted LLM Provider Plugins — Kestrel core owns the main routes: Anthropic, OpenAI, Gemini/Google, Vertex AI, Ollama, OpenRouter, Claude plan, and Codex plan.
- **docs/code_reviews/CLAUDE_PR_REVIEW.md** — Claude CLI PR Review Protocol — All PRs for the `kestrel-llms` provider monorepo must receive a Claude CLI review before merge.
- **docs/code_reviews/claude-initial-import.md** — I can't write the review file in the current permission mode.
- **docs/code_reviews/claude-pr-10.md** — Claude Review: PR #10 — - PR: https://github.com/KestrelSovereignAI/kestrel-llms/pull/10 - Title: Add AWS Bedrock provider package - Author: UncleSaurus - Base: main - Head: codex/add-bedrock-provider - Reviewed: 2026-05-26…
- **docs/code_reviews/claude-pr-2.md** — Claude Review: PR #2 — - PR: https://github.com/KestrelSovereignAI/kestrel-llms/pull/2 - Title: fix: preserve provider reasoning for tool roundtrips - Author: UncleSaurus - Base: main - Head: codex/deepseek-reasoning-round…
- **docs/code_reviews/claude-pr-4.md** — Now I have full context.
- **docs/code_reviews/claude-pr-6.md** — Claude Review: PR #6 — - PR: https://github.com/KestrelSovereignAI/kestrel-llms/pull/6 - Title: feat: declare provider capabilities - Author: UncleSaurus - Base: main - Head: codex/provider-capabilities - Reviewed: 2026-05…
- **docs/code_reviews/claude-pr-8.md** — Claude Review: PR #8 — - PR: https://github.com/KestrelSovereignAI/kestrel-llms/pull/8 - Title: Add Vertex AI provider package - Author: UncleSaurus - Base: main - Head: codex/add-vertex-provider - Reviewed: 2026-05-26T17:…

## `providers/`

- **providers/kestrel-llm-bedrock/pyproject.toml** — (configuration)
- **providers/kestrel-llm-bedrock/src/kestrel_llm_bedrock/__init__.py** — AWS Bedrock provider plugin for Kestrel Sovereign.
  - `class BedrockClients`; `def normalize_bedrock_messages(messages)`; `class BedrockAdapter`
- **providers/kestrel-llm-deepseek/pyproject.toml** — (configuration)
- **providers/kestrel-llm-deepseek/src/kestrel_llm_deepseek/__init__.py** — DeepSeek provider plugin for Kestrel Sovereign.
  - `class DeepSeekAdapter`
- **providers/kestrel-llm-kimi/pyproject.toml** — (configuration)
- **providers/kestrel-llm-kimi/src/kestrel_llm_kimi/__init__.py** — Moonshot Kimi provider plugin for Kestrel Sovereign.
  - `def normalize_kimi_messages(messages)`; `class KimiAdapter`
- **providers/kestrel-llm-openai-compat/pyproject.toml** — (configuration)
- **providers/kestrel-llm-openai-compat/src/kestrel_llm_openai_compat/__init__.py** — Shared helpers for OpenAI-compatible Kestrel LLM provider plugins.
  - `def openai_compatible_capabilities()`; `def normalize_messages(messages)`; `def completion_kwargs(format, tools, response_format, kwargs, …)`; `def to_llm_response(response)`; `async def stream_with_tool_calls(client, model, messages, tools, …)`
- **providers/kestrel-llm-vertex/pyproject.toml** — (configuration)
- **providers/kestrel-llm-vertex/src/kestrel_llm_vertex/__init__.py** — Google Vertex AI provider plugin for Kestrel Sovereign.
  - `def normalize_vertex_messages(messages)`; `class VertexAIAdapter`
- **providers/kestrel-llm-xai/pyproject.toml** — (configuration)
- **providers/kestrel-llm-xai/src/kestrel_llm_xai/__init__.py** — xAI Grok provider plugin for Kestrel Sovereign.
  - `class XAIAdapter`
- **providers/kestrel-llms/pyproject.toml** — (configuration)
- **providers/kestrel-llms/src/kestrel_llms/__init__.py** — Convenience package for installing Kestrel LLM provider extras.

## `scripts/`

- **scripts/claude_pr_review.sh** — —
- **scripts/generate_repo_map.py** — Generate REPO_MAP.md — a file-tree + per-file purpose index for this repo.
  - `class FileEntry`; `def repo_name()`; `def tracked_files()`; `def is_excluded(path)`; `def first_sentence(text, max_chars)`; `def summarize_python(path)`; `def summarize_markdown(path)`; `def summarize_other(path)`; `…`
- **scripts/llm_provider_packages.sh** — —

## `tests/`

- **tests/unit/providers/test_llm_provider_plugins.py** — —
  - `def test_cloud_provider_factories_build_route_info(monkeypatch, module_name, class_name, vendor, …)`; `def test_kimi_accepts_kimi_api_key_alias(monkeypatch)`; `def test_vertex_factory_uses_api_key_mode(monkeypatch)`; `def test_vertex_factory_accepts_gemini_api_key_alias(monkeypatch)`; `def test_vertex_factory_uses_vertex_project_mode(monkeypatch)`; `def test_bedrock_factory_uses_profile_and_region(monkeypatch)`; `def test_bedrock_factory_uses_static_credentials(monkeypatch)`; `def test_normalize_messages_json_encodes_tool_call_arguments_without_mutating()`; `…`
