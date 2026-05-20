# Kestrel LLM Provider Monorepo

This directory is the seed for the future `KestrelSovereignAI/kestrel-llms`
monorepo: one repository, many independently published provider packages.

Core Kestrel keeps the main LLM routes in-tree: OpenAI, Anthropic, Gemini /
Vertex, Ollama, OpenRouter, Claude plan, and Codex plan. Additional providers
live here as installable plugins that register through the
`kestrel_sovereign.llm_providers` entry-point group.

## Packages

| Package | Route | Backend | Status |
| --- | --- | --- | --- |
| `kestrel-llm-deepseek` | `deepseek:api` | Cloud, OpenAI-compatible | First wave |
| `kestrel-llm-xai` | `xai:api` | Cloud, OpenAI-compatible Grok | First wave |
| `kestrel-llm-kimi` | `kimi:api` | Cloud, OpenAI-compatible Moonshot/Kimi | First wave |
| `kestrel-llm-openai-compat` | n/a | Shared OpenAI-compatible adapter helpers | Internal helper |
| `kestrel-llms` | n/a | Meta-package with install extras | First wave, cloud providers only |

`llama.cpp` is intentionally not part of the first-wave plugin set. Kestrel
core already supports `llama_cpp` as a local OpenAI-compatible route through
`OpenAIAdapter`, including local/no-key auth, local privacy routing, prompt
cache body extensions, and model mismatch guards. The experimental
`kestrel-llm-llama-cpp` package is not referenced by the meta-package.

## Monorepo Rules

- Each provider remains a separate PyPI distribution.
- Provider packages depend on `kestrel-sovereign-sdk`, not `kestrel_sovereign`.
- Runtime integration happens through entry points only.
- Provider-specific SDKs are allowed only inside that provider package.
- OpenAI-compatible providers must construct clients with `max_retries=0`.
- Every PR must include a Claude CLI review before merge.

## Install Shape

Install one provider:

```bash
pip install kestrel-llm-deepseek
```

Future convenience meta-package:

```bash
pip install "kestrel-llms[deepseek,xai,kimi]"
pip install "kestrel-llms[cloud]"
pip install "kestrel-llms[all]"
```

## Verification

From the Kestrel checkout:

```bash
scripts/llm_provider_packages.sh verify
```

## Publishing

Create a PyPI Pending Trusted Publisher for each package before its first
release:

```text
Owner: KestrelSovereignAI
Repository: kestrel-llms
Workflow: publish.yml
Environment: pypi
```

Then run the `Publish to PyPI` workflow manually for one package at a time.
Publish dependency packages first:

```text
kestrel-llm-openai-compat
kestrel-llm-deepseek
kestrel-llm-xai
kestrel-llm-kimi
kestrel-llms
```

Before merging a PR:

```bash
scripts/claude_pr_review.sh <pr-number-or-url>
```
