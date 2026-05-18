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
| `kestrel-llm-llama-cpp` | `llama_cpp:local` | Local/LAN OpenAI-compatible server | First wave |
| `kestrel-llms` | n/a | Meta-package with install extras | First wave |

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
pip install "kestrel-llms[local]"
pip install "kestrel-llms[all]"
```

## Verification

From the Kestrel checkout:

```bash
scripts/llm_provider_packages.sh verify
```

Before merging a PR:

```bash
scripts/claude_pr_review.sh <pr-number-or-url>
```
