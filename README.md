# Kestrel LLM Provider Monorepo

One repository with independently published provider packages for Kestrel.

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
| `kestrel-llm-vertex` | `vertex_ai:api` | Google Gen AI SDK | Available |
| `kestrel-llm-bedrock` | `bedrock:api` | AWS Bedrock Converse | Available |
| `kestrel-llm-openai-compat` | n/a | Shared OpenAI-compatible adapter helpers | Internal helper |
| `kestrel-llms` | n/a | Meta-package with install extras | First wave, cloud providers only |

`llama.cpp` is intentionally not part of this plugin monorepo. Kestrel core
already supports `llama_cpp` as a local OpenAI-compatible route through
`OpenAIAdapter`, including local/no-key auth, local privacy routing, prompt
cache body extensions, and model mismatch guards.

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

Install through the meta-package:

```bash
pip install "kestrel-llms[deepseek,xai,kimi]"
pip install "kestrel-llms[cloud]"
pip install "kestrel-llms[all]"
```

## Completion contract in 0.2

All packages require `kestrel-sovereign-sdk>=0.49.0`. Each response carries the
native provider's `stop_reason`, including unknown values. Missing evidence
stays `None`; the host decides whether a generation is safe to accept.

Both streaming methods yield text followed by exactly one final `LLMResponse`.
The final response arrives after native response cleanup. A transport, protocol,
or cleanup failure raises without a successful terminal response. Plain
streaming rejects requested or returned tool calls. Vertex tool streaming keeps
its advertised non-streaming handoff; Bedrock streaming does not support schema
projection and refuses it before sending a request.

Bedrock's non-streaming schema projection keeps the native `tool_use` stop
reason and usage while returning the assembled object as content. Native
streams can complete zero-argument calls without argument fragments; those
calls carry `{}`. Nonempty malformed JSON and non-object arguments fail.

Cancelling a Bedrock stream waits for the native read in progress before it
returns. On Windows that read may wait for further wire data or botocore's
read timeout (60 seconds by default). The original read stays owned until it
settles; the adapter then closes the response and discards any late event.

This is a breaking provider interface release. Callers must accept the final
response on plain streams and use the client returned by `create_provider` for
Vertex and Bedrock operations, including model discovery. Close those clients
with `await info.client.close()` when the provider is retired. Vertex accepts
`base_url` and `api_version`; Bedrock accepts `endpoint_url` and `ca_bundle` for
custom endpoints. Automatic Google client-certificate mTLS is refused in Vertex
project mode because the owned HTTP transport cannot perform that SDK endpoint
switch. API-key mode selects the Gemini API. If `GOOGLE_GENAI_USE_VERTEXAI` or
`GOOGLE_GENAI_USE_ENTERPRISE` requests Vertex express mode, construction fails
before opening a client. Use Vertex project credentials, or clear those flags
to select the Gemini API explicitly.

## Verification

From this checkout, without installing Core:

```bash
uv sync --locked --group test
scripts/llm_provider_packages.sh verify
```

The native SDK tests use owned loopback HTTP/SSE, TLS and AWS event-stream peers.
They require no vendor credentials or local model server. CI covers Python
3.11–3.14 on Linux and Windows.

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
kestrel-llm-vertex
kestrel-llm-bedrock
kestrel-llms
```

Before merging a PR:

```bash
scripts/claude_pr_review.sh <pr-number-or-url>
```
