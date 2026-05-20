# Epic: Extracted LLM Provider Plugins

Kestrel core owns the main routes: Anthropic, OpenAI, Gemini/Google, Vertex AI,
Ollama, OpenRouter, Claude plan, and Codex plan. Additional model vendors live
in this monorepo as separately published provider packages.

## Tracking

- Kestrel core umbrella: KestrelSovereignAI/kestrel-sovereign#1279
- Layout seed: KestrelSovereignAI/kestrel-sovereign#1280
- DeepSeek: KestrelSovereignAI/kestrel-sovereign#1282
- xAI/Grok: KestrelSovereignAI/kestrel-sovereign#1283
- Kimi/Moonshot: KestrelSovereignAI/kestrel-sovereign#1281
- Shared OpenAI-compatible helpers: part of first-wave package hardening.
- Build and publish pipeline: KestrelSovereignAI/kestrel-sovereign#1285
- Claude CLI review gate: KestrelSovereignAI/kestrel-sovereign#1284

`llama.cpp` is resolved in Kestrel core as a local OpenAI-compatible route, not
as a first-wave external provider. The plugin package is experimental and is
not included in the `kestrel-llms` meta-package.

## Acceptance

- Each provider builds as an independent PyPI distribution.
- Provider packages depend on `kestrel-sovereign-sdk`, not Kestrel core.
- Kestrel discovers providers through `kestrel_sovereign.llm_providers`.
- `scripts/llm_provider_packages.sh verify` compiles, builds, and tests all
  packages.
- Every PR includes a Claude CLI review.
