# kestrel-llms — Agent Instructions

See [README.md](README.md) for the monorepo overview and package table.

## Code Index

- [REPO_MAP.md](REPO_MAP.md) — generated per-file index (one-line purpose + public Python symbols; regenerated nightly by `.github/workflows/repo-map.yml`).

## Package Structure

```
kestrel-llms/
├── pyproject.toml                 # uv workspace root (not a published package)
├── providers/
│   ├── kestrel-llms/              # Meta-package with install extras ([all], per-provider)
│   ├── kestrel-llm-openai-compat/ # Shared OpenAI-compatible adapter helpers (internal dep)
│   ├── kestrel-llm-deepseek/      # deepseek:api route
│   ├── kestrel-llm-xai/           # xai:api route (Grok)
│   ├── kestrel-llm-kimi/          # kimi:api route (Moonshot)
│   ├── kestrel-llm-bedrock/       # AWS Bedrock route
│   └── kestrel-llm-vertex/        # Vertex AI route
├── scripts/                       # CI helper + repo-map generator
└── tests/unit/providers/          # Plugin contract tests
```

Each `providers/*` directory is a separate PyPI distribution with its own
`pyproject.toml` and version.

## Entry Points

Providers register through the `kestrel_sovereign.llm_providers` entry-point
group, e.g. `bedrock = "kestrel_llm_bedrock:BedrockAdapter"`. Runtime
integration happens through entry points only — packages depend on
`kestrel-sovereign-sdk`, never on `kestrel_sovereign`.

## Key Files to Read First

1. `README.md` — package table, routes, monorepo rules
2. `providers/kestrel-llm-openai-compat/src/kestrel_llm_openai_compat/__init__.py` — shared adapter base most cloud providers build on
3. `tests/unit/providers/test_llm_provider_plugins.py` — the plugin contract every provider must satisfy

## Running Tests

```bash
uv run pytest tests/
```

CI runs `.github/workflows/llm-provider-packages.yml` on PRs.

## Publishing

No tag trigger and no tag==version gate (unlike the single-package repos).
After merging a version bump, dispatch manually:

```bash
gh workflow run publish.yml -f package=<dist-name> -f ref=main -f use_pypi_environment=true
```

## Agent-Specific Instructions

- Adapters must advertise only the capabilities the route actually serves —
  no aspirational capability flags.
- One provider per package; shared logic goes in `kestrel-llm-openai-compat`,
  not copy-paste.
- Keep provider packages dependency-light; heavy SDKs (e.g. boto3) belong only
  in the package that needs them.
