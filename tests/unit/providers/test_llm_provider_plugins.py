from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from kestrel_sdk.llm import LLMResponse, ToolCallStarted

ROOT = Path(__file__).resolve().parents[3]
for src in [
    ROOT / "providers/kestrel-llm-deepseek/src",
    ROOT / "providers/kestrel-llm-xai/src",
    ROOT / "providers/kestrel-llm-kimi/src",
    ROOT / "providers/kestrel-llm-openai-compat/src",
]:
    sys.path.insert(0, str(src))

from kestrel_sovereign.llm.provider_registry import (  # noqa: E402
    LLM_PROVIDER_ENTRY_POINT_GROUP,
    ProviderRegistry,
)
from kestrel_llm_openai_compat import normalize_messages  # noqa: E402
from kestrel_llm_kimi import normalize_kimi_messages  # noqa: E402


def _entry_point(name: str, cls):
    ep = MagicMock()
    ep.name = name
    ep.value = f"{cls.__module__}:{cls.__name__}"
    ep.load.return_value = cls
    return ep


def _patch_entry_points(entry_points: list):
    eps = MagicMock()
    eps.select.return_value = entry_points
    return patch("kestrel_sovereign.entrypoints.importlib.metadata.entry_points", return_value=eps)


@pytest.mark.parametrize(
    ("module_name", "class_name", "vendor", "env_var", "base_url", "default_model"),
    [
        ("kestrel_llm_deepseek", "DeepSeekAdapter", "deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-v4-flash"),
        ("kestrel_llm_xai", "XAIAdapter", "xai", "XAI_API_KEY", "https://api.x.ai/v1", "grok-4.3"),
        ("kestrel_llm_kimi", "KimiAdapter", "kimi", "MOONSHOT_API_KEY", "https://api.moonshot.ai/v1", "kimi-k2.6"),
    ],
)
def test_cloud_provider_factories_build_route_info(
    monkeypatch,
    module_name,
    class_name,
    vendor,
    env_var,
    base_url,
    default_model,
):
    module = __import__(module_name)
    cls = getattr(module, class_name)
    monkeypatch.setenv(env_var, "sk-test")
    fake_client = MagicMock()

    with patch.object(module.openai, "AsyncOpenAI", return_value=fake_client) as async_openai:
        info = cls.create_provider({})

    async_openai.assert_called_once_with(api_key="sk-test", base_url=base_url, max_retries=0)
    assert info.name == f"{vendor}:api"
    assert info.vendor == vendor
    assert info.route == "api"
    assert info.client is fake_client
    assert info.model == default_model
    assert info.is_cloud is True
    assert info.is_local is False


def test_kimi_accepts_kimi_api_key_alias(monkeypatch):
    import kestrel_llm_kimi

    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.setenv("KIMI_API_KEY", "sk-kimi")

    with patch.object(kestrel_llm_kimi.openai, "AsyncOpenAI") as async_openai:
        kestrel_llm_kimi.KimiAdapter.create_provider({})

    assert async_openai.call_args.kwargs["api_key"] == "sk-kimi"


class _AsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _chunk(delta=None, usage=None):
    choices = [] if delta is None else [SimpleNamespace(delta=delta)]
    return SimpleNamespace(choices=choices, usage=usage)


def _tool_delta(index, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_normalize_messages_json_encodes_tool_call_arguments_without_mutating():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"q": "kestrel", "limit": 2},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]

    normalized = normalize_messages(messages)

    arguments = normalized[1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert arguments == '{"q": "kestrel", "limit": 2}'
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {
        "q": "kestrel",
        "limit": 2,
    }


def test_kimi_normalize_messages_adds_reasoning_content_to_tool_call_history():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"q": "kestrel"},
                    },
                }
            ],
        },
    ]

    normalized = normalize_kimi_messages(messages)

    assert normalized[1]["reasoning_content"] == ""
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == '{"q": "kestrel"}'
    assert "reasoning_content" not in messages[1]


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("kestrel_llm_deepseek", "DeepSeekAdapter"),
        ("kestrel_llm_xai", "XAIAdapter"),
        ("kestrel_llm_kimi", "KimiAdapter"),
    ],
)
async def test_openai_compatible_plugins_stream_with_tools(module_name, class_name):
    module = __import__(module_name)
    adapter = getattr(module, class_name)()
    captured_kwargs = {}

    chunks = [
        _chunk(SimpleNamespace(content="I'll check. ", tool_calls=None)),
        _chunk(SimpleNamespace(content=None, reasoning_content="Need lookup.", tool_calls=None)),
        _chunk(
            SimpleNamespace(
                content=None,
                tool_calls=[_tool_delta(0, id="call_1", name="lookup", arguments='{"q"')],
            )
        ),
        _chunk(
            SimpleNamespace(
                content=None,
                tool_calls=[_tool_delta(0, arguments=':"kestrel"}')],
            )
        ),
        _chunk(None, usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12)),
    ]

    async def create(**kwargs):
        captured_kwargs.update(kwargs)
        assert kwargs["stream"] is True
        assert kwargs["tools"] == [{"type": "function", "function": {"name": "lookup"}}]
        assert kwargs["tool_choice"] == "auto"
        return _AsyncStream(chunks)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    items = [
        item
        async for item in adapter.get_streaming_response_with_tools(
            client,
            "model",
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_previous",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": {"q": "previous"},
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_previous", "content": "ok"},
                {"role": "user", "content": "hi"},
            ],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            max_tokens=20,
        )
    ]

    assert items[0] == "I'll check. "
    assert items[1] == ToolCallStarted(index=0, id="call_1", name="lookup")
    final = items[-1]
    assert isinstance(final, LLMResponse)
    assert final.content == "I'll check. "
    assert final.input_tokens == 5
    assert final.output_tokens == 7
    assert final.total_tokens == 12
    assert final.raw == {"reasoning_content": "Need lookup."}
    assert final.tool_calls[0].id == "call_1"
    assert final.tool_calls[0].name == "lookup"
    assert final.tool_calls[0].arguments == {"q": "kestrel"}
    sent_arguments = captured_kwargs["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert sent_arguments == '{"q": "previous"}'
    if module_name == "kestrel_llm_kimi":
        assert captured_kwargs["messages"][0]["reasoning_content"] == ""
    else:
        assert "reasoning_content" not in captured_kwargs["messages"][0]


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("kestrel_llm_deepseek", "DeepSeekAdapter"),
        ("kestrel_llm_xai", "XAIAdapter"),
        ("kestrel_llm_kimi", "KimiAdapter"),
    ],
)
async def test_openai_compatible_plugins_get_response_normalizes_tool_history(
    module_name,
    class_name,
):
    module = __import__(module_name)
    adapter = getattr(module, class_name)()
    captured_kwargs = {}

    async def create(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=None)
                )
            ],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7),
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    response = await adapter.get_response(
        client,
        "model",
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_previous",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": {"q": "previous"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_previous", "content": "ok"},
            {"role": "user", "content": "hi"},
        ],
    )

    assert response.content == "done"
    sent_arguments = captured_kwargs["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert sent_arguments == '{"q": "previous"}'
    if module_name == "kestrel_llm_kimi":
        assert captured_kwargs["messages"][0]["reasoning_content"] == ""
    else:
        assert "reasoning_content" not in captured_kwargs["messages"][0]


def test_registry_discovers_first_wave_plugins(monkeypatch):
    import kestrel_llm_deepseek
    import kestrel_llm_kimi
    import kestrel_llm_xai

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("XAI_API_KEY", "sk-xai")
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-kimi")

    entry_points = [
        _entry_point("deepseek", kestrel_llm_deepseek.DeepSeekAdapter),
        _entry_point("xai", kestrel_llm_xai.XAIAdapter),
        _entry_point("kimi", kestrel_llm_kimi.KimiAdapter),
    ]
    config = {
        "route_priority": ["deepseek:api", "xai:api", "kimi:api"],
        "vendors": {
            "deepseek": {"routes": {"api": {}}},
            "xai": {"routes": {"api": {}}},
            "kimi": {"routes": {"api": {}}},
        },
    }

    with (
        _patch_entry_points(entry_points),
        patch.object(kestrel_llm_deepseek.openai, "AsyncOpenAI", return_value=MagicMock()),
        patch.object(kestrel_llm_xai.openai, "AsyncOpenAI", return_value=MagicMock()),
        patch.object(kestrel_llm_kimi.openai, "AsyncOpenAI", return_value=MagicMock()),
    ):
        providers = ProviderRegistry(config).initialize_providers()

    assert [provider.name for provider in providers] == [
        "deepseek:api",
        "xai:api",
        "kimi:api",
    ]
    assert {provider.name for provider in providers if provider.is_local} == set()
    assert {provider.name for provider in providers if provider.is_cloud} == {
        "deepseek:api",
        "xai:api",
        "kimi:api",
    }


def test_entry_point_group_stays_stable():
    assert LLM_PROVIDER_ENTRY_POINT_GROUP == "kestrel_sovereign.llm_providers"


def _load_pyproject(package: str) -> dict:
    with (ROOT / "providers" / package / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


@pytest.mark.parametrize(
    ("package", "entry_point"),
    [
        ("kestrel-llm-deepseek", "deepseek"),
        ("kestrel-llm-xai", "xai"),
        ("kestrel-llm-kimi", "kimi"),
    ],
)
def test_provider_pyprojects_keep_plugin_contract(package, entry_point):
    pyproject = _load_pyproject(package)
    project = pyproject["project"]

    assert project["name"] == package
    assert "kestrel-sovereign-sdk>=0.14.1,<1" in project["dependencies"]
    assert "kestrel-llm-openai-compat>=0.1.5,<0.2" in project["dependencies"]
    assert not any(dep.startswith("kestrel_sovereign") for dep in project["dependencies"])

    entry_points = pyproject["project"]["entry-points"][LLM_PROVIDER_ENTRY_POINT_GROUP]
    assert list(entry_points) == [entry_point]


def test_meta_package_extras_track_first_wave_packages():
    pyproject = _load_pyproject("kestrel-llms")
    extras = pyproject["project"]["optional-dependencies"]

    assert set(extras) == {"deepseek", "xai", "kimi", "cloud", "all"}
    assert extras["deepseek"] == ["kestrel-llm-deepseek>=0.1.6,<0.2"]
    assert extras["xai"] == ["kestrel-llm-xai>=0.1.6,<0.2"]
    assert extras["kimi"] == ["kestrel-llm-kimi>=0.1.6,<0.2"]
    assert set(extras["cloud"]) == {
        "kestrel-llm-deepseek>=0.1.6,<0.2",
        "kestrel-llm-xai>=0.1.6,<0.2",
        "kestrel-llm-kimi>=0.1.6,<0.2",
    }
    assert set(extras["all"]) == set(extras["cloud"])


def test_shared_openai_compat_package_has_no_entry_point():
    pyproject = _load_pyproject("kestrel-llm-openai-compat")
    project = pyproject["project"]

    assert project["name"] == "kestrel-llm-openai-compat"
    assert "kestrel-sovereign-sdk>=0.14.1,<1" in project["dependencies"]
    assert "entry-points" not in pyproject["project"]
