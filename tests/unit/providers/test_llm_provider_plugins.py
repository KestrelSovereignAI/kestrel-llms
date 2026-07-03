from __future__ import annotations

import sys
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel
from kestrel_sdk.llm import (
    LLMResponse,
    ModelCategory,
    ProviderCapabilities,
    StructuredOutputMode,
    ToolCall,
    ToolCallStarted,
    ToolStreamingMode,
    VisionInputMode,
)

ROOT = Path(__file__).resolve().parents[3]
for src in [
    ROOT / "providers/kestrel-llm-deepseek/src",
    ROOT / "providers/kestrel-llm-xai/src",
    ROOT / "providers/kestrel-llm-kimi/src",
    ROOT / "providers/kestrel-llm-vertex/src",
    ROOT / "providers/kestrel-llm-bedrock/src",
    ROOT / "providers/kestrel-llm-openai-compat/src",
]:
    sys.path.insert(0, str(src))

from kestrel_sovereign.llm.provider_registry import (  # noqa: E402
    LLM_PROVIDER_ENTRY_POINT_GROUP,
    ProviderRegistry,
)
from kestrel_llm_openai_compat import (  # noqa: E402
    normalize_messages,
    openai_compatible_capabilities,
    to_llm_response,
)
from kestrel_llm_kimi import normalize_kimi_messages  # noqa: E402
from kestrel_llm_vertex import normalize_vertex_messages  # noqa: E402
from kestrel_llm_bedrock import normalize_bedrock_messages  # noqa: E402


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


def test_vertex_factory_uses_api_key_mode(monkeypatch):
    import kestrel_llm_vertex

    fake_client = MagicMock()
    fake_genai = SimpleNamespace(Client=MagicMock(return_value=fake_client))
    monkeypatch.setattr(kestrel_llm_vertex, "_load_genai", lambda: fake_genai)
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-google")

    info = kestrel_llm_vertex.VertexAIAdapter.create_provider({})

    fake_genai.Client.assert_called_once_with(api_key="sk-google")
    assert info.name == "vertex_ai:api"
    assert info.vendor == "vertex_ai"
    assert info.route == "api"
    assert info.client is fake_client
    assert info.model == "gemini-2.5-flash"
    assert info.is_cloud is True
    assert info.is_local is False


def test_vertex_factory_accepts_gemini_api_key_alias(monkeypatch):
    import kestrel_llm_vertex

    fake_client = MagicMock()
    fake_genai = SimpleNamespace(Client=MagicMock(return_value=fake_client))
    monkeypatch.setattr(kestrel_llm_vertex, "_load_genai", lambda: fake_genai)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini")

    info = kestrel_llm_vertex.VertexAIAdapter.create_provider({})

    fake_genai.Client.assert_called_once_with(api_key="sk-gemini")
    assert info.client is fake_client


def test_vertex_factory_uses_vertex_project_mode(monkeypatch):
    import kestrel_llm_vertex

    fake_client = MagicMock()
    fake_genai = SimpleNamespace(Client=MagicMock(return_value=fake_client))
    monkeypatch.setattr(kestrel_llm_vertex, "_load_genai", lambda: fake_genai)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "kestrel-project")

    info = kestrel_llm_vertex.VertexAIAdapter.create_provider({"location": "us-east5"})

    fake_genai.Client.assert_called_once_with(
        vertexai=True,
        project="kestrel-project",
        location="us-east5",
    )
    assert info.client is fake_client


def test_bedrock_factory_uses_profile_and_region(monkeypatch):
    import kestrel_llm_bedrock

    fake_runtime_client = MagicMock()
    fake_management_client = MagicMock()
    fake_session = MagicMock()
    fake_session.client.side_effect = [fake_runtime_client, fake_management_client]
    fake_boto3 = SimpleNamespace(Session=MagicMock(return_value=fake_session))
    retry_config = object()
    monkeypatch.setattr(kestrel_llm_bedrock, "_load_boto3", lambda: fake_boto3)
    monkeypatch.setattr(kestrel_llm_bedrock, "_client_config", lambda: retry_config)

    info = kestrel_llm_bedrock.BedrockAdapter.create_provider(
        {"profile": "prod", "region": "us-west-2", "model": "anthropic.test"}
    )

    fake_boto3.Session.assert_called_once_with(profile_name="prod")
    assert fake_session.client.call_args_list[0].args == ("bedrock-runtime",)
    assert fake_session.client.call_args_list[0].kwargs == {
        "region_name": "us-west-2",
        "endpoint_url": None,
        "config": retry_config,
    }
    assert fake_session.client.call_args_list[1].args == ("bedrock",)
    assert fake_session.client.call_args_list[1].kwargs == {
        "region_name": "us-west-2",
        "endpoint_url": None,
        "config": retry_config,
    }
    assert info.name == "bedrock:api"
    assert info.vendor == "bedrock"
    assert info.route == "api"
    assert info.client.runtime is fake_runtime_client
    assert info.client.management is fake_management_client
    assert info.model == "anthropic.test"
    assert info.is_cloud is True
    assert info.is_local is False


def test_bedrock_factory_uses_static_credentials(monkeypatch):
    import kestrel_llm_bedrock

    fake_session = MagicMock()
    fake_boto3 = SimpleNamespace(Session=MagicMock(return_value=fake_session))
    monkeypatch.setattr(kestrel_llm_bedrock, "_load_boto3", lambda: fake_boto3)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")

    kestrel_llm_bedrock.BedrockAdapter.create_provider(
        {
            "aws_access_key_id": "key",
            "aws_secret_access_key": "secret",
            "aws_session_token": "token",
        }
    )

    fake_boto3.Session.assert_called_once_with(
        aws_access_key_id="key",
        aws_secret_access_key="secret",
        aws_session_token="token",
    )
    assert [call.args for call in fake_session.client.call_args_list] == [
        ("bedrock-runtime",),
        ("bedrock",),
    ]
    assert fake_session.client.call_args_list[0].kwargs["region_name"] == "eu-central-1"
    assert fake_session.client.call_args_list[1].kwargs["region_name"] == "eu-central-1"


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


def test_vertex_normalize_messages_converts_chat_tool_history():
    messages = [
        {"role": "system", "content": "Be brief."},
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
        {
            "role": "tool",
            "name": "lookup",
            "tool_call_id": "call_1",
            "content": '{"ok": true}',
        },
    ]

    system_prompt, contents = normalize_vertex_messages(messages)

    assert system_prompt == "Be brief."
    assert contents == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {
            "role": "model",
            "parts": [
                {
                    "function_call": {
                        "id": "call_1",
                        "name": "lookup",
                        "args": {"q": "kestrel"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "lookup",
                        "response": {"result": {"ok": True}},
                    }
                }
            ],
        },
    ]


def test_bedrock_normalize_messages_converts_tool_history_and_system():
    messages = [
        {"role": "system", "content": "Be brief."},
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
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"ok": true}',
        },
    ]

    system, normalized = normalize_bedrock_messages(messages)

    assert system == [{"text": "Be brief."}]
    assert normalized == [
        {"role": "user", "content": [{"text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "call_1",
                        "name": "lookup",
                        "input": {"q": "kestrel"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "call_1",
                        "content": [{"json": {"ok": True}}],
                    }
                }
            ],
        },
    ]


def test_to_llm_response_preserves_raw_provider_reasoning_object():
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    reasoning_content="Need the tool.",
                    tool_calls=None,
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    response = to_llm_response(raw)

    assert response.raw is raw
    assert response.raw.choices[0].message.reasoning_content == "Need the tool."


def test_openai_compatible_capabilities_return_sdk_contract():
    capabilities = openai_compatible_capabilities(
        supports_vision=True,
        model_dependent=("vision",),
        notes=("example",),
    )

    assert capabilities == ProviderCapabilities(
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        supports_structured_output=True,
        structured_output_mode=StructuredOutputMode.JSON_SCHEMA,
        tool_streaming_mode=ToolStreamingMode.NATIVE_DELTA,
        vision_input_mode=VisionInputMode.OPENAI_IMAGE_URL,
        model_dependent=("vision",),
        notes=("example",),
    )


def test_openai_compatible_capabilities_can_disable_feature_modes():
    capabilities = openai_compatible_capabilities(
        supports_structured_output=False,
    )

    assert capabilities.supports_vision is False
    assert capabilities.supports_structured_output is False
    assert capabilities.structured_output_mode == StructuredOutputMode.NONE
    assert capabilities.vision_input_mode == VisionInputMode.NONE


@pytest.mark.parametrize(
    (
        "module_name",
        "class_name",
        "supports_vision",
        "structured_output_mode",
        "tool_streaming_mode",
        "vision_input_mode",
        "model_dependent",
    ),
    [
        (
            "kestrel_llm_deepseek",
            "DeepSeekAdapter",
            False,
            StructuredOutputMode.JSON_SCHEMA,
            ToolStreamingMode.NATIVE_DELTA,
            VisionInputMode.NONE,
            {"structured_output"},
        ),
        (
            "kestrel_llm_xai",
            "XAIAdapter",
            True,
            StructuredOutputMode.JSON_SCHEMA,
            ToolStreamingMode.NATIVE_DELTA,
            VisionInputMode.OPENAI_IMAGE_URL,
            {"vision", "structured_output"},
        ),
        (
            "kestrel_llm_kimi",
            "KimiAdapter",
            False,
            StructuredOutputMode.JSON_SCHEMA,
            ToolStreamingMode.NATIVE_DELTA,
            VisionInputMode.NONE,
            {"structured_output"},
        ),
        (
            "kestrel_llm_vertex",
            "VertexAIAdapter",
            True,
            StructuredOutputMode.PROVIDER_NATIVE,
            ToolStreamingMode.NONSTREAM_FALLBACK,
            VisionInputMode.GEMINI_INLINE_DATA,
            {"tools", "vision", "structured_output"},
        ),
        (
            "kestrel_llm_bedrock",
            "BedrockAdapter",
            True,
            StructuredOutputMode.TOOL_FORCED,
            ToolStreamingMode.NATIVE_DELTA,
            VisionInputMode.PROVIDER_NATIVE,
            {"tools", "vision", "structured_output", "streaming"},
        ),
    ],
)
def test_provider_plugins_declare_capabilities(
    module_name,
    class_name,
    supports_vision,
    structured_output_mode,
    tool_streaming_mode,
    vision_input_mode,
    model_dependent,
):
    module = __import__(module_name)
    adapter = getattr(module, class_name)()

    capabilities = adapter.provider_capabilities()

    assert capabilities.supports_tools is True
    assert capabilities.supports_streaming is True
    assert capabilities.supports_vision is supports_vision
    assert capabilities.supports_structured_output is True
    assert capabilities.structured_output_mode == structured_output_mode
    assert capabilities.tool_streaming_mode == tool_streaming_mode
    assert capabilities.vision_input_mode == vision_input_mode
    assert set(capabilities.model_dependent) == model_dependent
    assert capabilities.notes


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
        _chunk(SimpleNamespace(reasoning_content="Need health. ", content=None, tool_calls=None)),
        _chunk(SimpleNamespace(reasoning_content="Use lookup.", content=None, tool_calls=None)),
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
    assert final.raw == {"reasoning_content": "Need health. Use lookup.Need lookup."}
    assert final.input_tokens == 5
    assert final.output_tokens == 7
    assert final.total_tokens == 12
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


async def test_vertex_get_response_builds_generation_config_with_tools_and_schema():
    from kestrel_llm_vertex import VertexAIAdapter

    class Answer(BaseModel):
        ok: bool

    adapter = VertexAIAdapter()
    captured_kwargs = {}
    raw = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text=None, function_call=SimpleNamespace(
                            id="call_1",
                            name="lookup",
                            args={"q": "kestrel"},
                        ))
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=2,
            candidates_token_count=3,
            total_token_count=5,
        ),
    )

    async def generate_content(**kwargs):
        captured_kwargs.update(kwargs)
        return raw

    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    )

    response = await adapter.get_response(
        client,
        "gemini-test",
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "health?"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup health",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ],
        response_format=Answer,
        max_tokens=20,
        temperature=0,
    )

    assert captured_kwargs["model"] == "gemini-test"
    assert captured_kwargs["contents"] == [{"role": "user", "parts": [{"text": "health?"}]}]
    config = captured_kwargs["config"]
    assert config["system_instruction"] == "Be brief."
    assert config["max_output_tokens"] == 20
    assert config["temperature"] == 0
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"]["title"] == "Answer"
    assert config["tools"] == [
        {
            "function_declarations": [
                {
                    "name": "lookup",
                    "description": "Lookup health",
                    "parameters_json_schema": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                }
            ]
        }
    ]
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == {"q": "kestrel"}
    assert response.input_tokens == 2
    assert response.output_tokens == 3
    assert response.total_tokens == 5


def test_vertex_tool_schema_keys_validate_with_google_genai():
    from google.genai import types
    from kestrel_llm_vertex import _generation_config

    config = _generation_config(
        system_prompt=None,
        format=None,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup health",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ],
        response_format=None,
        kwargs={},
    )

    sdk_config = types.GenerateContentConfig(**config)

    assert sdk_config.model_dump(exclude_none=True, by_alias=True)["tools"] == [
        {
            "functionDeclarations": [
                {
                    "description": "Lookup health",
                    "name": "lookup",
                    "parametersJsonSchema": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                }
            ]
        }
    ]


def test_vertex_extract_response_tolerates_text_property_errors():
    from kestrel_llm_vertex import _extract_response

    class EmptyResponse:
        candidates = []
        usage_metadata = None

        @property
        def text(self):
            raise ValueError("No text parts")

    response = _extract_response(EmptyResponse())

    assert response.content is None
    assert response.tool_calls is None


async def test_vertex_streaming_with_tools_uses_nonstreaming_handoff():
    from kestrel_llm_vertex import VertexAIAdapter

    adapter = VertexAIAdapter()
    response = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="lookup", arguments={})],
    )

    async def get_response(*args, **kwargs):
        return response

    adapter.get_response = get_response

    items = [
        item
        async for item in adapter.get_streaming_response_with_tools(
            MagicMock(),
            "model",
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
        )
    ]

    assert items[0] == ToolCallStarted(index=0, id="call_1", name="lookup")
    assert items[1] is response


class _AsyncModelList:
    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


async def test_vertex_list_models_uses_async_client():
    from kestrel_llm_vertex import VertexAIAdapter

    async def list_models():
        return _AsyncModelList(
            [
                SimpleNamespace(
                    name="publishers/google/models/gemini-2.5-flash",
                    display_name="Gemini 2.5 Flash",
                    description="Fast Gemini model",
                    input_token_limit=1000000,
                ),
                SimpleNamespace(
                    name="publishers/google/models/text-embedding-005",
                    display_name="Text Embedding 005",
                    description="Embedding model",
                    input_token_limit=2048,
                ),
                SimpleNamespace(
                    name="publishers/google/models/imagen-4.0-generate-preview",
                    display_name="Imagen 4",
                    description="Image model",
                    input_token_limit=None,
                ),
            ]
        )

    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(list=list_models)),
        models=SimpleNamespace(list=MagicMock(side_effect=AssertionError("sync list used"))),
    )

    models = await VertexAIAdapter().list_models(client)

    assert len(models) == 3
    assert models[0].id == "publishers/google/models/gemini-2.5-flash"
    assert models[0].provider == "vertex_ai"
    assert models[0].supports_vision is True
    assert models[0].supports_tools is True
    assert models[0].supports_streaming is True
    assert models[1].category == ModelCategory.EMBEDDING
    assert models[1].supports_tools is False
    assert models[1].supports_streaming is False
    assert models[2].category == ModelCategory.IMAGE
    assert models[2].supports_tools is False
    assert models[2].supports_streaming is False


async def test_bedrock_get_response_builds_converse_payload_with_tools_and_schema():
    from kestrel_llm_bedrock import BedrockAdapter

    class Answer(BaseModel):
        ok: bool

    client = MagicMock()
    client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "structured_1",
                            "name": "kestrel_structured_response",
                            "input": {"ok": True},
                        }
                    }
                ]
            }
        },
        "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
    }

    response = await BedrockAdapter().get_response(
        client,
        "anthropic.test",
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "health?"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup health",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ],
        response_format=Answer,
        max_tokens=20,
        temperature=0,
    )

    payload = client.converse.call_args.kwargs
    assert payload["modelId"] == "anthropic.test"
    assert payload["system"] == [{"text": "Be brief."}]
    assert payload["messages"] == [{"role": "user", "content": [{"text": "health?"}]}]
    assert payload["inferenceConfig"] == {"maxTokens": 20, "temperature": 0}
    assert payload["toolConfig"]["toolChoice"] == {
        "tool": {"name": "kestrel_structured_response"}
    }
    assert payload["toolConfig"]["tools"][0]["toolSpec"]["name"] == "lookup"
    assert payload["toolConfig"]["tools"][1]["toolSpec"]["name"] == "kestrel_structured_response"
    assert response.content == '{"ok": true}'
    assert response.input_tokens == 2
    assert response.output_tokens == 3
    assert response.total_tokens == 5


async def test_bedrock_streaming_with_tools_emits_marker_and_final_response():
    from kestrel_llm_bedrock import BedrockAdapter

    client = MagicMock()
    # A single-pass iterator (not a reusable list), so the stream is consumed
    # exactly like a real botocore EventStream — and so the F407 off-loop pump
    # is exercised rather than masked by a re-iterable list.
    client.converse_stream.return_value = {
        "stream": iter([
            {"contentBlockDelta": {"delta": {"text": "Checking. "}}},
            {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "start": {
                        "toolUse": {
                            "toolUseId": "call_1",
                            "name": "lookup",
                        }
                    },
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": '{"q"'}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": ':"kestrel"}'}},
                }
            },
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 7, "totalTokens": 12}}},
        ])
    }

    items = [
        item
        async for item in BedrockAdapter().get_streaming_response_with_tools(
            client,
            "model",
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
        )
    ]

    assert items[0] == "Checking. "
    assert items[1] == ToolCallStarted(index=1, id="call_1", name="lookup")
    final = items[-1]
    assert isinstance(final, LLMResponse)
    assert final.content == "Checking. "
    assert final.tool_calls[0].id == "call_1"
    assert final.tool_calls[0].name == "lookup"
    assert final.tool_calls[0].arguments == {"q": "kestrel"}
    assert final.input_tokens == 5
    assert final.output_tokens == 7
    assert final.total_tokens == 12


def test_bedrock_normalize_coalesces_parallel_tool_results():
    """F408: two tool messages after a 2-call assistant turn must collapse into
    ONE user message with two toolResult blocks, or Bedrock's role-alternation
    validation rejects the consecutive user messages."""
    messages = [
        {"role": "user", "content": "do two things"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "a", "arguments": {}}},
                {"id": "call_2", "type": "function",
                 "function": {"name": "b", "arguments": {}}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"r": 1}'},
        {"role": "tool", "tool_call_id": "call_2", "content": '{"r": 2}'},
    ]

    _system, normalized = normalize_bedrock_messages(messages)

    assert [m["role"] for m in normalized] == ["user", "assistant", "user"]
    tool_results = normalized[-1]["content"]
    assert [b["toolResult"]["toolUseId"] for b in tool_results] == ["call_1", "call_2"]


class _ThreadRecordingStream:
    """EventStream stand-in whose iteration records the thread it runs on, so a
    test can prove the (blocking) reads happen off the event loop (F407)."""

    def __init__(self, events):
        self._events = list(events)
        self._index = 0
        self.threads = []

    def __iter__(self):
        return self

    def __next__(self):
        self.threads.append(threading.current_thread())
        if self._index >= len(self._events):
            raise StopIteration
        event = self._events[self._index]
        self._index += 1
        return event


async def test_bedrock_streaming_iterates_off_the_event_loop():
    """F407: the botocore EventStream must be pumped in a worker thread, not
    iterated on the event loop where each blocking read freezes the host."""
    from kestrel_llm_bedrock import BedrockAdapter

    main_thread = threading.current_thread()
    stream = _ThreadRecordingStream(
        [{"contentBlockDelta": {"delta": {"text": "hi"}}}]
    )
    client = MagicMock()
    client.converse_stream.return_value = {"stream": stream}

    chunks = [
        chunk
        async for chunk in BedrockAdapter().get_streaming_response(
            client, "model", [{"role": "user", "content": "x"}]
        )
    ]

    assert chunks == ["hi"]
    assert stream.threads  # the stream was actually iterated
    assert all(t is not main_thread for t in stream.threads)


class _BlockingRecordingStream:
    """EventStream stand-in that yields some events then BLOCKS (like a live
    Bedrock stream mid-generation) until ``close()`` is called."""

    def __init__(self, initial_events):
        self._events = list(initial_events)
        self._index = 0
        self._gate = threading.Event()
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._events):
            event = self._events[self._index]
            self._index += 1
            return event
        # Emulate a still-open Bedrock request: block until closed.
        self._gate.wait(timeout=5.0)
        raise StopIteration

    def close(self):
        self.closed = True
        self._gate.set()


async def test_bedrock_streaming_closes_stream_on_early_exit():
    """F407 (codex P2): if the consumer stops early, the underlying stream is
    closed so the worker-thread pump stops draining the live Bedrock request."""
    from kestrel_llm_bedrock import BedrockAdapter

    stream = _BlockingRecordingStream(
        [{"contentBlockDelta": {"delta": {"text": "hi"}}}]
    )
    client = MagicMock()
    client.converse_stream.return_value = {"stream": stream}

    gen = BedrockAdapter().get_streaming_response(
        client, "model", [{"role": "user", "content": "x"}]
    )
    first = await gen.__anext__()
    assert first == "hi"

    await gen.aclose()  # simulate a disconnecting consumer

    assert stream.closed is True


async def test_bedrock_list_models_uses_foundation_model_api(monkeypatch):
    from kestrel_llm_bedrock import BedrockAdapter, BedrockClients

    runtime_client = MagicMock()
    bedrock_client = MagicMock()
    bedrock_client.list_foundation_models.return_value = {
        "modelSummaries": [
            {
                "modelId": "anthropic.claude-3-5-sonnet",
                "modelName": "Claude 3.5 Sonnet",
                "providerName": "Anthropic",
            },
            {
                "modelId": "amazon.titan-embed-text-v2:0",
                "modelName": "Titan Embed",
                "providerName": "Amazon",
            },
        ]
    }
    models = await BedrockAdapter().list_models(
        BedrockClients(runtime=runtime_client, management=bedrock_client, region="us-west-2")
    )

    assert [model.id for model in models] == [
        "anthropic.claude-3-5-sonnet",
        "amazon.titan-embed-text-v2:0",
    ]
    assert models[0].category == ModelCategory.CHAT
    assert models[0].supports_tools is True
    assert models[1].category == ModelCategory.EMBEDDING
    assert models[1].supports_tools is False


async def test_bedrock_list_models_rejects_raw_runtime_client():
    from kestrel_llm_bedrock import BedrockAdapter

    with pytest.raises(ValueError, match="BedrockClients"):
        await BedrockAdapter().list_models(object())


def test_registry_discovers_first_wave_plugins(monkeypatch):
    import kestrel_llm_bedrock
    import kestrel_llm_deepseek
    import kestrel_llm_kimi
    import kestrel_llm_vertex
    import kestrel_llm_xai

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("XAI_API_KEY", "sk-xai")
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-kimi")
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-google")

    entry_points = [
        _entry_point("deepseek", kestrel_llm_deepseek.DeepSeekAdapter),
        _entry_point("xai", kestrel_llm_xai.XAIAdapter),
        _entry_point("kimi", kestrel_llm_kimi.KimiAdapter),
        _entry_point("vertex_ai", kestrel_llm_vertex.VertexAIAdapter),
        _entry_point("bedrock", kestrel_llm_bedrock.BedrockAdapter),
    ]
    config = {
        "route_priority": ["deepseek:api", "xai:api", "kimi:api", "vertex_ai:api", "bedrock:api"],
        "vendors": {
            "deepseek": {"routes": {"api": {}}},
            "xai": {"routes": {"api": {}}},
            "kimi": {"routes": {"api": {}}},
            "vertex_ai": {"routes": {"api": {}}},
            "bedrock": {"routes": {"api": {"region": "us-west-2"}}},
        },
    }

    with (
        _patch_entry_points(entry_points),
        patch.object(kestrel_llm_deepseek.openai, "AsyncOpenAI", return_value=MagicMock()),
        patch.object(kestrel_llm_xai.openai, "AsyncOpenAI", return_value=MagicMock()),
        patch.object(kestrel_llm_kimi.openai, "AsyncOpenAI", return_value=MagicMock()),
        patch.object(kestrel_llm_vertex, "_load_genai") as load_genai,
        patch.object(kestrel_llm_bedrock, "_load_boto3") as load_boto3,
    ):
        load_genai.return_value = SimpleNamespace(Client=MagicMock(return_value=MagicMock()))
        fake_bedrock_session = MagicMock()
        fake_bedrock_session.client.return_value = MagicMock()
        load_boto3.return_value = SimpleNamespace(Session=MagicMock(return_value=fake_bedrock_session))
        providers = ProviderRegistry(config).initialize_providers()

    assert [provider.name for provider in providers] == [
        "deepseek:api",
        "xai:api",
        "kimi:api",
        "vertex_ai:api",
        "bedrock:api",
    ]
    assert {provider.name for provider in providers if provider.is_local} == set()
    assert {provider.name for provider in providers if provider.is_cloud} == {
        "deepseek:api",
        "xai:api",
        "kimi:api",
        "vertex_ai:api",
        "bedrock:api",
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
        ("kestrel-llm-vertex", "vertex_ai"),
        ("kestrel-llm-bedrock", "bedrock"),
    ],
)
def test_provider_pyprojects_keep_plugin_contract(package, entry_point):
    pyproject = _load_pyproject(package)
    project = pyproject["project"]

    assert project["name"] == package
    assert "kestrel-sovereign-sdk>=0.17.0,<1" in project["dependencies"]
    if package == "kestrel-llm-vertex":
        assert "google-genai>=1.75.0,<3" in project["dependencies"]
    elif package == "kestrel-llm-bedrock":
        assert "boto3>=1.34,<2" in project["dependencies"]
    else:
        assert "kestrel-llm-openai-compat>=0.1.7,<0.2" in project["dependencies"]
    assert not any(dep.startswith("kestrel_sovereign") for dep in project["dependencies"])

    entry_points = pyproject["project"]["entry-points"][LLM_PROVIDER_ENTRY_POINT_GROUP]
    assert list(entry_points) == [entry_point]


def test_meta_package_extras_track_first_wave_packages():
    pyproject = _load_pyproject("kestrel-llms")
    extras = pyproject["project"]["optional-dependencies"]

    assert set(extras) == {"deepseek", "xai", "kimi", "vertex", "bedrock", "cloud", "all"}
    assert extras["bedrock"] == ["kestrel-llm-bedrock>=0.1.0,<0.2"]
    assert extras["deepseek"] == ["kestrel-llm-deepseek>=0.1.8,<0.2"]
    assert extras["xai"] == ["kestrel-llm-xai>=0.1.8,<0.2"]
    assert extras["kimi"] == ["kestrel-llm-kimi>=0.1.8,<0.2"]
    assert extras["vertex"] == ["kestrel-llm-vertex>=0.1.0,<0.2"]
    assert set(extras["cloud"]) == {
        "kestrel-llm-bedrock>=0.1.0,<0.2",
        "kestrel-llm-deepseek>=0.1.8,<0.2",
        "kestrel-llm-xai>=0.1.8,<0.2",
        "kestrel-llm-kimi>=0.1.8,<0.2",
        "kestrel-llm-vertex>=0.1.0,<0.2",
    }
    assert set(extras["all"]) == set(extras["cloud"])


def test_shared_openai_compat_package_has_no_entry_point():
    pyproject = _load_pyproject("kestrel-llm-openai-compat")
    project = pyproject["project"]

    assert project["name"] == "kestrel-llm-openai-compat"
    assert "kestrel-sovereign-sdk>=0.17.0,<1" in project["dependencies"]
    assert "entry-points" not in pyproject["project"]
