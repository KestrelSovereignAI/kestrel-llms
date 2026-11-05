from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel
from completion_peer import chunk, owned_client, openai_response, owned_vertex_client, owned_bedrock_client
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


from kestrel_llm_openai_compat import (  # noqa: E402
    normalize_messages,
    openai_compatible_capabilities,
    to_llm_response,
)
from kestrel_llm_kimi import normalize_kimi_messages  # noqa: E402
from kestrel_llm_vertex import normalize_vertex_messages  # noqa: E402
from kestrel_llm_bedrock import normalize_bedrock_messages  # noqa: E402


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


async def test_vertex_factory_uses_api_key_mode(monkeypatch):
    from kestrel_llm_vertex import VertexAIAdapter, VertexClient
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-google")
    info = VertexAIAdapter.create_provider({})
    try:
        assert info.name == "vertex_ai:api"
        assert info.vendor == "vertex_ai" and info.route == "api"
        assert isinstance(info.client, VertexClient)
        assert info.model == "gemini-2.5-flash"
        assert info.is_cloud and not info.is_local
    finally:
        await info.client.close()


async def test_vertex_factory_accepts_gemini_api_key_alias(monkeypatch):
    from kestrel_llm_vertex import VertexAIAdapter
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini")
    info = VertexAIAdapter.create_provider({})
    try:
        assert info.client.genai.aio is not None
    finally:
        await info.client.close()


async def test_vertex_factory_uses_vertex_project_mode(monkeypatch):
    import google.auth
    from google.auth.credentials import AnonymousCredentials
    from kestrel_llm_vertex import VertexAIAdapter
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "kestrel-project")
    monkeypatch.setattr(google.auth, "default", lambda **kwargs: (AnonymousCredentials(), "kestrel-project"))
    info = VertexAIAdapter.create_provider({"location": "us-east5"})
    try:
        assert info.client.genai.vertexai is True
    finally:
        await info.client.close()


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


async def test_to_llm_response_preserves_raw_provider_reasoning_object(native_peer):
    payload = openai_response().model_dump(mode="json")
    payload["choices"][0]["message"].update(content=None, reasoning_content="Need the tool.")
    with native_peer([payload]) as peer:
        async with owned_client(peer) as client:
            raw = await client.chat.completions.create(model="owned-model", messages=[])
            response = to_llm_response(raw)
    assert response.raw is raw
    assert response.raw.choices[0].message.reasoning_content == "Need the tool."
    assert response.stop_reason == "stop"


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
async def test_openai_compatible_plugins_stream_with_tools(module_name, class_name, native_peer):
    adapter = getattr(__import__(module_name), class_name)()
    frames = [
        chunk({"reasoning_content": "Need health. "}),
        chunk({"reasoning_content": "Use lookup."}),
        chunk({"content": "I'll check. "}),
        chunk({"reasoning_content": "Need lookup."}),
        chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                               "function": {"name": "lookup", "arguments": '{"q"'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": ':"kestrel"}'}}]}, reason="tool_calls"),
        chunk(usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}),
    ]
    with native_peer(frames) as peer:
        async with owned_client(peer) as client:
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
        captured_kwargs = peer.calls[0]["body"]

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
    module_name, class_name, native_peer,
):
    adapter = getattr(__import__(module_name), class_name)()
    payload = openai_response().model_dump(mode="json")
    payload["choices"][0]["message"]["content"] = "done"
    payload["usage"] = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    with native_peer([payload]) as peer:
        async with owned_client(peer) as client:
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
        captured_kwargs = peer.calls[0]["body"]

    assert response.content == "done"
    sent_arguments = captured_kwargs["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert sent_arguments == '{"q": "previous"}'
    if module_name == "kestrel_llm_kimi":
        assert captured_kwargs["messages"][0]["reasoning_content"] == ""
    else:
        assert "reasoning_content" not in captured_kwargs["messages"][0]


async def test_vertex_get_response_builds_generation_config_with_tools_and_schema(native_peer):
    from kestrel_llm_vertex import VertexAIAdapter
    class Answer(BaseModel):
        ok: bool
    raw = {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {
        "id": "call_1", "name": "lookup", "args": {"q": "kestrel"}}}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5}}
    with native_peer([raw], wire="vertex", json_response=True) as peer:
        async with owned_vertex_client(peer) as client:
            response = await VertexAIAdapter().get_response(client, "gemini-test", [
                {"role": "system", "content": "Be brief."}, {"role": "user", "content": "health?"}],
                tools=[{"type": "function", "function": {"name": "lookup", "description": "Lookup health",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}],
                response_format=Answer, max_tokens=20, temperature=0)
        payload = peer.calls[0]["body"]
        assert payload["contents"] == [{"role": "user", "parts": [{"text": "health?"}]}]
        assert payload["systemInstruction"]["parts"] == [{"text": "Be brief."}]
        config = payload["generationConfig"]
        assert config["maxOutputTokens"] == 20 and config["temperature"] == 0
        assert config["responseMimeType"] == "application/json"
        assert config["responseJsonSchema"]["title"] == "Answer"
        assert payload["tools"][0]["functionDeclarations"][0]["name"] == "lookup"
        assert response.tool_calls[0] == ToolCall(id="call_1", name="lookup", arguments={"q": "kestrel"})
        assert (response.input_tokens, response.output_tokens, response.total_tokens) == (2, 3, 5)
        assert response.stop_reason == "STOP"


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


async def test_vertex_streaming_with_tools_uses_nonstreaming_handoff(native_peer):
    from kestrel_llm_vertex import VertexAIAdapter
    raw = {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {
        "id": "call_1", "name": "lookup", "args": {}}}]}, "finishReason": "STOP"}]}
    with native_peer([raw], wire="vertex", json_response=True) as peer:
        async with owned_vertex_client(peer) as client:
            items = [item async for item in VertexAIAdapter().get_streaming_response_with_tools(
                client, "owned-model", [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "lookup"}}])]
        assert items[0] == ToolCallStarted(index=0, id="call_1", name="lookup")
        assert items[1].tool_calls[0].name == "lookup"
        assert items[1].stop_reason == "STOP"
        assert peer.calls[0]["path"].endswith(":generateContent")


async def test_vertex_list_models_uses_async_client(native_peer):
    from kestrel_llm_vertex import VertexAIAdapter
    raw = {"models": [
        {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash", "inputTokenLimit": 1000000},
        {"name": "models/text-embedding-005", "displayName": "Text Embedding 005"},
        {"name": "models/imagen-4.0-generate-preview", "displayName": "Imagen 4"}]}
    with native_peer([raw], wire="vertex", json_response=True) as peer:
        async with owned_vertex_client(peer) as client:
            models = await VertexAIAdapter().list_models(client)
        assert len(models) == 3
        assert models[0].id == "models/gemini-2.5-flash"
        assert models[0].supports_vision and models[0].supports_tools and models[0].supports_streaming
        assert models[1].category == ModelCategory.EMBEDDING and not models[1].supports_tools
        assert models[2].category == ModelCategory.IMAGE and not models[2].supports_streaming
        assert len(peer.calls) == 1


async def test_bedrock_get_response_builds_converse_payload_with_tools_and_schema(native_peer):
    from kestrel_llm_bedrock import BedrockAdapter
    class Answer(BaseModel):
        ok: bool
    raw = {"output": {"message": {"role": "assistant", "content": [{"toolUse": {
        "toolUseId": "structured_1", "name": "kestrel_structured_response", "input": {"ok": True}}}]}},
        "stopReason": "tool_use", "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
        "metrics": {"latencyMs": 1}}
    with native_peer([raw], wire="bedrock", json_response=True) as peer:
        async with owned_bedrock_client(peer) as client:
            response = await BedrockAdapter().get_response(client, "anthropic.test", [
                {"role": "system", "content": "Be brief."}, {"role": "user", "content": "health?"}],
                tools=[{"type": "function", "function": {"name": "lookup", "description": "Lookup health",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}],
                response_format=Answer, max_tokens=20, temperature=0)
        payload = peer.calls[0]["body"]
        assert payload["system"] == [{"text": "Be brief."}]
        assert payload["messages"] == [{"role": "user", "content": [{"text": "health?"}]}]
        assert payload["inferenceConfig"] == {"maxTokens": 20, "temperature": 0}
        assert payload["toolConfig"]["toolChoice"] == {"tool": {"name": "kestrel_structured_response"}}
        assert [t["toolSpec"]["name"] for t in payload["toolConfig"]["tools"]] == ["lookup", "kestrel_structured_response"]
        assert response.content == '{"ok": true}' and response.stop_reason == "tool_use"
        assert (response.input_tokens, response.output_tokens, response.total_tokens) == (2, 3, 5)


async def test_bedrock_streaming_with_tools_emits_marker_and_final_response(native_peer):
    from kestrel_llm_bedrock import BedrockAdapter
    raw = [
        {"contentBlockDelta": {"delta": {"text": "Checking. "}}},
        {"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "call_1", "name": "lookup"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"q"'}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": ':"kestrel"}'}}}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 7, "totalTokens": 12}}}]
    with native_peer(raw, wire="bedrock") as peer:
        async with owned_bedrock_client(peer) as client:
            items = [item async for item in BedrockAdapter().get_streaming_response_with_tools(
                client, "owned-model", [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "lookup"}}])]
        assert items[0] == "Checking. "
        assert items[1] == ToolCallStarted(index=1, id="call_1", name="lookup")
        final = items[-1]
        assert final.content == "Checking. " and final.stop_reason == "tool_use"
        assert final.tool_calls[0] == ToolCall(id="call_1", name="lookup", arguments={"q": "kestrel"})
        assert (final.input_tokens, final.output_tokens, final.total_tokens) == (5, 7, 12)
        assert "description" not in peer.calls[0]["body"]["toolConfig"]["tools"][0]["toolSpec"]


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


async def test_bedrock_list_models_uses_foundation_model_api(native_peer):
    from kestrel_llm_bedrock import BedrockAdapter
    raw = {"modelSummaries": [
        {"modelId": "anthropic.claude-3-5-sonnet", "modelName": "Claude 3.5 Sonnet", "providerName": "Anthropic"},
        {"modelId": "amazon.titan-embed-text-v2:0", "modelName": "Titan Embed", "providerName": "Amazon"}]}
    with native_peer([raw], wire="bedrock", json_response=True) as peer:
        async with owned_bedrock_client(peer) as client:
            models = await BedrockAdapter().list_models(client)
        assert [model.id for model in models] == ["anthropic.claude-3-5-sonnet", "amazon.titan-embed-text-v2:0"]
        assert models[0].category == ModelCategory.CHAT and models[0].supports_tools
        assert models[1].category == ModelCategory.EMBEDDING and not models[1].supports_tools
        assert peer.calls[0]["aws_signed"]


async def test_bedrock_list_models_rejects_raw_runtime_client():
    from kestrel_llm_bedrock import BedrockAdapter
    with pytest.raises(TypeError, match="owned clients"):
        await BedrockAdapter().list_models(object())
