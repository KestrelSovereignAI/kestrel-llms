"""Native tool calls must retain their identity and contain a JSON object."""

import json

import pytest

from kestrel_sdk.llm import LLMProtocolError, LLMResponse
from kestrel_sdk.testing import drain_streaming_with_tools
from kestrel_llm_deepseek import DeepSeekAdapter
from kestrel_llm_xai import XAIAdapter
from kestrel_llm_kimi import KimiAdapter
from kestrel_llm_bedrock import BedrockAdapter
from completion_peer import chunk, openai_response, owned_client, owned_bedrock_client


ADAPTERS = [DeepSeekAdapter, XAIAdapter, KimiAdapter]


def openai_tool_frames(call_id, name, arguments, streaming):
    call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
    if streaming:
        return [chunk({"tool_calls": [{"index": 0, **call}]}, reason="tool_calls")]
    response = openai_response("tool_calls").model_dump(mode="json")
    response["choices"][0]["message"] = {"role": "assistant", "content": None, "tool_calls": [call]}
    return [response]


@pytest.mark.parametrize("adapter_type", ADAPTERS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("call_id,name,arguments", [
    ("", "write", "{}"), ("owned", "", "{}"),
    ("owned", "write", "[]"), ("owned", "write", "null"),
    ("owned", "write", '"text"'), ("owned", "write", "{"),
    ("owned", "write", ""),
])
async def test_openai_native_malformed_tool_calls_never_complete(native_peer, adapter_type, streaming, call_id, name, arguments):
    observed = []
    with native_peer(openai_tool_frames(call_id, name, arguments, streaming)) as peer:
        async with owned_client(peer) as client:
            with pytest.raises(LLMProtocolError):
                if streaming:
                    async for item in adapter_type().get_streaming_response_with_tools(client, "owned-model", []):
                        observed.append(item)
                else:
                    await adapter_type().get_response(client, "owned-model", [])
            assert not any(isinstance(item, LLMResponse) for item in observed)
            assert peer.responses and all(response.is_closed for response in peer.responses)


@pytest.mark.parametrize("adapter_type", ADAPTERS)
@pytest.mark.parametrize("second_id,second_name", [("owned", "write"), ("other", "write"), ("owned", "other")])
async def test_openai_native_tool_identity_is_not_concatenated(native_peer, adapter_type, second_id, second_name):
    frames = [
        chunk({"tool_calls": [{"index": 0, "id": "owned", "type": "function",
            "function": {"name": "write", "arguments": '{"value":'}}]}),
        chunk({"tool_calls": [{"index": 0, "id": second_id, "type": "function",
            "function": {"name": second_name, "arguments": '"owned"}'}}]}, reason="tool_calls"),
    ]
    with native_peer(frames) as peer:
        async with owned_client(peer) as client:
            stream = adapter_type().get_streaming_response_with_tools(client, "owned-model", [])
            if (second_id, second_name) == ("owned", "write"):
                result = await drain_streaming_with_tools(stream)
                call, = result.final_response.tool_calls
                assert (call.id, call.name, call.arguments) == ("owned", "write", {"value": "owned"})
            else:
                with pytest.raises(LLMProtocolError):
                    _ = [item async for item in stream]
            assert all(response.is_closed for response in peer.responses)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("arguments", [None, [], "text", 17, True])
async def test_bedrock_native_non_object_arguments_are_refused(native_peer, streaming, arguments):
    if streaming:
        frames = [
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "owned", "name": "write"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": json.dumps(arguments)}}}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
    else:
        frames = [{"output": {"message": {"role": "assistant", "content": [{"toolUse": {
            "toolUseId": "owned", "name": "write", "input": arguments}}]}},
            "stopReason": "tool_use", "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "metrics": {"latencyMs": 1}}]
    with native_peer(frames, wire="bedrock", json_response=not streaming) as peer:
        async with owned_bedrock_client(peer) as client:
            with pytest.raises(LLMProtocolError):
                if streaming:
                    _ = [item async for item in BedrockAdapter().get_streaming_response_with_tools(client, "owned-model", [{"role": "user", "content": "owned"}])]
                else:
                    await BedrockAdapter().get_response(client, "owned-model", [{"role": "user", "content": "owned"}])


@pytest.mark.parametrize("call_id,name", [(None, "write"), ("", "write"), ("owned", None), ("owned", "")])
async def test_bedrock_native_missing_identity_is_not_invented(native_peer, call_id, name):
    frames = [{"output": {"message": {"role": "assistant", "content": [{"toolUse": {
        "toolUseId": call_id, "name": name, "input": {}}}]}}, "stopReason": "tool_use"}]
    with native_peer(frames, wire="bedrock", json_response=True) as peer:
        async with owned_bedrock_client(peer) as client:
            with pytest.raises(LLMProtocolError):
                await BedrockAdapter().get_response(client, "owned-model", [{"role": "user", "content": "owned"}])


@pytest.mark.parametrize("arguments", ["{", ""])
async def test_bedrock_native_invalid_json_does_not_become_raw_arguments(native_peer, arguments):
    frames = [
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "owned", "name": "write"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": arguments}}}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]
    with native_peer(frames, wire="bedrock") as peer:
        async with owned_bedrock_client(peer) as client:
            with pytest.raises(LLMProtocolError):
                _ = [item async for item in BedrockAdapter().get_streaming_response_with_tools(client, "owned-model", [{"role": "user", "content": "owned"}])]


@pytest.mark.parametrize("adapter_type", ADAPTERS)
@pytest.mark.parametrize("streaming", [False, True])
async def test_native_raw_key_remains_ordinary_object_data(native_peer, adapter_type, streaming):
    arguments = {"_raw": "native data"}
    with native_peer(openai_tool_frames("owned", "write", json.dumps(arguments), streaming)) as peer:
        async with owned_client(peer) as client:
            if streaming:
                result = await drain_streaming_with_tools(adapter_type().get_streaming_response_with_tools(client, "owned-model", []))
                response = result.final_response
            else:
                response = await adapter_type().get_response(client, "owned-model", [])
            call, = response.tool_calls
            assert call.arguments == arguments
