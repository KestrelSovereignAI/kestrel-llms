"""Native Google SSE and signed AWS binary event-stream completion evidence."""

import asyncio
import json

from pydantic import BaseModel
import pytest

from kestrel_sdk.llm import LLMProtocolError, LLMResponse
from kestrel_sdk.testing import drain_streaming_with_tools
from kestrel_llm_vertex import VertexAIAdapter
from kestrel_llm_bedrock import BedrockAdapter
from completion_peer import owned_vertex_client, owned_bedrock_client


def vertex_chunk(text=None, reason=None, usage=None):
    response = {}
    if text is not None or reason is not None:
        candidate = {"index": 0, "content": {"role": "model", "parts": [] if text is None else [{"text": text}]}}
        if reason is not None:
            candidate["finishReason"] = reason
        response["candidates"] = [candidate]
    if usage is not None:
        response["usageMetadata"] = usage
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get_streaming_response", "get_streaming_response_with_tools"])
@pytest.mark.parametrize("reason", ["STOP", "MAX_TOKENS", "SAFETY", "future_reason", None])
async def test_vertex_native_stream_reason_and_usage(native_peer, method, reason):
    frames = [vertex_chunk("owned "), vertex_chunk("text", reason), vertex_chunk(usage={
        "promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5})]
    with native_peer(frames, wire="vertex") as peer:
        async with owned_vertex_client(peer) as client:
            result = await drain_streaming_with_tools(getattr(VertexAIAdapter(), method)(
                client, "owned-model", [{"role": "user", "content": "owned"}]))
            assert result.text == "owned text"
            assert result.final_response.stop_reason == reason
            assert result.final_response.total_tokens == 5
            assert peer.responses and all(response.is_closed for response in peer.responses)
        assert len(peer.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get_streaming_response", "get_streaming_response_with_tools"])
@pytest.mark.parametrize("reason", ["end_turn", "max_tokens", "guardrail_intervened", "future_reason", None])
async def test_bedrock_native_stream_reason_and_usage(native_peer, method, reason):
    frames = [{"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "owned text"}}}]
    if reason is not None:
        frames.append({"messageStop": {"stopReason": reason}})
    frames.append({"metadata": {"usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5}}})
    with native_peer(frames, wire="bedrock") as peer:
        async with owned_bedrock_client(peer) as client:
            result = await drain_streaming_with_tools(getattr(BedrockAdapter(), method)(
                client, "owned-model", [{"role": "user", "content": "owned"}]))
            assert result.text == "owned text"
            assert result.final_response.stop_reason == reason
            assert result.final_response.total_tokens == 5
        assert len(peer.calls) == 1 and peer.calls[0]["aws_signed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("frames", [
    [{"messageStop": {"stopReason": "end_turn"}}, {"messageStop": {"stopReason": "end_turn"}}],
    [{"messageStop": {"stopReason": "end_turn"}}, {"contentBlockDelta": {"delta": {"text": "trailing"}}}],
])
async def test_bedrock_native_contradiction_is_not_a_terminal(native_peer, frames):
    observed = []
    with native_peer(frames, wire="bedrock") as peer:
        async with owned_bedrock_client(peer) as client:
            with pytest.raises(LLMProtocolError):
                async for item in BedrockAdapter().get_streaming_response_with_tools(
                    client, "owned-model", [{"role": "user", "content": "owned"}]):
                    observed.append(item)
    assert not any(isinstance(item, LLMResponse) for item in observed)


@pytest.mark.asyncio
async def test_vertex_native_contradiction_is_not_a_terminal(native_peer):
    observed = []
    with native_peer([vertex_chunk(reason="STOP"), vertex_chunk("trailing")], wire="vertex") as peer:
        async with owned_vertex_client(peer) as client:
            with pytest.raises(LLMProtocolError):
                async for item in VertexAIAdapter().get_streaming_response_with_tools(
                    client, "owned-model", [{"role": "user", "content": "owned"}]):
                    observed.append(item)
    assert not any(isinstance(item, LLMResponse) for item in observed)


@pytest.mark.asyncio
async def test_vertex_early_close_releases_the_native_response(native_peer):
    with native_peer([vertex_chunk("partial"), vertex_chunk(reason="STOP")],
                     wire="vertex", pause_after_first=True) as peer:
        async with owned_vertex_client(peer) as client:
            stream = VertexAIAdapter().get_streaming_response(client, "owned-model", [{"role": "user", "content": "owned"}])
            assert await anext(stream) == "partial"
            await stream.aclose()
            assert peer.responses and all(response.is_closed for response in peer.responses)
            assert not peer.pause_timed_out.is_set()


@pytest.mark.asyncio
async def test_bedrock_early_close_releases_the_native_response(native_peer):
    frames = [{"contentBlockDelta": {"delta": {"text": "partial"}}}, {"messageStop": {"stopReason": "end_turn"}}]
    with native_peer(frames, wire="bedrock", pause_after_first=True) as peer:
        async with owned_bedrock_client(peer) as client:
            stream = BedrockAdapter().get_streaming_response(client, "owned-model", [{"role": "user", "content": "owned"}])
            assert await anext(stream) == "partial"
            await stream.aclose()
            assert peer.raw_responses and all(response.closed for response in peer.raw_responses)
            assert not peer.pause_timed_out.is_set()


@pytest.mark.asyncio
async def test_bedrock_structured_projection_preserves_native_evidence(native_peer):
    class Payload(BaseModel):
        value: str

    native = {
        "output": {"message": {"role": "assistant", "content": [{"toolUse": {
            "toolUseId": "owned-schema", "name": "kestrel_structured_response", "input": {"value": "owned"}}}]}},
        "stopReason": "tool_use", "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
        "metrics": {"latencyMs": 1},
    }
    with native_peer([native], wire="bedrock", json_response=True) as peer:
        async with owned_bedrock_client(peer) as client:
            result = await BedrockAdapter().get_response(client, "owned-model", [{"role": "user", "content": "owned"}], response_format=Payload)
            assert json.loads(result.content) == {"value": "owned"}
            assert result.tool_calls is None
            assert result.stop_reason == "tool_use"
            assert (result.input_tokens, result.output_tokens, result.total_tokens) == (2, 3, 5)
        assert len(peer.calls) == 1 and peer.calls[0]["aws_signed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get_streaming_response", "get_streaming_response_with_tools"])
async def test_bedrock_streamed_schema_is_refused_before_wire(native_peer, method):
    class Payload(BaseModel):
        value: str

    with native_peer([], wire="bedrock") as peer:
        async with owned_bedrock_client(peer) as client:
            with pytest.raises(LLMProtocolError):
                async for _ in getattr(BedrockAdapter(), method)(client, "owned-model", [{"role": "user", "content": "owned"}], response_format=Payload):
                    pass
        assert peer.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vertex", "bedrock"])
async def test_backend_plain_requested_tools_are_refused_before_wire(native_peer, backend):
    with native_peer([], wire=backend) as peer:
        if backend == "vertex":
            async with owned_vertex_client(peer) as client:
                with pytest.raises(LLMProtocolError):
                    async for _ in VertexAIAdapter().get_streaming_response(client, "owned-model", [], tools=[{}]):
                        pass
        else:
            async with owned_bedrock_client(peer) as client:
                with pytest.raises(LLMProtocolError):
                    async for _ in BedrockAdapter().get_streaming_response(client, "owned-model", [], tools=[{}]):
                        pass
        assert peer.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("frames", [
    [{"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": "{}"}}}}],
    [{"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "owned", "name": "write"}}}}] * 2,
    [{"contentBlockStart": {"start": {"toolUse": {"toolUseId": "owned", "name": "write"}}}}],
    [{"contentBlockStart": {"contentBlockIndex": -1, "start": {"toolUse": {"toolUseId": "owned", "name": "write"}}}}],
    [{"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {"name": "write"}}}}],
    [{"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "owned"}}}}],
])
async def test_bedrock_native_tool_blocks_reject_missing_or_duplicate_start(native_peer, frames):
    observed = []
    with native_peer(frames, wire="bedrock") as peer:
        async with owned_bedrock_client(peer) as client:
            with pytest.raises(LLMProtocolError, match="existing native tool block|new non-negative integer index|native id and name"):
                async for item in BedrockAdapter().get_streaming_response_with_tools(client, "owned-model", [{"role": "user", "content": "owned"}]):
                    observed.append(item)
            assert not any(isinstance(item, LLMResponse) for item in observed)
            assert peer.raw_responses and all(response.closed for response in peer.raw_responses)
