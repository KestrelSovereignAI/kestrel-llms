"""Real OpenAI SDK HTTP/SSE parsing against an owned loopback protocol peer."""

import asyncio

from openai import APIError
import pytest

from kestrel_sdk.llm import LLMProtocolError, LLMResponse
from kestrel_sdk.testing import assert_response_contract, drain_streaming_with_tools
from kestrel_llm_deepseek import DeepSeekAdapter
from kestrel_llm_kimi import KimiAdapter
from kestrel_llm_xai import XAIAdapter
from completion_peer import chunk, owned_client, openai_response

ADAPTERS = [DeepSeekAdapter, XAIAdapter, KimiAdapter]


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", ADAPTERS)
@pytest.mark.parametrize("method", ["get_streaming_response", "get_streaming_response_with_tools"])
@pytest.mark.parametrize("reason", ["stop", "length", "content_filter", "future_provider_reason", None])
async def test_native_text_stream_preserves_reason_and_usage(native_peer, adapter_type, method, reason):
    frames = [chunk({"content": "owned "}), chunk({"content": "text"}, reason=reason),
              chunk(usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})]
    with native_peer(frames) as peer:
        async with owned_client(peer) as client:
            result = await drain_streaming_with_tools(
                getattr(adapter_type(), method)(client, "owned-model", [{"role": "user", "content": "owned"}]))
            assert result.text == "owned text"
            assert result.final_response.content == "owned text"
            assert result.final_response.stop_reason == reason
            assert (result.final_response.input_tokens, result.final_response.output_tokens,
                    result.final_response.total_tokens) == (2, 3, 5)
            assert peer.responses and all(response.is_closed for response in peer.responses)
        assert len(peer.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", ADAPTERS)
async def test_native_usage_trailers_and_repeated_reason_do_not_erase_stop(native_peer, adapter_type):
    frames = [chunk({"content": "text"}, reason="stop"), chunk({}, reason="stop"), chunk({}),
              chunk(usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})]
    with native_peer(frames) as peer:
        async with owned_client(peer) as client:
            result = await drain_streaming_with_tools(
                adapter_type().get_streaming_response_with_tools(client, "owned-model", []))
            assert result.final_response.stop_reason == "stop"
            assert result.final_response.total_tokens == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", ADAPTERS)
@pytest.mark.parametrize("frames", [
    [chunk({}, reason="length"), chunk({}, reason="stop")],
    [chunk({}, reason="stop"), chunk({"content": "trailing"})],
    [chunk({}, reason="stop"), chunk({"reasoning_content": "trailing"})],
    [chunk({}, reason="stop"), chunk({"tool_calls": [{"index": 0, "id": "owned", "function": {"name": "write", "arguments": "{}"}}]})],
    [chunk({"content": "alternative"}, index=1)],
    [chunk({}, reason=17)],
    [chunk({}, reason="")],
])
async def test_native_contradictory_stream_never_emits_a_terminal(native_peer, adapter_type, frames):
    observed = []
    with native_peer(frames) as peer:
        async with owned_client(peer) as client:
            with pytest.raises(LLMProtocolError):
                async for item in adapter_type().get_streaming_response_with_tools(client, "owned-model", []):
                    observed.append(item)
            assert not any(isinstance(item, LLMResponse) for item in observed)
            assert peer.responses and all(response.is_closed for response in peer.responses)
        assert len(peer.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", ADAPTERS)
async def test_plain_requested_tools_are_refused_before_wire(native_peer, adapter_type):
    with native_peer([chunk({}, reason="stop")]) as peer:
        async with owned_client(peer) as client:
            with pytest.raises(LLMProtocolError):
                async for _ in adapter_type().get_streaming_response(
                    client, "owned-model", [], tools=[{"type": "function", "function": {"name": "write"}}]):
                    pass
        assert peer.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", ADAPTERS)
async def test_plain_unrequested_tool_intent_is_refused(native_peer, adapter_type):
    frames = [chunk({"tool_calls": [{"index": 0, "id": "call", "type": "function",
                                    "function": {"name": "write", "arguments": "{}"}}]}, reason="tool_calls")]
    with native_peer(frames) as peer:
        async with owned_client(peer) as client:
            with pytest.raises(LLMProtocolError):
                async for _ in adapter_type().get_streaming_response(client, "owned-model", []):
                    pass
            assert peer.responses and all(response.is_closed for response in peer.responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", ADAPTERS)
async def test_sparse_tool_indices_preserve_marker_stream_order(native_peer, adapter_type):
    frames = [
        chunk({"tool_calls": [{"index": index, "id": str(index), "type": "function",
                               "function": {"name": "write_" + str(index), "arguments": "{}"}}]})
        for index in [2, 0]
    ] + [chunk({}, reason="tool_calls")]
    with native_peer(frames) as peer:
        async with owned_client(peer) as client:
            result = await drain_streaming_with_tools(
                adapter_type().get_streaming_response_with_tools(client, "owned-model", []))
            assert [marker.index for marker in result.tool_starts] == [2, 0]
            assert [call.id for call in result.final_response.tool_calls] == ["2", "0"]
            assert result.final_response.stop_reason == "tool_calls"


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", ADAPTERS)
async def test_native_partial_stream_is_closed_before_early_close_returns(native_peer, adapter_type):
    with native_peer([chunk({"content": "partial"}), chunk({}, reason="stop")], pause_after_first=True) as peer:
        async with owned_client(peer) as client:
            stream = adapter_type().get_streaming_response_with_tools(client, "owned-model", [])
            assert await anext(stream) == "partial"
            await stream.aclose()
            assert peer.responses and all(response.is_closed for response in peer.responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", ADAPTERS)
async def test_native_error_is_propagated_without_terminal(native_peer, adapter_type):
    observed = []
    frames = [chunk({"content": "partial"}), {"error": {"type": "owned_failure", "message": "owned stream failure"}}]
    with native_peer(frames) as peer:
        async with owned_client(peer) as client:
            with pytest.raises(APIError):
                async for item in adapter_type().get_streaming_response_with_tools(client, "owned-model", []):
                    observed.append(item)
            assert not any(isinstance(item, LLMResponse) for item in observed)
            assert peer.responses and all(response.is_closed for response in peer.responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", ADAPTERS)
@pytest.mark.parametrize("reason", ["stop", "length", "tool_calls", "content_filter", "function_call"])
async def test_native_nonstream_reason_and_usage_survive_http(native_peer, adapter_type, reason):
    with native_peer([openai_response(reason).model_dump(mode="json")]) as peer:
        async with owned_client(peer) as client:
            response = await adapter_type().get_response(client, "owned-model", [])
            assert response.stop_reason == reason
            assert (response.input_tokens, response.output_tokens, response.total_tokens) == (2, 3, 5)
            assert_response_contract(response)
        assert len(peer.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", ADAPTERS)
@pytest.mark.parametrize("method", ["get_streaming_response", "get_streaming_response_with_tools"])
async def test_empty_native_stream_has_unknown_terminal_evidence(native_peer, adapter_type, method):
    with native_peer([]) as peer:
        async with owned_client(peer) as client:
            result = await drain_streaming_with_tools(getattr(adapter_type(), method)(client, "owned-model", []))
            assert result.text == ""
            assert result.final_response.stop_reason is None
            assert result.final_response.tool_calls is None


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", ADAPTERS)
@pytest.mark.parametrize("reason", ["tool_calls", "function_call"])
async def test_plain_native_tool_reason_without_calls_is_refused(native_peer, adapter_type, reason):
    with native_peer([chunk({}, reason=reason)]) as peer:
        async with owned_client(peer) as client:
            with pytest.raises(LLMProtocolError):
                async for _ in adapter_type().get_streaming_response(client, "owned-model", []):
                    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", ADAPTERS)
async def test_native_cancel_closes_response_before_propagating(native_peer, adapter_type):
    with native_peer([chunk({"content": "partial"}), chunk({}, reason="stop")], pause_after_first=True) as peer:
        async with owned_client(peer) as client:
            stream = adapter_type().get_streaming_response_with_tools(client, "owned-model", [])
            assert await anext(stream) == "partial"
            started = asyncio.Event()

            async def read_next():
                started.set()
                return await anext(stream)

            task = asyncio.create_task(read_next())
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert peer.responses and all(response.is_closed for response in peer.responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", ADAPTERS)
async def test_native_close_failure_cannot_produce_terminal(native_peer, adapter_type, monkeypatch):
    failure = RuntimeError("owned native close failure")

    async def inject_close_failure(response):
        close = response.aclose

        async def fail_after_close():
            await close()
            raise failure

        monkeypatch.setattr(response, "aclose", fail_after_close)

    with native_peer([chunk({"content": "text"}, reason="stop")]) as peer:
        async with owned_client(peer, after_response=inject_close_failure) as client:
            observed = []
            with pytest.raises(RuntimeError) as raised:
                async for item in adapter_type().get_streaming_response_with_tools(client, "owned-model", []):
                    observed.append(item)
            assert raised.value is failure
            assert not any(isinstance(item, LLMResponse) for item in observed)
            assert peer.responses and all(response.is_closed for response in peer.responses)
