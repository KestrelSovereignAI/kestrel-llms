"""Physical native SDK response ownership, cancellation and client shutdown."""

import asyncio
from contextlib import contextmanager

import pytest

import kestrel_llm_bedrock as bedrock
from kestrel_llm_vertex import VertexAIAdapter
from kestrel_sdk.llm import LLMResponse
from completion_peer import owned_bedrock_client, owned_vertex_client


MESSAGES = [{"role": "user", "content": "owned"}]


@contextmanager
def loop_errors():
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    errors = []
    loop.set_exception_handler(lambda loop, context: errors.append(context))
    try:
        yield errors
    finally:
        loop.set_exception_handler(previous)
        assert errors == []


def frames(backend):
    if backend == "vertex":
        return [{"candidates": [{"content": {"role": "model", "parts": [{"text": "partial"}]}}]},
                {"candidates": [{"finishReason": "STOP"}]}]
    return [{"contentBlockDelta": {"delta": {"text": "partial"}}},
            {"messageStop": {"stopReason": "end_turn"}}]


@pytest.mark.parametrize("backend,tls", [("vertex", False), ("bedrock", False), ("bedrock", True)])
async def test_cancel_native_blocked_read_joins_before_return(native_peer, monkeypatch, backend, tls):
    """The second task owns cancellation and then closes the suspended facade."""
    with native_peer(frames(backend), wire=backend, tls=tls, pause_after_first=True) as peer, loop_errors():
        factory = owned_vertex_client if backend == "vertex" else owned_bedrock_client
        adapter = VertexAIAdapter() if backend == "vertex" else bedrock.BedrockAdapter()
        async with factory(peer) as client:
            stream = adapter.get_streaming_response(client, "owned-model", MESSAGES)
            assert await anext(stream) == "partial"
            entered = asyncio.Event()
            if backend == "bedrock":
                submit = bedrock._submit
                loop = asyncio.get_running_loop()

                def observe_submit(function, *args, **kwargs):
                    if function is next:
                        def observed_next():
                            loop.call_soon_threadsafe(entered.set)
                            return function(*args, **kwargs)
                        return submit(observed_next)
                    return submit(function, *args, **kwargs)

                monkeypatch.setattr(bedrock, "_submit", observe_submit)

            async def read():
                if backend == "vertex":
                    entered.set()
                return await anext(stream)

            task = asyncio.create_task(read())
            await asyncio.wait_for(entered.wait(), 2)
            # The peer withholds the second frame; an event-loop ticker must run.
            for _ in range(4):
                await asyncio.sleep(0.01)
                assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            assert task.cancelled()
            await stream.aclose()
            responses = peer.responses if backend == "vertex" else peer.raw_responses
            assert responses and all(r.is_closed if backend == "vertex" else r.closed for r in responses)
            assert not peer.pause_timed_out.is_set()
            assert len(peer.calls) == 1
            await asyncio.sleep(0)


async def test_cancel_bedrock_start_waits_for_native_call_and_closes_capture(native_peer):
    with native_peer(frames("bedrock"), wire="bedrock", pause_before_headers=True) as peer, loop_errors():
        async with owned_bedrock_client(peer) as client:
            stream = bedrock.BedrockAdapter().get_streaming_response(client, "owned-model", MESSAGES)
            task = asyncio.create_task(anext(stream))
            await asyncio.wait_for(peer.headers_reached.wait(), 2)
            task.cancel()
            await asyncio.sleep(0.03)
            assert not task.done(), "Cancellation detached the native start call"
            peer.release_headers.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            assert task.cancelled()
            assert peer.raw_responses and all(r.closed for r in peer.raw_responses)
            assert not peer.pause_timed_out.is_set()
            await stream.aclose()


@pytest.mark.parametrize("backend", ["vertex", "bedrock"])
async def test_terminal_is_delivered_only_after_physical_close(native_peer, backend):
    with native_peer(frames(backend), wire=backend) as peer, loop_errors():
        factory = owned_vertex_client if backend == "vertex" else owned_bedrock_client
        adapter = VertexAIAdapter() if backend == "vertex" else bedrock.BedrockAdapter()
        async with factory(peer) as client:
            terminal = []
            async for item in adapter.get_streaming_response_with_tools(client, "owned-model", MESSAGES):
                if isinstance(item, LLMResponse):
                    responses = peer.responses if backend == "vertex" else peer.raw_responses
                    assert responses and all(r.is_closed if backend == "vertex" else r.closed for r in responses)
                    if backend == "bedrock":
                        assert "stream" not in item.raw
                    terminal.append(item)
            assert len(terminal) == 1


@pytest.mark.parametrize("backend", ["vertex", "bedrock"])
@pytest.mark.parametrize("operation", ["get_response", "get_streaming_response", "get_streaming_response_with_tools", "list_models"])
async def test_wrong_client_is_refused_before_wire(native_peer, backend, operation):
    with native_peer([], wire=backend) as peer:
        adapter = VertexAIAdapter() if backend == "vertex" else bedrock.BedrockAdapter()
        with pytest.raises(TypeError, match="owned client"):
            if operation == "list_models":
                await adapter.list_models(object())
            elif operation == "get_response":
                await adapter.get_response(object(), "owned-model", MESSAGES)
            else:
                await anext(getattr(adapter, operation)(object(), "owned-model", MESSAGES))
        assert peer.calls == []


async def test_vertex_automatic_mtls_is_refused(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "true")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="mTLS"):
        VertexAIAdapter.create_provider({"project_id": "owned-project"})


async def test_vertex_api_key_mode_does_not_use_automatic_mtls(native_peer, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "true")
    with native_peer([], wire="vertex") as peer:
        async with owned_vertex_client(peer) as client:
            assert not client.http.is_closed
        assert client.http.is_closed and peer.calls == []


async def test_owned_vertex_client_closes_its_transport(native_peer):
    with native_peer([], wire="vertex") as peer:
        async with owned_vertex_client(peer) as client:
            assert not client.http.is_closed
        assert client.http.is_closed


async def test_owned_bedrock_client_closes_both_pools(native_peer, monkeypatch):
    with native_peer([], wire="bedrock") as peer:
        calls = []
        async with owned_bedrock_client(peer) as client:
            for name in ("runtime", "management"):
                native = getattr(client, name)
                close = native.close

                def observe_close(close=close, name=name):
                    close()
                    calls.append(name)

                monkeypatch.setattr(native, "close", observe_close)
        assert calls == ["runtime", "management"]


@pytest.mark.parametrize("backend", ["vertex", "bedrock"])
async def test_native_cleanup_failure_prevents_terminal(native_peer, monkeypatch, backend):
    failure = RuntimeError("owned physical-close failure")
    observed = []
    with native_peer(frames(backend), wire=backend) as peer, loop_errors():
        factory = owned_vertex_client if backend == "vertex" else owned_bedrock_client
        adapter = VertexAIAdapter() if backend == "vertex" else bedrock.BedrockAdapter()
        async with factory(peer) as client:
            if backend == "vertex":
                async def inject(response):
                    close = response.aclose

                    async def fail_after_close():
                        await close()
                        raise failure

                    monkeypatch.setattr(response, "aclose", fail_after_close)

                client.http.event_hooks["response"].append(inject)
            else:
                def inject(http_response, **kwargs):
                    close = http_response.raw.close

                    def fail_after_close():
                        close()
                        raise failure

                    monkeypatch.setattr(http_response.raw, "close", fail_after_close)

                client.runtime.meta.events.register("after-call.bedrock-runtime.ConverseStream", inject)
            with pytest.raises(RuntimeError) as raised:
                async for item in adapter.get_streaming_response_with_tools(client, "owned-model", MESSAGES):
                    observed.append(item)
            assert raised.value is failure
            assert not any(isinstance(item, LLMResponse) for item in observed)
            responses = peer.responses if backend == "vertex" else peer.raw_responses
            assert responses and all(r.is_closed if backend == "vertex" else r.closed for r in responses)


async def test_bedrock_shutdown_failure_propagates_after_join_and_close(native_peer, monkeypatch):
    failure = RuntimeError("owned shutdown failure")
    with native_peer(frames("bedrock"), wire="bedrock", pause_after_first=True) as peer, loop_errors():
        async with owned_bedrock_client(peer) as client:
            stream = bedrock.BedrockAdapter().get_streaming_response(client, "owned-model", MESSAGES)
            assert await anext(stream) == "partial"
            raw, = peer.raw_responses
            shutdown = raw.shutdown

            def fail_after_shutdown():
                shutdown()
                raise failure

            monkeypatch.setattr(raw, "shutdown", fail_after_shutdown)
            entered = asyncio.Event()
            reads = []
            loop = asyncio.get_running_loop()
            submit = bedrock._submit

            def observe_submit(function, *args, **kwargs):
                if function is next:
                    def read():
                        loop.call_soon_threadsafe(entered.set)
                        return function(*args, **kwargs)
                    future = submit(read)
                    reads.append(future)
                    return future
                return submit(function, *args, **kwargs)

            monkeypatch.setattr(bedrock, "_submit", observe_submit)
            task = asyncio.create_task(anext(stream))
            await asyncio.wait_for(entered.wait(), 2)
            await asyncio.sleep(0.02)
            assert not task.done()
            task.cancel()
            with pytest.raises(RuntimeError) as raised:
                await asyncio.wait_for(task, 3)
            assert raised.value is failure
            assert reads and all(read.done() and not read.cancelled() for read in reads)
            assert raw.closed and not peer.pause_timed_out.is_set()
            await stream.aclose()
