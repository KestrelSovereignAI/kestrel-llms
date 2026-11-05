"""Physical native SDK response ownership, cancellation and client shutdown."""

import asyncio
import threading
from types import SimpleNamespace
from contextlib import asynccontextmanager, contextmanager

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


@pytest.mark.parametrize("mode_variable", [None, "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE"])
async def test_vertex_api_key_mode_does_not_use_automatic_mtls(native_peer, monkeypatch, mode_variable):
    monkeypatch.setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "true")
    for variable in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE"):
        monkeypatch.delenv(variable, raising=False)
    if mode_variable is not None:
        monkeypatch.setenv(mode_variable, "true")
    with native_peer(frames("vertex"), wire="vertex") as peer:
        if mode_variable is not None:
            with pytest.raises(ValueError, match="Vertex express"):
                VertexAIAdapter.create_provider({"api_key": "owned-key", "base_url": peer.origin})
            assert peer.calls == []
            return
        async with owned_vertex_client(peer) as client:
            assert client.genai.vertexai is False
            assert not client.http.is_closed
            items = [item async for item in VertexAIAdapter().get_streaming_response(client, "owned-model", MESSAGES)]
            assert items[0] == "partial" and items[-1].stop_reason == "STOP"
            assert peer.responses and all(response.is_closed for response in peer.responses)
        assert client.http.is_closed and len(peer.calls) == 1


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





@asynccontextmanager
async def observed_bedrock_read(monkeypatch, raw, *, peer, stream, shutdown_failure=None):
    """Observe the real native read, shutdown and close without replacing them.

    The settled read is held until the test settles it, so every OS shows the
    join: shutdown wakes a blocked POSIX read, while Winsock leaves it blocked.
    """
    loop = asyncio.get_running_loop()
    loop_thread = threading.current_thread()
    observed = SimpleNamespace(entered=asyncio.Event(), returned=asyncio.Event(), shut=asyncio.Event(),
                               reads=[], tasks=[], shutdowns=[], closes=[],
                               settle=threading.Event(), settle_timed_out=threading.Event())
    submit, shutdown, close = bedrock._submit, raw.shutdown, raw.close

    def observe_submit(function, *args, **kwargs):
        if function is not next:
            return submit(function, *args, **kwargs)

        def read():
            loop.call_soon_threadsafe(observed.entered.set)
            try:
                return function(*args, **kwargs)
            finally:
                loop.call_soon_threadsafe(observed.returned.set)
                if not observed.settle.wait(timeout=10):
                    observed.settle_timed_out.set()

        future = submit(read)
        observed.reads.append(future)
        return future

    def observe_shutdown():
        observed.shutdowns.append([read.done() for read in observed.reads])
        try:
            shutdown()
        finally:
            observed.shut.set()
        if shutdown_failure is not None:
            raise shutdown_failure

    def observe_close():
        # urllib3 also closes from the reading thread when it reaches EOF.
        if threading.current_thread() is loop_thread:
            observed.closes.append([read.done() for read in observed.reads])
        close()

    monkeypatch.setattr(bedrock, "_submit", observe_submit)
    monkeypatch.setattr(raw, "shutdown", observe_shutdown)
    monkeypatch.setattr(raw, "close", observe_close)
    try:
        yield observed
    finally:
        peer.release.set()
        peer.release_second.set()
        observed.settle.set()
        if observed.tasks or observed.reads:
            done, pending = await asyncio.wait({*observed.tasks, *observed.reads}, timeout=3)
            for future in done:
                if not future.cancelled():
                    future.exception()
            assert not pending, "Native observation cleanup did not join its original work"
        await stream.aclose()


async def test_cancel_vertex_blocked_read_closes_without_wire_progress(native_peer):
    """Async native I/O is interrupted by cancellation itself on every OS."""
    with native_peer(frames("vertex"), wire="vertex", pause_after_first=True) as peer, loop_errors():
        async with owned_vertex_client(peer) as client:
            stream = VertexAIAdapter().get_streaming_response(client, "owned-model", MESSAGES)
            assert await anext(stream) == "partial"
            task = asyncio.create_task(anext(stream))
            # The peer withholds the second frame; an event-loop ticker must run.
            for _ in range(4):
                await asyncio.sleep(0.01)
                assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            assert task.cancelled()
            await stream.aclose()
            assert peer.responses and all(r.is_closed for r in peer.responses)
            assert not peer.pause_timed_out.is_set()
            assert len(peer.calls) == 1


@pytest.mark.parametrize("tls", [False, True])
async def test_cancel_bedrock_blocked_read_is_owned_until_it_settles(native_peer, monkeypatch, tls):
    """Cancellation shuts the read side, then joins that exact native read.

    Winsock does not wake a receive already blocked when its read side is shut,
    so the cancelled caller waits until wire progress settles the read. It is
    never told the stream stopped before then.
    """
    with native_peer(frames("bedrock"), wire="bedrock", tls=tls, pause_after_first=True) as peer, loop_errors():
        async with owned_bedrock_client(peer) as client:
            stream = bedrock.BedrockAdapter().get_streaming_response(client, "owned-model", MESSAGES)
            assert await anext(stream) == "partial"
            raw, = peer.raw_responses
            async with observed_bedrock_read(monkeypatch, raw, peer=peer, stream=stream) as observed:
                task = asyncio.create_task(anext(stream))
                observed.tasks.append(task)
                await asyncio.wait_for(observed.entered.wait(), 2)
                for _ in range(4):
                    await asyncio.sleep(0.01)
                    assert not task.done()
                task.cancel()
                await asyncio.sleep(0)
                assert observed.shut.is_set(), "Native read-side shutdown was not invoked"
                read, = observed.reads
                task.cancel()
                for _ in range(4):
                    await asyncio.sleep(0.01)
                    assert not read.done() and not task.done() and observed.closes == []
                peer.release.set()
                observed.settle.set()
                done, _ = await asyncio.wait({task}, timeout=3)
                assert done == {task} and task.cancelled()
                assert read.done() and not read.cancelled()
                assert observed.shutdowns == [[False]] and observed.closes == [[True]] and raw.closed
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
            assert len(peer.calls) == 1
            assert not peer.pause_timed_out.is_set() and not observed.settle_timed_out.is_set()


async def test_cancel_bedrock_never_delivers_a_late_native_event(native_peer, monkeypatch):
    """A tool event whose native read settles after withdrawal is retrieved, never delivered."""
    events = [{"contentBlockDelta": {"delta": {"text": "partial"}}},
              {"contentBlockStart": {"contentBlockIndex": 0,
                                     "start": {"toolUse": {"toolUseId": "late-tool", "name": "owned_tool"}}}},
              {"messageStop": {"stopReason": "tool_use"}}]
    with native_peer(events, wire="bedrock", pause_after_first=True, pause_after_second=True) as peer, loop_errors():
        async with owned_bedrock_client(peer) as client:
            stream = bedrock.BedrockAdapter().get_streaming_response_with_tools(client, "owned-model", MESSAGES)
            assert await anext(stream) == "partial"
            raw, = peer.raw_responses
            async with observed_bedrock_read(monkeypatch, raw, peer=peer, stream=stream) as observed:
                task = asyncio.create_task(anext(stream))
                observed.tasks.append(task)
                await asyncio.wait_for(observed.entered.wait(), 2)
                peer.release.set()
                await asyncio.wait_for(observed.returned.wait(), 2)
                read, = observed.reads
                assert not read.done() and not task.done()
                task.cancel()
                await asyncio.sleep(0)
                assert observed.shut.is_set(), "Native read-side shutdown was not invoked"
                observed.settle.set()
                done, _ = await asyncio.wait({task}, timeout=3)
                assert done == {task} and task.cancelled()
                assert "contentBlockStart" in read.result()
                assert observed.shutdowns == [[False]] and observed.closes == [[True]] and raw.closed
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
            assert len(peer.calls) == 1 and not peer.pause_timed_out.is_set()


async def test_bedrock_shutdown_failure_propagates_after_join_and_close(native_peer, monkeypatch):
    failure = OSError("owned shutdown failure")
    with native_peer(frames("bedrock"), wire="bedrock", pause_after_first=True) as peer, loop_errors():
        async with owned_bedrock_client(peer) as client:
            stream = bedrock.BedrockAdapter().get_streaming_response(client, "owned-model", MESSAGES)
            assert await anext(stream) == "partial"
            raw, = peer.raw_responses
            async with observed_bedrock_read(monkeypatch, raw, peer=peer, stream=stream, shutdown_failure=failure) as observed:
                task = asyncio.create_task(anext(stream))
                observed.tasks.append(task)
                await asyncio.wait_for(observed.entered.wait(), 2)
                await asyncio.sleep(0.02)
                assert not task.done()
                task.cancel()
                await asyncio.sleep(0)
                assert observed.shut.is_set(), "Native read-side shutdown was not invoked"
                read, = observed.reads
                for _ in range(4):
                    await asyncio.sleep(0.01)
                    assert not read.done() and not task.done() and observed.closes == []
                peer.release.set()
                observed.settle.set()
                done, _ = await asyncio.wait({task}, timeout=3)
                assert done == {task} and not task.cancelled(), "Shutdown failure must survive cancellation"
                assert task.exception() is failure
                assert isinstance(failure.__context__, asyncio.CancelledError)
                assert read.done() and not read.cancelled()
                assert observed.shutdowns == [[False]] and observed.closes == [[True]] and raw.closed
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
            assert len(peer.calls) == 1
            assert not peer.pause_timed_out.is_set() and not observed.settle_timed_out.is_set()
