"""AWS Bedrock provider plugin for Kestrel Sovereign."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, AsyncIterator, Dict, List, Optional, Type

from pydantic import BaseModel
from botocore.eventstream import EventStream
from urllib3.response import HTTPResponse

from kestrel_sdk.llm import (
    LLMAdapter,
    LLMProtocolError,
    LLMResponse,
    ModelCategory,
    ModelInfo,
    ProviderCapabilities,
    ProviderInfo,
    StructuredOutputMode,
    ToolCall,
    ToolCallStarted,
    ToolStreamingMode,
    VisionInputMode,
)

DEFAULT_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
STRUCTURED_OUTPUT_TOOL = "kestrel_structured_response"


@dataclass(frozen=True)
class BedrockClients:
    runtime: Any
    management: Any
    region: str
    responses: ContextVar = field(default_factory=lambda: ContextVar("bedrock_native_responses", default=None), repr=False, compare=False)

    def __post_init__(self) -> None:
        def capture(http_response: Any, **kwargs: Any) -> None:
            sink = self.responses.get()
            if sink is not None:
                sink.append(http_response.raw)

        self.runtime.meta.events.register("after-call.bedrock-runtime.ConverseStream", capture)

    async def close(self) -> None:
        async with AsyncExitStack() as cleanup:
            cleanup.push_async_callback(_to_thread, self.management.close)
            if self.runtime is not self.management:
                cleanup.push_async_callback(_to_thread, self.runtime.close)


def _load_boto3() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 package not installed. Install with: pip install kestrel-llm-bedrock"
        ) from exc
    return boto3


def _client_config() -> Any:
    from botocore.config import Config

    return Config(retries={"total_max_attempts": 1})


def _region_from_config(config: Dict[str, Any]) -> str:
    return (
        config.get("region")
        or config.get("region_name")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def _client_from_config(config: Dict[str, Any]) -> Any:
    boto3 = _load_boto3()

    session_kwargs: Dict[str, Any] = {}
    if config.get("profile"):
        session_kwargs["profile_name"] = config["profile"]
    if config.get("aws_access_key_id"):
        session_kwargs["aws_access_key_id"] = config["aws_access_key_id"]
    if config.get("aws_secret_access_key"):
        session_kwargs["aws_secret_access_key"] = config["aws_secret_access_key"]
    if config.get("aws_session_token"):
        session_kwargs["aws_session_token"] = config["aws_session_token"]

    session = boto3.Session(**session_kwargs)
    region = _region_from_config(config)
    common_kwargs = {
        "region_name": region,
        "endpoint_url": config.get("endpoint_url"),
    }
    common_kwargs["config"] = _client_config()
    if config.get("ca_bundle"):
        common_kwargs["verify"] = config["ca_bundle"]
    return BedrockClients(
        runtime=session.client(
            "bedrock-runtime",
            **common_kwargs,
        ),
        management=session.client(
            "bedrock",
            **common_kwargs,
        ),
        region=region,
    )


def _require_clients(client: Any) -> BedrockClients:
    if not isinstance(client, BedrockClients):
        raise TypeError("Bedrock operations require the owned clients returned by create_provider.")
    return client


def _runtime_client(client: Any) -> Any:
    return _require_clients(client).runtime


def _management_client(client: Any) -> Any:
    return _require_clients(client).management


def _text_block(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, str):
        return {"text": value}
    return {"text": json.dumps(value)}


def _image_block(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    image_url = item.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url.startswith("data:") or "," not in url:
        return None
    header, data = url.split(",", 1)
    mime_type = header[5:].split(";", 1)[0] or "image/png"
    image_format = mime_type.rsplit("/", 1)[-1].replace("jpg", "jpeg")
    return {"image": {"format": image_format, "source": {"bytes": base64.b64decode(data)}}}


def _content_item_to_block(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        return {"text": item}
    if not isinstance(item, dict):
        return _text_block(item)
    if item.get("type") == "text":
        return _text_block(item.get("text"))
    if item.get("type") == "image_url":
        return _image_block(item)
    if item.get("type") == "input_image":
        data = item.get("data")
        if isinstance(data, str):
            data = base64.b64decode(data)
        if isinstance(data, bytes):
            mime_type = item.get("mime_type") or "image/png"
            image_format = mime_type.rsplit("/", 1)[-1].replace("jpg", "jpeg")
            return {"image": {"format": image_format, "source": {"bytes": data}}}
    return item


def _content_to_blocks(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        return [block for item in content if (block := _content_item_to_block(item))]
    block = _text_block(content)
    return [block] if block else []


def _tool_result_content(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        try:
            return [{"json": json.loads(content)}]
        except json.JSONDecodeError:
            return [{"text": content}]
    return [{"json": content}]


def normalize_bedrock_messages(
    messages: List[Dict[str, Any]],
) -> tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Convert Kestrel/OpenAI-style chat history into Bedrock Converse messages."""
    system: List[Dict[str, str]] = []
    bedrock_messages: List[Dict[str, Any]] = []

    def _emit(bedrock_role: str, blocks: List[Dict[str, Any]]) -> None:
        """Append content, coalescing into the previous message when it has the
        SAME role.

        Bedrock's Converse API validates strict user/assistant alternation and
        rejects consecutive same-role messages. Sovereign emits one ``role:
        'tool'`` message per tool call, so a parallel-tool turn (2+ calls) would
        otherwise map to consecutive ``user`` messages and fail the whole turn
        with a ValidationException. Merging same-role content keeps N parallel
        toolResult blocks in a single user message per the contract (F408).
        """
        if not blocks:
            return
        if bedrock_messages and bedrock_messages[-1]["role"] == bedrock_role:
            bedrock_messages[-1]["content"].extend(blocks)
        else:
            bedrock_messages.append({"role": bedrock_role, "content": list(blocks)})

    for message in messages:
        role = message.get("role")
        if role == "system":
            content = message.get("content")
            if isinstance(content, str) and content:
                system.append({"text": content})
            continue

        if role == "tool":
            tool_use_id = message.get("tool_call_id") or message.get("id") or "tool_result"
            _emit(
                "user",
                [
                    {
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": _tool_result_content(message.get("content")),
                        }
                    }
                ],
            )
            continue

        blocks = _content_to_blocks(message.get("content"))
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"_raw": arguments}
                blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": call.get("id") or f"bedrock_tool_{uuid.uuid4().hex}",
                            "name": function.get("name") or "",
                            "input": arguments,
                        }
                    }
                )
            bedrock_role = "assistant"
        else:
            bedrock_role = "user"

        _emit(bedrock_role, blocks)

    return system, bedrock_messages


def _convert_tools(
    tools: Optional[List[Dict[str, Any]]],
    response_format: Optional[Type[BaseModel]] = None,
) -> Optional[Dict[str, Any]]:
    converted = []
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        function = tool.get("function") or {}
        converted.append(
            {
                "toolSpec": {
                    "name": function["name"],
                    **({"description": function["description"]} if "description" in function else {}),
                    "inputSchema": {
                        "json": function.get("parameters", {"type": "object", "properties": {}})
                    },
                }
            }
        )

    if response_format is not None and issubclass(response_format, BaseModel):
        schema = response_format.model_json_schema()
        converted.append(
            {
                "toolSpec": {
                    "name": STRUCTURED_OUTPUT_TOOL,
                    "description": f"Return a structured {response_format.__name__} response.",
                    "inputSchema": {"json": schema},
                }
            }
        )

    if not converted:
        return None

    config: Dict[str, Any] = {"tools": converted}
    if response_format is not None and issubclass(response_format, BaseModel):
        config["toolChoice"] = {"tool": {"name": STRUCTURED_OUTPUT_TOOL}}
    else:
        config["toolChoice"] = {"auto": {}}
    return config


def _inference_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if "max_tokens" in kwargs:
        config["maxTokens"] = kwargs["max_tokens"]
    if "temperature" in kwargs:
        config["temperature"] = kwargs["temperature"]
    if "top_p" in kwargs:
        config["topP"] = kwargs["top_p"]
    if "stop_sequences" in kwargs:
        config["stopSequences"] = kwargs["stop_sequences"]
    return config


def _request_payload(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    response_format: Optional[Type[BaseModel]],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    system, bedrock_messages = normalize_bedrock_messages(messages)
    payload: Dict[str, Any] = {"modelId": model, "messages": bedrock_messages}
    if system:
        payload["system"] = system
    inference_config = _inference_config(kwargs)
    if inference_config:
        payload["inferenceConfig"] = inference_config
    tool_config = _convert_tools(tools, response_format)
    if tool_config:
        payload["toolConfig"] = tool_config
    return payload


def _usage_tokens(response: Dict[str, Any]) -> tuple[Optional[int], Optional[int], Optional[int]]:
    usage = response.get("usage") or {}
    input_tokens = usage.get("inputTokens")
    output_tokens = usage.get("outputTokens")
    total_tokens = usage.get("totalTokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def _native_stop_reason(response: Dict[str, Any]) -> Optional[str]:
    reason = response.get("stopReason")
    if reason is not None and (type(reason) is not str or not reason):
        raise LLMProtocolError("Bedrock stopReason must be a non-empty string or absent.")
    return reason


def _tool_call(call_id: Any, name: Any, arguments: Any) -> ToolCall:
    """Require a complete native identity and object arguments."""
    if any(type(value) is not str or not value for value in (call_id, name)):
        raise LLMProtocolError("Bedrock tool call requires its native id and name.")
    if not isinstance(arguments, dict):
        raise LLMProtocolError("Bedrock tool arguments must contain a JSON object.")
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _extract_response(response: Dict[str, Any]) -> LLMResponse:
    stop_reason = _native_stop_reason(response)
    message = (response.get("output") or {}).get("message") or {}
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []

    for block in message.get("content") or []:
        if "text" in block:
            text_parts.append(block["text"])
        if "toolUse" in block:
            tool_use = block["toolUse"]
            if not isinstance(tool_use, dict):
                raise LLMProtocolError("Bedrock toolUse must be an object.")
            tool_calls.append(_tool_call(tool_use.get("toolUseId"), tool_use.get("name"), tool_use.get("input")))

    input_tokens, output_tokens, total_tokens = _usage_tokens(response)
    return LLMResponse(
        content="".join(text_parts) or None,
        tool_calls=tool_calls or None,
        raw=response,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        stop_reason=stop_reason,
    )


def _submit(function: Any, *args: Any, **kwargs: Any) -> asyncio.Future:
    context = copy_context()
    return asyncio.get_running_loop().run_in_executor(None, context.run, partial(function, *args, **kwargs))


async def _joined(call: asyncio.Future, *, withdrawn: bool = False) -> Any:
    """Join the exact native operation even when its observer is cancelled."""
    cancelled = None
    while not call.done():
        try:
            await asyncio.wait((call,))
        except asyncio.CancelledError as error:
            cancelled = cancelled or error
    if cancelled is not None:
        call.exception()
        raise cancelled
    if withdrawn:
        call.exception()
        return None
    return call.result()


async def _to_thread(function: Any, **kwargs: Any) -> Any:
    return await _joined(_submit(function, **kwargs))


_STREAM_DONE = object()


class _EventStreamReader:
    """One demand-driven native read and the public HTTP response it owns."""

    def __init__(self, events: EventStream, raw: HTTPResponse) -> None:
        self.events = iter(events)
        self.raw = raw
        self.read = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.read = _submit(next, self.events, _STREAM_DONE)
        await asyncio.wait((self.read,))
        value = self.read.result()
        self.read = None
        if value is _STREAM_DONE:
            raise StopAsyncIteration
        return value

    async def close(self) -> None:
        try:
            if self.read is not None and not self.read.done() and self.raw.connection is not None:
                # SHUT_RD wakes POSIX reads. Winsock can keep an active
                # receive pending until wire progress or the SDK read timeout.
                # Retirement still joins this exact read before closing it.
                try:
                    self.raw.shutdown()
                except RuntimeError:
                    if self.raw.connection is not None:
                        raise
        finally:
            try:
                if self.read is not None:
                    # Shutdown can make a withdrawn read report truncated wire
                    # data. Retrieve its outcome; never deliver it as a result.
                    await _joined(self.read, withdrawn=True)
            finally:
                self.raw.close()


@asynccontextmanager
async def _owned_event_stream(client: BedrockClients, payload: Dict[str, Any]):
    client = _require_clients(client)
    responses = []
    reader = None
    token = client.responses.set(responses)
    try:
        call = _submit(client.runtime.converse_stream, **payload)
    finally:
        client.responses.reset(token)
    try:
        response = await _joined(call)
        events = response.get("stream")
        if not isinstance(events, EventStream) or len(responses) != 1 or not isinstance(responses[0], HTTPResponse):
            raise LLMProtocolError("Bedrock streaming requires one owned native event stream and HTTP response.")
        reader = _EventStreamReader(events, responses[0])
        yield reader, {key: value for key, value in response.items() if key != "stream"}
    finally:
        if reader is not None:
            await reader.close()
        else:
            async with AsyncExitStack() as cleanup:
                for raw in responses:
                    cleanup.callback(raw.close)


class BedrockAdapter(LLMAdapter):
    provider_name = "bedrock"
    default_model = DEFAULT_MODEL

    @classmethod
    def create_provider(cls, config: Dict[str, Any]) -> ProviderInfo:
        region = _region_from_config(config)
        return ProviderInfo(
            name="bedrock:api",
            vendor="bedrock",
            route="api",
            client=_client_from_config(config),
            adapter=cls(),
            model=config.get("model") or cls.default_model,
            is_cloud=True,
            is_local=False,
            base_url=config.get("endpoint_url"),
            selection_hints=list(
                config.get("selection_hints")
                or ["aws", "multi-provider", "private-cloud"]
            ),
        )

    def display_name(self) -> str:
        return "AWS Bedrock"

    def key_env_var(self) -> Optional[str]:
        return "AWS_ACCESS_KEY_ID"

    def substrate_type(self) -> str:
        return "multi-provider"

    def deliberation_style(self) -> str:
        return "sequential"

    def provider_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_vision=True,
            supports_structured_output=True,
            structured_output_mode=StructuredOutputMode.TOOL_FORCED,
            tool_streaming_mode=ToolStreamingMode.NATIVE_DELTA,
            vision_input_mode=VisionInputMode.PROVIDER_NATIVE,
            model_dependent=("tools", "vision", "structured_output", "streaming"),
            notes=(
                "Uses AWS Bedrock Runtime Converse APIs.",
                "Feature support depends on the selected Bedrock model family.",
                "Structured output is implemented with a forced Bedrock tool call.",
                "Structured output requires the non-streaming method.",
            ),
        )

    async def get_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if format == "json" and response_format is None:
            messages = [
                {"role": "system", "content": "Return valid JSON."},
                *messages,
            ]
        payload = _request_payload(model, messages, tools, response_format, kwargs)
        response = await _to_thread(_runtime_client(client).converse, **payload)
        llm_response = _extract_response(response)
        if response_format is not None and llm_response.tool_calls:
            structured = next(
                (
                    call.arguments
                    for call in llm_response.tool_calls
                    if call.name == STRUCTURED_OUTPUT_TOOL
                ),
                None,
            )
            if structured is not None:
                return replace(llm_response, content=json.dumps(structured), tool_calls=None)
        return llm_response

    async def get_streaming_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | LLMResponse]:
        if tools:
            raise LLMProtocolError("Plain Bedrock streaming does not accept tools.")
        async with aclosing(self.get_streaming_response_with_tools(
            client, model, messages, response_format=response_format, **kwargs,
        )) as stream:
            async for item in stream:
                if isinstance(item, ToolCallStarted) or (
                    isinstance(item, LLMResponse)
                    and (item.has_tool_calls or item.stop_reason == "tool_use")
                ):
                    raise LLMProtocolError("Plain Bedrock streaming received unexpected tool evidence.")
                yield item

    async def get_streaming_response_with_tools(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | ToolCallStarted | LLMResponse]:
        if response_format is not None:
            raise LLMProtocolError("Bedrock structured output requires the non-streaming method.")
        payload = _request_payload(model, messages, tools, None, kwargs)

        text_parts: List[str] = []
        tool_blocks: Dict[int, Dict[str, Any]] = {}
        input_tokens = output_tokens = total_tokens = None
        stop_reason = None
        stop_seen = False

        async with _owned_event_stream(_require_clients(client), payload) as (events, response):
            async for event in events:
                if "messageStop" in event:
                    if stop_seen:
                        raise LLMProtocolError("Bedrock stream returned more than one messageStop.")
                    stop_seen = True
                    stop_reason = _native_stop_reason(event["messageStop"])
                    continue
                if stop_seen and ("contentBlockStart" in event or "contentBlockDelta" in event):
                    raise LLMProtocolError("Bedrock stream returned content after terminal evidence.")
                if "metadata" in event:
                    input_tokens, output_tokens, total_tokens = _usage_tokens(event["metadata"])
                    continue

                start = event.get("contentBlockStart", {}).get("start", {})
                if "toolUse" in start:
                    index = event["contentBlockStart"].get("contentBlockIndex")
                    if type(index) is not int or index < 0 or index in tool_blocks:
                        raise LLMProtocolError("Bedrock tool block requires a new non-negative integer index.")
                    tool_use = start["toolUse"]
                    if any(type(tool_use.get(key)) is not str or not tool_use[key] for key in ("toolUseId", "name")):
                        raise LLMProtocolError("Bedrock tool block requires its native id and name.")
                    tool_blocks[index] = {
                        "id": tool_use["toolUseId"],
                        "name": tool_use["name"],
                        "input": "",
                    }
                    yield ToolCallStarted(
                        index=index,
                        id=tool_blocks[index]["id"],
                        name=tool_blocks[index]["name"],
                    )
                    continue

                delta_event = event.get("contentBlockDelta", {})
                delta = delta_event.get("delta", {})
                if "text" in delta:
                    text_parts.append(delta["text"])
                    yield delta["text"]
                if "toolUse" in delta:
                    index = delta_event.get("contentBlockIndex")
                    if type(index) is not int or index not in tool_blocks:
                        raise LLMProtocolError("Bedrock tool delta requires an existing native tool block.")
                    current = tool_blocks[index]
                    fragment = delta["toolUse"].get("input")
                    if type(fragment) is not str:
                        raise LLMProtocolError("Bedrock tool argument fragments must be strings.")
                    current["input"] += fragment

        tool_calls = []
        for current in tool_blocks.values():
            try:
                # A native zero-argument tool block need not carry input deltas.
                arguments = json.loads(current["input"]) if current["input"] else {}
            except json.JSONDecodeError as error:
                raise LLMProtocolError("Bedrock tool arguments must contain valid JSON.") from error
            tool_calls.append(_tool_call(current["id"], current["name"], arguments))
        yield LLMResponse(
            content="".join(text_parts) or None,
            tool_calls=tool_calls or None,
            raw=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            stop_reason=stop_reason,
        )


    async def list_models(self, client: Any = None) -> List[ModelInfo]:
        bedrock = _management_client(client)
        response = await _to_thread(bedrock.list_foundation_models)
        models = []
        for item in response.get("modelSummaries") or []:
            model_id = item.get("modelId")
            if not model_id:
                continue
            lower_id = model_id.lower()
            category = ModelCategory.EMBEDDING if "embed" in lower_id else ModelCategory.CHAT
            is_chat = category == ModelCategory.CHAT
            models.append(
                ModelInfo(
                    id=model_id,
                    provider="bedrock",
                    display_name=item.get("modelName") or model_id,
                    category=category,
                    description=item.get("providerName"),
                    supports_vision=is_chat and any(x in lower_id for x in ("claude-3", "nova", "llama")),
                    supports_tools=is_chat,
                    supports_streaming=is_chat,
                )
            )
        return models
