"""Shared helpers for OpenAI-compatible Kestrel LLM provider plugins."""

from __future__ import annotations

import json
from contextlib import aclosing
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

from pydantic import BaseModel

from kestrel_sdk.llm import (
    LLMProtocolError,
    LLMResponse,
    ProviderCapabilities,
    StructuredOutputMode,
    ToolCall,
    ToolCallStarted,
    ToolStreamingMode,
    VisionInputMode,
)

STANDARD_COMPLETION_KWARGS = (
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "extra_body",
)
REASONING_COMPLETION_KWARGS = (*STANDARD_COMPLETION_KWARGS, "reasoning_effort")


def openai_compatible_capabilities(
    *,
    supports_vision: bool = False,
    supports_structured_output: bool = True,
    model_dependent: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> ProviderCapabilities:
    """Return SDK capability metadata for OpenAI-compatible providers.

    Per-model support may still vary by provider/model; use
    ``model_dependent`` to mark those caveats.
    """
    return ProviderCapabilities(
        supports_tools=True,
        supports_streaming=True,
        supports_vision=supports_vision,
        supports_structured_output=supports_structured_output,
        structured_output_mode=StructuredOutputMode.JSON_SCHEMA
        if supports_structured_output
        else StructuredOutputMode.NONE,
        tool_streaming_mode=ToolStreamingMode.NATIVE_DELTA,
        vision_input_mode=VisionInputMode.OPENAI_IMAGE_URL
        if supports_vision
        else VisionInputMode.NONE,
        model_dependent=tuple(model_dependent),
        notes=tuple(notes),
    )


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize Chat Completions messages without mutating the caller.

    Kestrel exposes tool-call arguments to downstream code as parsed dicts.
    OpenAI-compatible APIs require assistant ``tool_calls`` history to carry
    ``function.arguments`` as a JSON string on the wire.
    """
    normalized = []
    for msg in messages:
        if "tool_calls" not in msg or not msg["tool_calls"]:
            normalized.append(msg)
            continue

        new_msg = {**msg}
        new_tool_calls = []
        for tool_call in msg["tool_calls"]:
            new_tool_call = {**tool_call}
            function = tool_call.get("function")
            if isinstance(function, dict) and "arguments" in function:
                arguments = function["arguments"]
                if isinstance(arguments, dict):
                    new_tool_call["function"] = {
                        **function,
                        "arguments": json.dumps(arguments),
                    }
            new_tool_calls.append(new_tool_call)
        new_msg["tool_calls"] = new_tool_calls
        normalized.append(new_msg)
    return normalized


MessageNormalizer = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]


def completion_kwargs(
    format: Optional[str],
    tools: Optional[List[Dict[str, Any]]],
    response_format: Optional[Type[BaseModel]],
    kwargs: Dict[str, Any],
    passthrough_keys: Iterable[str] = STANDARD_COMPLETION_KWARGS,
) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    if response_format is not None and issubclass(response_format, BaseModel):
        schema = response_format.model_json_schema()
        schema["additionalProperties"] = False
        extra["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": schema,
                "strict": True,
            },
        }
    elif format == "json":
        extra["response_format"] = {"type": "json_object"}

    if tools:
        extra["tools"] = tools
        extra["tool_choice"] = "auto"

    if "max_tokens" in kwargs:
        extra["max_completion_tokens"] = kwargs["max_tokens"]

    for key in passthrough_keys:
        if key in kwargs:
            extra[key] = kwargs[key]
    return extra


def _single_choice(choices: Any, *, allow_empty: bool = False) -> Any:
    """One selected generation; usage-only stream trailers may have none."""
    if not isinstance(choices, list):
        raise LLMProtocolError("OpenAI completion choices must be a list.")
    if not choices and allow_empty:
        return None
    if len(choices) != 1 or type(getattr(choices[0], "index", None)) is not int or choices[0].index != 0:
        raise LLMProtocolError("OpenAI completion must contain one choice with index zero.")
    return choices[0]


def _native_finish_reason(choice: Any) -> Optional[str]:
    reason = getattr(choice, "finish_reason", None)
    if reason is not None and (type(reason) is not str or not reason):
        raise LLMProtocolError("OpenAI finish_reason must be a non-empty string or null.")
    return reason


def _tool_call(call_id: Any, name: Any, arguments: Any) -> ToolCall:
    """Decode one complete native call without repairing malformed evidence."""
    if any(type(value) is not str or not value for value in (call_id, name)):
        raise LLMProtocolError("OpenAI tool call requires its native id and name.")
    if type(arguments) is not str:
        raise LLMProtocolError("OpenAI tool arguments must be a JSON object string.")
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise LLMProtocolError("OpenAI tool arguments must contain valid JSON.") from error
    if not isinstance(parsed, dict):
        raise LLMProtocolError("OpenAI tool arguments must contain a JSON object.")
    return ToolCall(id=call_id, name=name, arguments=parsed)


def to_llm_response(response: Any) -> LLMResponse:
    """Convert a non-streaming SDK response while preserving the raw object.

    Kestrel core knows how to extract provider ``reasoning_content`` from
    ``raw.choices[0].message``. Keep the full response here so callers retain
    provider-specific metadata and usage details.
    """
    choice = _single_choice(getattr(response, "choices", None))
    message = choice.message
    tool_calls = None
    if getattr(message, "tool_calls", None):
        tool_calls = []
        for call in message.tool_calls:
            function = getattr(call, "function", None)
            tool_calls.append(_tool_call(
                getattr(call, "id", None), getattr(function, "name", None),
                getattr(function, "arguments", None),
            ))

    usage = getattr(response, "usage", None)
    return LLMResponse(
        content=getattr(message, "content", None),
        tool_calls=tool_calls,
        raw=response,
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        stop_reason=_native_finish_reason(choice),
    )


async def stream_with_tool_calls(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Type[BaseModel]] = None,
    passthrough_keys: Iterable[str] = STANDARD_COMPLETION_KWARGS,
    message_normalizer: MessageNormalizer = normalize_messages,
    **kwargs: Any,
) -> AsyncIterator[Union[str, ToolCallStarted, LLMResponse]]:
    """Stream text and OpenAI-compatible tool calls.

    Kestrel's streaming honesty layer requires a ``ToolCallStarted`` marker
    as soon as a provider begins a tool call. OpenAI-compatible backends expose
    tool calls as indexed deltas, so this helper accumulates each index and
    emits exactly one marker for each call before yielding the final assembled
    ``LLMResponse``.
    """
    extra = completion_kwargs(None, tools, response_format, kwargs, passthrough_keys=passthrough_keys)
    extra["stream_options"] = {"include_usage": True}
    stream = await client.chat.completions.create(
        model=model, messages=message_normalizer(messages), stream=True, **extra,
    )
    tool_calls: Dict[int, Dict[str, str]] = {}
    text_content = ""
    reasoning_content = ""
    input_tokens = output_tokens = total_tokens = None
    stop_reason = None

    async with stream:
        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                if getattr(usage, "prompt_tokens", None) is not None:
                    input_tokens = usage.prompt_tokens
                if getattr(usage, "completion_tokens", None) is not None:
                    output_tokens = usage.completion_tokens
                if getattr(usage, "total_tokens", None) is not None:
                    total_tokens = usage.total_tokens

            choice = _single_choice(getattr(chunk, "choices", None), allow_empty=True)
            if choice is None:
                continue
            reason = _native_finish_reason(choice)
            delta = choice.delta
            content = getattr(delta, "content", None)
            reasoning = getattr(delta, "reasoning_content", None)
            deltas = getattr(delta, "tool_calls", None) or []
            if stop_reason is not None:
                if content or reasoning or deltas:
                    raise LLMProtocolError("OpenAI stream returned content after terminal evidence.")
                if reason is not None and reason != stop_reason:
                    raise LLMProtocolError("OpenAI stream returned conflicting terminal reasons.")

            if isinstance(reasoning, str) and reasoning:
                reasoning_content += reasoning
            if isinstance(content, str) and content:
                text_content += content
                yield content
            for delta_call in deltas:
                index = delta_call.index
                if type(index) is not int or index < 0:
                    raise LLMProtocolError("OpenAI tool delta requires a non-negative integer index.")
                is_new = index not in tool_calls
                if is_new:
                    tool_calls[index] = {"id": "", "name": "", "arguments": ""}
                current = tool_calls[index]
                function = getattr(delta_call, "function", None)
                for key, value in (("id", getattr(delta_call, "id", None)), ("name", getattr(function, "name", None))):
                    if value is None:
                        continue
                    if type(value) is not str or not value:
                        raise LLMProtocolError("OpenAI tool identity fields must be non-empty strings.")
                    if current[key] and current[key] != value:
                        raise LLMProtocolError("OpenAI tool delta returned a conflicting native identity.")
                    current[key] = value
                if function is not None:
                    arguments = getattr(function, "arguments", None)
                    if arguments is not None:
                        if type(arguments) is not str:
                            raise LLMProtocolError("OpenAI tool argument fragments must be strings.")
                        current["arguments"] += arguments
                if is_new:
                    yield ToolCallStarted(index=index, id=current["id"] or None, name=current["name"] or None)
            if reason is not None:
                stop_reason = reason

    parsed_tool_calls = []
    for current in tool_calls.values():
        # Native streams may finish a zero-argument call without JSON deltas.
        parsed_tool_calls.append(_tool_call(current["id"], current["name"], current["arguments"] or "{}"))
    yield LLMResponse(
        content=text_content or None, tool_calls=parsed_tool_calls or None,
        raw={"reasoning_content": reasoning_content} if reasoning_content else None,
        input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens,
        stop_reason=stop_reason,
    )


async def stream_text_only(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Type[BaseModel]] = None,
    passthrough_keys: Iterable[str] = STANDARD_COMPLETION_KWARGS,
    message_normalizer: MessageNormalizer = normalize_messages,
    **kwargs: Any,
) -> AsyncIterator[Union[str, LLMResponse]]:
    """Project the canonical stream without accepting any tool evidence."""
    if tools:
        raise LLMProtocolError("Plain streaming does not accept tools; use the with-tools method.")
    async with aclosing(stream_with_tool_calls(
        client, model, messages, None, response_format,
        passthrough_keys=passthrough_keys, message_normalizer=message_normalizer, **kwargs,
    )) as stream:
        async for item in stream:
            if isinstance(item, ToolCallStarted) or (
                isinstance(item, LLMResponse)
                and (item.has_tool_calls or item.stop_reason in {"tool_calls", "function_call"})
            ):
                raise LLMProtocolError("Plain streaming received unexpected tool evidence.")
            yield item
