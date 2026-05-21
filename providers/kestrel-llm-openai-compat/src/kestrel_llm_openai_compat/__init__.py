"""Shared helpers for OpenAI-compatible Kestrel LLM provider plugins."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional, Type, Union

from pydantic import BaseModel

from kestrel_sdk.llm import LLMResponse, ToolCall, ToolCallStarted

STANDARD_COMPLETION_KWARGS = (
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "extra_body",
)
REASONING_COMPLETION_KWARGS = (*STANDARD_COMPLETION_KWARGS, "reasoning_effort")


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


def to_llm_response(response: Any) -> LLMResponse:
    message = response.choices[0].message
    tool_calls = None
    if getattr(message, "tool_calls", None):
        tool_calls = []
        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                args = {"_raw": call.function.arguments}
            tool_calls.append(
                ToolCall(id=call.id, name=call.function.name, arguments=args)
            )

    usage = getattr(response, "usage", None)
    return LLMResponse(
        content=getattr(message, "content", None),
        tool_calls=tool_calls,
        raw=response,
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
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
    extra = completion_kwargs(
        None,
        tools,
        response_format,
        kwargs,
        passthrough_keys=passthrough_keys,
    )
    extra["stream_options"] = {"include_usage": True}

    stream = await client.chat.completions.create(
        model=model,
        messages=message_normalizer(messages),
        stream=True,
        **extra,
    )

    tool_calls: Dict[int, Dict[str, str]] = {}
    text_content = ""
    reasoning_content = ""
    input_tokens = None
    output_tokens = None
    total_tokens = None

    async for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

        if not getattr(chunk, "choices", None):
            continue

        delta = chunk.choices[0].delta
        delta_reasoning = getattr(delta, "reasoning_content", None)
        if isinstance(delta_reasoning, str) and delta_reasoning:
            reasoning_content += delta_reasoning

        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            text_content += content
            yield content

        for tc_delta in getattr(delta, "tool_calls", None) or []:
            idx = tc_delta.index
            is_new = idx not in tool_calls
            if is_new:
                tool_calls[idx] = {"id": "", "name": "", "arguments": ""}

            current = tool_calls[idx]
            if getattr(tc_delta, "id", None):
                current["id"] += tc_delta.id

            function = getattr(tc_delta, "function", None)
            if function:
                if getattr(function, "name", None):
                    current["name"] += function.name
                if getattr(function, "arguments", None):
                    current["arguments"] += function.arguments

            if is_new:
                yield ToolCallStarted(
                    index=idx,
                    id=current["id"] or None,
                    name=current["name"] or None,
                )

    if tool_calls:
        parsed_tool_calls = []
        for idx in sorted(tool_calls):
            tc_data = tool_calls[idx]
            try:
                args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
            except json.JSONDecodeError:
                args = {"_raw": tc_data["arguments"]}

            parsed_tool_calls.append(
                ToolCall(
                    id=tc_data["id"],
                    name=tc_data["name"],
                    arguments=args,
                )
            )

        yield LLMResponse(
            content=text_content or None,
            tool_calls=parsed_tool_calls,
            raw={"reasoning_content": reasoning_content} if reasoning_content else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
