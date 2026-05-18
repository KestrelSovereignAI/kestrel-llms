"""Shared helpers for OpenAI-compatible Kestrel LLM provider plugins."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Type

from pydantic import BaseModel

from kestrel_sdk.llm import LLMResponse, ToolCall

STANDARD_COMPLETION_KWARGS = (
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "extra_body",
)
REASONING_COMPLETION_KWARGS = (*STANDARD_COMPLETION_KWARGS, "reasoning_effort")


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

