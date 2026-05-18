"""xAI Grok provider plugin for Kestrel Sovereign."""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Type

import openai
from pydantic import BaseModel

from kestrel_sdk.llm import LLMAdapter, LLMResponse, ModelCategory, ModelInfo, ProviderInfo, ToolCall


class XAIAdapter(LLMAdapter):
    provider_name = "xai"
    default_base_url = "https://api.x.ai/v1"
    default_model = "grok-4.3"
    env_var = "XAI_API_KEY"

    @classmethod
    def create_provider(cls, config: Dict[str, Any]) -> ProviderInfo:
        api_key = config.get("api_key") or os.environ.get(config.get("api_key_env") or cls.env_var)
        if not api_key:
            raise ValueError(f"xai:api requires {config.get('api_key_env') or cls.env_var}")
        base_url = config.get("base_url") or cls.default_base_url
        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        return ProviderInfo(
            name="xai:api",
            vendor="xai",
            route="api",
            client=client,
            adapter=cls(),
            model=config.get("model") or cls.default_model,
            is_cloud=True,
            is_local=False,
            base_url=base_url,
            selection_hints=list(config.get("selection_hints") or ["current-events", "reasoning"]),
        )

    def display_name(self) -> str:
        return "xAI Grok"

    def key_env_var(self) -> str:
        return self.env_var

    def substrate_type(self) -> str:
        return "grok"

    def deliberation_style(self) -> str:
        return "sequential"

    async def get_response(self, client: openai.AsyncOpenAI, model: str, messages: List[Dict[str, Any]], format: Optional[str] = None, tools: Optional[List[Dict[str, Any]]] = None, response_format: Optional[Type[BaseModel]] = None, **kwargs: Any) -> LLMResponse:
        response = await client.chat.completions.create(model=model, messages=messages, **_completion_kwargs(format, tools, response_format, kwargs))
        return _to_llm_response(response)

    async def get_streaming_response(self, client: openai.AsyncOpenAI, model: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, response_format: Optional[Type[BaseModel]] = None, **kwargs: Any) -> AsyncIterator[str]:
        stream = await client.chat.completions.create(model=model, messages=messages, stream=True, **_completion_kwargs(None, tools, response_format, kwargs))
        async for chunk in stream:
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                yield content

    async def list_models(self, client: openai.AsyncOpenAI) -> List[ModelInfo]:
        models = await client.models.list()
        return [ModelInfo(id=item.id, provider="xai", display_name=item.id, category=ModelCategory.CHAT, supports_tools=True, supports_streaming=True) for item in getattr(models, "data", []) if getattr(item, "id", None)]


def _completion_kwargs(format: Optional[str], tools: Optional[List[Dict[str, Any]]], response_format: Optional[Type[BaseModel]], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    if response_format is not None and issubclass(response_format, BaseModel):
        schema = response_format.model_json_schema()
        schema["additionalProperties"] = False
        extra["response_format"] = {"type": "json_schema", "json_schema": {"name": response_format.__name__, "schema": schema, "strict": True}}
    elif format == "json":
        extra["response_format"] = {"type": "json_object"}
    if tools:
        extra["tools"] = tools
        extra["tool_choice"] = "auto"
    if "max_tokens" in kwargs:
        extra["max_completion_tokens"] = kwargs["max_tokens"]
    for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "reasoning_effort", "extra_body"):
        if key in kwargs:
            extra[key] = kwargs[key]
    return extra


def _to_llm_response(response: Any) -> LLMResponse:
    message = response.choices[0].message
    tool_calls = None
    if getattr(message, "tool_calls", None):
        tool_calls = []
        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                args = {"_raw": call.function.arguments}
            tool_calls.append(ToolCall(id=call.id, name=call.function.name, arguments=args))
    usage = getattr(response, "usage", None)
    return LLMResponse(content=getattr(message, "content", None), tool_calls=tool_calls, raw=response, input_tokens=getattr(usage, "prompt_tokens", None), output_tokens=getattr(usage, "completion_tokens", None), total_tokens=getattr(usage, "total_tokens", None))
