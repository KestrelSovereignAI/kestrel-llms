"""Moonshot Kimi provider plugin for Kestrel Sovereign."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, List, Optional, Type

import openai
from pydantic import BaseModel

from kestrel_llm_openai_compat import (
    REASONING_COMPLETION_KWARGS,
    completion_kwargs,
    normalize_messages,
    stream_with_tool_calls,
    to_llm_response,
)
from kestrel_sdk.llm import (
    LLMAdapter,
    LLMResponse,
    ModelCategory,
    ModelInfo,
    ProviderInfo,
    ToolCallStarted,
)


class KimiAdapter(LLMAdapter):
    provider_name = "kimi"
    default_base_url = "https://api.moonshot.ai/v1"
    default_model = "kimi-k2.6"
    env_var = "MOONSHOT_API_KEY"

    @classmethod
    def create_provider(cls, config: Dict[str, Any]) -> ProviderInfo:
        api_key = (
            config.get("api_key")
            or os.environ.get(config.get("api_key_env") or cls.env_var)
            or os.environ.get("KIMI_API_KEY")
        )
        if not api_key:
            raise ValueError(f"kimi:api requires {config.get('api_key_env') or cls.env_var}")
        base_url = config.get("base_url") or cls.default_base_url
        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        return ProviderInfo(
            name="kimi:api",
            vendor="kimi",
            route="api",
            client=client,
            adapter=cls(),
            model=config.get("model") or cls.default_model,
            is_cloud=True,
            is_local=False,
            base_url=base_url,
            selection_hints=list(
                config.get("selection_hints") or ["long-context", "agentic-coding"]
            ),
        )

    def display_name(self) -> str:
        return "Kimi"

    def key_env_var(self) -> str:
        return self.env_var

    def substrate_type(self) -> str:
        return "kimi"

    def deliberation_style(self) -> str:
        return "sequential"

    async def get_response(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        extra = completion_kwargs(
            format,
            tools,
            response_format,
            kwargs,
            passthrough_keys=REASONING_COMPLETION_KWARGS,
        )
        response = await client.chat.completions.create(
            model=model,
            messages=normalize_messages(messages),
            **extra,
        )
        return to_llm_response(response)

    async def get_streaming_response(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        extra = completion_kwargs(
            None,
            tools,
            response_format,
            kwargs,
            passthrough_keys=REASONING_COMPLETION_KWARGS,
        )
        stream = await client.chat.completions.create(
            model=model,
            messages=normalize_messages(messages),
            stream=True,
            **extra,
        )
        async for chunk in stream:
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                yield content

    async def get_streaming_response_with_tools(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | ToolCallStarted | LLMResponse]:
        async for item in stream_with_tool_calls(
            client,
            model,
            messages,
            tools,
            response_format,
            passthrough_keys=REASONING_COMPLETION_KWARGS,
            **kwargs,
        ):
            yield item

    async def list_models(self, client: openai.AsyncOpenAI) -> List[ModelInfo]:
        models = await client.models.list()
        return [
            ModelInfo(
                id=item.id,
                provider="kimi",
                display_name=item.id,
                category=ModelCategory.CHAT,
                supports_tools=True,
                supports_streaming=True,
            )
            for item in getattr(models, "data", [])
            if getattr(item, "id", None)
        ]
