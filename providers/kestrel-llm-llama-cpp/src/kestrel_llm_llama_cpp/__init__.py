"""llama.cpp provider plugin for Kestrel Sovereign."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, List, Optional, Type

import openai
from pydantic import BaseModel

from kestrel_llm_openai_compat import completion_kwargs, to_llm_response
from kestrel_sdk.llm import LLMAdapter, LLMResponse, ModelCategory, ModelInfo, ProviderInfo


class LlamaCppAdapter(LLMAdapter):
    provider_name = "llama_cpp"
    default_base_url = "http://localhost:8000/v1"
    default_model = "local-model"

    @classmethod
    def create_provider(cls, config: Dict[str, Any]) -> ProviderInfo:
        base_url = (
            config.get("base_url")
            or os.environ.get("LLAMA_CPP_BASE_URL")
            or cls.default_base_url
        )
        api_key = (
            config.get("api_key")
            or os.environ.get(config.get("api_key_env") or "LLAMA_CPP_API_KEY")
            or "local"
        )
        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        return ProviderInfo(
            name="llama_cpp:local",
            vendor="llama_cpp",
            route="local",
            client=client,
            adapter=cls(),
            model=config.get("model") or os.environ.get("LLAMA_CPP_MODEL") or cls.default_model,
            is_cloud=False,
            is_local=True,
            base_url=base_url,
            selection_hints=list(config.get("selection_hints") or ["local", "private"]),
        )

    def display_name(self) -> str:
        return "llama.cpp"

    def substrate_type(self) -> str:
        return "local"

    def key_env_var(self) -> str:
        return "LLAMA_CPP_API_KEY"

    def deliberation_style(self) -> str:
        return "parallel"

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
        extra = completion_kwargs(format, tools, response_format, kwargs)
        response = await client.chat.completions.create(model=model, messages=messages, **extra)
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
        extra = completion_kwargs(None, tools, response_format, kwargs)
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **extra,
        )
        async for chunk in stream:
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                yield content

    async def list_models(self, client: openai.AsyncOpenAI) -> List[ModelInfo]:
        models = await client.models.list()
        return [
            ModelInfo(
                id=item.id,
                provider="llama_cpp",
                display_name=item.id,
                category=ModelCategory.CHAT,
                supports_tools=False,
                supports_streaming=True,
            )
            for item in getattr(models, "data", [])
            if getattr(item, "id", None)
        ]
