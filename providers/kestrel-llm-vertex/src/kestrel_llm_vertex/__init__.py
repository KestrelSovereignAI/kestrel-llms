"""Google Vertex AI provider plugin for Kestrel Sovereign."""

from __future__ import annotations

import base64
import json
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from kestrel_sdk.llm import (
    LLMAdapter,
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


def _load_genai() -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "google-genai package not installed. Install with: pip install kestrel-llm-vertex"
        ) from exc
    return genai


def _client_from_config(config: Dict[str, Any]) -> Any:
    genai = _load_genai()
    api_key = (
        config.get("api_key")
        or os.environ.get(config.get("api_key_env") or "GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if api_key:
        return genai.Client(api_key=api_key)

    project_id = (
        config.get("project_id")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT_ID")
        or _project_from_credentials(config.get("credentials_file"))
    )
    if not project_id:
        raise ValueError(
            "vertex_ai:api requires GOOGLE_API_KEY, GEMINI_API_KEY, "
            "GOOGLE_CLOUD_PROJECT, GCP_PROJECT_ID, or credentials with project_id"
        )

    client_kwargs: Dict[str, Any] = {
        "vertexai": True,
        "project": project_id,
        "location": config.get("location")
        or os.environ.get("GOOGLE_CLOUD_LOCATION")
        or "us-central1",
    }
    credentials_file = config.get("credentials_file") or os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    if credentials_file:
        from google.oauth2 import service_account

        client_kwargs["credentials"] = service_account.Credentials.from_service_account_file(
            credentials_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    return genai.Client(**client_kwargs)


def _project_from_credentials(credentials_file: Optional[str] = None) -> Optional[str]:
    path = credentials_file or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("project_id")
    return value if isinstance(value, str) and value else None


def _tool_name_from_tool_message(message: Dict[str, Any]) -> str:
    for key in ("name", "tool_name"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return "tool_result"


def _text_part(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, str):
        return {"text": value}
    return {"text": json.dumps(value)}


def _content_item_to_part(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        return {"text": item}
    if not isinstance(item, dict):
        return _text_part(item)

    item_type = item.get("type")
    if item_type == "text":
        return _text_part(item.get("text"))
    if item_type == "image_url":
        image_url = item.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if isinstance(url, str) and url.startswith("data:") and "," in url:
            header, data = url.split(",", 1)
            mime_type = header[5:].split(";", 1)[0] or "image/png"
            return {"inline_data": {"mime_type": mime_type, "data": data}}
        if isinstance(url, str):
            return {"file_data": {"file_uri": url}}
    if item_type == "input_image":
        data = item.get("data")
        if isinstance(data, bytes):
            data = base64.b64encode(data).decode("ascii")
        if isinstance(data, str):
            return {
                "inline_data": {
                    "mime_type": item.get("mime_type") or "image/png",
                    "data": data,
                }
            }
    return item


def _content_to_parts(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        return [part for item in content if (part := _content_item_to_part(item))]
    part = _text_part(content)
    return [part] if part else []


def normalize_vertex_messages(
    messages: List[Dict[str, Any]],
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """Convert Kestrel/OpenAI-style chat history into google-genai contents."""
    system_prompt = None
    contents: List[Dict[str, Any]] = []

    for message in messages:
        if "parts" in message:
            if message.get("role") == "_system":
                parts = message.get("parts") or []
                if parts and isinstance(parts[0], dict):
                    system_prompt = parts[0].get("text")
                continue
            contents.append(message)
            continue

        role = message.get("role")
        if role == "system":
            text = message.get("content")
            if isinstance(text, str) and text:
                system_prompt = f"{system_prompt}\n{text}" if system_prompt else text
            continue

        if role == "tool":
            content = message.get("content")
            try:
                response_value = json.loads(content) if isinstance(content, str) else content
            except json.JSONDecodeError:
                response_value = content
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": _tool_name_from_tool_message(message),
                                "response": {"result": response_value},
                            }
                        }
                    ],
                }
            )
            continue

        parts = _content_to_parts(message.get("content"))
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
                parts.append(
                    {
                        "function_call": {
                            "id": call.get("id"),
                            "name": function.get("name"),
                            "args": arguments,
                        }
                    }
                )
            vertex_role = "model"
        else:
            vertex_role = "user"

        if parts:
            contents.append({"role": vertex_role, "parts": parts})

    return system_prompt, contents


def _convert_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    declarations = []
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        function = tool.get("function") or {}
        declarations.append(
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters_json_schema": function.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return [{"function_declarations": declarations}] if declarations else None


def _generation_config(
    system_prompt: Optional[str],
    format: Optional[str],
    tools: Optional[List[Dict[str, Any]]],
    response_format: Optional[Type[BaseModel]],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if system_prompt:
        config["system_instruction"] = system_prompt
    if "max_tokens" in kwargs:
        config["max_output_tokens"] = kwargs["max_tokens"]
    if "temperature" in kwargs:
        config["temperature"] = kwargs["temperature"]
    if "top_p" in kwargs:
        config["top_p"] = kwargs["top_p"]
    if "top_k" in kwargs:
        config["top_k"] = kwargs["top_k"]

    tool_config = _convert_tools(tools)
    if tool_config:
        config["tools"] = tool_config

    if response_format is not None and issubclass(response_format, BaseModel):
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = response_format.model_json_schema()
    elif format == "json":
        config["response_mime_type"] = "application/json"

    return config


def _extract_response(response: Any) -> LLMResponse:
    content = None
    parsed_tool_calls = []

    for candidate in getattr(response, "candidates", None) or []:
        candidate_content = getattr(candidate, "content", None)
        for part in getattr(candidate_content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                content = f"{content or ''}{text}"
            function_call = getattr(part, "function_call", None)
            if function_call:
                parsed_tool_calls.append(
                    ToolCall(
                        id=getattr(function_call, "id", None)
                        or f"vertex_call_{uuid.uuid4().hex}",
                        name=getattr(function_call, "name", None) or "",
                        arguments=dict(getattr(function_call, "args", None) or {}),
                    )
                )

    if content is None:
        try:
            content = getattr(response, "text", None)
        except ValueError:
            content = None

    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    total_tokens = getattr(usage, "total_token_count", None)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return LLMResponse(
        content=content,
        tool_calls=parsed_tool_calls or None,
        raw=response,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


class VertexAIAdapter(LLMAdapter):
    provider_name = "vertex_ai"
    default_model = "gemini-2.5-flash"

    @classmethod
    def create_provider(cls, config: Dict[str, Any]) -> ProviderInfo:
        client = _client_from_config(config)
        return ProviderInfo(
            name="vertex_ai:api",
            vendor="vertex_ai",
            route="api",
            client=client,
            adapter=cls(),
            model=config.get("model") or cls.default_model,
            is_cloud=True,
            is_local=False,
            base_url=config.get("base_url"),
            selection_hints=list(
                config.get("selection_hints") or ["google-cloud", "vision", "structured-output"]
            ),
        )

    def display_name(self) -> str:
        return "Google Vertex AI"

    def key_env_var(self) -> Optional[str]:
        return "GOOGLE_API_KEY"

    def substrate_type(self) -> str:
        return "gemini"

    def deliberation_style(self) -> str:
        return "sequential"

    def provider_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_vision=True,
            supports_structured_output=True,
            structured_output_mode=StructuredOutputMode.PROVIDER_NATIVE,
            tool_streaming_mode=ToolStreamingMode.NONSTREAM_FALLBACK,
            vision_input_mode=VisionInputMode.GEMINI_INLINE_DATA,
            model_dependent=("tools", "vision", "structured_output"),
            notes=(
                "Uses the google-genai SDK with Vertex AI or Google API-key mode.",
                "Structured output uses Vertex/Gemini response_json_schema.",
                "Streaming tool calls use a non-streaming detection pass before yielding tool handoff.",
            ),
        )

    def create_messages(
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[Union[str, bytes]]] = None,
    ) -> List[Dict[str, Any]]:
        messages = []
        if system_prompt:
            messages.append({"role": "_system", "parts": [{"text": system_prompt}]})
        parts = []
        if user_prompt:
            parts.append({"text": user_prompt})
        for image in images or []:
            data = base64.b64encode(image).decode("ascii") if isinstance(image, bytes) else image
            parts.append({"inline_data": {"mime_type": "image/png", "data": data}})
        if parts:
            messages.append({"role": "user", "parts": parts})
        return messages

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
        genai_client = client if client else _client_from_config({})
        system_prompt, contents = normalize_vertex_messages(messages)
        response = await genai_client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=_generation_config(system_prompt, format, tools, response_format, kwargs)
            or None,
        )
        return _extract_response(response)

    async def get_streaming_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        genai_client = client if client else _client_from_config({})
        system_prompt, contents = normalize_vertex_messages(messages)
        stream = await genai_client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=_generation_config(system_prompt, None, tools, response_format, kwargs)
            or None,
        )
        async for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text
                continue
            for candidate in getattr(chunk, "candidates", None) or []:
                for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
                    text = getattr(part, "text", None)
                    if text:
                        yield text

    async def get_streaming_response_with_tools(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | ToolCallStarted | LLMResponse]:
        if tools:
            # google-genai can expose function calls in streams, but Kestrel's
            # current Vertex contract declares non-stream fallback for tool
            # handoff so callers always receive a complete ToolCall.
            response = await self.get_response(
                client,
                model,
                messages,
                tools=tools,
                response_format=response_format,
                **kwargs,
            )
            if response.tool_calls:
                for index, tool_call in enumerate(response.tool_calls):
                    yield ToolCallStarted(index=index, id=tool_call.id, name=tool_call.name)
                yield response
                return
            if response.content:
                yield response.content
            return

        async for chunk in self.get_streaming_response(
            client,
            model,
            messages,
            response_format=response_format,
            **kwargs,
        ):
            yield chunk

    async def list_models(self, client: Any = None) -> List[ModelInfo]:
        genai_client = client if client else _client_from_config({})
        models = []
        # AsyncPager.__anext__ fetches the next page when the current page is
        # exhausted, so async iteration walks all SDK pages.
        async_models = await genai_client.aio.models.list()
        async for item in async_models:
            model_id = getattr(item, "name", None) or str(item)
            display_name = getattr(item, "display_name", None) or model_id
            lower_id = model_id.lower()
            category = ModelCategory.CHAT
            if "embed" in lower_id:
                category = ModelCategory.EMBEDDING
            elif "image" in lower_id or "imagen" in lower_id:
                category = ModelCategory.IMAGE
            is_chat_model = category == ModelCategory.CHAT
            models.append(
                ModelInfo(
                    id=model_id,
                    provider="vertex_ai",
                    display_name=display_name,
                    category=category,
                    description=getattr(item, "description", None),
                    context_limit=getattr(item, "input_token_limit", None),
                    supports_vision=is_chat_model
                    and any(token in lower_id for token in ("gemini", "vision")),
                    supports_tools=is_chat_model,
                    supports_streaming=is_chat_model,
                )
            )
        return models
