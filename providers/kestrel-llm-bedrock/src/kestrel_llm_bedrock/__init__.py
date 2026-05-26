"""AWS Bedrock provider plugin for Kestrel Sovereign."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Type

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

DEFAULT_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
STRUCTURED_OUTPUT_TOOL = "kestrel_structured_response"


@dataclass(frozen=True)
class BedrockClients:
    runtime: Any
    management: Any
    region: str


def _load_boto3() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 package not installed. Install with: pip install kestrel-llm-bedrock"
        ) from exc
    return boto3


def _client_config() -> Any:
    try:
        from botocore.config import Config
    except ImportError:
        return None
    return Config(retries={"max_attempts": 1})


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
    client_config = _client_config()
    if client_config is not None:
        common_kwargs["config"] = client_config
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


def _runtime_client(client: Any) -> Any:
    return client.runtime if isinstance(client, BedrockClients) else client


def _management_client(client: Any) -> Any:
    if isinstance(client, BedrockClients):
        return client.management
    if hasattr(client, "list_foundation_models"):
        return client
    if client is not None:
        raise ValueError(
            "Bedrock list_models requires BedrockClients, a Bedrock management client, or no client."
        )

    kwargs = {
        "region_name": os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1",
    }
    client_config = _client_config()
    if client_config is not None:
        kwargs["config"] = client_config
    return _load_boto3().Session().client("bedrock", **kwargs)


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

    for message in messages:
        role = message.get("role")
        if role == "system":
            content = message.get("content")
            if isinstance(content, str) and content:
                system.append({"text": content})
            continue

        if role == "tool":
            tool_use_id = message.get("tool_call_id") or message.get("id") or "tool_result"
            bedrock_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": _tool_result_content(message.get("content")),
                            }
                        }
                    ],
                }
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

        if blocks:
            bedrock_messages.append({"role": bedrock_role, "content": blocks})

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
                    "description": function.get("description", ""),
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


def _extract_response(response: Dict[str, Any]) -> LLMResponse:
    message = (response.get("output") or {}).get("message") or {}
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []

    for block in message.get("content") or []:
        if "text" in block:
            text_parts.append(block["text"])
        if "toolUse" in block:
            tool_use = block["toolUse"]
            tool_calls.append(
                ToolCall(
                    id=tool_use.get("toolUseId") or f"bedrock_tool_{uuid.uuid4().hex}",
                    name=tool_use.get("name") or "",
                    arguments=tool_use.get("input") or {},
                )
            )

    input_tokens, output_tokens, total_tokens = _usage_tokens(response)
    return LLMResponse(
        content="".join(text_parts) or None,
        tool_calls=tool_calls or None,
        raw=response,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


async def _to_thread(func: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(func, **kwargs)


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
                return LLMResponse(
                    content=json.dumps(structured),
                    raw=response,
                    input_tokens=llm_response.input_tokens,
                    output_tokens=llm_response.output_tokens,
                    total_tokens=llm_response.total_tokens,
                )
        return llm_response

    async def get_streaming_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        payload = _request_payload(model, messages, tools, response_format, kwargs)
        response = await _to_thread(_runtime_client(client).converse_stream, **payload)
        for event in response.get("stream") or []:
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            if "text" in delta:
                yield delta["text"]

    async def get_streaming_response_with_tools(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | ToolCallStarted | LLMResponse]:
        payload = _request_payload(model, messages, tools, response_format, kwargs)
        response = await _to_thread(_runtime_client(client).converse_stream, **payload)

        text_parts: List[str] = []
        tool_blocks: Dict[int, Dict[str, Any]] = {}
        input_tokens = output_tokens = total_tokens = None

        for event in response.get("stream") or []:
            if "metadata" in event:
                input_tokens, output_tokens, total_tokens = _usage_tokens(event["metadata"])
                continue

            start = event.get("contentBlockStart", {}).get("start", {})
            if "toolUse" in start:
                index = event["contentBlockStart"].get("contentBlockIndex", len(tool_blocks))
                tool_use = start["toolUse"]
                tool_blocks[index] = {
                    "id": tool_use.get("toolUseId") or f"bedrock_tool_{uuid.uuid4().hex}",
                    "name": tool_use.get("name") or "",
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
                index = delta_event.get("contentBlockIndex", 0)
                current = tool_blocks.setdefault(
                    index,
                    {"id": f"bedrock_tool_{uuid.uuid4().hex}", "name": "", "input": ""},
                )
                current["input"] += delta["toolUse"].get("input", "")

        if tool_blocks:
            tool_calls = []
            for index in sorted(tool_blocks):
                current = tool_blocks[index]
                try:
                    arguments = json.loads(current["input"]) if current["input"] else {}
                except json.JSONDecodeError:
                    arguments = {"_raw": current["input"]}
                tool_calls.append(
                    ToolCall(
                        id=current["id"],
                        name=current["name"],
                        arguments=arguments,
                    )
                )
            yield LLMResponse(
                content="".join(text_parts) or None,
                tool_calls=tool_calls,
                raw=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
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
