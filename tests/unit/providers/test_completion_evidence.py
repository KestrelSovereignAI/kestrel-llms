"""Provider-native completion evidence survives SDK response conversion."""

import json

import botocore.parsers
import botocore.session
from google.genai import types
from completion_peer import openai_response
import pytest

from kestrel_sdk.llm import LLMProtocolError
from kestrel_sdk.testing import assert_response_contract
from kestrel_llm_openai_compat import to_llm_response
from kestrel_llm_vertex import _extract_response as vertex_response
from kestrel_llm_bedrock import _extract_response as bedrock_response




@pytest.mark.parametrize("reason", ["stop", "tool_calls", "length", "content_filter", "function_call"])
def test_openai_native_reason_is_preserved(reason):
    native = openai_response(reason)
    response = to_llm_response(native)
    assert response.raw is native
    assert response.stop_reason == reason
    assert (response.input_tokens, response.output_tokens, response.total_tokens) == (2, 3, 5)
    assert_response_contract(response)


@pytest.mark.parametrize("indices", [(), (1,), (0, 1)])
def test_openai_rejects_missing_or_alternative_choices(indices):
    with pytest.raises(LLMProtocolError):
        to_llm_response(openai_response(indices=indices))


@pytest.mark.parametrize("reason", [types.FinishReason.STOP, types.FinishReason.MAX_TOKENS, types.FinishReason.SAFETY])
def test_vertex_native_enum_value_is_preserved(reason):
    native = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(text="owned text")]),
                                    finish_reason=reason)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=2, candidates_token_count=3, total_token_count=5),
    )
    response = vertex_response(native)
    assert response.stop_reason == reason.value
    assert type(response.stop_reason) is str
    assert response.raw is native
    assert (response.input_tokens, response.output_tokens, response.total_tokens) == (2, 3, 5)
    assert_response_contract(response)


def test_vertex_never_merges_alternative_candidates():
    native = types.GenerateContentResponse(candidates=[
        types.Candidate(content=types.Content(parts=[types.Part(text=text)]), finish_reason=types.FinishReason.STOP)
        for text in ["first", "alternative"]
    ])
    with pytest.raises(LLMProtocolError):
        vertex_response(native)


def test_vertex_no_candidates_keeps_unknown_evidence():
    native = types.GenerateContentResponse(candidates=[])
    response = vertex_response(native)
    assert response.stop_reason is None
    assert response.raw is native
    assert_response_contract(response)


@pytest.mark.parametrize("reason", ["end_turn", "tool_use", "max_tokens", "guardrail_intervened", "future_provider_reason"])
def test_bedrock_native_parser_reason_is_preserved(reason):
    model = botocore.session.get_session().get_service_model("bedrock-runtime")
    parser = botocore.parsers.create_parser(model.metadata["protocol"])
    native = parser.parse({
        "status_code": 200, "headers": {},
        "body": json.dumps({
            "output": {"message": {"role": "assistant", "content": [{"text": "owned text"}]}},
            "stopReason": reason, "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
            "metrics": {"latencyMs": 1},
        }).encode(),
    }, model.operation_model("Converse").output_shape)
    response = bedrock_response(native)
    assert response.stop_reason == reason
    assert response.raw is native
    assert (response.input_tokens, response.output_tokens, response.total_tokens) == (2, 3, 5)
    assert_response_contract(response)
