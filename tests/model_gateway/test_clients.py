from types import SimpleNamespace

import pytest

from proactive_assistant.model_gateway import (
    FakeModelClient,
    ModelGatewayError,
    ModelOutputValidationError,
    ModelRequest,
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
)


def make_request() -> ModelRequest:
    return ModelRequest(
        model="gpt-5.2",
        instructions="Return JSON.",
        input_text="{}",
        response_schema_name="TestSchema",
        response_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        max_output_tokens=100,
        metadata={"contract": "test"},
    )


def test_fake_model_client_records_requests() -> None:
    client = FakeModelClient({"ok": True})

    response = client.generate_structured(make_request())

    assert response.parsed == {"ok": True}
    assert response.provider == "fake"
    assert response.cached is True
    assert len(client.requests) == 1


def test_openai_responses_client_builds_structured_output_request() -> None:
    captured = {}

    class Responses:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return SimpleNamespace(
                id="resp_123",
                output_text='{"ok": true}',
                usage=SimpleNamespace(input_tokens=11, output_tokens=7),
            )

    client = OpenAIResponsesClient(client=SimpleNamespace(responses=Responses()))

    response = client.generate_structured(make_request())

    assert response.parsed == {"ok": True}
    assert response.raw_response_id == "resp_123"
    assert response.input_tokens == 11
    assert response.output_tokens == 7
    assert captured["text"]["format"]["type"] == "json_schema"
    assert captured["text"]["format"]["strict"] is True
    assert captured["store"] is False
    assert captured["max_output_tokens"] == 100
    assert captured["metadata"] == {"contract": "test"}


def test_openai_responses_client_rejects_non_json_output() -> None:
    class Responses:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(id="resp_123", output_text="not json", usage=None)

    client = OpenAIResponsesClient(client=SimpleNamespace(responses=Responses()))

    with pytest.raises(ModelOutputValidationError):
        client.generate_structured(make_request())


def test_openai_responses_client_wraps_request_failure() -> None:
    class Responses:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("upstream unavailable")

    client = OpenAIResponsesClient(client=SimpleNamespace(responses=Responses()))

    with pytest.raises(ModelGatewayError, match="OpenAI Responses request failed"):
        client.generate_structured(make_request())


def test_openai_chat_completions_client_builds_compatible_request() -> None:
    captured = {}

    class Completions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return SimpleNamespace(
                id="chatcmpl_123",
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage=SimpleNamespace(prompt_tokens=13, completion_tokens=5),
            )

    client = OpenAIChatCompletionsClient(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
        response_format="json_schema",
    )

    response = client.generate_structured(make_request())

    assert response.parsed == {"ok": True}
    assert response.provider == "openai_compatible_chat"
    assert response.raw_response_id == "chatcmpl_123"
    assert response.input_tokens == 13
    assert response.output_tokens == 5
    assert captured["model"] == "gpt-5.2"
    assert captured["messages"] == [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": "{}"},
    ]
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "TestSchema",
            "schema": make_request().response_schema,
            "strict": True,
        },
    }
    assert captured["max_tokens"] == 100


def test_openai_chat_completions_client_can_parse_fenced_json_object() -> None:
    class Completions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                id="chatcmpl_123",
                choices=[SimpleNamespace(message=SimpleNamespace(content='```json\n{"ok": true}\n```'))],
                usage=None,
            )

    client = OpenAIChatCompletionsClient(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
        response_format="json_object",
        max_tokens_param="max_completion_tokens",
    )

    response = client.generate_structured(make_request())

    assert response.parsed == {"ok": True}


def test_openai_chat_completions_client_can_parse_compatible_reasoning_content() -> None:
    class Completions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                id="chatcmpl_123",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="", reasoning_content='{"ok": true}'),
                    )
                ],
                usage=None,
            )

    client = OpenAIChatCompletionsClient(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
        response_format="json_object",
    )

    response = client.generate_structured(make_request())

    assert response.parsed == {"ok": True}


def test_openai_chat_completions_client_wraps_request_failure() -> None:
    class Completions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("upstream unavailable")

    client = OpenAIChatCompletionsClient(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )

    with pytest.raises(ModelGatewayError, match="OpenAI Chat Completions request failed"):
        client.generate_structured(make_request())


def test_openai_chat_completions_client_reports_empty_message_details() -> None:
    class Completions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                id="chatcmpl_123",
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=""))],
                usage=None,
            )

    client = OpenAIChatCompletionsClient(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
        response_format="json_schema",
    )

    with pytest.raises(ModelOutputValidationError, match="finish_reason=stop"):
        client.generate_structured(make_request())
