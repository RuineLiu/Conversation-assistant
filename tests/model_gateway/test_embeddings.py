from __future__ import annotations

from types import SimpleNamespace

from proactive_assistant.model_gateway import FakeEmbeddingClient, OpenAIEmbeddingClient


def test_fake_embedding_client_is_deterministic() -> None:
    client = FakeEmbeddingClient(dimensions=8)

    first = client.embed_texts(["客户报价确认"], model="fixture")
    second = client.embed_texts(["客户报价确认"], model="fixture")

    assert first.embeddings == second.embeddings
    assert first.provider == "fake_embeddings"


def test_openai_embedding_client_parses_openai_compatible_response() -> None:
    fake_client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                id="emb_001",
                data=[
                    SimpleNamespace(index=1, embedding=[0.3, 0.4]),
                    SimpleNamespace(index=0, embedding=[0.1, 0.2]),
                ],
                usage=SimpleNamespace(prompt_tokens=12),
            )
        )
    )
    client = OpenAIEmbeddingClient(client=fake_client)

    response = client.embed_texts(["a", "b"], model="embedding-test")

    assert response.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert response.provider == "openai_embeddings"
    assert response.input_tokens == 12
