"""GeminiEmbedder testado sem rede: um transporte falso do httpx responde no lugar da API."""

from __future__ import annotations

import json
import math

import httpx
import pytest

from assistant.embeddings import GeminiEmbedder


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def fake_api(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.headers.get("x-goog-api-key"), body))
        # vetor não normalizado (norma 5), como a API devolve com dimensão reduzida
        return httpx.Response(
            200, json={"embeddings": [{"values": [3.0, 4.0]} for _ in body["requests"]]}
        )

    return handler


def test_documents_and_query_use_different_task_types_and_are_normalized():
    seen: list = []
    emb = GeminiEmbedder("chave-teste", dim=2, client=make_client(fake_api(seen)))
    docs = emb.embed(["a", "b"])
    query = emb.embed_query("pergunta")
    assert docs == [[0.6, 0.8], [0.6, 0.8]]
    assert math.isclose(sum(v * v for v in query), 1.0)
    assert seen[0][0] == "chave-teste"
    assert {r["taskType"] for r in seen[0][1]["requests"]} == {"RETRIEVAL_DOCUMENT"}
    assert seen[1][1]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"
    assert seen[0][1]["requests"][0]["outputDimensionality"] == 2


def test_large_input_is_split_in_batches_of_100():
    seen: list = []
    emb = GeminiEmbedder("k", dim=2, client=make_client(fake_api(seen)))
    assert len(emb.embed([f"t{i}" for i in range(230)])) == 230
    assert [len(b["requests"]) for _, b in seen] == [100, 100, 30]


def test_rate_limit_is_retried_with_backoff():
    calls, waits = [], []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED"}})
        return httpx.Response(200, json={"embeddings": [{"values": [1.0, 0.0]}]})

    emb = GeminiEmbedder("k", dim=2, client=make_client(handler), sleep=waits.append)
    assert emb.embed_query("x") == [1.0, 0.0]
    assert waits == [1, 2]


def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"status": "INVALID_ARGUMENT"}})

    emb = GeminiEmbedder("k", dim=2, client=make_client(handler), sleep=lambda s: None)
    with pytest.raises(httpx.HTTPStatusError):
        emb.embed(["x"])
    assert len(calls) == 1


def test_missing_key_fails_early():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiEmbedder("")
