"""PgVectorStore contra um PostgreSQL real com pgvector.

Roda só quando TEST_DATABASE_URL está definida. Use um banco só para testes, porque as tabelas
são apagadas antes e depois de cada teste. Localmente:
    docker compose -p rag-mcp up -d pgvector
    export TEST_DATABASE_URL=postgresql://rag:rag-local-only@127.0.0.1:55433/rag_test
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from assistant.config import DEFAULT_SAMPLE_DIR
from assistant.llm import FakeLLM
from assistant.service import AssistantService

URL = os.getenv("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL não definida")


def _drop_all():
    import psycopg

    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS rag_chunks, rag_documents, rag_meta CASCADE")


@pytest.fixture
def pg_store():
    from assistant.store_pg import PgVectorStore

    _drop_all()
    store = PgVectorStore(URL, "teste-3d")
    store.reset()
    yield store
    store.reset()
    store.close()


def test_upsert_search_and_cosine_order(pg_store):
    pg_store.upsert_document("a.md", "A", ["norte", "leste"], [[1, 0, 0], [0, 1, 0]])
    pg_store.upsert_document("b.md", "B", ["nordeste"], [[0.7071, 0.7071, 0]])
    results = pg_store.search([1, 0, 0], k=3)
    assert [r.text for r in results] == ["norte", "nordeste", "leste"]
    assert results[0].score == pytest.approx(1.0, abs=1e-4)
    assert results[2].score == pytest.approx(0.0, abs=1e-4)
    assert pg_store.count_chunks() == 3


def test_reindexing_a_document_replaces_its_chunks(pg_store):
    pg_store.upsert_document("a.md", "v1", ["x", "y"], [[1, 0, 0], [0, 1, 0]])
    pg_store.upsert_document("a.md", "v2", ["z"], [[0, 0, 1]])
    assert pg_store.count_chunks() == 1
    assert pg_store.document_hash("a.md") == pg_store.content_hash("v2")
    assert pg_store.list_sources()[0]["chunks"] == 1


def test_dimension_and_embedder_mismatch_are_rejected(pg_store):
    from assistant.store_pg import PgVectorStore

    pg_store.upsert_document("a.md", "A", ["x"], [[1, 0, 0]])
    with pytest.raises(RuntimeError, match="dimensões"):
        pg_store.upsert_document("b.md", "B", ["y"], [[1, 0]])
    with pytest.raises(RuntimeError, match="embedder"):
        PgVectorStore(URL, "outro-modelo")


def test_full_service_works_on_pgvector(settings):
    _drop_all()
    svc = AssistantService(
        replace(settings, vector_store="pgvector", database_url=URL), llm=FakeLLM()
    )
    try:
        stats = svc.ingest(DEFAULT_SAMPLE_DIR)
        assert stats.chunks > 0 and svc.store.count_chunks() == stats.chunks
        top = svc.search("quanto de polvilho vai no pão de queijo?", k=1)[0]
        assert top.source.endswith("receita-pao-de-queijo.txt")
    finally:
        svc.close()
        _drop_all()
