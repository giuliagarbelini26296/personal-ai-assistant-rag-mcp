import pytest

from assistant.config import DEFAULT_SAMPLE_DIR
from assistant.context import count_tokens
from assistant.embeddings import HashingEmbedder
from assistant.retrieval import SearchResult, dedupe
from assistant.store import VectorStore


def test_ingest_is_idempotent(service):
    assert len(service.list_sources()) == 6
    again = service.ingest(DEFAULT_SAMPLE_DIR)
    assert again.indexed == 0 and again.unchanged == 6


def test_search_returns_relevant_source_with_score(service):
    results = service.search("temperatura do forno pão de queijo", k=3)
    assert results[0].source == "receita-pao-de-queijo.txt"
    assert 0 < results[0].score <= 1
    assert all(a.score >= b.score for a, b in zip(results, results[1:], strict=False))


def test_search_json_document(service):
    results = service.search("livro sobre TF-IDF e similaridade de cosseno", k=2)
    assert "lista-leitura.json" in {r.source for r in results}


def test_add_note_becomes_searchable(service):
    saved = service.add_note("Ideia de projeto", "Criar um bot que lembra de regar as plantas")
    assert saved == {"saved": "notas/ideia-de-projeto.md", "chunks": 1}
    assert service.search("regar plantas", k=1)[0].source == "notas/ideia-de-projeto.md"


def test_store_rejects_different_embedder(tmp_path):
    db = str(tmp_path / "x.sqlite3")
    VectorStore(db, "hash-bow-1024").close()
    with pytest.raises(RuntimeError, match="embedder"):
        VectorStore(db, "openai-text-embedding-3-small")


def test_store_search_cosine(tmp_path):
    emb = HashingEmbedder(dim=64)
    store = VectorStore(str(tmp_path / "s.sqlite3"), emb.name)
    texts = ["gato preto", "cachorro caramelo"]
    store.upsert_document("animais.md", "\n".join(texts), texts, emb.embed(texts))
    top = store.search(emb.embed(["gato preto"])[0], k=1)[0]
    assert top.text == "gato preto" and top.score == pytest.approx(1.0, abs=1e-5)


def test_dedupe_removes_near_duplicates():
    a = SearchResult("a.md", 0, "usar sqlite como vector store no prototipo", 0.9)
    b = SearchResult("b.md", 0, "usar sqlite como vector store no prototipo!", 0.8)
    c = SearchResult("c.md", 0, "receita de pao de queijo", 0.5)
    assert dedupe([a, b, c]) == [a, c]


def test_count_tokens_approximation():
    assert count_tokens("") == 0
    assert count_tokens("a" * 40) == 10
