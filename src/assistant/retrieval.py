"""Retrieval híbrido simples: cosseno (vetorial) + sobreposição de palavras-chave, com dedupe."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .embeddings import Embedder, tokenize
from .observability import SPAN_KIND, tracer
from .store import Store


@dataclass(frozen=True)
class SearchResult:
    source: str
    chunk_index: int
    text: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def keyword_score(query: str, text: str) -> float:
    """Fração dos termos da pergunta que aparecem no chunk (0..1)."""
    q_terms = set(tokenize(query))
    if not q_terms:
        return 0.0
    return len(q_terms & set(tokenize(text))) / len(q_terms)


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def dedupe(results: list[SearchResult], threshold: float = 0.8) -> list[SearchResult]:
    """Remove chunks quase idênticos (ex.: overlap grande ou nota duplicada)."""
    kept: list[SearchResult] = []
    for r in results:
        if all(jaccard(r.text, k.text) < threshold for k in kept):
            kept.append(r)
    return kept


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        vector_weight: float = 0.7,
        min_score: float = 0.05,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.vector_weight = vector_weight
        self.min_score = min_score

    def search(self, query: str, k: int = 4) -> list[SearchResult]:
        with tracer().start_as_current_span("retrieval.search") as span:
            span.set_attribute(SPAN_KIND, "RETRIEVER")
            span.set_attribute("input.value", query[:500])
            span.set_attribute("retrieval.k", k)
            span.set_attribute("retrieval.embedder", getattr(self.embedder, "name", "?"))
            results = self._search(query, k)
            span.set_attribute("retrieval.returned", len(results))
            if results:
                span.set_attribute("retrieval.top_score", results[0].score)
            for i, r in enumerate(results):  # documentos recuperados, na ordem
                prefix = f"retrieval.documents.{i}.document"
                span.set_attribute(f"{prefix}.id", f"{r.source}#{r.chunk_index}")
                span.set_attribute(f"{prefix}.score", r.score)
                span.set_attribute(f"{prefix}.content", r.text[:1000])
            return results

    def _search(self, query: str, k: int) -> list[SearchResult]:
        with tracer().start_as_current_span("embedding.query") as span:
            span.set_attribute(SPAN_KIND, "EMBEDDING")
            span.set_attribute("embedding.model_name", getattr(self.embedder, "name", "?"))
            span.set_attribute("gen_ai.operation.name", "embeddings")
            span.set_attribute("gen_ai.request.model", getattr(self.embedder, "name", "?"))
            query_vec = self.embedder.embed_query(query)
        # Busca mais candidatos do que o necessário e reordena (rerank leve).
        with tracer().start_as_current_span("vector_store.search") as span:
            span.set_attribute("db.system", type(self.store).__name__)
            candidates = self.store.search(query_vec, k=max(k * 4, 10))
            span.set_attribute("vector_store.candidates", len(candidates))
        w = self.vector_weight
        rescored = [
            SearchResult(
                c.source,
                c.chunk_index,
                c.text,
                round(w * c.score + (1 - w) * keyword_score(query, c.text), 4),
            )
            for c in candidates
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return [r for r in dedupe(rescored) if r.score >= self.min_score][:k]
