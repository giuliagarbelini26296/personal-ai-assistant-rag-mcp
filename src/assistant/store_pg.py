"""Vector store em PostgreSQL com a extensão pgvector.

Conceitos usados aqui (detalhes em docs/estudo/01-embeddings-e-pgvector.md):

- Coluna `vector(n)`: tipo do pgvector que guarda um vetor de n números.
- Operador `<=>`: distância de cosseno (0 = mesma direção, 2 = direções opostas).
  A similaridade é `1 - distância`.
- Índice HNSW: grafo que acha vizinhos próximos sem comparar a pergunta com todas as linhas.
  É uma busca aproximada (ANN): muito mais rápida em bases grandes, com perda mínima de recall.

A dimensão dos vetores só é conhecida no primeiro documento, por isso a tabela de chunks é
criada na primeira gravação. Trocar de embedder exige apagar os dados, porque vetores de
modelos diferentes não são comparáveis.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

try:
    import psycopg
    from pgvector.psycopg import register_vector
except ImportError as exc:  # pragma: no cover - depende do extra instalado
    raise RuntimeError("Instale o extra: pip install -e '.[pgvector]'") from exc

from .store import StoredChunk, content_hash

BASE_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS rag_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rag_documents (
    source TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    n_chunks INTEGER NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
"""


class PgVectorStore:
    def __init__(self, database_url: str, embedder_name: str) -> None:
        self._conn = psycopg.connect(database_url, autocommit=True)
        self._conn.execute(BASE_SCHEMA)
        register_vector(self._conn)
        self._embedder_name = embedder_name
        self._dim = self._meta("dim")
        self._check_embedder(embedder_name)

    def _meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM rag_meta WHERE key = %s", (key,)).fetchone()
        return row[0] if row else None

    def _check_embedder(self, name: str) -> None:
        current = self._meta("embedder")
        if current is None:
            self._conn.execute("INSERT INTO rag_meta VALUES ('embedder', %s)", (name,))
        elif current != name:
            raise RuntimeError(
                f"O banco foi criado com o embedder '{current}', mas o atual é '{name}'. "
                "Apague as tabelas rag_* e rode a ingestão de novo."
            )

    def _ensure_chunks_table(self, dim: int) -> None:
        if self._dim is not None:
            if int(self._dim) != dim:
                raise RuntimeError(f"Vetores com {dim} dimensões; o banco usa {self._dim}.")
            return
        with self._conn.transaction():
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    source TEXT NOT NULL REFERENCES rag_documents(source) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector({int(dim)}) NOT NULL
                );
                CREATE INDEX IF NOT EXISTS rag_chunks_source ON rag_chunks(source);
                CREATE INDEX IF NOT EXISTS rag_chunks_hnsw
                    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
                """
            )
            self._conn.execute("INSERT INTO rag_meta VALUES ('dim', %s)", (str(dim),))
        self._dim = str(dim)

    def _has_chunks_table(self) -> bool:
        return self._dim is not None

    content_hash = staticmethod(content_hash)

    def document_hash(self, source: str) -> str | None:
        row = self._conn.execute(
            "SELECT content_hash FROM rag_documents WHERE source = %s", (source,)
        ).fetchone()
        return row[0] if row else None

    def upsert_document(
        self, source: str, content: str, chunks: list[str], vectors: list[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks e vectors precisam ter o mesmo tamanho")
        if vectors:
            self._ensure_chunks_table(len(vectors[0]))
        now = datetime.now(UTC)
        with self._conn.transaction():
            if self._has_chunks_table():
                self._conn.execute("DELETE FROM rag_chunks WHERE source = %s", (source,))
            self._conn.execute(
                """INSERT INTO rag_documents VALUES (%s, %s, %s, %s)
                   ON CONFLICT (source) DO UPDATE SET content_hash = EXCLUDED.content_hash,
                   n_chunks = EXCLUDED.n_chunks, updated_at = EXCLUDED.updated_at""",
                (source, content_hash(content), len(chunks), now),
            )
            if chunks:
                with self._conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO rag_chunks (source, chunk_index, text, embedding) "
                        "VALUES (%s, %s, %s, %s)",
                        [
                            (source, i, text, np.array(vec, dtype=np.float32))
                            for i, (text, vec) in enumerate(zip(chunks, vectors, strict=True))
                        ],
                    )

    def search(self, query_vector: list[float], k: int = 5) -> list[StoredChunk]:
        if not self._has_chunks_table():
            return []
        rows = self._conn.execute(
            """SELECT source, chunk_index, text, 1 - (embedding <=> %s) AS score
               FROM rag_chunks ORDER BY embedding <=> %s LIMIT %s""",
            (np.array(query_vector, dtype=np.float32),) * 2 + (k,),
        ).fetchall()
        return [StoredChunk(s, i, t, float(score)) for s, i, t, score in rows]

    def list_sources(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT source, n_chunks, updated_at FROM rag_documents ORDER BY source"
        ).fetchall()
        return [
            {"source": s, "chunks": n, "updated_at": u.isoformat(timespec="seconds")}
            for s, n, u in rows
        ]

    def count_chunks(self) -> int:
        if not self._has_chunks_table():
            return 0
        return self._conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]

    def reset(self) -> None:
        """Apaga tudo (usado nos testes e ao trocar de embedder)."""
        self._conn.execute("DROP TABLE IF EXISTS rag_chunks, rag_documents, rag_meta CASCADE")
        self._conn.execute(BASE_SCHEMA)
        self._dim = None
        self._check_embedder(self._embedder_name)

    def close(self) -> None:
        self._conn.close()
