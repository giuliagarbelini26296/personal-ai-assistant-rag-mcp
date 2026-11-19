"""Vector store simples em SQLite: vetores salvos como JSON, similaridade de cosseno com numpy."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    source TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    n_chunks INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL REFERENCES documents(source) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
"""


@dataclass(frozen=True)
class StoredChunk:
    source: str
    chunk_index: int
    text: str
    score: float


class VectorStore:
    def __init__(self, db_path: str, embedder_name: str) -> None:
        # check_same_thread=False: FastAPI pode executar rotas síncronas em threads diferentes.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._check_embedder(embedder_name)

    def _check_embedder(self, name: str) -> None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
        if row is None:
            self._conn.execute("INSERT INTO meta VALUES ('embedder', ?)", (name,))
            self._conn.commit()
        elif row[0] != name:
            raise RuntimeError(
                f"O banco foi criado com o embedder '{row[0]}', mas o atual é '{name}'. "
                "Apague o arquivo do banco e rode a ingestão de novo."
            )

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def document_hash(self, source: str) -> str | None:
        row = self._conn.execute(
            "SELECT content_hash FROM documents WHERE source = ?", (source,)
        ).fetchone()
        return row[0] if row else None

    def upsert_document(
        self, source: str, content: str, chunks: list[str], vectors: list[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks e vectors precisam ter o mesmo tamanho")
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self._conn:  # transação
            self._conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
            self._conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?)",
                (source, self.content_hash(content), len(chunks), now),
            )
            self._conn.executemany(
                "INSERT INTO chunks (source, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
                [
                    (source, i, text, json.dumps(vec))
                    for i, (text, vec) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )

    def search(self, query_vector: list[float], k: int = 5) -> list[StoredChunk]:
        rows = self._conn.execute(
            "SELECT source, chunk_index, text, embedding FROM chunks"
        ).fetchall()
        if not rows:
            return []
        matrix = np.array([json.loads(r[3]) for r in rows], dtype=np.float32)
        query = np.array(query_vector, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
        scores = (matrix @ query) / np.where(norms == 0, 1.0, norms)
        top = np.argsort(-scores)[:k]
        return [StoredChunk(rows[i][0], rows[i][1], rows[i][2], float(scores[i])) for i in top]

    def list_sources(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT source, n_chunks, updated_at FROM documents ORDER BY source"
        ).fetchall()
        return [{"source": s, "chunks": n, "updated_at": u} for s, n, u in rows]

    def count_chunks(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
