"""Leitura de documentos (Markdown, TXT, JSON) e indexação no vector store."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_text
from .embeddings import Embedder
from .store import Store, content_hash

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".md", ".txt", ".json"}


@dataclass(frozen=True)
class Document:
    source: str
    text: str


@dataclass
class IngestStats:
    files_seen: int = 0
    indexed: int = 0
    unchanged: int = 0
    chunks: int = 0


def _json_to_text(data: object) -> list[str]:
    """Achata um JSON em linhas legíveis ('chave: valor') para que possa ser buscado."""
    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            if isinstance(value, dict | list):
                lines.append(f"{key}:")
                lines.extend(_json_to_text(value))
            else:
                lines.append(f"{key}: {value}")
        return lines
    if isinstance(data, list):
        return [
            "- " + "; ".join(f"{k}: {v}" for k, v in item.items())
            if isinstance(item, dict)
            else f"- {item}"
            for item in data
        ]
    return [str(data)]


def read_document(path: Path, root: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        raw = "\n".join(_json_to_text(json.loads(raw)))
    return Document(source=path.relative_to(root).as_posix(), text=raw)


def load_documents(path: Path) -> list[Document]:
    path = Path(path)
    if path.is_file():
        return [read_document(path, path.parent)]
    if not path.is_dir():
        raise FileNotFoundError(f"Caminho não encontrado: {path}")
    files = sorted(p for p in path.rglob("*") if p.suffix in SUPPORTED_SUFFIXES and p.is_file())
    return [read_document(p, path) for p in files]


def index_document(
    doc: Document, store: Store, embedder: Embedder, chunk_size: int, overlap: int
) -> int:
    chunks = [c.text for c in chunk_text(doc.text, chunk_size, overlap)]
    if not chunks:
        return 0
    store.upsert_document(doc.source, doc.text, chunks, embedder.embed(chunks))
    return len(chunks)


def ingest_path(
    path: Path, store: Store, embedder: Embedder, chunk_size: int, overlap: int
) -> IngestStats:
    stats = IngestStats()
    for doc in load_documents(path):
        stats.files_seen += 1
        if store.document_hash(doc.source) == content_hash(doc.text):
            stats.unchanged += 1
            continue
        n = index_document(doc, store, embedder, chunk_size, overlap)
        stats.indexed += 1
        stats.chunks += n
        logger.info("documento indexado", extra={"source": doc.source, "chunks": n})
    return stats


def slugify(title: str) -> str:
    from .embeddings import normalize

    slug = re.sub(r"[^a-z0-9]+", "-", normalize(title)).strip("-")
    return slug[:60] or "nota"
