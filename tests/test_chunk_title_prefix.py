"""Cabeçalho contextual: o título do documento repetido em cada chunk (opcional)."""

from assistant.embeddings import HashingEmbedder
from assistant.ingest import Document, document_title, index_document


class MemoryStore:
    def __init__(self):
        self.chunks = {}

    def upsert_document(self, source, content, chunks, vectors):
        self.chunks[source] = chunks


def test_document_title():
    assert document_title("# Copom: 282ª reunião\n\ntexto") == "Copom: 282ª reunião"
    assert document_title("sem título") is None


def test_title_is_repeated_only_when_enabled():
    text = "# Copom de novembro\n\n" + " ".join(f"p{i}" for i in range(300))
    doc = Document("copom/x.md", text)

    off, on = MemoryStore(), MemoryStore()
    index_document(doc, off, HashingEmbedder(), chunk_size=120, overlap=30)
    index_document(doc, on, HashingEmbedder(), chunk_size=120, overlap=30, title_prefix=True)

    assert len(on.chunks["copom/x.md"]) == len(off.chunks["copom/x.md"]) > 1
    assert not off.chunks["copom/x.md"][1].startswith("[")
    first, *rest = on.chunks["copom/x.md"]
    assert first.startswith("# Copom de novembro")  # o 1º já tem o título; não duplica
    assert all(c.startswith("[Copom de novembro]\n") for c in rest)
