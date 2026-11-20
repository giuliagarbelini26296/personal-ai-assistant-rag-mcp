"""Camada de serviço compartilhada pela CLI, pela API REST e pelo servidor MCP."""

from __future__ import annotations

from pathlib import Path

from .agent import Agent
from .config import Settings
from .context import ConversationMemory
from .embeddings import Embedder, get_embedder
from .ingest import Document, IngestStats, index_document, ingest_path, slugify
from .llm import LLM, get_llm
from .observability import setup_tracing
from .retrieval import Retriever, SearchResult
from .store import open_store
from .tools import ToolRegistry


class AssistantService:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.settings = settings or Settings()
        setup_tracing(
            self.settings.otel_exporter, self.settings.otel_endpoint, self.settings.otel_jsonl_path
        )
        self.embedder = embedder or get_embedder(self.settings)
        self.store = open_store(self.settings, self.embedder.name)
        self.retriever = Retriever(self.store, self.embedder)
        self.tools = ToolRegistry(self)
        self._llm = llm

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = get_llm(self.settings)
        return self._llm

    def ingest(self, path: str | Path) -> IngestStats:
        s = self.settings
        return ingest_path(Path(path), self.store, self.embedder, s.chunk_size, s.chunk_overlap)

    def search(self, query: str, k: int = 4) -> list[SearchResult]:
        return self.retriever.search(query, k)

    def add_note(self, title: str, content: str) -> dict:
        source = f"notas/{slugify(title)}.md"
        doc = Document(source=source, text=f"# {title}\n\n{content}")
        n = index_document(
            doc, self.store, self.embedder, self.settings.chunk_size, self.settings.chunk_overlap
        )
        return {"saved": source, "chunks": n}

    def list_sources(self) -> list[dict]:
        return self.store.list_sources()

    def new_agent(self, memory: ConversationMemory | None = None) -> Agent:
        return Agent(
            self.llm,
            self.tools,
            memory=memory,
            max_steps=self.settings.max_agent_steps,
            context_budget=self.settings.context_token_budget,
        )

    def close(self) -> None:
        self.store.close()
