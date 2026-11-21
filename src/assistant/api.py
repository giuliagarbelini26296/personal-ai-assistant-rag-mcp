"""API REST (FastAPI): /health, /ingest, /search, /chat."""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import DEFAULT_SAMPLE_DIR, Settings
from .context import ConversationMemory
from .logging_utils import setup_logging
from .service import AssistantService

logger = logging.getLogger("assistant.api")


class IngestRequest(BaseModel):
    path: str = Field(default=str(DEFAULT_SAMPLE_DIR), description="Pasta ou arquivo a indexar")


class IngestResponse(BaseModel):
    files_seen: int
    indexed: int
    unchanged: int
    chunks: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(default=4, ge=1, le=10)


class SearchHit(BaseModel):
    source: str
    chunk_index: int
    text: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchHit]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, description="Reutilize para manter o histórico")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str]
    steps: int
    tool_calls: list[dict]
    # Observabilidade: use o trace_id para achar esta resposta no Phoenix.
    trace_id: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


def create_app(service: AssistantService | None = None) -> FastAPI:
    settings = service.settings if service else Settings()
    setup_logging(settings.log_level)
    app = FastAPI(title="Personal AI Assistant — RAG + MCP", version=__version__)
    app.state.service = service or AssistantService(settings)
    app.state.sessions: dict[str, ConversationMemory] = {}

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("erro não tratado", extra={"path": request.url.path})
        return JSONResponse(status_code=500, content={"detail": "Erro interno no servidor."})

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        """Exige o cabeçalho X-API-Key quando API_KEY está configurada.

        `secrets.compare_digest` compara em tempo constante: o tempo de resposta não revela
        quantos caracteres da chave estavam certos.
        """
        expected = app.state.service.settings.api_key
        if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
            raise HTTPException(status_code=401, detail="API key ausente ou inválida.")

    protected = [Depends(require_api_key)]

    @app.get("/health")
    def health() -> dict:
        svc: AssistantService = app.state.service
        return {
            "status": "ok",
            "version": __version__,
            "chunks_indexed": svc.store.count_chunks(),
            "embedder": svc.embedder.name,
            "llm_provider": svc.settings.llm_provider,
        }

    @app.post("/ingest", response_model=IngestResponse, dependencies=protected)
    def ingest(body: IngestRequest) -> IngestResponse:
        root = Path(app.state.service.settings.ingest_root).resolve()
        path = Path(body.path).resolve()
        # resolve() desfaz "..", e is_relative_to barra caminhos fora da pasta permitida
        if not path.is_relative_to(root):
            raise HTTPException(status_code=403, detail="Caminho fora da pasta permitida.")
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Caminho não encontrado: {body.path}")
        stats = app.state.service.ingest(path)
        return IngestResponse(**vars(stats))

    @app.post("/search", response_model=SearchResponse, dependencies=protected)
    def search(body: SearchRequest) -> SearchResponse:
        results = app.state.service.search(body.query, body.k)
        return SearchResponse(results=[SearchHit(**r.to_dict()) for r in results])

    @app.post("/chat", response_model=ChatResponse, dependencies=protected)
    def chat(body: ChatRequest) -> ChatResponse:
        sessions: dict[str, ConversationMemory] = app.state.sessions
        session_id = body.session_id or uuid.uuid4().hex[:12]
        memory = sessions.setdefault(session_id, ConversationMemory())
        try:
            result = app.state.service.new_agent(memory).run(body.message)
        except RuntimeError as exc:  # ex.: provedor de LLM mal configurado
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return ChatResponse(session_id=session_id, **vars(result))

    return app
