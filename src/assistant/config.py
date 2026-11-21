"""Configuração lida de variáveis de ambiente (com padrões que funcionam offline)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    db_path: str = field(default_factory=lambda: _env_str("ASSISTANT_DB_PATH", "assistant.sqlite3"))
    # sqlite (padrão, sem servidor) | pgvector (PostgreSQL com a extensão pgvector)
    vector_store: str = field(default_factory=lambda: _env_str("VECTOR_STORE", "sqlite"))
    database_url: str = field(default_factory=lambda: _env_str("DATABASE_URL"))
    embedding_provider: str = field(default_factory=lambda: _env_str("EMBEDDING_PROVIDER", "hash"))
    embedding_model: str = field(default_factory=lambda: _env_str("EMBEDDING_MODEL"))
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 768))
    gemini_api_key: str = field(default_factory=lambda: _env_str("GEMINI_API_KEY"))
    llm_provider: str = field(default_factory=lambda: _env_str("LLM_PROVIDER", "fake"))
    llm_model: str = field(default_factory=lambda: _env_str("LLM_MODEL"))
    openai_api_key: str = field(default_factory=lambda: _env_str("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: _env_str("OPENAI_BASE_URL"))
    ollama_base_url: str = field(
        default_factory=lambda: _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 120))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 30))
    # Repete o título do documento (1ª linha "# ...") no início de cada chunk. Ajuda quando os
    # documentos se parecem muito e o que os diferencia (data, nome) só aparece no título.
    chunk_title_prefix: bool = field(
        default_factory=lambda: (
            _env_str("CHUNK_TITLE_PREFIX", "false").lower() in {"1", "true", "sim"}
        )
    )
    context_token_budget: int = field(
        default_factory=lambda: _env_int("CONTEXT_TOKEN_BUDGET", 1500)
    )
    max_agent_steps: int = field(default_factory=lambda: _env_int("MAX_AGENT_STEPS", 4))
    log_level: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO"))
    # Observabilidade: none (padrão) | console | jsonl | otlp
    otel_exporter: str = field(default_factory=lambda: _env_str("OTEL_EXPORTER", "none"))
    otel_endpoint: str = field(
        default_factory=lambda: _env_str(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:6006/v1/traces"
        )
    )
    otel_jsonl_path: str = field(
        default_factory=lambda: _env_str("OTEL_JSONL_PATH", "traces/spans.jsonl")
    )
