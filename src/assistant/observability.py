"""Observabilidade com OpenTelemetry: traces de cada pergunta, tokens, custo e latência.

Conceitos (detalhes em docs/estudo/03-observabilidade.md):

- **Trace**: a história completa de uma pergunta. É formado por **spans**.
- **Span**: uma etapa com início, fim e atributos (ex.: "llm.chat", 820 ms, 1.200 tokens).
  Spans ficam aninhados: `agent.run` > `llm.chat` / `tool.search_notes` > `retrieval.search`.
- **Exporter**: para onde os spans vão. Aqui: nenhum (padrão), console, arquivo JSONL, OTLP
  (protocolo padrão, aceito por Phoenix, Langfuse, Jaeger, Grafana Tempo, Datadog...) ou
  Azure Monitor (Application Insights).

Usar OpenTelemetry, e não o SDK de um fornecedor, permite trocar a ferramenta de visualização
sem mexer no código. Os nomes de atributos seguem as convenções `gen_ai.*` do OpenTelemetry
para IA generativa.

Se o SDK não estiver instalado ou `OTEL_EXPORTER=none`, a API do OpenTelemetry vira "no-op":
os spans são criados e descartados quase sem custo, e o código não precisa de `if`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from opentelemetry import trace

logger = logging.getLogger(__name__)
TRACER_NAME = "personal-ai-assistant"

# Preço em US$ por 1 milhão de tokens (entrada, saída), plano Standard pago.
# Fonte: https://ai.google.dev/gemini-api/docs/pricing (consultado em 20/11/2026).
# No plano gratuito o custo real é zero; o valor estimado mostra quanto custaria em produção.
# A saída inclui os tokens de "raciocínio" (thinking) dos modelos Gemini 3.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
}

_lock = threading.Lock()
_configured = False


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Custo estimado da chamada, ou None quando o preço do modelo não está na tabela.

    Variáveis LLM_PRICE_INPUT_PER_M e LLM_PRICE_OUTPUT_PER_M sobrescrevem a tabela.
    """
    env_in, env_out = os.getenv("LLM_PRICE_INPUT_PER_M"), os.getenv("LLM_PRICE_OUTPUT_PER_M")
    if env_in and env_out:
        price_in, price_out = float(env_in), float(env_out)
    elif model in PRICES_PER_MILLION:
        price_in, price_out = PRICES_PER_MILLION[model]
    else:
        return None
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 8)


# Convenção OpenInference (lida pelo Phoenix e por outras ferramentas de LLM). Gravamos estes
# atributos junto com os gen_ai.* do OpenTelemetry: cada ferramenta entende uma convenção.
SPAN_KIND = "openinference.span.kind"


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


class JsonlSpanExporter:
    """Grava cada span como uma linha JSON. Bom para análise local e para o script
    `scripts/trace_report.py`, sem precisar subir nenhum serviço."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans) -> object:
        from opentelemetry.sdk.trace.export import SpanExportResult

        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            for s in spans:
                parent = s.parent.span_id if s.parent else None
                fh.write(
                    json.dumps(
                        {
                            "trace_id": format(s.context.trace_id, "032x"),
                            "span_id": format(s.context.span_id, "016x"),
                            "parent_id": format(parent, "016x") if parent else None,
                            "name": s.name,
                            "start_ns": s.start_time,
                            "duration_ms": round((s.end_time - s.start_time) / 1e6, 2),
                            "status": s.status.status_code.name,
                            "attributes": dict(s.attributes or {}),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def setup_tracing(
    exporter: str = "none",
    otlp_endpoint: str = "http://127.0.0.1:6006/v1/traces",
    jsonl_path: str = "traces/spans.jsonl",
    service_name: str = TRACER_NAME,
) -> bool:
    """Configura o envio de traces uma única vez por processo. Devolve True se ativou.

    `exporter` aceita vários destinos separados por vírgula, ex.: "otlp,jsonl".
    """
    global _configured
    kinds = [k.strip().lower() for k in exporter.split(",") if k.strip()]
    if not kinds or kinds == ["none"]:
        return False
    unknown = set(kinds) - {"console", "jsonl", "otlp", "azure"}
    if unknown:
        raise ValueError(f"OTEL_EXPORTER desconhecido: {sorted(unknown)}")
    with _lock:
        if _configured:
            return True
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )
        except ImportError:
            logger.warning(
                "OTEL_EXPORTER=%s, mas o extra não está instalado: "
                "pip install -e '.[observability]'",
                exporter,
            )
            return False
        provider = TracerProvider(
            resource=Resource.create(
                # project.name: nome do projeto na tela do Phoenix
                {"service.name": service_name, "openinference.project.name": service_name}
            )
        )
        if "console" in kinds:
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        if "jsonl" in kinds:
            provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(jsonl_path)))
        if "otlp" in kinds:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            # Batch: envia em lotes, em segundo plano, sem atrasar a resposta ao usuário.
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
        if "azure" in kinds:
            # Application Insights (Azure Monitor). A connection string vem da variável
            # APPLICATIONINSIGHTS_CONNECTION_STRING, definida pela infraestrutura (Bicep).
            from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

            provider.add_span_processor(BatchSpanProcessor(AzureMonitorTraceExporter()))
        trace.set_tracer_provider(provider)
        _configured = True
        logger.info("tracing ativo", extra={"exporter": ",".join(kinds)})
        return True


def shutdown_tracing() -> None:
    """Envia os spans pendentes (importante em scripts curtos antes de sair)."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
