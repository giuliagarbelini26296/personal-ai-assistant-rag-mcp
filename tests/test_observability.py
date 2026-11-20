"""Observabilidade: cada pergunta gera um trace com a árvore de spans, tokens e custo.

Os spans vão para um exporter em memória, então o teste consegue inspecionar tudo sem rede.
"""

from __future__ import annotations

import json
import logging

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from assistant.agent import Agent
from assistant.llm import FakeLLM, LLMResponse, ToolCall, Usage
from assistant.logging_utils import JsonFormatter
from assistant.observability import JsonlSpanExporter, estimate_cost_usd, tracer

EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def _provider():
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
    trace.set_tracer_provider(provider)  # só pode ser definido uma vez por processo
    yield


@pytest.fixture(autouse=True)
def _clear():
    EXPORTER.clear()


class MeteredLLM:
    """Responde como um LLM real: chama a busca e depois responde, informando tokens."""

    provider = "gemini"
    model = "gemini-3.5-flash-lite"
    name = "gemini:gemini-3.5-flash-lite"

    def __init__(self):
        self.turn = 0

    def chat(self, messages, tools):
        self.turn += 1
        if self.turn == 1:
            return LLMResponse(
                tool_calls=[ToolCall("c1", "search_notes", '{"query": "polvilho"}')],
                usage=Usage(1000, 50, 30),
            )
        return LLMResponse(content="Use polvilho [fonte: receita]", usage=Usage(2000, 100))


def spans_by_name():
    return {s.name: s for s in EXPORTER.get_finished_spans()}


def test_question_produces_nested_spans_with_tokens_and_cost(service):
    result = Agent(MeteredLLM(), service.tools).run("quanto polvilho?")
    spans = spans_by_name()
    assert {
        "agent.run",
        "llm.chat",
        "tool.search_notes",
        "retrieval.search",
        "embedding.query",
        "vector_store.search",
    } <= set(spans)

    root = spans["agent.run"]
    assert root.parent is None
    assert spans["tool.search_notes"].parent.span_id == root.context.span_id
    assert spans["retrieval.search"].parent.span_id == spans["tool.search_notes"].context.span_id
    assert {s.context.trace_id for s in EXPORTER.get_finished_spans()} == {root.context.trace_id}

    # tokens somados das duas chamadas ao LLM: 1000+2000 de entrada, 50+100 de saída
    assert (result.input_tokens, result.output_tokens) == (3000, 150)
    assert root.attributes["gen_ai.usage.input_tokens"] == 3000
    # 3000 * 0,30/1M + 150 * 2,50/1M = 0,000900 + 0,000375
    assert result.cost_usd == pytest.approx(0.001275)
    assert root.attributes["llm.cost_usd"] == pytest.approx(0.001275)
    assert result.trace_id == format(root.context.trace_id, "032x")
    assert result.latency_ms > 0

    llm_spans = [s for s in EXPORTER.get_finished_spans() if s.name == "llm.chat"]
    assert [s.attributes["agent.step"] for s in llm_spans] == [1, 2]
    assert llm_spans[0].attributes["gen_ai.usage.reasoning_tokens"] == 30
    assert llm_spans[0].attributes["gen_ai.response.tool_calls"] == ("search_notes",)
    assert spans["tool.search_notes"].attributes["retrieval.top_score"] > 0


def test_fake_llm_without_usage_has_no_cost(service):
    result = Agent(FakeLLM(), service.tools).run("quanto polvilho vai na receita?")
    assert result.input_tokens == 0 and result.cost_usd is None
    assert "llm.cost_usd" not in spans_by_name()["agent.run"].attributes


def test_tool_error_marks_span_as_error(service):
    class BadArgs(MeteredLLM):
        def chat(self, messages, tools):
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(tool_calls=[ToolCall("c1", "search_notes", '{"k": 99}')])
            return LLMResponse(content="não consegui")

    Agent(BadArgs(), service.tools).run("x")
    assert spans_by_name()["tool.search_notes"].status.status_code.name == "ERROR"


def test_exception_is_recorded_on_root_span(service):
    class Broken:
        def chat(self, messages, tools):
            raise TimeoutError("api lenta")

    with pytest.raises(TimeoutError):
        Agent(Broken(), service.tools).run("x")
    root = spans_by_name()["agent.run"]
    assert root.status.status_code.name == "ERROR"
    assert root.events and root.events[0].name == "exception"


def test_logs_carry_the_trace_id():
    record = logging.makeLogRecord({"msg": "oi", "levelname": "INFO", "name": "t"})
    with tracer().start_as_current_span("x") as span:
        payload = json.loads(JsonFormatter().format(record))
    assert payload["trace_id"] == format(span.get_span_context().trace_id, "032x")
    assert "trace_id" not in json.loads(JsonFormatter().format(record))


def test_jsonl_exporter_writes_one_line_per_span(tmp_path, service):
    path = tmp_path / "spans.jsonl"
    exporter = JsonlSpanExporter(str(path))
    Agent(MeteredLLM(), service.tools).run("quanto polvilho?")
    exporter.export(EXPORTER.get_finished_spans())
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    root = next(s for s in lines if s["name"] == "agent.run")
    assert root["parent_id"] is None and root["attributes"]["gen_ai.usage.input_tokens"] == 3000
    assert all(s["trace_id"] == root["trace_id"] for s in lines)


def test_cost_estimate_uses_table_and_env_override(monkeypatch):
    assert estimate_cost_usd("gemini-3.5-flash-lite", 1_000_000, 0) == 0.30
    assert estimate_cost_usd("modelo-desconhecido", 10, 10) is None
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_M", "1")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_M", "2")
    assert estimate_cost_usd("modelo-desconhecido", 1_000_000, 1_000_000) == 3.0


def test_chat_api_returns_trace_and_usage(service):
    from fastapi.testclient import TestClient

    from assistant.api import create_app

    body = (
        TestClient(create_app(service)).post("/chat", json={"message": "quanto polvilho?"}).json()
    )
    assert body["trace_id"] and len(body["trace_id"]) == 32
    assert body["latency_ms"] > 0 and "cost_usd" in body
