"""Loop de agente com tool calling: o LLM decide quais ferramentas chamar, até N passos."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from opentelemetry.trace import Status, StatusCode

from .context import ConversationMemory, fit_results_to_budget, message_tokens
from .llm import LLM, ToolCall
from .observability import SPAN_KIND, current_trace_id, estimate_cost_usd, tracer
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Você é um assistente pessoal de estudos que responde em português.
Regras:
- Para perguntas sobre as notas do usuário, use a ferramenta search_notes antes de responder.
- Responda somente com base nos trechos retornados e cite as fontes no formato [fonte: arquivo].
- Se não encontrar a informação, diga isso claramente. Nunca invente.
- Se uma ferramenta retornar "error", corrija os argumentos e tente de novo,
  ou explique o problema."""


@dataclass
class AgentResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    # textos dos trechos que o modelo recebeu (usado na avaliação de fidelidade)
    contexts: list[str] = field(default_factory=list)
    steps: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    # Observabilidade: consumo da pergunta inteira (todas as chamadas ao LLM somadas)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: float = 0.0
    trace_id: str | None = None


def _tool_call_message(tc: ToolCall) -> dict:
    message = {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": tc.arguments},
    }
    if tc.extra:
        message["extra_content"] = tc.extra  # ex.: thought_signature do Gemini 3
    return message


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        memory: ConversationMemory | None = None,
        max_steps: int = 4,
        context_budget: int = 1500,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.memory = memory or ConversationMemory()
        self.max_steps = max_steps
        self.context_budget = context_budget

    def _run_tool(self, name: str, arguments: str, messages: list[dict]) -> dict:
        with tracer().start_as_current_span(f"tool.{name}") as span:
            span.set_attribute(SPAN_KIND, "TOOL")
            span.set_attribute("tool.name", name)
            span.set_attribute("tool.arguments", arguments[:500])
            span.set_attribute("input.value", arguments[:500])
            result = self.tools.execute(name, arguments)
            if "error" in result:
                span.set_status(Status(StatusCode.ERROR, str(result["error"])[:200]))
            if "results" in result:
                # Só entra no prompt o que cabe no orçamento restante de tokens.
                used = sum(message_tokens(m) for m in messages)
                remaining = max(200, self.context_budget - used)
                before = len(result["results"])
                result["results"] = fit_results_to_budget(result["results"], remaining)
                span.set_attribute("retrieval.results", before)
                span.set_attribute("retrieval.results_in_prompt", len(result["results"]))
                if result["results"]:
                    span.set_attribute("retrieval.top_score", result["results"][0]["score"])
                    span.set_attribute(
                        "retrieval.sources", [r["source"] for r in result["results"]]
                    )
            return result

    def _call_llm(self, messages: list[dict], step: int, result: AgentResult):
        model = getattr(self.llm, "model", getattr(self.llm, "name", "desconhecido"))
        with tracer().start_as_current_span("llm.chat") as span:
            span.set_attribute(SPAN_KIND, "LLM")
            span.set_attribute("llm.model_name", model)
            span.set_attribute("llm.provider", getattr(self.llm, "provider", "fake"))
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.system", getattr(self.llm, "provider", "fake"))
            span.set_attribute("gen_ai.request.model", model)
            span.set_attribute("agent.step", step)
            span.set_attribute("gen_ai.request.message_count", len(messages))
            response = self.llm.chat(messages, self.tools.specs())
            span.set_attribute(
                "gen_ai.response.tool_calls", [tc.name for tc in response.tool_calls]
            )
            if response.content:
                span.set_attribute("output.value", response.content[:2000])
            if response.usage:
                u = response.usage
                span.set_attribute("gen_ai.usage.input_tokens", u.input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", u.output_tokens)
                span.set_attribute("gen_ai.usage.reasoning_tokens", u.reasoning_tokens)
                span.set_attribute("llm.token_count.prompt", u.input_tokens)
                span.set_attribute("llm.token_count.completion", u.output_tokens)
                span.set_attribute("llm.token_count.total", u.input_tokens + u.output_tokens)
                cost = estimate_cost_usd(model, u.input_tokens, u.output_tokens)
                if cost is not None:
                    span.set_attribute("llm.cost_usd", cost)
                    result.cost_usd = round((result.cost_usd or 0.0) + cost, 8)
                result.input_tokens += u.input_tokens
                result.output_tokens += u.output_tokens
            return response

    def run(self, question: str) -> AgentResult:
        started = time.perf_counter()
        with tracer().start_as_current_span("agent.run") as span:
            span.set_attribute(SPAN_KIND, "AGENT")
            span.set_attribute("agent.question", question[:500])
            span.set_attribute("input.value", question[:500])
            span.set_attribute("agent.max_steps", self.max_steps)
            result = AgentResult(answer="", trace_id=current_trace_id())
            try:
                self._loop(question, result)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise
            finally:
                result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
                span.set_attribute("agent.steps", result.steps)
                span.set_attribute("agent.sources", result.sources)
                span.set_attribute("agent.answer_chars", len(result.answer))
                span.set_attribute("output.value", result.answer[:2000])
                span.set_attribute("gen_ai.usage.input_tokens", result.input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", result.output_tokens)
                if result.cost_usd is not None:
                    span.set_attribute("llm.cost_usd", result.cost_usd)
            logger.info(
                "pergunta respondida",
                extra={
                    "steps": result.steps,
                    "latency_ms": result.latency_ms,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cost_usd": result.cost_usd,
                },
            )
            return result

    def _loop(self, question: str, result: AgentResult) -> None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self.memory.as_messages()]
        messages.append({"role": "user", "content": question})
        sources: list[str] = []

        for step in range(1, self.max_steps + 1):
            result.steps = step
            response = self._call_llm(messages, step, result)
            if not response.tool_calls:
                result.answer = (response.content or "").strip()
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [_tool_call_message(tc) for tc in response.tool_calls],
                }
            )
            for tc in response.tool_calls:
                output = self._run_tool(tc.name, tc.arguments, messages)
                for r in output.get("results", []):
                    if r["source"] not in sources:
                        sources.append(r["source"])
                    result.contexts.append(f"[{r['source']}] {r['text']}")
                result.tool_calls.append(
                    {"name": tc.name, "arguments": tc.arguments, "ok": "error" not in output}
                )
                logger.info("tool executada", extra={"tool": tc.name, "ok": "error" not in output})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(output, ensure_ascii=False),
                    }
                )
        else:
            result.answer = (
                f"Não cheguei a uma resposta final em {self.max_steps} passos. "
                "Tente reformular a pergunta."
            )

        if sources and "[fonte:" not in result.answer:
            result.answer += "\n\nFontes consultadas: " + ", ".join(sources)
        result.sources = sources
        self.memory.add("user", question)
        self.memory.add("assistant", result.answer)
