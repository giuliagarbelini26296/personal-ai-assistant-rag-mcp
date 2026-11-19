"""Loop de agente com tool calling: o LLM decide quais ferramentas chamar, até N passos."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .context import ConversationMemory, fit_results_to_budget, message_tokens
from .llm import LLM
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
    steps: int = 0
    tool_calls: list[dict] = field(default_factory=list)


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
        result = self.tools.execute(name, arguments)
        if "results" in result:
            # Só entra no prompt o que cabe no orçamento restante de tokens.
            used = sum(message_tokens(m) for m in messages)
            remaining = max(200, self.context_budget - used)
            result["results"] = fit_results_to_budget(result["results"], remaining)
        return result

    def run(self, question: str) -> AgentResult:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self.memory.as_messages()]
        messages.append({"role": "user", "content": question})
        result = AgentResult(answer="")
        sources: list[str] = []

        for step in range(1, self.max_steps + 1):
            result.steps = step
            response = self.llm.chat(messages, self.tools.specs())
            if not response.tool_calls:
                result.answer = (response.content or "").strip()
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in response.tool_calls
                    ],
                }
            )
            for tc in response.tool_calls:
                output = self._run_tool(tc.name, tc.arguments, messages)
                for r in output.get("results", []):
                    if r["source"] not in sources:
                        sources.append(r["source"])
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
        return result
