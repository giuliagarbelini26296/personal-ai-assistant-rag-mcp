"""Gerenciamento de contexto: contagem aproximada de tokens, orçamento e memória da conversa."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def count_tokens(text: str) -> int:
    """Aproximação: ~4 caracteres por token (boa o bastante para orçamento, não para cobrança)."""
    return math.ceil(len(text) / 4) if text else 0


def message_tokens(message: dict) -> int:
    content = message.get("content") or ""
    extra = sum(count_tokens(str(tc)) for tc in message.get("tool_calls", []) or [])
    return count_tokens(content) + extra + 4  # +4 de overhead por mensagem


def fit_results_to_budget(results: list[dict], budget: int) -> list[dict]:
    """Mantém os resultados em ordem de relevância até estourar o orçamento de tokens.

    O último resultado que não cabe inteiro é truncado em vez de descartado.
    """
    selected: list[dict] = []
    used = 0
    for r in results:
        cost = count_tokens(r["text"]) + 20  # 20 ~ metadados (fonte, score)
        if used + cost <= budget:
            selected.append(r)
            used += cost
            continue
        remaining_chars = (budget - used - 20) * 4
        if remaining_chars > 80:
            selected.append({**r, "text": r["text"][:remaining_chars] + "…"})
        break
    return selected


def _shorten(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class ConversationMemory:
    """Janela das últimas mensagens + resumo acumulado (rolling summary) das antigas.

    O resumo é extrativo e determinístico (primeiras palavras de cada turno), para não
    depender de uma chamada extra ao LLM. Uma evolução natural é resumir com o próprio LLM.
    """

    max_messages: int = 6
    max_summary_tokens: int = 300
    messages: list[dict] = field(default_factory=list)
    summary: str = ""

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        while len(self.messages) > self.max_messages:
            old = self.messages.pop(0)
            label = "Usuário" if old["role"] == "user" else "Assistente"
            self.summary = (self.summary + f"\n- {label}: {_shorten(old['content'])}").strip()
        # Corta o início do resumo se ele passar do orçamento.
        while count_tokens(self.summary) > self.max_summary_tokens and "\n" in self.summary:
            self.summary = self.summary.split("\n", 1)[1]

    def as_messages(self) -> list[dict]:
        prefix = []
        if self.summary:
            prefix.append(
                {"role": "system", "content": f"Resumo da conversa anterior:\n{self.summary}"}
            )
        return prefix + list(self.messages)
