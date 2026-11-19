"""Provedores de LLM no formato de chat/tool calling compatível com a API da OpenAI.

- `OpenAIChatLLM`: OpenAI ou qualquer endpoint compatível (ex.: Ollama em /v1).
- `FakeLLM`: determinístico, sem rede. Usado nos testes e no modo demo offline.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings
from .embeddings import tokenize


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON em string, como na API da OpenAI


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class OpenAIChatLLM:
    def __init__(self, model: str, api_key: str, base_url: str = "", name: str = "openai") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Instale o extra: pip install -e '.[openai]'") from exc
        self.model = model
        self.name = f"{name}:{model}"
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model, messages=messages, tools=tools or None, temperature=0
        )
        msg = response.choices[0].message
        calls = [
            ToolCall(tc.id, tc.function.name, tc.function.arguments or "{}")
            for tc in (msg.tool_calls or [])
        ]
        return LLMResponse(content=msg.content, tool_calls=calls)


NOTE_PATTERN = r"^(anote|salve (uma )?nota|adicione (uma )?nota)\s*[:\-]?\s*(.+)$"


def _new_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


class FakeLLM:
    """LLM falso baseado em regras, para rodar tudo sem chave de API.

    - Se receber `script`, devolve as respostas na ordem (útil para testes do loop).
    - Caso contrário: escolhe uma ferramenta pela intenção da pergunta e, depois de
      receber o resultado, monta uma resposta extrativa citando as fontes.
    """

    name = "fake"

    def __init__(self, script: list[LLMResponse] | None = None) -> None:
        self._script = list(script) if script else None
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append(list(messages))
        if self._script is not None:
            if not self._script:
                return LLMResponse(content="(script vazio)")
            return self._script.pop(0)

        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        question = messages[last_user]["content"]
        tool_msgs = [m for m in messages[last_user + 1 :] if m["role"] == "tool"]
        if not tool_msgs:
            return self._choose_tool(question)
        return LLMResponse(content=self._answer(question, json.loads(tool_msgs[-1]["content"])))

    @staticmethod
    def _choose_tool(question: str) -> LLMResponse:
        q = question.strip()
        note = re.match(NOTE_PATTERN, q, re.I | re.S)
        if note:
            content = note.group(4).strip()
            title = " ".join(content.split()[:6])
            name, args = "add_note", {"title": title, "content": content}
        elif re.search(r"\b(quais|liste|listar)\b.*\b(fontes|documentos|arquivos)\b", q, re.I):
            name, args = "list_sources", {}
        elif re.search(r"\b(hoje|data atual|que dia)\b", q, re.I):
            name, args = "get_current_date", {}
        else:
            name, args = "search_notes", {"query": q, "k": 3}
        return LLMResponse(
            tool_calls=[ToolCall(_new_id(), name, json.dumps(args, ensure_ascii=False))]
        )

    @staticmethod
    def _answer(question: str, result: dict) -> str:
        if "error" in result:
            return f"Não consegui usar a ferramenta: {result['error']}"
        if "results" in result:
            return FakeLLM._extractive_answer(question, result["results"])
        if "saved" in result:
            return f"Nota salva em {result['saved']} ({result['chunks']} chunk(s) indexado(s))."
        if "sources" in result:
            names = ", ".join(s["source"] for s in result["sources"]) or "nenhuma"
            return f"Tenho {len(result['sources'])} fonte(s) indexada(s): {names}."
        if "date" in result:
            return f"Hoje é {result['weekday']}, {result['date']}."
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _extractive_answer(question: str, results: list[dict], max_sentences: int = 3) -> str:
        if not results:
            return "Não encontrei nada sobre isso nas suas notas."
        q_terms = set(tokenize(question))
        candidates: list[tuple[int, int, str, str, set[str]]] = []
        seen: set[str] = set()
        for rank, r in enumerate(results):
            for pos, part in enumerate(re.split(r"\n+|(?<=[.!?])\s+", r["text"])):
                sentence = re.sub(r"^\s*(?:[-*#]+|\d+\.)\s*", "", part).replace("**", "").strip()
                matched = q_terms & set(tokenize(sentence))
                if matched and len(sentence) > 15 and sentence not in seen:
                    seen.add(sentence)
                    candidates.append((rank, pos, sentence, r["source"], matched))
        # Peso tipo IDF: termos que aparecem em muitas frases valem menos.
        df = Counter(t for *_, matched in candidates for t in matched)
        scored = sorted(
            (-sum(1 / df[t] for t in matched), rank, pos, sentence, source)
            for rank, pos, sentence, source, matched in candidates
        )
        if not scored:
            top = results[0]
            snippet = top["text"][:200]
            return f'Trecho mais próximo: "{snippet}…" [fonte: {top["source"]}]'
        lines = [f"- {s} [fonte: {src}]" for *_, s, src in scored[:max_sentences]]
        return "Pelas suas notas:\n" + "\n".join(lines)


def get_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider.lower()
    if provider == "fake":
        return FakeLLM()
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY não definida para LLM_PROVIDER=openai")
        return OpenAIChatLLM(
            settings.llm_model or "gpt-4o-mini", settings.openai_api_key, settings.openai_base_url
        )
    if provider == "ollama":
        return OpenAIChatLLM(
            settings.llm_model or "llama3.1",
            api_key="ollama",  # o Ollama ignora a chave, mas o SDK exige um valor
            base_url=settings.ollama_base_url.rstrip("/") + "/v1",
            name="ollama",
        )
    raise ValueError(f"LLM_PROVIDER desconhecido: {settings.llm_provider!r}")
