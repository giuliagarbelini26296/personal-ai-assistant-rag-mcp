"""Ferramentas expostas ao LLM (function calling), com validação de argumentos via Pydantic."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from .service import AssistantService

logger = logging.getLogger(__name__)
WEEKDAYS_PT = [
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
]


class SearchNotesArgs(BaseModel):
    query: str = Field(min_length=2, max_length=500, description="Pergunta ou termos de busca")
    k: int = Field(default=4, ge=1, le=10, description="Quantidade de trechos a retornar")


class AddNoteArgs(BaseModel):
    title: str = Field(min_length=1, max_length=120, description="Título curto da nota")
    content: str = Field(min_length=1, max_length=5000, description="Conteúdo da nota")


class NoArgs(BaseModel):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], dict]

    def spec(self) -> dict:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": schema},
        }


class ToolRegistry:
    def __init__(self, service: AssistantService, clock: Callable[[], datetime] = datetime.now):
        self._service = service
        self._clock = clock
        self._tools = {
            t.name: t
            for t in [
                Tool(
                    "search_notes",
                    "Busca trechos relevantes nas notas e documentos pessoais do usuário.",
                    SearchNotesArgs,
                    self._search_notes,
                ),
                Tool(
                    "add_note",
                    "Salva uma nova nota pessoal e a indexa para buscas futuras.",
                    AddNoteArgs,
                    self._add_note,
                ),
                Tool("list_sources", "Lista os documentos indexados.", NoArgs, self._list_sources),
                Tool("get_current_date", "Retorna a data de hoje.", NoArgs, self._current_date),
            ]
        }

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def execute(self, name: str, arguments: str | dict | None) -> dict:
        """Executa uma ferramenta. Nunca levanta exceção: erros voltam como {"error": ...}
        para que o modelo possa corrigir os argumentos na próxima rodada."""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Ferramenta desconhecida: {name}. Disponíveis: {self.names}"}
        try:
            raw = json.loads(arguments) if isinstance(arguments, str) and arguments else arguments
            args = tool.args_model.model_validate(raw or {})
        except json.JSONDecodeError as exc:
            return {"error": f"Argumentos não são JSON válido: {exc.msg}"}
        except ValidationError as exc:
            details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return {"error": f"Argumentos inválidos para {name}: {details}"}
        try:
            return tool.handler(args)
        except Exception as exc:  # erro inesperado vira mensagem para o modelo
            logger.exception("falha na ferramenta", extra={"tool": name})
            return {"error": f"Falha ao executar {name}: {exc}"}

    # --- handlers -------------------------------------------------------------------------
    def _search_notes(self, args: SearchNotesArgs) -> dict:
        results = self._service.search(args.query, args.k)
        return {"results": [r.to_dict() for r in results]}

    def _add_note(self, args: AddNoteArgs) -> dict:
        return self._service.add_note(args.title, args.content)

    def _list_sources(self, _: NoArgs) -> dict:
        return {"sources": self._service.list_sources()}

    def _current_date(self, _: NoArgs) -> dict:
        now = self._clock()
        return {"date": now.date().isoformat(), "weekday": WEEKDAYS_PT[now.weekday()]}
