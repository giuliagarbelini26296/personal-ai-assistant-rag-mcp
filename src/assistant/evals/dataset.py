"""Conjunto de avaliação ("golden set"): perguntas com a resposta esperada conhecida.

Cada linha de `data/eval/golden.jsonl` tem:
- `question`: a pergunta, como o usuário faria;
- `category`: direta | parafrase | multi-fonte | sem-resposta;
- `expected_sources`: documentos que precisam aparecer na busca (vazio = a resposta não existe);
- `facts`: fatos que a resposta precisa conter. Cada fato é uma lista de formas aceitas
  (ex.: ["2 xícaras", "duas xícaras"]);
- `reference`: a resposta ideal, usada pelo juiz.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import PROJECT_ROOT

DEFAULT_GOLDEN = PROJECT_ROOT / "data" / "eval" / "golden.jsonl"
CATEGORIES = {"direta", "parafrase", "multi-fonte", "sem-resposta"}


@dataclass(frozen=True)
class EvalItem:
    id: str
    category: str
    question: str
    expected_sources: tuple[str, ...]
    facts: tuple[tuple[str, ...], ...]
    reference: str

    @property
    def answerable(self) -> bool:
        return bool(self.expected_sources)


def load_golden(path: str | Path = DEFAULT_GOLDEN) -> list[EvalItem]:
    items: list[EvalItem] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw["id"] in seen:
            raise ValueError(f"linha {n}: id repetido {raw['id']!r}")
        if raw["category"] not in CATEGORIES:
            raise ValueError(f"linha {n}: categoria desconhecida {raw['category']!r}")
        if (raw["category"] == "sem-resposta") == bool(raw["expected_sources"]):
            raise ValueError(f"linha {n}: só 'sem-resposta' pode ficar sem fontes esperadas")
        seen.add(raw["id"])
        items.append(
            EvalItem(
                id=raw["id"],
                category=raw["category"],
                question=raw["question"],
                expected_sources=tuple(raw["expected_sources"]),
                facts=tuple(tuple(f) for f in raw["facts"]),
                reference=raw["reference"],
            )
        )
    return items
