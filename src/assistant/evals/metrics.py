"""Métricas de avaliação. Todas são funções puras: fáceis de testar e de explicar.

Busca (retrieval), olhando a lista de documentos devolvida, em ordem:
- hit@k: 1 se algum documento esperado aparece entre os k primeiros, senão 0.
- recall@k: fração dos documentos esperados que aparecem entre os k primeiros.
- reciprocal rank: 1/posição do primeiro documento certo (1º lugar = 1, 2º = 0,5...).
  A média disso é o MRR (Mean Reciprocal Rank), que premia acertar mais no topo.

Resposta (geração), sem precisar de outro LLM:
- fact recall: fração dos fatos esperados que aparecem na resposta.
- citação: a resposta usou alguma fonte esperada.
- abstenção: para perguntas sem resposta nas notas, o assistente admitiu não saber.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..embeddings import normalize


def unique_sources(sources: Sequence[str]) -> list[str]:
    """Vários chunks podem vir do mesmo documento; avaliamos por documento, na ordem."""
    out: list[str] = []
    for s in sources:
        if s not in out:
            out.append(s)
    return out


def _matches(source: str, expected: str) -> bool:
    return source == expected or source.endswith("/" + expected)


def first_relevant_rank(sources: Sequence[str], expected: Sequence[str]) -> int | None:
    for i, s in enumerate(unique_sources(sources), 1):
        if any(_matches(s, e) for e in expected):
            return i
    return None


def hit_at_k(sources: Sequence[str], expected: Sequence[str], k: int) -> float:
    rank = first_relevant_rank(sources, expected)
    return 1.0 if rank is not None and rank <= k else 0.0


def recall_at_k(sources: Sequence[str], expected: Sequence[str], k: int) -> float:
    top = unique_sources(sources)[:k]
    found = sum(1 for e in expected if any(_matches(s, e) for s in top))
    return found / len(expected) if expected else 0.0


def reciprocal_rank(sources: Sequence[str], expected: Sequence[str]) -> float:
    rank = first_relevant_rank(sources, expected)
    return 1.0 / rank if rank else 0.0


def fact_recall(answer: str, facts: Sequence[Sequence[str]]) -> float:
    if not facts:
        return 1.0
    text = normalize(answer)
    found = sum(1 for alternatives in facts if any(normalize(a) in text for a in alternatives))
    return found / len(facts)


def cites_expected(sources: Sequence[str], expected: Sequence[str]) -> bool:
    return any(_matches(s, e) for s in sources for e in expected)


ABSTENTION = re.compile(
    r"n[aã]o (encontrei|h[aá]|tenho|consta|sei|aparece|est[aá]|foi poss[ií]vel|localizei)"
    r"|nenhuma (informa|nota|men[cç])|sem informa",
    re.IGNORECASE,
)


def abstained(answer: str) -> bool:
    return bool(ABSTENTION.search(answer))


def mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
