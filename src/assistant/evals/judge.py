"""LLM como juiz ("LLM-as-a-judge").

Métricas por texto (fact recall) não percebem uma resposta certa escrita de outro jeito, nem
uma resposta que inventa algo a mais. Um segundo modelo lê a pergunta, os trechos usados e a
resposta, e dá notas com critérios fixos.

Boas práticas usadas aqui:
- Critérios explícitos, com escala curta (1 a 5) e o que cada ponta significa.
- Saída em JSON, validada; se vier inválida, o item fica marcado como erro (não vira nota 0).
- Juiz diferente do modelo avaliado (por padrão, gemini-3.5-flash julga gemini-3.5-flash-lite),
  para reduzir o viés de um modelo preferir as próprias respostas.
- Temperatura 0, para notas mais estáveis.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

JUDGE_PROMPT = """Você avalia respostas de um assistente que responde perguntas sobre as notas
pessoais de uma pessoa.

Dê duas notas de 1 a 5:

FIDELIDADE: tudo o que a resposta afirma está nos TRECHOS?
5 = tudo vem dos trechos; 3 = algo não está nos trechos, mas é secundário;
1 = inventa informação importante.
Se a resposta só diz que não encontrou a informação, a fidelidade é 5.

CORRETUDE: a resposta atende à pergunta, comparando com a RESPOSTA DE REFERÊNCIA?
5 = correta e completa; 3 = parcialmente correta ou incompleta; 1 = errada ou não responde.
Se a referência diz que as notas não têm a informação, 5 = o assistente admitiu não saber;
1 = inventou uma resposta.

PERGUNTA:
{question}

TRECHOS QUE O ASSISTENTE RECEBEU:
{contexts}

RESPOSTA DE REFERÊNCIA:
{reference}

RESPOSTA DO ASSISTENTE:
{answer}

Responda só com JSON, sem texto antes ou depois:
{{"fidelidade": <1-5>, "corretude": <1-5>, "justificativa": "<uma frase>"}}"""


@dataclass(frozen=True)
class Verdict:
    faithfulness: int | None
    correctness: int | None
    reason: str
    error: str | None = None


def parse_verdict(raw: str) -> Verdict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return Verdict(None, None, "", error="resposta do juiz sem JSON")
    try:
        data = json.loads(match.group(0))
        f, c = int(data["fidelidade"]), int(data["corretude"])
    except (ValueError, KeyError, TypeError) as exc:
        return Verdict(None, None, "", error=f"JSON inválido do juiz: {exc}")
    if not (1 <= f <= 5 and 1 <= c <= 5):
        return Verdict(None, None, "", error="nota fora da escala 1-5")
    return Verdict(f, c, str(data.get("justificativa", ""))[:300])


class Judge:
    def __init__(self, complete: Callable[[str], str], name: str) -> None:
        self._complete = complete
        self.name = name

    def evaluate(self, question: str, contexts: list[str], reference: str, answer: str) -> Verdict:
        prompt = JUDGE_PROMPT.format(
            question=question,
            contexts="\n".join(contexts) or "(nenhum trecho)",
            reference=reference,
            answer=answer,
        )
        try:
            return parse_verdict(self._complete(prompt))
        except Exception as exc:  # erro de rede/cota não pode derrubar a avaliação inteira
            return Verdict(None, None, "", error=f"falha ao chamar o juiz: {type(exc).__name__}")
