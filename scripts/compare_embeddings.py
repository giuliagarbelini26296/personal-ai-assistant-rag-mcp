"""Compara a busca com embeddings por hashing (palavras) e com o Gemini (significado).

Uso (precisa de GEMINI_API_KEY no ambiente; o banco é temporário e apagado no fim):
    python scripts/compare_embeddings.py

As perguntas foram escritas SEM repetir as palavras dos documentos, para mostrar a diferença
entre busca por palavras e busca semântica.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace

from assistant.config import DEFAULT_SAMPLE_DIR, Settings
from assistant.llm import FakeLLM
from assistant.service import AssistantService

# (pergunta, documento que deveria aparecer em primeiro)
QUESTIONS = [
    ("qual ingrediente deixa o salgadinho mineiro elástico?", "receita-pao-de-queijo.txt"),
    ("quando o grupo vai se encontrar de novo?", "reuniao-grupo-estudos.md"),
    ("como rodar várias tarefas ao mesmo tempo em Python sem threads?", "estudos/python-async.md"),
    ("qual trilha com queda d'água está no roteiro das férias?", "viagem-chapada-diamantina.md"),
    ("como o modelo encontra trechos relevantes antes de responder?", "estudos/rag-anotacoes.md"),
    ("quais livros ainda preciso terminar?", "lista-leitura.json"),
]


def run(provider: str, db: str) -> list[tuple[str, float, bool]]:
    settings = replace(Settings(), embedding_provider=provider, db_path=db, log_level="WARNING")
    svc = AssistantService(settings, llm=FakeLLM())
    try:
        svc.ingest(DEFAULT_SAMPLE_DIR)
        out = []
        for question, expected in QUESTIONS:
            top = svc.search(question, k=1)
            source = top[0].source if top else "(nada)"
            out.append((source, top[0].score if top else 0.0, source.endswith(expected)))
        return out
    finally:
        svc.close()


def mark(result: tuple[str, float, bool]) -> str:
    source, score, ok = result
    return ("OK  " if ok else "ERRO") + f" {source.split('/')[-1][:18]:18} {score:.2f}"


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("Defina GEMINI_API_KEY para rodar a comparação.")
    with tempfile.TemporaryDirectory() as tmp:
        hashed = run("hash", f"{tmp}/hash.sqlite3")
        gemini = run("gemini", f"{tmp}/gemini.sqlite3")
    print(f"{'pergunta':62} | {'hashing':28} | gemini")
    for (q, _), h, g in zip(QUESTIONS, hashed, gemini, strict=True):
        print(f"{q[:62]:62} | {mark(h):28} | {mark(g)}")
    print(
        f"\nacertos no 1º lugar: hashing {sum(h[2] for h in hashed)}/{len(QUESTIONS)}"
        f" · gemini {sum(g[2] for g in gemini)}/{len(QUESTIONS)}"
    )


if __name__ == "__main__":
    main()
