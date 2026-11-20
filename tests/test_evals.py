"""Testes da avaliação: métricas, conjunto de referência, juiz e execução (tudo offline)."""

from __future__ import annotations

import json

import pytest

from assistant.evals import metrics as m
from assistant.evals.dataset import DEFAULT_GOLDEN, load_golden
from assistant.evals.judge import Judge, parse_verdict
from assistant.evals.runner import evaluate_answers, evaluate_retrieval, to_markdown

# ---- métricas de busca -------------------------------------------------------------------


def test_rank_metrics_use_unique_documents_in_order():
    retrieved = ["a.md", "a.md", "b.md", "c.md"]  # dois chunks do mesmo documento
    assert m.first_relevant_rank(retrieved, ["b.md"]) == 2
    assert m.hit_at_k(retrieved, ["b.md"], 1) == 0.0
    assert m.hit_at_k(retrieved, ["b.md"], 2) == 1.0
    assert m.reciprocal_rank(retrieved, ["c.md"]) == pytest.approx(1 / 3)
    assert m.reciprocal_rank(retrieved, ["z.md"]) == 0.0


def test_recall_counts_each_expected_document_and_matches_subfolders():
    retrieved = ["estudos/rag.md", "x.md"]
    assert m.recall_at_k(retrieved, ["rag.md", "lista.json"], 4) == 0.5
    assert m.recall_at_k(retrieved, ["estudos/rag.md"], 4) == 1.0
    assert m.recall_at_k(retrieved, ["rag.md"], 0) == 0.0


# ---- métricas de resposta ----------------------------------------------------------------


def test_fact_recall_accepts_alternatives_and_ignores_accents_and_case():
    facts = [["2 xícaras", "duas xícaras"], ["polvilho"]]
    assert m.fact_recall("Use DUAS XICARAS de Polvilho azedo", facts) == 1.0
    assert m.fact_recall("Use polvilho", facts) == 0.5
    assert m.fact_recall("qualquer coisa", []) == 1.0


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("Não encontrei essa informação nas suas notas.", True),
        ("Nas notas não há menção à senha.", True),
        ("Nenhuma nota fala disso.", True),
        ("A senha é 1234.", False),
    ],
)
def test_abstention_detection(answer, expected):
    assert m.abstained(answer) is expected


def test_mean_rounds_and_handles_empty():
    assert m.mean([1, 0, 0]) == 0.3333
    assert m.mean([]) == 0.0


# ---- conjunto de referência --------------------------------------------------------------


def test_golden_set_is_valid_and_points_to_real_documents():
    from assistant.config import DEFAULT_SAMPLE_DIR

    items = load_golden()
    assert len(items) >= 20
    assert {i.category for i in items} >= {"direta", "parafrase", "sem-resposta"}
    docs = {p.relative_to(DEFAULT_SAMPLE_DIR).as_posix() for p in DEFAULT_SAMPLE_DIR.rglob("*.*")}
    for item in items:
        assert set(item.expected_sources) <= docs, item.id


def test_golden_loader_rejects_inconsistent_items(tmp_path):
    bad = tmp_path / "g.jsonl"
    line = {
        "id": "x",
        "category": "sem-resposta",
        "question": "q",
        "expected_sources": ["a"],
        "facts": [],
        "reference": "r",
    }
    bad.write_text(json.dumps(line))
    with pytest.raises(ValueError, match="sem-resposta"):
        load_golden(bad)
    line.update(category="outra")
    bad.write_text(json.dumps(line))
    with pytest.raises(ValueError, match="categoria"):
        load_golden(bad)
    assert DEFAULT_GOLDEN.exists()


# ---- juiz ----------------------------------------------------------------------------------


def test_verdict_parsing_accepts_json_inside_text_and_rejects_bad_scores():
    v = parse_verdict('Aqui está:\n{"fidelidade": 5, "corretude": 3, "justificativa": "ok"}')
    assert (v.faithfulness, v.correctness, v.reason, v.error) == (5, 3, "ok", None)
    assert parse_verdict("sem json").error
    assert parse_verdict('{"fidelidade": 9, "corretude": 3}').error
    assert parse_verdict('{"fidelidade": "x"}').error


def test_judge_failures_become_errors_not_zero_scores():
    def boom(prompt):
        raise TimeoutError

    v = Judge(boom, "fake").evaluate("q", [], "r", "a")
    assert v.faithfulness is None and "TimeoutError" in v.error


def test_judge_prompt_contains_question_contexts_reference_and_answer():
    seen = {}

    def complete(prompt):
        seen["p"] = prompt
        return '{"fidelidade": 4, "corretude": 5, "justificativa": "x"}'

    Judge(complete, "fake").evaluate("PERG", ["[a.md] TRECHO"], "REF", "RESP")
    assert all(s in seen["p"] for s in ["PERG", "[a.md] TRECHO", "REF", "RESP"])


# ---- execução ponta a ponta (hashing + FakeLLM, sem rede) ----------------------------------


def test_retrieval_report_on_golden_set(service):
    report = evaluate_retrieval(service, load_golden(), k=4)
    assert report.summary["n"] == sum(1 for i in load_golden() if i.answerable)
    assert 0.7 <= report.summary["hit_at_4"] <= 1.0
    assert "parafrase" in report.by_category
    assert "## Resumo" in to_markdown(report)


def test_answers_report_with_fake_judge_and_pause(service):
    items = [i for i in load_golden() if i.id in {"reuniao-01", "fora-01"}]
    calls, waits = [], []
    judge = Judge(lambda p: calls.append(p) or '{"fidelidade": 5, "corretude": 4}', "fake")
    report = evaluate_answers(service, items, judge=judge, pause=2, sleep=waits.append)
    assert report.summary["n"] == 2 and report.summary["errors"] == 0
    assert report.summary["faithfulness"] == 5.0 and len(calls) == 2
    assert waits == [2, 2]  # pausa antes do 2º item e antes do 2º julgamento
    assert report.items[0]["fact_recall"] is not None and "abstained" in report.items[1]
