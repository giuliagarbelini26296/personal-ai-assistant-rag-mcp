"""Executa as avaliações e monta os relatórios (JSON para máquina, Markdown para pessoas)."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..context import ConversationMemory
from ..service import AssistantService
from . import metrics as m
from .dataset import EvalItem
from .judge import Judge


@dataclass
class Report:
    kind: str
    config: dict
    summary: dict
    by_category: dict
    items: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def _avg(rows: list[dict], key: str) -> float | None:
    """Média só dos itens que têm a métrica. Itens com erro de execução ficam de fora:
    um erro de rede não é uma resposta ruim, e somar como zero esconderia o problema real."""
    values = [r[key] for r in rows if r.get(key) is not None and not r.get("error")]
    return m.mean(values) if values else None


def _group(items: list[dict], keys: list[str]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        groups[it["category"]].append(it)
    return {
        cat: {"n": len(rows), **{k: _avg(rows, k) for k in keys}}
        for cat, rows in sorted(groups.items())
    }


def evaluate_retrieval(svc: AssistantService, items: list[EvalItem], k: int = 4) -> Report:
    rows: list[dict] = []
    unanswerable_scores: list[float] = []
    for it in items:
        results = svc.search(it.question, k=k)
        sources = [r.source for r in results]
        if not it.answerable:
            # Sem resposta nas notas: guardamos o maior score para calibrar um limite de corte.
            unanswerable_scores.append(results[0].score if results else 0.0)
            continue
        rows.append(
            {
                "id": it.id,
                "category": it.category,
                "question": it.question,
                "expected": list(it.expected_sources),
                "retrieved": m.unique_sources(sources),
                "top_score": results[0].score if results else 0.0,
                "hit_at_1": m.hit_at_k(sources, it.expected_sources, 1),
                f"hit_at_{k}": m.hit_at_k(sources, it.expected_sources, k),
                f"recall_at_{k}": m.recall_at_k(sources, it.expected_sources, k),
                "reciprocal_rank": m.reciprocal_rank(sources, it.expected_sources),
            }
        )
    keys = ["hit_at_1", f"hit_at_{k}", f"recall_at_{k}", "reciprocal_rank", "top_score"]
    summary = {"n": len(rows), **{key: m.mean([r[key] for r in rows]) for key in keys}}
    summary["mrr"] = summary.pop("reciprocal_rank")
    summary["mean_top_score_unanswerable"] = m.mean(unanswerable_scores)
    return Report(
        kind="retrieval",
        config={"embedder": svc.embedder.name, "store": svc.settings.vector_store, "k": k},
        summary=summary,
        by_category=_group(rows, keys),
        items=rows,
    )


def evaluate_answers(
    svc: AssistantService,
    items: list[EvalItem],
    judge: Judge | None = None,
    pause: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Report:
    rows: list[dict] = []
    for n, it in enumerate(items):
        if n and pause:
            sleep(pause)  # respeita a cota gratuita da API
        started = time.perf_counter()
        try:
            result = svc.new_agent(memory=ConversationMemory()).run(it.question)
            answer, sources, contexts, error = result.answer, result.sources, result.contexts, None
            usage = {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                "trace_id": result.trace_id,
            }
        except Exception as exc:  # um item com erro não derruba os outros
            answer, sources, contexts, error = "", [], [], f"{type(exc).__name__}: {exc}"[:200]
            usage = {}
        row = {
            "id": it.id,
            "category": it.category,
            "question": it.question,
            "answer": answer,
            "sources": sources,
            "latency_s": round(time.perf_counter() - started, 2),
            "error": error,
            **usage,
        }
        if it.answerable:
            row["fact_recall"] = m.fact_recall(answer, it.facts)
            row["cited"] = 1.0 if m.cites_expected(sources, it.expected_sources) else 0.0
        else:
            row["abstained"] = 1.0 if m.abstained(answer) else 0.0
        if judge and not error:
            if n and pause:
                sleep(pause)
            v = judge.evaluate(it.question, contexts, it.reference, answer)
            row.update(
                faithfulness=v.faithfulness,
                correctness=v.correctness,
                judge_reason=v.reason,
                judge_error=v.error,
            )
        rows.append(row)
    keys = [
        "fact_recall",
        "cited",
        "abstained",
        "faithfulness",
        "correctness",
        "latency_s",
        "input_tokens",
        "output_tokens",
        "cost_usd",
    ]
    summary = {
        "n": len(rows),
        "errors": sum(1 for r in rows if r["error"]),
        "judge_errors": sum(1 for r in rows if r.get("judge_error")),
        **{k: _avg(rows, k) for k in keys},
    }
    return Report(
        kind="answers",
        config={
            "llm": getattr(svc.llm, "name", type(svc.llm).__name__),
            "judge": judge.name if judge else None,
            "embedder": svc.embedder.name,
            "store": svc.settings.vector_store,
        },
        summary=summary,
        by_category=_group(rows, keys),
        items=rows,
    )


def to_markdown(report: Report) -> str:
    lines = [
        f"# Avaliação: {report.kind}",
        "",
        f"Gerado em {report.created_at} · configuração: "
        + ", ".join(f"`{k}={v}`" for k, v in report.config.items()),
        "",
        "## Resumo",
        "",
    ]
    lines += ["| métrica | valor |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in report.summary.items()]
    if report.by_category:
        cols = list(next(iter(report.by_category.values())).keys())
        lines += [
            "",
            "## Por categoria",
            "",
            "| categoria | " + " | ".join(cols) + " |",
            "|---|" + "---|" * len(cols),
        ]
        lines += [
            f"| {cat} | " + " | ".join(str(v.get(c, "")) for c in cols) + " |"
            for cat, v in report.by_category.items()
        ]
    fails = [
        r
        for r in report.items
        if r.get("hit_at_1") == 0.0
        or r.get("correctness") in (1, 2)
        or r.get("abstained") == 0.0
        or r.get("error")
    ]
    if fails:
        lines += ["", "## Itens para olhar", ""]
        for r in fails:
            detail = (
                r.get("judge_reason") or r.get("error") or f"recuperado: {r.get('retrieved', '')}"
            )
            lines.append(f"- **{r['id']}** ({r['category']}): {r['question']} — {detail}")
    return "\n".join(lines) + "\n"


def save(report: Report, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{report.kind}-{report.config.get('embedder', '')}".replace(":", "-")
    base.with_suffix(".json").write_text(report.to_json(), encoding="utf-8")
    base.with_suffix(".md").write_text(to_markdown(report), encoding="utf-8")
    return base.with_suffix(".md")
