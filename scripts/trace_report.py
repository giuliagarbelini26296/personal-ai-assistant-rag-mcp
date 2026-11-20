"""Resumo dos traces gravados em JSONL (OTEL_EXPORTER=jsonl ou otlp,jsonl).

    python scripts/trace_report.py [traces/spans.jsonl]

Mostra, por pergunta: latência, passos, tokens e custo estimado. No fim, p50/p95 da latência
e a fatia de tempo de cada etapa (LLM, busca, embedding, banco). É o tipo de número que um time
de operações de IA acompanha: onde está o tempo, quanto custa e o que piorou.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[idx]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def report(spans: list[dict]) -> str:
    roots = [s for s in spans if s["name"] == "agent.run"]
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        by_trace[s["trace_id"]].append(s)
    lines = [f"{'pergunta':52} {'ms':>7} {'passos':>6} {'tok in':>7} {'tok out':>7} {'US$':>10}"]
    stage_ms: dict[str, float] = defaultdict(float)
    for r in sorted(roots, key=lambda x: x["start_ns"]):
        a = r["attributes"]
        cost = a.get("llm.cost_usd")
        lines.append(
            f"{str(a.get('agent.question', ''))[:52]:52} {r['duration_ms']:7.0f} "
            f"{a.get('agent.steps', 0):6} {a.get('gen_ai.usage.input_tokens', 0):7} "
            f"{a.get('gen_ai.usage.output_tokens', 0):7} "
            f"{(f'{cost:.6f}' if cost is not None else '—'):>10}"
        )
        for s in by_trace[r["trace_id"]]:
            if s["name"] in {"llm.chat", "embedding.query", "vector_store.search"}:
                stage_ms[s["name"]] += s["duration_ms"]
    latencies = [r["duration_ms"] for r in roots]
    total_ms = sum(latencies) or 1.0
    costs = [r["attributes"].get("llm.cost_usd") for r in roots]
    known = [c for c in costs if c is not None]
    lines += [
        "",
        f"perguntas: {len(roots)} · latência p50 {percentile(latencies, 50):.0f} ms"
        f" · p95 {percentile(latencies, 95):.0f} ms",
        f"custo estimado: total US$ {sum(known):.6f}"
        + (f" · médio US$ {sum(known) / len(known):.6f} por pergunta" if known else ""),
        "tempo por etapa: "
        + " · ".join(f"{k} {v / total_ms:.0%}" for k, v in sorted(stage_ms.items())),
    ]
    return "\n".join(lines)


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "traces/spans.jsonl")
    if not path.exists():
        raise SystemExit(f"Arquivo não encontrado: {path}. Rode com OTEL_EXPORTER=jsonl.")
    print(report(load(path)))


if __name__ == "__main__":
    main()
