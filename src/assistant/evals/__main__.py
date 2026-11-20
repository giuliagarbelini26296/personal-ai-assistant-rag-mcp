"""CLI da avaliação.

    python -m assistant.evals retrieval [--k 4] [--min-hit-rate 0.8] [--min-mrr 0.7]
    python -m assistant.evals answers [--limit 8] [--judge] [--pause 4]

A configuração (embedder, banco, LLM) vem das mesmas variáveis de ambiente do assistente.
Com VECTOR_STORE=sqlite, a avaliação usa um banco temporário novo, para ser reproduzível.
Os limites `--min-*` fazem o comando sair com erro, o que reprova o CI (quality gate).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from ..config import DEFAULT_SAMPLE_DIR, Settings
from ..logging_utils import setup_logging
from ..service import AssistantService
from .dataset import DEFAULT_GOLDEN, load_golden
from .judge import Judge
from .runner import evaluate_answers, evaluate_retrieval, save


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m assistant.evals", description=__doc__.split("\n")[0]
    )
    p.add_argument("kind", choices=["retrieval", "answers"])
    p.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    p.add_argument("--out", default="eval_results")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--min-hit-rate", type=float, help="mínimo de hit@k (0 a 1)")
    p.add_argument("--min-mrr", type=float, help="mínimo de MRR (0 a 1)")
    p.add_argument("--limit", type=int, help="avaliar só os N primeiros itens (economiza cota)")
    p.add_argument("--only", help="ids separados por vírgula")
    p.add_argument("--judge", action="store_true", help="usar um LLM como juiz")
    p.add_argument("--judge-model", default="gemini-3.5-flash")
    p.add_argument("--pause", type=float, default=0.0, help="segundos entre chamadas ao LLM")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    setup_logging("WARNING")
    items = load_golden(args.golden)
    if args.only:
        wanted = set(args.only.split(","))
        items = [i for i in items if i.id in wanted]
    if args.limit:
        items = items[: args.limit]

    with tempfile.TemporaryDirectory() as tmp:
        if settings.vector_store == "sqlite":
            settings = replace(settings, db_path=str(Path(tmp) / "eval.sqlite3"))
        svc = AssistantService(settings)
        try:
            svc.ingest(DEFAULT_SAMPLE_DIR)
            if args.kind == "retrieval":
                report = evaluate_retrieval(svc, items, k=args.k)
            else:
                judge = None
                if args.judge:
                    from ..llm import GEMINI_OPENAI_URL, OpenAIChatLLM

                    jllm = OpenAIChatLLM(
                        args.judge_model, settings.gemini_api_key, GEMINI_OPENAI_URL, name="gemini"
                    )
                    judge = Judge(jllm.complete, jllm.name)
                report = evaluate_answers(svc, items, judge=judge, pause=args.pause)
        finally:
            svc.close()

    path = save(report, args.out)
    print(path.read_text(encoding="utf-8"))
    print(f"Relatório salvo em {path} (e .json)")

    failed = []
    hit_key = f"hit_at_{args.k}"
    if args.min_hit_rate is not None and report.summary.get(hit_key, 0) < args.min_hit_rate:
        failed.append(f"{hit_key}={report.summary[hit_key]} < {args.min_hit_rate}")
    if args.min_mrr is not None and report.summary.get("mrr", 0) < args.min_mrr:
        failed.append(f"mrr={report.summary['mrr']} < {args.min_mrr}")
    if failed:
        print("REPROVADO no limite de qualidade: " + "; ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
