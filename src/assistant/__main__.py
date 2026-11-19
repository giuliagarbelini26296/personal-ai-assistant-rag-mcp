"""CLI: python -m assistant {ingest,ask,search,serve-api,serve-mcp}."""

from __future__ import annotations

import argparse
import json
import sys

from .config import DEFAULT_SAMPLE_DIR, Settings
from .logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="assistant", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Indexa uma pasta/arquivo (padrão: data/sample)")
    p.add_argument("path", nargs="?", default=str(DEFAULT_SAMPLE_DIR))

    p = sub.add_parser("ask", help="Pergunta ao agente (usa LLM_PROVIDER, padrão: fake)")
    p.add_argument("question")
    p.add_argument("--json", action="store_true", help="Saída em JSON com fontes e tool calls")

    p = sub.add_parser("search", help="Busca direta no vector store (sem LLM)")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=4)

    p = sub.add_parser("serve-api", help="Sobe a API REST com uvicorn")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8101)

    sub.add_parser("serve-mcp", help="Sobe o servidor MCP via stdio")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()

    if args.command == "serve-mcp":
        from .mcp_server import main as mcp_main

        mcp_main()
        return 0

    if args.command == "serve-api":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
        return 0

    setup_logging("WARNING" if settings.log_level == "INFO" else settings.log_level)
    from .service import AssistantService

    service = AssistantService(settings)
    try:
        if args.command == "ingest":
            stats = service.ingest(args.path)
            print(
                f"Arquivos: {stats.files_seen} | indexados: {stats.indexed} | "
                f"sem mudança: {stats.unchanged} | chunks novos: {stats.chunks} | "
                f"total de chunks no banco: {service.store.count_chunks()}"
            )
        elif args.command == "search":
            for r in service.search(args.query, args.k):
                preview = " ".join(r.text.split())[:100]
                print(f"[{r.score:.3f}] {r.source}#{r.chunk_index}: {preview}…")
        elif args.command == "ask":
            result = service.new_agent().run(args.question)
            if args.json:
                print(json.dumps(vars(result), ensure_ascii=False, indent=2))
            else:
                print(result.answer)
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
