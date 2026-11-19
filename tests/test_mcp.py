"""Testes do servidor MCP: in-process (rápido) e via stdio (subprocesso real)."""

import asyncio
import json
import os
import sys

from mcp import Client, StdioServerParameters

from assistant.config import PROJECT_ROOT
from assistant.mcp_server import build_mcp_server


def _payload(result) -> dict:
    assert not result.is_error, result
    return json.loads(result.content[0].text)


def test_mcp_tools_in_process(service):
    async def scenario():
        async with Client(build_mcp_server(service)) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert names == {"search_notes", "add_note", "list_sources"}

            found = _payload(await client.call_tool("search_notes", {"query": "Poço Azul", "k": 2}))
            assert found["results"][0]["source"] == "viagem-chapada-diamantina.md"

            saved = _payload(
                await client.call_tool("add_note", {"title": "MCP", "content": "Testar no Claude"})
            )
            assert saved["saved"] == "notas/mcp.md"

            bad = await client.call_tool("search_notes", {"query": "x"})
            assert bad.is_error

            resource = await client.read_resource("notes://sources")
            sources = json.loads(resource.contents[0].text)
            assert "notas/mcp.md" in {s["source"] for s in sources}

    asyncio.run(scenario())


def test_mcp_over_stdio(service, settings):
    """Sobe `python -m assistant serve-mcp` como subprocesso e fala o protocolo via stdio."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "assistant", "serve-mcp"],
        env={**os.environ, "ASSISTANT_DB_PATH": settings.db_path, "LOG_LEVEL": "WARNING"},
        cwd=str(PROJECT_ROOT),
    )

    async def scenario():
        async with Client(params) as client:
            listed = _payload(await client.call_tool("list_sources", {}))
            assert len(listed["sources"]) == 6

    asyncio.run(scenario())
