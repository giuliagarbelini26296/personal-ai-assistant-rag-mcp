"""Servidor MCP (stdio) expondo as ferramentas de notas para clientes como Claude Desktop/Code.

Usa o SDK oficial `mcp` 2.x, em que a antiga classe `FastMCP` passou a se chamar `MCPServer`.
"""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from .service import AssistantService


def build_mcp_server(service: AssistantService) -> MCPServer:
    server = MCPServer(
        name="personal-notes",
        instructions="Busca e adiciona notas pessoais de estudo. Cite a fonte de cada trecho.",
    )

    @server.tool()
    def search_notes(query: str, k: int = 4) -> dict:
        """Busca trechos relevantes nas notas pessoais. Retorna fonte, trecho e score."""
        args = service.tools.execute("search_notes", {"query": query, "k": k})
        if "error" in args:
            raise ValueError(args["error"])
        return args

    @server.tool()
    def add_note(title: str, content: str) -> dict:
        """Salva uma nova nota e a indexa para buscas futuras."""
        result = service.tools.execute("add_note", {"title": title, "content": content})
        if "error" in result:
            raise ValueError(result["error"])
        return result

    @server.tool()
    def list_sources() -> dict:
        """Lista os documentos indexados e quantos chunks cada um tem."""
        return {"sources": service.list_sources()}

    @server.resource("notes://sources", name="sources", mime_type="application/json")
    def sources_resource() -> str:
        """Lista de documentos indexados (JSON)."""
        return json.dumps(service.list_sources(), ensure_ascii=False)

    return server


def main() -> None:
    from .logging_utils import setup_logging

    service = AssistantService()
    setup_logging(service.settings.log_level)  # logs vão para stderr; stdout é do protocolo
    build_mcp_server(service).run("stdio")


if __name__ == "__main__":
    main()
