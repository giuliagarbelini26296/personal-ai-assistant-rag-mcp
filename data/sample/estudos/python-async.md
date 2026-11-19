# Anotações: async em Python

- `async def` cria uma corrotina; ela só executa quando é aguardada com `await`.
- `asyncio.run(main())` é o ponto de entrada mais simples para rodar código assíncrono.
- `asyncio.gather` executa várias corrotinas de forma concorrente e junta os resultados.
- Async ajuda em tarefas de I/O (rede, disco), não em tarefas pesadas de CPU.
- FastAPI aceita rotas `async def` e rotas normais; as normais rodam em um thread pool.

Exercício da semana: escrever um script que baixa 5 páginas em paralelo com httpx.AsyncClient.
