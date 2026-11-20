# Etapa 1 — Embeddings semânticos e banco vetorial (pgvector)

Guia de estudo da primeira evolução do projeto. A ideia é você conseguir explicar cada decisão numa entrevista.

## O problema que esta etapa resolve

A versão anterior usava **embeddings por hashing**: cada palavra vira uma posição num vetor. Isso só encontra documentos que **repetem as palavras** da pergunta.

Pergunta: *"qual ingrediente deixa o salgadinho mineiro elástico?"*
O documento certo é a receita de pão de queijo, mas ela não tem as palavras "salgadinho", "mineiro" nem "elástico". Por isso o hashing não acha nada.

Um **embedding semântico** (aqui, `gemini-embedding-001`) foi treinado para colocar textos com **o mesmo significado** perto um do outro, mesmo com palavras diferentes.

Resultado real em 20/11/2026 (`python scripts/compare_embeddings.py`):

| Busca | Acertos no 1º lugar (6 perguntas) | Faixa de score dos acertos |
|---|---|---|
| Hashing (palavras) | 4/6 | 0,13 a 0,30 |
| Gemini (significado) | **6/6** | 0,45 a 0,72 |

## Conceitos

### Embedding
Uma lista de números (vetor) que representa o significado de um texto. Textos parecidos têm vetores que apontam para direções parecidas.

### Similaridade de cosseno
Mede o ângulo entre dois vetores: **1** = mesma direção, **0** = sem relação, **−1** = opostos. Não depende do tamanho do vetor, só da direção.

### Por que normalizar
O Gemini gera 3.072 números, mas pedimos 768 (`outputDimensionality`) para economizar espaço. Vetores **truncados não vêm com comprimento 1** — no teste, a norma veio 0,59. Por isso `GeminiEmbedder` normaliza tudo (`l2_normalize`). Com vetores de comprimento 1, produto escalar e cosseno dão o mesmo resultado.

### Tipo de tarefa (`taskType`)
O modelo recebe uma dica do uso do texto:
- `RETRIEVAL_DOCUMENT` para o que vai para o banco;
- `RETRIEVAL_QUERY` para a pergunta.

Isso aproxima "perguntas" das "respostas", e não só de textos parecidos com a pergunta. Por isso o código tem `embed()` (documentos) e `embed_query()` (pergunta).

### Banco vetorial e pgvector
`pgvector` é uma extensão do PostgreSQL que adiciona:
- o tipo `vector(768)`;
- o operador `<=>` (distância de cosseno);
- índices para busca de vizinhos próximos.

Vantagem sobre um banco vetorial separado: os vetores ficam **no mesmo banco dos dados da aplicação**, com transações, backup e SQL normais.

### Índice HNSW e busca aproximada (ANN)
Sem índice, o banco compara a pergunta com **todas** as linhas (busca exata). Com milhões de linhas isso fica lento.

O HNSW monta um grafo de "vizinhos" e caminha por ele até os mais próximos. É **aproximado** (pode perder um vizinho raramente), mas muito mais rápido. Parâmetros importantes, se perguntarem: `m` e `ef_construction` (qualidade do grafo) e `hnsw.ef_search` (quantos candidatos olhar na busca).

Com 8 trechos o índice não faz diferença; ele está aqui para mostrar como seria em produção.

### Busca híbrida
O `Retriever` continua combinando **70% vetor + 30% palavras-chave**. Isso ajuda com nomes próprios, códigos e siglas, que embeddings às vezes tratam mal.

## O que mudou no código

| Arquivo | Mudança |
|---|---|
| `src/assistant/embeddings.py` | `GeminiEmbedder`: lotes de até 100 textos, `taskType`, normalização e nova tentativa com espera crescente em erro 429/5xx |
| `src/assistant/store.py` | Protocolo `Store` e `open_store()`: o resto do código não sabe qual banco está em uso |
| `src/assistant/store_pg.py` | `PgVectorStore`: tabela com `vector(n)`, índice HNSW, busca com `<=>` |
| `docker-compose.yml` | PostgreSQL com pgvector na porta 55433 (só no seu computador) |
| `tests/test_gemini_embedder.py` | Testa o embedder **sem rede**, com uma API falsa (`httpx.MockTransport`) |
| `tests/test_pgvector_store.py` | Testa o banco de verdade; roda no CI com um serviço pgvector |
| `scripts/compare_embeddings.py` | A comparação da tabela acima |

## Como rodar

```bash
docker compose -p rag-mcp up -d pgvector
export EMBEDDING_PROVIDER=gemini VECTOR_STORE=pgvector GEMINI_API_KEY=sua-chave
export DATABASE_URL=postgresql://rag:rag-local-only@127.0.0.1:55433/rag
python -m assistant ingest data/sample
python -m assistant search "qual ingrediente deixa o salgadinho mineiro elástico?"
```

Rodar os testes, inclusive os do pgvector (eles usam o banco separado `rag_test`, porque apagam as tabelas):

```bash
TEST_DATABASE_URL=postgresql://rag:rag-local-only@127.0.0.1:55433/rag_test pytest -q
```

Ver os vetores direto no banco:

```bash
docker exec -it rag-mcp-pgvector-1 psql -U rag -d rag
SELECT source, chunk_index, vector_dims(embedding) FROM rag_chunks;
```

## Exercícios (para fazer sozinha)

1. Rode `search` com uma pergunta sua que não repita palavras das notas. O resultado faz sentido?
2. Mude `EMBEDDING_DIM` para 256 e reindexe (precisa apagar as tabelas `rag_*`). Os scores mudam? Por quê?
3. Na busca da receita, o 2º lugar (viagem, 0,38) ficou perto do 1º (0,45). O que isso diz sobre usar um `min_score` fixo? *(Isso leva à Etapa 2: medir a qualidade.)*

## Perguntas de entrevista

- **Por que não usar só busca por palavras?** Porque usuários perguntam com palavras diferentes das do documento.
- **Por que pgvector e não um banco vetorial dedicado?** Menos infraestrutura, transações e SQL. Bancos dedicados ganham em escala muito grande e filtros avançados.
- **O que acontece se trocar o modelo de embedding?** Vetores de modelos diferentes não são comparáveis: é preciso reindexar tudo. O código recusa misturar (`embedder` gravado em `rag_meta`).
- **Como lidar com limite de requisições da API?** Lotes, nova tentativa com espera crescente só em erros temporários (429/5xx) e nunca em erros do cliente (400).
- **O que é HNSW?** Índice de busca aproximada de vizinhos; troca uma perda mínima de precisão por muita velocidade.
