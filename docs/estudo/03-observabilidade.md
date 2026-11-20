# Etapa 3 — Observabilidade (traces, tokens, custo e latência)

A etapa 2 mede a qualidade **antes** de publicar. Esta etapa mostra o que acontece **em uso**: cada pergunta deixa um rastro com o caminho que percorreu, quanto tempo cada parte levou, quantos tokens gastou e quanto custaria.

É o trabalho do dia a dia de **operações de IA**: responder "por que essa resposta saiu errada?", "por que ficou lento?" e "quanto vai custar com mil usuários?".

![Trace de uma pergunta no Phoenix](../img/phoenix-trace.jpg)

## Conceitos

### Trace e span
- **Span:** uma etapa com início, fim e atributos. Ex.: `llm.chat`, 964 ms, 1.408 tokens.
- **Trace:** a árvore de spans de **uma** pergunta, ligada por um `trace_id`.

A árvore de uma pergunta neste projeto:

```
agent.run                      1,9 s   pergunta, resposta, tokens e custo totais
├── llm.chat (passo 1)         0,7 s   o modelo decide chamar search_notes
├── tool.search_notes          0,3 s   argumentos, nº de resultados, melhor score
│   └── retrieval.search       0,3 s   documentos recuperados, com score e trecho
│       ├── embedding.query    0,28 s  vetor da pergunta (API do Gemini)
│       └── vector_store.search  4 ms  busca no pgvector
└── llm.chat (passo 2)         1,0 s   resposta final
```

### Por que OpenTelemetry
OpenTelemetry é o **padrão aberto** de observabilidade. O código cria os spans uma vez, e o destino é escolhido por configuração:

| `OTEL_EXPORTER` | Para onde vai |
|---|---|
| `none` (padrão) | lugar nenhum; os spans são descartados quase sem custo |
| `console` | terminal |
| `jsonl` | arquivo `traces/spans.jsonl`, lido por `scripts/trace_report.py` |
| `otlp` | protocolo padrão: Phoenix, Langfuse, Jaeger, Grafana Tempo, Datadog... |
| `otlp,jsonl` | os dois ao mesmo tempo |

Trocar de Phoenix para Langfuse ou Datadog é só mudar o endereço, sem alterar o código.

### Convenções de nomes
Os atributos seguem **duas** convenções, porque cada ferramenta lê uma:
- `gen_ai.*` (oficial do OpenTelemetry): `gen_ai.request.model`, `gen_ai.usage.input_tokens`;
- **OpenInference** (lida pelo Phoenix): `openinference.span.kind` (AGENT, LLM, TOOL, RETRIEVER, EMBEDDING), `llm.token_count.prompt`, `input.value`, `output.value`.

Na primeira versão gravei só `gen_ai.*`, e o Phoenix mostrou todos os spans como "unknown" e sem tokens. **Convenção importa.**

### Tokens e custo
Cada resposta do Gemini informa os tokens de entrada e de saída. O agente soma as chamadas de uma pergunta e calcula o custo com o preço público (Standard, consultado em 20/11/2026):

| Modelo | Entrada (US$/1M) | Saída (US$/1M) |
|---|---|---|
| `gemini-3.5-flash-lite` | 0,30 | 2,50 |

No plano gratuito o custo real é zero. O número mostra **quanto custaria em produção**. A saída inclui os tokens de "raciocínio" (thinking), gravados em `gen_ai.usage.reasoning_tokens`.

### Ligando log e trace
Todo log JSON emitido durante uma pergunta ganha o campo `trace_id`. A API `/chat` também devolve `trace_id`, `latency_ms`, tokens e `cost_usd`. Com o `trace_id` de uma reclamação, você acha o trace exato no Phoenix.

## Resultados reais (20/11/2026, 8 perguntas, Gemini + pgvector)

```
perguntas: 8 · latência p50 1961 ms · p95 2207 ms
custo estimado: total US$ 0,006273 · médio US$ 0,000784 por pergunta
tempo por etapa: llm.chat 84% · embedding.query 16% · vector_store.search 0%
```

Leituras que valem para uma entrevista:
1. **O LLM é 84% do tempo.** Otimizar o banco vetorial não muda nada para o usuário (4 ms). Para ficar mais rápido: menos passos do agente, respostas mais curtas ou streaming.
2. **Toda pergunta faz 2 chamadas ao LLM** (decidir buscar + responder). Um fluxo fixo "sempre busca, depois responde" economizaria uma chamada; o agente troca isso por flexibilidade.
3. **A entrada domina os tokens** (~1.700 de entrada contra ~70 a 260 de saída). Prompt do sistema, definição das ferramentas e trechos recuperados vão em toda chamada. Cortar trechos irrelevantes reduz o custo direto.
4. **Custo: cerca de US$ 0,78 por mil perguntas** com este modelo. A resposta mais longa (async, 264 tokens de saída) custou 60% a mais que a média, por causa do preço maior da saída.
5. **O embedding da pergunta leva ~280 ms** (uma chamada de rede ao Gemini). Um cache de perguntas repetidas eliminaria esse tempo.

## Dois problemas reais desta etapa

1. **Traces perdidos em silêncio.** O Phoenix 21 vem com login ligado e recusou os spans com **401**. O assistente continuou funcionando, sem nenhum erro visível, e os traces simplesmente não apareceram. Lição: telemetria também falha; confira se os dados chegam (aqui, com a API do Phoenix). No ambiente local, o login foi desligado, e o Phoenix só escuta em `127.0.0.1`.
2. **Convenção errada:** spans sem `openinference.span.kind` apareceram como "unknown". Resolvido gravando as duas convenções.

## O que mudou no código

| Arquivo | Mudança |
|---|---|
| `src/assistant/observability.py` | configuração do OpenTelemetry, exportadores, preço e custo, `trace_id` atual |
| `src/assistant/agent.py` | spans `agent.run`, `llm.chat` (por passo) e `tool.*`; soma de tokens e custo |
| `src/assistant/retrieval.py` | spans `retrieval.search`, `embedding.query` e `vector_store.search`, com os documentos recuperados |
| `src/assistant/llm.py` | `Usage`: tokens de entrada, saída e raciocínio de cada resposta |
| `src/assistant/logging_utils.py` | `trace_id` nos logs JSON |
| `src/assistant/api.py` | `/chat` devolve `trace_id`, latência, tokens e custo |
| `src/assistant/evals/runner.py` | a avaliação registra tokens, custo e `trace_id` por pergunta |
| `scripts/trace_report.py` | p50/p95, custo e tempo por etapa a partir do JSONL |
| `tests/test_observability.py` | 8 testes com um exportador em memória (sem rede) |

## Como rodar

```bash
docker compose -p rag-mcp up -d pgvector phoenix
export EMBEDDING_PROVIDER=gemini LLM_PROVIDER=gemini VECTOR_STORE=pgvector GEMINI_API_KEY=sua-chave
export DATABASE_URL=postgresql://rag:rag-local-only@127.0.0.1:55433/rag
export OTEL_EXPORTER=otlp,jsonl
python -m assistant ask "Qual ingrediente deixa o salgadinho mineiro elástico?"
python scripts/trace_report.py
```

Abra **http://127.0.0.1:6006**, entre no projeto `personal-ai-assistant` e clique na pergunta.

## Exercícios

1. No Phoenix, abra o trace do "salgadinho mineiro" e veja no span `retrieval.search` quais documentos o modelo recebeu. Isso confirma o que a etapa 2 disse (a busca acertou e a geração falhou)?
2. Rode a mesma pergunta 3 vezes e compare a latência. Qual etapa varia mais?
3. Calcule quanto custariam 10 mil perguntas por dia durante um mês com os números do relatório. E se cada resposta tivesse o dobro de tokens de saída?
4. Desafio: crie um alerta simples no `trace_report.py` que avise quando o p95 passar de 3 s ou o custo médio passar de US$ 0,001.

## Perguntas de entrevista

- **O que você monitora numa aplicação com LLM?** Latência (p50/p95, por etapa), tokens e custo por requisição, taxa de erro e de chamadas de ferramenta com erro, qualidade (avaliações e feedback) e os documentos recuperados.
- **Trace, log e métrica: qual a diferença?** Métrica é um número agregado (p95 = 2,2 s); log é um evento; trace é o caminho de uma requisição. O `trace_id` nos logs liga os três.
- **Por que OpenTelemetry e não o SDK de uma ferramenta?** Padrão aberto, sem ficar preso a um fornecedor; trocar de ferramenta é configuração.
- **Como reduzir custo de um RAG?** Menos tokens de entrada (menos trechos e prompt menor), modelo menor para tarefas simples, cache de embeddings e de respostas, menos chamadas por pergunta.
- **Como saber se a telemetria está funcionando?** Verificar se os dados chegam no destino. Neste projeto, os spans foram recusados (401) sem nenhum erro visível na aplicação.
