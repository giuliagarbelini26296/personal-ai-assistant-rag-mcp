# Avaliação: answers

Gerado em 2026-11-20T20:24:58+00:00 · configuração: `llm=gemini:gemini-3.5-flash-lite`, `judge=gemini:gemini-3.5-flash`, `embedder=gemini-gemini-embedding-001-768`, `store=sqlite`

## Resumo

| métrica | valor |
|---|---|
| n | 24 |
| errors | 0 |
| judge_errors | 1 |
| fact_recall | 0.9524 |
| cited | 1.0 |
| abstained | 1.0 |
| faithfulness | 5.0 |
| correctness | 4.8261 |
| latency_s | 1.915 |

## Por categoria

| categoria | n | fact_recall | cited | abstained | faithfulness | correctness | latency_s |
|---|---|---|---|---|---|---|---|
| direta | 12 | 1.0 | 1.0 | None | 5.0 | 5.0 | 1.8058 |
| multi-fonte | 1 | 1.0 | 1.0 | None | 5.0 | 5.0 | 2.24 |
| parafrase | 8 | 0.875 | 1.0 | None | 5.0 | 4.5 | 2.1125 |
| sem-resposta | 3 | None | None | 1.0 | 5.0 | 5.0 | 1.7167 |

## Itens para olhar

- **receita-02** (parafrase): Qual ingrediente deixa o salgadinho mineiro elástico? — O assistente foi fiel aos trechos ao não inventar dados, mas falhou em responder à pergunta ao dizer que não encontrou a informação, enquanto a resposta de referência identificou o polvilho.
