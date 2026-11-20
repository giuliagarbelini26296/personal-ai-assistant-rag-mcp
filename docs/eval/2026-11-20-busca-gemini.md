# Avaliação: retrieval

Gerado em 2026-11-20T20:08:46+00:00 · configuração: `embedder=gemini-gemini-embedding-001-768`, `store=sqlite`, `k=4`

## Resumo

| métrica | valor |
|---|---|
| n | 21 |
| hit_at_1 | 0.9524 |
| hit_at_4 | 1.0 |
| recall_at_4 | 1.0 |
| top_score | 0.6506 |
| mrr | 0.9762 |
| mean_top_score_unanswerable | 0.4608 |

## Por categoria

| categoria | n | hit_at_1 | hit_at_4 | recall_at_4 | reciprocal_rank | top_score |
|---|---|---|---|---|---|---|
| direta | 12 | 0.9167 | 1.0 | 1.0 | 0.9583 | 0.6944 |
| multi-fonte | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.6014 |
| parafrase | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5911 |

## Itens para olhar

- **reuniao-04** (direta): O que cada participante deve levar para o próximo encontro? — recuperado: ['viagem-chapada-diamantina.md', 'reuniao-grupo-estudos.md', 'estudos/rag-anotacoes.md']
