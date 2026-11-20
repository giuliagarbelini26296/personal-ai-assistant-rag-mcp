# Avaliação: retrieval

Gerado em 2026-11-20T20:23:38+00:00 · configuração: `embedder=hash-bow-1024`, `store=sqlite`, `k=4`

## Resumo

| métrica | valor |
|---|---|
| n | 21 |
| hit_at_1 | 0.8095 |
| hit_at_4 | 0.9524 |
| recall_at_4 | 0.9524 |
| top_score | 0.2602 |
| mrr | 0.873 |
| mean_top_score_unanswerable | 0.0694 |

## Por categoria

| categoria | n | hit_at_1 | hit_at_4 | recall_at_4 | reciprocal_rank | top_score |
|---|---|---|---|---|---|---|
| direta | 12 | 0.8333 | 1.0 | 1.0 | 0.9028 | 0.3139 |
| multi-fonte | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.2245 |
| parafrase | 8 | 0.75 | 0.875 | 0.875 | 0.8125 | 0.1842 |

## Itens para olhar

- **receita-02** (parafrase): Qual ingrediente deixa o salgadinho mineiro elástico? — recuperado: []
- **receita-04** (parafrase): O que muda se eu trocar o tipo de amido da massa por um mais suave? — recuperado: ['estudos/rag-anotacoes.md', 'receita-pao-de-queijo.txt']
- **reuniao-04** (direta): O que cada participante deve levar para o próximo encontro? — recuperado: ['viagem-chapada-diamantina.md', 'estudos/rag-anotacoes.md', 'reuniao-grupo-estudos.md']
- **leitura-01** (direta): Qual livro da minha lista eu já terminei? — recuperado: ['viagem-chapada-diamantina.md', 'lista-leitura.json']
