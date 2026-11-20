# Etapa 2 — Avaliação do RAG (evals)

Na etapa 1, a comparação de 6 perguntas mostrou que o Gemini era melhor. Mas 6 perguntas escolhidas a dedo não provam nada. Esta etapa cria uma forma **repetível** de medir a qualidade, para responder com números: *"essa mudança melhorou ou piorou?"*

## Ideia central

Um sistema de RAG pode falhar em dois lugares diferentes:

1. **Busca (retrieval):** não trouxe o documento certo.
2. **Geração:** trouxe o documento certo, mas o modelo respondeu errado, inventou ou deixou de responder.

Por isso medimos as duas partes **separadamente**. Se medir só a resposta final, você não sabe onde consertar.

## O conjunto de referência (golden set)

`data/eval/golden.jsonl` tem 24 perguntas escritas à mão, cada uma com:

- documento(s) onde está a resposta (`expected_sources`);
- fatos que a resposta precisa ter (`facts`), com formas alternativas aceitas;
- a resposta ideal (`reference`), usada pelo juiz.

Categorias, porque cada uma testa uma coisa:

| Categoria | Testa | Exemplo |
|---|---|---|
| direta (12) | o básico | "Quem ficou responsável pelos testes?" |
| parafrase (8) | entender sem as mesmas palavras | "Qual ingrediente deixa o salgadinho mineiro elástico?" |
| multi-fonte (1) | juntar dois documentos | cosseno nos estudos **e** na lista de leitura |
| sem-resposta (3) | admitir que não sabe | "Qual é a senha do wi-fi da pousada?" |

As perguntas **sem resposta** são as mais importantes para confiança: um assistente que inventa é pior do que um que diz "não sei".

## Métricas de busca

| Métrica | Pergunta que responde | Como calcula |
|---|---|---|
| hit@1 | o documento certo veio em 1º? | 1 ou 0 por pergunta, depois a média |
| hit@k | veio entre os k primeiros? | idem, com k = 4 |
| recall@k | quantos dos documentos certos vieram? | importa na multi-fonte |
| MRR | quão alto ele aparece? | média de 1/posição (1º = 1; 2º = 0,5; 3º = 0,33) |

Resultados reais (20/11/2026, 21 perguntas com resposta):

| Embedder | hit@1 | hit@4 | MRR | Paráfrases hit@1 |
|---|---|---|---|---|
| Hashing (palavras) | 0,81 | 0,95 | 0,87 | 0,75 |
| Gemini (significado) | **0,95** | **1,00** | **0,98** | **1,00** |

### O achado sobre perguntas sem resposta
Nas 3 perguntas sem resposta, o **melhor trecho** encontrado teve nota média:
- **0,07** com hashing;
- **0,46** com Gemini.

Com embeddings semânticos, **sempre** aparece algo "parecido". Um limite fixo de nota não separa o relevante do irrelevante. Quem precisa admitir "não está nas notas" é o **modelo**, e isso se mede na avaliação das respostas.

## Métricas de resposta

Sem outro LLM (baratas e determinísticas):
- **fact recall:** fração dos fatos esperados presentes na resposta (ignora maiúsculas e acentos);
- **citação:** usou alguma fonte esperada;
- **abstenção:** nas perguntas sem resposta, admitiu não saber.

Com **LLM como juiz** (`--judge`):
- **fidelidade (1–5):** tudo o que a resposta afirma está nos trechos? Mede **alucinação**.
- **corretude (1–5):** responde à pergunta, comparando com a referência?

### Por que um juiz, se já tem fact recall?
Fact recall procura texto. Ele não percebe:
- resposta certa escrita de outro jeito ("dois copos" em vez de "2 xícaras");
- resposta que tem o fato certo **e** inventa outra coisa junto.

### Cuidados com o juiz (boas perguntas de entrevista)
- **Critérios escritos e escala curta** (1 a 5, com o que significa cada ponta).
- **Saída em JSON validada:** se o juiz responder fora do formato, o item vira **erro**, não nota zero.
- **Modelo diferente do avaliado:** `gemini-3.5-flash` julga `gemini-3.5-flash-lite`, para reduzir o viés de autopreferência.
- **Temperatura 0**, para notas estáveis.
- O juiz também erra: as notas baixas devem ser **lidas por uma pessoa** (a seção "Itens para olhar" do relatório existe para isso).

### Resultados das respostas (20/11/2026)

`gemini-3.5-flash-lite` respondendo, `gemini-3.5-flash` como juiz, busca com Gemini. Relatório completo em [docs/eval/2026-11-20-respostas-gemini.md](../eval/2026-11-20-respostas-gemini.md).

| Categoria | n | Fatos presentes | Citou a fonte | Admitiu não saber | Fidelidade | Corretude |
|---|---|---|---|---|---|---|
| direta | 12 | 1,00 | 1,00 | — | 5,0 | 5,0 |
| paráfrase | 8 | 0,88 | 1,00 | — | 5,0 | **4,5** |
| multi-fonte | 1 | 1,00 | 1,00 | — | 5,0 | 5,0 |
| sem resposta | 3 | — | — | **1,00** | 5,0 | 5,0 |
| **total** | 24 | 0,95 | 1,00 | 1,00 | **5,0** | **4,83** |

Latência média: 1,9 s por pergunta. Nenhuma alucinação detectada (fidelidade 5 em todas).

**A única falha (`receita-02`) é de geração, não de busca.** A busca trouxe a receita em 1º lugar, mas o modelo respondeu *"não há menção a um salgadinho mineiro"*. Foi literal demais: não ligou "salgadinho mineiro" a "pão de queijo". O prompt diz *"responda somente com base nos trechos"* e *"nunca invente"*, e o modelo exagerou na cautela. Esse é o equilíbrio clássico entre **não alucinar** e **ser útil**.

**Um item ficou sem nota do juiz** (`fora-03`), por limite de cota (`RateLimitError`). Ele aparece como erro do juiz, não como nota 0, e por isso a média de corretude é sobre 23 itens.

## Portão de qualidade no CI

O CI roda a avaliação da busca a cada push, com o embedder por hashing (sem chave, sem custo, sempre igual):

```bash
python -m assistant.evals retrieval --min-hit-rate 0.9 --min-mrr 0.8
```

Se uma mudança no chunking ou no ranking derrubar a qualidade, **o CI reprova**. Testado: com um limite impossível (0,99), o comando sai com erro.

A avaliação das respostas **não** roda no CI: custa chamadas de API, depende de cota e não é determinística. Ela roda sob demanda, antes de mudanças de prompt ou de modelo.

## Dois problemas reais que a avaliação encontrou

1. **Modelo descontinuado:** o `gemini-2.5-flash-lite` devolveu 404 ("no longer available to new users"). Trocado por `gemini-3.5-flash-lite`. Lição: fixe a versão do modelo e tenha um teste que avise quando ela sair do ar.
2. **Thought signature do Gemini 3:** a primeira chamada funcionava, mas a segunda (depois de usar a ferramenta) dava **400**. O Gemini 3 manda uma `thought_signature` em cada chamada de ferramenta e exige que ela volte no histórico. O agente passou a guardar e reenviar (`ToolCall.extra`). Há um teste para isso.

## Como rodar

```bash
# busca, com hashing (igual ao CI)
python -m assistant.evals retrieval
# busca, com Gemini
EMBEDDING_PROVIDER=gemini python -m assistant.evals retrieval
# respostas, com juiz (gasta cota: ~3 chamadas por pergunta)
EMBEDDING_PROVIDER=gemini LLM_PROVIDER=gemini python -m assistant.evals answers --judge --pause 3
# só algumas perguntas
python -m assistant.evals answers --only receita-02,fora-01 --judge
```

Os relatórios ficam em `eval_results/` (`.md` para ler, `.json` para comparar versões).

## Exercícios

1. Escreva 3 perguntas novas no `golden.jsonl` (uma de cada categoria) e rode de novo. Alguma métrica mudou?
2. Mude `CHUNK_SIZE` para 40 e rode a avaliação da busca com hashing. O portão de qualidade passa?
3. **Corrija o `receita-02` sem quebrar o resto.** Ajuste o `SYSTEM_PROMPT` em `src/assistant/agent.py` para permitir ligar sinônimos e descrições ao que está nos trechos (ex.: "salgadinho mineiro" = pão de queijo), mantendo a regra de não inventar. Rode primeiro só os itens sensíveis:
   `python -m assistant.evals answers --judge --only receita-02,fora-01,fora-02,fora-03 --pause 3`
   Depois rode tudo e compare com o relatório de 20/11. Se as perguntas **sem resposta** começarem a ganhar respostas inventadas, a mudança piorou o sistema, mesmo que o `receita-02` tenha melhorado.

## Perguntas de entrevista

- **Como você sabe que o seu RAG funciona?** Conjunto de referência com categorias, métricas de busca e de resposta separadas, juiz para alucinação e portão no CI.
- **O que é MRR?** Média de 1/posição do primeiro resultado certo; premia acertar no topo.
- **Como medir alucinação?** Fidelidade: comparar cada afirmação da resposta com os trechos recuperados (LLM como juiz ou verificação por afirmação).
- **Por que não rodar a avaliação com LLM no CI?** Custo, cota, latência e resultados não determinísticos. No CI fica a parte determinística.
- **Quais os riscos do LLM como juiz?** Viés de posição e de autopreferência, sensibilidade ao prompt e notas instáveis. Mitigação: critérios claros, modelo diferente, temperatura 0 e revisão humana das notas baixas.
