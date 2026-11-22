# Etapa 4 — O assistente na nuvem (Azure)

As etapas anteriores rodavam no computador. Esta coloca a API **em produção**: endereço HTTPS público, banco gerenciado, segredos num cofre, traces num monitor e deploy automático a cada push. Tudo descrito em código (`infra/main.bicep`), tudo dentro do plano de estudante.

## Arquitetura

```
GitHub (push na main)
   │ build ──► GitHub Container Registry (imagem pública)
   │ deploy ── OIDC, sem senha ──► Azure
   ▼
Azure Container Apps (API, escala a zero, só HTTPS)
   │ identidade gerenciada
   ├──► Key Vault: chave do Gemini, API key, URL do banco
   ├──► PostgreSQL Flexible (B1ms) + pgvector
   └──► Application Insights ◄── Log Analytics (teto de 1 GB/dia)
```

| Peça | Serviço | Por quê |
|---|---|---|
| API | Container Apps (consumo) | escala a zero: sem uso, sem custo; HTTPS e domínio prontos |
| Imagem | GitHub Container Registry | grátis para repositório público (o registro do Azure custa ~US$ 5/mês) |
| Banco | PostgreSQL Flexible B1ms | incluído nos 12 meses gratuitos; aceita pgvector |
| Segredos | Key Vault + identidade gerenciada | a app lê os segredos com a própria identidade; nenhum segredo fica no código ou no GitHub |
| Observabilidade | Application Insights | os mesmos spans OpenTelemetry da etapa 3, agora em um monitor gerenciado |
| Infraestrutura | Bicep | toda a infraestrutura versionada e reproduzível com um comando |
| Deploy | GitHub Actions + OIDC | o GitHub prova quem é com um token temporário; não existe senha do Azure guardada |

## Segurança antes de expor na internet

- **API key** (`X-API-Key`) nas rotas `/ingest`, `/search` e `/chat`, comparada com `secrets.compare_digest` (tempo constante). Sem isso, qualquer pessoa gastaria a cota do Gemini. O `/health` fica aberto para o Azure saber se a app está viva.
- **`/ingest` restrito a uma pasta** (`INGEST_ROOT`). Antes aceitava qualquer caminho: com acesso à API, alguém poderia mandar indexar `/etc`. O caminho é resolvido antes da verificação, então `..` não escapa.
- **Container sem root** (uid 10001) e **só HTTPS** (HTTP recebe 301).
- **Menor privilégio:**
  - a identidade da app só **lê** segredos (Key Vault Secrets User);
  - a identidade do deploy é **Contributor só no grupo de recursos**;
  - o OIDC só aceita tokens da branch `main` **deste** repositório.
- **Banco:** aceita conexões de serviços do Azure (o Container Apps de consumo não tem IP fixo), mas exige TLS e senha. A alternativa fechada (rede virtual e endpoint privado) custa mais. É uma escolha consciente para portfólio.
- **Custo:** a assinatura de estudante não tem cartão; se o crédito acabasse, os recursos parariam, sem cobrança. O Log Analytics tem teto de 1 GB por dia.

## Resultados (21/11/2026, em produção)

| Pergunta | Resposta | Latência | Custo |
|---|---|---|---|
| Qual o orçamento da viagem para a Chapada? | R$ 3.200 por pessoa [viagem-chapada-diamantina.md] | 1,5 s | US$ 0,0005 |
| Quem ficou responsável pelos testes? | Carla [reuniao-grupo-estudos.md] | 1,0 s | US$ 0,0004 |
| Qual é a senha do wi-fi da pousada? | não encontrou nas notas (não inventou) | 1,5 s | US$ 0,0005 |

- **Trace de uma pergunta no Application Insights** (consulta KQL nas tabelas `AppRequests` e `AppDependencies`):

```
POST /chat                 2.210 ms
└─ agent.run               1.479 ms
   ├─ llm.chat (passo 1)     572 ms   decide buscar
   ├─ tool.search_notes        9 ms
   │  └─ retrieval.search      9 ms   embedding local 0 ms, pgvector 3 ms
   └─ llm.chat (passo 2)     897 ms   responde
```

  O LLM é quase todo o tempo do agente, como na etapa 3. Há cerca de **730 ms no endpoint fora do agente** que os spans atuais não explicam (ver exercícios).
- **Deploy automático:** cerca de 3 min do push até a versão nova no ar, com teste de fumaça (o `/health` responde e o `/search` sem chave dá 401).
- **Infraestrutura completa:** a primeira criação levou 3 min (o Postgres já existia de uma tentativa anterior; do zero, cerca de 7 min).

## Problemas reais desta etapa

Quase nada funcionou de primeira. Todos estes problemas são de produção, e nenhum aparecia rodando local:

1. **Ambiente "Express".** O Azure passou a criar ambientes do Container Apps num modo novo, que **não aceita segredos do Key Vault** (`ExpressKeyVaultSecretsNotSupported`). A documentação (link do próprio erro) levava ao portal. A API resolveu: o campo `environmentMode` aparece na versão mais nova, e uma requisição com valor inválido listou os válidos (`WorkloadProfiles`, `ConsumptionOnly`, `Reserved`, `Express`, `Archived`).
2. **Formato novo do token OIDC do GitHub.** O `subject` passou a incluir IDs imutáveis (`repo:dono@ID/nome@ID:ref:...`), e o login falhou com `AADSTS700213`. A vantagem: um repositório apagado e recriado com o mesmo nome não herda a confiança.
3. **Pegadinha do zsh.** Ao corrigir o item 2, `"$REPO_ID:ref"` virou `...ef`, porque o zsh leu `:r` como modificador ("remover extensão"). Só foi pego porque conferi o valor gravado byte a byte. A correção é usar `${REPO_ID}`.
4. **Cota diária do Gemini.** O primeiro `/ingest` deu **500**. Os logs mostraram que era o 429 do Gemini: a cota de 1.000 embeddings do dia tinha acabado no projeto do Banco Central. Duas correções:
   - limite do provedor agora vira **503 com `Retry-After`**, e não "erro interno";
   - o embedder virou **parâmetro** do Bicep (`hash` ou `gemini`).
5. **Impasse ao trocar o embedder.** O assistente recusa abrir um banco criado com outro embedder; é a proteção contra misturar vetores incompatíveis. Na troca:
   - a versão nova não subia (o banco dizia "gemini");
   - a versão antiga continuava no ar e regravava "gemini" a cada reinício.

   A saída foi, nesta ordem: desativar a versão antiga; limpar as tabelas vazias, abrindo o banco **só para o meu IP** e fechando logo depois (com `trap`); e criar uma versão nova. Uma versão em `ActivationFailed` não volta com restart.
6. **Logs pagos à toa.** O SDK do Azure Monitor registrava cada envio de telemetria em INFO, várias linhas por requisição, e cada linha conta na cota do Log Analytics. O logger `azure` passou para WARNING.
7. **Falso positivo do gitleaks.** Ele acusou o ID público de um papel do Azure (`4633458b-...`, igual em todos os tenants). O caso foi registrado em `.gitleaksignore`. Antes disso, um push passou com o aviso, porque o `| tail` engoliu o código de saída. O comando foi corrigido para parar se houver achado.

## Como rodar

```bash
az login
az group create -n rg-assistant -l canadacentral
az deployment group create -g rg-assistant --template-file infra/main.bicep \
  --parameters containerImage=ghcr.io/<usuario>/personal-ai-assistant-rag-mcp:latest \
               geminiApiKey=... apiKey=... dbAdminPassword=... adminPrincipalId=$(az ad signed-in-user show --query id -o tsv)

# depois do primeiro deploy, os seguintes são automáticos (push na main)
K=$(az keyvault secret show --vault-name <kv> -n api-key --query value -o tsv)
curl -X POST https://<app>/ingest -H "X-API-Key: $K" -H 'Content-Type: application/json' -d '{"path":"/app/data/sample"}'
curl -X POST https://<app>/chat   -H "X-API-Key: $K" -H 'Content-Type: application/json' -d '{"message":"Qual o orçamento da viagem?"}'
```

## Exercícios
1. Troque o embedder de volta para `gemini` quando a cota voltar. Quais passos do item 5 você precisa repetir? Como automatizar isso?
2. Escreva uma consulta KQL no Log Analytics que mostre o p95 de `llm.chat` por hora.
3. Investigue os ~730 ms do `/chat` que ficam fora do `agent.run`. Crie spans para a memória de sessão e para a montagem da resposta e descubra onde está o tempo.
4. Desafio: troque a regra "serviços do Azure" do banco por uma rede virtual com endpoint privado. Quanto passaria a custar?

## Perguntas de entrevista
- **Como você guarda segredos na nuvem?** Num cofre (Key Vault), lidos pela identidade gerenciada da app; nada no código, na imagem ou no CI.
- **O que é OIDC no deploy?** O GitHub emite um token de curta duração dizendo "sou a main deste repositório", e o Azure confia nesse emissor. Não existe senha para vazar nem para trocar.
- **Por que infraestrutura como código?** É reproduzível, revisável em pull request e não depende de cliques no portal.
- **Como você controla custo?** Escala a zero, serviços dentro do plano gratuito, teto de logs, logs de bibliotecas em WARNING e uma assinatura sem cartão.
- **Como você diagnostica um 500 em produção?** Logs estruturados no Log Analytics e o `trace_id` para achar a requisição; aqui, o "erro interno" era o 429 do provedor de IA.
