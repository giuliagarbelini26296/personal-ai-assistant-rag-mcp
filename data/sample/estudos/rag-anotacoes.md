# Anotações de estudo: RAG (Retrieval-Augmented Generation)

## O que é
RAG é uma técnica em que, antes de pedir uma resposta ao LLM, o sistema busca trechos
relevantes em uma base de documentos e os coloca no prompt como contexto. Assim o modelo
responde com base em informação atualizada e verificável, e não só no que aprendeu no treino.

## Etapas principais
1. Ingestão: carregar documentos (Markdown, PDF, TXT) e limpar o texto.
2. Chunking: dividir o texto em pedaços menores. Usar overlap (sobreposição) entre chunks
   ajuda a não cortar uma ideia no meio.
3. Embeddings: transformar cada chunk em um vetor numérico.
4. Vector store: guardar os vetores e permitir busca por similaridade de cosseno.
5. Retrieval: dado uma pergunta, buscar os k chunks mais parecidos.
6. Geração: montar o prompt com os chunks e pedir a resposta citando as fontes.

## Dúvidas para revisar
- Qual o tamanho ideal de chunk? Depende do tipo de documento; começar com 150 a 300 palavras.
- Busca híbrida combina similaridade vetorial com busca por palavra-chave (tipo BM25).
- Reranking reordena os resultados com um modelo mais caro, só para os top-k.
