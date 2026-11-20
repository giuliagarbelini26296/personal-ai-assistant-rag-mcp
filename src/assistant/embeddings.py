"""Provedores de embeddings atrás de uma interface simples.

- `HashingEmbedder`: padrão, determinístico, 100% offline (bag-of-words com hashing trick).
- `OpenAIEmbedder`: usa a API da OpenAI (ou endpoint compatível).
- `OllamaEmbedder`: usa um servidor Ollama local (`/api/embed`).
- `GeminiEmbedder`: modelo semântico do Google (`gemini-embedding-001`), com tipo de tarefa
  diferente para documentos e perguntas.

Todo embedder tem `embed(textos)` para documentos e `embed_query(texto)` para perguntas.
Alguns modelos geram vetores melhores quando sabem se o texto é um documento ou uma pergunta.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import unicodedata
from collections import Counter
from typing import Protocol

import httpx

from .config import Settings

# Lista curta de stopwords em português (suficiente para o exemplo).
STOPWORDS = frozenset(
    """a o as os um uma uns umas de do da dos das em no na nos nas por para com sem e ou
    que se ao aos à às é ser foi são como mais menos muito já não sim isso esse essa este esta
    eu você ele ela nós eles elas me te lhe seu sua meu minha qual quais quando onde porque
    sobre entre até também só the of and to in is""".split()
)


def normalize(text: str) -> str:
    """Minúsculas e sem acentos (para 'polvilho' == 'Polvilho' e 'reunião' == 'reuniao')."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", normalize(text)) if t not in STOPWORDS and len(t) > 1]


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def l2_normalize(vec: list[float]) -> list[float]:
    """Deixa o vetor com comprimento 1, para que o produto escalar seja igual ao cosseno."""
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class HashingEmbedder:
    """Vetores esparsos de unigramas + bigramas projetados em `dim` posições via hash.

    Não é semântico como um modelo neural, mas é rápido, reprodutível e não precisa de rede —
    ideal para testes e para a demo offline.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hash-bow-{dim}"

    def _index(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        sign = 1.0 if value & 1 else -1.0
        return (value >> 1) % self.dim, sign

    def embed_one(self, text: str) -> list[float]:
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        vec = [0.0] * self.dim
        for feature, count in Counter(features).items():
            idx, sign = self._index(feature)
            vec[idx] += sign * (1.0 + math.log(count))  # tf sublinear
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_one(text)


class OpenAIEmbedder:
    def __init__(self, model: str, api_key: str, base_url: str = "") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depende do extra instalado
            raise RuntimeError("Instale o extra: pip install -e '.[openai]'") from exc
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY não definida para EMBEDDING_PROVIDER=openai")
        self.model = model
        self.name = f"openai-{model}"
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.name = f"ollama-{model}"
        self._url = base_url.rstrip("/") + "/api/embed"

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(self._url, json={"model": self.model, "input": texts}, timeout=60)
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class GeminiEmbedder:
    """Embeddings semânticos pela API do Gemini (`batchEmbedContents`).

    - `task_type`: RETRIEVAL_DOCUMENT para o que vai para o banco e RETRIEVAL_QUERY para a
      pergunta. O modelo foi treinado para aproximar perguntas das respostas certas.
    - `dim`: o modelo gera 3072 dimensões; pedimos menos (padrão 768) para economizar espaço.
      Vetores truncados não vêm normalizados, então normalizamos aqui.
    - Erros 429/5xx são temporários: tentamos de novo com espera crescente.
    """

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
    MAX_BATCH = 100  # limite de itens por requisição da API

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dim: int = 768,
        client: httpx.Client | None = None,
        max_retries: int = 4,
        sleep=time.sleep,
    ) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY não definida para EMBEDDING_PROVIDER=gemini")
        self.model, self.dim = model, dim
        self.name = f"gemini-{model}-{dim}"
        self._key = api_key
        self._client = client or httpx.Client(timeout=60)
        self._max_retries = max_retries
        self._sleep = sleep

    def _call(self, texts: list[str], task_type: str) -> list[list[float]]:
        body = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                    "outputDimensionality": self.dim,
                }
                for t in texts
            ]
        }
        for attempt in range(self._max_retries + 1):
            response = self._client.post(
                self.URL.format(model=self.model),
                json=body,
                headers={"x-goog-api-key": self._key},
            )
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self._max_retries:
                    response.raise_for_status()
                self._sleep(min(2**attempt, 30))
                continue
            response.raise_for_status()
            return [l2_normalize(e["values"]) for e in response.json()["embeddings"]]
        raise RuntimeError("inalcançável")  # pragma: no cover

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.MAX_BATCH):
            vectors.extend(self._call(texts[i : i + self.MAX_BATCH], "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._call([text], "RETRIEVAL_QUERY")[0]


def get_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider.lower()
    if provider == "hash":
        return HashingEmbedder()
    if provider == "openai":
        return OpenAIEmbedder(
            settings.embedding_model or "text-embedding-3-small",
            settings.openai_api_key,
            settings.openai_base_url,
        )
    if provider == "ollama":
        return OllamaEmbedder(
            settings.embedding_model or "nomic-embed-text", settings.ollama_base_url
        )
    if provider == "gemini":
        return GeminiEmbedder(
            settings.gemini_api_key,
            settings.embedding_model or "gemini-embedding-001",
            settings.embedding_dim,
        )
    raise ValueError(f"EMBEDDING_PROVIDER desconhecido: {settings.embedding_provider!r}")
