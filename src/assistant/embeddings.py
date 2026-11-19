"""Provedores de embeddings atrás de uma interface simples.

- `HashingEmbedder`: padrão, determinístico, 100% offline (bag-of-words com hashing trick).
- `OpenAIEmbedder`: usa a API da OpenAI (ou endpoint compatível).
- `OllamaEmbedder`: usa um servidor Ollama local (`/api/embed`).
"""

from __future__ import annotations

import hashlib
import math
import re
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


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.name = f"ollama-{model}"
        self._url = base_url.rstrip("/") + "/api/embed"

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(self._url, json={"model": self.model, "input": texts}, timeout=60)
        response.raise_for_status()
        return response.json()["embeddings"]


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
    raise ValueError(f"EMBEDDING_PROVIDER desconhecido: {settings.embedding_provider!r}")
