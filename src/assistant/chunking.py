"""Divisão de texto em chunks por palavras, com sobreposição (overlap)."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[Chunk]:
    """Quebra `text` em janelas de `chunk_size` palavras, repetindo `overlap` palavras.

    Contamos palavras (e não caracteres) para não cortar palavras ao meio.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser > 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap deve estar entre 0 e chunk_size - 1")

    # Guardamos a posição de cada palavra para recortar o texto original e
    # preservar as quebras de linha (úteis para separar frases/itens depois).
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    if not spans:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    for start in range(0, len(spans), step):
        window = spans[start : start + chunk_size]
        chunks.append(Chunk(index=len(chunks), text=text[window[0][0] : window[-1][1]]))
        if start + chunk_size >= len(spans):
            break
    return chunks
