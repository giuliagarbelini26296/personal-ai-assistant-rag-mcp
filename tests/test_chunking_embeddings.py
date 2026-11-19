import pytest

from assistant.chunking import chunk_text
from assistant.embeddings import HashingEmbedder, normalize, tokenize


def test_chunk_text_overlap():
    text = " ".join(f"p{i}" for i in range(25))
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    # janelas começam nas palavras 0, 7, 14 e 21 (passo = 10 - 3)
    assert [c.index for c in chunks] == [0, 1, 2, 3]
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-3:] == second[:3]  # as últimas 3 palavras se repetem
    assert chunks[-1].text.split()[-1] == "p24"  # nada se perde no final


def test_chunk_text_preserves_newlines_and_handles_empty():
    assert chunk_text("") == []
    assert chunk_text("linha um\nlinha dois", chunk_size=10, overlap=2)[0].text == (
        "linha um\nlinha dois"
    )


@pytest.mark.parametrize("size,overlap", [(0, 0), (5, 5), (5, -1)])
def test_chunk_text_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=size, overlap=overlap)


def test_tokenize_removes_accents_and_stopwords():
    assert normalize("Reunião") == "reuniao"
    assert tokenize("A reunião do grupo é às 19h") == ["reuniao", "grupo", "19h"]


def test_hashing_embedder_is_deterministic_and_normalized():
    emb = HashingEmbedder(dim=256)
    v1, v2 = emb.embed(["pão de queijo", "pão de queijo"])
    assert v1 == v2
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-9
    assert emb.embed([""])[0] == [0.0] * 256
