from assistant.context import ConversationMemory, fit_results_to_budget


def test_fit_results_to_budget_keeps_order_and_truncates():
    results = [{"source": f"{i}.md", "text": "x" * 400} for i in range(5)]  # ~100 tokens cada
    fitted = fit_results_to_budget(results, budget=300)
    assert [r["source"] for r in fitted] == ["0.md", "1.md", "2.md"]
    assert fitted[-1]["text"].endswith("…")  # o terceiro foi truncado


def test_memory_window_and_rolling_summary():
    mem = ConversationMemory(max_messages=2)
    mem.add("user", "pergunta 1")
    mem.add("assistant", "resposta 1")
    mem.add("user", "pergunta 2")
    msgs = mem.as_messages()
    assert msgs[0]["role"] == "system" and "Usuário: pergunta 1" in msgs[0]["content"]
    assert [m["content"] for m in msgs[1:]] == ["resposta 1", "pergunta 2"]


def test_summary_respects_token_budget():
    mem = ConversationMemory(max_messages=1, max_summary_tokens=30)
    for i in range(20):
        mem.add("user", f"mensagem número {i} " + "texto " * 10)
    assert len(mem.summary) / 4 <= 30 + 50  # mantém apenas as linhas mais recentes
    assert "mensagem número 18" in mem.summary
