import json

from assistant.agent import Agent
from assistant.llm import FakeLLM, LLMResponse, ToolCall


def test_tool_specs_follow_openai_format(service):
    specs = service.tools.specs()
    names = [s["function"]["name"] for s in specs]
    assert names == ["search_notes", "add_note", "list_sources", "get_current_date"]
    params = specs[0]["function"]["parameters"]
    assert params["required"] == ["query"] and params["properties"]["k"]["maximum"] == 10


def test_tool_validation_errors_are_returned_not_raised(service):
    assert "Argumentos inválidos" in service.tools.execute("search_notes", {"query": "x"})["error"]
    assert "JSON" in service.tools.execute("search_notes", "{nao json")["error"]
    assert "desconhecida" in service.tools.execute("apagar_tudo", {})["error"]


def test_current_date_tool(service):
    assert service.tools.execute("get_current_date", "{}") == {
        "date": "2026-11-19",
        "weekday": "quinta-feira",
    }


def test_agent_answers_with_citation(service):
    result = service.new_agent().run("Quando é a próxima reunião do grupo?")
    assert "17/11" in result.answer
    assert "[fonte: reuniao-grupo-estudos.md]" in result.answer
    assert result.sources[0] == "reuniao-grupo-estudos.md"
    assert result.steps == 2
    assert result.tool_calls[0]["name"] == "search_notes"


def test_agent_add_note_then_find_it(service):
    agent = service.new_agent()
    saved = agent.run("Anote: comprar polvilho azedo para a receita de sábado")
    assert "notas/" in saved.answer
    found = agent.run("O que preciso comprar para a receita de sábado?")
    assert any(s.startswith("notas/") for s in found.sources)


def test_agent_recovers_from_tool_error(service):
    """O modelo erra os argumentos, recebe o erro e tenta de novo."""
    script = [
        LLMResponse(tool_calls=[ToolCall("c1", "search_notes", json.dumps({"query": ""}))]),
        LLMResponse(tool_calls=[ToolCall("c2", "search_notes", json.dumps({"query": "forno"}))]),
        LLMResponse(content="Asse a 200 °C [fonte: receita-pao-de-queijo.txt]"),
    ]
    llm = FakeLLM(script=script)
    result = Agent(llm, service.tools, max_steps=4).run("temperatura do forno?")
    assert [c["ok"] for c in result.tool_calls] == [False, True]
    assert result.steps == 3
    tool_msgs = [m for m in llm.calls[1] if m["role"] == "tool"]
    assert "error" in json.loads(tool_msgs[0]["content"])


def test_agent_stops_at_max_steps(service):
    looping = [LLMResponse(tool_calls=[ToolCall(f"c{i}", "list_sources", "{}")]) for i in range(10)]
    result = Agent(FakeLLM(script=looping), service.tools, max_steps=3).run("oi")
    assert result.steps == 3 and "3 passos" in result.answer


def test_agent_appends_sources_when_model_forgets_to_cite(service):
    script = [
        LLMResponse(tool_calls=[ToolCall("c1", "search_notes", '{"query": "Chapada orçamento"}')]),
        LLMResponse(content="O orçamento é R$ 3.200 por pessoa."),
    ]
    result = Agent(FakeLLM(script=script), service.tools).run("orçamento?")
    assert "Fontes consultadas: viagem-chapada-diamantina.md" in result.answer


def test_agent_uses_conversation_memory(service):
    agent = service.new_agent()
    agent.memory.max_messages = 2
    agent.run("Quais fontes estão indexadas?")
    agent.run("Qual a temperatura do forno?")
    agent.run("Quando é a próxima reunião?")
    last_prompt = agent.llm.calls[-1]
    assert last_prompt[1]["role"] == "system"
    assert "Resumo da conversa anterior" in last_prompt[1]["content"]


def test_provider_extra_data_is_sent_back_with_tool_calls(service):
    """O Gemini 3 recusa a conversa se a thought_signature da chamada não voltar no histórico."""
    from assistant.llm import LLMResponse, ToolCall

    seen = []

    class SignatureLLM:
        name = "fake-signature"

        def __init__(self):
            self.turn = 0

        def chat(self, messages, tools):
            seen.append(messages)
            self.turn += 1
            if self.turn == 1:
                extra = {"google": {"thought_signature": "abc"}}
                return LLMResponse(
                    tool_calls=[ToolCall("c1", "search_notes", '{"query": "polvilho"}', extra)]
                )
            return LLMResponse(content="ok")

    from assistant.agent import Agent

    result = Agent(SignatureLLM(), service.tools).run("polvilho?")
    assistant_msg = next(m for m in seen[1] if m["role"] == "assistant")
    assert assistant_msg["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": "abc"}
    }
    assert result.contexts and result.contexts[0].startswith("[")
