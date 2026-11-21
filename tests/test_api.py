import pytest
from fastapi.testclient import TestClient

from assistant.api import create_app


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks_indexed"] > 0


def test_search_endpoint(client):
    resp = client.post("/search", json={"query": "Morro do Pai Inácio pôr do sol", "k": 2})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["source"] == "viagem-chapada-diamantina.md"
    assert "x-request-id" in resp.headers


def test_search_validation_error(client):
    assert client.post("/search", json={"query": "a"}).status_code == 422
    assert client.post("/search", json={"query": "ok ok", "k": 50}).status_code == 422


def test_ingest_endpoint(client, tmp_path):
    (tmp_path / "nova.md").write_text("# Nova nota\nEstudar MCP na sexta.", encoding="utf-8")
    body = client.post("/ingest", json={"path": str(tmp_path)}).json()
    assert body["indexed"] == 1
    missing = str(tmp_path / "nao-existe")  # dentro da pasta permitida, mas não existe
    assert client.post("/ingest", json={"path": missing}).status_code == 404


def test_chat_keeps_session(client):
    first = client.post("/chat", json={"message": "Qual o orçamento da viagem para a Chapada?"})
    assert first.status_code == 200
    data = first.json()
    assert "3.200" in data["answer"] and data["sources"]
    second = client.post(
        "/chat", json={"message": "Quais fontes estão indexadas?", "session_id": data["session_id"]}
    )
    assert second.json()["session_id"] == data["session_id"]
    assert client.app.state.sessions[data["session_id"]].messages[0]["role"] == "user"
