"""Proteções para expor a API na internet: API key e /ingest restrito a uma pasta."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from assistant.api import create_app
from assistant.service import AssistantService


@pytest.fixture
def secured(settings):
    return TestClient(create_app(AssistantService(replace(settings, api_key="chave-de-teste"))))


def test_health_stays_open(secured):
    assert secured.get("/health").status_code == 200  # o Azure usa para saber se a app está viva


@pytest.mark.parametrize(
    "path,body",
    [
        ("/search", {"query": "viagem"}),
        ("/chat", {"message": "oi"}),
        ("/ingest", {}),
    ],
)
def test_protected_routes_need_the_key(secured, path, body):
    assert secured.post(path, json=body).status_code == 401
    assert secured.post(path, json=body, headers={"X-API-Key": "errada"}).status_code == 401


def test_right_key_is_accepted(secured):
    resp = secured.post(
        "/search", json={"query": "viagem"}, headers={"X-API-Key": "chave-de-teste"}
    )
    assert resp.status_code == 200


def test_ingest_outside_allowed_folder_is_forbidden(service, tmp_path):
    client = TestClient(create_app(service))
    assert client.post("/ingest", json={"path": "/etc"}).status_code == 403
    # ".." não escapa da pasta: o caminho é resolvido antes da verificação
    escape = str(tmp_path / ".." / "..")
    assert client.post("/ingest", json={"path": escape}).status_code == 403
