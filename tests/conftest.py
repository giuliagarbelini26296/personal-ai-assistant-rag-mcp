from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from assistant.config import DEFAULT_SAMPLE_DIR, Settings
from assistant.llm import FakeLLM
from assistant.service import AssistantService


@pytest.fixture
def settings(tmp_path) -> Settings:
    return replace(
        Settings(),
        db_path=str(tmp_path / "test.sqlite3"),
        embedding_provider="hash",
        llm_provider="fake",
        log_level="WARNING",
    )


@pytest.fixture
def service(settings) -> AssistantService:
    svc = AssistantService(settings, llm=FakeLLM())
    svc.tools._clock = lambda: datetime(2026, 11, 19, 10, 0)
    svc.ingest(DEFAULT_SAMPLE_DIR)
    yield svc
    svc.close()
