from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture(autouse=True)
def openrouter_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "test-model")


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summarize_uses_configured_summarizer(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_summarize(
        text: str,
        client: object,
        model: str,
    ) -> main.SummarizeResponse:
        assert text == "FastAPI validates request data. Pydantic defines the contract."
        assert client is not None
        assert model == "test-model"
        return main.SummarizeResponse(
            summary="FastAPI validates request data",
            model="test-model",
        )

    monkeypatch.setattr(main, "summarize_text", fake_summarize)

    response = client.post(
        "/summarize",
        json={"text": "FastAPI validates request data. Pydantic defines the contract."},
    )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "FastAPI validates request data",
        "model": "test-model",
    }


def test_summarize_rejects_empty_text(client: TestClient) -> None:
    response = client.post("/summarize", json={"text": ""})

    assert response.status_code == 422


def test_summarize_returns_safe_error_when_provider_is_unavailable(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable_summarizer(
        text: str,
        client: object,
        model: str,
    ) -> main.SummarizeResponse:
        raise main.SummarizationUnavailableError

    monkeypatch.setattr(main, "summarize_text", unavailable_summarizer)

    response = client.post("/summarize", json={"text": "A short request."})

    assert response.status_code == 503
    assert response.json() == {"detail": "Summarization is temporarily unavailable."}
