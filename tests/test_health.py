from fastapi.testclient import TestClient

import main


def test_health_returns_ok() -> None:
    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summarize_uses_configured_summarizer(monkeypatch) -> None:
    async def fake_summarize(text: str) -> main.SummarizeResponse:
        assert text == "FastAPI validates request data. Pydantic defines the contract."
        return main.SummarizeResponse(
            summary="FastAPI validates request data",
            model="test-model",
        )

    monkeypatch.setattr(main, "summarize_text", fake_summarize)

    response = TestClient(main.app).post(
        "/summarize",
        json={"text": "FastAPI validates request data. Pydantic defines the contract."},
    )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "FastAPI validates request data",
        "model": "test-model",
    }


def test_summarize_rejects_empty_text() -> None:
    response = TestClient(main.app).post("/summarize", json={"text": ""})

    assert response.status_code == 422
