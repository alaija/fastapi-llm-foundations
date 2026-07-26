from fastapi.testclient import TestClient

from main import app


def test_health_returns_ok() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summarize_returns_first_sentence() -> None:
    response = TestClient(app).post(
        "/summarize",
        json={"text": "FastAPI validates request data. Pydantic defines the contract."},
    )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "FastAPI validates request data",
        "model": "fake-llm",
    }


def test_summarize_rejects_empty_text() -> None:
    response = TestClient(app).post("/summarize", json={"text": ""})

    assert response.status_code == 422
