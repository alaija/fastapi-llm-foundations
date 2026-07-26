from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

import main

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def openrouter_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "test-model")


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with main.lifespan(main.app):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as test_client:
            yield test_client


async def test_health_returns_ok(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_summarize_uses_configured_summarizer(
    client: httpx.AsyncClient,
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

    response = await client.post(
        "/summarize",
        json={"text": "FastAPI validates request data. Pydantic defines the contract."},
    )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "FastAPI validates request data",
        "model": "test-model",
    }


async def test_summarize_rejects_empty_text(client: httpx.AsyncClient) -> None:
    response = await client.post("/summarize", json={"text": ""})

    assert response.status_code == 422


async def test_summarize_returns_safe_error_when_provider_is_unavailable(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable_summarizer(
        text: str,
        client: object,
        model: str,
    ) -> main.SummarizeResponse:
        raise main.SummarizationUnavailableError

    monkeypatch.setattr(main, "summarize_text", unavailable_summarizer)

    response = await client.post("/summarize", json={"text": "A short request."})

    assert response.status_code == 503
    assert response.json() == {"detail": "Summarization is temporarily unavailable."}
