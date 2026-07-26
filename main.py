from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class HealthResponse(BaseModel):
    status: str


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class SummarizeResponse(BaseModel):
    summary: str
    model: str


class SummarizationUnavailableError(Exception):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openrouter_api_key: SecretStr
    openrouter_base_url: str
    openrouter_model: str


def load_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]  # Values are supplied by BaseSettings.
    except ValidationError as error:
        raise RuntimeError(
            "Define OPENROUTER_API_KEY, OPENROUTER_BASE_URL, and OPENROUTER_MODEL in .env."
        ) from error


# FastAPI runs lifespan setup before serving requests and cleanup at shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=settings.openrouter_base_url,
    )
    app.state.openrouter_client = client
    app.state.openrouter_model = settings.openrouter_model

    try:
        yield
    finally:
        await client.close()


app = FastAPI(title="FastAPI LLM Foundations", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


async def summarize_text(
    text: str,
    client: AsyncOpenAI,
    model: str,
) -> SummarizeResponse:
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the input in one sentence. Return only the summary.",
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=120,
        )
    except APIError as error:
        raise SummarizationUnavailableError from error

    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise SummarizationUnavailableError

    return SummarizeResponse(summary=content.strip(), model=response.model)


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(payload: SummarizeRequest, request: Request) -> SummarizeResponse:
    # app.state is dynamic; these values were created and validated during lifespan startup.
    client = cast(AsyncOpenAI, request.app.state.openrouter_client)
    model = cast(str, request.app.state.openrouter_model)
    try:
        return await summarize_text(payload.text, client, model)
    except SummarizationUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail="Summarization is temporarily unavailable.",
        ) from error
