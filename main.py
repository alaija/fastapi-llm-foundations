import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


class HealthResponse(BaseModel):
    status: str


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class SummarizeResponse(BaseModel):
    summary: str
    model: str


app = FastAPI(title="FastAPI LLM Foundations")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


async def summarize_text(text: str) -> SummarizeResponse:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    client = AsyncOpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    response = await client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
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
    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise RuntimeError("OpenRouter returned an empty summary")

    return SummarizeResponse(summary=content.strip(), model=response.model)


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    return await summarize_text(request.text)
