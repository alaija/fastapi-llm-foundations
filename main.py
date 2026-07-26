from fastapi import FastAPI
from pydantic import BaseModel, Field


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


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    summary = request.text.split(".", maxsplit=1)[0].strip()
    return SummarizeResponse(summary=summary, model="fake-llm")
