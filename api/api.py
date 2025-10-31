# 文件路径：api/api.py
import hashlib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import time
import re
import uvicorn

from core.learn_mate_core import cached_rag, rag_cache, health_check

app = FastAPI(
    title="LearnMate RAG API",
    description="Chapter 2 课程智能问答引擎",
    version="1.0.0"
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    response_time_ms: int
    cache_hit: bool


@app.post("/api/v1/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    start = time.time()
    answer = cached_rag(request.question)
    elapsed = int((time.time() - start) * 1000)

    sources = re.findall(r'chunk_\d+', answer)
    cache_key = f"rag_v1:{hashlib.md5(request.question.strip().lower().encode()).hexdigest()}"
    cache_hit = cache_key in rag_cache

    return AskResponse(
        answer=answer,
        sources=sources,
        response_time_ms=elapsed,
        cache_hit=cache_hit
    )


@app.get("/health")
def health():
    return health_check()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)