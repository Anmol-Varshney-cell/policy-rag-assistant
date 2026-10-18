"""FastAPI web app for the Policy RAG Assistant.

Vercel finds the `app` instance in this file automatically and serves it
as a Python serverless function. The vector index is built lazily on the
first question (one batched embedding call) and cached in memory.
"""

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag_system.rag_pipeline import RAGPipeline

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "policies"

app = FastAPI(title="Policy RAG Assistant")

_rag = None
_init_lock = asyncio.Lock()


async def get_rag() -> RAGPipeline:
    """Lazily build the RAG pipeline and vector index once per instance."""
    global _rag
    if _rag is None:
        async with _init_lock:
            if _rag is None:
                rag = RAGPipeline(str(DATA_DIR), index_dir="/tmp/vector_index")
                # /tmp is the only writable path on serverless; always rebuild there
                await rag.build_index(force_rebuild=True)
                _rag = rag
    return _rag


class AskRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def home():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health():
    return {"status": "ok", "gemini_key_configured": bool(os.getenv("GEMINI_API_KEY"))}


@app.post("/api/ask")
async def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        return {"error": "Question cannot be empty."}
    if not os.getenv("GEMINI_API_KEY"):
        return {"error": "GEMINI_API_KEY is not configured on the server."}
    rag = await get_rag()
    result = await rag.answer_question(question)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
    }
