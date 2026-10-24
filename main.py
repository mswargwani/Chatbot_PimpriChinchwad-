"""
main.py  -  FASTAPI SERVER
Exposes the chat API and serves the frontend from ./static
Run:  uvicorn main:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag_backend

STATIC_DIR = Path(__file__).parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the RAG chain once at startup (loads embeddings + vector DB)
    state["chains"] = rag_backend.build_chains()
    yield
    state.clear()


app = FastAPI(title="Conversational RAG Chatbot", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=100)
    mode: str = Field(default="general", pattern="^(general|pcmc)$")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Plain `def` endpoints run in a thread pool, so the blocking LLM call
# does not freeze the server.
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return rag_backend.ask(state["chains"], req.mode, req.message.strip(), req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


@app.get("/api/history/{session_id}")
def history(session_id: str):
    return {"messages": rag_backend.get_history(session_id)}


@app.delete("/api/history/{session_id}")
def clear(session_id: str):
    rag_backend.clear_history(session_id)
    return {"status": "cleared"}


# Serve the frontend (must be mounted LAST so /api routes take priority)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
