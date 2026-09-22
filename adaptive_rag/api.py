from __future__ import annotations

from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .confidence import ConfidencePolicy
from .engine import AdaptiveRAG
from .generator import ExtractiveGenerator, OllamaGenerator
from .ingestion import ingest_path
from .semantic import HashingSemanticBackend, LSASemanticBackend, SentenceTransformerBackend

app = FastAPI(title="AdaptiveRAG", version="0.3.0")
engine: AdaptiveRAG | None = None
engine_config: dict[str, str] = {}


class BuildRequest(BaseModel):
    path: str
    semantic_backend: str = Field(default="sentence-transformer", pattern="^(sentence-transformer|lsa|hashing)$")
    generation_backend: str = Field(default="extractive", pattern="^(extractive|ollama)$")
    ollama_model: str | None = None
    calibration_path: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=6, ge=1, le=20)
    generate: bool = False
    reference_time: datetime | None = None


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).with_name("static") / "index.html")


@app.get("/health")
def health():
    return {
        "ok": True,
        "indexed_chunks": len(engine.chunks) if engine else 0,
        "config": engine_config,
    }


@app.post("/build")
def build(req: BuildRequest):
    global engine, engine_config
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    chunks = ingest_path(p)
    if not chunks:
        raise HTTPException(status_code=422, detail="No supported, non-empty documents were found")
    backend = (HashingSemanticBackend() if req.semantic_backend == "hashing" else LSASemanticBackend() if req.semantic_backend == "lsa" else SentenceTransformerBackend())
    generator = OllamaGenerator(model=req.ollama_model) if req.generation_backend == "ollama" else ExtractiveGenerator()
    try:
        policy = ConfidencePolicy(req.calibration_path) if req.calibration_path else ConfidencePolicy()
        candidate = AdaptiveRAG(semantic_backend=backend, generator=generator, confidence_policy=policy)
        candidate.build(chunks)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    engine = candidate
    engine_config = {
        "semantic_backend": req.semantic_backend,
        "generation_backend": req.generation_backend,
        "source_path": str(p),
        "calibration": req.calibration_path or "none",
    }
    return {"chunks": len(chunks), **engine_config}


@app.post("/query")
def query(req: QueryRequest):
    if engine is None:
        raise HTTPException(status_code=409, detail="Build an index first with POST /build")
    try:
        return engine.retrieve(
            req.query, k=req.k, generate=req.generate, reference_time=req.reference_time
        ).to_dict()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
