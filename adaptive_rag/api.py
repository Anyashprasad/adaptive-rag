from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .confidence import ConfidencePolicy
from .engine import AdaptiveRAG
from .generator import ExtractiveGenerator
from .ingestion import ingest_path
from .semantic import LSASemanticBackend

app = FastAPI(title="AdaptiveRAG", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine: AdaptiveRAG | None = None
engine_meta: dict = {}


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=6, ge=1, le=20)
    reference_time: datetime | None = None


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).with_name("static") / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "indexed_chunks": len(engine.chunks) if engine else 0}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Accept a PDF or text file, index it, ready for querying."""
    global engine, engine_meta
    suffix = Path(file.filename or "doc").suffix.lower() or ".txt"
    allowed = {".pdf", ".txt", ".md", ".csv", ".json", ".log", ".py", ".html", ".yaml", ".yml"}
    if suffix not in allowed:
        raise HTTPException(status_code=422, detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(allowed))}")
    tmp_dir = tempfile.mkdtemp(prefix="adaptiverag_")
    try:
        dest = Path(tmp_dir) / (file.filename or f"document{suffix}")
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        chunks = ingest_path(dest)
        if not chunks:
            raise HTTPException(status_code=422, detail="Could not extract any text from the document.")
        candidate = AdaptiveRAG(
            chunks,
            semantic_backend=LSASemanticBackend(),
            generator=ExtractiveGenerator(),
            confidence_policy=ConfidencePolicy(),
        )
        engine = candidate
        engine_meta = {"filename": file.filename, "chunks": len(chunks)}
        return {"ok": True, "filename": file.filename, "chunks": len(chunks)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query")
def query(req: QueryRequest):
    if engine is None:
        raise HTTPException(status_code=409, detail="Upload a document first.")
    try:
        result = engine.retrieve(req.query, k=req.k, generate=True, reference_time=req.reference_time)
        r = result.to_dict()
        # Return clean hits — just text, source, page info
        hits = []
        for h in r["hits"]:
            hit = {
                "text": h["text"],
                "source": Path(h["source"]).name,
                "score": h["score"],
                "page": h["metadata"].get("printed_page") or h["metadata"].get("pdf_page"),
                "channel": h["channel"],
            }
            hits.append(hit)
        return {
            "query": r["query"],
            "route": r["trace"]["route"],
            "hits": hits,
            "answer": r["answer"],
        }
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# Keep /build for backward-compat with CLI/tests but hide from user-facing UI
@app.post("/build", include_in_schema=False)
def build_legacy(req: dict):
    raise HTTPException(status_code=410, detail="Use POST /upload instead.")
