from pathlib import Path
from fastapi.testclient import TestClient

import adaptive_rag.api as api_module

ROOT = Path(__file__).resolve().parents[1]


def test_api_end_to_end_build_query_and_ui():
    api_module.engine = None
    api_module.engine_config = {}
    client = TestClient(api_module.app)

    assert client.get("/").status_code == 200
    assert "AdaptiveRAG" in client.get("/").text
    assert client.post("/query", json={"query": "hello"}).status_code == 409

    build = client.post("/build", json={
        "path": str(ROOT / "sample_data"),
        "semantic_backend": "hashing",
        "generation_backend": "extractive",
    })
    assert build.status_code == 200, build.text
    assert build.json()["chunks"] >= 1

    response = client.post("/query", json={
        "query": "What happened on 4 September 2026 at 11:42 am?",
        "k": 3,
        "generate": True,
    })
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["trace"]["first_stage"] == "lexical"
    assert data["hits"]
    assert "Grounded evidence" in data["answer"]
    assert "project name" in data["hits"][0]["text"].lower()

    health = client.get("/health").json()
    assert health["indexed_chunks"] >= 1
    assert health["config"]["semantic_backend"] == "hashing"


def test_api_resolves_relative_dates_with_reference_time():
    api_module.engine = None
    client = TestClient(api_module.app)
    client.post("/build", json={
        "path": str(ROOT / "sample_data"),
        "semantic_backend": "hashing",
    })
    response = client.post("/query", json={
        "query": "What happened yesterday at 11:42 am?",
        "reference_time": "2026-09-05T18:30:00+05:30",
    })
    assert response.status_code == 200
    data = response.json()
    assert "2026-09-04" in data["analysis"]["dates"]
    assert data["trace"]["first_stage"] == "lexical"
