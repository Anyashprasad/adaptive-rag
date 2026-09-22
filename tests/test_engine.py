from adaptive_rag.chunking import chunk_text
from adaptive_rag.engine import AdaptiveRAG
from adaptive_rag.models import DocumentChunk
from adaptive_rag.semantic import HashingSemanticBackend

TEXT = """
On 4 September 2026 at 11:42 am the faculty said the project name deadline is 8 September 2026.
Students may choose related topics, but their implementation and technology stack must differ.
AdaptiveRAG prioritizes exact retrieval for timestamps and semantic retrieval for conceptual questions.
"""


def make_rag():
    chunks = chunk_text(TEXT, "fixture.txt", chunk_words=28, overlap_words=4)
    return AdaptiveRAG(chunks, semantic_backend=HashingSemanticBackend())


def test_exact_query_gets_evidence():
    r = make_rag().retrieve("What was said on 4 September 2026 at 11:42 am?")
    assert r.hits
    assert r.trace.first_stage == "lexical"
    assert "deadline" in r.hits[0].chunk.text.lower()
    assert r.hits[0].features.get("date_match") is True
    assert r.hits[0].features.get("time_match") is True


def test_conceptual_query_semantic_first():
    r = make_rag().retrieve("Why must the project implementation be different?")
    assert r.trace.first_stage == "semantic"
    assert r.hits


def test_empty_query_rejected():
    try:
        make_rag().retrieve("   ")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("empty query should fail")


def test_semantic_fallback_cannot_violate_exact_timestamp_constraint():
    chunks = [
        DocumentChunk(
            "right", "Meeting happened on 4 September 2026 at 11:42 and covered project topics.", "events.txt", 0,
            {"dates": ["2026-09-04"], "times": ["11:42"], "datetimes": ["2026-09-04T11:42"]},
        ),
        DocumentChunk(
            "wrong", "A long conceptually rich explanation about project topics and meetings on 5 September 2026 at 11:42.", "events.txt", 1,
            {"dates": ["2026-09-05"], "times": ["11:42"], "datetimes": ["2026-09-05T11:42"]},
        ),
    ]
    rag = AdaptiveRAG(chunks, semantic_backend=HashingSemanticBackend())
    r = rag.retrieve("explain what happened on 4 September 2026 at 11:42", k=2)
    assert all(h.chunk.chunk_id != "wrong" for h in r.hits)
    assert r.hits and r.hits[0].chunk.chunk_id == "right"
