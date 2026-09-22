from adaptive_rag.engine import AdaptiveRAG
from adaptive_rag.models import DocumentChunk
from adaptive_rag.semantic import HashingSemanticBackend


def test_bare_page_prefers_visible_printed_page_over_physical_pdf_page():
    chunks = [
        DocumentChunk("a", "wrong physical page", "book.pdf", 0, {"pdf_page": 581, "printed_page": "565", "dates": [], "times": [], "datetimes": []}),
        DocumentChunk("b", "correct visible page about Gibbs sampling", "book.pdf", 1, {"pdf_page": 597, "printed_page": "581", "dates": [], "times": [], "datetimes": []}),
    ]
    rag = AdaptiveRAG(chunks, semantic_backend=HashingSemanticBackend())
    r = rag.retrieve("find and summarize whatever is on page 581", k=2)
    assert r.trace.first_stage == "lexical"
    assert r.hits[0].chunk.chunk_id == "b"
    assert r.hits[0].features["printed_page_match"] is True
    assert r.trace.fallback_triggered is False


def test_explicit_pdf_page_targets_physical_page():
    chunks = [
        DocumentChunk("a", "physical page", "book.pdf", 0, {"pdf_page": 581, "printed_page": "565", "dates": [], "times": [], "datetimes": []}),
        DocumentChunk("b", "printed page", "book.pdf", 1, {"pdf_page": 597, "printed_page": "581", "dates": [], "times": [], "datetimes": []}),
    ]
    rag = AdaptiveRAG(chunks, semantic_backend=HashingSemanticBackend())
    r = rag.retrieve("summarize PDF page 581", k=2)
    assert r.hits[0].chunk.chunk_id == "a"
    assert r.hits[0].features["pdf_page_match"] is True


def test_page_lookup_returns_chunks_in_reading_order():
    chunks = [
        DocumentChunk("p1", "page start", "book.pdf", 10, {"pdf_page": 597, "printed_page": "581", "dates": [], "times": [], "datetimes": []}),
        DocumentChunk("p2", "middle content", "book.pdf", 11, {"pdf_page": 597, "printed_page": "581", "dates": [], "times": [], "datetimes": []}),
        DocumentChunk("p3", "footer 581", "book.pdf", 12, {"pdf_page": 597, "printed_page": "581", "dates": [], "times": [], "datetimes": []}),
    ]
    rag = AdaptiveRAG(chunks, semantic_backend=HashingSemanticBackend())
    r = rag.retrieve("summarize page 581", k=3)
    assert [h.chunk.chunk_id for h in r.hits] == ["p1", "p2", "p3"]


def test_missing_page_returns_no_evidence_and_never_semantic_fallback():
    chunks = [
        DocumentChunk("a", "this talks about page-like concepts", "book.pdf", 0,
                      {"pdf_page": 10, "printed_page": "1", "dates": [], "times": [], "datetimes": []}),
    ]
    rag = AdaptiveRAG(chunks, semantic_backend=HashingSemanticBackend())
    r = rag.retrieve("summarize page 581", k=6)
    assert r.hits == []
    assert r.trace.fallback_triggered is False
    assert "semantic fallback suppressed" in " ".join(r.trace.reasons)


def test_page_range_resolves_all_requested_pages_in_reading_order():
    chunks = [
        DocumentChunk("a1", "start 581", "book.pdf", 0, {"pdf_page": 597, "printed_page": "581", "page_chunk_index": 0, "dates": [], "times": [], "datetimes": []}),
        DocumentChunk("a2", "end 581", "book.pdf", 1, {"pdf_page": 597, "printed_page": "581", "page_chunk_index": 1, "dates": [], "times": [], "datetimes": []}),
        DocumentChunk("b1", "start 582", "book.pdf", 2, {"pdf_page": 598, "printed_page": "582", "page_chunk_index": 0, "dates": [], "times": [], "datetimes": []}),
        DocumentChunk("c1", "start 583", "book.pdf", 3, {"pdf_page": 599, "printed_page": "583", "page_chunk_index": 0, "dates": [], "times": [], "datetimes": []}),
    ]
    rag = AdaptiveRAG(chunks, semantic_backend=HashingSemanticBackend())
    r = rag.retrieve("summarize pages 581-583", k=1)
    assert r.analysis.page_labels == ["581", "582", "583"]
    assert [h.chunk.chunk_id for h in r.hits] == ["a1", "a2", "b1", "c1"]


def test_roman_numeral_printed_page_is_supported():
    chunks = [
        DocumentChunk("front", "Notation and symbols", "book.pdf", 0,
                      {"pdf_page": 13, "printed_page": "xi", "dates": [], "times": [], "datetimes": []}),
        DocumentChunk("body", "Chapter content", "book.pdf", 1,
                      {"pdf_page": 581, "printed_page": "565", "dates": [], "times": [], "datetimes": []}),
    ]
    rag = AdaptiveRAG(chunks, semantic_backend=HashingSemanticBackend())
    r = rag.retrieve("what is on page xi?", k=2)
    assert r.analysis.page_labels == ["xi"]
    assert [h.chunk.chunk_id for h in r.hits] == ["front"]


def test_filename_terms_disambiguate_same_printed_page_across_documents():
    chunks = [
        DocumentChunk("a", "generic material", "/tmp/deep_learning.pdf", 0,
                      {"source_name": "deep_learning.pdf", "pdf_page": 597, "printed_page": "581", "dates": [], "times": [], "datetimes": []}),
        DocumentChunk("b", "generic material", "/tmp/other_book.pdf", 1,
                      {"source_name": "other_book.pdf", "pdf_page": 581, "printed_page": "581", "dates": [], "times": [], "datetimes": []}),
    ]
    rag = AdaptiveRAG(chunks, semantic_backend=HashingSemanticBackend())
    r = rag.retrieve("summarize page 581 of deep_learning.pdf", k=2)
    assert r.hits[0].chunk.chunk_id == "a"


def test_page_locator_resolves_empty_extraction_without_inventing_other_content():
    chunks = [
        DocumentChunk("empty", "", "scan.pdf", 0,
                      {"pdf_page": 5, "printed_page": None, "text_extracted": False, "dates": [], "times": [], "datetimes": []}),
        DocumentChunk("other", "very semantically relevant summary text", "scan.pdf", 1,
                      {"pdf_page": 6, "printed_page": None, "text_extracted": True, "dates": [], "times": [], "datetimes": []}),
    ]
    rag = AdaptiveRAG(chunks, semantic_backend=HashingSemanticBackend())
    r = rag.retrieve("summarize PDF page 5", k=2)
    assert [h.chunk.chunk_id for h in r.hits] == ["empty"]
    assert any("no extractable text" in reason for reason in r.trace.reasons)


def test_exact_page_delivery_removes_chunk_overlap_without_losing_content():
    chunks = [
        DocumentChunk("a", "one two three four", "book.pdf", 0,
                      {"pdf_page": 17, "printed_page": "1", "page_chunk_index": 0,
                       "word_start": 0, "word_end": 4, "dates": [], "times": [], "datetimes": []}),
        DocumentChunk("b", "three four five six", "book.pdf", 1,
                      {"pdf_page": 17, "printed_page": "1", "page_chunk_index": 1,
                       "word_start": 2, "word_end": 6, "dates": [], "times": [], "datetimes": []}),
    ]
    rag = AdaptiveRAG(chunks, semantic_backend=HashingSemanticBackend())
    r = rag.retrieve("page 1", k=1)
    assert [h.chunk.text for h in r.hits] == ["one two three four", "five six"]
    assert r.hits[1].features["overlap_trimmed"] == 2
