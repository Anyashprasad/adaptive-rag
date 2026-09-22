from adaptive_rag.router import QueryRouter


def test_temporal_query_routes_exact_first():
    a = QueryRouter().analyze("What happened on 4 September 2026 at 11:42 am?")
    assert a.route in {"exact_temporal", "mixed"}
    assert "2026-09-04" in a.dates
    assert "11:42" in a.times


def test_conceptual_query_routes_semantic_first():
    a = QueryRouter().analyze("Why must every student's implementation be unique?")
    assert a.route == "conceptual"


def test_mixed_query_routes_parallel():
    a = QueryRouter().analyze("Explain what changed after 4 September 2026")
    assert a.route == "mixed"


def test_page_lookup_routes_exact_first():
    a = QueryRouter().analyze("Find and summarize whatever is on page 581")
    assert a.route == "exact_temporal"
    assert a.page_numbers == [581]
    assert a.page_scope == "any"


def test_explicit_pdf_page_scope():
    a = QueryRouter().analyze("summarize PDF page 581")
    assert a.page_numbers == [581]
    assert a.page_scope == "pdf"
