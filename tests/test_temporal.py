from datetime import datetime, timezone

from adaptive_rag.router import QueryRouter
from adaptive_rag.text import parse_temporal_constraints


def test_relative_yesterday_is_resolved_from_reference_time():
    ref = datetime(2026, 9, 5, 18, 30, tzinfo=timezone.utc)
    a = QueryRouter().analyze("What happened yesterday at 11:42?", reference_time=ref)
    assert "2026-09-04" in a.dates
    assert a.date_start == "2026-09-04"
    assert a.date_end == "2026-09-04"
    assert a.times == ["11:42"]
    assert a.route == "exact_temporal"


def test_date_range_is_normalized():
    t = parse_temporal_constraints(
        "What happened between 3 September 2026 and 5 September 2026?"
    )
    assert t["operator"] == "range"
    assert t["date_start"] == "2026-09-03"
    assert t["date_end"] == "2026-09-05"


def test_last_days_range_uses_reference_time():
    ref = datetime(2026, 9, 5, 18, 30, tzinfo=timezone.utc)
    t = parse_temporal_constraints("show events from the last 3 days", reference_time=ref)
    assert t["date_start"] == "2026-09-03"
    assert t["date_end"] == "2026-09-05"


def test_datetime_pair_extraction_does_not_cross_events():
    from adaptive_rag.chunking import chunk_text
    from adaptive_rag.lexical import LexicalIndex
    text = "[04/09/2026, 11:46 am] event A. [05/09/2026, 11:42 am] event B."
    chunks = chunk_text(text, "events.txt", chunk_words=80, overlap_words=0)
    a = QueryRouter().analyze("What happened on 4 September 2026 at 11:42 am?")
    hit = LexicalIndex(chunks).search(a.query, a, k=1)[0]
    assert hit.features["date_match"] is True
    assert hit.features["time_match"] is True
    assert hit.features["datetime_match"] is False
    assert hit.features["exact_coverage"] == 0.0
