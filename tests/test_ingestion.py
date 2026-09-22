from pathlib import Path

from adaptive_rag.ingestion import ingest_path

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_ingestion_skips_unsupported_and_reads_text_json_csv(tmp_path):
    (tmp_path / "a.txt").write_text("hello 4 September 2026 at 11:42", encoding="utf-8")
    (tmp_path / "b.json").write_text('{"notice":"deadline 8 September 2026"}', encoding="utf-8")
    (tmp_path / "c.csv").write_text("id,time\nABC-42,17:00\n", encoding="utf-8")
    (tmp_path / "d.bin").write_bytes(b"123")
    chunks = ingest_path(tmp_path)
    assert len(chunks) >= 3
    joined = "\n".join(c.text for c in chunks)
    assert "ABC-42" in joined
    assert any("2026-09-04" in c.metadata["dates"] for c in chunks)
    assert any("11:42" in c.metadata["times"] for c in chunks)


def test_pdf_ingestion_extracts_text_and_metadata():
    chunks = ingest_path(FIXTURES / "sample_notice.pdf")
    assert chunks
    joined = " ".join(c.text for c in chunks)
    assert "PDF-TEST-42" in joined
    assert any("2026-09-08" in c.metadata["dates"] for c in chunks)
    assert any("17:00" in c.metadata["times"] for c in chunks)


def test_pdf_ingestion_preserves_page_metadata():
    chunks = ingest_path(FIXTURES / "sample_notice.pdf")
    assert all("pdf_page" in c.metadata for c in chunks)
