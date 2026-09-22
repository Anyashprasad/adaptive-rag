from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from .chunking import chunk_text
from .models import DocumentChunk

TEXT_EXTS = {".txt", ".md", ".log", ".py", ".js", ".ts", ".html", ".yaml", ".yml"}
ROMAN_RE = re.compile(r"[ivxlcdm]{1,10}", re.IGNORECASE)
SECTION_LINE_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\s+(.+?)\s*$")
CHAPTER_HEADER_RE = re.compile(r"^CHAPTER\s+(\d+)\b(?:\.\s*(.*))?", re.IGNORECASE)


def _read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".json":
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return json.dumps(obj, indent=2, ensure_ascii=False)
    if ext == ".csv":
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            for row in csv.reader(f):
                rows.append(" | ".join(row))
        return "\n".join(rows)
    raise ValueError(f"Unsupported file type: {ext}")


def _normalize_page_label(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().lower()
    if re.fullmatch(r"\d{1,6}", value) or ROMAN_RE.fullmatch(value):
        return value
    return None


def _infer_printed_page(text: str) -> str | None:
    """Text-only fallback for visible page labels (footer/header)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates = list(reversed(lines[-2:])) + lines[:2]
    for value in candidates:
        label = _normalize_page_label(value)
        if label is not None:
            return label
    return None


def _infer_printed_page_pymupdf(page, text: str) -> str | None:
    """Prefer the PDF's declared page label, then geometric footer/header labels.

    Geometry avoids accidentally treating a standalone number in body content as the
    visible page number. This matters for books where PDF page 597 is printed page 581.
    """
    try:
        declared = _normalize_page_label(page.get_label())
        if declared:
            return declared
    except Exception:
        pass

    # Fast path: most text PDFs expose a footer/header label as one of the outer
    # text lines. Avoid a second page extraction unless that fails.
    textual = _infer_printed_page(text)
    if textual:
        return textual

    try:
        height = float(page.rect.height)
        footer: list[tuple[float, str]] = []
        header: list[tuple[float, str]] = []
        for block in page.get_text("blocks", sort=True):
            _x0, y0, _x1, y1, raw, *_ = block
            value = " ".join(str(raw).split())
            label = _normalize_page_label(value)
            if label is None:
                continue
            if y0 >= height * 0.84:
                footer.append((float(y0), label))
            elif y1 <= height * 0.14:
                header.append((float(y0), label))
        if footer:
            return sorted(footer, key=lambda x: x[0], reverse=True)[0][1]
        if header:
            return sorted(header, key=lambda x: x[0])[0][1]
    except Exception:
        pass
    return None


def _detect_structure(
    text: str,
    previous_chapter: str | None,
    previous_section: str | None,
    previous_section_title: str | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Best-effort chapter/section propagation for book-like PDFs.

    We only accept section headings when the page itself has a CHAPTER header (or a
    chapter was already active), which keeps table-of-contents lines from poisoning
    structural metadata.
    """
    lines = [" ".join(ln.split()) for ln in text.splitlines() if ln.strip()]
    chapter = previous_chapter
    section = previous_section
    section_title = previous_section_title
    page_heading: str | None = None

    page_chapter = None
    for line in lines[:12]:
        m = CHAPTER_HEADER_RE.match(line)
        if m:
            page_chapter = m.group(1)
            break
    if page_chapter and page_chapter != chapter:
        chapter = page_chapter
        section = None
        section_title = None

    if chapter:
        prefix = f"{chapter}."
        # Real section headings are usually near the top. Reject dot-leader / trailing
        # page-number TOC-style lines.
        for line in lines[:30]:
            m = SECTION_LINE_RE.match(line)
            if not m or not m.group(1).startswith(prefix):
                continue
            title = m.group(2).strip()
            if re.search(r"\.{3,}", title) or re.search(r"\s\d{1,4}$", title):
                continue
            section = m.group(1)
            section_title = title
            page_heading = f"{section} {title}"
            break

    if page_heading is None and page_chapter:
        # The running chapter header is provenance, not necessarily a semantic heading.
        page_heading = f"Chapter {page_chapter}"
    return chapter, section, section_title, page_heading


def _empty_page_chunk(source: str, source_name: str, page_idx: int, printed: str | None, official_label: str | None, global_idx: int) -> DocumentChunk:
    digest = hashlib.sha1(f"{source}:page:{page_idx}:empty".encode("utf-8")).hexdigest()[:12]
    return DocumentChunk(
        chunk_id=f"{source}:p{page_idx}:{digest}",
        text="",
        source=source,
        chunk_index=global_idx,
        metadata={
            "source_name": source_name,
            "pdf_page": page_idx,
            "pdf_label": str(page_idx),
            "document_page_label": official_label,
            "printed_page": printed,
            "page_chunk_index": 0,
            "page_chunk_count": 1,
            "text_extracted": False,
            "dates": [], "times": [], "datetimes": [], "word_count": 0,
            "word_start": 0, "word_end": 0,
            "content_sha1": hashlib.sha1(b"").hexdigest(),
        },
    )


def _ingest_pdf(path: Path, chunk_words: int, overlap_words: int) -> list[DocumentChunk]:
    source = str(path)
    source_name = path.name
    chunks: list[DocumentChunk] = []
    global_idx = 0
    current_chapter = current_section = current_section_title = None

    # PyMuPDF is dramatically faster on large books; keep pypdf as a zero-config fallback.
    try:
        import pymupdf
        doc = pymupdf.open(str(path))
        for i, page in enumerate(doc):
            page_idx = i + 1
            text = page.get_text("text", sort=True) or ""
            official_label = None
            try:
                official_label = _normalize_page_label(page.get_label())
            except Exception:
                pass
            visible = _infer_printed_page_pymupdf(page, text)
            current_chapter, current_section, current_section_title, page_heading = _detect_structure(
                text, current_chapter, current_section, current_section_title
            )

            if not text.strip():
                c = _empty_page_chunk(source, source_name, page_idx, visible, official_label, global_idx)
                c.metadata.update({
                    "chapter": current_chapter, "section": current_section,
                    "section_title": current_section_title, "page_heading": page_heading,
                })
                chunks.append(c)
                global_idx += 1
                continue

            page_chunks = chunk_text(text, source, chunk_words, overlap_words)
            count = len(page_chunks)
            for local_idx, c in enumerate(page_chunks):
                c.chunk_index = global_idx
                c.chunk_id = f"{source}:p{page_idx}:{c.chunk_id.rsplit(':', 1)[-1]}"
                c.metadata.update({
                    "source_name": source_name,
                    "pdf_page": page_idx,
                    "pdf_label": str(page_idx),
                    "document_page_label": official_label,
                    "printed_page": visible,
                    "page_chunk_index": local_idx,
                    "page_chunk_count": count,
                    "text_extracted": True,
                    "chapter": current_chapter,
                    "section": current_section,
                    "section_title": current_section_title,
                    "page_heading": page_heading,
                })
                chunks.append(c)
                global_idx += 1
        doc.close()
        return chunks
    except ImportError:
        pass

    from pypdf import PdfReader
    reader = PdfReader(str(path))
    labels = getattr(reader, "page_labels", None) or []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        official_label = _normalize_page_label(str(labels[page_idx - 1])) if page_idx - 1 < len(labels) else None
        visible = official_label or _infer_printed_page(text)
        current_chapter, current_section, current_section_title, page_heading = _detect_structure(
            text, current_chapter, current_section, current_section_title
        )
        if not text.strip():
            c = _empty_page_chunk(source, source_name, page_idx, visible, official_label, global_idx)
            c.metadata.update({
                "chapter": current_chapter, "section": current_section,
                "section_title": current_section_title, "page_heading": page_heading,
            })
            chunks.append(c)
            global_idx += 1
            continue
        page_chunks = chunk_text(text, source, chunk_words, overlap_words)
        count = len(page_chunks)
        for local_idx, c in enumerate(page_chunks):
            c.chunk_index = global_idx
            c.chunk_id = f"{source}:p{page_idx}:{c.chunk_id.rsplit(':', 1)[-1]}"
            c.metadata.update({
                "source_name": source_name,
                "pdf_page": page_idx,
                "pdf_label": str(page_idx),
                "document_page_label": official_label,
                "printed_page": visible,
                "page_chunk_index": local_idx,
                "page_chunk_count": count,
                "text_extracted": True,
                "chapter": current_chapter,
                "section": current_section,
                "section_title": current_section_title,
                "page_heading": page_heading,
            })
            chunks.append(c)
            global_idx += 1
    return chunks


def ingest_path(path: str | Path, chunk_words: int = 180, overlap_words: int = 35) -> list[DocumentChunk]:
    path = Path(path)
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    chunks: list[DocumentChunk] = []
    for file in files:
        try:
            if file.suffix.lower() == ".pdf":
                chunks.extend(_ingest_pdf(file, chunk_words, overlap_words))
                continue
            text = _read_file(file)
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        source = str(file)
        file_chunks = chunk_text(text, source, chunk_words, overlap_words)
        for c in file_chunks:
            c.metadata.setdefault("source_name", file.name)
            c.metadata.setdefault("text_extracted", True)
        chunks.extend(file_chunks)
    return chunks
