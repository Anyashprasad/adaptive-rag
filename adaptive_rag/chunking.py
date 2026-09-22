from __future__ import annotations

import hashlib
from .models import DocumentChunk
from .text import normalize_dates, normalize_datetimes, normalize_times


def chunk_text(text: str, source: str, chunk_words: int = 180, overlap_words: int = 35) -> list[DocumentChunk]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    chunks: list[DocumentChunk] = []
    for idx, start in enumerate(range(0, len(words), step)):
        end = min(len(words), start + chunk_words)
        piece = " ".join(words[start:end]).strip()
        if not piece:
            continue
        digest = hashlib.sha1(f"{source}:{idx}:{piece[:120]}".encode("utf-8")).hexdigest()[:12]
        content_sha1 = hashlib.sha1(piece.encode("utf-8")).hexdigest()
        chunks.append(DocumentChunk(
            chunk_id=f"{source}:{digest}",
            text=piece,
            source=source,
            chunk_index=idx,
            metadata={
                "dates": normalize_dates(piece),
                "times": normalize_times(piece),
                "datetimes": normalize_datetimes(piece),
                "word_count": len(piece.split()),
                "word_start": start,
                "word_end": end,
                "content_sha1": content_sha1,
            },
        ))
        if end >= len(words):
            break
    return chunks
