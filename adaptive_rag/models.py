from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Route = Literal["exact_temporal", "conceptual", "mixed"]


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryAnalysis:
    query: str
    route: Route
    exactness: float
    conceptuality: float
    dates: list[str] = field(default_factory=list)
    times: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    quoted_phrases: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    # page_numbers is retained for API/backward compatibility. page_labels is the
    # canonical representation and also supports Roman-numeral front matter.
    page_numbers: list[int] = field(default_factory=list)
    page_labels: list[str] = field(default_factory=list)
    page_scope: str | None = None
    chapter_numbers: list[str] = field(default_factory=list)
    section_numbers: list[str] = field(default_factory=list)
    date_start: str | None = None
    date_end: str | None = None
    time_start: str | None = None
    time_end: str | None = None
    temporal_operator: str | None = None
    reference_time: str | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def has_structural_locator(self) -> bool:
        return bool(self.page_labels or self.chapter_numbers or self.section_numbers)


@dataclass
class SearchHit:
    chunk: DocumentChunk
    score: float
    channel: str
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "text": self.chunk.text,
            "source": self.chunk.source,
            "chunk_index": self.chunk.chunk_index,
            "metadata": self.chunk.metadata,
            "score": round(float(self.score), 6),
            "channel": self.channel,
            "features": self.features,
        }


@dataclass
class RetrievalTrace:
    route: Route
    first_stage: str
    fallback_triggered: bool
    confidence: float
    threshold: float
    base_lexical_weight: float
    base_semantic_weight: float
    lexical_weight: float
    semantic_weight: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class RetrievalResponse:
    query: str
    analysis: QueryAnalysis
    trace: RetrievalTrace
    hits: list[SearchHit]
    answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "analysis": asdict(self.analysis),
            "trace": asdict(self.trace),
            "hits": [h.to_dict() for h in self.hits],
            "answer": self.answer,
        }
