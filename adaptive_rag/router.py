from __future__ import annotations

import re
from datetime import datetime
from .models import QueryAnalysis
from .text import parse_temporal_constraints

QUOTE_RE = re.compile(r'["“](.+?)["”]')
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
ID_RE = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9_-]*\d+[A-Z0-9_-]*|0x[0-9a-fA-F]+|CVE-\d{4}-\d+|[\w.-]+@[\w.-]+)\b", re.IGNORECASE)
CONCEPT_WORDS = {
    "why", "how", "explain", "summarize", "summary", "reason", "meaning", "compare",
    "difference", "idea", "concept", "relationship", "cause", "implication", "overview",
    "purpose", "benefit", "works", "work",
}
EXACT_WORDS = {"exact", "date", "time", "when", "where", "filename", "file", "id", "code", "error", "deadline"}
TEMPORAL_WORDS = {"today", "yesterday", "tomorrow", "before", "after", "since", "until", "between", "during", "last", "past"}

# Supports: page 581, pages 581-583, pp. 581–583, printed page xiv, PDF page 12.
PAGE_RE = re.compile(
    r"\b(?:(pdf|printed|book)\s+)?(?:pages?|p\.?|pp\.?|pg\.?)\s+"
    r"(\d{1,5}|[ivxlcdm]{1,10})"
    r"(?:\s*(?:-|–|—|to|through)\s*(\d{1,5}))?\b",
    re.IGNORECASE,
)
CHAPTER_RE = re.compile(r"\bchapter\s+(\d{1,4})\b", re.IGNORECASE)
SECTION_RE = re.compile(r"(?:\bsection\s+|§\s*)(\d+(?:\.\d+){1,5})\b", re.IGNORECASE)


def _expand_page_match(first: str, last: str | None, cap: int = 100) -> list[str]:
    first = first.lower()
    if not last:
        return [first]
    if not first.isdigit() or not last.isdigit():
        return [first]
    a, b = int(first), int(last)
    if b < a:
        a, b = b, a
    if b - a + 1 > cap:
        # Do not let an accidental huge range explode the query analysis.
        return [str(a), str(b)]
    return [str(n) for n in range(a, b + 1)]


class QueryRouter:
    def analyze(self, query: str, reference_time: datetime | None = None) -> QueryAnalysis:
        temporal = parse_temporal_constraints(query, reference_time=reference_time)
        dates = list(temporal["dates"])
        times = list(temporal["times"])
        quoted = QUOTE_RE.findall(query)
        numbers = NUMBER_RE.findall(query)
        ids = ID_RE.findall(query)

        raw_page_matches = PAGE_RE.findall(query)
        page_labels: list[str] = []
        scopes: set[str] = set()
        for scope, first, last in raw_page_matches:
            if scope:
                scopes.add(scope.lower())
            for label in _expand_page_match(first, last or None):
                if label not in page_labels:
                    page_labels.append(label)
        page_numbers = [int(p) for p in page_labels if p.isdigit()]
        page_scope = None
        if page_labels:
            if "pdf" in scopes:
                page_scope = "pdf"
            elif scopes & {"printed", "book"}:
                page_scope = "printed"
            else:
                page_scope = "any"

        chapter_numbers = list(dict.fromkeys(CHAPTER_RE.findall(query)))
        section_numbers = list(dict.fromkeys(SECTION_RE.findall(query)))
        tokens = set(re.findall(r"[a-z]+", query.lower()))

        exact_evidence = 0.0
        reasons: list[str] = []
        for label, values, weight in [
            ("date", dates, 0.34), ("time", times, 0.28), ("quoted phrase", quoted, 0.32),
            ("identifier", ids, 0.30), ("page reference", page_labels, 0.58),
            ("section reference", section_numbers, 0.58), ("chapter reference", chapter_numbers, 0.52),
            ("number", numbers, 0.08),
        ]:
            if values:
                exact_evidence += weight
                reasons.append(f"contains {label}")
        if (temporal["date_start"] or temporal["date_end"]) and not dates:
            exact_evidence += 0.34
            reasons.append("contains normalized date range")
        exact_kw = tokens & EXACT_WORDS
        if exact_kw:
            exact_evidence += min(0.18, 0.05 * len(exact_kw))
            reasons.append("uses exact/lookup wording")
        temporal_kw = tokens & TEMPORAL_WORDS
        if temporal_kw:
            exact_evidence += min(0.16, 0.04 * len(temporal_kw))
            reasons.append("contains temporal constraint wording")

        concept_hits = tokens & CONCEPT_WORDS
        conceptuality = min(1.0, 0.26 + 0.16 * len(concept_hits)) if concept_hits else 0.20
        if concept_hits:
            reasons.append("asks for conceptual interpretation")

        exactness = min(1.0, exact_evidence)
        structural_locator = bool(page_labels or chapter_numbers or section_numbers)
        if structural_locator:
            route = "exact_temporal"
            reasons.append("structural locator forces exact-first retrieval")
        elif exactness >= 0.32 and conceptuality >= 0.42:
            route = "mixed"
        elif exactness >= 0.34:
            route = "exact_temporal"
        else:
            route = "conceptual"

        if not reasons:
            reasons.append("no strong exact markers; semantic-first is safer")

        if temporal["operator"] == "range":
            reasons.append("normalized temporal range")
        elif temporal["operator"] in {"before", "after"}:
            reasons.append(f"normalized temporal {temporal['operator']} constraint")

        return QueryAnalysis(
            query=query, route=route, exactness=round(exactness, 3),
            conceptuality=round(conceptuality, 3), dates=dates, times=times,
            numbers=numbers, quoted_phrases=quoted, ids=ids,
            page_numbers=page_numbers, page_labels=page_labels, page_scope=page_scope,
            chapter_numbers=chapter_numbers, section_numbers=section_numbers,
            date_start=temporal["date_start"], date_end=temporal["date_end"],
            time_start=temporal["time_start"], time_end=temporal["time_end"],
            temporal_operator=temporal["operator"], reference_time=temporal["reference_time"],
            reasons=reasons,
        )
