from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from .models import DocumentChunk, QueryAnalysis, SearchHit
from .text import tokenize, within_iso_range


class LexicalIndex:
    """Small in-memory BM25 + exact/temporal/structural constraints.

    Structural locators (page/chapter/section) are resolved as metadata constraints,
    never merely as BM25 boosts. This prevents semantic-looking false positives such
    as returning page 478 for a request for page 581.
    """

    def __init__(self, chunks: list[DocumentChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.searchable_texts: list[str] = []
        for c in chunks:
            md = c.metadata
            metadata_text = " ".join(str(v) for v in [
                md.get("source_name") or Path(c.source).name,
                md.get("page_heading") or "",
                md.get("section_title") or "",
                md.get("section") or "",
                md.get("chapter") or "",
            ] if v)
            self.searchable_texts.append((c.text + " " + metadata_text).strip())
        self.tokens = [tokenize(t) for t in self.searchable_texts]
        self.tf = [Counter(t) for t in self.tokens]
        self.lengths = [len(t) for t in self.tokens]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter()
        for toks in self.tokens:
            self.df.update(set(toks))
        self.n = len(chunks)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    @staticmethod
    def _norm_label(value) -> str | None:
        if value is None:
            return None
        return str(value).strip().lower() or None


    @staticmethod
    def _deoverlap_page_hits(page_hits: list[SearchHit]) -> list[SearchHit]:
        """Trim intentional chunk overlap for exact page delivery.

        The semantic index keeps overlapping chunks for recall, but when a page is
        requested as a hard locator the downstream LLM should see each page token once.
        """
        out: list[SearchHit] = []
        last_end_by_page: dict[tuple[str, int], int] = {}
        for hit in page_hits:
            md = hit.chunk.metadata
            pdf_page = int(md.get("pdf_page") or 0)
            key = (hit.chunk.source, pdf_page)
            start = int(md.get("word_start") or 0)
            end = int(md.get("word_end") or start)
            previous_end = last_end_by_page.get(key, start)
            trim_words = max(0, previous_end - start)
            words = hit.chunk.text.split()
            if trim_words >= len(words) and words:
                last_end_by_page[key] = max(previous_end, end)
                continue
            if trim_words > 0:
                from .models import DocumentChunk
                new_md = dict(md)
                new_md["retrieval_trimmed_overlap_words"] = trim_words
                new_chunk = DocumentChunk(
                    chunk_id=hit.chunk.chunk_id,
                    text=" ".join(words[trim_words:]),
                    source=hit.chunk.source,
                    chunk_index=hit.chunk.chunk_index,
                    metadata=new_md,
                )
                hit = SearchHit(new_chunk, hit.score, hit.channel, dict(hit.features))
                hit.features["overlap_trimmed"] = trim_words
            last_end_by_page[key] = max(previous_end, end)
            out.append(hit)
        return out

    def _structural_results(self, hits: list[SearchHit], analysis: QueryAnalysis, k: int) -> list[SearchHit] | None:
        """Return strict structural-locator results, or None when no locator exists."""
        if not analysis.has_structural_locator:
            return None

        # Page locators are special: a bare page means the visible printed label first,
        # physical PDF page only if no printed-label match exists anywhere.
        if analysis.page_labels:
            printed = [h for h in hits if h.features.get("printed_page_match")]
            physical = [h for h in hits if h.features.get("pdf_page_match")]
            if analysis.page_scope == "printed":
                candidates = printed
                mode = "printed"
            elif analysis.page_scope == "pdf":
                candidates = physical
                mode = "pdf"
            else:
                candidates = printed if printed else physical
                mode = "printed" if printed else "pdf"
            if not candidates:
                return []

            requested = set(analysis.page_labels)
            by_source: dict[str, list[SearchHit]] = defaultdict(list)
            for h in candidates:
                by_source[h.chunk.source].append(h)

            def source_rank(item: tuple[str, list[SearchHit]]):
                _source, source_hits = item
                labels = set()
                for h in source_hits:
                    if mode == "printed":
                        label = self._norm_label(h.chunk.metadata.get("printed_page"))
                    else:
                        label = self._norm_label(h.chunk.metadata.get("pdf_page"))
                    if label in requested:
                        labels.add(label)
                return (len(labels), max((h.score for h in source_hits), default=0.0))

            best_source, page_hits = max(by_source.items(), key=source_rank)
            del best_source
            page_hits.sort(key=lambda h: (
                int(h.chunk.metadata.get("pdf_page") or 0),
                int(h.chunk.metadata.get("page_chunk_index") or 0),
                h.chunk.chunk_index,
            ))
            # A single exact page should be supplied completely. Large page ranges are
            # bounded to protect downstream context windows while still exceeding normal k.
            cap = 64 if len(requested) > 1 else 32
            return self._deoverlap_page_hits(page_hits[:cap])

        # Explicit sections/chapters are also strict metadata filters. They may be long,
        # so k remains the user's context-budget control rather than returning a whole book.
        if analysis.section_numbers:
            candidates = [h for h in hits if h.features.get("section_match")]
            if not candidates:
                return []
            by_source: dict[str, list[SearchHit]] = defaultdict(list)
            for h in candidates:
                by_source[h.chunk.source].append(h)
            _, selected = max(by_source.items(), key=lambda item: (len(item[1]), max(h.score for h in item[1])))
            selected.sort(key=lambda h: h.chunk.chunk_index)
            return selected[:k]

        if analysis.chapter_numbers:
            candidates = [h for h in hits if h.features.get("chapter_match")]
            if not candidates:
                return []
            by_source: dict[str, list[SearchHit]] = defaultdict(list)
            for h in candidates:
                by_source[h.chunk.source].append(h)
            _, selected = max(by_source.items(), key=lambda item: (len(item[1]), max(h.score for h in item[1])))
            # Keep BM25 order within a chapter because a whole chapter is usually far larger
            # than a context window and the query may contain additional topical terms.
            selected.sort(key=lambda h: h.score, reverse=True)
            return selected[:k]
        return []

    def search(self, query: str, analysis: QueryAnalysis, k: int = 8) -> list[SearchHit]:
        qtokens = tokenize(query)
        qset = set(qtokens)
        hits: list[SearchHit] = []
        query_lower = query.lower().strip()
        requested_page_labels = set(p.lower() for p in analysis.page_labels)
        requested_chapters = set(analysis.chapter_numbers)
        requested_sections = set(analysis.section_numbers)

        for i, chunk in enumerate(self.chunks):
            score = 0.0
            matched_terms: set[str] = set()
            for term in qtokens:
                f = self.tf[i].get(term, 0)
                if not f:
                    continue
                matched_terms.add(term)
                denom = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avgdl, 1e-9))
                score += self._idf(term) * (f * (self.k1 + 1)) / denom

            text_lower = chunk.text.lower()
            quoted_matches = sum(1 for phrase in analysis.quoted_phrases if phrase.lower() in text_lower)
            id_matches = sum(1 for ident in analysis.ids if ident.lower() in text_lower)

            printed_label = self._norm_label(chunk.metadata.get("printed_page"))
            pdf_label = self._norm_label(chunk.metadata.get("pdf_page"))
            printed_page_match = bool(requested_page_labels and printed_label in requested_page_labels)
            pdf_page_match = bool(requested_page_labels and pdf_label in requested_page_labels)
            if analysis.page_scope == "printed":
                page_match = printed_page_match
                page_boost = 12.0 if printed_page_match else 0.0
            elif analysis.page_scope == "pdf":
                page_match = pdf_page_match
                page_boost = 12.0 if pdf_page_match else 0.0
            else:
                page_match = printed_page_match or pdf_page_match
                page_boost = 12.0 if printed_page_match else 5.0 if pdf_page_match else 0.0

            chapter_value = self._norm_label(chunk.metadata.get("chapter"))
            section_value = self._norm_label(chunk.metadata.get("section"))
            chapter_match = bool(requested_chapters and chapter_value in requested_chapters)
            section_match = bool(requested_sections and section_value in requested_sections)
            structure_boost = 8.0 * int(section_match) + 5.0 * int(chapter_match)

            chunk_dates = chunk.metadata.get("dates", [])
            chunk_times = chunk.metadata.get("times", [])
            chunk_datetimes = chunk.metadata.get("datetimes", [])
            date_range_matches = sum(
                1 for d in chunk_dates if within_iso_range(d, analysis.date_start, analysis.date_end)
            ) if (analysis.date_start or analysis.date_end) else 0
            time_range_matches = sum(
                1 for t in chunk_times if within_iso_range(t, analysis.time_start, analysis.time_end)
            ) if (analysis.time_start or analysis.time_end) else 0
            joint_temporal_requested = bool((analysis.date_start or analysis.date_end) and (analysis.time_start or analysis.time_end))
            joint_temporal_matches = 0
            if joint_temporal_requested:
                for dt in chunk_datetimes:
                    if "T" not in dt:
                        continue
                    d, t = dt.split("T", 1)
                    if within_iso_range(d, analysis.date_start, analysis.date_end) and within_iso_range(t, analysis.time_start, analysis.time_end):
                        joint_temporal_matches += 1

            # Do not combine a date from one event and time from another when event-level
            # timestamp metadata is available.
            if joint_temporal_requested and chunk_datetimes:
                temporal_boost = 5.0 * joint_temporal_matches
            else:
                temporal_boost = 2.6 * date_range_matches + 2.1 * time_range_matches

            literal_match = 1 if len(query_lower) < 100 and query_lower and query_lower in text_lower else 0
            boost = (
                2.6 * quoted_matches + temporal_boost + 1.8 * id_matches
                + page_boost + structure_boost + 0.9 * literal_match
            )
            total = score + boost
            # Structural matches must survive even if the query words are absent from text.
            if total <= 0 and not (page_match or chapter_match or section_match):
                continue

            if joint_temporal_requested and chunk_datetimes:
                temporal_constraints = 1
                temporal_satisfied = int(bool(joint_temporal_matches))
            else:
                temporal_constraints = int(bool(analysis.date_start or analysis.date_end)) + int(bool(analysis.time_start or analysis.time_end))
                temporal_satisfied = (
                    int(bool(date_range_matches)) * int(bool(analysis.date_start or analysis.date_end))
                    + int(bool(time_range_matches)) * int(bool(analysis.time_start or analysis.time_end))
                )
            structural_constraints = int(bool(analysis.page_labels)) + int(bool(analysis.chapter_numbers)) + int(bool(analysis.section_numbers))
            exact_constraints = len(analysis.quoted_phrases) + len(analysis.ids) + temporal_constraints + structural_constraints
            satisfied_constraints = (
                quoted_matches + id_matches + temporal_satisfied
                + int(page_match) + int(chapter_match) + int(section_match)
            )
            exact_coverage = satisfied_constraints / exact_constraints if exact_constraints else 0.0
            token_coverage = len(matched_terms) / max(1, len(qset))

            hits.append(SearchHit(
                chunk=chunk,
                score=total,
                channel="lexical",
                features={
                    "bm25": round(score, 4),
                    "exact_boost": round(boost, 4),
                    "date_match": bool(date_range_matches),
                    "time_match": bool(time_range_matches),
                    "datetime_match": bool(joint_temporal_matches),
                    "id_match": bool(id_matches),
                    "page_match": bool(page_match),
                    "printed_page_match": bool(printed_page_match),
                    "pdf_page_match": bool(pdf_page_match),
                    "chapter_match": bool(chapter_match),
                    "section_match": bool(section_match),
                    "quoted_match": bool(quoted_matches),
                    "literal_match": bool(literal_match),
                    "exact_coverage": round(exact_coverage, 4),
                    "token_coverage": round(token_coverage, 4),
                    "text_extracted": bool(chunk.metadata.get("text_extracted", True)),
                },
            ))
        hits.sort(key=lambda h: h.score, reverse=True)

        structural = self._structural_results(hits, analysis, k)
        if structural is not None:
            return structural
        return hits[:k]
