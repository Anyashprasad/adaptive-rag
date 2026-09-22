from __future__ import annotations

from datetime import datetime
from .confidence import ConfidencePolicy
from .fusion import adaptive_weights, confidence_adjusted_weights, weighted_rrf_custom
from .generator import ExtractiveGenerator
from .lexical import LexicalIndex
from .models import DocumentChunk, QueryAnalysis, RetrievalResponse, RetrievalTrace, SearchHit
from .router import QueryRouter
from .semantic import SemanticBackend, SentenceTransformerBackend
from .text import within_iso_range


class AdaptiveRAG:
    def __init__(
        self,
        chunks: list[DocumentChunk] | None = None,
        semantic_backend: SemanticBackend | None = None,
        generator=None,
        confidence_policy: ConfidencePolicy | None = None,
    ):
        self.router = QueryRouter()
        self.policy = confidence_policy or ConfidencePolicy()
        self.semantic = semantic_backend or SentenceTransformerBackend()
        self.generator = generator or ExtractiveGenerator()
        self.chunks: list[DocumentChunk] = []
        self.lexical: LexicalIndex | None = None
        if chunks:
            self.build(chunks)

    def build(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index with zero chunks.")
        self.chunks = chunks
        self.lexical = LexicalIndex(chunks)
        self.semantic.build(chunks)

    @staticmethod
    def _has_hard_content_constraints(analysis: QueryAnalysis) -> bool:
        return bool(
            analysis.quoted_phrases or analysis.ids
            or analysis.date_start or analysis.date_end
            or analysis.time_start or analysis.time_end
        )

    @staticmethod
    def _satisfies_hard_content_constraints(hit: SearchHit, analysis: QueryAnalysis) -> bool:
        """Validate brittle literal/temporal constraints against the chunk itself.

        Semantic similarity may rank a conceptually related chunk highly, but it is never
        allowed to substitute a different date, timestamp, ID, or quoted literal.
        """
        chunk = hit.chunk
        text_lower = chunk.text.lower()
        if analysis.quoted_phrases and not any(p.lower() in text_lower for p in analysis.quoted_phrases):
            return False
        if analysis.ids and not any(i.lower() in text_lower for i in analysis.ids):
            return False

        dates = chunk.metadata.get("dates", [])
        times = chunk.metadata.get("times", [])
        datetimes = chunk.metadata.get("datetimes", [])
        want_date = bool(analysis.date_start or analysis.date_end)
        want_time = bool(analysis.time_start or analysis.time_end)

        if want_date and want_time and datetimes:
            if not any(
                "T" in dt
                and within_iso_range(dt.split("T", 1)[0], analysis.date_start, analysis.date_end)
                and within_iso_range(dt.split("T", 1)[1], analysis.time_start, analysis.time_end)
                for dt in datetimes
            ):
                return False
        else:
            if want_date and not any(within_iso_range(d, analysis.date_start, analysis.date_end) for d in dates):
                return False
            if want_time and not any(within_iso_range(t, analysis.time_start, analysis.time_end) for t in times):
                return False
        return True

    @classmethod
    def _constraint_filter(cls, hits: list[SearchHit], analysis: QueryAnalysis) -> list[SearchHit]:
        if not cls._has_hard_content_constraints(analysis):
            return hits
        return [h for h in hits if cls._satisfies_hard_content_constraints(h, analysis)]

    @staticmethod
    def _dedupe_hits(hits: list[SearchHit], k: int | None = None) -> list[SearchHit]:
        """Remove exact duplicate evidence while preserving rank/order.

        Overlap between neighboring chunks is intentional, but identical extracted text
        (common with duplicated PDF layers/pages) should not consume context twice.
        """
        seen: set[tuple[str, str]] = set()
        out: list[SearchHit] = []
        for h in hits:
            digest = str(h.chunk.metadata.get("content_sha1") or "")
            key = (h.chunk.source, digest or " ".join(h.chunk.text.split()))
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
            if k is not None and len(out) >= k:
                break
        return out

    def retrieve(
        self,
        query: str,
        k: int = 6,
        generate: bool = False,
        reference_time: datetime | None = None,
    ) -> RetrievalResponse:
        if not self.lexical:
            raise RuntimeError("Index is empty. Call build() with document chunks first.")
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if k < 1:
            raise ValueError("k must be >= 1")

        analysis = self.router.analyze(query, reference_time=reference_time)
        base_lw, base_sw = adaptive_weights(analysis)
        used_lw, used_sw = base_lw, base_sw
        reasons = list(analysis.reasons)

        # Structural references are hard metadata locators. They bypass confidence routing:
        # either resolve the requested page/chapter/section exactly or return no evidence.
        if analysis.has_structural_locator:
            first = "lexical"
            hits = self.lexical.search(query, analysis, k=max(k, 8))
            hits = self._dedupe_hits(hits)
            fallback = False
            used_lw, used_sw = 1.0, 0.0
            if hits:
                confidence = 1.0
                threshold = 1.0
                reasons.append("structural locator resolved exactly; semantic fallback suppressed")
                if any(not h.chunk.metadata.get("text_extracted", True) for h in hits):
                    reasons.append("target resolved, but at least one page has no extractable text")
            else:
                confidence = 0.0
                threshold = 1.0
                reasons.append("structural locator was not found; semantic fallback suppressed to prevent wrong-page evidence")
            trace = RetrievalTrace(
                route=analysis.route, first_stage=first, fallback_triggered=fallback,
                confidence=confidence, threshold=threshold,
                base_lexical_weight=base_lw, base_semantic_weight=base_sw,
                lexical_weight=used_lw, semantic_weight=used_sw, reasons=reasons,
            )
            answer = self.generator.generate(query, hits) if generate else None
            return RetrievalResponse(query=query, analysis=analysis, trace=trace, hits=hits, answer=answer)

        fallback = False
        if analysis.route == "exact_temporal":
            first = "lexical"
            primary = self._constraint_filter(self.lexical.search(query, analysis, k=max(k, 8)), analysis)
            lconf = self.policy.score("lexical", primary, analysis)
            confidence = lconf
            threshold = self.policy.threshold("lexical", analysis)
            if confidence >= threshold:
                hits = primary[:k]
                used_lw, used_sw = 1.0, 0.0
                reasons.append("lexical confidence cleared adaptive threshold")
            else:
                fallback = True
                secondary = self._constraint_filter(self.semantic.search(query, k=max(k * 4, 32)), analysis)
                sconf = self.policy.score("semantic", secondary, analysis)
                used_lw, used_sw = confidence_adjusted_weights(analysis, lconf, sconf)
                hits = weighted_rrf_custom(primary, secondary, used_lw, used_sw, k=k)
                if self._has_hard_content_constraints(analysis):
                    reasons.append("semantic fallback was constraint-filtered before fusion")
                reasons.append("lexical confidence was weak; semantic fallback + confidence-adjusted fusion activated")

        elif analysis.route == "conceptual":
            first = "semantic"
            primary = self.semantic.search(query, k=max(k, 8))
            sconf = self.policy.score("semantic", primary, analysis)
            confidence = sconf
            threshold = self.policy.threshold("semantic", analysis)
            if confidence >= threshold:
                hits = primary[:k]
                used_lw, used_sw = 0.0, 1.0
                reasons.append("semantic confidence cleared adaptive threshold")
            else:
                fallback = True
                secondary = self.lexical.search(query, analysis, k=max(k, 8))
                lconf = self.policy.score("lexical", secondary, analysis)
                used_lw, used_sw = confidence_adjusted_weights(analysis, lconf, sconf)
                hits = weighted_rrf_custom(secondary, primary, used_lw, used_sw, k=k)
                reasons.append("semantic confidence was weak; lexical fallback + confidence-adjusted fusion activated")

        else:
            first = "parallel"
            lexical = self._constraint_filter(self.lexical.search(query, analysis, k=max(k, 8)), analysis)
            semantic = self._constraint_filter(self.semantic.search(query, k=max(k * 4, 32)), analysis)
            lconf = self.policy.score("lexical", lexical, analysis)
            sconf = self.policy.score("semantic", semantic, analysis)
            confidence = max(lconf, sconf)
            threshold = min(self.policy.threshold("lexical", analysis), self.policy.threshold("semantic", analysis))
            used_lw, used_sw = confidence_adjusted_weights(analysis, lconf, sconf)
            hits = weighted_rrf_custom(lexical, semantic, used_lw, used_sw, k=k)
            if self._has_hard_content_constraints(analysis):
                reasons.append("both channels were hard-constraint filtered before fusion")
            reasons.append("mixed query ran both channels with query- and confidence-dependent fusion")

        hits = self._dedupe_hits(hits, k=k)
        trace = RetrievalTrace(
            route=analysis.route,
            first_stage=first,
            fallback_triggered=fallback,
            confidence=round(confidence, 4),
            threshold=round(threshold, 4),
            base_lexical_weight=base_lw,
            base_semantic_weight=base_sw,
            lexical_weight=used_lw,
            semantic_weight=used_sw,
            reasons=reasons,
        )
        answer = self.generator.generate(query, hits) if generate else None
        return RetrievalResponse(query=query, analysis=analysis, trace=trace, hits=hits, answer=answer)
