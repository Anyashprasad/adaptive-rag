from __future__ import annotations

from collections import defaultdict
from .models import QueryAnalysis, SearchHit


def adaptive_weights(analysis: QueryAnalysis) -> tuple[float, float]:
    # Exact-heavy queries pull toward lexical; conceptual queries pull toward semantic.
    lex = 0.30 + 0.58 * analysis.exactness - 0.12 * analysis.conceptuality
    lex = max(0.18, min(0.82, lex))
    return round(lex, 3), round(1.0 - lex, 3)


def confidence_adjusted_weights(
    analysis: QueryAnalysis, lexical_confidence: float, semantic_confidence: float
) -> tuple[float, float]:
    """Blend query intent with observed channel reliability.

    Query intent chooses the prior; confidence acts as evidence. A small floor prevents one
    retriever from disappearing because confidence is itself uncertain.
    """
    lw, sw = adaptive_weights(analysis)
    lrel = (0.15 + 0.85 * max(0.0, min(1.0, lexical_confidence))) ** 2
    srel = (0.15 + 0.85 * max(0.0, min(1.0, semantic_confidence))) ** 2
    la, sa = lw * lrel, sw * srel
    total = la + sa
    if total <= 0:
        return lw, sw
    return round(la / total, 3), round(sa / total, 3)


def weighted_rrf_custom(
    lexical: list[SearchHit],
    semantic: list[SearchHit],
    lexical_weight: float,
    semantic_weight: float,
    k: int = 8,
    rrf_k: int = 60,
) -> list[SearchHit]:
    scores: dict[str, float] = defaultdict(float)
    hits_by_id: dict[str, SearchHit] = {}
    channels: dict[str, set[str]] = defaultdict(set)

    for weight, results, channel in [
        (lexical_weight, lexical, "lexical"),
        (semantic_weight, semantic, "semantic"),
    ]:
        for rank, hit in enumerate(results, start=1):
            cid = hit.chunk.chunk_id
            scores[cid] += weight / (rrf_k + rank)
            hits_by_id.setdefault(cid, hit)
            channels[cid].add(channel)

    ordered = sorted(scores, key=scores.get, reverse=True)[:k]
    out: list[SearchHit] = []
    max_score = max((scores[c] for c in ordered), default=1.0)
    for cid in ordered:
        base = hits_by_id[cid]
        out.append(SearchHit(
            chunk=base.chunk,
            score=scores[cid] / max_score if max_score else 0.0,
            channel="+".join(sorted(channels[cid])),
            features={
                "rrf": round(scores[cid], 6),
                "lexical_weight": round(lexical_weight, 3),
                "semantic_weight": round(semantic_weight, 3),
            },
        ))
    return out


def weighted_rrf(
    lexical: list[SearchHit], semantic: list[SearchHit], analysis: QueryAnalysis,
    k: int = 8, rrf_k: int = 60,
) -> list[SearchHit]:
    lw, sw = adaptive_weights(analysis)
    return weighted_rrf_custom(lexical, semantic, lw, sw, k=k, rrf_k=rrf_k)


def static_rrf(
    lexical: list[SearchHit], semantic: list[SearchHit],
    k: int = 8, lexical_weight: float = 0.5, semantic_weight: float = 0.5, rrf_k: int = 60,
) -> list[SearchHit]:
    return weighted_rrf_custom(lexical, semantic, lexical_weight, semantic_weight, k=k, rrf_k=rrf_k)
