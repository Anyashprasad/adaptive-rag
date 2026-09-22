from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from adaptive_rag.confidence import ConfidencePolicy
from adaptive_rag.engine import AdaptiveRAG
from adaptive_rag.fusion import static_rrf
from adaptive_rag.ingestion import ingest_path
from adaptive_rag.lexical import LexicalIndex
from adaptive_rag.router import QueryRouter
from adaptive_rag.semantic import HashingSemanticBackend, LSASemanticBackend, SentenceTransformerBackend

SYSTEM_NAMES = ["lexical_only", "semantic_only", "static_hybrid", "adaptive_rag"]


def relevant_rank(hits, gold_source: str) -> int | None:
    for i, hit in enumerate(hits, 1):
        if Path(hit.chunk.source).name == gold_source:
            return i
    return None


def score_case(rank: int | None) -> tuple[float, float, float]:
    if rank is None:
        return 0.0, 0.0, 0.0
    return 1.0, 1.0 / rank, 1.0 / math.log2(rank + 1)


def empty_bucket():
    return {name: {"recall": [], "mrr": [], "ndcg": [], "latency_ms": []} for name in SYSTEM_NAMES}


def summarize(vals, k: int, include_latency: bool = True):
    out = {
        f"recall@{k}": statistics.mean(vals["recall"]) if vals["recall"] else 0.0,
        "mrr": statistics.mean(vals["mrr"]) if vals["mrr"] else 0.0,
        f"ndcg@{k}": statistics.mean(vals["ndcg"]) if vals["ndcg"] else 0.0,
    }
    if include_latency:
        out.update({
            "mean_latency_ms": statistics.mean(vals["latency_ms"]) if vals["latency_ms"] else 0.0,
            "p50_latency_ms": statistics.median(vals["latency_ms"]) if vals["latency_ms"] else 0.0,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hashing", "lsa", "sentence-transformer"], default="hashing")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--calibration", default=None)
    args = ap.parse_args()

    root = Path(__file__).parent
    cases = json.loads((root / "evaluation/cases.json").read_text())
    chunks = ingest_path(root / "benchmark_corpus", chunk_words=120, overlap_words=20)
    backend = HashingSemanticBackend() if args.backend == "hashing" else LSASemanticBackend() if args.backend == "lsa" else SentenceTransformerBackend()
    policy = ConfidencePolicy(args.calibration) if args.calibration else ConfidencePolicy()
    rag = AdaptiveRAG(chunks, semantic_backend=backend, confidence_policy=policy)
    lexical = LexicalIndex(chunks)
    router = QueryRouter()

    systems = empty_bucket()
    by_category = {category: empty_bucket() for category in sorted({c["category"] for c in cases})}
    route_hits = 0
    fallbacks = 0
    details = []

    for case in cases:
        q = case["query"]
        analysis = router.analyze(q)
        route_hits += int(analysis.route == case["expected_route"])

        t = time.perf_counter(); lex = lexical.search(q, analysis, k=args.k); lex_ms=(time.perf_counter()-t)*1000
        t = time.perf_counter(); sem = backend.search(q, k=args.k); sem_ms=(time.perf_counter()-t)*1000
        t = time.perf_counter(); hyb = static_rrf(lexical.search(q, analysis, k=max(args.k,8)), backend.search(q, k=max(args.k,8)), k=args.k); hyb_ms=(time.perf_counter()-t)*1000
        t = time.perf_counter(); ada = rag.retrieve(q, k=args.k); ada_ms=(time.perf_counter()-t)*1000
        fallbacks += int(ada.trace.fallback_triggered)

        result_map = {
            "lexical_only": (lex, lex_ms),
            "semantic_only": (sem, sem_ms),
            "static_hybrid": (hyb, hyb_ms),
            "adaptive_rag": (ada.hits, ada_ms),
        }
        case_detail = {"query": q, "category": case["category"], "expected_route": case["expected_route"], "got_route": analysis.route, "ranks": {}}
        for name, (hits, latency_ms) in result_map.items():
            rank = relevant_rank(hits, case["gold_source"])
            recall, mrr, ndcg = score_case(rank)
            for bucket in (systems[name], by_category[case["category"]][name]):
                bucket["recall"].append(recall); bucket["mrr"].append(mrr); bucket["ndcg"].append(ndcg); bucket["latency_ms"].append(latency_ms)
            case_detail["ranks"][name] = rank
        details.append(case_detail)

    report = {
        "backend": args.backend,
        "calibration": args.calibration or "none",
        "k": args.k,
        "cases": len(cases),
        "router_accuracy": route_hits / len(cases),
        "fallback_rate": fallbacks / len(cases),
        "systems": {name: summarize(vals, args.k) for name, vals in systems.items()},
        "by_category": {
            category: {name: summarize(vals, args.k, include_latency=False) for name, vals in category_data.items()}
            for category, category_data in by_category.items()
        },
        "case_details": details,
    }

    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
