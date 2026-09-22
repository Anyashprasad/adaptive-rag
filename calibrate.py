from __future__ import annotations

import argparse
import json
from pathlib import Path
from adaptive_rag.confidence import ConfidencePolicy
from adaptive_rag.ingestion import ingest_path
from adaptive_rag.lexical import LexicalIndex
from adaptive_rag.router import QueryRouter
from adaptive_rag.semantic import HashingSemanticBackend, LSASemanticBackend, SentenceTransformerBackend


def is_correct(hits, gold_source):
    if gold_source is None:
        return 0
    return int(bool(hits) and Path(hits[0].chunk.source).name == gold_source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hashing", "lsa", "sentence-transformer"], default="hashing")
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()
    root = Path(__file__).parent
    cases = json.loads((root / "evaluation/calibration_cases.json").read_text())
    chunks = ingest_path(root / "benchmark_corpus", chunk_words=120, overlap_words=20)
    semantic = HashingSemanticBackend() if args.backend == "hashing" else LSASemanticBackend() if args.backend == "lsa" else SentenceTransformerBackend()
    semantic.build(chunks)
    lexical = LexicalIndex(chunks)
    router = QueryRouter(); policy = ConfidencePolicy()
    scores = {"lexical": [], "semantic": []}; labels = {"lexical": [], "semantic": []}
    for case in cases:
        a = router.analyze(case["query"])
        lex = lexical.search(case["query"], a, k=3); sem = semantic.search(case["query"], k=3)
        scores["lexical"].append(policy.raw_score("lexical", lex, a)); labels["lexical"].append(is_correct(lex, case["gold_source"]))
        scores["semantic"].append(policy.raw_score("semantic", sem, a)); labels["semantic"].append(is_correct(sem, case["gold_source"]))
    for ch in ("lexical", "semantic"):
        policy.fit_channel(ch, scores[ch], labels[ch])
    policy.save(args.out)
    print(json.dumps({
        "backend": args.backend,
        "out": args.out,
        "calibrators": {k: {"a": v.a, "b": v.b} for k, v in policy.calibrators.items()},
        "label_counts": {k: {"positive": sum(labels[k]), "total": len(labels[k])} for k in labels},
    }, indent=2))


if __name__ == "__main__":
    main()
