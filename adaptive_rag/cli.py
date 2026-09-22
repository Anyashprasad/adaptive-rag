from __future__ import annotations

import argparse
import json
from datetime import datetime
from .confidence import ConfidencePolicy
from .engine import AdaptiveRAG
from .generator import ExtractiveGenerator, OllamaGenerator
from .ingestion import ingest_path
from .semantic import HashingSemanticBackend, LSASemanticBackend, SentenceTransformerBackend


def main():
    parser = argparse.ArgumentParser(description="AdaptiveRAG demo")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("query", help="Question to retrieve evidence for")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--backend", choices=["sentence-transformer", "lsa", "hashing"], default="sentence-transformer")
    parser.add_argument("--fast", action="store_true", help="Alias for --backend hashing")
    parser.add_argument("--generate", action="store_true", help="Generate a grounded answer")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for generation instead of extractive evidence")
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--reference-time", default=None, help="ISO datetime used to resolve today/yesterday/tomorrow")
    parser.add_argument("--calibration", default=None, help="Path to backend-specific confidence calibration JSON")
    args = parser.parse_args()

    chunks = ingest_path(args.path)
    backend_name = "hashing" if args.fast else args.backend
    semantic = HashingSemanticBackend() if backend_name == "hashing" else LSASemanticBackend() if backend_name == "lsa" else SentenceTransformerBackend()
    generator = OllamaGenerator(model=args.ollama_model) if args.ollama else ExtractiveGenerator()
    policy = ConfidencePolicy(args.calibration) if args.calibration else ConfidencePolicy()
    rag = AdaptiveRAG(chunks, semantic_backend=semantic, generator=generator, confidence_policy=policy)
    reference_time = datetime.fromisoformat(args.reference_time) if args.reference_time else None
    result = rag.retrieve(args.query, k=args.k, generate=args.generate, reference_time=reference_time)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
