# AdaptiveRAG v0.3

**Query-aware hybrid retrieval with exact/temporal-first routing, semantic-first conceptual search, confidence-driven fallback, and explainable fusion.**

AdaptiveRAG is a GenAI / Prompt Engineering project that changes the retrieval strategy based on the query instead of always applying the same hybrid search recipe.

## What is different

A conventional hybrid RAG system often runs sparse/lexical and dense/vector retrieval together, then applies a fixed fusion rule. AdaptiveRAG adds a retrieval controller:

- **Exact / temporal query** -> lexical/BM25 + literal/date/time/ID constraints first.
- **Conceptual query** -> semantic retrieval first.
- **Mixed query** -> both channels in parallel.
- Weak primary confidence triggers the opposite channel.
- Fusion combines a **query-derived prior** with **observed retriever confidence** before weighted RRF.
- Relative dates (`yesterday`, `today`, `tomorrow`) and simple ranges (`between`, `before`, `after`, `last N days`) are normalized.
- Date + time pairs are tracked at event level so a date from one event cannot silently combine with a time from another event.
- Every response exposes the route, confidence, threshold, base weights, final weights, fallback decision, and evidence sources.
- Confidence can be calibrated per retrieval backend using a dependency-free Platt calibrator.
- **Hard structural locators** (`page 581`, `PDF page 581`, `pages 581-583`, `page xi`, `section 16.3`, `chapter 16`) are metadata constraints, not relevance hints.
- Bare page numbers prefer the page number visibly printed in the document; explicit `PDF page N` targets the physical PDF page.
- Page extraction preserves physical page, printed page, chapter, section, source name, chunk position, and content hashes.
- Exact-page delivery removes intentional chunk overlap before handing evidence downstream, so page text is not duplicated in the LLM context.
- Missing structural locators fail closed: semantic fallback is suppressed rather than returning a semantically related but wrong page/section.
- Semantic fallback for exact dates/times/IDs/quoted literals is constraint-filtered before fusion.

The database is not the contribution. The controller can later sit above Tencent Cloud VectorDB, Qdrant, pgvector, Elasticsearch, or another sparse/vector store.

## Architecture

```text
                           USER QUERY
                               |
                     exact/temporal parser
                               |
                         Query Router
                /--------------+--------------\
               /               |               \
       exact/temporal      conceptual          mixed
              |                 |               |
        lexical first      semantic first     parallel
              |                 |               |
         confidence          confidence      confidence
           strong?             strong?       both sides
          /      \            /      \           |
       yes       no         yes       no          |
        |         |          |         |          |
    primary    semantic   primary    lexical      |
     only      fallback     only     fallback     |
          \       |          |       /            |
           \------+----------+------+-------------/
                          |
             query prior + channel confidence
                          |
                    weighted RRF
                          |
             hard-constraint / provenance guard
                          |
                   evidence + trace
                          |
                extractive / Ollama LLM
```

## Backends

AdaptiveRAG has three semantic backends:

- `sentence-transformer` - production-oriented dense embeddings (`all-MiniLM-L6-v2` by default).
- `lsa` - fully offline NumPy Latent Semantic Analysis baseline, useful for demos and validation.
- `hashing` - deterministic smoke-test backend. It validates orchestration only and should **not** be used to make semantic-quality claims.

Lexical retrieval is an in-memory BM25-style engine with exact/temporal boosts and strict structural metadata resolution. It is intentionally replaceable.

### Hard locator behavior

Examples:

```text
page 581            -> visible/printed page 581
PDF page 581        -> physical PDF page 581
pages 581-583       -> printed-page range, in reading order
page xi             -> Roman-numeral front matter
section 16.3        -> chunks carrying section 16.3 metadata
```

If a requested page/section does not exist in the indexed document, AdaptiveRAG returns no grounded evidence. It does **not** fall back to a vaguely similar semantic result.

## Install

Core + tests:

```bash
cd adaptive-rag
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

For real SentenceTransformer embeddings:

```bash
uv pip install -e '.[semantic,dev]'
```

## CLI

Fully offline run:

```bash
adaptive-rag sample_data \
  "What happened on 4 September 2026 at 11:42 am?" \
  --backend lsa --generate
```

Relative-time query with an explicit reference clock:

```bash
adaptive-rag sample_data \
  "What happened yesterday at 11:42 am?" \
  --backend lsa --generate \
  --reference-time '2026-09-05T18:30:00+05:30'
```

Use Ollama for the final grounded answer:

```bash
adaptive-rag sample_data \
  "Why must the implementation be different?" \
  --backend lsa --generate --ollama --ollama-model qwen2.5:7b
```

## Web/API demo

```bash
fastapi dev adaptive_rag/api.py
```

Open `http://127.0.0.1:8000`.

The UI lets you select the semantic backend, generation backend, corpus path, and an optional backend-specific confidence calibration JSON.

Build through the API:

```bash
curl -X POST http://127.0.0.1:8000/build \
  -H 'content-type: application/json' \
  -d '{
    "path":"benchmark_corpus",
    "semantic_backend":"lsa",
    "generation_backend":"extractive",
    "calibration_path":"calibration_lsa.json"
  }'
```

Query:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{
    "query":"Why was CVE-2026-1234 on 2 September 2026 important?",
    "k":3,
    "generate":true
  }'
```

## Calibrate confidence

Calibration must be trained for the retrieval backend whose score distribution it will be used with.

```bash
python calibrate.py --backend lsa --out calibration_lsa.json
```

The file stores Platt-scaling parameters for lexical and semantic confidence. The included calibration dataset is only a small project/demo fixture; use a larger held-out dataset before making production claims.

## Benchmark

The benchmark compares:

1. lexical-only retrieval
2. semantic-only retrieval
3. fixed 50/50 static hybrid RRF
4. AdaptiveRAG

```bash
python benchmark.py \
  --backend lsa \
  --calibration calibration_lsa.json \
  --k 3 \
  --json-out validation_benchmark_lsa_v03.json
```

On the included **12-query synthetic validation corpus**, the calibrated offline LSA run produced:

| System | Recall@3 | MRR | nDCG@3 |
|---|---:|---:|---:|
| Lexical only | 1.000 | 1.000 | 1.000 |
| LSA semantic only | 1.000 | 0.806 | 0.855 |
| Static hybrid | 1.000 | 0.958 | 0.969 |
| **AdaptiveRAG** | **1.000** | **1.000** | **1.000** |

Router accuracy on the labeled exact/conceptual/mixed cases was 12/12. The corpus is intentionally small and lexical-friendly, so these numbers are **validation evidence, not a claim that AdaptiveRAG universally beats BM25 or dense RAG**. The real evaluation should be repeated on a larger corpus using the SentenceTransformer or eventual VectorDB backend.

See `VALIDATION.md` for the complete validation record and limitations.

## Ingestion

Supported file types:

- TXT / Markdown / LOG
- Python / JS / TS / HTML / YAML
- JSON
- CSV
- text-based PDF, with printed/PDF page and book-structure metadata

Scanned PDFs need an OCR preprocessing step for text retrieval; OCR is intentionally not hidden inside the core retriever. Physical blank/scanned pages are still preserved as metadata-only page records, so an exact `PDF page N` lookup can resolve the page without inventing text from somewhere else.

## Project structure

```text
adaptive_rag/
  api.py          FastAPI + web UI
  chunking.py     chunking + temporal metadata
  confidence.py   raw confidence + Platt calibration
  engine.py       adaptive retrieval controller
  fusion.py       query/confidence-weighted RRF
  generator.py    extractive + Ollama generation
  ingestion.py    document loaders
  lexical.py      BM25/exact/temporal retrieval
  router.py       query classification
  semantic.py     SentenceTransformer / LSA / hashing
  text.py         token/date/time/range parsing
benchmark.py      four-system retrieval benchmark
calibrate.py      backend-specific confidence calibration
evaluation/       labeled benchmark/calibration cases
benchmark_corpus/ validation corpus
tests/            unit + API/integration tests
```

## Current limitations

- Indexes are in-memory and are rebuilt at startup.
- The benchmark corpus is small and synthetic.
- The LSA backend is an offline validation baseline, not a replacement for a modern embedding model.
- Relative-time interpretation needs a correct reference clock/timezone from the caller.
- Temporal parsing is deterministic and intentionally conservative; it is not a full natural-language date parser.
- No learned cross-encoder reranker yet.
- The API accepts server-local corpus paths, which is appropriate for a local demo but should be sandboxed or replaced with controlled upload/storage APIs before public deployment.
- Text-based PDFs work; scanned PDFs require OCR for textual evidence, though their physical page locations are preserved.

## Next useful upgrades

- Tencent Cloud VectorDB/Qdrant storage adapter with sparse + dense retrieval.
- Cross-encoder reranking after adaptive fusion.
- Larger benchmark with held-out calibration/test splits.
- Persistent index/cache.
- More temporal operators and document-level metadata filters.
- Prompt-injection detector on retrieved content.
