# AdaptiveRAG v0.3 Validation Record

Validation date: 2026-09-05

This record separates behavior actually executed in the repository from external model paths that were unavailable in the sandbox.

## 1. Automated regression suite

Command:

```bash
python -m pytest -q
```

Result:

```text
30 passed
```

Coverage includes:

- exact/temporal, conceptual, and mixed routing
- explicit and relative date/time normalization
- date ranges and event-level date+time pairing
- confidence calibration persistence
- JSON/CSV/text/PDF ingestion
- physical PDF page metadata and visible printed-page metadata
- bare printed-page vs explicit physical-PDF-page disambiguation
- Roman-numeral printed pages
- page ranges
- exact page delivery in reading order
- exact page overlap removal without content loss
- same page number across multiple source documents
- missing-page fail-closed behavior (no semantic fallback)
- metadata-only/empty PDF page resolution
- exact timestamp constraint filtering during semantic fallback
- retrieval engine, FastAPI, extractive evidence, and Ollama HTTP contract tests

## 2. Real 801-page PDF regression

The uploaded 801-page *Deep Learning* book was ingested through the real PDF loader and indexed with the deterministic hashing semantic backend so the test measured the retrieval controller rather than model-download availability.

Observed ingestion/index properties:

- indexed chunks: **2,274**
- visible printed page `581` maps to physical PDF page **597**
- physical PDF page `581` carries visible printed page **565**
- page 597 metadata resolves to chapter **16**, section **16.3**, `Sampling from Graphical Models`

Exact query:

```text
find and summarize whatever is on page 581
```

Result:

- route: `exact_temporal`
- first stage: `lexical`
- semantic fallback: **false**
- returned page: physical **597**, printed **581**
- returned chunks: **4**, in page reading order

The exact-page delivery path trims the configured 35-word chunk overlap from subsequent page chunks. Concatenating the returned evidence produced **473 words**, exactly equal to the normalized 473-word text extracted directly from physical PDF page 597. This verifies complete page delivery with no missing words and no duplicated overlap.

Additional real-document probes:

| Query | Result |
|---|---|
| `summarize PDF page 581` | physical page 581 / printed page 565 |
| `summarize pages 581-583` | printed pages 581-583 / physical pages 597-599, ordered |
| `what is on page xi?` | Roman-numeral printed page xi / physical page 13 |
| `summarize page 9999` | zero hits; semantic fallback suppressed |
| `summarize section 16.3` | only chunks carrying section 16.3 metadata |

This regression directly covers the failure that originally returned Chapter 12 material for a page-581 request.

## 3. Hard-constraint safety behavior

Structural locators are now treated as filters rather than ranking hints. A nonexistent page/section returns no evidence. The semantic channel is not allowed to reinterpret it.

For non-structural exact constraints, semantic fallback is still available, but candidate chunks are filtered before fusion when the query contains brittle evidence such as:

- normalized dates or date ranges
- times/timestamps
- exact identifiers such as CVEs/error codes
- quoted literal phrases

A regression case with a conceptually similar event on the wrong date confirms that the wrong timestamp cannot survive the fallback filter.

## 4. Retrieval benchmark

Command:

```bash
python benchmark.py --backend lsa \
  --calibration calibration_lsa.json \
  --json-out validation_benchmark_lsa_v03.json
```

Dataset: 10 small corpus documents, 12 labeled queries across exact, conceptual, and mixed categories.

| System | Recall@3 | MRR | nDCG@3 |
|---|---:|---:|---:|
| Lexical only | 1.000 | 1.000 | 1.000 |
| LSA semantic only | 1.000 | 0.806 | 0.855 |
| Static 50/50 hybrid RRF | 1.000 | 0.958 | 0.969 |
| AdaptiveRAG | 1.000 | 1.000 | 1.000 |

Router accuracy: **12/12**. Included calibration fallback rate: **5/12 (41.7%)**.

These are correctness/demo numbers from a tiny lexical-friendly corpus, not a universal performance claim.

## 5. Packaging/API/generation boundary

The retriever intentionally does not own summarization quality. `ExtractiveGenerator` remains a debugging/evidence surface; a real grounded summary is delegated to the configured LLM (for example Ollama).

The retrieval layer's contract is now: deliver the correct, provenance-rich evidence in the correct order and fail closed when hard constraints cannot be satisfied.

The FastAPI build/query/health flow and Ollama-compatible HTTP contract remain covered by automated tests. A live Uvicorn v0.3 server was also built against the real 801-page PDF using the hashing backend: `/build` indexed 2,274 chunks; `page 581` returned four hits from physical page 597; `PDF page 581` returned four hits from physical page 581; and `page 9999` returned zero hits with no semantic fallback.

A `0.3.0` wheel was built with `pip wheel`, installed into a separate target directory, imported from that wheel snapshot, and used to execute a retrieval smoke query successfully.

## 6. External dense-model limitation

A real `sentence-transformers/all-MiniLM-L6-v2` quality run still requires model weights on the target machine. The sentence-transformer path is implemented, but no claim about MiniLM retrieval quality is made from the sandbox-only LSA/hashing validation.

Recommended target-machine run:

```bash
uv pip install -e '.[semantic,dev]'
python calibrate.py --backend sentence-transformer --out calibration_minilm.json
python benchmark.py --backend sentence-transformer --k 3 \
  --calibration calibration_minilm.json \
  --json-out benchmark_minilm.json
```

## Validation conclusion

v0.3 closes the real-world page-retrieval failure mode: page boundaries and visible page labels survive ingestion; hard locators cannot fall through into unrelated semantic evidence; exact-page chunks are returned in reading order with overlap removed; and exact temporal/literal constraints remain enforced during semantic fallback.
