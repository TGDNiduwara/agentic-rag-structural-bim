# Agentic RAG for Structural Engineering & BIM Automation

A modular Python RAG framework specialized for **structural engineering codes**,
**seismic load differential equations**, and **BIM/FEA API automation**
(Revit / SAP2000), with hybrid retrieval, an agentic query loop, and an ML
failure-probability classifier. Everything runs locally (no LLM API keys, no
model downloads).

## Quick start

```bash
pip install -r requirements.txt

python generate_synthetic_corpus.py        # Step 0: 50,000-record corpus (~9 s)
python rag_engine.py "your query here"     # Smoke test / demo CLI
python benchmark_system.py                 # Step 3: 500-query benchmark
python optimize_loop.py                    # Step 3b: 100-iteration optimization
```

> `data/` (corpus, queries, labeled nodes) and `index_store/` (cached index
> artifacts) are generated locally and excluded from version control — the
> first script recreates them deterministically (seed 42).

## Project layout

| File | Purpose |
|---|---|
| `generate_synthetic_corpus.py` | Generates `data/corpus.json` (50,000 records: 20k building codes, 8k seismic equations, 12k Revit API, 10k SAP2000 API), `data/structural_nodes.json` (10k ML-labeled structural nodes) and `data/queries.json` (500 benchmark queries with cluster ground truth). Deterministic (seed 42). |
| `rag_engine.py` | The engine: chunking strategies, local dense embeddings (TF-IDF -> TruncatedSVD/LSA), vectorized BM25, hybrid fusion, agentic loop (routing, self-check, query expansion/PRF), `LogisticRegression` failure classifier, disk-cached indexes. |
| `benchmark_system.py` | Runs 500 queries; reports precision@5, recall@8, hit@1, MRR, latency (mean/p50/p95), classifier ROC-AUC, per-category breakdown. Saves `results/benchmark_results.json`. |
| `optimize_loop.py` | Autonomous 100-iteration optimization: chunking sweep -> fusion/BM25/PRF grid -> dense-vector variants -> classifier regularization -> greedy hill climb -> final benchmark. Writes `config.json` (best config) and `results/optimization_log.jsonl`. |
| `config.json` | Winning configuration from the optimization loop (used by default by `benchmark_system.py`). |
| `index_store/` | Cached index artifacts keyed by config hash (dense matrix, BM25 CSC, vocab, chunk table). |
| `results/` | Benchmark JSONs + optimization log. |

## Architecture

```
query ──> intent router (4 categories) ──> hybrid retrieval
                                            ├─ dense: TF-IDF(1-2g, sublinear) -> SVD(384) -> cosine
                                            └─ lexical: Okapi BM25 (CSC, indptr scoring)
                       ──> weighted min-max fusion (alpha) / RRF
                       ──> agentic self-check: raw-cosine floor -> synonym + PRF expansion
                       ──> category rerank -> top-k
                       ──> ML failure classifier (LogisticRegression on 8 structural
                           node features: demand/capacity ratio, drift, axial ratio,
                           ductility, corrosion, zone PGA, connection slip, duration)
                       ──> extractive answer + failure probabilities per hit
```

Chunking strategies: `record`, `recursive` (LangChain `RecursiveCharacterTextSplitter`),
`sentence_window`, `concat` (chunks may span records). Index artifacts are cached
under `index_store/<sha1>` keyed by the index-affecting config subset, so
query-time knobs are swept without rebuilds.

## Benchmark results (500 queries)

| Metric | Baseline | Optimized | Delta |
|---|---|---|---|
| precision@5 | 0.7112 | **0.8200** | +15.3% |
| recall@8 | 0.7350 | **0.8260** | +12.4% |
| hit@1 | 0.7820 | **0.8860** | +13.3% |
| MRR | 0.8399 | **0.9299** | +10.7% |
| latency p95 (ms) | 25.2 | 35.6 | +10 ms (still far under budget) |
| classifier ROC-AUC (held-out) | 0.9311 | 0.9311 | – |

Winning changes found by the optimization loop:
1. **Bigram TF-IDF features** (`ngram_hi: 2`) — the single largest gain (+0.08 score),
   disambiguating terms like "base shear" vs "shear wall".
2. **SVD dims 256 -> 384** for the dense LSA embeddings.
3. **Dense-heavy fusion** (`alpha: 0.85`) — bigram-augmented dense scores dominate BM25.
4. Chunking stayed at record level: records are ~600 chars, so larger recursive
   windows never split them, and sentence-window chunking fragmented context.

## Notes

- Retrieval ground truth = topic clusters (4-9 records sharing one engineering
  topic phrase), so recall is measured over record sets, precision over chunks.
- The "answer" is extractive (top snippets); the pipeline is LLM-pluggable at
  the composition step if a local/global LLM is added later.
- The failure classifier is deliberately insensitive to regularization `C`
  (the synthetic node problem is nearly separable; AUC pinned at ~0.93).
