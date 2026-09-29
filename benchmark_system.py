#!/usr/bin/env python3
"""
benchmark_system.py
===================
Runs 500 complex structural engineering queries through the Agentic RAG engine
and reports precision, recall, ranking and latency metrics to the terminal,
plus the ML failure-classifier ROC-AUC.

Usage:
    python benchmark_system.py                     # 500 queries, config.json or defaults
    python benchmark_system.py --queries 100       # subset
    python benchmark_system.py --config my.json    # explicit config
    python benchmark_system.py --out results/x.json

Ground truth: each benchmark query carries the ids of its topic cluster
(4-9 records sharing one engineering topic). A chunk is *relevant* if any of
its source record ids belongs to the relevant set.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from rag_engine import AgenticRAG, DEFAULT_CONFIG  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

P_K = 5    # precision cutoff (chunk level)
R_K = 8    # recall cutoff (record level)


def evaluate_query(res: dict, relevant: set[str]) -> dict:
    hits = res["hits"]
    top = hits[:max(P_K, R_K)]

    def is_rel(h):
        return any(sid in relevant for sid in h["src_ids"])

    rel_flags = [is_rel(h) for h in top]
    prec = sum(rel_flags[:P_K]) / P_K
    covered = set()
    for h in top[:R_K]:
        covered.update(sid for sid in h["src_ids"] if sid in relevant)
    recall = len(covered) / len(relevant) if relevant else 0.0
    hit1 = 1.0 if rel_flags and rel_flags[0] else 0.0
    mrr = 0.0
    for i, flag in enumerate(rel_flags, start=1):
        if flag:
            mrr = 1.0 / i
            break
    return {"precision": prec, "recall": recall, "hit1": hit1, "mrr": mrr}


def summarize(rows: list[dict]) -> dict:
    def mean(key):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0
    lat = [r["latency_ms"] for r in rows]
    lat_sorted = sorted(lat)
    p = lambda q: lat_sorted[min(len(lat_sorted) - 1, int(q * len(lat_sorted)))]
    return {
        "n": len(rows),
        "precision@5": round(mean("precision"), 4),
        "recall@8": round(mean("recall"), 4),
        "hit@1": round(mean("hit1"), 4),
        "mrr": round(mean("mrr"), 4),
        "latency_mean_ms": round(statistics.fmean(lat), 2) if lat else 0.0,
        "latency_p50_ms": round(p(0.50), 2),
        "latency_p95_ms": round(p(0.95), 2),
    }


def run_benchmark(config: dict, n_queries: int = 500, label: str = "benchmark",
                  queries_path: Path | None = None) -> dict:
    with open(queries_path or (DATA_DIR / "queries.json"), encoding="utf-8") as f:
        queries = json.load(f)["queries"][:n_queries]

    t0 = time.time()
    engine = AgenticRAG(cfg=config).build_or_load()
    load_s = time.time() - t0

    rows, cls_pairs = [], []
    for q in queries:
        res = engine.query(q["query"])
        m = evaluate_query(res, set(q["relevant_ids"]))
        m["latency_ms"] = res["latency_ms"]
        m["qid"] = q["qid"]
        m["category"] = q["category"]
        rows.append(m)
        for prob in res.get("failure_probabilities", []):
            pass
        for h in res.get("hits", []):
            if "failure_probability" in h:
                cls_pairs.append((h["failure_label"], h["failure_probability"]))

    overall = summarize(rows)
    by_cat = {}
    for cat in sorted({r["category"] for r in rows}):
        by_cat[cat] = summarize([r for r in rows if r["category"] == cat])

    # ML classifier metrics: held-out AUC + AUC over retrieval-time predictions
    try:
        from sklearn.metrics import roc_auc_score
        if cls_pairs:
            ys, ps = zip(*cls_pairs)
            retrieval_auc = float(roc_auc_score(ys, ps)) if len(set(ys)) > 1 else None
        else:
            retrieval_auc = None
    except Exception:
        retrieval_auc = None

    result = {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "index_ev": getattr(engine.index, "_ev", {}),
        "index_key": getattr(engine.index, "key", None),
        "classifier_auc_test": engine.classifier.auc_test,
        "classifier_auc_retrieval": retrieval_auc,
        "overall": overall,
        "by_category": by_cat,
        "engine_load_s": round(load_s, 2),
        "rows": rows,
    }
    return result


def print_report(r: dict) -> None:
    o = r["overall"]
    print("=" * 72)
    print(f"BENCHMARK REPORT - {r['label']}  ({r['timestamp']})")
    print("=" * 72)
    print(f"Queries executed     : {o['n']}")
    print(f"Engine load time     : {r['engine_load_s']} s "
          f"(index build {r['index_ev'].get('build_s', '-')} s, "
          f"{r['index_ev'].get('n_chunks', '-')} chunks, "
          f"vocab {r['index_ev'].get('vocab', '-')}, dims {r['index_ev'].get('dims', '-')})")
    print("-" * 72)
    print("RETRIEVAL METRICS")
    print(f"  precision@5        : {o['precision@5']:.4f}")
    print(f"  recall@8           : {o['recall@8']:.4f}")
    print(f"  hit@1              : {o['hit@1']:.4f}")
    print(f"  MRR                : {o['mrr']:.4f}")
    print("LATENCY")
    print(f"  mean               : {o['latency_mean_ms']:.2f} ms")
    print(f"  p50 / p95          : {o['latency_p50_ms']:.2f} / {o['latency_p95_ms']:.2f} ms")
    print("ML FAILURE CLASSIFIER")
    print(f"  ROC-AUC (held-out) : {r['classifier_auc_test']:.4f}")
    ra = r['classifier_auc_retrieval']
    print(f"  ROC-AUC (retrieved): {ra:.4f}" if ra is not None else
          "  ROC-AUC (retrieved): n/a")
    print("PER CATEGORY")
    for cat, c in r["by_category"].items():
        print(f"  {cat:<18} P@5={c['precision@5']:.3f}  R@8={c['recall@8']:.3f} "
              f"hit@1={c['hit@1']:.3f}  lat_p95={c['latency_p95_ms']:.1f}ms")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--config", type=str, default=None,
                    help="path to config json (default: config.json if present)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--label", type=str, default="benchmark")
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else ROOT / "config.json"
    if cfg_path.exists():
        config = json.loads(cfg_path.read_text())
    else:
        config = DEFAULT_CONFIG

    result = run_benchmark(config, n_queries=args.queries, label=args.label)
    print_report(result)

    out = Path(args.out) if args.out else RESULTS_DIR / "benchmark_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
