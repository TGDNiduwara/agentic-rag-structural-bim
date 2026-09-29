#!/usr/bin/env python3
"""
optimize_loop.py
================
Autonomous optimization loop: runs 100 benchmark-evaluated iterations against
the Agentic RAG engine, edits configuration (chunking strategy, vector-search
parameters, fusion, ML regularization), re-evaluates, and keeps the best.

Phases
------
  A  (1-11)    chunking strategy sweep (each variant = one index build)
  B  (12-47)   fusion / BM25 / PRF parameter grid on the top-3 chunkings
               (indexes cached in memory + on disk -> no rebuilds)
  C  (48-55)   dense embedding variants (SVD dims, n-grams, min_df, sublinear)
  D  (56-67)   ML classifier regularization (C, class_weight) + rerank knobs
  E  (68-98)   greedy coordinate-descent hill climb around the best config
  F  (99)      final full 500-query benchmark of the best config ->
               config.json + results/benchmark_final.json

Objective
---------
  score = 0.40*recall@8 + 0.25*precision@5 + 0.15*hit@1 + 0.15*MRR
          + 0.05*classifier_AUC - 0.10*max(0, p95_latency_ms - 120)/120

Validation: 350 of the 500 seeded-shuffled queries (held deterministic);
the final iteration reports the full 500-query benchmark.
"""
from __future__ import annotations

import copy
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from rag_engine import AgenticRAG, DEFAULT_CONFIG, HybridIndex, index_key  # noqa: E402
from benchmark_system import evaluate_query, run_benchmark, print_report  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LOG_PATH = RESULTS_DIR / "optimization_log.jsonl"
ITERS = 100
VAL_QUERIES = 350
IDX_CACHE_MAX = 3


def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Optimizer:
    def __init__(self, iters: int = ITERS, val_n: int = VAL_QUERIES, seed: int = 42):
        self.iters = iters
        self.rng = random.Random(seed)
        self.records, self.nodes, self.queries = self._load_data()
        self.val = self.queries[:val_n]
        self.idx_cache: dict[str, HybridIndex] = {}
        self.cache_order: list[str] = []
        self.best = {"score": -1e9, "cfg": None, "iter": -1}
        self.history: list[dict] = []

    # ------------------------------------------------------------------ data
    @staticmethod
    def _load_data():
        with open(DATA_DIR / "corpus.json", encoding="utf-8") as f:
            records = json.load(f)["records"]
        with open(DATA_DIR / "structural_nodes.json", encoding="utf-8") as f:
            nodes = json.load(f)["nodes"]
        with open(DATA_DIR / "queries.json", encoding="utf-8") as f:
            queries = json.load(f)["queries"]
        return records, nodes, queries

    # ----------------------------------------------------------------- index
    def get_engine(self, cfg: dict) -> AgenticRAG:
        eng = AgenticRAG(cfg=cfg, records=self.records)
        key = index_key(cfg)
        ix = self.idx_cache.get(key)
        if ix is None:
            ix = HybridIndex.load(cfg)
            if ix is None:
                t0 = time.time()
                ix = HybridIndex(cfg)
                ix.build(self.records, verbose=False)
                ix.save()
                print(f"    [build] {key} {ix._ev['n_chunks']:,} chunks "
                      f"in {time.time() - t0:.1f}s", flush=True)
            self.idx_cache[key] = ix
            self.cache_order.append(key)
            while len(self.idx_cache) > IDX_CACHE_MAX:
                old = self.cache_order.pop(0)
                self.idx_cache.pop(old, None)
        # query-time BM25 params may differ from the cached index build
        ix.bm25.set_params(float(cfg["bm25"]["k1"]), float(cfg["bm25"]["b"]))
        eng.index = ix
        from rag_engine import FailureClassifier
        eng.classifier = FailureClassifier(cfg)
        train_nodes = self.nodes + [
            {"features": r["meta"]["node_features"],
             "failure_label": r["meta"]["failure_label"]}
            for r in eng.node_records]
        eng.classifier.train(train_nodes)
        return eng

    # ------------------------------------------------------------------ eval
    def evaluate(self, cfg: dict, n_queries: int | None = None) -> dict:
        qs = self.val if n_queries is None else self.queries[:n_queries]
        t0 = time.time()
        eng = self.get_engine(cfg)
        build_s = time.time() - t0
        t1 = time.time()
        rows, cls_pairs = [], []
        for q in qs:
            res = eng.query(q["query"])
            m = evaluate_query(res, set(q["relevant_ids"]))
            m["latency_ms"] = res["latency_ms"]
            rows.append(m)
            for h in res.get("hits", []):
                if "failure_probability" in h:
                    cls_pairs.append((h["failure_label"], h["failure_probability"]))
        n = len(rows)
        avg = lambda k: sum(r[k] for r in rows) / n
        lat = sorted(r["latency_ms"] for r in rows)
        p95 = lat[min(n - 1, int(0.95 * n))]
        try:
            from sklearn.metrics import roc_auc_score
            auc = (float(roc_auc_score(*zip(*cls_pairs)))
                   if len(cls_pairs) > 2 and len({y for y, _ in cls_pairs}) > 1
                   else eng.classifier.auc_test)
        except Exception:
            auc = eng.classifier.auc_test
        metrics = {
            "precision@5": round(avg("precision"), 4),
            "recall@8": round(avg("recall"), 4),
            "hit@1": round(avg("hit1"), 4),
            "mrr": round(avg("mrr"), 4),
            "p95_ms": round(p95, 2),
            "auc": round(auc, 4),
        }
        score = (0.40 * metrics["recall@8"] + 0.25 * metrics["precision@5"]
                 + 0.15 * metrics["hit@1"] + 0.15 * metrics["mrr"]
                 + 0.05 * metrics["auc"]
                 - 0.10 * max(0.0, p95 - 120.0) / 120.0)
        metrics["eval_s"] = round(time.time() - t1, 2)
        return {"score": round(score, 5), "metrics": metrics,
                "build_s": round(build_s, 2)}

    # -------------------------------------------------------------- schedule
    def phase_patches(self) -> list[tuple[str, dict]]:
        """Phases A-D: fixed, ordered patches."""
        plan: list[tuple[str, dict]] = []
        ch = lambda s, sz, ov: {"chunking": {"strategy": s, "chunk_size": sz,
                                             "chunk_overlap": ov}}
        # Phase A - chunking sweep (baseline "record" already evaluated as iter 0)
        for patch in [ch("recursive", 600, 0), ch("recursive", 600, 150),
                      ch("recursive", 900, 150), ch("recursive", 1200, 200),
                      ch("recursive", 1600, 250), ch("recursive", 2400, 300),
                      ch("sentence_window", 900, 150), ch("sentence_window", 1600, 200),
                      ch("concat", 900, 150), ch("concat", 1200, 200),
                      ch("concat", 1600, 300)]:
            plan.append(("A-chunk", patch))
        # Phase B - fusion/BM25/PRF grid per top-3 chunking (filled at runtime)
        return plan

    def phase_b_patches(self, top_cfgs: list[dict]) -> list[tuple[str, dict]]:
        out = []
        for base in top_cfgs:
            tag = base["chunking"]["strategy"] + str(base["chunking"]["chunk_size"])
            grid = [
                {"fusion": {"alpha": a}} for a in (0.35, 0.45, 0.65, 0.75)
            ] + [
                {"fusion": {"method": "rrf"}},
                {"bm25": {"k1": 1.2, "b": 0.4}}, {"bm25": {"k1": 1.2, "b": 0.6}},
                {"bm25": {"k1": 2.0, "b": 0.6}}, {"bm25": {"k1": 2.0, "b": 0.75}},
                {"agent": {"prf_docs": 3, "prf_terms": 5}},
                {"agent": {"prf_docs": 5, "prf_terms": 8}},
                {"rerank": {"candidate_pool": 300}},
                {"bm25": {"k1": 1.2, "b": 0.6}, "fusion": {"alpha": 0.45}},
            ]
            for g in grid:
                out.append((f"B-{tag}", deep_merge(base, g)))
        return out

    def phase_c_patches(self, best_cfg: dict) -> list[tuple[str, dict]]:
        d = lambda **kw: {"dense": kw}
        return [
            ("C-dense128", deep_merge(best_cfg, d(svd_dims=128))),
            ("C-dense384", deep_merge(best_cfg, d(svd_dims=384))),
            ("C-dense512", deep_merge(best_cfg, d(svd_dims=512))),
            ("C-ngram2", deep_merge(best_cfg, d(ngram_hi=2))),
            ("C-mindf1", deep_merge(best_cfg, d(min_df=1))),
            ("C-mindf5", deep_merge(best_cfg, d(min_df=5))),
            ("C-lintf", deep_merge(best_cfg, d(sublinear_tf=False))),
            ("C-ngram2-384", deep_merge(best_cfg, d(ngram_hi=2, svd_dims=384))),
        ]

    def phase_d_patches(self, best_cfg: dict) -> list[tuple[str, dict]]:
        out = []
        for C in (0.05, 0.1, 0.3, 3.0, 10.0, 30.0):
            out.append((f"D-C{C}", deep_merge(best_cfg, {"classifier": {"C": C}})))
        out.append(("D-balanced", deep_merge(best_cfg,
                    {"classifier": {"class_weight": "balanced"}})))
        for b in (0.05, 0.10, 0.20):
            out.append((f"D-boost{b}", deep_merge(best_cfg,
                        {"rerank": {"category_boost": b}})))
        out.append(("D-topk6", deep_merge(best_cfg, {"rerank": {"top_k": 6}})))
        out.append(("D-topk10", deep_merge(best_cfg, {"rerank": {"top_k": 10}})))
        return out

    def hill_candidates(self, best_cfg: dict, rng: random.Random) -> list[dict]:
        """One greedy sweep of coordinate moves around the best config."""
        c = best_cfg
        cand = []
        a = c["fusion"]["alpha"]
        for da in (-0.07, -0.03, 0.03, 0.07):
            na = min(0.9, max(0.1, a + da))
            if abs(na - a) > 1e-6:
                cand.append({"fusion": {"alpha": round(na, 3)}})
        k1, b = c["bm25"]["k1"], c["bm25"]["b"]
        for dk in (-0.2, 0.2):
            cand.append({"bm25": {"k1": round(min(2.5, max(0.8, k1 + dk)), 2)}})
        for db in (-0.08, 0.08):
            cand.append({"bm25": {"b": round(min(0.9, max(0.2, b + db)), 2)}})
        ag = c["agent"]
        if ag["prf_docs"] > 0:
            cand.append({"agent": {"prf_docs": max(0, ag["prf_docs"] - 2)}})
            cand.append({"agent": {"prf_docs": ag["prf_docs"] + 2,
                                   "prf_terms": ag["prf_terms"] + 2}})
        else:
            cand.append({"agent": {"prf_docs": 3, "prf_terms": 5}})
        if ag["score_floor"] > 0:
            cand.append({"agent": {"self_check": False}})
        rr = c["rerank"]
        for db in (-0.05, 0.05, 0.10):
            nb = round(min(0.5, max(0.0, rr["category_boost"] + db)), 2)
            cand.append({"rerank": {"category_boost": nb}})
        cand.append({"rerank": {"candidate_pool":
                                300 if rr["candidate_pool"] < 300 else 100}})
        cl = c["classifier"]
        for fc in (0.5, 2.0):
            cand.append({"classifier": {"C": round(cl["C"] * fc, 3)}})
        # occasional dense rebuild moves (expensive -> limited)
        if rng.random() < 0.25:
            dims = c["dense"]["svd_dims"]
            cand.append({"dense": {"svd_dims": int(dims * rng.choice([0.66, 1.5]))}})
        return cand

    # ------------------------------------------------------------------- run
    def run(self):
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(LOG_PATH, "a", encoding="utf-8")

        def log(it, phase, cfg, res):
            entry = {"iter": it, "phase": phase, "score": res["score"],
                     "metrics": res["metrics"], "build_s": res["build_s"],
                     "config": cfg, "is_best": res["score"] > self.best["score"]}
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()

        def consider(it, phase, cfg, res):
            mark = ""
            if res["score"] > self.best["score"]:
                self.best = {"score": res["score"], "cfg": copy.deepcopy(cfg),
                             "iter": it}
                mark = "  <-- NEW BEST"
            log(it, phase, cfg, res)
            m = res["metrics"]
            print(f"[{it:>3}] {phase:<12} score={res['score']:.4f} "
                  f"P@5={m['precision@5']:.3f} R@8={m['recall@8']:.3f} "
                  f"hit@1={m['hit@1']:.3f} MRR={m['mrr']:.3f} "
                  f"p95={m['p95_ms']:.0f}ms AUC={m['auc']:.3f}{mark}", flush=True)

        it = 0
        # ---- iter 0: baseline ----
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        res = self.evaluate(cfg)
        consider(it, "baseline", cfg, res)
        it += 1

        # ---- Phase A: chunking sweep ----
        chunk_scores = [(copy.deepcopy(cfg), res["score"])]
        for patch in self.phase_patches():
            if it >= 12:
                break
            cfg_i = deep_merge(DEFAULT_CONFIG, patch[1])
            try:
                res = self.evaluate(cfg_i)
            except Exception as e:
                print(f"[{it}] {patch[0]} FAILED: {e}", flush=True)
                res = {"score": -1e9, "metrics": {}, "build_s": 0}
                log(it, patch[0], cfg_i, res)
            consider(it, patch[0], cfg_i, res)
            chunk_scores.append((copy.deepcopy(cfg_i), res["score"]))
            it += 1

        # ---- Phase B: fusion/BM25/PRF grid on top-3 chunkings ----
        top3 = [c for c, _ in sorted(chunk_scores, key=lambda x: -x[1])[:3]]
        for name, cfg_i in self.phase_b_patches(top3):
            if it >= 48:
                break
            try:
                res = self.evaluate(cfg_i)
                consider(it, name, cfg_i, res)
            except Exception as e:
                print(f"[{it}] {name} FAILED: {e}", flush=True)
                log(it, name, cfg_i, {"score": -1e9, "metrics": {}, "build_s": 0})
            it += 1

        # ---- Phase C: dense variants (rebuilds) ----
        for name, cfg_i in self.phase_c_patches(self.best["cfg"]):
            if it >= 56:
                break
            try:
                res = self.evaluate(cfg_i)
                consider(it, name, cfg_i, res)
            except Exception as e:
                print(f"[{it}] {name} FAILED: {e}", flush=True)
                log(it, name, cfg_i, {"score": -1e9, "metrics": {}, "build_s": 0})
            it += 1

        # ---- Phase D: classifier / rerank ----
        for name, cfg_i in self.phase_d_patches(self.best["cfg"]):
            if it >= 68:
                break
            try:
                res = self.evaluate(cfg_i)
                consider(it, name, cfg_i, res)
            except Exception as e:
                print(f"[{it}] {name} FAILED: {e}", flush=True)
                log(it, name, cfg_i, {"score": -1e9, "metrics": {}, "build_s": 0})
            it += 1

        # ---- Phase E: greedy hill climb to fill up to iter 98 ----
        while it <= 98:
            base = self.best["cfg"]
            cands = self.hill_candidates(base, self.rng)
            progressed = False
            for patch in cands:
                if it > 98:
                    break
                cfg_i = deep_merge(base, patch)
                try:
                    res = self.evaluate(cfg_i)
                    consider(it, "E-hill", cfg_i, res)
                    progressed = True
                except Exception as e:
                    print(f"[{it}] E-hill FAILED: {e}", flush=True)
                    log(it, "E-hill", cfg_i, {"score": -1e9, "metrics": {},
                                              "build_s": 0})
                it += 1
            if not progressed:
                break

        # ---- Iter 99 (final): full 500-query benchmark of best config ----
        best_cfg = self.best["cfg"]
        (ROOT / "config.json").write_text(json.dumps(best_cfg, indent=2))
        print(f"\nBest config (iter {self.best['iter']}, "
              f"val score {self.best['score']:.4f}) written to config.json",
              flush=True)
        final = run_benchmark(best_cfg, n_queries=500, label="final-optimized")
        with open(RESULTS_DIR / "benchmark_final.json", "w", encoding="utf-8") as f:
            json.dump(final, f, indent=1)
        print_report(final)
        log(99, "F-final", best_cfg,
            {"score": self.best["score"],
             "metrics": final["overall"], "build_s": 0})
        log_f.close()

        # prune stale index caches except the best + default
        keep = {index_key(best_cfg), index_key(DEFAULT_CONFIG)}
        for d in (ROOT / "index_store").iterdir():
            if d.is_dir() and d.name not in keep:
                shutil.rmtree(d, ignore_errors=True)
        print("Optimization loop complete. Full log: "
              f"{LOG_PATH}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    Optimizer().run()
    print(f"Total wall time: {(time.time() - t0) / 60:.1f} min")
