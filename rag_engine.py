#!/usr/bin/env python3
"""
rag_engine.py
=============
Agentic RAG engine for Structural Engineering, extreme loadings and BIM/SAP API
automation.

Pipeline
--------
  1. Chunking    : "record"          -> one chunk per corpus record
                   "recursive"       -> LangChain RecursiveCharacterTextSplitter
                                        (chunk_size / chunk_overlap)
                   "sentence_window" -> sliding groups of sentences per record
                   "concat"          -> records concatenated per category, then
                                        split; chunks may span records
  2. Dense local embeddings : TF-IDF (sublinear, smooth idf, l2) -> TruncatedSVD
                   (LSA) -> l2. Fully local, no model downloads.
  3. Lexical     : vectorized BM25 (Okapi) on a CSR term-count matrix.
  4. Hybrid fuse : weighted min-max fusion (alpha) or Reciprocal Rank Fusion.
  5. Agentic loop: intent routing -> retrieval -> self-check -> query expansion
                   (domain synonyms + pseudo-relevance feedback) -> category
                   rerank -> compose extractive answer.
  6. ML sub-module: scikit-learn LogisticRegression classifies retrieved
                   structural nodes by failure probability (ROC-AUC reported).

Index artifacts are cached under index_store/<hash>/ keyed by the
index-affecting part of the config (chunking + dense params), so the optimizer
can sweep query-time knobs (BM25 k1/b, fusion alpha, PRF, classifier C) without
rebuilding.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:  # LangChain integration (used for the "recursive" strategy)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except Exception:  # pragma: no cover
    HAS_LANGCHAIN = False

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index_store"
SEED = 42

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "chunking": {"strategy": "record", "chunk_size": 900, "chunk_overlap": 150},
    "dense": {"enabled": True, "sublinear_tf": True, "min_df": 2,
              "ngram_hi": 1, "svd_dims": 256},
    "bm25": {"enabled": True, "k1": 1.5, "b": 0.75},
    "fusion": {"method": "weighted", "alpha": 0.55},
    "rerank": {"category_boost": 0.0, "top_k": 8, "candidate_pool": 150},
    "agent": {"self_check": True, "score_floor": 0.20, "max_rounds": 1,
              "prf_docs": 0, "prf_terms": 0, "prf_weight": 0.35},
    "classifier": {"C": 1.0, "class_weight": None, "max_iter": 300},
}

FEATURE_KEYS = ["demand_capacity_ratio", "drift_ratio", "axial_load_ratio",
                "curvature_ductility", "corrosion_pct", "zone_pga",
                "connection_slip_mm", "load_duration_factor"]

SYNONYMS = {
    "earthquake": ["seismic", "ground motion"],
    "formula": ["equation", "derivation"],
    "equation": ["derivation", "formula"],
    "requirement": ["provisions", "requirements"],
    "code": ["provisions", "standard"],
    "automate": ["API", "method"],
    "script": ["API", "example"],
    "python": ["comtypes", "example"],
    "c#": ["C#", "snippet"],
    "deflection": ["drift", "displacement"],
    "load": ["load", "demand"],
}

STOPWORDS = set("""a an and are as at be by for from has have in into is it its
of on or per shall should that the their then these this to under upon was were
will with within without can may must shall note example section using use
value values given where which when all any each than also""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def ngram_tokens(tokens: list[str], n: int) -> list[str]:
    if n <= 1:
        return tokens
    return tokens + ["_".join(tokens[i:i + k])
                     for k in range(2, n + 1)
                     for i in range(len(tokens) - k + 1)]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def chunk_records(records: list[dict], cfg: dict) -> list[dict]:
    """Return chunks: {text, src_ids (record ids), category}."""
    strat = cfg["chunking"]["strategy"]
    size = int(cfg["chunking"].get("chunk_size", 900))
    overlap = int(cfg["chunking"].get("chunk_overlap", 150))
    chunks: list[dict] = []

    def add(text, src, cat):
        text = text.strip()
        if len(text) >= 40:
            chunks.append({"text": text, "src_ids": list(src), "category": cat})

    if strat == "record":
        for r in records:
            add(f"{r['title']}. {r['content']}", [r["id"]], r["category"])

    elif strat == "recursive":
        if HAS_LANGCHAIN:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size, chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", " ", ""])
            for r in records:
                for piece in splitter.split_text(f"{r['title']}. {r['content']}"):
                    add(piece, [r["id"]], r["category"])
        else:  # fallback sliding window
            for r in records:
                text = f"{r['title']}. {r['content']}"
                step = max(1, size - overlap)
                for i in range(0, len(text), step):
                    add(text[i:i + size], [r["id"]], r["category"])

    elif strat == "sentence_window":
        for r in records:
            sents = [r["title"]] + _split_sentences(r["content"])
            k = max(2, size // 220)          # sentences per window
            stride = max(1, k - max(1, overlap // 220))
            for i in range(0, len(sents), stride):
                win = " ".join(sents[i:i + k])
                if len(win) >= 40:
                    add(win, [r["id"]], r["category"])
                if i + k >= len(sents):
                    break

    elif strat == "concat":
        # concatenate records per category; chunks may span record boundaries
        splitter = None
        if HAS_LANGCHAIN:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size, chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", " ", ""])
        by_cat: dict[str, list[dict]] = {}
        for r in records:
            by_cat.setdefault(r["category"], []).append(r)
        for cat, recs in by_cat.items():
            pieces, offsets, pos = [], [], 0
            for r in recs:
                text = f"{r['title']}. {r['content']}\n"
                offsets.append((pos, pos + len(text) - 1, r["id"]))
                pieces.append(text)
                pos += len(text)
            corpus = "".join(pieces)
            spans = []
            if splitter is not None:
                spans = splitter.split_text(corpus)
                # locate each split piece by scanning sequentially
                cursor = 0
                located = []
                for piece in spans:
                    start = corpus.find(piece[:60], cursor)
                    if start < 0:
                        start = cursor
                    end = start + len(piece) - 1
                    located.append((start, end))
                    cursor = max(cursor, start + 1)
            else:
                step = max(1, size - overlap)
                located = [(i, i + size - 1) for i in range(0, len(corpus), step)]
            for (a, b) in located:
                src = [rid for (s, e, rid) in offsets if s <= b and e >= a]
                if src:
                    add(corpus[a:b + 1], src, cat)

    else:
        raise ValueError(f"unknown chunking strategy: {strat}")
    return chunks


# ---------------------------------------------------------------------------
# Vectorized BM25
# ---------------------------------------------------------------------------

class BM25Index:
    """Vectorized Okapi BM25 over a term-major (CSC) count matrix."""

    def __init__(self, count_csc: sp.csc_matrix, idf: np.ndarray,
                 doc_denom: np.ndarray, k1: float = 1.5, b: float = 0.75):
        self.X = count_csc.tocsc()      # (n_docs, n_terms) term-major access
        self.idf = idf
        self.doc_denom = doc_denom      # k1 * (1 - b + b * len/avg) per doc
        self.k1, self.b = k1, b
        self.len_norm = doc_denom / max(k1, 1e-9)  # provisional until build sets it

    def set_params(self, k1: float, b: float):
        self.k1, self.b = k1, b
        self.doc_denom = self.k1 * self.len_norm

    def scores(self, query_ids: np.ndarray) -> np.ndarray:
        n_docs = self.X.shape[0]
        scores = np.zeros(n_docs)
        if query_ids.size == 0:
            return scores
        indptr, indices, data = self.X.indptr, self.X.indices, self.X.data
        for j in query_ids:
            s, e = indptr[j], indptr[j + 1]
            if s == e:
                continue
            docs = indices[s:e]
            tf = data[s:e]
            contrib = tf * (self.k1 + 1.0) / (tf + self.doc_denom[docs])
            scores[docs] += self.idf[j] * contrib
        return scores


# ---------------------------------------------------------------------------
# Hybrid index (dense LSA + BM25) with on-disk cache
# ---------------------------------------------------------------------------

def index_key(cfg: dict) -> str:
    subset = {
        "chunking": cfg["chunking"],
        "dense": cfg["dense"],
        "seed": SEED,
    }
    h = hashlib.sha1(json.dumps(subset, sort_keys=True).encode()).hexdigest()[:12]
    return h


class HybridIndex:
    """Builds / loads and searches the hybrid index for one config."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.key = index_key(cfg)
        self.chunks: list[dict] = []
        self.count: sp.csr_matrix | None = None
        self.bm25: BM25Index | None = None
        self.dense: np.ndarray | None = None          # (n_chunks, dims) l2
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.svd_components: np.ndarray | None = None  # (dims, vocab)

    # ---------------- build ----------------
    def build(self, records: list[dict], verbose: bool = True):
        t0 = time.time()
        self.chunks = chunk_records(records, self.cfg)
        dcfg = self.cfg["dense"]
        ngram_hi = int(dcfg.get("ngram_hi", 1))
        min_df = int(dcfg.get("min_df", 2))
        sublinear = bool(dcfg.get("sublinear_tf", True))

        tok_lists = [ngram_tokens(tokenize(c["text"]), ngram_hi) for c in self.chunks]
        raw_tok = [tokenize(c["text"]) for c in self.chunks]  # BM25 uses unigrams

        # document frequencies (per tf-idf n-gram)
        from collections import Counter
        df_counter: Counter = Counter()
        for toks in tok_lists:
            df_counter.update(set(toks))
        kept = [(t, df) for t, df in sorted(df_counter.items()) if df >= min_df]
        vocab = {t: i for i, (t, df) in enumerate(kept)}
        self.vocab = vocab
        n_docs, n_terms = len(tok_lists), len(vocab)

        # vectorized CSR construction: ones -> COO -> CSR with summed duplicates
        import itertools
        lens = np.fromiter((len(t) for t in tok_lists), dtype=np.int64, count=n_docs)
        total = int(lens.sum())
        rows = np.repeat(np.arange(n_docs, dtype=np.int64), lens)
        cols = np.fromiter((vocab.get(t, -1) for t in itertools.chain.from_iterable(tok_lists)),
                           dtype=np.int64, count=total)
        keep = cols >= 0
        counts = sp.coo_matrix((np.ones(int(keep.sum()), dtype=np.float64),
                                (rows[keep], cols[keep])),
                               shape=(n_docs, n_terms)).tocsr()
        counts.sum_duplicates()

        # BM25 idf (Robertson) over unigram vocab subset
        uni_vocab = {t: i for t, i in vocab.items() if "_" not in t or ngram_hi == 1}
        # df for unigrams computed directly
        uni_df = np.zeros(n_terms)
        for t, j in vocab.items():
            if ngram_hi == 1 or "_" not in t:
                uni_df[j] = df_counter[t]
        bm25_idf = np.log(1.0 + (n_docs - uni_df + 0.5) / (uni_df + 0.5))
        avg_len = max(1.0, lens.mean())
        len_norm = lens / avg_len
        k1 = float(self.cfg["bm25"].get("k1", 1.5))
        b = float(self.cfg["bm25"].get("b", 0.75))
        self.bm25 = BM25Index(counts.tocsc(), bm25_idf, k1 * (1.0 - b + b * len_norm), k1, b)
        self.bm25.len_norm = len_norm

        # dense: TF-IDF -> SVD
        df_col = counts.getnnz(axis=0).astype(np.float64)
        idf = np.log((1.0 + n_docs) / (1.0 + df_col)) + 1.0
        tf = counts.copy()
        if sublinear:
            tf.data = 1.0 + np.log(np.maximum(tf.data, 1e-12))
        tf = tf.multiply(idf).tocsr()
        norms = np.sqrt(tf.multiply(tf).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        tf = sp.diags(1.0 / norms) @ tf
        self.idf = idf
        self.count = counts

        dims = int(dcfg.get("svd_dims", 256))
        dims = max(2, min(dims, n_terms - 1, n_docs - 1))
        svd = TruncatedSVD(n_components=dims, algorithm="randomized",
                           n_iter=5, random_state=SEED)
        dense = svd.fit_transform(tf)
        dn = np.linalg.norm(dense, axis=1, keepdims=True)
        dn[dn == 0] = 1.0
        self.dense = dense / dn
        self.svd_components = svd.components_

        self._ev = {"build_s": time.time() - t0, "n_chunks": n_docs,
                    "vocab": n_terms, "dims": int(dims)}
        if verbose:
            print(f"[index] built {n_docs:,} chunks, vocab {n_terms:,}, "
                  f"dims {dims} in {self._ev['build_s']:.1f}s")

    # ---------------- cache ----------------
    def save(self):
        d = INDEX_DIR / self.key
        d.mkdir(parents=True, exist_ok=True)
        sp.save_npz(d / "counts.npz", self.count)
        np.savez_compressed(d / "dense.npz", dense=self.dense,
                            idf=self.idf, svd=self.svd_components,
                            bm25_idf=self.bm25.idf, len_norm=self.bm25.len_norm)
        with open(d / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f, protocol=4)
        with open(d / "vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f, protocol=4)
        with open(d / "info.json", "w") as f:
            json.dump({"cfg_subset": {"chunking": self.cfg["chunking"],
                                      "dense": self.cfg["dense"]},
                       "ev": self._ev}, f)

    @classmethod
    def load(cls, cfg: dict) -> "HybridIndex | None":
        d = INDEX_DIR / index_key(cfg)
        if not (d / "info.json").exists():
            return None
        obj = cls(cfg)
        info = json.loads((d / "info.json").read_text())
        # only the index-affecting subset must match; query-time params are free
        if info.get("cfg_subset") != {"chunking": cfg["chunking"], "dense": cfg["dense"]}:
            return None
        obj.count = sp.load_npz(d / "counts.npz").tocsr()
        z = np.load(d / "dense.npz")
        obj.dense = z["dense"]; obj.idf = z["idf"]; obj.svd_components = z["svd"]
        with open(d / "chunks.pkl", "rb") as f:
            obj.chunks = pickle.load(f)
        with open(d / "vocab.pkl", "rb") as f:
            obj.vocab = pickle.load(f)
        k1 = float(cfg["bm25"].get("k1", 1.5))
        b = float(cfg["bm25"].get("b", 0.75))
        obj.bm25 = BM25Index(obj.count.tocsc(), z["bm25_idf"],
                             k1 * (1.0 - b + b * z["len_norm"]), k1, b)
        obj.bm25.len_norm = z["len_norm"]
        obj._ev = info["ev"]
        return obj

    # ---------------- query-side transforms ----------------
    def _query_tfidf(self, text: str):
        toks = ngram_tokens(tokenize(text), int(self.cfg["dense"].get("ngram_hi", 1)))
        ids, cnts = [], []
        local: dict[int, int] = {}
        for t in toks:
            j = self.vocab.get(t)
            if j is not None:
                local[j] = local.get(j, 0) + 1
        if not local:
            return None
        items = sorted(local.items())
        cols = np.array([i for i, _ in items])
        vals = np.array([float(c) for _, c in items])
        if self.cfg["dense"].get("sublinear_tf", True):
            vals = 1.0 + np.log(vals)
        vals = vals * self.idf[cols]
        vec = sp.csr_matrix((vals, (np.zeros_like(cols), cols)),
                            shape=(1, len(self.vocab)))
        n = np.sqrt(vec.multiply(vec).sum())
        if n > 0:
            vec = vec.multiply(1.0 / n).tocsr()
        return vec, cols

    def dense_scores(self, text: str) -> np.ndarray:
        qt = self._query_tfidf(text)
        if qt is None or self.dense is None:
            return np.full(len(self.chunks), -1.0)
        vec, _ = qt
        q = np.asarray(vec.dot(self.svd_components.T)).ravel()
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        return self.dense @ q

    def bm25_scores(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        ids = np.array(sorted({self.vocab[t] for t in toks if t in self.vocab}))
        return self.bm25.scores(ids) if self.bm25 else np.zeros(len(self.chunks))


# ---------------------------------------------------------------------------
# ML sub-module: structural failure-probability classifier
# ---------------------------------------------------------------------------

class FailureClassifier:
    """Logistic regression on structural node features -> failure probability."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=float(cfg["classifier"].get("C", 1.0)),
            class_weight=cfg["classifier"].get("class_weight"),
            max_iter=int(cfg["classifier"].get("max_iter", 300)),
            random_state=SEED)
        self.auc_test = None
        self.trained = False

    def train(self, nodes: list[dict]):
        X = np.array([[n["features"][k] for k in FEATURE_KEYS] for n in nodes])
        y = np.array([int(n["failure_label"]) for n in nodes])
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(y))
        cut = int(0.7 * len(y))
        tr, te = idx[:cut], idx[cut:]
        Xtr = self.scaler.fit_transform(X[tr])
        self.model.fit(Xtr, y[tr])
        Xte = self.scaler.transform(X[te])
        p = self.model.predict_proba(Xte)[:, 1]
        self.auc_test = float(roc_auc_score(y[te], p))
        self.trained = True
        return self.auc_test

    def predict(self, features: dict) -> float:
        x = np.array([[features.get(k, 0.0) for k in FEATURE_KEYS]])
        return float(self.model.predict_proba(self.scaler.transform(x))[0, 1])

    def predict_batch(self, rows) -> list[float]:
        X = np.array([[f[k] for k in FEATURE_KEYS] if isinstance(f, dict) else list(f)
                      for f in rows], dtype=float)
        return self.model.predict_proba(self.scaler.transform(X))[:, 1].tolist()


# ---------------------------------------------------------------------------
# Agentic RAG facade
# ---------------------------------------------------------------------------

CATEGORY_HINTS = {
    "building_code": ["code", "provision", "requirement", "standard", "shall",
                      "combination", "factor", "phi", "detailing", "inspection"],
    "seismic_equation": ["equation", "derivation", "spectrum", "modal", "damping",
                         "wn", "rad/s", "eom", "newmark", "integrate", "formula"],
    "revit_api": ["revit", "autodesk", "family", "transaction", "elementid",
                  "collector", "pyrevit", "bim", "parameter", "wall"],
    "sap2000_api": ["sap2000", "oapi", "sapmodel", "comtypes", "frameobj",
                    "analysis", "jointdispl", "csi", "kN", "combo"],
}


class AgenticRAG:
    def __init__(self, cfg: dict | None = None, records: list[dict] | None = None,
                 use_cache: bool = True, verbose: bool = False):
        self.cfg = json.loads(json.dumps(cfg)) if cfg else json.loads(json.dumps(DEFAULT_CONFIG))
        self.verbose = verbose
        self.records = records
        self.index: HybridIndex | None = None
        self.classifier: FailureClassifier | None = None
        self.node_records: list[dict] = []          # corpus records w/ node features
        self.rec_by_id: dict[str, dict] = {}
        self._load_data()

    # ---------------- data ----------------
    def _load_data(self):
        if self.records is None:
            with open(DATA_DIR / "corpus.json", encoding="utf-8") as f:
                self.records = json.load(f)["records"]
        self.rec_by_id = {r["id"]: r for r in self.records}
        self.node_records = [r for r in self.records
                             if "node_features" in r.get("meta", {})]
        with open(DATA_DIR / "structural_nodes.json", encoding="utf-8") as f:
            self.nodes = json.load(f)["nodes"]

    def build_or_load(self):
        if use_cache_idx(self.cfg) and (ix := HybridIndex.load(self.cfg)):
            self.index = ix
            if self.verbose:
                print(f"[index] loaded cached {index_key(self.cfg)} "
                      f"({len(ix.chunks):,} chunks)")
        else:
            self.index = HybridIndex(self.cfg)
            self.index.build(self.records, verbose=True)
            self.index.save()
        self.classifier = FailureClassifier(self.cfg)
        train_nodes = self.nodes + [
            {"features": r["meta"]["node_features"],
             "failure_label": r["meta"]["failure_label"]}
            for r in self.node_records]
        self.classifier.train(train_nodes)
        return self

    # ---------------- agent steps ----------------
    def route(self, query: str) -> tuple[str, float]:
        toks = set(tokenize(query))
        scores = {c: sum(1 for h in hs if h in toks or h in query.lower())
                  for c, hs in CATEGORY_HINTS.items()}
        best = max(scores, key=lambda c: scores[c])
        conf = scores[best] / max(1.0, max(scores.values()) if max(scores.values()) else 1.0)
        return (best, min(1.0, conf)) if scores[best] > 0 else ("any", 0.0)

    def _retrieve_raw(self, query: str) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (candidate_idx, fused_scores, raw_max_dense_cosine)."""
        pool = int(self.cfg["rerank"].get("candidate_pool", 150))
        dcfg, bcfg, fcfg = self.cfg["dense"], self.cfg["bm25"], self.cfg["fusion"]
        n = len(self.index.chunks)
        ds = self.index.dense_scores(query) if dcfg.get("enabled", True) else np.full(n, -1.0)
        bs = self.index.bm25_scores(query) if bcfg.get("enabled", True) else np.zeros(n)
        raw_max = float(ds.max()) if dcfg.get("enabled", True) else 0.0

        def top_idx(arr, k):
            k = min(k, len(arr))
            return np.argpartition(-arr, k - 1)[:k]

        cand = np.unique(np.concatenate([
            top_idx(ds, pool) if dcfg.get("enabled", True) else np.array([], dtype=int),
            top_idx(bs, pool) if bcfg.get("enabled", True) else np.array([], dtype=int)]))
        method = fcfg.get("method", "weighted")
        alpha = float(fcfg.get("alpha", 0.55))
        if method == "rrf":
            k = 60.0
            d_rank = np.argsort(np.argsort(-ds[cand]))
            b_rank = np.argsort(np.argsort(-bs[cand]))
            fused = 1.0 / (k + d_rank + 1) + 1.0 / (k + b_rank + 1)
        else:
            def mm(x):
                lo, hi = x.min(), x.max()
                return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
            fused = alpha * mm(ds[cand]) + (1.0 - alpha) * mm(bs[cand])
        return cand, fused, raw_max

    def _expand(self, query: str, cand: np.ndarray, fused: np.ndarray) -> str:
        words = set(query.lower().split())
        for w in list(words):
            if w in SYNONYMS:
                words.update(SYNONYMS[w])
        prf_docs = int(self.cfg["agent"].get("prf_docs", 0))
        prf_terms = int(self.cfg["agent"].get("prf_terms", 0))
        if prf_docs > 0 and prf_terms > 0 and len(cand) > 0:
            top = cand[np.argsort(-fused)[:prf_docs]]
            from collections import Counter
            cnt = Counter()
            for i in top:
                cnt.update(tokenize(self.index.chunks[i]["text"]))
            for t, _ in cnt.most_common(prf_terms):
                words.add(t)
        return query + " " + " ".join(sorted(words - set(query.lower().split())))

    def query(self, query: str, top_k: int | None = None) -> dict:
        t0 = time.time()
        if top_k is None:
            top_k = int(self.cfg["rerank"].get("top_k", 8))
        intent, conf = self.route(query)
        boost = float(self.cfg["rerank"].get("category_boost", 0.0))
        cand, fused, raw_max = self._retrieve_raw(query)

        rounds = 0
        max_rounds = int(self.cfg["agent"].get("max_rounds", 1)) \
            if self.cfg["agent"].get("self_check", True) else 0
        while (rounds < max_rounds and len(cand) and
               raw_max < float(self.cfg["agent"].get("score_floor", 0.20))):
            query = self._expand(query, cand, fused)
            cand, fused, raw_max = self._retrieve_raw(query)
            rounds += 1
        self._last_expansion_rounds = rounds

        order = np.argsort(-fused)
        hits = []
        for pos in order[: int(self.cfg["rerank"].get("candidate_pool", 150))]:
            i = cand[pos]
            ch = self.index.chunks[i]
            s = float(fused[pos])
            if boost > 0 and intent == ch["category"]:
                s *= (1.0 + boost)
            hits.append((s, i, ch))
        hits.sort(key=lambda x: -x[0])

        out_hits, preds = [], []
        top = hits[:top_k]
        # one batched prediction for every node-tagged record among the top hits
        flat = [(pos, rid) for pos, (s, i, ch) in enumerate(top) for rid in ch["src_ids"]
                if rid in self.rec_by_id
                and "node_features" in self.rec_by_id[rid].get("meta", {})]
        probs_map = {}
        if flat and self.classifier:
            Xb = [[self.rec_by_id[rid]["meta"]["node_features"][k] for k in FEATURE_KEYS]
                  for _, rid in flat]
            pb = self.classifier.predict_batch(Xb)
            probs_map = {(pos, rid): float(p) for (pos, rid), p in zip(flat, pb)}
        for pos, (s, i, ch) in enumerate(top):
            h = {"chunk": i, "score": round(s, 4), "category": ch["category"],
                 "src_ids": ch["src_ids"], "text": ch["text"]}
            for rid in ch["src_ids"]:
                if (pos, rid) in probs_map:
                    p = probs_map[(pos, rid)]
                    h["failure_probability"] = round(p, 4)
                    h["failure_label"] = self.rec_by_id[rid]["meta"]["failure_label"]
                    preds.append(p)
            out_hits.append(h)

        answer = " | ".join(h["text"][:220] for h in out_hits[:3])
        return {
            "query": query,
            "intent": intent,
            "expansion_rounds": rounds,
            "hits": out_hits,
            "failure_probabilities": preds,
            "answer": answer,
            "latency_ms": round((time.time() - t0) * 1000.0, 2),
        }


def use_cache_idx(cfg: dict) -> bool:
    return True


# ---------------------------------------------------------------------------
# Demo CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "seismic base shear design requirements for SDC D"
    engine = AgenticRAG(verbose=True).build_or_load()
    print(f"Classifier held-out AUC: {engine.classifier.auc_test:.4f}")
    res = engine.query(q)
    print(json.dumps({k: v for k, v in res.items() if k != "hits"},
                     indent=2)[:900])
    for h in res["hits"][:5]:
        print(f"  [{h['score']:.3f}] {h['category']:<17} "
              f"{h['src_ids']} failP={h.get('failure_probability', '-')}")
