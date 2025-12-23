from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from src.utils import clean_text, env

_TFIDF_AVAILABLE = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    _TFIDF_AVAILABLE = False

_EMB_AVAILABLE = True
try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    _EMB_AVAILABLE = False


def ensure_store_dir() -> str:
    store = env("KX_STORE_DIR", "data/kx_store")
    os.makedirs(store, exist_ok=True)
    return store


def _meta_path() -> str:
    return os.path.join(ensure_store_dir(), "kx_meta.json")


def _backend_path() -> str:
    return os.path.join(ensure_store_dir(), "kx_backend.json")


def _faiss_index_path() -> str:
    return os.path.join(ensure_store_dir(), "kx_faiss.index")


def _tfidf_path() -> str:
    return os.path.join(ensure_store_dir(), "kx_tfidf.pkl")


@dataclass
class KXBackend:
    name: str
    dim: int | None = None


def kx_index_exists() -> bool:
    return os.path.exists(_meta_path()) and (
        os.path.exists(_faiss_index_path()) or os.path.exists(_tfidf_path())
    )


def _save_meta(meta: List[Dict[str, Any]]) -> None:
    with open(_meta_path(), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_meta() -> List[Dict[str, Any]]:
    with open(_meta_path(), "r", encoding="utf-8") as f:
        return json.load(f)


def _save_backend(b: KXBackend) -> None:
    with open(_backend_path(), "w", encoding="utf-8") as f:
        json.dump({"name": b.name, "dim": b.dim}, f, ensure_ascii=False, indent=2)


def _load_backend() -> KXBackend | None:
    if not os.path.exists(_backend_path()):
        return None
    with open(_backend_path(), "r", encoding="utf-8") as f:
        d = json.load(f)
    return KXBackend(name=d.get("name", "tfidf"), dim=d.get("dim"))


def _get_model() -> Any:
    model_name = env("KX_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformer(model_name, device="cpu", trust_remote_code=False)


def _build_faiss_index(texts: List[str]) -> Tuple[Any, int]:
    if not _EMB_AVAILABLE:
        raise RuntimeError("Embeddings/FAISS not available")

    model = _get_model()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embs = np.asarray(embs).astype("float32")
    dim = int(embs.shape[1])

    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index, dim


def _save_faiss(index: Any) -> None:
    faiss.write_index(index, _faiss_index_path())


def _load_faiss() -> Any:
    return faiss.read_index(_faiss_index_path())


def _search_faiss(index: Any, query: str, top_k: int) -> Tuple[List[float], List[int]]:
    model = _get_model()
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q).astype("float32")
    scores, ids = index.search(q, top_k)
    return scores[0].tolist(), ids[0].tolist()


def _build_tfidf(texts: List[str]) -> Dict[str, Any]:
    if not _TFIDF_AVAILABLE:
        raise RuntimeError("TF-IDF backend not available (scikit-learn missing)")

    vec = TfidfVectorizer(max_features=80_000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    return {"vectorizer": vec, "matrix": X}


def _save_tfidf(obj: Dict[str, Any]) -> None:
    with open(_tfidf_path(), "wb") as f:
        pickle.dump(obj, f)


def _load_tfidf() -> Dict[str, Any]:
    with open(_tfidf_path(), "rb") as f:
        return pickle.load(f)


def _search_tfidf(obj: Dict[str, Any], query: str, top_k: int) -> Tuple[List[float], List[int]]:
    vec = obj["vectorizer"]
    X = obj["matrix"]
    q = vec.transform([query])
    scores = (X @ q.T).toarray().ravel()
    if scores.size == 0:
        return [], []
    idx = np.argsort(-scores)[:top_k]
    return scores[idx].tolist(), idx.tolist()


def build_kx_index(chunks: List[Dict[str, Any]], doc_name: str = "KX") -> Dict[str, Any]:
    """
    chunks: [{"chunk_id":..., "page":..., "text":..., "pdf_name":..., "start":..., "end":...}, ...]
    """
    meta: List[Dict[str, Any]] = []
    texts: List[str] = []

    for i, c in enumerate(chunks):
        txt = clean_text(c.get("text", ""))
        if not txt:
            continue
        meta.append(
            {
                "i": i,
                "doc_name": doc_name,
                "pdf_name": c.get("pdf_name"),
                "chunk_id": c.get("chunk_id"),
                "page": c.get("page"),
                "start": c.get("start"),
                "end": c.get("end"),
                "text": txt,
            }
        )
        texts.append(txt)

    _save_meta(meta)

    use_embeddings = env("KX_USE_EMBEDDINGS", "1") == "1"
    if use_embeddings:
        try:
            index, dim = _build_faiss_index(texts)
            _save_faiss(index)
            _save_backend(KXBackend(name="st_faiss", dim=dim))
            return {"backend": "st_faiss", "chunks": len(meta), "dim": dim}
        except Exception:
            _save_backend(KXBackend(name="tfidf", dim=None))

    tfidf_obj = _build_tfidf(texts)
    _save_tfidf(tfidf_obj)
    _save_backend(KXBackend(name="tfidf", dim=None))
    return {"backend": "tfidf", "chunks": len(meta), "dim": None}


def search_kx(
    query: str,
    top_k: int = 6,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    if not kx_index_exists():
        return []

    meta = _load_meta()
    backend = _load_backend() or KXBackend(name="tfidf", dim=None)

    if backend.name == "st_faiss" and os.path.exists(_faiss_index_path()):
        index = _load_faiss()
        scores, ids = _search_faiss(index, query, top_k)
    else:
        tfidf_obj = _load_tfidf()
        scores, ids = _search_tfidf(tfidf_obj, query, top_k)

    out: List[Dict[str, Any]] = []
    for score, idx in zip(scores, ids):
        if idx < 0 or idx >= len(meta):
            continue
        if float(score) < min_score:
            continue
        m = meta[int(idx)]
        out.append(
            {
                "score": float(score),
                "doc_name": m.get("doc_name"),
                "pdf_name": m.get("pdf_name"),
                "page": m.get("page"),
                "chunk_id": m.get("chunk_id"),
                "start": m.get("start"),
                "end": m.get("end"),
                "text": m.get("text"),
            }
        )
    return out


def mmr_diversify(
    hits: List[Dict[str, Any]],
    lambda_mult: float = 0.75,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """Lightweight MMR diversification using token overlap (no extra embeddings)."""
    if not hits:
        return []
    chosen: List[Dict[str, Any]] = []
    pool = hits[:]

    def toks(s: str) -> set[str]:
        return set(clean_text(s).lower().split())

    pool_tokens = [toks(h.get("text", "")) for h in pool]

    while pool and len(chosen) < top_k:
        if not chosen:
            best = 0
        else:
            chosen_tokens = [toks(c.get("text", "")) for c in chosen]
            best_score = -1e9
            best = 0
            for i, h in enumerate(pool):
                rel = float(h.get("score", 0.0))
                sim = 0.0
                for ct in chosen_tokens:
                    inter = len(pool_tokens[i] & ct)
                    union = len(pool_tokens[i] | ct) or 1
                    sim = max(sim, inter / union)
                mmr = lambda_mult * rel - (1.0 - lambda_mult) * sim
                if mmr > best_score:
                    best_score = mmr
                    best = i
        chosen.append(pool.pop(best))
        pool_tokens.pop(best)

    return chosen


def filter_kx_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop obvious TOC/index noise."""
    out: List[Dict[str, Any]] = []
    for h in hits:
        t = clean_text(h.get("text", ""))
        if not t:
            continue
        tl = t.lower()
        if "table of contents" in tl or "contenido" in tl or "índice" in tl or "indice" in tl:
            continue
        if t.count(".") > 16 and len(t) < 700:
            continue
        out.append(h)
    return out
