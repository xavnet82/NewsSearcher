from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from src.utils import clean_text, env

# --- Optional deps (TF-IDF fallback) ---
_TFIDF_AVAILABLE = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    _TFIDF_AVAILABLE = False

# --- Optional deps (SentenceTransformers) ---
_ST_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    _ST_AVAILABLE = False

# ---------------- Paths ----------------
def ensure_store_dir() -> str:
    d = env("KX_STORE_DIR", "data/kx_store")
    os.makedirs(d, exist_ok=True)
    return d

def _meta_path() -> str:
    return os.path.join(ensure_store_dir(), "kx_meta.json")

def _backend_path() -> str:
    return os.path.join(ensure_store_dir(), "kx_backend.json")

def _faiss_index_path() -> str:
    return os.path.join(ensure_store_dir(), "kx_faiss.index")

def _tfidf_path() -> str:
    return os.path.join(ensure_store_dir(), "kx_tfidf.pkl")


# ---------------- Backend selection ----------------
@dataclass
class KXBackend:
    name: str  # "st_faiss" or "tfidf"
    dim: int | None = None

def _save_backend(b: KXBackend) -> None:
    with open(_backend_path(), "w", encoding="utf-8") as f:
        json.dump({"name": b.name, "dim": b.dim}, f, ensure_ascii=False, indent=2)

def _load_backend() -> KXBackend | None:
    if not os.path.exists(_backend_path()):
        return None
    with open(_backend_path(), "r", encoding="utf-8") as f:
        d = json.load(f)
    return KXBackend(name=d.get("name", "tfidf"), dim=d.get("dim"))

def kx_index_exists() -> bool:
    # Either FAISS or TFIDF artifacts must exist
    return os.path.exists(_meta_path()) and (os.path.exists(_faiss_index_path()) or os.path.exists(_tfidf_path()))

def _save_meta(meta: List[Dict[str, Any]]) -> None:
    with open(_meta_path(), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _load_meta() -> List[Dict[str, Any]]:
    with open(_meta_path(), "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- Embeddings backend (SentenceTransformer + FAISS) ----------------
def _get_embedder_cpu() -> "SentenceTransformer":
    """
    Carga robusta en CPU. Si el entorno fuerza 'meta' y falla, lanzará excepción.
    """
    if not _ST_AVAILABLE:
        raise RuntimeError("sentence-transformers no disponible")

    model_name = env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Intento explícito en CPU. Evita GPU, evita remote code.
    return SentenceTransformer(
        model_name,
        device="cpu",
        trust_remote_code=False,
    )

def _build_faiss_index(texts: List[str]) -> Tuple[Any, int]:
    import faiss  # faiss-cpu debe estar instalado
    embedder = _get_embedder_cpu()

    emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.array(emb).astype("float32")
    dim = emb.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, dim

def _search_faiss(index: Any, query: str, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    embedder = _get_embedder_cpu()
    q = clean_text(query)
    q_emb = embedder.encode([q], normalize_embeddings=True, show_progress_bar=False)
    q_emb = np.array(q_emb).astype("float32")
    scores, ids = index.search(q_emb, top_k)
    return scores[0], ids[0]

def _save_faiss(index: Any) -> None:
    import faiss
    faiss.write_index(index, _faiss_index_path())

def _load_faiss() -> Any:
    import faiss
    return faiss.read_index(_faiss_index_path())


# ---------------- TF-IDF fallback backend (NO TORCH) ----------------
def _build_tfidf(texts: List[str]) -> Dict[str, Any]:
    if not _TFIDF_AVAILABLE:
        raise RuntimeError("scikit-learn no disponible para TF-IDF fallback")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=80000,
        ngram_range=(1, 2),
        stop_words=None,
    )
    X = vectorizer.fit_transform(texts)
    return {"vectorizer": vectorizer, "X": X}

def _save_tfidf(obj: Dict[str, Any]) -> None:
    with open(_tfidf_path(), "wb") as f:
        pickle.dump(obj, f)

def _load_tfidf() -> Dict[str, Any]:
    with open(_tfidf_path(), "rb") as f:
        return pickle.load(f)

def _search_tfidf(tfidf_obj: Dict[str, Any], query: str, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    q = clean_text(query)
    vectorizer = tfidf_obj["vectorizer"]
    X = tfidf_obj["X"]
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, X)[0]
    top_idx = np.argsort(-sims)[:top_k]
    return sims[top_idx], top_idx


# ---------------- Public API ----------------
def build_kx_index(chunks: List[Dict[str, Any]], doc_name: str = "KX") -> Dict[str, Any]:
    """
    chunks: [{"chunk_id":..., "page":..., "text":...}, ...]
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
                "chunk_id": c.get("chunk_id"),
                "page": c.get("page"),
                "text": txt,
            }
        )
        texts.append(txt)

    if not meta:
        raise ValueError("No hay chunks con texto para indexar.")

    _save_meta(meta)

    # 1) Intento embeddings + FAISS (si está instalado y el modelo carga)
    use_embeddings = env("KX_USE_EMBEDDINGS", "1") == "1"
    if use_embeddings:
        try:
            index, dim = _build_faiss_index(texts)
            _save_faiss(index)
            _save_backend(KXBackend(name="st_faiss", dim=dim))
            return {"backend": "st_faiss", "chunks": len(meta), "dim": dim}
        except Exception as e:
            # Si falla (tu caso), caemos a TF-IDF
            _save_backend(KXBackend(name="tfidf", dim=None))
            # seguimos abajo con tfidf

    # 2) TF-IDF fallback (sin torch)
    tfidf_obj = _build_tfidf(texts)
    _save_tfidf(tfidf_obj)
    _save_backend(KXBackend(name="tfidf", dim=None))
    return {"backend": "tfidf", "chunks": len(meta), "dim": None}

def search_kx(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
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
        m = meta[int(idx)]
        out.append(
            {
                "score": float(score),
                "doc_name": m.get("doc_name"),
                "page": m.get("page"),
                "chunk_id": m.get("chunk_id"),
                "text": m.get("text"),
            }
        )
    return out
