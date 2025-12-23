from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .utils import env, clean_text

from functools import lru_cache
from sentence_transformers import SentenceTransformer
from src.utils import env

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    model_name = env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Forzamos CPU explícitamente. Evita caminos raros de device/meta.
    embedder = SentenceTransformer(
        model_name,
        device="cpu",
        trust_remote_code=False,
    )
    return embedder


def ensure_faiss_dir() -> str:
    d = env("FAISS_DIR", "data/faiss")
    os.makedirs(d, exist_ok=True)
    return d

def _index_path() -> str:
    return os.path.join(ensure_faiss_dir(), "kx.index")

def _meta_path() -> str:
    return os.path.join(ensure_faiss_dir(), "kx_meta.json")

def build_kx_index(chunks: List[Dict[str, Any]], doc_name: str) -> Dict[str, Any]:
    embedder = get_embedder()
    texts = [clean_text(c["text"]) for c in chunks]
    embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embs = np.array(embs).astype("float32")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, _index_path())

    meta = []
    for i, c in enumerate(chunks):
        meta.append({
            "i": i,
            "doc_name": doc_name,
            "chunk_id": c.get("chunk_id"),
            "page": c.get("page"),
            "text": c.get("text"),
        })
    with open(_meta_path(), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"chunks": len(chunks), "dim": dim}

def kx_index_exists() -> bool:
    return os.path.exists(_index_path()) and os.path.exists(_meta_path())

def load_index() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index = faiss.read_index(_index_path())
    with open(_meta_path(), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def search_kx(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not kx_index_exists():
        return []
    index, meta = load_index()
    embedder = get_embedder()
    q = clean_text(query)
    q_emb = embedder.encode([q], normalize_embeddings=True, show_progress_bar=False)
    q_emb = np.array(q_emb).astype("float32")

    scores, idxs = index.search(q_emb, top_k)
    out = []
    for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
        if i < 0 or i >= len(meta):
            continue
        m = meta[i]
        out.append({
            "score": float(score),
            "doc_name": m.get("doc_name"),
            "chunk_id": m.get("chunk_id"),
            "page": m.get("page"),
            "text": m.get("text"),
        })
    return out
