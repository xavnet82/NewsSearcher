from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os

from pypdf import PdfReader

from .utils import clean_text, sha1, env


def ensure_kx_dir() -> str:
    kx_dir = env("KX_DIR", "data/kx_pdfs")
    os.makedirs(kx_dir, exist_ok=True)
    return kx_dir


def save_uploaded_pdf(file_bytes: bytes, filename: str) -> str:
    kx_dir = ensure_kx_dir()
    safe_name = filename.replace("/", "_").replace("\\", "_")
    path = os.path.join(kx_dir, safe_name)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


def extract_pdf_text(pdf_path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(pdf_path)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = clean_text(txt)
        if txt:
            pages.append((i, txt))
    return pages


def _looks_like_toc(s: str) -> bool:
    s2 = s.lower()
    if "table of contents" in s2 or "contenido" in s2 or "índice" in s2 or "indice" in s2:
        return True
    if s.count(".") > 12 and len(s) < 700:
        return True
    return False


def chunk_pages(
    pages: List[Tuple[int, str]],
    pdf_name: str,
    chunk_size: int = 900,
    overlap: int = 180,
    min_chars: int = 140,
) -> List[Dict[str, Any]]:
    """Chunk by character window with real overlap and traceable metadata."""
    chunks: List[Dict[str, Any]] = []
    for page_num, text in pages:
        if _looks_like_toc(text):
            continue

        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + chunk_size)
            chunk = text[start:end]
            if len(chunk) >= min_chars:
                chunks.append(
                    {
                        "chunk_id": sha1(f"{pdf_name}:{page_num}:{start}:{chunk[:60]}"),
                        "pdf_name": pdf_name,
                        "page": page_num,
                        "start": start,
                        "end": end,
                        "text": chunk,
                    }
                )
            if end >= n:
                break
            # REAL overlap
            start = max(0, end - overlap)
            if start == end:  # safety
                start = end + 1
    return chunks
