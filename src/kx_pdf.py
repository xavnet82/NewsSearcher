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

def extract_pdf_text(path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = clean_text(text)
        if text:
            pages.append((i + 1, text))
    return pages

def chunk_text(pages: List[Tuple[int, str]], chunk_size: int = 900, overlap: int = 150) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end]
            if len(chunk) > 120:
                chunks.append({
                    "chunk_id": sha1(f"{page_num}:{start}:{chunk[:40]}"),
                    "page": page_num,
                    "text": chunk
                })
            start = max(end - overlap, end)
    return chunks
