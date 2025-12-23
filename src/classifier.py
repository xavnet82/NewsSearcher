from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .utils import clean_text


TREND_MAP: Dict[str, List[str]] = {
    "GenAI": ["genai", "generative ai", "llm", "foundation model", "copilot"],
    "Ciberseguridad": ["cyber", "ransomware", "zero trust", "security", "niso", "iso 27001"],
    "Cloud & Data": ["cloud", "data platform", "lakehouse", "databricks", "snowflake", "azure", "aws", "gcp"],
    "Soberanía / Regulación": ["sovereign", "sovereignty", "sovereign cloud", "dora", "nis2", "gdpr", "regulation"],
}

MARKET_MAP: Dict[str, List[str]] = {
    "Banca": ["bank", "banking", "fintech", "payments"],
    "Seguros": ["insurance", "insurer"],
    "Sector Público": ["public sector", "government", "ministry", "municipal"],
    "Energía": ["energy", "utilities", "grid"],
}

SERVICE_MAP: Dict[str, List[str]] = {
    "Strategy & Consulting": ["strategy", "consulting", "operating model", "transformation"],
    "Technology": ["implementation", "migration", "platform", "architecture"],
    "Security": ["security", "soc", "iam", "zero trust"],
    "Managed Services": ["managed services", "outsourcing", "run", "operate"],
}


def _hits(text: str, terms: List[str]) -> int:
    t = text.lower()
    n = 0
    for term in terms:
        term = term.lower().strip()
        if not term:
            continue
        if len(term) <= 4:
            if re.search(rf"\b{re.escape(term)}\b", t):
                n += 1
        else:
            if term in t:
                n += 1
    return n


def classify(title: str, summary: str) -> Dict[str, List[str]]:
    text = clean_text(f"{title} {summary}")
    out: Dict[str, List[str]] = {"trends": [], "markets": [], "services": []}

    for k, terms in TREND_MAP.items():
        if _hits(text, terms) > 0:
            out["trends"].append(k)

    for k, terms in MARKET_MAP.items():
        if _hits(text, terms) > 0:
            out["markets"].append(k)

    for k, terms in SERVICE_MAP.items():
        if _hits(text, terms) > 0:
            out["services"].append(k)

    return out


def hard_filter(title: str, summary: str, must_terms: List[str]) -> Tuple[bool, List[str]]:
    """Return (passes, matched_terms)."""
    if not must_terms:
        return True, []
    text = clean_text(f"{title} {summary}").lower()
    matched: List[str] = []
    for term in must_terms:
        term = term.strip()
        if not term:
            continue
        tl = term.lower()
        if len(tl) <= 4:
            ok = re.search(rf"\b{re.escape(tl)}\b", text) is not None
        else:
            ok = tl in text
        if ok:
            matched.append(term)
    return (len(matched) > 0), matched
