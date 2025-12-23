from __future__ import annotations
from typing import Dict, List, Tuple
from .utils import clean_text

# Diccionarios simples (ajustables en UI si quieres)
TREND_MAP = {
    "GenAI/LLM": ["genai", "generative ai", "llm", "gpt", "foundation model"],
    "Cloud": ["cloud", "aws", "azure", "gcp", "kubernetes"],
    "Sovereign Cloud": ["sovereign", "sovereignty", "data residency", "gaia-x"],
    "Cyber": ["cyber", "security", "ransomware", "zero trust"],
    "Regulation": ["regulation", "regulatory", "compliance", "gdpr", "ai act"],
    "Data": ["data platform", "lakehouse", "analytics", "governance"],
}

MARKET_MAP = {
    "Financial Services": ["bank", "banking", "fintech", "insurance", "payments"],
    "Public Sector": ["government", "public sector", "ministry", "agency", "municipal"],
    "Health": ["hospital", "healthcare", "pharma", "clinical"],
    "Industrial": ["manufacturing", "automotive", "energy", "utilities"],
}

SERVICE_MAP = {
    "Strategy & Consulting": ["strategy", "operating model", "transformation", "cost", "target operating model"],
    "Cloud": ["cloud", "migration", "landing zone", "platform engineering"],
    "Data/AI": ["data", "ai", "ml", "genai", "analytics"],
    "Security": ["security", "cyber", "iam", "soc", "zero trust"],
}

# Heurística muy simple de alcance
GLOBAL_HINTS = ["global", "worldwide", "international", "eu", "europe", "united states", "usa", "asia"]

def _match_categories(text: str, mapping: Dict[str, List[str]]) -> List[str]:
    hits = []
    for k, terms in mapping.items():
        for t in terms:
            if t in text:
                hits.append(k)
                break
    return hits

def classify(news: Dict) -> Dict:
    title = clean_text(news.get("title", "")).lower()
    summary = clean_text(news.get("summary", "")).lower()
    text = f"{title} {summary}"

    trends = _match_categories(text, TREND_MAP)
    markets = _match_categories(text, MARKET_MAP)
    services = _match_categories(text, SERVICE_MAP)

    scope = "Regional/Local"
    if any(h in text for h in GLOBAL_HINTS):
        scope = "Global"

    return {
        "trends": trends or ["Other"],
        "markets": markets or ["Unknown"],
        "services": services or ["Unknown"],
        "scope": scope,
    }

def hard_filter(news: Dict, must_have_any: List[str], max_age_days: int | None = None) -> Tuple[bool, str]:
    """
    Returns (keep?, reason_if_dropped)
    must_have_any: terms that must appear in title+summary
    """
    title = clean_text(news.get("title", "")).lower()
    summary = clean_text(news.get("summary", "")).lower()
    text = f"{title} {summary}"

    if not title and not summary:
        return False, "Sin contenido"

    if must_have_any:
        if not any(t.lower() in text for t in must_have_any):
            return False, "No cumple términos mínimos"

    # max_age_days se evalúa en scoring.py (days_ago); aquí lo dejamos opcional si quieres endurecer
    return True, ""
