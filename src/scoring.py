from __future__ import annotations
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone
import math

from .utils import clean_text, clamp

DEFAULT_KEYWORDS = [
    "ai", "artificial intelligence", "genai", "llm",
    "cloud", "sovereign cloud", "regulation", "compliance",
    "cyber", "security", "data", "public sector", "banking", "financial"
]

DEFAULT_ENTITIES = [
    "accenture", "aws", "azure", "google", "microsoft", "openai",
    "sap", "oracle", "ibm", "deloitte", "pwc", "ey", "kpmg"
]

def _days_ago_iso(iso_str: str) -> float:
    if not iso_str:
        return 365.0
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (now - dt).total_seconds() / 86400.0)
    except Exception:
        return 365.0

def score_news_item(
    item: Dict[str, Any],
    kx_hits: List[Dict[str, Any]],
    weights: Dict[str, float],
    keywords: List[str],
    entities: List[str],
) -> Tuple[float, Dict[str, float], List[str]]:
    title = clean_text(item.get("title", "")).lower()
    summary = clean_text(item.get("summary", "")).lower()
    text = f"{title} {summary}"

    days = _days_ago_iso(item.get("published_at") or "")
    recency = math.exp(-days / 10.0)
    recency_score = 100.0 * clamp(recency, 0.0, 1.0)

    ent_hits = 0
    ent_tags = []
    for e in entities:
        if e.lower() in text:
            ent_hits += 1
            ent_tags.append(e)
    entity_score = 100.0 * clamp(ent_hits / max(1, min(6, len(entities))), 0.0, 1.0)

    kw_hits = 0
    kw_tags = []
    for k in keywords:
        if k.lower() in text:
            kw_hits += 1
            kw_tags.append(k)
    keyword_score = 100.0 * clamp(kw_hits / max(1, min(8, len(keywords))), 0.0, 1.0)

    kx_best = max([h.get("score", 0.0) for h in kx_hits], default=0.0)
    kx_score = 100.0 * clamp(float(kx_best), 0.0, 1.0)

    comps = {
        "recency": recency_score,
        "entity": entity_score,
        "keyword": keyword_score,
        "kx": kx_score,
    }

    total = (
        weights.get("recency", 0.25) * recency_score +
        weights.get("entity", 0.25) * entity_score +
        weights.get("keyword", 0.20) * keyword_score +
        weights.get("kx", 0.30) * kx_score
    )
    total = clamp(total, 0.0, 100.0)

    tags = sorted(set(ent_tags[:4] + kw_tags[:4]))
    return total, comps, tags
