from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .utils import clean_text


def _parse_iso(dt: Optional[str]) -> Optional[datetime]:
    if not dt:
        return None
    try:
        if dt.endswith("Z"):
            dt = dt[:-1] + "+00:00"
        return datetime.fromisoformat(dt).astimezone(timezone.utc)
    except Exception:
        return None


def recency_score(published_at_iso: Optional[str]) -> float:
    """0..1. Exponential decay with 10-day scale."""
    dt = _parse_iso(published_at_iso)
    if not dt:
        return 0.2
    now = datetime.now(timezone.utc)
    days = max(0.0, (now - dt).total_seconds() / 86400.0)
    return float(math.exp(-days / 10.0))


def keyword_hit_ratio(text: str, keywords: List[str], cap: int) -> float:
    t = clean_text(text).lower()
    hits = 0
    for kw in keywords:
        kw = (kw or "").strip().lower()
        if not kw:
            continue
        if kw in t:
            hits += 1
    return min(hits, cap) / float(cap or 1)


def score_item(
    title: str,
    summary: str,
    published_at: Optional[str],
    acc_keywords: List[str],
    entity_keywords: List[str],
    kx_hits: List[Dict[str, Any]],
    classif: Dict[str, List[str]],
    weights: Optional[Dict[str, float]] = None,
    kx_min_evidence: float = 0.35,
) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (final_score_0_100, explain_dict)

    Pillars (0..100):
    - Freshness (0..25)
    - Accenture relevance (0..30): keywords/entities on title+summary
    - Market/service fit (0..25): #tags matched
    - KX alignment (0..20): evidence gated by similarity
    """
    weights = weights or {"freshness": 25, "relevance": 30, "fit": 25, "kx": 20}

    text = clean_text(f"{title} {summary}")
    r = recency_score(published_at)
    freshness = r * weights["freshness"]

    rel_kw = keyword_hit_ratio(text, acc_keywords, cap=8)
    rel_ent = keyword_hit_ratio(text, entity_keywords, cap=6)
    relevance = (0.6 * rel_kw + 0.4 * rel_ent) * weights["relevance"]

    n_tags = len(classif.get("markets", [])) + len(classif.get("services", [])) + len(classif.get("trends", []))
    fit_ratio = min(n_tags, 6) / 6.0
    fit = fit_ratio * weights["fit"]

    kx_best = max([float(h.get("score", 0.0)) for h in kx_hits], default=0.0)
    if kx_best < kx_min_evidence:
        kx = 0.0
        kx_gated = True
    else:
        kx = min(1.0, kx_best) * weights["kx"]
        kx_gated = False

    total = freshness + relevance + fit + kx
    total = max(0.0, min(100.0, total))

    explain = {
        "pillars": {
            "freshness": round(freshness, 2),
            "relevance": round(relevance, 2),
            "fit": round(fit, 2),
            "kx_alignment": round(kx, 2),
        },
        "signals": {
            "recency_0_1": round(r, 3),
            "keyword_ratio": round(rel_kw, 3),
            "entity_ratio": round(rel_ent, 3),
            "kx_best": round(kx_best, 3),
            "kx_gated": kx_gated,
            "tags": classif,
        },
    }
    return float(total), explain
