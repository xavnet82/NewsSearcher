from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import feedparser
import requests
from bs4 import BeautifulSoup

from .utils import clean_text, env, sha1, safe_fetch_text


def fetch_rss(urls: List[str], provider: str = "rss") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in urls:
        if not u:
            continue
        d = feedparser.parse(u)
        for e in d.entries:
            title = clean_text(getattr(e, "title", "") or "")
            link = clean_text(getattr(e, "link", "") or "")
            summary = clean_text(getattr(e, "summary", "") or getattr(e, "description", "") or "")
            source = clean_text(getattr(getattr(e, "source", None), "title", "") or getattr(d.feed, "title", "") or provider)

            published_iso = None
            if getattr(e, "published_parsed", None):
                try:
                    published_iso = datetime(*e.published_parsed[:6], tzinfo=timezone.utc).isoformat()
                except Exception:
                    published_iso = None

            guid = clean_text(getattr(e, "id", "") or getattr(e, "guid", "") or "")
            base = link or guid or title
            item_id = sha1(base)

            out.append(
                {
                    "id": item_id,
                    "title": title,
                    "summary": summary,
                    "source": source,
                    "url": link,
                    "published_at": published_iso,
                    "provider": provider,
                    "raw": dict(e),
                }
            )
    return out


def build_google_news_rss_urls(query: str, hl: str = "es", gl: str = "ES", ceid: str = "ES:es") -> List[str]:
    base = "https://news.google.com/rss/search"
    params = {"q": query, "hl": hl, "gl": gl, "ceid": ceid}
    return [f"{base}?{urlencode(params)}"]


def fetch_newsapi(query: str, from_date: Optional[str] = None, language: str = "en", page_size: int = 30) -> List[Dict[str, Any]]:
    key = env("NEWSAPI_KEY", "")
    if not key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "language": language, "pageSize": page_size, "sortBy": "publishedAt"}
    if from_date:
        params["from"] = from_date

    r = requests.get(url, params=params, headers={"X-Api-Key": key}, timeout=10)
    r.raise_for_status()
    data = r.json()
    items = data.get("articles", []) or []

    out: List[Dict[str, Any]] = []
    for a in items:
        title = clean_text(a.get("title", "") or "")
        link = clean_text(a.get("url", "") or "")
        summary = clean_text(a.get("description", "") or a.get("content", "") or "")
        source = clean_text((a.get("source") or {}).get("name", "") or "NewsAPI")
        published_iso = clean_text(a.get("publishedAt", "") or "") or None
        item_id = sha1(link or title)

        out.append(
            {
                "id": item_id,
                "title": title,
                "summary": summary,
                "source": source,
                "url": link,
                "published_at": published_iso,
                "provider": "newsapi",
                "raw": a,
            }
        )
    return out


def fetch_url_snippet(url: str, max_chars: int = 450) -> Dict[str, Any]:
    """Best-effort HTML snippet extraction with SSRF/size guards."""
    html, status = safe_fetch_text(url)
    if not html:
        return {"ok": False, "status": status, "snippet": ""}

    try:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        metas = soup.find_all("meta")
        meta_desc = ""
        for m in metas:
            name = (m.get("name") or "").lower()
            prop = (m.get("property") or "").lower()
            if name == "description" or prop == "og:description":
                meta_desc = clean_text(m.get("content") or "")
                if meta_desc:
                    break

        if meta_desc:
            return {"ok": True, "status": "ok", "snippet": meta_desc[:max_chars]}

        text = clean_text(soup.get_text(" ", strip=True))
        text = text[: max_chars * 2]
        return {"ok": True, "status": "ok", "snippet": text[:max_chars]}
    except Exception as e:
        return {"ok": False, "status": f"error:{type(e).__name__}", "snippet": ""}
