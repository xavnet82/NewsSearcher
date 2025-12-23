from __future__ import annotations
from typing import Any, Dict, List
import requests
import feedparser
from bs4 import BeautifulSoup

from .utils import clean_text, sha1, parse_date_to_iso, env

def fetch_rss(feeds: List[str], max_items: int = 50) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for url in feeds:
        d = feedparser.parse(url)
        for e in d.entries[:max_items]:
            title = clean_text(getattr(e, "title", "") or "")
            link = getattr(e, "link", "") or ""
            summary = clean_text(getattr(e, "summary", "") or getattr(e, "description", "") or "")
            published = None
            if hasattr(e, "published"):
                published = parse_date_to_iso(getattr(e, "published", ""))
            source = clean_text(getattr(d.feed, "title", "") or url)
            news_id = sha1(link or title)

            items.append({
                "id": news_id,
                "title": title,
                "summary": summary,
                "source": source,
                "url": link,
                "published_at": published,
                "provider": "rss"
            })
    return items

def fetch_newsapi(query: str, page_size: int = 50) -> List[Dict[str, Any]]:
    key = env("NEWSAPI_KEY", "")
    if not key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "language": "en",
    }
    headers = {"X-Api-Key": key}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    out = []
    for a in data.get("articles", []):
        title = clean_text(a.get("title", "") or "")
        link = a.get("url", "") or ""
        summary = clean_text(a.get("description", "") or "")
        source = clean_text((a.get("source") or {}).get("name", "") or "NewsAPI")
        published = a.get("publishedAt")
        news_id = sha1(link or title)
        out.append({
            "id": news_id,
            "title": title,
            "summary": summary,
            "source": source,
            "url": link,
            "published_at": published,
            "provider": "newsapi"
        })
    return out

def fetch_url_snippet(url: str, timeout: int = 12) -> str:
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        desc = ""
        m = soup.find("meta", attrs={"name": "description"})
        if m and m.get("content"):
            desc = clean_text(m.get("content"))

        paras = []
        for p in soup.find_all("p")[:4]:
            t = clean_text(p.get_text(" "))
            if len(t) > 60:
                paras.append(t)
        body = " ".join(paras)

        combined = clean_text(" ".join([desc, body]))
        return combined[:1200]
    except Exception:
        return ""

def fetch_google_news_rss(queries: list[str], hl: str, gl: str, ceid: str, max_items: int = 50):
    feeds = []
    for q in queries:
        q_enc = requests.utils.quote(q)
        feeds.append(f"https://news.google.com/rss/search?q={q_enc}&hl={hl}&gl={gl}&ceid={ceid}")

    return fetch_rss(feeds, max_items=max_items)
