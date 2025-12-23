from __future__ import annotations
import os
import sqlite3
from typing import Any, Dict, List

from .utils import env, now_utc_iso, safe_json

def get_db_path() -> str:
    path = env("DB_PATH", "data/app.db")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = connect()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS news_items (
        id TEXT PRIMARY KEY,
        title TEXT,
        summary TEXT,
        source TEXT,
        url TEXT,
        published_at TEXT,
        raw_json TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        created_at TEXT,
        params_json TEXT,
        results_json TEXT
    )
    """)

    conn.commit()
    conn.close()

def upsert_news(item: Dict[str, Any]) -> None:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO news_items (id, title, summary, source, url, published_at, raw_json)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
        title=excluded.title,
        summary=excluded.summary,
        source=excluded.source,
        url=excluded.url,
        published_at=excluded.published_at,
        raw_json=excluded.raw_json
    """, (
        item["id"], item.get("title"), item.get("summary"), item.get("source"),
        item.get("url"), item.get("published_at"), safe_json(item)
    ))
    conn.commit()
    conn.close()

def news_exists(news_id: str) -> bool:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM news_items WHERE id = ? LIMIT 1", (news_id,))
    row = cur.fetchone()
    conn.close()
    return row is not None

def list_news(limit: int = 200) -> List[Dict[str, Any]]:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, summary, source, url, published_at, raw_json
        FROM news_items
        ORDER BY published_at DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "title": r["title"],
            "summary": r["summary"],
            "source": r["source"],
            "url": r["url"],
            "published_at": r["published_at"],
        })
    return out

def save_run(run_id: str, params: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO runs (run_id, created_at, params_json, results_json)
    VALUES (?, ?, ?, ?)
    """, (run_id, now_utc_iso(), safe_json(params), safe_json(results)))
    conn.commit()
    conn.close()
