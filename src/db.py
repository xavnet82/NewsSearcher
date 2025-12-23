from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .utils import env


def _db_path() -> str:
    path = env("DB_PATH", "data/app.db")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_items (
                id TEXT PRIMARY KEY,
                title TEXT,
                summary TEXT,
                source TEXT,
                url TEXT,
                published_at TEXT,
                provider TEXT,
                raw_json TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT,
                params_json TEXT,
                results_json TEXT
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_published_at ON news_items(published_at);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_provider ON news_items(provider);")
        conn.commit()


def upsert_news(item: Dict[str, Any]) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO news_items(id, title, summary, source, url, published_at, provider, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                summary=excluded.summary,
                source=excluded.source,
                url=excluded.url,
                published_at=excluded.published_at,
                provider=excluded.provider,
                raw_json=excluded.raw_json;
            """,
            (
                item.get("id"),
                item.get("title"),
                item.get("summary"),
                item.get("source"),
                item.get("url"),
                item.get("published_at"),
                item.get("provider"),
                json.dumps(item.get("raw", {}), ensure_ascii=False),
            ),
        )
        conn.commit()


def get_news(limit: int = 500) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM news_items ORDER BY published_at DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(r) for r in rows]


def save_run(run_id: str, params: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO runs(run_id, created_at, params_json, results_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                created_at=excluded.created_at,
                params_json=excluded.params_json,
                results_json=excluded.results_json;
            """,
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(params, ensure_ascii=False),
                json.dumps(results, ensure_ascii=False),
            ),
        )
        conn.commit()


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return dict(row) if row else None
