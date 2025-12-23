from __future__ import annotations
import os
import re
import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

def env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None and v != "" else default

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def parse_date_to_iso(dt_str: str) -> Optional[str]:
    if not dt_str:
        return None
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
