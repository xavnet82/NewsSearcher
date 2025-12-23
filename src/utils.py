from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import ipaddress
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -------------------------- Safe URL fetching --------------------------

_PRIVATE_NETS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _host_to_ip(host: str) -> Optional[ipaddress._BaseAddress]:
    try:
        ip = ipaddress.ip_address(host)
        return ip
    except ValueError:
        pass

    try:
        ip_str = socket.gethostbyname(host)
        return ipaddress.ip_address(ip_str)
    except Exception:
        return None


def is_private_host(host: str) -> bool:
    if not host:
        return True
    ip = _host_to_ip(host)
    if ip is None:
        return True
    return any(ip in net for net in _PRIVATE_NETS)


def validate_public_http_url(url: str) -> Tuple[bool, str]:
    """Basic SSRF guard: only http/https, non-private hosts."""
    try:
        p = urlparse(url)
    except Exception:
        return False, "invalid_url"

    if p.scheme not in ("http", "https"):
        return False, "invalid_scheme"

    if not p.netloc:
        return False, "missing_host"

    host = p.hostname or ""
    if is_private_host(host):
        return False, "private_host"

    return True, "ok"


@dataclass
class FetchPolicy:
    timeout_s: float = 6.0
    max_bytes: int = 350_000
    max_redirects: int = 2
    user_agent: str = "Acn2Agent/1.1 (+streamlit)"


def safe_fetch_text(url: str, policy: Optional[FetchPolicy] = None) -> Tuple[Optional[str], str]:
    """Fetch small HTML/text safely (best-effort). Returns (text, status)."""
    policy = policy or FetchPolicy()
    ok, reason = validate_public_http_url(url)
    if not ok:
        return None, reason

    try:
        with requests.Session() as s:
            s.max_redirects = policy.max_redirects
            r = s.get(
                url,
                timeout=policy.timeout_s,
                headers={"User-Agent": policy.user_agent, "Accept": "text/html,application/xhtml+xml"},
                allow_redirects=True,
                stream=True,
            )
            r.raise_for_status()

            buf = bytearray()
            for chunk in r.iter_content(chunk_size=16_384):
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) > policy.max_bytes:
                    return None, "too_large"

            r.encoding = r.encoding or "utf-8"
            return buf.decode(r.encoding, errors="replace"), "ok"
    except Exception as e:
        return None, f"error:{type(e).__name__}"


def to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
