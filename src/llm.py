from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.utils import clean_text, env


def llm_available() -> bool:
    return bool(env("OPENAI_API_KEY", "").strip())


def _client() -> OpenAI:
    return OpenAI(api_key=env("OPENAI_API_KEY", ""))


def _kx_compact(kx_hits: List[Dict[str, Any]], max_hits: int = 4, max_chars: int = 420) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in (kx_hits or [])[:max_hits]:
        out.append(
            {
                "evidence_id": f"{h.get('pdf_name','KX')}#p{h.get('page')}:{h.get('chunk_id')}",
                "pdf_name": h.get("pdf_name"),
                "page": h.get("page"),
                "score": round(float(h.get("score", 0.0)), 3),
                "text": clean_text(h.get("text", ""))[:max_chars],
            }
        )
    return out


def _schema_stub() -> Dict[str, Any]:
    return {
        "brief": "",
        "why_it_matters": [],
        "implications_for_accenture": [],
        "kx_context": [],
        "kx_evidence_ids": [],
    }


def ensure_structured_output(obj: Any) -> Dict[str, Any]:
    base = _schema_stub()
    if not isinstance(obj, dict):
        return base
    for k in base.keys():
        if k in obj:
            base[k] = obj[k]
    for k in ["why_it_matters", "implications_for_accenture", "kx_context", "kx_evidence_ids"]:
        if not isinstance(base[k], list):
            base[k] = []
        base[k] = [clean_text(str(x)) for x in base[k] if clean_text(str(x))]
    base["brief"] = clean_text(str(base.get("brief", "")))
    return base


def generate_insights_heuristic(
    title: str,
    summary: str,
    kx_hits: List[Dict[str, Any]],
    kx_gated: bool,
) -> Dict[str, Any]:
    text = clean_text(f"{title}. {summary}")
    why = []
    if title:
        why.append(f"El titular sugiere un cambio/relevancia: {title[:120]}")
    if summary:
        why.append(f"Resumen indica: {summary[:160]}")
    why = why[:3]

    impl = [
        "Validar si afecta a cuentas/sectores prioritarios.",
        "Identificar servicio/competencia asociada y preparar POV.",
    ]

    kx_context = []
    kx_ids = []
    if (not kx_gated) and kx_hits:
        top = kx_hits[0]
        kx_context.append(f"KX menciona contexto relacionado (p.{top.get('page')})")
        kx_ids.append(f"{top.get('pdf_name','KX')}#p{top.get('page')}:{top.get('chunk_id')}")
    return {
        "brief": text[:240],
        "why_it_matters": why,
        "implications_for_accenture": impl,
        "kx_context": kx_context,
        "kx_evidence_ids": kx_ids,
    }


def generate_insights_llm(
    title: str,
    summary: str,
    url: str,
    kx_hits: List[Dict[str, Any]],
    kx_gated: bool,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    if not llm_available():
        return _schema_stub()

    model = model or env("OPENAI_MODEL", "gpt-4o-mini")
    kx = _kx_compact(kx_hits)

    system = (
        "Eres un analista. Devuelve SOLO JSON válido (sin markdown). "
        "No inventes hechos. Usa únicamente la noticia y las evidencias KX proporcionadas. "
        "Si no hay evidencia KX sólida (kx_gated=true), deja kx_context y kx_evidence_ids vacíos."
    )

    user = {
        "news": {"title": title, "summary": summary, "url": url},
        "kx_gated": bool(kx_gated),
        "kx_evidence": kx,
        "output_schema": _schema_stub(),
        "instructions": [
            "brief: 1-2 frases (<=240 chars).",
            "why_it_matters: 2-4 bullets concretos.",
            "implications_for_accenture: 2-4 bullets accionables.",
            "kx_context: 1-3 bullets SOLO si kx_gated=false.",
            "kx_evidence_ids: lista de evidence_id usados (subset de los proporcionados).",
        ],
    }

    try:
        client = _client()
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            max_tokens=int(env("OPENAI_MAX_TOKENS", "450")),
            timeout=float(env("OPENAI_TIMEOUT", "20")),
        )
        txt = resp.choices[0].message.content or ""
        obj = json.loads(txt)
        return ensure_structured_output(obj)
    except Exception:
        return _schema_stub()
