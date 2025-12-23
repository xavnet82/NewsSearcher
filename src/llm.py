# src/llm.py
from __future__ import annotations

import json
from typing import Any, Dict, List

from src.utils import clean_text, env


def llm_available() -> bool:
    return bool(env("OPENAI_API_KEY", "").strip())


def _kx_compact(kx_hits: List[Dict[str, Any]], max_hits: int = 3, max_chars: int = 350) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in (kx_hits or [])[:max_hits]:
        out.append(
            {
                "doc": h.get("doc_name", "KX"),
                "page": h.get("page", None),
                "score": float(h.get("score", 0.0)),
                "snippet": clean_text((h.get("text") or "")[:max_chars]),
            }
        )
    return out


def generate_insights_llm(item: Dict[str, Any], kx_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Devuelve JSON estructurado ALINEADO AL RETO + enriquecimiento KX con citas.
    Incluye timeouts y límites de tokens para no colgar Streamlit.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=env("OPENAI_API_KEY", ""),
        timeout=float(env("OPENAI_TIMEOUT", "15")),
    )
    model = env("OPENAI_MODEL", "gpt-4o-mini")

    title = clean_text(item.get("title", ""))
    summary = clean_text(item.get("summary", ""))
    source = clean_text(item.get("source", ""))
    published = item.get("published_at", "")
    url = item.get("url", "")

    kx_evs = _kx_compact(kx_hits, max_hits=3, max_chars=350)

    prompt = f"""
Eres un analista de inteligencia de noticias para Accenture. Devuelve SOLO JSON válido (sin markdown).
Objetivo: output estructurado de noticia y enriquecimiento con evidencias internas (KX) si existen.

NOTICIA:
- titulo: {title}
- resumen: {summary}
- fuente: {source}
- fecha: {published}
- link: {url}

EVIDENCIAS KX (internas, pueden estar vacías):
{kx_evs}

INSTRUCCIONES:
1) Genera:
- descripcion_breve: 1-2 frases, neutral, sin inventar.
- por_que_importa: lista de 2-4 bullets concretos.
- implicaciones_para_accenture: lista de 2-4 bullets accionables.
2) Si hay evidencias KX, añade:
- kx_enriquecimiento.resumen_contexto: 1-2 frases conectando noticia con KX.
- kx_enriquecimiento.evidencias: copia SOLO evidencias usadas (doc,page,score,snippet).
3) Si no hay KX: kx_enriquecimiento.resumen_contexto="" y evidencias=[].
4) No inventes datos. Si falta info, mantén generalidad y usa "podría"/"potencialmente".

FORMATO JSON EXACTO:
{{
  "descripcion_breve": "...",
  "por_que_importa": ["..."],
  "implicaciones_para_accenture": ["..."],
  "kx_enriquecimiento": {{
    "resumen_contexto": "...",
    "evidencias": [{{"doc":"...","page":0,"score":0.0,"snippet":"..."}}]
  }}
}}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=int(env("OPENAI_MAX_TOKENS", "420")),
    )

    content = resp.choices[0].message.content or "{}"

    try:
        data = json.loads(content)
    except Exception:
        # fallback seguro
        data = {
            "descripcion_breve": summary[:240] if summary else title,
            "por_que_importa": [],
            "implicaciones_para_accenture": [],
            "kx_enriquecimiento": {"resumen_contexto": "", "evidencias": []},
        }

    # harden schema
    if not isinstance(data, dict):
        data = {
            "descripcion_breve": summary[:240] if summary else title,
            "por_que_importa": [],
            "implicaciones_para_accenture": [],
            "kx_enriquecimiento": {"resumen_contexto": "", "evidencias": []},
        }

    data.setdefault("descripcion_breve", "")
    data.setdefault("por_que_importa", [])
    data.setdefault("implicaciones_para_accenture", [])
    data.setdefault("kx_enriquecimiento", {"resumen_contexto": "", "evidencias": []})

    # types
    if not isinstance(data["por_que_importa"], list):
        data["por_que_importa"] = [str(data["por_que_importa"])] if data["por_que_importa"] else []
    if not isinstance(data["implicaciones_para_accenture"], list):
        data["implicaciones_para_accenture"] = [str(data["implicaciones_para_accenture"])] if data["implicaciones_para_accenture"] else []
    if not isinstance(data["kx_enriquecimiento"], dict):
        data["kx_enriquecimiento"] = {"resumen_contexto": "", "evidencias": []}
    data["kx_enriquecimiento"].setdefault("resumen_contexto", "")
    data["kx_enriquecimiento"].setdefault("evidencias", [])
    if not isinstance(data["kx_enriquecimiento"]["evidencias"], list):
        data["kx_enriquecimiento"]["evidencias"] = []

    # Si el modelo no devolvió evidencias pero hay KX disponible, las adjuntamos como referencia
    if kx_evs and not data["kx_enriquecimiento"]["evidencias"]:
        data["kx_enriquecimiento"]["evidencias"] = kx_evs

    return data


def generate_insights_heuristic(item: Dict[str, Any], kx_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fallback determinista: output alineado al reto + enriquecimiento KX (si existe) con citas.
    """
    title = clean_text(item.get("title", ""))
    summary = clean_text(item.get("summary", ""))

    evs = _kx_compact(kx_hits, max_hits=3, max_chars=300)
    kx_ctx = ""
    if evs:
        kx_ctx = "Existe contexto interno relacionado en KX; revisar evidencias citadas para alinear mensaje, posicionamiento y acciones."

    text = (title + " " + summary).lower()

    por_que: List[str] = []
    if "acquisition" in text or "acquire" in text or "adquis" in text:
        por_que.append("Podría implicar cambios en capacidades, posicionamiento competitivo o cartera de servicios.")
    if "regulation" in text or "regulación" in text or "compliance" in text:
        por_que.append("Podría afectar a compliance, operaciones y demanda de asesoramiento/regtech.")
    if "cyber" in text or "ciber" in text or "security" in text:
        por_que.append("Podría indicar variaciones en demanda y prioridades de ciberseguridad en clientes.")
    if "ai" in text or "genai" in text or "generative" in text:
        por_que.append("Podría afectar a prioridades de inversión en IA/GenAI y necesidades de adopción responsable.")

    if not por_que:
        por_que = [
            "Puede afectar a percepción de mercado, clientes y competitividad en servicios.",
            "Puede activar oportunidades de conversación comercial en sectores clave.",
        ]

    implicaciones = [
        "Identificar sectores/clientes impactados (FS/PS/Products) y preparar talking points.",
        "Mapear oportunidades en servicios (Cloud, Data, Security, GenAI, Operating Model) relacionadas con la noticia.",
    ]
    if evs:
        implicaciones.append("Contrastar con KX interno para asegurar consistencia con posicionamiento y mensajes aprobados.")

    return {
        "descripcion_breve": summary[:260] if summary else title,
        "por_que_importa": por_que[:4],
        "implicaciones_para_accenture": implicaciones[:4],
        "kx_enriquecimiento": {
            "resumen_contexto": kx_ctx,
            "evidencias": evs,
        },
    }
