from __future__ import annotations
from typing import Any, Dict, List
from .utils import env, clean_text

def llm_available() -> bool:
    return bool(env("OPENAI_API_KEY", ""))

def generate_insights_llm(
    news: Dict[str, Any],
    kx_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(api_key=env("OPENAI_API_KEY", ""))

    model = env("OPENAI_MODEL", "gpt-4o-mini")

    kx_evidence = []
    for h in kx_hits[:5]:
        kx_evidence.append({
            "doc": h.get("doc_name"),
            "page": h.get("page"),
            "text": (h.get("text") or "")[:450]
        })

    prompt = f"""
You are an enterprise news intelligence agent for Accenture Strategy & Consulting.

TASK:
Given a news item and internal KX evidence snippets, produce a concise structured output:
- descripcion (1-2 sentences)
- por_que_importa (2-4 bullets)
- implicaciones_para_accenture (2-4 bullets)
- nivel_confianza (low/medium/high) based on evidence relevance

NEWS:
Title: {news.get("title","")}
Summary: {news.get("summary","")}
Source: {news.get("source","")}
URL: {news.get("url","")}

KX EVIDENCE SNIPPETS (internal):
{kx_evidence}

Return ONLY valid JSON with keys:
descripcion, por_que_importa, implicaciones_para_accenture, nivel_confianza
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or "{}"

    import json
    try:
        return json.loads(content)
    except Exception:
        return {
            "descripcion": clean_text(news.get("summary", ""))[:240],
            "por_que_importa": ["(LLM output not parseable)"],
            "implicaciones_para_accenture": ["(LLM output not parseable)"],
            "nivel_confianza": "low",
        }

def generate_insights_heuristic(
    news: Dict[str, Any],
    kx_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    title = clean_text(news.get("title", ""))
    summary = clean_text(news.get("summary", ""))

    evid = []
    for h in kx_hits[:3]:
        evid.append(f"KX({h.get('doc_name')} p.{h.get('page')} sim={h.get('score',0):.2f})")

    why = []
    if "regul" in (title + " " + summary).lower():
        why.append("Posible impacto regulatorio que puede afectar decisiones de tecnología y compliance.")
    if "ai" in (title + " " + summary).lower() or "artificial intelligence" in (title + " " + summary).lower():
        why.append("Tendencia de IA con potencial impacto en demanda de casos de uso, plataforma y gobierno.")
    if not why:
        why.append("Señal de mercado potencialmente relevante; revisar impacto en clientes/industria.")

    imp = [
        "Oportunidad para reforzar posicionamiento y propuestas asociadas (según KX y capacidades existentes).",
        "Revisar cuentas/sectores impactados y preparar talking points para clientes."
    ]
    if evid:
        imp.append("Evidencia interna utilizada: " + "; ".join(evid))

    return {
        "descripcion": summary[:260] if summary else title[:260],
        "por_que_importa": why[:4],
        "implicaciones_para_accenture": imp[:4],
        "nivel_confianza": "medium" if kx_hits else "low",
    }
