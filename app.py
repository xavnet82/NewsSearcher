# app.py
from __future__ import annotations

import html
import re
import uuid
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from src import db
from src.classifier import classify, hard_filter
from src.kx_pdf import chunk_text, extract_pdf_text, save_uploaded_pdf
from src.llm import generate_insights_heuristic, generate_insights_llm, llm_available
from src.news import fetch_newsapi, fetch_rss, fetch_url_snippet
from src.rag import build_kx_index, kx_index_exists, search_kx
from src.scoring import (
    DEFAULT_ENTITIES,
    DEFAULT_KEYWORDS,
    score_dimensions_boost,
    score_news_item,
)
from src.utils import clean_text, safe_json, sha1


# --- Streamlit context guard (reduce SessionInfo issues) ---
def has_st_ctx() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def ss_get(key: str, default):
    if not has_st_ctx():
        return default
    try:
        if key not in st.session_state:
            st.session_state[key] = default
        return st.session_state[key]
    except Exception:
        return default


def ss_set(key: str, value) -> None:
    if not has_st_ctx():
        return
    try:
        st.session_state[key] = value
    except Exception:
        pass


def ui_progress(initial: float = 0.0):
    if not has_st_ctx():
        return None
    try:
        return st.progress(initial)
    except Exception:
        return None


# ---------------- init ----------------
load_dotenv()

st.set_page_config(page_title="Acn2Agent · Accenture News + KX (PDF)", layout="wide")
st.title("Acn2Agent · Accenture News + KX (PDF)")
st.caption("Buscar (Google News RSS) → enriquecer (KX) → filtrar → rankear → output estructurado")

# ---------------- global CSS (wrap URLs, tidy cards) ----------------
st.markdown(
    """
<style>
/* Evita que URLs largas rompan el layout */
.acn-card, .acn-card * {
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}

.acn-card {
  border: 1px solid rgba(49,51,63,0.20);
  border-radius: 14px;
  padding: 14px 16px;
  margin: 10px 0;
  background: rgba(255,255,255,0.02);
}

.acn-card-title {
  font-weight: 800;
  font-size: 16px;
  margin-bottom: 8px;
}

.acn-chip {
  display: inline-block;
  padding: 2px 8px;
  margin-right: 6px;
  border-radius: 999px;
  border: 1px solid rgba(49,51,63,0.20);
  font-size: 12px;
  opacity: 0.85;
}

.acn-small {
  font-size: 12px;
  opacity: 0.85;
}

.acn-kv {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

@media (max-width: 900px) {
  .acn-kv { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- UI helpers ----------------
def card(title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div class="acn-card">
          <div class="acn-card-title">{html.escape(title)}</div>
          <div>{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bullets(items: List[str]) -> str:
    if not items:
        return "—"
    safe_lines: List[str] = []
    for x in items:
        s = str(x).replace("\n", " ").strip()
        safe_lines.append(f"• {html.escape(s)}")
    return "<br/>".join(safe_lines)


def shorten_url(u: str, max_len: int = 80) -> str:
    u = (u or "").strip()
    if len(u) <= max_len:
        return u
    return u[: max_len - 3] + "..."


def link_html(url: str) -> str:
    if not url:
        return "—"
    safe_u = html.escape(url)
    label = html.escape(shorten_url(url))
    return f'<a href="{safe_u}" target="_blank" rel="noopener noreferrer">{label}</a>'


def build_google_news_rss_urls(queries: List[str], hl: str, gl: str, ceid: str) -> List[str]:
    urls: List[str] = []
    for q in queries:
        q_enc = requests.utils.quote(q)
        urls.append(f"https://news.google.com/rss/search?q={q_enc}&hl={hl}&gl={gl}&ceid={ceid}")
    return urls


# ---------------- KX: heuristic to drop TOC/index-like chunks ----------------
_RE_MANY_DOTS = re.compile(r"\.{4,}")
_RE_PAGE_LIST = re.compile(r"\b\d{1,4}\b(?:\s*[-–—]\s*\b\d{1,4}\b)?")
_RE_HEADERS = re.compile(r"^(contents|table of contents|índice|indice|index)\b", re.IGNORECASE)


def is_toc_like(text: str) -> bool:
    """
    Heurística simple para detectar secciones tipo índice/TOC:
    - muchas líneas cortas
    - muchos números (páginas)
    - patrones de puntos (......)
    - keywords típicas (Índice, Contents)
    """
    if not text:
        return False
    t = text.strip()
    if len(t) < 180:
        return False

    head = t[:200].strip().lower()
    if _RE_HEADERS.search(head):
        return True

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 8:
        short_lines = sum(1 for ln in lines if len(ln) <= 60)
        if short_lines / max(1, len(lines)) > 0.75:
            # si además hay muchos números/páginas, casi seguro es TOC
            nums = len(_RE_PAGE_LIST.findall(t))
            if nums >= 10:
                return True

    dots = len(_RE_MANY_DOTS.findall(t))
    if dots >= 2:
        return True

    # muchos números respecto a texto
    nums = len(_RE_PAGE_LIST.findall(t))
    if nums >= 20 and len(t) < 2500:
        return True

    return False


def filter_kx_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not hits:
        return hits
    filtered = [h for h in hits if not is_toc_like(h.get("text", "") or "")]
    return filtered if filtered else hits  # si nos quedamos sin nada, devolvemos original


# ---------------- sidebar config ----------------
st.sidebar.header("Configuración")

st.sidebar.subheader("Google News (RSS por búsqueda)")
use_gnews = st.sidebar.checkbox("Usar Google News Search RSS", value=True)

gnews_queries_txt = st.sidebar.text_area(
    "Queries (una por línea)",
    value=(
        "Accenture\n"
        "Accenture AI\n"
        "Accenture generative AI\n"
        "Accenture OpenAI\n"
        "Accenture Microsoft\n"
        "Accenture AWS\n"
        "Accenture cybersecurity\n"
        "Accenture public sector\n"
        "Accenture banking\n"
        "Accenture acquisition"
    ),
    height=170,
)
gnews_queries = [x.strip() for x in gnews_queries_txt.splitlines() if x.strip()]

gnews_region = st.sidebar.selectbox("Región Google News", ["ES (español)", "US (english)", "GB (english)"], index=0)
if gnews_region == "ES (español)":
    hl, gl, ceid = "es", "ES", "ES:es"
elif gnews_region == "US (english)":
    hl, gl, ceid = "en-US", "US", "US:en"
else:
    hl, gl, ceid = "en-GB", "GB", "GB:en"

st.sidebar.subheader("RSS generales (opcional)")
default_feeds = [
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://www.theverge.com/rss/index.xml",
]
feeds_text = st.sidebar.text_area("RSS feeds (uno por línea)", value="\n".join(default_feeds), height=120)
rss_feeds = [f.strip() for f in feeds_text.splitlines() if f.strip()]

st.sidebar.subheader("NewsAPI (opcional)")
use_newsapi = st.sidebar.checkbox("Usar NewsAPI", value=False)
newsapi_query = st.sidebar.text_input(
    "NewsAPI query",
    value="Accenture OR ACN OR (Accenture AND AI) OR (Accenture AND cloud) OR (Accenture AND consulting)",
)
news_limit = st.sidebar.slider("Máx noticias por fuente", 10, 200, 50, 10)

st.sidebar.divider()
st.sidebar.subheader("KX (PDF)")
chunk_size = st.sidebar.slider("Tamaño chunk (chars)", 400, 1600, 900, 100)
chunk_overlap = st.sidebar.slider("Overlap (chars)", 0, 400, 150, 25)
drop_toc = st.sidebar.checkbox("Excluir secciones tipo Índice/TOC del KX", value=True)

st.sidebar.divider()
st.sidebar.subheader("Scoring base (0–100)")
w_recency = st.sidebar.slider("Peso recencia", 0.0, 1.0, 0.25, 0.05)
w_entity = st.sidebar.slider("Peso entidades", 0.0, 1.0, 0.25, 0.05)
w_kw = st.sidebar.slider("Peso keywords", 0.0, 1.0, 0.20, 0.05)
w_kx = st.sidebar.slider("Peso similitud KX", 0.0, 1.0, 0.30, 0.05)
weights = {"recency": w_recency, "entity": w_entity, "keyword": w_kw, "kx": w_kx}

keywords = st.sidebar.text_area("Keywords (uno por línea)", value="\n".join(DEFAULT_KEYWORDS), height=140)
kw_list = [k.strip() for k in keywords.splitlines() if k.strip()]

entities = st.sidebar.text_area("Entidades (uno por línea)", value="\n".join(DEFAULT_ENTITIES), height=120)
ent_list = [e.strip() for e in entities.splitlines() if e.strip()]

st.sidebar.divider()
st.sidebar.subheader("Filtros / ranking")
top_k_kx = st.sidebar.slider("Top K evidencias KX por noticia", 1, 10, 5, 1)
top_n = st.sidebar.slider("Top N noticias en ranking", 5, 50, 20, 5)
enrich_with_url = st.sidebar.checkbox("Intentar extraer snippet desde URL si falta resumen", value=True)

st.sidebar.subheader("Filtro mínimo (hard filter)")
st.sidebar.caption("Si se queda en 0, deja esto vacío para depurar.")
must_terms_txt = st.sidebar.text_area(
    "Debe contener al menos uno (uno por línea). Vacío = sin filtro.",
    value="Accenture\nACN",
    height=90,
)
must_terms = [x.strip() for x in must_terms_txt.splitlines() if x.strip()]

st.sidebar.divider()
st.sidebar.subheader("LLM (opcional)")
use_llm = st.sidebar.checkbox("Generar insights con LLM (requiere OPENAI_API_KEY)", value=False)
st.sidebar.caption(f"LLM disponible: {'Sí' if llm_available() else 'No'}")
MAX_LLM_CALLS = st.sidebar.slider("Máx llamadas LLM por ejecución", 0, 20, 5, 1)

run_agent = st.sidebar.button("▶ Run Agent", type="primary")


# ---------------- KX upload / build index ----------------
st.subheader("1) Cargar KX (PDF) y construir índice")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("Sube uno o varios PDFs (KX)", type=["pdf"], accept_multiple_files=True)

with col2:
    st.metric("Índice KX existe", "Sí" if kx_index_exists() else "No")

if uploaded and st.button("📚 Construir / Re-construir índice KX", type="secondary"):
    all_chunks: List[Dict[str, Any]] = []
    dropped = 0

    for f in uploaded:
        path = save_uploaded_pdf(f.getvalue(), f.name)
        pages = extract_pdf_text(path)
        chunks = chunk_text(pages, chunk_size=chunk_size, overlap=chunk_overlap)

        for c in chunks:
            c["doc_name"] = f.name
            if drop_toc and is_toc_like(c.get("text", "") or ""):
                dropped += 1
                continue
            all_chunks.append(c)

    if not all_chunks:
        st.error("No se extrajo texto útil de los PDFs (o todo se filtró como TOC).")
    else:
        try:
            meta = build_kx_index(
                chunks=[{"chunk_id": c["chunk_id"], "page": c["page"], "text": c["text"]} for c in all_chunks],
                doc_name="KX_MULTI_PDF",
            )
            st.success(
                f"Índice KX construido. Backend: {meta.get('backend','?')} · Chunks: {meta.get('chunks')} · Filtrados TOC: {dropped}"
            )
        except Exception as e:
            st.error(f"Error construyendo índice KX: {e}")

st.divider()
st.subheader("2) Ejecutar agente (news + KX)")


# ---------------- pipeline helpers ----------------
def normalize_and_store(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    db.init_db()

    stored: List[Dict[str, Any]] = []
    for it in news_items:
        title = clean_text(it.get("title", ""))
        url = it.get("url", "")
        summary = clean_text(it.get("summary", ""))

        if enrich_with_url and (not summary or len(summary) < 40):
            snippet = fetch_url_snippet(url)
            if snippet:
                summary = snippet

        norm = {
            "id": it.get("id") or sha1(url or title),
            "title": title,
            "summary": summary,
            "source": clean_text(it.get("source", "")),
            "url": url,
            "published_at": it.get("published_at"),
            "provider": it.get("provider", "unknown"),
        }

        if not db.news_exists(norm["id"]):
            db.upsert_news(norm)

        stored.append(norm)
    return stored


def ensure_structured_output(insights: Any, item: Dict[str, Any], kx_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normaliza el output para que SIEMPRE tenga las secciones del reto.
    """
    out: Dict[str, Any] = insights if isinstance(insights, dict) else {}

    # claves esperadas
    out.setdefault("descripcion_breve", clean_text(item.get("summary", ""))[:350] or "—")
    out.setdefault("por_que_importa", out.get("por_que_importa") or [])
    out.setdefault("implicaciones_para_accenture", out.get("implicaciones_para_accenture") or [])

    # KX enrichment: si el LLM/heurística no lo puso, añadimos un básico desde hits
    if "kx_enriquecimiento" not in out:
        evs = []
        for h in (kx_hits or [])[:3]:
            evs.append(
                {
                    "doc": h.get("doc_name", "KX"),
                    "page": h.get("page", "?"),
                    "score": float(h.get("score", 0.0)),
                    "snippet": clean_text(h.get("text", "") or "")[:700],
                }
            )
        resumen_ctx = ""
        if evs:
            resumen_ctx = "Se han recuperado evidencias relevantes del KX para contextualizar la noticia."
        out["kx_enriquecimiento"] = {"resumen_contexto": resumen_ctx, "evidencias": evs}

    # Link original
    out.setdefault("link", item.get("url"))

    return out


def run_pipeline() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    llm_calls = 0

    diag: Dict[str, Any] = {
        "rss_feeds_count": len(rss_feeds),
        "google_news_enabled": bool(use_gnews),
        "google_news_region": {"hl": hl, "gl": gl, "ceid": ceid},
        "google_news_queries_count": len(gnews_queries),
        "rss_items": 0,
        "google_news_items": 0,
        "newsapi_items": 0,
        "ingested_total": 0,
        "kept_after_filter": 0,
        "dropped_after_filter": 0,
        "drop_reasons": {},
        "kx_index_exists": bool(kx_index_exists()),
        "must_terms": must_terms,
        "use_llm": bool(use_llm),
        "llm_available": bool(llm_available()),
        "max_llm_calls": int(MAX_LLM_CALLS),
        "llm_calls_used": 0,
    }

    rss_items = fetch_rss(rss_feeds, max_items=news_limit) if rss_feeds else []
    diag["rss_items"] = len(rss_items)

    gnews_items: List[Dict[str, Any]] = []
    if use_gnews and gnews_queries:
        gnews_urls = build_google_news_rss_urls(gnews_queries, hl=hl, gl=gl, ceid=ceid)
        gnews_items = fetch_rss(gnews_urls, max_items=news_limit)
    diag["google_news_items"] = len(gnews_items)

    api_items = fetch_newsapi(newsapi_query, page_size=news_limit) if use_newsapi else []
    diag["newsapi_items"] = len(api_items)

    ingested = normalize_and_store(rss_items + gnews_items + api_items)
    diag["ingested_total"] = len(ingested)

    results: List[Dict[str, Any]] = []

    pbar = ui_progress(0.0)
    total = max(1, len(ingested))

    for i, it in enumerate(ingested):
        if pbar is not None:
            try:
                pbar.progress(min(1.0, (i + 1) / total))
            except Exception:
                pass

        keep, drop_reason = hard_filter(it, must_have_any=must_terms)
        if not keep:
            diag["dropped_after_filter"] += 1
            diag["drop_reasons"][drop_reason] = diag["drop_reasons"].get(drop_reason, 0) + 1
            continue

        diag["kept_after_filter"] += 1
        query = f"{it.get('title','')} {it.get('summary','')}".strip()

        # KX enrichment
        try:
            kx_hits = search_kx(query, top_k=top_k_kx) if kx_index_exists() else []
            kx_hits = filter_kx_hits(kx_hits)
        except Exception as e:
            diag["kx_error"] = str(e)
            kx_hits = []

        # Clasificación
        classif = classify(it)

        # Score
        base_score, base_comps, tags = score_news_item(it, kx_hits, weights, kw_list, ent_list)
        dim_scores = score_dimensions_boost(classif, boosts={})
        final_score = 0.75 * base_score + 0.25 * (
            0.25 * dim_scores["scope"]
            + 0.25 * dim_scores["trend"]
            + 0.25 * dim_scores["market"]
            + 0.25 * dim_scores["service"]
        )

        # Insights
        use_llm_now = use_llm and llm_available() and llm_calls < int(MAX_LLM_CALLS)
        try:
            if use_llm_now:
                insights = generate_insights_llm(it, kx_hits)
                llm_calls += 1
            else:
                insights = generate_insights_heuristic(it, kx_hits)
        except Exception as e:
            diag["llm_error"] = str(e)
            insights = generate_insights_heuristic(it, kx_hits)

        diag["llm_calls_used"] = llm_calls

        output = ensure_structured_output(insights, it, kx_hits)

        results.append(
            {
                **it,
                "score": float(final_score),
                "components": {**base_comps, **dim_scores},
                "classification": classif,
                "tags": tags,
                "output": output,
                "kx_evidence": kx_hits,
            }
        )

    if pbar is not None:
        try:
            pbar.empty()
        except Exception:
            pass

    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return results[:top_n], diag


# ---------------- Run ----------------
if run_agent:
    if (not rss_feeds) and (not use_gnews) and (not use_newsapi):
        st.error("Activa al menos una fuente: Google News RSS, RSS generales o NewsAPI.")
    else:
        try:
            with st.spinner("Ejecutando agente..."):
                results, diag = run_pipeline()
        except Exception as e:
            st.error(f"Error ejecutando pipeline: {e}")
            results, diag = [], {"pipeline_error": str(e)}

        run_id = str(uuid.uuid4())

        params = {
            "rss_feeds": rss_feeds,
            "use_gnews": use_gnews,
            "gnews_region": {"hl": hl, "gl": gl, "ceid": ceid},
            "gnews_queries": gnews_queries,
            "use_newsapi": use_newsapi,
            "newsapi_query": newsapi_query,
            "news_limit": news_limit,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "drop_toc": drop_toc,
            "weights": weights,
            "top_k_kx": top_k_kx,
            "top_n": top_n,
            "must_terms": must_terms,
            "use_llm": use_llm,
            "max_llm_calls": int(MAX_LLM_CALLS),
        }

        try:
            db.save_run(run_id, params, results)
        except Exception as e:
            diag["db_save_error"] = str(e)

        ss_set("last_results", results)
        ss_set("last_run_id", run_id)
        ss_set("last_diag", diag)


# ---------------- Results UI ----------------
results = ss_get("last_results", [])
run_id = ss_get("last_run_id", None)
diag = ss_get("last_diag", None)

tab_resumen, tab_detalle, tab_diag = st.tabs(["Resumen", "Detalle", "Diagnóstico"])

with tab_diag:
    if diag:
        st.json(diag)
    else:
        st.info("Ejecuta el agente para ver diagnóstico.")

with tab_resumen:
    if not results:
        st.warning("No hay noticias para mostrar. Pulsa 'Run Agent' y revisa el diagnóstico.")
    else:
        st.success(f"Run completado. run_id={run_id}")

        # Página resumen: una “tarjeta por noticia” alineada al reto
        for i, item in enumerate(results, start=1):
            out = item.get("output") or {}
            classif = item.get("classification") or {}
            comps = item.get("components") or {}

            title = item.get("title", "")
            score = float(item.get("score", 0.0))
            src = item.get("source", "")
            published = item.get("published_at", "")
            url = item.get("url", "")

            st.markdown(f"### {i}. {html.escape(title)}")
            st.markdown(
                f"""
<div class="acn-small">
  <span class="acn-chip">Score: {score:.1f}</span>
  <span class="acn-chip">Fuente: {html.escape(src)}</span>
  <span class="acn-chip">Fecha: {html.escape(str(published))}</span>
</div>
""",
                unsafe_allow_html=True,
            )

            # Secciones del reto (IA)
            card("Descripción breve", html.escape(out.get("descripcion_breve", "") or "—"))
            card("Por qué importa", bullets([str(x) for x in (out.get("por_que_importa") or [])]))
            card(
                "Implicaciones potenciales para Accenture",
                bullets([str(x) for x in (out.get("implicaciones_para_accenture") or [])]),
            )
            card("Link a la fuente original", link_html(url))

            # Enriquecimiento KX (en resumen)
            kx_enr = out.get("kx_enriquecimiento") or {}
            kx_ctx = (kx_enr.get("resumen_contexto") or "").strip() or "—"
            card("Enriquecimiento KX (contexto interno)", html.escape(kx_ctx))

            evs = kx_enr.get("evidencias") or []
            if evs:
                ev_html = []
                for e in evs[:5]:
                    doc = html.escape(str(e.get("doc", "KX")))
                    page = html.escape(str(e.get("page", "?")))
                    sc = float(e.get("score", 0.0))
                    sn = html.escape(str(e.get("snippet", ""))[:500])
                    ev_html.append(f"<div><b>{doc}</b> · page {page} · sim {sc:.3f}<br/>{sn}</div>")
                card("Evidencias KX citadas (top)", "<br/><br/>".join(ev_html))
            else:
                card("Evidencias KX citadas (top)", "—")

            # Mini resumen de clasificación/score como “drivers”
            cls_html = f"""
<div class="acn-kv">
  <div><b>Alcance (scope):</b> {html.escape(str(classif.get("scope","—")))}</div>
  <div><b>Tendencias:</b> {html.escape(", ".join(classif.get("trends", []) or []) or "—")}</div>
  <div><b>Mercados:</b> {html.escape(", ".join(classif.get("markets", []) or []) or "—")}</div>
  <div><b>Servicios:</b> {html.escape(", ".join(classif.get("services", []) or []) or "—")}</div>
</div>
"""
            card("Clasificación (drivers)", cls_html)

            # Separador
            st.divider()

with tab_detalle:
    if not results:
        st.info("Ejecuta el agente para ver el detalle.")
    else:
        st.subheader("Ranking (tabla)")
        df = pd.DataFrame(
            [
                {
                    "score": r.get("score", 0.0),
                    "title": r.get("title", ""),
                    "source": r.get("source", ""),
                    "published_at": r.get("published_at", ""),
                    "url": r.get("url", ""),
                    "provider": r.get("provider", ""),
                }
                for r in results
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Detalle por noticia")
        titles = [f"{i+1}. [{r.get('score', 0):.1f}] {r.get('title', '')[:110]}" for i, r in enumerate(results)]
        sel = st.selectbox("Selecciona una noticia", options=list(range(len(results))), format_func=lambda i: titles[i])
        item = results[int(sel)]

        left, right = st.columns([2, 1])

        with left:
            st.markdown(f"### {item.get('title','')}")
            st.write(
                f"**Fuente:** {item.get('source','')}  ·  **Fecha:** {item.get('published_at','')}  ·  **Proveedor:** {item.get('provider','')}"
            )
            if item.get("url"):
                st.link_button("Abrir noticia", item["url"])

            out = item.get("output") or {}

            st.markdown("#### Resumen estructurado (reto)")
            card("Descripción breve", html.escape(out.get("descripcion_breve", "") or "—"))
            card("Por qué importa", bullets([str(x) for x in (out.get("por_que_importa") or [])]))
            card(
                "Implicaciones potenciales para Accenture",
                bullets([str(x) for x in (out.get("implicaciones_para_accenture") or [])]),
            )
            card("Link a la fuente original", link_html(item.get("url", "")))

            st.markdown("#### Output JSON (debug)")
            st.code(safe_json(out), language="json")

        with right:
            st.markdown("#### Evidencias KX (top-k) recuperadas")
            kx = item.get("kx_evidence", [])
            if not kx:
                st.info("No hay evidencias KX (o el índice KX no está construido).")
            else:
                kx = filter_kx_hits(kx)
                for h in kx:
                    st.markdown(
                        f"**Sim:** {h.get('score', 0):.3f} · **Doc:** {h.get('doc_name', 'KX')} · **Page:** {h.get('page', '?')}"
                    )
                    st.caption((h.get("text", "") or "")[:700])

        st.subheader("Export")
        st.download_button(
            "Descargar resultados (JSON)",
            data=safe_json(results),
            file_name=f"acn2agent_results_{run_id}.json",
            mime="application/json",
        )
