from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from src import db
from src.classifier import classify, hard_filter
from src.kx_pdf import extract_pdf_text, chunk_pages, save_uploaded_pdf
from src.llm import generate_insights_heuristic, generate_insights_llm, llm_available
from src.news import build_google_news_rss_urls, fetch_newsapi, fetch_rss, fetch_url_snippet
from src.rag import build_kx_index, filter_kx_hits, kx_index_exists, mmr_diversify, search_kx
from src.scoring import score_item
from src.utils import clean_text, env, to_json


# ----------------------------- Streamlit setup -----------------------------
st.set_page_config(page_title="Acn2Agent • News + APIs + KX", layout="wide")

CARD_CSS = """
<style>
/* lightweight card styling */
.kpi {
  display:flex; align-items:center; gap:10px; margin:6px 0 10px 0;
  font-size: 0.95rem;
}
.badge {
  display:inline-block; padding:2px 8px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.15);
  background: rgba(255,255,255,0.05);
  margin-right: 6px;
  font-size: 0.78rem;
}
.muted { color: rgba(255,255,255,0.70); }
hr.soft { border: none; height: 1px; background: rgba(255,255,255,0.10); margin: 10px 0; }
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)


# ----------------------------- Helpers -----------------------------
def impact_bucket(score_0_100: float) -> str:
    if score_0_100 >= 75:
        return "ALTO"
    if score_0_100 >= 50:
        return "MEDIO"
    return "BAJO"


def fmt_dt(dt_iso: str | None) -> str:
    if not dt_iso:
        return "-"
    try:
        if dt_iso.endswith("Z"):
            dt_iso = dt_iso[:-1] + "+00:00"
        dt = datetime.fromisoformat(dt_iso)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return dt_iso[:19]


def chips(label: str, values: List[str]) -> str:
    if not values:
        return ""
    inner = "".join([f'<span class="badge">{v}</span>' for v in values])
    return f"<div class='muted' style='margin-top:6px'><b>{label}:</b> {inner}</div>"


def render_kx_evidence(hits: List[Dict[str, Any]], max_show: int = 3) -> None:
    if not hits:
        st.info("Sin evidencia interna (KX) suficiente para esta noticia.")
        return

    rows = []
    for h in hits[:max_show]:
        rows.append(
            {
                "PDF": h.get("pdf_name") or h.get("doc_name") or "KX",
                "Página": h.get("page"),
                "Sim": round(float(h.get("score", 0.0)), 3),
                "Chunk": (h.get("chunk_id") or "")[:10],
                "Snippet": clean_text(h.get("text", ""))[:280],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_news_card(item: Dict[str, Any]) -> None:
    score = float(item.get("score", 0.0))
    bucket = impact_bucket(score)
    title = item.get("title") or "(sin título)"
    url = item.get("url") or ""
    source = item.get("source") or item.get("provider") or ""
    published = fmt_dt(item.get("published_at"))

    insights = item.get("insights") or {}
    why = insights.get("why_it_matters", []) or []
    impl = insights.get("implications_for_accenture", []) or []
    kx_ctx = insights.get("kx_context", []) or []

    c1, c2 = st.columns([0.72, 0.28], vertical_alignment="top")

    with c1:
        st.markdown(f"### [{title}]({url})" if url else f"### {title}")
        st.markdown(f"<div class='muted'>🗞️ {source} · 🕒 {published}</div>", unsafe_allow_html=True)

        if insights.get("brief"):
            st.write(insights["brief"])

        st.markdown("**Por qué importa**")
        if why:
            for b in why[:4]:
                st.write(f"- {b}")
        else:
            st.write("- (sin explicación)")

        st.markdown("**Implicaciones para Accenture**")
        if impl:
            for b in impl[:4]:
                st.write(f"- {b}")
        else:
            st.write("- (sin implicaciones)")

        if kx_ctx:
            st.markdown("**Contexto KX**")
            for b in kx_ctx[:3]:
                st.write(f"- {b}")

        st.markdown(chips("Tendencias", item.get("classif", {}).get("trends", [])), unsafe_allow_html=True)
        st.markdown(chips("Mercados", item.get("classif", {}).get("markets", [])), unsafe_allow_html=True)
        st.markdown(chips("Servicios", item.get("classif", {}).get("services", [])), unsafe_allow_html=True)

        with st.expander("Ver evidencias KX", expanded=False):
            render_kx_evidence(item.get("kx_hits", []), max_show=4)

        with st.expander("Ver explicación del score", expanded=False):
            st.code(to_json(item.get("score_explain", {})), language="json")

    with c2:
        st.markdown(
            f"<div class='kpi'><b>Impacto</b> {score:.1f}/100 <span class='badge'>{bucket}</span></div>",
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, max(0.0, score / 100.0)))
        pillars = (item.get("score_explain", {}) or {}).get("pillars", {})
        if pillars:
            st.caption("Desglose")
            for k, v in pillars.items():
                st.write(f"- {k}: **{v}**")


# ----------------------------- Cached fetchers -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_rss(urls: Tuple[str, ...], provider: str) -> List[Dict[str, Any]]:
    return fetch_rss(list(urls), provider=provider)


@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_newsapi(query: str, from_date: str | None, language: str, page_size: int) -> List[Dict[str, Any]]:
    return fetch_newsapi(query=query, from_date=from_date, language=language, page_size=page_size)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_snippet(url: str) -> Dict[str, Any]:
    return fetch_url_snippet(url)


# ----------------------------- App state -----------------------------
db.init_db()

if "results" not in st.session_state:
    st.session_state["results"] = []
if "diag" not in st.session_state:
    st.session_state["diag"] = {}


# ----------------------------- Sidebar config -----------------------------
st.sidebar.title("Acn2Agent")

q_default = "Accenture"
query = st.sidebar.text_input("Query principal", value=q_default)

st.sidebar.subheader("Fuentes externas")
use_gnews = st.sidebar.checkbox("Google News RSS", value=True)
hl = st.sidebar.selectbox("Google News hl", ["es", "en"], index=0)
gl = st.sidebar.selectbox("Google News gl", ["ES", "US", "GB"], index=0)
ceid = st.sidebar.text_input("Google News ceid", value=f"{gl}:{hl}")

use_newsapi = st.sidebar.checkbox("NewsAPI (si hay key)", value=False)
newsapi_lang = st.sidebar.selectbox("NewsAPI language", ["en", "es"], index=0)
days_back = st.sidebar.slider("Noticias de los últimos N días (NewsAPI)", 1, 30, 7)

st.sidebar.subheader("RSS adicionales")
rss_text = st.sidebar.text_area(
    "RSS feeds (uno por línea)",
    value="""https://www.accenture.com/us-en/blogs/blogs-rss.xml
https://www.bloomberg.com/feeds/podcasts/etf-report.xml""",
    height=90,
)
rss_feeds = [x.strip() for x in rss_text.splitlines() if x.strip()]

st.sidebar.subheader("Enriquecimiento")
enrich_summary = st.sidebar.checkbox("Enriquecer resumen con snippet (scrape seguro)", value=True)
must_terms = [x.strip() for x in st.sidebar.text_input("Must contain (coma)", value="Accenture").split(",") if x.strip()]

st.sidebar.subheader("KX (PDF) / RAG")
kx_top_k = st.sidebar.slider("Top K chunks KX", 1, 10, 6)
kx_min_score = st.sidebar.slider("Umbral evidencia KX", 0.0, 1.0, 0.35, 0.05)
use_mmr = st.sidebar.checkbox("Diversificar evidencias (MMR)", value=True)

st.sidebar.subheader("Scoring")
acc_kw = st.sidebar.text_input("Keywords (relevancia)", value="accenture, consulting, partnership, acquisition, cloud, security, genai")
ent_kw = st.sidebar.text_input("Entities (clientes/competidores)", value="aws, azure, google, ibm, deloitte, capgemini, tcs, infosys")
acc_keywords = [x.strip().lower() for x in acc_kw.split(",") if x.strip()]
entity_keywords = [x.strip().lower() for x in ent_kw.split(",") if x.strip()]

top_n = st.sidebar.slider("Top N", 5, 50, 15)

st.sidebar.subheader("LLM (opcional)")
use_llm = st.sidebar.checkbox("Usar LLM para insights", value=False)
model = st.sidebar.text_input("Modelo OpenAI", value=env("OPENAI_MODEL", "gpt-4o-mini"))
max_llm_calls = st.sidebar.slider("Máx llamadas LLM", 0, 30, 10)
llm_ok = llm_available()

if use_llm and not llm_ok:
    st.sidebar.warning("No hay OPENAI_API_KEY; se usará heurístico.")

st.sidebar.subheader("KX Index builder")
uploaded = st.sidebar.file_uploader("Sube PDF(s) KX", type=["pdf"], accept_multiple_files=True)
chunk_size = st.sidebar.slider("Chunk size", 400, 2000, 900, 50)
overlap = st.sidebar.slider("Overlap", 0, 400, 180, 20)
build_btn = st.sidebar.button("Construir / Reindexar KX")

run_btn = st.sidebar.button("Run Agent")


# ----------------------------- KX build -----------------------------
if build_btn:
    if not uploaded:
        st.sidebar.error("Sube al menos un PDF")
    else:
        all_chunks: List[Dict[str, Any]] = []
        with st.spinner("Extrayendo y chunking PDFs..."):
            for f in uploaded:
                pdf_path = save_uploaded_pdf(f.getvalue(), f.name)
                pages = extract_pdf_text(pdf_path)
                chunks = chunk_pages(pages, pdf_name=f.name, chunk_size=chunk_size, overlap=overlap)
                all_chunks.extend(chunks)

        with st.spinner("Construyendo índice KX..."):
            info = build_kx_index(all_chunks, doc_name="KX")
        st.sidebar.success(f"KX index listo: {info}")


# ----------------------------- Run pipeline -----------------------------
def run_agent() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "fetched": 0,
        "stored": 0,
        "filtered_out": 0,
        "scored": 0,
        "llm_calls": 0,
        "kx_index": kx_index_exists(),
    }

    items: List[Dict[str, Any]] = []

    if use_gnews:
        gurls = build_google_news_rss_urls(query, hl=hl, gl=gl, ceid=ceid)
        items.extend(cached_fetch_rss(tuple(gurls), provider="gnews"))

    if rss_feeds:
        items.extend(cached_fetch_rss(tuple(rss_feeds), provider="rss"))

    if use_newsapi:
        from_date = (datetime.now(timezone.utc) - timedelta(days=int(days_back))).date().isoformat()
        try:
            items.extend(cached_fetch_newsapi(query, from_date, newsapi_lang, 30))
        except Exception as e:
            diag["newsapi_error"] = str(e)

    diag["fetched"] = len(items)

    # Normalize + persist + enrich
    normalized: List[Dict[str, Any]] = []
    for it in items:
        title = clean_text(it.get("title", ""))
        summary = clean_text(it.get("summary", ""))
        url = clean_text(it.get("url", ""))
        if enrich_summary and (len(summary) < 80) and url:
            sn = cached_snippet(url)
            if sn.get("ok") and sn.get("snippet"):
                summary = clean_text(sn["snippet"])
        norm = {
            "id": it.get("id"),
            "title": title,
            "summary": summary,
            "source": it.get("source"),
            "url": url,
            "published_at": it.get("published_at"),
            "provider": it.get("provider"),
            "raw": it.get("raw", {}),
        }
        normalized.append(norm)
        db.upsert_news(norm)
        diag["stored"] += 1

    results: List[Dict[str, Any]] = []
    for it in normalized:
        passes, matched_terms = hard_filter(it["title"], it["summary"], must_terms)
        if not passes:
            diag["filtered_out"] += 1
            continue

        classif = classify(it["title"], it["summary"])

        # KX retrieval
        query_text = clean_text(f"{it['title']} {it['summary']}")
        kx_hits = []
        if kx_index_exists():
            kx_hits = search_kx(query_text, top_k=kx_top_k, min_score=0.0)
            kx_hits = filter_kx_hits(kx_hits)
            if use_mmr:
                kx_hits = mmr_diversify(kx_hits, top_k=min(4, len(kx_hits)))

        # Score
        score, explain = score_item(
            it["title"],
            it["summary"],
            it.get("published_at"),
            acc_keywords=acc_keywords,
            entity_keywords=entity_keywords,
            kx_hits=kx_hits,
            classif=classif,
            kx_min_evidence=kx_min_score,
        )
        kx_gated = bool((explain.get("signals", {}) or {}).get("kx_gated", False))

        # Insights
        insights: Dict[str, Any]
        if use_llm and llm_ok and diag["llm_calls"] < max_llm_calls:
            insights = generate_insights_llm(
                title=it["title"],
                summary=it["summary"],
                url=it.get("url", ""),
                kx_hits=kx_hits,
                kx_gated=kx_gated,
                model=model,
            )
            diag["llm_calls"] += 1
            if not insights.get("brief"):
                insights = generate_insights_heuristic(it["title"], it["summary"], kx_hits, kx_gated)
        else:
            insights = generate_insights_heuristic(it["title"], it["summary"], kx_hits, kx_gated)

        results.append(
            {
                **it,
                "matched_terms": matched_terms,
                "classif": classif,
                "kx_hits": kx_hits,
                "score": score,
                "score_explain": explain,
                "insights": insights,
            }
        )
        diag["scored"] += 1

    results.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return results[: int(top_n)], diag


if run_btn:
    with st.spinner("Ejecutando agente..."):
        results, diag = run_agent()
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    params = {
        "query": query,
        "use_gnews": use_gnews,
        "use_newsapi": use_newsapi,
        "rss_feeds": rss_feeds,
        "must_terms": must_terms,
        "kx": {"top_k": kx_top_k, "min_score": kx_min_score, "use_mmr": use_mmr},
        "top_n": top_n,
        "use_llm": use_llm,
        "model": model,
    }
    db.save_run(run_id, params, results)
    st.session_state["results"] = results
    st.session_state["diag"] = diag
    st.success(f"Listo: {len(results)} noticias (run_id={run_id})")


# ----------------------------- Main UI -----------------------------
st.title("Acn2Agent • News + APIs + KX")
st.caption("Agente de noticias relevantes para Accenture con enriquecimiento KX (PDF) y ranking por impacto.")

results: List[Dict[str, Any]] = st.session_state.get("results", []) or []
diag: Dict[str, Any] = st.session_state.get("diag", {}) or {}

tab1, tab2, tab3 = st.tabs(["Resumen", "Detalle", "Diagnóstico"])

with tab1:
    if not results:
        st.info("Ejecuta 'Run Agent' desde la barra lateral.")
    else:
        for i, it in enumerate(results, start=1):
            st.markdown(f"## {i}. {impact_bucket(float(it.get('score',0)))} · {float(it.get('score',0)):.1f}/100")
            render_news_card(it)
            st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

with tab2:
    if not results:
        st.info("Sin resultados.")
    else:
        df = pd.DataFrame(
            [
                {
                    "score": round(float(r.get("score", 0.0)), 2),
                    "impact": impact_bucket(float(r.get("score", 0.0))),
                    "published": fmt_dt(r.get("published_at")),
                    "title": r.get("title"),
                    "source": r.get("source") or r.get("provider"),
                    "url": r.get("url"),
                    "markets": ", ".join((r.get("classif", {}) or {}).get("markets", [])),
                    "services": ", ".join((r.get("classif", {}) or {}).get("services", [])),
                    "trends": ", ".join((r.get("classif", {}) or {}).get("trends", [])),
                    "kx_best": (r.get("score_explain", {}) or {}).get("signals", {}).get("kx_best", 0.0),
                }
                for r in results
            ]
        )

        st.dataframe(df, use_container_width=True, hide_index=True)

        pick = st.selectbox(
            "Ver detalle",
            options=list(range(len(results))),
            format_func=lambda i: results[i].get("title", "(sin título)")[:80],
        )
        item = results[int(pick)]
        st.subheader(item.get("title", "(sin título)"))
        if item.get("url"):
            st.write(item.get("url"))
        st.write(item.get("summary", ""))

        st.markdown("### Evidencias KX")
        render_kx_evidence(item.get("kx_hits", []), max_show=6)

        st.markdown("### Output estructurado")
        st.code(to_json(item.get("insights", {})), language="json")

with tab3:
    st.subheader("Diagnóstico")
    st.code(to_json(diag), language="json")
    st.subheader("KX Index")
    st.write("Existe índice KX:", kx_index_exists())
