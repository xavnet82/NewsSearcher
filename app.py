from __future__ import annotations

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

# ---------------- init ----------------
load_dotenv()
db.init_db()

st.set_page_config(page_title="Acn2Agent · Accenture News + KX (PDF)", layout="wide")

st.title("Acn2Agent · Accenture News + KX (PDF)")
st.caption("Buscar (Google News RSS) → enriquecer (KX) → filtrar → rankear → output estructurado")


# ---------------- UI helpers ----------------
def card(title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(49,51,63,0.20);
            border-radius: 14px;
            padding: 14px 16px;
            margin: 8px 0;
            background: rgba(255,255,255,0.02);
        ">
          <div style="font-weight: 700; font-size: 16px; margin-bottom: 8px;">{title}</div>
          <div style="font-size: 14px; line-height: 1.45;">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_google_news_rss_urls(queries: List[str], hl: str, gl: str, ceid: str) -> List[str]:
    urls: List[str] = []
    for q in queries:
        q_enc = requests.utils.quote(q)
        urls.append(f"https://news.google.com/rss/search?q={q_enc}&hl={hl}&gl={gl}&ceid={ceid}")
    return urls


# ---------------- sidebar config ----------------
st.sidebar.header("Configuración")

# 1) Google News RSS (búsqueda)
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

# 2) RSS generales (opcional)
st.sidebar.subheader("RSS generales (opcional)")
default_feeds = [
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://www.theverge.com/rss/index.xml",
]
feeds_text = st.sidebar.text_area("RSS feeds (uno por línea)", value="\n".join(default_feeds), height=120)
rss_feeds = [f.strip() for f in feeds_text.splitlines() if f.strip()]

# 3) NewsAPI (opcional)
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

# Hard filter
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
    for f in uploaded:
        path = save_uploaded_pdf(f.getvalue(), f.name)
        pages = extract_pdf_text(path)
        chunks = chunk_text(pages, chunk_size=chunk_size, overlap=chunk_overlap)
        for c in chunks:
            c["doc_name"] = f.name
        all_chunks.extend(chunks)

    if not all_chunks:
        st.error("No se extrajo texto de los PDFs. (¿Son escaneados? En ese caso se requeriría OCR.)")
    else:
        meta = build_kx_index(
            chunks=[{"chunk_id": c["chunk_id"], "page": c["page"], "text": c["text"]} for c in all_chunks],
            doc_name="KX_MULTI_PDF",
        )
        st.success(f"Índice KX construido. Chunks: {meta['chunks']} · Dim: {meta['dim']}")

st.divider()

# ---------------- Agent pipeline ----------------
st.subheader("2) Ejecutar agente (news + KX)")


def normalize_and_store(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def run_pipeline() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    MAX_LLM_CALLS = 5
    llm_calls = 0   # ← DEFINICIÓN CORRECTA AQUÍ
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
    }

    # 1) Ingesta
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

    # 2) Filtrado + enriquecimiento + scoring
    results: List[Dict[str, Any]] = []

    for it in ingested:
        keep, drop_reason = hard_filter(it, must_have_any=must_terms)
        if not keep:
            diag["dropped_after_filter"] += 1
            diag["drop_reasons"][drop_reason] = diag["drop_reasons"].get(drop_reason, 0) + 1
            continue

        diag["kept_after_filter"] += 1

        query = f"{it.get('title','')} {it.get('summary','')}".strip()

        # KX enrichment
        kx_hits = search_kx(query, top_k=top_k_kx) if kx_index_exists() else []

        # Clasificación
        classif = classify(it)

        # Score base
        base_score, base_comps, tags = score_news_item(it, kx_hits, weights, kw_list, ent_list)

        # Score dimensiones
        dim_scores = score_dimensions_boost(classif, boosts={})

        # Score final
        final_score = 0.75 * base_score + 0.25 * (
            0.25 * dim_scores["scope"]
            + 0.25 * dim_scores["trend"]
            + 0.25 * dim_scores["market"]
            + 0.25 * dim_scores["service"]
        )

        # Explicación estructurada (LLM o heurística)
        try:
            use_llm_now = (
                use_llm
                and llm_available()
                and llm_calls < MAX_LLM_CALLS
            )
            
            if use_llm_now:
                try:
                    insights = generate_insights_llm(it, kx_hits)
                    llm_calls += 1
                except Exception as e:
                    insights = generate_insights_heuristic(it, kx_hits)
            else:
                insights = generate_insights_heuristic(it, kx_hits)

        except Exception as e:
            # Si el LLM falla, no rompemos el run; degradamos a heurístico y lo reflejamos en diag
            diag["llm_error"] = str(e)
            insights = generate_insights_heuristic(it, kx_hits)

        descripcion = insights.get("descripcion", "")
        por_que = insights.get("por_que_importa", [])
        implicaciones = insights.get("implicaciones_para_accenture", [])

        # Blindaje de tipos
        if not isinstance(por_que, list):
            por_que = [str(por_que)] if por_que else []
        if not isinstance(implicaciones, list):
            implicaciones = [str(implicaciones)] if implicaciones else []

        results.append(
            {
                **it,
                "score": float(final_score),
                "components": {**base_comps, **dim_scores},
                "classification": classif,
                "tags": tags,
                "output": {
                    "descripcion_breve": descripcion,
                    "por_que_importa": por_que,
                    "implicaciones_para_accenture": implicaciones,
                    "link": it.get("url"),
                },
                "kx_evidence": kx_hits,
            }
        )

    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return results[:top_n], diag


# ---------------- Run ----------------
if run_agent:
    if (not rss_feeds) and (not use_gnews) and (not use_newsapi):
        st.error("Activa al menos una fuente: Google News RSS, RSS generales o NewsAPI.")
    else:
        with st.spinner("Ejecutando agente..."):
            results, diag = run_pipeline()
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
                "weights": weights,
                "top_k_kx": top_k_kx,
                "top_n": top_n,
                "must_terms": must_terms,
                "use_llm": use_llm,
            }

            db.save_run(run_id, params, results)

            st.session_state["last_results"] = results
            st.session_state["last_run_id"] = run_id
            st.session_state["last_diag"] = diag


# ---------------- Results UI ----------------
results = st.session_state.get("last_results", [])
run_id = st.session_state.get("last_run_id", None)
diag = st.session_state.get("last_diag", None)

if diag:
    with st.expander("Diagnóstico de ingesta / filtros", expanded=True):
        st.json(diag)

if results:
    st.success(f"Run completado. run_id={run_id}")

    df = pd.DataFrame(
        [
            {
                "score": r.get("score", 0.0),
                "title": r.get("title", ""),
                "source": r.get("source", ""),
                "published_at": r.get("published_at", ""),
                "scope": (r.get("classification") or {}).get("scope", ""),
                "trends": ", ".join((r.get("classification") or {}).get("trends", [])),
                "markets": ", ".join((r.get("classification") or {}).get("markets", [])),
                "services": ", ".join((r.get("classification") or {}).get("services", [])),
                "url": r.get("url", ""),
                "provider": r.get("provider", ""),
            }
            for r in results
        ]
    )

    st.subheader("Ranking")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Detalle")
    titles = [f"{i+1}. [{r.get('score', 0):.1f}] {r.get('title', '')[:110]}" for i, r in enumerate(results)]
    idx = st.selectbox("Selecciona una noticia", options=list(range(len(results))), format_func=lambda i: titles[i])

    item = results[idx]
    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"### {item.get('title','')}")
        st.write(
            f"**Fuente:** {item.get('source','')}  ·  **Fecha:** {item.get('published_at','')}  ·  **Proveedor:** {item.get('provider','')}"
        )
        if item.get("url"):
            st.link_button("Abrir noticia", item["url"])

        st.markdown("#### Output (resumen estructurado)")
        output = item.get("output") or {}
        out_struct = {
            "titulo": item.get("title", ""),
            "descripcion_breve": output.get("descripcion_breve", ""),
            "por_que_importa": output.get("por_que_importa", []),
            "implicaciones_para_accenture": output.get("implicaciones_para_accenture", []),
            "link": output.get("link") or item.get("url"),
            "score_impacto": item.get("score", 0.0),
        }
        st.code(safe_json(out_struct), language="json")

        # Cards: Clasificación + principales drivers del score
        classif = item.get("classification") or {}
        comps = item.get("components") or {}

        c1, c2 = st.columns(2)

        with c1:
            trends = classif.get("trends", [])
            markets = classif.get("markets", [])
            services = classif.get("services", [])
            scope = classif.get("scope", "")

            body = (
                f"<b>Alcance:</b> {scope or '—'}<br/>"
                f"<b>Tendencias:</b> {', '.join(trends) if trends else '—'}<br/>"
                f"<b>Mercados:</b> {', '.join(markets) if markets else '—'}<br/>"
                f"<b>Servicios:</b> {', '.join(services) if services else '—'}<br/>"
            )
            card("Clasificación", body)

        with c2:
            keys_order = ["recency", "entity", "keyword", "kx", "scope", "trend", "market", "service"]
            lines: List[Tuple[str, float]] = []
            for k in keys_order:
                if k in comps:
                    try:
                        lines.append((k, float(comps.get(k))))
                    except Exception:
                        pass

            lines.sort(key=lambda x: x[1], reverse=True)
            top_lines = lines[:6]

            if top_lines:
                body = "<br/>".join([f"<b>{k}:</b> {v:.1f}" for k, v in top_lines])
            else:
                body = "—"
            card("Componentes del score (top drivers)", body)

    with right:
        st.markdown("#### Evidencias KX (top-k)")
        kx = item.get("kx_evidence", [])
        if not kx:
            st.info("No hay evidencias KX (o el índice KX no está construido).")
        else:
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

else:
    st.warning("No hay noticias para mostrar. Pulsa 'Run Agent' y revisa el diagnóstico.")
    if diag:
        st.write("Pistas rápidas:")
        st.write(f"- Google News items: {diag.get('google_news_items')}")
        st.write(f"- RSS items: {diag.get('rss_items')}")
        st.write(f"- NewsAPI items: {diag.get('newsapi_items')}")
        st.write(f"- Ingestadas: {diag.get('ingested_total')}")
        st.write(f"- Tras filtro: {diag.get('kept_after_filter')}")
        if diag.get("llm_error"):
            st.error(f"LLM error: {diag.get('llm_error')}")

