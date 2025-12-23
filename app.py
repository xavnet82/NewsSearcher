from __future__ import annotations

import uuid
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src import db
from src.classifier import classify, hard_filter
from src.kx_pdf import chunk_text, extract_pdf_text, save_uploaded_pdf
from src.llm import (
    generate_insights_heuristic,
    generate_insights_llm,
    llm_available,
)
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

st.set_page_config(page_title="Acn2Agent · News + KX (PDF)", layout="wide")

st.title("Acn2Agent · News + APIs + KX (PDF)")
st.caption("Agentic RAG: ingest → enrich(KX) → filter → rank → explain → structured output")

# ---------------- sidebar config ----------------
st.sidebar.header("Configuración")

default_feeds = [
    "https://www.reuters.com/rssFeed/technologyNews",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://www.theverge.com/rss/index.xml",
]
feeds_text = st.sidebar.text_area(
    "RSS feeds (uno por línea)",
    value="\n".join(default_feeds),
    height=120,
)
rss_feeds = [f.strip() for f in feeds_text.splitlines() if f.strip()]

use_newsapi = st.sidebar.checkbox("Usar NewsAPI (opcional)", value=False)
newsapi_query = st.sidebar.text_input(
    "NewsAPI query",
    value="Accenture OR artificial intelligence OR cloud regulation",
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

keywords = st.sidebar.text_area(
    "Keywords (uno por línea)",
    value="\n".join(DEFAULT_KEYWORDS),
    height=140,
)
kw_list = [k.strip() for k in keywords.splitlines() if k.strip()]

entities = st.sidebar.text_area(
    "Entidades (uno por línea)",
    value="\n".join(DEFAULT_ENTITIES),
    height=120,
)
ent_list = [e.strip() for e in entities.splitlines() if e.strip()]

st.sidebar.divider()
st.sidebar.subheader("Filtros / ranking")
top_k_kx = st.sidebar.slider("Top K evidencias KX por noticia", 1, 10, 5, 1)
top_n = st.sidebar.slider("Top N noticias en ranking", 5, 50, 20, 5)

enrich_with_url = st.sidebar.checkbox(
    "Intentar extraer snippet desde URL si falta resumen",
    value=True,
)

# Filtro mínimo (hard filter)
st.sidebar.subheader("Filtro mínimo")
must_terms_txt = st.sidebar.text_area(
    "Debe contener al menos uno (uno por línea)",
    value="Accenture\ndeloitte\npwc\ney\nkpmg\naws\nazure\nopenai",
    height=120,
)
must_terms = [x.strip() for x in must_terms_txt.splitlines() if x.strip()]

st.sidebar.divider()
st.sidebar.subheader("LLM (opcional)")
use_llm = st.sidebar.checkbox(
    "Generar insights con LLM (requiere OPENAI_API_KEY)",
    value=False,
)
st.sidebar.caption(f"LLM disponible: {'Sí' if llm_available() else 'No'}")

run_agent = st.sidebar.button("▶ Run Agent", type="primary")

# ---------------- KX upload / build index ----------------
st.subheader("1) Cargar KX (PDF) y construir índice")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader(
        "Sube uno o varios PDFs (KX)",
        type=["pdf"],
        accept_multiple_files=True,
    )
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

def run_pipeline() -> List[Dict[str, Any]]:
    # 1) Ingesta
    rss_items = fetch_rss(rss_feeds, max_items=news_limit) if rss_feeds else []
    api_items = fetch_newsapi(newsapi_query, page_size=news_limit) if use_newsapi else []
    ingested = normalize_and_store(rss_items + api_items)

    # 2) Enriquecer + clasificar + score
    results: List[Dict[str, Any]] = []

    for it in ingested:
        keep, _drop_reason = hard_filter(it, must_have_any=must_terms)
        if not keep:
            continue

        query = f"{it.get('title', '')} {it.get('summary', '')}".strip()

        # KX enrichment (RAG)
        kx_hits = search_kx(query, top_k=top_k_kx) if kx_index_exists() else []

        # Clasificación (tendencia/alcance/mercado/servicio)
        classif = classify(it)

        # Score base
        base_score, base_comps, tags = score_news_item(it, kx_hits, weights, kw_list, ent_list)

        # Score dimensiones (scope/trend/market/service)
        dim_scores = score_dimensions_boost(classif, boosts={})

        # Mezcla final (75% base, 25% dimensiones)
        final_score = 0.75 * base_score + 0.25 * (
            0.25 * dim_scores["scope"]
            + 0.25 * dim_scores["trend"]
            + 0.25 * dim_scores["market"]
            + 0.25 * dim_scores["service"]
        )

        # Explicación estructurada
        if use_llm and llm_available():
            insights = generate_insights_llm(it, kx_hits)
        else:
            insights = generate_insights_heuristic(it, kx_hits)

        descripcion = insights.get("descripcion", "")
        por_que = insights.get("por_que_importa", [])
        implicaciones = insights.get("implicaciones_para_accenture", [])

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

    # 3) Ranking
    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return results[:top_n]

if run_agent:
    if not rss_feeds and not use_newsapi:
        st.error("Configura al menos RSS o activa NewsAPI.")
    else:
        with st.spinner("Ejecutando agente..."):
            results = run_pipeline()
            run_id = str(uuid.uuid4())

            params = {
                "rss_feeds": rss_feeds,
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

# ---------------- Results UI ----------------
results = st.session_state.get("last_results", [])
run_id = st.session_state.get("last_run_id", None)

if results:
    st.success(f"Run completado. run_id={run_id}")

    df = pd.DataFrame(
        [
            {
                "score": r["score"],
                "title": r["title"],
                "source": r["source"],
                "published_at": r["published_at"],
                "scope": r.get("classification", {}).get("scope", ""),
                "trends": ", ".join(r.get("classification", {}).get("trends", [])),
                "markets": ", ".join(r.get("classification", {}).get("markets", [])),
                "services": ", ".join(r.get("classification", {}).get("services", [])),
                "url": r["url"],
                "provider": r.get("provider", ""),
            }
            for r in results
        ]
    )

    st.subheader("Ranking")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Detalle")
    titles = [f"{i+1}. [{r['score']:.1f}] {r['title'][:110]}" for i, r in enumerate(results)]
    idx = st.selectbox(
        "Selecciona una noticia",
        options=list(range(len(results))),
        format_func=lambda i: titles[i],
    )

    item = results[idx]
    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"### {item['title']}")
        st.write(
            f"**Fuente:** {item.get('source','')}  ·  **Fecha:** {item.get('published_at','')}  ·  **Proveedor:** {item.get('provider','')}"
        )
        if item.get("url"):
            st.link_button("Abrir noticia", item["url"])

        st.markdown("#### Output (resumen estructurado)")
        out_struct = {
            "titulo": item["title"],
            **item["output"],
            "score_impacto": item["score"],
        }
        st.code(safe_json(out_struct), language="json")

        st.markdown("#### Clasificación (tendencia/alcance/mercado/servicio)")
        st.json(item.get("classification", {}))

        st.markdown("#### Componentes del score")
        st.json(item.get("components", {}))

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
    st.info("Carga KX (opcional) y ejecuta el agente desde la barra lateral.")
