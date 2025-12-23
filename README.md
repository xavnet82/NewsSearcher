# Acn2Agent (Streamlit) — News + APIs + KX (PDF) Agentic RAG

Streamlit app that:
- Ingests news (RSS, optional NewsAPI)
- Uploads KX PDFs and builds a semantic index (embeddings + FAISS)
- Runs an "agent" pipeline: ingest → dedupe → enrich (RAG) → score → explain → structured output
- Shows ranked results + KX evidence snippets + exportable JSON

## Quickstart (local)

### 1) Create venv
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
