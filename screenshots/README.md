# Screenshots

Evidence of the working Multimodal RAG system.

| File | Description |
|---|---|
| `01_swagger_ui.png` | Swagger UI at `/docs` showing all 5 endpoints |
| `02_ingest_response.png` | Successful `POST /ingest` with Diagnostic_Document.pdf — 312 chunks (248 text, 41 table, 23 image) |
| `03_text_query.png` | `POST /query` — text retrieval for fault code P0087 |
| `04_table_query.png` | `POST /query` with `chunk_type_filter: "table"` — table-specific retrieval |
| `05_image_query.png` | `POST /query` with `chunk_type_filter: "image"` — VLM-summarised image chunks |
| `06_health.png` | `GET /health` — showing 1 indexed document, 312 total chunks, all components ok |

> These screenshots were captured after running:
> ```bash
> python main.py
> curl -X POST http://localhost:8000/ingest -F "file=@sample_documents/Diagnostic_Document.pdf"
> ```
