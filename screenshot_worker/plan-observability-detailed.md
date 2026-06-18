# Plan: Observability Platform — Detailed

## 1. Architecture Overview
```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Browser    │     │   AI Agent   │     │   Another Agent  │
└──────┬───────┘     └──────┬───────┘     └────────┬─────────┘
       │                    │                      │
       └────────────────────┼──────────────────────┘
                            │  HTTPS / WSS
                    ┌───────▼────────┐
                    │  API Gateway   │  (axum, rate limiting, API keys)
                    └───────┬────────┘
                            │
       ┌────────────────────┼──────────────────────┐
       │                    │                      │
┌──────▼──────┐    ┌────────▼──────┐    ┌─────────▼────────┐
│  Web UI     │    │  Frame Store  │    │  Vector Index    │
│  (React)    │    │  (SQLite/FS)  │    │  (pgvector /     │
│             │    │               │    │   chroma / qdrant)│
└─────────────┘    └───────────────┘    └──────────────────┘
```

## 2. Backend Server Extensions
- Framework: `axum` (same Rust ecosystem as worker).
- Middleware:
  - `tower_http::trace` for request logging.
  - `tower::limit::RateLimitLayer` for public endpoints.
  - Custom `RequireApiKey` extractor for agent endpoints.

## 3. REST API Specification
```yaml
GET  /api/v1/clients                    → List all clients with last seen
GET  /api/v1/clients/{id}/frames        → Paginated frame list
GET  /api/v1/clients/{id}/frames/latest → Redirect to latest image
GET  /api/v1/clients/{id}/frames/{fid}  → Serve image file
WS   /ws/agents                         → Real-time frame stream (JSON metadata + image URL)
POST /api/v1/query                      → RAG query over OCR text
GET  /api/v1/health                     → Server health check
```

## 4. Web Dashboard (React + TypeScript)
- **Live Grid**: WebSocket `/ws/dashboard` receives `FrameEvent` broadcasts; renders masonry grid.
- **Client Detail**: Route `/client/:id` shows calendar + hour selector + thumbnail strip.
- **Search Page**: Full-text search over OCR index; results link to exact frame.
- **Settings**: API key rotation, retention policy override per client.

## 5. OCR Pipeline (Server-Side)
- Worker thread pool or async task queue.
- Run `tesseract` or ONNX-based model (e.g., `easyocr` via Python FFI, or Rust `ocrs`) on each incoming frame.
- Store text in SQLite: `ocr_text(frame_id, text, lang)`.
- Emit `OcrCompleted` event; update vector index.

## 6. Vector Index & RAG
- Embed OCR text using lightweight local model (`sentence-transformers/all-MiniLM-L6-v2` via `ort` in Rust, or call external embedding service).
- Store in `pgvector` or `chroma`.
- Query endpoint:
  - Parse natural language query.
  - Embed → vector search (top-k=10).
  - Return: `{ frames: [{ client, timestamp, image_url, text_snippet, score }] }`.

## 7. AI Agent Connectivity
- **Local agents**: Connect to `/ws/agents` with API key; subscribe to specific clients or all.
- **Cloud agents**: Same endpoint; assume network reachability (or use reverse tunnel).
- **Agent protocol**: JSON messages:
  ```json
  { "type": "subscribe", "clients": ["dev-laptop-01"] }
  { "type": "frame", "client": "dev-laptop-01", "timestamp": "...", "ocr": "...", "image_url": "..." }
  ```

## 8. Deployment
- Docker Compose: server + PostgreSQL (for vector) + optional OCR worker container.
- Reverse proxy (nginx/traefik) for TLS termination.
- Environment-based config: `.env` for secrets, `config.yaml` for behavior.

## 9. Security Considerations
- API keys for agent access (no user auth in v1).
- CORS restricted to known dashboard origin.
- Image files served with `Content-Disposition: attachment` to prevent hotlinking.
- Retention enforcement: cron job or async task deletes old frames and index entries.

## 10. Milestones
| # | Deliverable | ETA |
|---|-------------|-----|
| 1 | REST API + frame serving | Day 6 |
| 2 | React dashboard (grid + timeline) | Day 8 |
| 3 | OCR pipeline integration | Day 10 |
| 4 | Vector index + RAG query | Day 12 |
| 5 | Agent WebSocket + documentation | Day 14 |
