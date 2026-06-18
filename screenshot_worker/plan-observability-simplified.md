# Plan: Observability Platform — Simplified

## Goal
A web application running on the server that aggregates all screenshot workers into a browsable, searchable dashboard for human operators and AI agents.

## Phase 1: Web Dashboard
- Single-page app (React or Vue) served by the server.
- Grid view: live thumbnails of all connected clients.
- Timeline view: scrollable history per client (date picker).
- Search: filter by client name, date range.

## Phase 2: AI Integration Hooks
- Expose `GET /api/clients/{name}/frames?since=` for agent consumption.
- WebSocket endpoint `/ws/agents` for real-time push to subscribed AI agents.
- Simple API key header auth for agent endpoints.

## Phase 3: Shared Knowledge
- Each client’s OCR results (processed server-side) indexed into a shared vector DB.
- Simple RAG endpoint: `POST /api/query` → returns relevant past screenshots + snippets.

## Deferred to Detailed Plan
- Multi-tenant auth, RBAC, advanced analytics, real-time collaboration.
