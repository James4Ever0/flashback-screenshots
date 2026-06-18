#!/usr/bin/env python3
"""
Screenshot Worker — Simplified WebSocket Server (Bootstrap)

Receives screenshot frames from go-bootstrap and android-bootstrap clients,
displays metadata without storing images.

Recommended Python version: 3.11+
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import datetime
import json
import logging
import signal
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import websockets

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("ws-server")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ClientInfo:
    """Tracks a connected WebSocket client."""

    ws: Any
    client_id: str
    name: str = "unknown"
    version: str = "unknown"
    connected_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    frames_received: int = 0
    bytes_received: int = 0
    last_ping_at: float = field(default_factory=time.time)
    last_frame_at: float | None = None


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------

# Heuristic: if timestamp > 10 billion, it's milliseconds (year ~2286 in seconds)
_MILLIS_THRESHOLD = 10_000_000_000


def normalize_timestamp(ts: int) -> datetime.datetime:
    """Convert a client timestamp (seconds or millis) to UTC datetime."""
    if ts > _MILLIS_THRESHOLD:
        return datetime.datetime.fromtimestamp(ts / 1000.0, tz=datetime.timezone.utc)
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)


def describe_timestamp(ts: int) -> str:
    """Return a human-readable description of the timestamp unit detected."""
    unit = "ms" if ts > _MILLIS_THRESHOLD else "s"
    dt = normalize_timestamp(ts)
    return f"{ts} {unit} → {dt.isoformat()}"


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class ServerState:
    """Shared mutable state protected by asyncio locks."""

    def __init__(self) -> None:
        self.clients: dict[str, ClientInfo] = {}
        self.lock = asyncio.Lock()
        self.shutdown_event = asyncio.Event()

    async def register(self, client: ClientInfo) -> None:
        async with self.lock:
            self.clients[client.client_id] = client
        logger.info(
            "[%s] Connected  | name=%s | version=%s | remote=%s | total_clients=%d",
            client.client_id[:8],
            client.name,
            client.version,
            client.ws.remote_address,
            len(self.clients),
        )

    async def unregister(self, client_id: str) -> None:
        async with self.lock:
            info = self.clients.pop(client_id, None)
        if info:
            duration = datetime.datetime.utcnow() - info.connected_at
            logger.info(
                "[%s] Disconnected | name=%s | frames=%d | bytes=%s | duration=%s",
                client_id[:8],
                info.name,
                info.frames_received,
                _human_size(info.bytes_received),
                duration,
            )

    async def update_client_name(self, client_id: str, name: str, version: str) -> None:
        async with self.lock:
            if client_id in self.clients:
                self.clients[client_id].name = name
                self.clients[client_id].version = version

    async def record_frame(self, client_id: str, data_len: int) -> None:
        async with self.lock:
            if client_id in self.clients:
                self.clients[client_id].frames_received += 1
                self.clients[client_id].bytes_received += data_len
                self.clients[client_id].last_frame_at = time.time()

    async def list_clients(self) -> list[ClientInfo]:
        async with self.lock:
            return list(self.clients.values())

    def request_shutdown(self) -> None:
        self.shutdown_event.set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def _make_json(data: dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Message handlers
# ---------------------------------------------------------------------------

async def _handle_hello(
    state: ServerState,
    client: ClientInfo,
    payload: dict[str, Any],
) -> None:
    """Process a client hello message."""
    name = payload.get("name", "unknown")
    version = payload.get("version", "unknown")
    await state.update_client_name(client.client_id, name, version)
    logger.info(
        "[%s] Hello      | name=%s | version=%s",
        client.client_id[:8],
        name,
        version,
    )


async def _handle_frame(
    state: ServerState,
    client: ClientInfo,
    payload: dict[str, Any],
) -> None:
    """Process a screenshot frame — decode, log metadata, discard data."""
    filename = payload.get("filename", "unknown")
    timestamp_raw = payload.get("timestamp", 0)
    data_b64 = payload.get("data", "")

    # Decode base64 to compute actual size, then discard
    try:
        raw_data = base64.b64decode(data_b64)
        data_size = len(raw_data)
    except Exception as exc:
        logger.warning(
            "[%s] Frame      | FAILED to decode base64 | filename=%s | error=%s",
            client.client_id[:8],
            filename,
            exc,
        )
        return

    await state.record_frame(client.client_id, data_size)

    ts_desc = describe_timestamp(timestamp_raw)
    logger.info(
        "[%s] Frame      | name=%s | filename=%s | size=%s (%d bytes) | timestamp=%s",
        client.client_id[:8],
        client.name,
        filename,
        _human_size(data_size),
        data_size,
        ts_desc,
    )

    # Send ACK back to client
    ack = _make_json({"type": "ack", "filename": filename})
    try:
        await client.ws.send(ack)
    except Exception as exc:
        logger.warning(
            "[%s] ACK failed | filename=%s | error=%s",
            client.client_id[:8],
            filename,
            exc,
        )


async def _handle_pong(
    state: ServerState,
    client: ClientInfo,
    _payload: dict[str, Any],
) -> None:
    """Client replied to our ping."""
    client.last_ping_at = time.time()
    logger.debug("[%s] Pong       | name=%s", client.client_id[:8], client.name)


# ---------------------------------------------------------------------------
# Per-client connection handler
# ---------------------------------------------------------------------------

async def _ping_loop(state: ServerState, client: ClientInfo, interval: float) -> None:
    """Send periodic ping messages to keep the connection alive."""
    try:
        while not state.shutdown_event.is_set():
            await asyncio.sleep(interval)
            ping_msg = _make_json({"type": "ping"})
            try:
                await client.ws.send(ping_msg)
                logger.debug("[%s] Ping sent  | name=%s", client.client_id[:8], client.name)
            except websockets.exceptions.ConnectionClosed:
                return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("[%s] Ping loop ended | error=%s", client.client_id[:8], exc)


def _get_ws_path(ws: Any) -> str | None:
    """Get the request path from a websocket connection (13.x or 14.x compatible)."""
    # websockets 13.x and earlier
    path = getattr(ws, "path", None)
    if path is not None:
        return path
    # websockets 14.x — path is inside the request object
    request = getattr(ws, "request", None)
    if request is not None:
        return getattr(request, "path", None)
    return None


async def handle_client(
    ws: Any,
    state: ServerState,
    ping_interval: float,
) -> None:
    """Main handler for a single WebSocket connection."""
    # Extract name from query parameter if provided (both clients send ?name=...)
    query_name = "unknown"
    path = _get_ws_path(ws)
    if path:
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(path)
        qs = parse_qs(parsed.query)
        if "name" in qs:
            query_name = qs["name"][0]

    client_id = str(uuid.uuid4())
    client = ClientInfo(ws=ws, client_id=client_id, name=query_name)
    await state.register(client)

    # Start ping loop in background
    ping_task = asyncio.create_task(
        _ping_loop(state, client, ping_interval),
        name=f"ping-{client_id[:8]}",
    )

    try:
        async for message in ws:
            if isinstance(message, bytes):
                logger.warning(
                    "[%s] Binary msg ignored | %d bytes",
                    client_id[:8],
                    len(message),
                )
                continue

            # Parse JSON text message
            try:
                payload = json.loads(message)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "[%s] Invalid JSON | error=%s | msg=%.80s...",
                    client_id[:8],
                    exc,
                    message,
                )
                continue

            msg_type = payload.get("type", "unknown")
            handler: Callable[..., Any] | None = {
                "hello": _handle_hello,
                "frame": _handle_frame,
                "pong": _handle_pong,
            }.get(msg_type)

            if handler is None:
                logger.debug(
                    "[%s] Unknown msg type=%s | %.120s...",
                    client_id[:8],
                    msg_type,
                    str(payload),
                )
                continue

            await handler(state, client, payload)

    except websockets.exceptions.ConnectionClosedOK:
        logger.debug("[%s] Connection closed normally", client_id[:8])
    except websockets.exceptions.ConnectionClosedError as exc:
        logger.debug("[%s] Connection closed with error: %s", client_id[:8], exc)
    finally:
        ping_task.cancel()
        try:
            await ping_task
        except asyncio.CancelledError:
            pass
        await state.unregister(client_id)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

async def _status_reporter(state: ServerState, interval: int) -> None:
    """Periodic summary of connected clients and total activity."""
    while not state.shutdown_event.is_set():
        await asyncio.sleep(interval)
        clients = await state.list_clients()
        if not clients:
            continue
        total_frames = sum(c.frames_received for c in clients)
        total_bytes = sum(c.bytes_received for c in clients)
        logger.info(
            "Status       | clients=%d | total_frames=%d | total_bytes=%s",
            len(clients),
            total_frames,
            _human_size(total_bytes),
        )
        for c in clients:
            logger.info(
                "  - %-10s | frames=%d | bytes=%s | since=%s",
                c.name,
                c.frames_received,
                _human_size(c.bytes_received),
                c.connected_at.strftime("%H:%M:%S"),
            )


async def run_server(host: str, port: int, ping_interval: float, status_interval: int) -> None:
    """Start the WebSocket server and run until shutdown signal."""
    state = ServerState()

    def _stop(signum: int, _frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, shutting down...", sig_name)
        state.request_shutdown()

    # Register signal handlers for graceful shutdown
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, state.request_shutdown)
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, state.request_shutdown)
    except (ValueError, OSError, NotImplementedError):
        # Windows or non-main thread fallback
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

    status_task = asyncio.create_task(
        _status_reporter(state, status_interval),
        name="status-reporter",
    )

    # websockets 14.x passes only the connection object to the handler
    async def handler(ws: Any) -> None:
        await handle_client(ws, state, ping_interval)

    logger.info("Starting server on ws://%s:%d/ws", host, port)
    logger.info("  ping_interval=%.1fs  status_interval=%ds", ping_interval, status_interval)
    logger.info("Press Ctrl+C to stop")

    async with websockets.serve(
        handler,
        host,
        port,
        ping_interval=None,  # we handle application-level pings ourselves
        ping_timeout=None,
    ):
        await state.shutdown_event.wait()

    status_task.cancel()
    try:
        await status_task
    except asyncio.CancelledError:
        pass

    # Close remaining connections
    clients = await state.list_clients()
    for client in clients:
        await client.ws.close(code=1001, reason="Server shutting down")
    logger.info("Server stopped. Goodbye!")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Screenshot Worker — Simplified WebSocket Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Default: localhost:8080
  %(prog)s --host 0.0.0.0 --port 9000
  %(prog)s --host 0.0.0.0 --port 9000 --status-interval 60
        """,
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost). Use 0.0.0.0 for all interfaces.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080).",
    )
    parser.add_argument(
        "--ping-interval",
        type=float,
        default=30.0,
        help="Seconds between server-initiated ping messages (default: 30.0).",
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=60,
        help="Seconds between status summary logs (default: 60).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        asyncio.run(
            run_server(
                host=args.host,
                port=args.port,
                ping_interval=args.ping_interval,
                status_interval=args.status_interval,
            )
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
