"""Health and status endpoints."""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Request

from flashback.core.daemon import DaemonManager

router = APIRouter()


@router.get("/health")
async def health() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "healthy"}


@router.get("/status")
async def status(request: Request) -> Dict[str, Any]:
    """Full system status."""
    config = request.app.state.config
    db = request.app.state.db

    # Check daemon status
    backend_daemon = DaemonManager("backend")
    webui_daemon = DaemonManager("webui")

    # Get database stats
    db_stats = db.get_stats()

    return {
        "backend": {
            "running": backend_daemon.is_running(),
            "pid": backend_daemon.get_pid(),
        },
        "webui": {
            "running": webui_daemon.is_running(),
            "pid": webui_daemon.get_pid(),
        },
        "storage": {
            "screenshots_dir": str(config.screenshot_dir),
            "database_path": str(config.db_path),
            "screenshot_count": db_stats["total"],
            "oldest_screenshot": db_stats.get("oldest_timestamp"),
            "newest_screenshot": db_stats.get("newest_timestamp"),
        },
        "config": {
            "ocr_enabled": config.is_worker_enabled("ocr"),
            "embedding_enabled": config.is_worker_enabled("embedding"),
            "search_methods": [
                m for m in ["bm25", "embedding"] if config.is_search_enabled(m)
            ],
        },
    }
