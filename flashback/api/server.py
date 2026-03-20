"""FastAPI server for flashback web UI."""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from flashback.core.config import Config
from flashback.core.daemon import DaemonManager
from flashback.core.database import Database
from flashback.core.logger import get_logger

# Import routes
from flashback.api.routes import health, search, screenshots, config as config_route

logger = get_logger("api.server")


async def log_requests(request: Request, call_next):
    """Middleware to log all API requests."""
    start_time = time.time()

    # Log request
    logger.debug(
        f"REQUEST {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )

    # Process request
    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.debug(
        f"RESPONSE {request.method} {request.url.path} "
        f"status={response.status_code} time={process_time:.3f}s"
    )

    return response


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = config or Config()

    app = FastAPI(
        title="Flashback",
        description="Screenshot history search with OCR and semantic search",
        version="0.1.0",
    )

    # Add logging middleware
    app.middleware("http")(log_requests)

    # Store config and db in app state
    app.state.config = config
    app.state.db = Database(config.db_path)
    logger.info("FastAPI app created")

    # Include routers
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")
    app.include_router(screenshots.router, prefix="/api/v1")
    app.include_router(config_route.router, prefix="/api/v1")

    # Setup templates
    template_dir = Path(__file__).parent.parent / "web" / "templates"
    if template_dir.exists():
        app.state.templates = Jinja2Templates(directory=str(template_dir))

    # Setup static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Mount screenshot directory for serving
    screenshot_dir = config.screenshot_dir
    if screenshot_dir.exists():
        app.mount(
            "/screenshots",
            StaticFiles(directory=str(screenshot_dir)),
            name="screenshots",
        )

    @app.get("/")
    async def index(request: Request):
        """Serve the main web UI page."""
        search_methods = []
        if config.is_search_enabled("bm25"):
            search_methods.append({"id": "bm25", "name": "BM25 Text", "default": True})
        if config.is_search_enabled("embedding"):
            default = not config.is_search_enabled("bm25")
            search_methods.append(
                {"id": "embedding", "name": "Embedding (Semantic)", "default": default}
            )

        if hasattr(app.state, "templates"):
            return app.state.templates.TemplateResponse(
                "index.html",
                {"request": request, "search_methods": search_methods},
            )
        return {"message": "Flashback API", "search_methods": search_methods}

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(exc)},
        )

    return app


def main():
    """Main entry point for the API server."""
    import uvicorn

    config = Config()

    # Check if backend is running
    backend_daemon = DaemonManager("backend")
    if not backend_daemon.is_running():
        logger.warning("Backend daemon is not running!")
        logger.info("Start it with: flashback serve --daemon")

    host = config.webui_host
    port = config.webui_port

    logger.info(f"Starting web UI on http://{host}:{port}")

    app = create_app(config)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
