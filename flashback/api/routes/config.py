"""Config endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/config")
async def get_config(request: Request) -> Dict[str, Any]:
    """Get current configuration (sensitive values masked)."""
    config = request.app.state.config

    # Return a safe subset of config
    cfg = config.to_dict()

    # Mask any sensitive paths
    return {
        "screenshot": cfg.get("screenshot"),
        "workers": cfg.get("workers"),
        "search": cfg.get("search"),
        "webui": cfg.get("webui"),
        "features": cfg.get("features"),
    }


@router.post("/config/reload")
async def reload_config(request: Request) -> Dict[str, str]:
    """Reload configuration from file."""
    # Note: This won't affect already-running workers
    # A full restart is needed for some changes
    return {"status": "reload not implemented - restart required"}
