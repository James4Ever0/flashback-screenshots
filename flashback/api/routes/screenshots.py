"""Screenshot endpoints."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from flashback.core.database import Database

router = APIRouter()


def _record_to_dict(record, include_full_text: bool = False) -> Dict[str, Any]:
    """Convert ScreenshotRecord to dict."""
    preview = ""
    if record.ocr_text:
        preview = record.ocr_text[:200] + "..." if len(record.ocr_text) > 200 else record.ocr_text

    result = {
        "id": record.id,
        "timestamp": record.timestamp,
        "timestamp_formatted": record.timestamp_formatted,
        "screenshot_path": record.screenshot_path,
        "screenshot_url": f"/screenshots/{Path(record.screenshot_path).name}",
        "window_title": record.window_title,
        "ocr_text_preview": preview,
        "has_embedding": record.embedding_path is not None,
    }

    if include_full_text and record.ocr_text:
        result["ocr_text_full"] = record.ocr_text

    return result


@router.get("/screenshots")
async def list_screenshots(
    request: Request,
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    window_title: Optional[str] = None,
    has_ocr: Optional[bool] = None,
    has_embedding: Optional[bool] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """List screenshots with filtering."""
    db: Database = request.app.state.db

    # Parse time range
    start_ts = None
    end_ts = None

    if from_date:
        try:
            start_ts = datetime.fromisoformat(from_date).timestamp()
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid from date: {from_date}")

    if to_date:
        try:
            end_ts = datetime.fromisoformat(to_date).timestamp()
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid to date: {to_date}")

    # Get records
    if start_ts or end_ts:
        start_ts = start_ts or 0
        end_ts = end_ts or datetime.now().timestamp()
        records = db.search_by_time_range(start_ts, end_ts, limit=limit + offset)
    else:
        # Get all screenshots (limited)
        records = db.get_unprocessed_ocr(limit=limit + offset)

    # Apply filters
    filtered = []
    for record in records:
        if window_title and (not record.window_title or window_title.lower() not in record.window_title.lower()):
            continue
        if has_ocr is not None:
            has_ocr_data = record.ocr_path is not None
            if has_ocr != has_ocr_data:
                continue
        if has_embedding is not None:
            has_emb_data = record.embedding_path is not None
            if has_embedding != has_emb_data:
                continue
        filtered.append(record)

    total = len(filtered)
    results = filtered[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": [_record_to_dict(r) for r in results],
    }


@router.get("/screenshots/{timestamp}")
async def get_screenshot(
    request: Request,
    timestamp: float,
) -> Dict[str, Any]:
    """Get a specific screenshot by timestamp."""
    db: Database = request.app.state.db
    record = db.get_by_timestamp(timestamp)

    if not record:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    return _record_to_dict(record, include_full_text=True)


@router.get("/screenshots/{timestamp}/neighbors")
async def get_neighbors(
    request: Request,
    timestamp: float,
    before: int = Query(5, ge=0, le=50),
    after: int = Query(5, ge=0, le=50),
) -> Dict[str, Any]:
    """Get screenshots near a timestamp (timeline view)."""
    db: Database = request.app.state.db

    # Get center record
    center = db.get_by_timestamp(timestamp)
    if not center:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    # Get neighbors within window
    window_seconds = max(before, after) * 60 * 5  # Rough estimate: 5 min per screenshot
    all_neighbors = db.get_neighbors(timestamp, window_seconds=window_seconds)

    # Sort and separate
    all_neighbors.sort(key=lambda r: r.timestamp)

    center_idx = None
    for i, r in enumerate(all_neighbors):
        if abs(r.timestamp - timestamp) < 1:
            center_idx = i
            break

    if center_idx is None:
        center_idx = len(all_neighbors) // 2

    # Get before/after
    start_idx = max(0, center_idx - before)
    end_idx = min(len(all_neighbors), center_idx + after + 1)
    selected = all_neighbors[start_idx:end_idx]

    return {
        "center_timestamp": timestamp,
        "screenshots": [
            {
                **_record_to_dict(r),
                "is_center": abs(r.timestamp - timestamp) < 1,
                "relative_minutes": round((r.timestamp - timestamp) / 60, 1),
            }
            for r in selected
        ],
    }


@router.get("/screenshots/{timestamp}/ocr")
async def get_ocr(
    request: Request,
    timestamp: float,
    format: str = Query("json", enum=["json", "text"]),
) -> Any:
    """Get OCR text for a screenshot."""
    db: Database = request.app.state.db
    record = db.get_by_timestamp(timestamp)

    if not record:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    if not record.ocr_text:
        if format == "text":
            return ""
        return {"timestamp": timestamp, "text": "", "word_count": 0}

    if format == "text":
        return record.ocr_text

    return {
        "timestamp": timestamp,
        "text": record.ocr_text,
        "word_count": len(record.ocr_text.split()),
    }


@router.delete("/screenshots/{timestamp}")
async def delete_screenshot(
    request: Request,
    timestamp: float,
) -> Dict[str, str]:
    """Delete a screenshot."""
    import os

    db: Database = request.app.state.db
    record = db.get_by_timestamp(timestamp)

    if not record:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    # Delete files
    for key in ["screenshot_path", "ocr_path", "embedding_path"]:
        path_str = getattr(record, key, None)
        if path_str:
            try:
                os.remove(path_str)
            except FileNotFoundError:
                pass

    # Delete database record
    db.delete_record(timestamp)

    return {"status": "deleted", "timestamp": str(timestamp)}
