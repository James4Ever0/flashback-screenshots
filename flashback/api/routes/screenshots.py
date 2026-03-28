"""Screenshot endpoints."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

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


@router.get("/screenshots/timeline")
async def list_screenshots_timeline(
    request: Request,
    before_time: Optional[float] = Query(None, description="Get screenshots before this timestamp"),
    around_time: Optional[float] = Query(None, description="Get screenshots around this timestamp"),
    window_title: Optional[str] = Query(None, description="Filter by window title substring"),
    search_keyword: Optional[str] = Query(None, description="Filter by OCR text substring"),
    limit: int = Query(50, ge=1, le=100),
) -> Dict[str, Any]:
    """List screenshots ordered by time (most recent first) for timeline browsing.

    Args:
        before_time: If provided, get screenshots before this timestamp (for pagination)
        around_time: If provided, get screenshots centered around this timestamp
        window_title: Optional filter by window title substring
        search_keyword: Optional filter by OCR text substring
        limit: Number of screenshots to return
    """
    db: Database = request.app.state.db

    # Use filtered queries if filters are provided
    if window_title or search_keyword:
        if around_time:
            records = db.get_screenshots_around_time_with_filters(
                around_time,
                window_title=window_title,
                ocr_text=search_keyword,
                count=limit,
            )
        elif before_time:
            records = db.get_screenshots_ordered_with_filters(
                before_time=before_time,
                window_title=window_title,
                ocr_text=search_keyword,
                limit=limit,
            )
        else:
            records = db.get_screenshots_ordered_with_filters(
                window_title=window_title,
                ocr_text=search_keyword,
                limit=limit,
            )
    else:
        if around_time:
            records = db.get_screenshots_around_time(around_time, count=limit)
        elif before_time:
            records = db.get_screenshots_ordered(before_time=before_time, limit=limit)
        else:
            records = db.get_screenshots_ordered(limit=limit)

    # Get total count for reference
    total = db.count_screenshots_after()

    # Determine the time range displayed
    if records:
        time_from = min(r.timestamp for r in records)
        time_to = max(r.timestamp for r in records)
    else:
        time_from = time_to = around_time or before_time

    # Get oldest timestamp for reference
    oldest_ts = db.get_oldest_timestamp()

    return {
        "total": total,
        "limit": limit,
        "time_from": time_from,
        "time_to": time_to,
        "oldest_timestamp": oldest_ts,
        "results": [_record_to_dict(r) for r in records],
    }


@router.get("/screenshots/timeline/jump")
async def jump_to_time(
    request: Request,
    time: float = Query(..., description="Target timestamp to jump to"),
    count: int = Query(50, ge=1, le=100),
) -> Dict[str, Any]:
    """Jump to a specific time point and get nearby screenshots."""
    db: Database = request.app.state.db

    records = db.get_screenshots_around_time(time, count=count)

    if not records:
        raise HTTPException(status_code=404, detail="No screenshots found near the specified time")

    time_from = min(r.timestamp for r in records)
    time_to = max(r.timestamp for r in records)

    return {
        "jump_time": time,
        "count": len(records),
        "time_from": time_from,
        "time_to": time_to,
        "results": [_record_to_dict(r) for r in records],
    }

@router.get("/screenshots/now")
async def get_latest_screenshot(request: Request) -> FileResponse:
    """Get the latest screenshot as a file response."""
    import time

    db: Database = request.app.state.db
    config: Config = request.app.state.config
    latest = db.get_latest()
    if not latest:
        raise HTTPException(status_code=404, detail="No screenshots found")

    # Check age limit
    age_limit_seconds = config.get("webui.latest_screenshot_age_limit_seconds", 120)
    age_seconds = time.time() - latest.timestamp
    if age_seconds > age_limit_seconds:
        raise HTTPException(
            status_code=404,
            detail=f"Latest screenshot is {age_seconds:.0f}s old (limit: {age_limit_seconds}s)"
        )

    return FileResponse(latest.screenshot_path)

@router.get("/screenshots/by-id/{screenshot_id}")
async def get_screenshot_by_id(
    request: Request,
    screenshot_id: int,
) -> Dict[str, Any]:
    """Get a specific screenshot by ID."""
    db: Database = request.app.state.db
    record = db.get_by_id(screenshot_id)

    if not record:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    return _record_to_dict(record, include_full_text=True)


@router.get("/screenshots/by-id/{screenshot_id}/neighbors")
async def get_neighbors_by_id(
    request: Request,
    screenshot_id: int,
    before: int = Query(5, ge=0, le=50),
    after: int = Query(5, ge=0, le=50),
    window_title: Optional[str] = Query(None, description="Filter by window title substring"),
    search_keyword: Optional[str] = Query(None, description="Filter by OCR text substring"),
) -> Dict[str, Any]:
    """Get screenshots near a screenshot ID (timeline view).

    Args:
        screenshot_id: The ID of the center screenshot
        before: Number of screenshots to include before the center
        after: Number of screenshots to include after the center
        window_title: Optional filter by window title substring
        search_keyword: Optional filter by OCR text substring
    """
    db: Database = request.app.state.db

    # Get center record
    center = db.get_by_id(screenshot_id)
    if not center:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    # Get neighbors within window
    window_seconds = max(before, after) * 60 * 5  # Rough estimate: 5 min per screenshot

    # Use filtered query if filters are provided
    if window_title or search_keyword:
        all_neighbors = db.get_neighbors_with_filters(
            center.timestamp,
            window_title=window_title,
            ocr_text=search_keyword,
            window_seconds=window_seconds,
        )
    else:
        all_neighbors = db.get_neighbors(center.timestamp, window_seconds=window_seconds)

    # Sort and separate
    all_neighbors.sort(key=lambda r: r.timestamp)

    center_idx = None
    for i, r in enumerate(all_neighbors):
        if r.id == screenshot_id:
            center_idx = i
            break

    if center_idx is None:
        center_idx = len(all_neighbors) // 2

    # Get before/after
    start_idx = max(0, center_idx - before)
    end_idx = min(len(all_neighbors), center_idx + after + 1)
    selected = all_neighbors[start_idx:end_idx]

    return {
        "center_id": screenshot_id,
        "filters_applied": {
            "window_title": window_title,
            "search_keyword": search_keyword,
        },
        "screenshots": [
            {
                **_record_to_dict(r),
                "is_center": r.id == screenshot_id,
                "relative_minutes": round((r.timestamp - center.timestamp) / 60, 1),
            }
            for r in selected
        ],
    }


# Legacy timestamp-based endpoints (deprecated but kept for compatibility)
@router.get("/screenshots/{timestamp}")
async def get_screenshot(
    request: Request,
    timestamp: float,
) -> Dict[str, Any]:
    """Get a specific screenshot by timestamp (deprecated, use /by-id/{id})."""
    db: Database = request.app.state.db
    record = db.get_by_timestamp(timestamp)

    if not record:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    return _record_to_dict(record, include_full_text=True)


@router.get("/screenshots/{timestamp}/image")
async def preview_screenshot(request:Request, timestamp: float) -> FileResponse:
    """Get screenshot preview image"""
    db: Database = request.app.state.db
    record = db.get_by_timestamp(timestamp)

    if not record:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    if not record.screenshot_path:
        raise HTTPException(status_code=404, detail="Screenshot image not found")

    return FileResponse(record.screenshot_path)


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
