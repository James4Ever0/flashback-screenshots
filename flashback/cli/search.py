"""Search command for flashback."""

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from flashback.core.config import Config
from flashback.core.database import Database
from flashback.search.bm25 import BM25Search
from flashback.search.embedding import (
    HybridEmbeddingSearch,
    ImageEmbeddingSearch,
    TextEmbeddingSearch,
)
from flashback.search.fusion import reciprocal_rank_fusion

console = Console()


def parse_time(time_str: str) -> Optional[float]:
    """Parse time string (relative or absolute)."""
    if not time_str:
        return None

    # Check for relative time (e.g., "1d", "2h", "30m")
    if time_str.endswith("d"):
        days = int(time_str[:-1])
        return (datetime.now() - timedelta(days=days)).timestamp()
    if time_str.endswith("h"):
        hours = int(time_str[:-1])
        return (datetime.now() - timedelta(hours=hours)).timestamp()
    if time_str.endswith("m"):
        minutes = int(time_str[:-1])
        return (datetime.now() - timedelta(minutes=minutes)).timestamp()

    # Try absolute time
    try:
        return datetime.fromisoformat(time_str).timestamp()
    except ValueError:
        pass

    # Try common formats
    for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(time_str, fmt).timestamp()
        except ValueError:
            continue

    raise ValueError(f"Cannot parse time: {time_str}")


@click.command()
@click.argument("query", required=False)
@click.option("--image", "image_path", type=click.Path(exists=True), help="Search by image file")
@click.option(
    "--search-mode",
    "-m",
    default=None,
    type=click.Choice([
        "bm25_only",
        "text_embedding_only",
        "text_hybrid",
        "image_embedding_only",
        "text_to_image",
        "text_and_image",
        "comprehensive",
    ]),
    help="Search mode to use",
)
@click.option("--limit", "-n", default=20, help="Max results")
@click.option("--from", "from_time", help="Start time (ISO or relative: 1d, 2h, 30m)")
@click.option("--to", "to_time", help="End time (ISO or relative)")
@click.option(
    "--format", "-F",
    "output_format",
    default="table",
    type=click.Choice(["table", "json", "csv", "simple"]),
)
@click.option("--preview", "-p", is_flag=True, help="Show OCR text excerpt")
@click.option("--open", "open_result", is_flag=True, help="Open first result in image viewer")
@click.option(
    "--text-weight",
    type=float,
    default=0.5,
    help="Weight for text query in multi-modal search (0.0-1.0)",
)
@click.option(
    "--image-weight",
    type=float,
    default=0.5,
    help="Weight for image query in multi-modal search (0.0-1.0)",
)
@click.pass_context
def search(
    ctx,
    query: Optional[str],
    image_path: Optional[str],
    search_mode: Optional[str],
    limit: int,
    from_time: Optional[str],
    to_time: Optional[str],
    output_format: str,
    preview: bool,
    open_result: bool,
    text_weight: float,
    image_weight: float,
):
    """Search through screenshot history.

    Search for text in OCR data, use semantic search with embeddings,
    or search by image similarity.

    Examples:
      # Text search
      flashback search "meeting notes"
      flashback search -m text_hybrid "dashboard with charts"
      flashback search "error" --from 1d --to now

      # Image search
      flashback search --image screenshot.png --search-mode image_embedding_only
      flashback search --image reference.jpg -n 10

      # Multi-modal search (text + image)
      flashback search "meeting" --image whiteboard.jpg --search-mode text_and_image
      flashback search "bug" --image error_dialog.png -m comprehensive

      # Output options
      flashback search "TODO" -F json
      flashback search "login" -n 5 --open
    """
    config = Config(config_path=ctx.obj.get("config_path"))
    db = Database(config.db_path)

    # Validate inputs
    if not query and not image_path:
        console.print("[red]Error: Either QUERY or --image must be provided[/red]")
        ctx.exit(1)

    # Determine search mode
    if search_mode is None:
        if image_path and query:
            search_mode = "text_and_image"
        elif image_path:
            search_mode = "image_embedding_only"
        else:
            search_mode = config.get_default_search_mode()

    # Parse time range
    start_ts = None
    end_ts = None
    if from_time:
        try:
            start_ts = parse_time(from_time)
        except ValueError as e:
            console.print(f"[red]Invalid --from: {e}[/red]")
            return
    if to_time:
        try:
            end_ts = parse_time(to_time)
        except ValueError as e:
            console.print(f"[red]Invalid --to: {e}[/red]")
            return

    # Perform search
    results = []
    score_breakdown = {}

    with console.status("[bold green]Searching..."):
        if search_mode == "bm25_only":
            results, score_breakdown = _search_bm25(config, db, query, limit)
        elif search_mode == "text_embedding_only":
            results, score_breakdown = _search_text_embedding(config, db, query, limit)
        elif search_mode == "text_hybrid":
            results, score_breakdown = _search_text_hybrid(config, db, query, limit)
        elif search_mode == "image_embedding_only":
            results, score_breakdown = _search_image(config, db, image_path, limit)
        elif search_mode == "text_to_image":
            results, score_breakdown = _search_text_to_image(config, db, query, limit)
        elif search_mode in ("text_and_image", "comprehensive"):
            results, score_breakdown = _search_multi_modal(
                config, db, query, image_path, limit, text_weight, image_weight
            )
        else:
            console.print(f"[red]Unknown search mode: {search_mode}[/red]")
            return

    # Filter by time range
    if start_ts or end_ts:
        filtered_results = []
        for doc_id, score in results:
            record = db.get_by_id(doc_id)
            if not record:
                continue
            if start_ts and record.timestamp < start_ts:
                continue
            if end_ts and record.timestamp > end_ts:
                continue
            filtered_results.append((doc_id, score))
        results = filtered_results

    # Format output
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    _display_results(results, db, query, search_mode, output_format, preview, score_breakdown)

    # Open first result if requested
    if open_result and results:
        record = db.get_by_id(results[0][0])
        if record:
            viewer_cmd = config.get("viewer.command", "xdg-open")
            subprocess.Popen(
                [viewer_cmd, record.screenshot_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def _search_bm25(config, db, query: str, limit: int) -> tuple:
    """Perform BM25 search."""
    bm25 = BM25Search(config, db)
    results = bm25.search(query, top_k=limit * 2)
    return results, {"bm25": len(results)}


def _search_text_embedding(config, db, query: str, limit: int) -> tuple:
    """Perform text embedding search."""
    text_search = TextEmbeddingSearch(config, db)
    results = text_search.search(query, top_k=limit * 2)
    return results, {"text_embedding": len(results)}


def _search_text_hybrid(config, db, query: str, limit: int) -> tuple:
    """Perform hybrid BM25 + text embedding search."""
    bm25_results = []
    emb_results = []

    try:
        bm25 = BM25Search(config, db)
        bm25_results = bm25.search(query, top_k=limit * 2)
    except Exception as e:
        console.print(f"[yellow]BM25 search error: {e}[/yellow]")

    try:
        text_search = TextEmbeddingSearch(config, db)
        emb_results = text_search.search(query, top_k=limit * 2)
    except Exception as e:
        console.print(f"[yellow]Embedding search error: {e}[/yellow]")

    results = reciprocal_rank_fusion(bm25_results, emb_results, k=60, top_k=limit)
    return results, {"bm25": len(bm25_results), "text_embedding": len(emb_results)}


def _search_image(config, db, image_path: str, limit: int) -> tuple:
    """Perform image embedding search."""
    image_search = ImageEmbeddingSearch(config, db)
    results = image_search.search_by_image(image_path, top_k=limit * 2)
    return results, {"image_embedding": len(results)}


def _search_text_to_image(config, db, query: str, limit: int) -> tuple:
    """Perform text-to-image search (requires CLIP-like model)."""
    image_search = ImageEmbeddingSearch(config, db)
    results = image_search.search_by_text(query, top_k=limit * 2)
    return results, {"text_to_image": len(results)}


def _search_multi_modal(
    config, db, query: Optional[str], image_path: Optional[str], limit: int,
    text_weight: float, image_weight: float
) -> tuple:
    """Perform multi-modal search with text and/or image."""
    hybrid = HybridEmbeddingSearch(config, db)

    # Override weights
    original_weights = hybrid.weights.copy()
    hybrid.weights["text_weight"] = text_weight
    hybrid.weights["image_weight"] = image_weight

    image = None
    if image_path:
        from PIL import Image
        image = Image.open(image_path)

    results, metadata = hybrid.search_fused(
        text_query=query,
        image_query=image,
        top_k=limit,
    )

    # Restore weights
    hybrid.weights = original_weights

    return results, metadata


def _display_results(results, db, query, search_mode, output_format, preview, score_breakdown):
    """Display search results in the requested format."""
    formatted_results = []
    for doc_id, score in results:
        record = db.get_by_id(doc_id)
        if not record:
            continue
        formatted_results.append((record, score))

    if not formatted_results:
        console.print("[yellow]No results found[/yellow]")
        return

    if output_format == "simple":
        for record, _ in formatted_results:
            console.print(record.screenshot_path)
    elif output_format == "json":
        output = [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "path": r.screenshot_path,
                "window_title": r.window_title,
                "score": score,
                "ocr_preview": (r.ocr_text or "")[:200] if preview else None,
            }
            for r, score in formatted_results
        ]
        console.print(json.dumps({
            "query": query,
            "search_mode": search_mode,
            "score_breakdown": score_breakdown,
            "results": output,
        }, indent=2))
    elif output_format == "csv":
        console.print("id,timestamp,path,window_title,score")
        for record, score in formatted_results:
            console.print(
                f"{record.id},{record.timestamp_formatted},{record.screenshot_path},"
                f"{record.window_title or ''},{score:.4f}"
            )
    else:  # table
        table = Table(title=f"Search Results: \"{query or '(image query)'}\" ({search_mode})")
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Time", style="green")
        table.add_column("Score", style="yellow")
        table.add_column("Window", style="blue")
        if preview:
            table.add_column("Preview", style="dim")

        for i, (record, score) in enumerate(formatted_results, 1):
            row = [
                str(i),
                record.timestamp_formatted,
                f"{score:.2f}",
                (record.window_title or "")[:30],
            ]
            if preview:
                text = (record.ocr_text or "")[:100].replace("\n", " ")
                row.append(text + "..." if len(record.ocr_text or "") > 100 else text)
            table.add_row(*row)

        console.print(table)

        # Show score breakdown if available
        if score_breakdown:
            breakdown_str = ", ".join(
                f"{k}: {v}" for k, v in score_breakdown.items() if not k.endswith("_error")
            )
            if breakdown_str:
                console.print(f"\n[dim]Score breakdown: {breakdown_str}[/dim]")

        console.print(f"\n[dim]Press 1-{min(len(formatted_results), 9)} to view, or q to quit[/dim]")
