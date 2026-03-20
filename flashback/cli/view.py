"""View command for flashback."""

import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console

from flashback.core.config import Config
from flashback.core.database import Database

console = Console()


@click.command()
@click.argument("timestamp_or_path")
@click.option("--text", "-t", is_flag=True, help="Show OCR text in terminal")
@click.option("--neighbors", "-n", default=0, help="Show N timeline neighbors")
@click.option("--copy", "-c", is_flag=True, help="Copy image path to clipboard")
@click.option("--export", "export_path", type=click.Path(), help="Copy image to path")
@click.pass_context
def view(ctx, timestamp_or_path, text, neighbors, copy, export_path):
    """View a specific screenshot.

    Opens the screenshot in your system's image viewer, or shows OCR text.

    TIMESTAMP_OR_PATH can be:
      - Timestamp: YYYYMMDD_HHMMSS or YYYY-MM-DD HH:MM:SS
      - Full path to screenshot file

    Examples:
      flashback view 20240320_142312
      flashback view "2024-03-20 14:23:12"
      flashback view /path/to/screenshot.png -t
      flashback view 20240320_142312 -n 10
      flashback view 20240320_142312 --export ~/Downloads/screenshot.png
    """
    config = Config(config_path=ctx.obj.get("config_path"))
    db = Database(config.db_path)

    # Parse timestamp or path
    timestamp = None
    screenshot_path = None

    if Path(timestamp_or_path).exists():
        screenshot_path = timestamp_or_path
    else:
        # Parse timestamp
        ts_str = timestamp_or_path.replace("_", " ").replace("-", "")
        for fmt in ["%Y%m%d %H%M%S", "%Y%m%d%H%M%S"]:
            try:
                from datetime import datetime

                dt = datetime.strptime(ts_str[:15], fmt)
                timestamp = dt.timestamp()
                break
            except ValueError:
                continue

    # Get record from database
    record = None
    if timestamp:
        record = db.get_by_timestamp(timestamp)
    elif screenshot_path:
        # Find by path
        # This is inefficient but works for now
        all_records = db.get_unprocessed_ocr(limit=10000)
        for r in all_records:
            if r.screenshot_path == screenshot_path:
                record = r
                break

    if not record:
        console.print(f"[red]Screenshot not found: {timestamp_or_path}[/red]")
        sys.exit(1)

    # Handle options
    if text:
        if record.ocr_text:
            console.print(record.ocr_text)
        else:
            console.print("[yellow]No OCR text available for this screenshot[/yellow]")
        return

    if copy:
        try:
            import pyperclip

            pyperclip.copy(record.screenshot_path)
            console.print(f"[green]Copied: {record.screenshot_path}[/green]")
        except ImportError:
            console.print("[red]pyperclip not installed. Run: pip install pyperclip[/red]")
        return

    if export_path:
        import shutil

        shutil.copy(record.screenshot_path, export_path)
        console.print(f"[green]Exported to: {export_path}[/green]")
        return

    # Show neighbors if requested
    if neighbors > 0:
        neighbor_records = db.get_neighbors(record.timestamp, window_seconds=neighbors * 300)
        console.print(f"\n[bold]Timeline Context ({neighbors} neighbors):[/bold]\n")

        for r in sorted(neighbor_records, key=lambda x: x.timestamp):
            rel = ""
            if r.timestamp < record.timestamp:
                rel = f"[-{int((record.timestamp - r.timestamp) / 60)}m]"
            elif r.timestamp > record.timestamp:
                rel = f"[+{int((r.timestamp - record.timestamp) / 60)}m]"
            else:
                rel = "[NOW]"

            marker = " <--" if abs(r.timestamp - record.timestamp) < 1 else ""
            console.print(f"{rel} {r.timestamp_formatted} - {r.window_title or 'Unknown'}{marker}")
        console.print()

    # Open in image viewer
    viewer_cmd = config.get("viewer.command", "xdg-open")
    viewer_args = config.get("viewer.args", ["{path}"])

    # Replace {path} placeholder
    args = [arg.replace("{path}", record.screenshot_path) for arg in viewer_args]
    cmd = [viewer_cmd] + args

    console.print(f"[dim]Opening: {record.screenshot_path}[/dim]")
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
