"""Status command for flashback."""

import json
import time

import click
from rich.console import Console
from rich.table import Table

from flashback.core.config import Config
from flashback.core.daemon import DaemonManager
from flashback.core.database import Database

console = Console()


@click.command()
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--watch", "-w", type=int, help="Watch mode (refresh every N seconds)")
@click.pass_context
def status(ctx, json_output, watch):
    """Check backend health and status.

    Examples:
      flashback status
      flashback status --json
      flashback status -w 5
    """
    config = Config(config_path=ctx.obj.get("config_path"))

    def get_status():
        backend = DaemonManager("backend")
        webui = DaemonManager("webui")
        db = Database(config.db_path)
        stats = db.get_stats()

        return {
            "backend": {
                "running": backend.is_running(),
                "pid": backend.get_pid(),
            },
            "webui": {
                "running": webui.is_running(),
                "pid": webui.get_pid(),
            },
            "database": {
                "screenshot_count": stats["total"],
                "with_ocr": stats.get("with_ocr", 0),
                "with_embedding": stats.get("with_embedding", 0),
                "oldest_timestamp": stats.get("oldest_timestamp"),
                "newest_timestamp": stats.get("newest_timestamp"),
            },
        }

    def render_table(data):
        table = Table(title="Flashback Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")

        backend = data["backend"]
        status_str = "[green]Running[/green]" if backend["running"] else "[red]Stopped[/red]"
        if backend["pid"]:
            status_str += f" (PID: {backend['pid']})"
        table.add_row("Backend", status_str)

        webui = data["webui"]
        status_str = "[green]Running[/green]" if webui["running"] else "[red]Stopped[/red]"
        if webui["pid"]:
            status_str += f" (PID: {webui['pid']})"
        table.add_row("Web UI", status_str)

        db = data["database"]
        table.add_row("Screenshots", str(db["screenshot_count"]))
        table.add_row("With OCR", str(db.get("with_ocr", 0)))
        table.add_row("With Embeddings", str(db.get("with_embedding", 0)))

        console.print(table)

    if watch:
        try:
            while True:
                console.clear()
                data = get_status()
                if json_output:
                    console.print(json.dumps(data, indent=2))
                else:
                    render_table(data)
                time.sleep(watch)
        except KeyboardInterrupt:
            console.print()
    else:
        data = get_status()
        if json_output:
            console.print(json.dumps(data, indent=2))
        else:
            render_table(data)
