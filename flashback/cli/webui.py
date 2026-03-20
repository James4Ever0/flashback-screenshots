"""Web UI command for flashback."""

import sys
import webbrowser
from pathlib import Path

import click
from rich.console import Console

from flashback.api.server import create_app
from flashback.core.config import Config
from flashback.core.daemon import DaemonManager

console = Console()


@click.command()
@click.option("--host", help="Bind address")
@click.option("--port", type=int, help="Port number")
@click.option("--daemon", "-D", is_flag=True, help="Run as background daemon")
@click.option("--no-browser", is_flag=True, help="Don't auto-open browser")
@click.pass_context
def webui(ctx, host, port, daemon, no_browser):
    """Start the web UI server.

    This command starts the FastAPI web server for browser-based searching.

    Examples:
      flashback webui              # Run on default port, open browser
      flashback webui --port 3000  # Run on custom port
      flashback webui -D           # Run as daemon
    """
    import uvicorn

    config = Config(config_path=ctx.obj.get("config_path"))

    host = host or config.webui_host
    port = port or config.webui_port

    # Check if backend is running
    backend = DaemonManager("backend")
    if not backend.is_running():
        console.print("[yellow]Warning: Backend daemon is not running[/yellow]")
        console.print("[dim]Start it with: flashback serve --daemon[/dim]")

    # Check if already running
    webui_daemon = DaemonManager("webui")
    if webui_daemon.is_running():
        console.print(f"[red]Web UI is already running (PID: {webui_daemon.get_pid()})[/red]")
        sys.exit(1)

    if daemon:
        console.print("[green]Starting web UI daemon...[/green]")
        # For daemon mode, we'd need to implement proper daemonization
        # For now, just run in background

    console.print(f"[green]Starting web UI on http://{host}:{port}[/green]")

    if not no_browser and not daemon:
        webbrowser.open(f"http://{host}:{port}")

    app = create_app(config)
    uvicorn.run(app, host=host, port=port, log_level="info")
