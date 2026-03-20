"""Stop command for flashback."""

import click
from rich.console import Console

from flashback.core.daemon import DaemonManager

console = Console()


@click.command()
@click.option("--backend", is_flag=True, help="Stop only backend daemon")
@click.option("--webui", is_flag=True, help="Stop only webui daemon")
@click.option("--all", "stop_all", is_flag=True, help="Stop all daemons (default)")
@click.option("--force", "-f", is_flag=True, help="Force kill (SIGKILL)")
def stop(backend, webui, stop_all, force):
    """Stop running daemon(s).

    Examples:
      flashback stop           # Stop all daemons
      flashback stop --backend # Stop only backend
      flashback stop -f        # Force kill
    """
    if not backend and not webui:
        stop_all = True

    stopped = []

    if stop_all or backend:
        mgr = DaemonManager("backend")
        if mgr.is_running():
            if mgr.stop(force=force):
                stopped.append("backend")

    if stop_all or webui:
        mgr = DaemonManager("webui")
        if mgr.is_running():
            if mgr.stop(force=force):
                stopped.append("webui")

    if stopped:
        console.print(f"[green]Stopped: {', '.join(stopped)}[/green]")
    else:
        console.print("[yellow]No running daemons found[/yellow]")
