"""Logs command for flashback."""

import subprocess
import sys

import click
from rich.console import Console

from flashback.core.daemon import DaemonManager

console = Console()


@click.command()
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--lines", "-n", default=50, help="Show last N lines")
@click.option("--backend", is_flag=True, help="Show backend logs")
@click.option("--webui", is_flag=True, help="Show webui logs")
@click.option("--clear", is_flag=True, help="Clear log files")
def logs(follow, lines, backend, webui, clear):
    """View daemon logs.

    Examples:
      flashback logs
      flashback logs -f
      flashback logs -n 100 --backend
    """
    if not backend and not webui:
        backend = True

    if clear:
        if backend:
            mgr = DaemonManager("backend")
            log_file = mgr.get_log_path()
            if log_file.exists():
                log_file.write_text("")
                console.print(f"[green]Cleared: {log_file}[/green]")

        if webui:
            mgr = DaemonManager("webui")
            log_file = mgr.get_log_path()
            if log_file.exists():
                log_file.write_text("")
                console.print(f"[green]Cleared: {log_file}[/green]")
        return

    log_files = []

    if backend:
        mgr = DaemonManager("backend")
        log_files.append(mgr.get_log_path())

    if webui:
        mgr = DaemonManager("webui")
        log_files.append(mgr.get_log_path())

    for log_file in log_files:
        if not log_file.exists():
            console.print(f"[yellow]Log file not found: {log_file}[/yellow]")
            continue

        if follow:
            # Use tail -f
            try:
                subprocess.run(["tail", "-f", "-n", str(lines), str(log_file)])
            except KeyboardInterrupt:
                console.print()
        else:
            content = mgr.read_logs(lines=lines)
            console.print(f"[bold]{log_file}:[/bold]")
            console.print(content)
