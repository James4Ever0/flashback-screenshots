"""Serve command for flashback backend."""

import signal
import sys
import time
from pathlib import Path

import click
from rich.console import Console

from flashback.core.config import Config
from flashback.core.daemon import DaemonManager, daemonize
from flashback.core.database import Database
from flashback.workers.screenshot import ScreenshotWorker
from flashback.workers.ocr import OCRWorker
from flashback.workers.embedding import EmbeddingWorker
from flashback.workers.cleanup import CleanupWorker
from flashback.workers.window_title import WindowTitleWorker

console = Console()


def check_dependencies(config: Config):
    """Check if required dependencies are available."""
    errors = []

    if config.is_worker_enabled("ocr"):
        try:
            import pytesseract
            # Try to run tesseract
            pytesseract.get_tesseract_version()
        except Exception as e:
            errors.append(f"OCR (tesseract) not available: {e}")
            errors.append("  Install: sudo apt-get install tesseract-ocr")

    if config.is_worker_enabled("embedding"):
        try:
            import sentence_transformers
        except ImportError:
            errors.append("Embedding (sentence-transformers) not installed")
            errors.append("  Install: pip install sentence-transformers")

    if errors:
        for error in errors:
            console.print(f"[red]{error}[/red]")
        if click.confirm("Continue without these features?"):
            # Disable problematic workers
            for worker in ["ocr", "embedding"]:
                if any(worker in e for e in errors):
                    config.set(f"workers.{worker}.enabled", False)
        else:
            sys.exit(1)


def run_workers(config: Config):
    """Run all enabled workers."""
    db = Database(config.db_path)
    workers = []

    # Screenshot worker (core)
    if config.is_worker_enabled("screenshot"):
        workers.append(ScreenshotWorker(config=config, db=db))

    # OCR worker
    if config.is_worker_enabled("ocr"):
        try:
            workers.append(OCRWorker(config=config, db=db))
        except Exception as e:
            console.print(f"[yellow]Failed to start OCR worker: {e}[/yellow]")

    # Embedding worker
    if config.is_worker_enabled("embedding"):
        try:
            workers.append(EmbeddingWorker(config=config, db=db))
        except Exception as e:
            console.print(f"[yellow]Failed to start Embedding worker: {e}[/yellow]")

    # Cleanup worker
    if config.is_worker_enabled("cleanup"):
        workers.append(CleanupWorker(config=config, db=db))

    # Window title worker
    if config.is_worker_enabled("window_title"):
        try:
            workers.append(WindowTitleWorker(config=config, db=db))
        except Exception as e:
            console.print(f"[yellow]Failed to start Window Title worker: {e}[/yellow]")

    if not workers:
        console.print("[red]No workers enabled![/red]")
        sys.exit(1)

    # Start all workers
    for worker in workers:
        worker.start()
        console.print(f"[green]Started {worker.name}[/green]")

    # Setup signal handlers
    running = True

    def signal_handler(signum, frame):
        nonlocal running
        console.print("\n[yellow]Shutting down...[/yellow]")
        running = False
        for worker in workers:
            worker.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Keep main thread alive
    while running:
        # Check if any worker died
        for worker in workers:
            if not worker.is_alive() and running:
                console.print(f"[red]{worker.name} died unexpectedly[/red]")
        time.sleep(1)

    # Wait for workers to finish
    for worker in workers:
        worker.join(timeout=5)

    console.print("[green]Shutdown complete[/green]")


@click.command()
@click.option("--daemon", "-D", is_flag=True, help="Run as background daemon")
@click.option("--foreground", "-F", is_flag=True, help="Run in foreground (default)")
@click.option("--pid-file", type=click.Path(path_type=Path), help="Write PID to file")
@click.option("--log-file", type=click.Path(path_type=Path), help="Write logs to file")
@click.pass_context
def serve(ctx, daemon, foreground, pid_file, log_file):
    """Start the backend daemon (screenshot capture and workers).

    This command starts all enabled workers including screenshot capture,
    OCR, embedding generation, and cleanup.

    Examples:
      flashback serve              # Run in foreground
      flashback serve --daemon     # Run as background daemon
      flashback serve -D --pid-file /tmp/flashback.pid
    """
    # Load config
    config = Config(config_path=ctx.obj.get("config_path"))

    # Check if already running
    daemon_mgr = DaemonManager("backend")
    if daemon_mgr.is_running():
        console.print(f"[red]Backend is already running (PID: {daemon_mgr.get_pid()})[/red]")
        sys.exit(1)

    # Check dependencies
    check_dependencies(config)

    if daemon:
        # Run as daemon
        console.print("[green]Starting backend daemon...[/green]")

        if pid_file:
            daemon_mgr.pid_file = pid_file
        if log_file:
            daemon_mgr.log_file = log_file

        pid = daemonize(run_workers, daemon_mgr, config)
        if pid:
            console.print(f"[green]Daemon started (PID: {pid})[/green]")
        else:
            console.print("[red]Failed to start daemon[/red]")
            sys.exit(1)
    else:
        # Run in foreground
        console.print("[bold]Flashback Backend[/bold]")
        console.print(f"Data directory: {config.data_dir}")
        console.print(f"Workers: {[w for w in ['screenshot', 'ocr', 'embedding', 'cleanup', 'window_title'] if config.is_worker_enabled(w)]}")
        console.print("Press Ctrl+C to stop\n")

        run_workers(config)
