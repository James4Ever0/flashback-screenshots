"""Main CLI entry point for flashback."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

# Import subcommands
from flashback.cli.serve import serve
from flashback.cli.webui import webui
from flashback.cli.status import status
from flashback.cli.search import search
from flashback.cli.view import view
from flashback.cli.config_cmd import config
from flashback.cli.stop import stop
from flashback.cli.logs import logs

console = Console()


@click.group()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--data-dir", "-d",
    type=click.Path(path_type=Path),
    help="Override data directory",
)
@click.option("--verbose", "-v", count=True, help="Increase verbosity (-v, -vv, -vvv)")
@click.option("--quiet", "-q", is_flag=True, help="Only show errors")
@click.option("--debug", is_flag=True, help="Enable debug logging (same as -vv)")
@click.option("--trace", is_flag=True, help="Enable trace logging (maximum verbosity)")
@click.option("--log-file", type=click.Path(path_type=Path), help="Log to file")
@click.version_option(version="0.1.0", prog_name="flashback")
@click.pass_context
def cli(ctx, config, data_dir, verbose, quiet, debug, trace, log_file):
    """Flashback - Search through your screenshot history.

    Flashback captures screenshots, performs OCR, and creates semantic embeddings
to make your visual history searchable.

    Commands:
      serve     Start the backend daemon (captures screenshots)
      webui     Start the web UI server
      status    Check daemon health and status
      search    Search screenshots from command line
      view      View a specific screenshot
      config    Manage configuration
      stop      Stop running daemon(s)
      logs      View daemon logs

    Examples:
      flashback serve --daemon          # Start backend as daemon
      flashback search "meeting notes"  # Search for text in screenshots
      flashback view 20240320_142312    # View specific screenshot
      flashback -vv serve              # Start with debug logging
    """
    # Import here to avoid circular imports
    from flashback.core.config import Config
    from flashback.core.logging_config import setup_logging, get_log_level_from_verbosity

    # Determine log level
    if trace:
        level = "DEBUG"
    elif debug:
        level = "DEBUG"
    else:
        level = get_log_level_from_verbosity(verbose, quiet)

    # Load config and setup logging
    cfg = Config(config_path=config)
    if data_dir:
        cfg.set("data_dir", str(data_dir))

    setup_logging(cfg, level=level, log_file=str(log_file) if log_file else None, trace=trace)

    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["data_dir"] = data_dir
    ctx.obj["verbose"] = verbose or debug or trace


# Register subcommands
cli.add_command(serve)
cli.add_command(webui)
cli.add_command(status)
cli.add_command(search)
cli.add_command(view)
cli.add_command(config, name="config")
cli.add_command(stop)
cli.add_command(logs)


def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
