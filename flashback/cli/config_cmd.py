"""Config command for flashback."""

from pathlib import Path

import click
from rich.console import Console
from rich.syntax import Syntax

from flashback.core.config import Config, get_config_dir

console = Console()


@click.group()
def config():
    """Manage flashback configuration.

    Examples:
      flashback config show
      flashback config edit
      flashback config init
      flashback config get workers.ocr.enabled
      flashback config set workers.screenshot.interval_seconds 300
    """
    pass


@config.command()
@click.pass_context
def show(ctx):
    """Display current configuration."""
    config = Config(config_path=ctx.obj.get("config_path"))
    cfg_dict = config.to_dict()

    import yaml

    yaml_str = yaml.dump(cfg_dict, default_flow_style=False, sort_keys=True)
    console.print(Syntax(yaml_str, "yaml"))


@config.command()
def edit():
    """Open configuration in $EDITOR."""
    import os
    import subprocess

    config_path = get_config_dir() / "config.yaml"

    editor = os.environ.get("EDITOR", "nano")
    subprocess.call([editor, str(config_path)])


@config.command()
@click.option("--path", type=click.Path(path_type=Path), help="Config file location")
def init(path):
    """Create default configuration file."""
    config_path = Config.create_default(path)
    console.print(f"[green]Created configuration: {config_path}[/green]")


@config.command()
@click.argument("key")
@click.pass_context
def get(ctx, key):
    """Get a configuration value."""
    config = Config(config_path=ctx.obj.get("config_path"))
    value = config.get(key)

    if value is None:
        console.print(f"[yellow]Key not found: {key}[/yellow]")
    else:
        console.print(value)


@config.command()
@click.argument("key")
@click.argument("value")
@click.pass_context
def set(ctx, key, value):
    """Set a configuration value."""
    config = Config(config_path=ctx.obj.get("config_path"))

    # Try to parse as int, float, bool, or keep as string
    parsed_value = value
    try:
        parsed_value = int(value)
    except ValueError:
        try:
            parsed_value = float(value)
        except ValueError:
            if value.lower() in ("true", "yes", "on"):
                parsed_value = True
            elif value.lower() in ("false", "no", "off"):
                parsed_value = False

    config.set(key, parsed_value)
    config.save()
    console.print(f"[green]Set {key} = {parsed_value}[/green]")


@config.command()
def validate():
    """Validate configuration file."""
    try:
        Config()
        console.print("[green]Configuration is valid[/green]")
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")


@config.command("test-embedding")
@click.option(
    "--type",
    "embedding_type",
    type=click.Choice(["text", "image"]),
    required=True,
    help="Type of embedding to test (text or image)",
)
@click.option(
    "--write",
    is_flag=True,
    help="Write detected dimension to configuration file",
)
@click.option(
    "--image",
    "test_image_path",
    type=click.Path(exists=True),
    help="Path to test image for image embedding test",
)
@click.pass_context
def test_embedding(ctx, embedding_type: str, write: bool, test_image_path: str = None):
    """Test embedding API connection and detect dimension.

    This command tests the connection to the configured embedding API
    and automatically detects the embedding dimension.

    Examples:
      # Test text embedding API
      flashback config test-embedding --type text

      # Test image embedding API
      flashback config test-embedding --type image

      # Test and save dimension to config
      flashback config test-embedding --type text --write
      flashback config test-embedding --type image --write
    """
    from flashback.core.embedding_client import EmbeddingAPIClient

    config = Config(config_path=ctx.obj.get("config_path"))

    # Get the appropriate config
    if embedding_type == "text":
        api_config = config.get_text_embedding_config()
        if not api_config.get("model"):
            console.print("[red]Error: No text embedding model configured[/red]")
            console.print("[yellow]Set workers.embedding.text.model in your config[/yellow]")
            return
    else:  # image
        api_config = config.get_image_embedding_config()
        if not api_config.get("model"):
            console.print("[red]Error: No image embedding model configured[/red]")
            console.print("[yellow]Set workers.embedding.image.model in your config[/yellow]")
            return

    # Create client
    try:
        client = EmbeddingAPIClient(
            base_url=api_config.get("base_url", "https://api.openai.com/v1"),
            api_key=api_config.get("api_key", ""),
            model=api_config["model"],
            dimension=None,  # Don't validate during testing
            extra_headers=api_config.get("extra_headers", {}),
            name=embedding_type,
        )
    except Exception as e:
        console.print(f"[red]Failed to create embedding client: {e}[/red]")
        return

    # Test connection
    console.print(f"[bold]Testing {embedding_type} embedding API...[/bold]")
    console.print(f"  Base URL: {client.base_url}")
    console.print(f"  Model: {client.model}")
    console.print("")

    if embedding_type == "text":
        result = client.test_connection()
    else:  # image
        result = client.test_image_embedding(test_image_path)

    if result["success"]:
        console.print(f"[green]✓ Connection successful![/green]")
        console.print(f"  Detected dimension: [bold]{result['dimension']}[/bold]")

        if write:
            config.set_embedding_dimension(embedding_type, result["dimension"])
            config.save()
            console.print(f"[green]✓ Saved dimension to configuration[/green]")
        else:
            console.print(f"\n[dim]To save this dimension, run:[/dim]")
            console.print(f"  flashback config test-embedding --type {embedding_type} --write")
    else:
        console.print(f"[red]✗ Connection failed:[/red]")
        console.print(f"  {result['message']}")

        console.print(f"\n[yellow]Troubleshooting:[/yellow]")
        if embedding_type == "text":
            console.print("  1. Verify workers.embedding.text.base_url is correct")
            console.print("  2. Check that workers.embedding.text.api_key is valid")
            console.print("  3. Ensure the model name is correct")
        else:
            console.print("  1. Verify workers.embedding.image.base_url is correct")
            console.print("  2. Ensure you're using a vision-capable model (e.g., llava)")
            console.print("  3. Check that the embedding server supports image inputs")
