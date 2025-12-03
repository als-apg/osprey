"""Configuration export command.

This module provides the 'osprey export-config' command which displays
the osprey's default configuration template. This helps users understand what
configuration options are available when creating new projects.
"""

from pathlib import Path

import click
import yaml
from jinja2 import Template
from rich.syntax import Syntax

from osprey.cli.styles import Styles, console


@click.command(name="export-config")
@click.option(
    "--project", "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Project directory (default: current directory or OSPREY_PROJECT env var)"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file (default: print to console)"
)
@click.option(
    "--format",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    help="Output format (default: yaml)"
)
def export_config(project: str, output: str, format: str):
    """Export osprey default configuration template.

    .. deprecated::
       Use 'osprey config export' instead. This command is kept for backward
       compatibility but will be removed in a future version.

    Displays the osprey's default configuration template that is used
    when creating new projects with 'osprey init'. This is useful for:

    \b
      - Understanding available configuration options
      - Seeing default values for models, services, etc.
      - Debugging configuration issues
      - Creating custom configurations

    The exported configuration shows the complete osprey template
    rendered with example values.

    Examples:

    \b
      # Display to console with syntax highlighting
      $ osprey export-config

      # Save to file
      $ osprey export-config -o osprey-defaults.yml

      # Export as JSON
      $ osprey export-config --format json -o osprey-defaults.json
    """
    # Show deprecation warning
    console.print(
        "⚠️  [yellow]DEPRECATED:[/yellow] 'osprey export-config' is deprecated.",
        style=Styles.WARNING
    )
    console.print(
        "   Use [bold cyan]osprey config export[/bold cyan] instead.\n",
        style=Styles.DIM
    )

    try:
        # Load osprey's configuration template (known location in osprey structure)
        template_path = Path(__file__).parent.parent / "templates" / "project" / "config.yml.j2"

        if not template_path.exists():
            console.print(
                "❌ Could not locate osprey configuration template.",
                style=Styles.ERROR
            )
            console.print(
                f"   Expected at: {template_path}",
                style=Styles.DIM
            )
            raise click.Abort()

        # Read and render the template with example values
        with open(template_path) as f:
            template_content = f.read()

        template = Template(template_content)
        rendered_config = template.render(
            project_name="example_project",
            package_name="example_project",
            project_root="/path/to/example_project",
            hostname="localhost",
            default_provider="cborg",
            default_model="anthropic/claude-haiku"
        )

        # Parse the rendered config as YAML
        config_data = yaml.safe_load(rendered_config)

        # Format output based on requested format
        if format == "yaml":
            output_str = yaml.dump(
                config_data,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )
        else:  # json
            import json
            output_str = json.dumps(config_data, indent=2, ensure_ascii=False)

        # Output to file or console
        if output:
            output_path = Path(output)
            output_path.write_text(output_str)
            console.print(
                f"✅ Configuration exported to: [bold]{output_path}[/bold]"
            )
        else:
            # Print to console with syntax highlighting
            console.print("\n[bold]Osprey Default Configuration:[/bold]\n")
            syntax = Syntax(
                output_str,
                format,
                theme="monokai",
                line_numbers=False,
                word_wrap=True
            )
            console.print(syntax)
            console.print(
                "\n[dim]💡 Tip: Save to file with --output flag[/dim]"
            )

    except KeyboardInterrupt:
        console.print("\n⚠️  Operation cancelled", style=Styles.WARNING)
        raise click.Abort()
    except Exception as e:
        console.print(f"❌ Failed to export configuration: {e}", style=Styles.ERROR)
        import os
        if os.environ.get("DEBUG"):
            import traceback
            console.print(traceback.format_exc(), style=Styles.DIM)
        raise click.Abort()


if __name__ == "__main__":
    export_config()

