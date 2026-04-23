"""Project initialization command.

This module provides the 'osprey init' command which creates new
projects from templates with Claude Code integration (MCP servers,
hooks, and rules).
"""

from pathlib import Path

import click

from .project_utils import _clear_claude_code_project_state
from .styles import Messages, Styles, console
from .templates.manager import TemplateManager


@click.command()
@click.argument("project_name")
@click.option(
    "--template",
    "-t",
    type=click.Choice(["hello_world", "control_assistant"], case_sensitive=False),
    default="control_assistant",
    help="Application template to use (hello_world, control_assistant)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=".",
    help="Output directory for project (default: current directory)",
)
@click.option(
    "--force", "-f", is_flag=True, help="Force overwrite if project directory already exists"
)
def init(
    project_name: str,
    template: str,
    output_dir: str,
    force: bool,
):
    """Create a new project.

    Creates a project configured for Claude Code with MCP servers,
    hooks, and rules.

    PROJECT_NAME: Name of your project (e.g., my-assistant, beamline-agent)

    \b
      # Create project with default control_assistant template
      $ osprey init my-assistant

      # Create in specific location
      $ osprey init my-assistant --output-dir /projects

      # Force overwrite if directory exists
      $ osprey init my-assistant --force
    """
    console.print(f"Creating project: [header]{project_name}[/header]")

    try:
        # Create template manager
        manager = TemplateManager()

        # Show available templates
        available_templates = manager.list_app_templates()
        if template not in available_templates:
            console.print(
                f"✗ Template '{template}' not found.\n"
                f"Available templates: {', '.join(available_templates)}",
                style=Styles.ERROR,
            )
            raise click.Abort()

        console.print(f"  Using template: [accent]{template}[/accent]")
        console.print("  Mode: Claude Code")

        from .templates.scaffolding import detect_environment_variables

        detected_env = detect_environment_variables()
        if detected_env:
            console.print(f"  Detected {len(detected_env)} environment variable(s) from system:")
            for env_var in detected_env.keys():
                console.print(f"     • {env_var}", style=Styles.DIM)

        # Handle existing directory
        output_path = Path(output_dir).resolve()
        project_path = output_path / project_name

        if project_path.exists():
            if force:
                msg = Messages.warning(f"Removing existing directory: {project_path}")
                console.print(f"  !  {msg}")
                import shutil

                shutil.rmtree(project_path)
                console.print(f"  {Messages.success('Removed existing directory')}")
            else:
                console.print(
                    f"✗ Directory '{project_path}' already exists.\n"
                    f"   Use --force to overwrite, or choose a different name.",
                    style=Styles.ERROR,
                )
                raise click.Abort()

        _clear_claude_code_project_state(project_path)

        project_path = manager.create_project(
            project_name=project_name,
            output_dir=output_path,
            template_name=template,
            registry_style="extend",
            context={},
        )

        manager.generate_manifest(
            project_dir=project_path,
            project_name=project_name,
            template_name=template,
            registry_style="extend",
            context={
                "default_provider": "anthropic",
                "default_model": "haiku",
            },
        )

        console.print("  ✓ Creating project configuration...", style=Styles.SUCCESS)
        console.print("  ✓ Creating Claude Code integration...", style=Styles.SUCCESS)

        # Initialize git repo
        import os
        import subprocess

        # Check if project is inside an existing git repo
        inside_existing_repo = False
        parent_root: Path | None = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=project_path,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                parent_root = Path(result.stdout.strip()).resolve()
                if parent_root != project_path.resolve():
                    inside_existing_repo = True
        except FileNotFoundError:
            pass  # git not installed — handled below

        try:
            subprocess.run(["git", "init"], cwd=project_path, check=True, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=project_path, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial project from osprey init"],
                cwd=project_path,
                check=True,
                capture_output=True,
                env={
                    **os.environ,
                    "GIT_AUTHOR_NAME": "osprey",
                    "GIT_AUTHOR_EMAIL": "osprey@init",
                    "GIT_COMMITTER_NAME": "osprey",
                    "GIT_COMMITTER_EMAIL": "osprey@init",
                },
            )
            console.print("  ✓ Initialized git repository", style=Styles.SUCCESS)
            if inside_existing_repo:
                console.print(
                    f"  !  Note: created a nested git repo inside {parent_root}.\n"
                    "     This is required for Claude Code project isolation (it uses\n"
                    "     the git root to discover .claude/ settings). The parent repo\n"
                    "     will treat this directory as opaque.",
                    style=Styles.WARNING,
                )
        except FileNotFoundError:
            console.print(
                "  !  git not found — project created but not initialized as a git repo.\n"
                "     Claude Code requires git. Run 'git init && git add . && git commit'"
                " manually.",
                style=Styles.WARNING,
            )
        except subprocess.CalledProcessError:
            console.print(
                "  !  git init succeeded but initial commit failed.\n"
                "     Run 'git add . && git commit' manually.",
                style=Styles.WARNING,
            )

        # Check if API keys were detected and .env was created
        from osprey.models.provider_registry import PROVIDER_API_KEYS

        api_key_names = {v for v in PROVIDER_API_KEYS.values() if v is not None}
        has_api_keys = any(key in detected_env for key in api_key_names)

        if has_api_keys:
            console.print("  ✓ Created .env with detected API keys", style=Styles.SUCCESS)

        console.print(f"\n✓ Project created successfully at: [bold]{project_path}[/bold]")

        # Show next steps (Claude Code focused)
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  1. {Messages.command(f'cd {project_name}')}")

        if has_api_keys:
            console.print("  2. # .env already configured with detected API keys")
            console.print(f"  3. {Messages.command('claude')}")
        else:
            console.print(f"  2. {Messages.command('cp .env.example .env')}")
            console.print("  3. # Edit .env with your API keys")
            console.print(f"  4. {Messages.command('claude')}")

        console.print(
            "\n[dim]Your project is a standalone git repo. Claude Code will use"
            " project-local settings only.[/dim]"
        )

        # Recommend installing the build-interview skill for production work
        console.print(
            "\n[success]Next:[/success] install the build-interview skill for production work:"
        )
        console.print("  [accent]uv run osprey skills install build-interview[/accent]")

    except ValueError as e:
        console.print(f"✗ Error: {e}", style=Styles.ERROR)
        raise click.Abort() from e
    except Exception as e:
        console.print(f"✗ Unexpected error: {e}", style=Styles.ERROR)
        import traceback

        console.print(traceback.format_exc(), style=Styles.DIM)
        raise click.Abort() from e


if __name__ == "__main__":
    init()
