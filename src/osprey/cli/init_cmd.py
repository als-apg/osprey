"""Project initialization command.

This module provides the 'osprey init' command which creates new
projects from templates with Claude Code integration (MCP servers,
hooks, and rules).

New workflow: init generates a BuildProfile YAML, which is then
consumed by 'osprey build' to assemble a project. Use --quick to
combine both steps.
"""

from pathlib import Path

import click

from .project_utils import _clear_claude_code_project_state
from .styles import Messages, Styles, console
from .templates.manager import TemplateManager


def _get_examples_dir() -> Path:
    """Return the path to bundled example profiles."""
    from importlib.resources import files

    return Path(str(files("osprey").joinpath("profiles/examples")))


def _list_examples() -> list[str]:
    """Return sorted list of bundled example profile names (without .yml)."""
    examples_dir = _get_examples_dir()
    if not examples_dir.exists():
        return []
    return sorted(p.stem for p in examples_dir.glob("*.yml"))


@click.command()
@click.argument("project_name")
@click.option(
    "--template",
    "-t",
    default="control_assistant",
    help="Application template to use (hello_world, control_assistant, lattice_design)",
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
@click.option(
    "--provider",
    type=click.Choice(
        ["anthropic", "openai", "google", "cborg", "ollama", "amsc", "als-apg"],
        case_sensitive=False,
    ),
    default=None,
    help="AI provider to configure (anthropic, openai, google, cborg, ollama, amsc, als-apg)",
)
@click.option(
    "--model",
    type=click.Choice(["haiku", "sonnet", "opus"], case_sensitive=False),
    default=None,
    help="Model tier to use (haiku, sonnet, opus)",
)
@click.option(
    "--channel-finder-mode",
    type=click.Choice(["in_context", "hierarchical", "middle_layer", "all"], case_sensitive=False),
    default=None,
    help="Channel finder pipeline mode (control_assistant template only)",
)
@click.option(
    "--example",
    "-e",
    default=None,
    metavar="NAME",
    help="Copy a bundled example profile to <project_name>.yml instead of creating a project",
)
@click.option(
    "--list-examples",
    is_flag=True,
    help="List available bundled example profiles and exit",
)
@click.option(
    "--interactive",
    is_flag=True,
    help="Run the interactive wizard to generate a BuildProfile YAML",
)
@click.option(
    "--quick",
    default=None,
    metavar="EXAMPLE",
    help="Copy EXAMPLE profile and immediately run 'osprey build' on it",
)
def init(
    project_name: str,
    template: str,
    output_dir: str,
    force: bool,
    provider: str,
    model: str,
    channel_finder_mode: str,
    example: str,
    list_examples: bool,
    interactive: bool,
    quick: str,
):
    """Create a new project.

    Creates a project configured for Claude Code with MCP servers,
    hooks, and rules.

    PROJECT_NAME: Name of your project (e.g., my-assistant, beamline-agent)

    New profile-based workflow:

    \b
      # Copy an example profile to customise
      $ osprey init my-project --example control-assistant

      # List available examples
      $ osprey init . --list-examples

      # Copy example and build immediately
      $ osprey init my-project --quick control-assistant

      # Run interactive wizard to generate a profile
      $ osprey init my-project --interactive

    Legacy direct-creation workflow:

    \b
      # Create project with default control_assistant template
      $ osprey init my-assistant

      # Create in specific location
      $ osprey init my-assistant --output-dir /projects

      # Force overwrite if directory exists
      $ osprey init my-assistant --force

      # Create with specific AI provider and model
      $ osprey init my-assistant --provider cborg --model anthropic/claude-haiku
    """
    # --- Flag: --list-examples ---
    if list_examples:
        examples = _list_examples()
        if not examples:
            console.print("No bundled example profiles found.", style=Styles.WARNING)
        else:
            console.print("[bold]Available example profiles:[/bold]")
            for name in examples:
                console.print(f"  • {name}")
            console.print(
                f"\nUse: [accent]osprey init <project> --example <name>[/accent] to copy one."
            )
        return

    # --- Flag: --example ---
    if example is not None:
        _copy_example_profile(project_name, example, output_dir, force)
        return

    # --- Flag: --quick ---
    if quick is not None:
        _quick_build(project_name, quick, output_dir, force)
        return

    # --- Flag: --interactive ---
    if interactive:
        _run_interactive_profile_wizard(project_name, output_dir)
        return

    # --- Legacy direct-creation workflow ---
    _create_project_direct(
        project_name=project_name,
        template=template,
        output_dir=output_dir,
        force=force,
        provider=provider,
        model=model,
        channel_finder_mode=channel_finder_mode,
    )


def _copy_example_profile(
    project_name: str, example: str, output_dir: str, force: bool
) -> None:
    """Copy a bundled example profile to <project_name>.yml."""
    examples_dir = _get_examples_dir()
    src = examples_dir / f"{example}.yml"
    if not src.exists():
        available = _list_examples()
        console.print(
            f"[error]✗[/error] Example '{example}' not found.\n"
            f"Available: {', '.join(available) if available else 'none'}",
            style=Styles.ERROR,
        )
        raise click.Abort()

    output_path = Path(output_dir).resolve()
    dst = output_path / f"{project_name}.yml"

    if dst.exists() and not force:
        console.print(
            f"[error]✗[/error] '{dst}' already exists. Use --force to overwrite.",
            style=Styles.ERROR,
        )
        raise click.Abort()

    import shutil

    shutil.copy(src, dst)
    console.print(f"[success]✓[/success] Copied example profile to [bold]{dst}[/bold]")
    console.print(
        f"\nEdit [accent]{dst.name}[/accent] then run:\n"
        f"  [accent]osprey build {project_name} {dst.name}[/accent]"
    )


def _quick_build(project_name: str, example: str, output_dir: str, force: bool) -> None:
    """Copy an example profile and immediately run osprey build."""
    import tempfile

    examples_dir = _get_examples_dir()
    src = examples_dir / f"{example}.yml"
    if not src.exists():
        available = _list_examples()
        console.print(
            f"[error]✗[/error] Example '{example}' not found.\n"
            f"Available: {', '.join(available) if available else 'none'}",
            style=Styles.ERROR,
        )
        raise click.Abort()

    output_path = Path(output_dir).resolve()

    # Write a temporary profile with the project name injected
    import yaml as _yaml

    raw = _yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    raw["name"] = project_name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", dir=output_path, delete=False, encoding="utf-8"
    ) as tmp:
        _yaml.dump(raw, tmp, default_flow_style=False, allow_unicode=True)
        tmp_path = Path(tmp.name)

    console.print(
        f"[success]✓[/success] Generated profile from example '{example}'"
    )
    console.print(f"  Building [accent]{project_name}[/accent] ...")

    try:
        from .build_cmd import build

        ctx = click.get_current_context()
        force_flag = ["--force"] if force else []
        ctx.invoke(
            build,
            project_name=project_name,
            profile=str(tmp_path),
            output_dir=str(output_path),
            force=force,
            stream=False,
            skip_lifecycle=False,
            skip_deps=False,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _run_interactive_profile_wizard(project_name: str, output_dir: str) -> None:
    """Run the interactive wizard and write a BuildProfile YAML."""
    try:
        from .init_wizard import run_interactive_profile_wizard as _wizard

        output_path = Path(output_dir).resolve()
        _wizard(project_name, output_path)
    except ImportError:
        console.print(
            "[error]✗[/error] Interactive mode requires the 'questionary' package.\n"
            f"Install with: [accent]uv add questionary[/accent]",
            style=Styles.ERROR,
        )
        raise click.Abort()


def _create_project_direct(
    project_name: str,
    template: str,
    output_dir: str,
    force: bool,
    provider: str | None,
    model: str | None,
    channel_finder_mode: str | None,
) -> None:
    """Legacy direct project creation path (original init behavior)."""
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

        context = {}
        if provider:
            context["default_provider"] = provider
        if model:
            context["default_model"] = model
        if channel_finder_mode:
            context["channel_finder_mode"] = channel_finder_mode

        project_path = manager.create_project(
            project_name=project_name,
            output_dir=output_path,
            template_name=template,
            registry_style="extend",
            context=context,
        )

        manifest_context = {
            "default_provider": context.get("default_provider", "anthropic"),
            "default_model": context.get("default_model", "haiku"),
        }
        if channel_finder_mode:
            manifest_context["channel_finder_mode"] = channel_finder_mode
        manager.generate_manifest(
            project_dir=project_path,
            project_name=project_name,
            template_name=template,
            registry_style="extend",
            context=manifest_context,
        )

        console.print("  ✓ Creating project configuration...", style=Styles.SUCCESS)
        console.print("  ✓ Creating Claude Code integration...", style=Styles.SUCCESS)

        # Initialize git repo
        import os
        import subprocess

        # Check if project is inside an existing git repo
        inside_existing_repo = False
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

        # Show provider/model configuration status
        if provider or model:
            if provider:
                console.print(
                    f"  ✓ Configured AI provider: [accent]{provider}[/accent]",
                    style=Styles.SUCCESS,
                )
            if model:
                console.print(
                    f"  ✓ Configured model: [accent]{model}[/accent]", style=Styles.SUCCESS
                )

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

        if provider:
            from osprey.cli.claude_code_resolver import CLAUDE_CODE_PROVIDERS

            prov_def = CLAUDE_CODE_PROVIDERS.get(provider, {})
            if prov_def:
                console.print(f"\n[bold]Shell setup for '{provider}':[/bold]")
                console.print(
                    f"  ✓ Required in ~/.zshrc:  "
                    f"[accent]export {prov_def['auth_secret_env']}=<your-key>[/accent]"
                )
                console.print("  ✗ Remove from ~/.zshrc:  any ANTHROPIC_* exports")
                console.print("  → Launch with:           [accent]osprey claude chat[/accent]")

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
