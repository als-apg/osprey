"""Workflow management commands for AI-assisted development.

This module provides commands for exporting and working with Osprey's
AI workflow documentation files that guide AI coding assistants through
common development tasks.
"""

import shutil
from pathlib import Path

import click

from .styles import Messages, Styles, console


def get_workflows_source_path() -> Path | None:
    """Get the path to bundled workflow files using importlib.resources.

    DEPRECATED: This now points to assist/tasks for backward compatibility.
    New code should use osprey.cli.tasks_cmd functions instead.

    Returns:
        Path to the assist/tasks directory in the installed package,
        or None if not found.
    """
    try:
        # Python 3.11+ compatible way to access package resources
        # This works for both installed packages and development mode
        from importlib.resources import files

        # Point to assist/tasks instead of deprecated workflows directory
        tasks_ref = files("osprey").joinpath("assist", "tasks")

        # Convert to Path - handle both Traversable and Path objects
        if hasattr(tasks_ref, "__fspath__"):
            # It's a real Path
            return Path(tasks_ref)
        else:
            # It's a Traversable, convert via str
            return Path(str(tasks_ref))
    except Exception as e:
        console.print(f"{Messages.error('Error locating workflow files:')} {e}", style=Styles.ERROR)
        return None


@click.group(name="workflows", invoke_without_command=True)
@click.pass_context
def workflows(ctx):
    """Manage AI workflow documentation files.

    DEPRECATED: Use 'osprey tasks' and 'osprey claude' instead.

    The workflows have been consolidated into the assist system:
      - osprey tasks list         # List all available tasks
      - osprey tasks show X       # Show task details
      - osprey claude install X   # Install task as Claude Code skill

    This command is kept for backward compatibility but will be removed
    in a future version.

    Examples:

    \b
      # Export to current directory (creates osprey-workflows/)
      osprey workflows export

      # Export to custom location
      osprey workflows export --output ~/my-workflows

      # List available workflows
      osprey workflows list
    """
    # Show deprecation warning
    console.print(
        "\n[yellow]⚠ DEPRECATED:[/yellow] 'osprey workflows' is deprecated. "
        "Use [cyan]osprey tasks[/cyan] and [cyan]osprey claude[/cyan] instead.\n"
    )
    console.print("  [dim]osprey tasks list[/dim]         - List all available tasks")
    console.print("  [dim]osprey tasks show X[/dim]       - Show task details")
    console.print("  [dim]osprey claude install X[/dim]   - Install task as Claude Code skill\n")

    if ctx.invoked_subcommand is None:
        # Default action: export to current directory
        ctx.invoke(export)


@workflows.command()
def list():
    """List all available workflow files.

    Shows workflow files bundled with the installed Osprey package.
    These can be exported using 'osprey workflows export'.
    """
    source = get_workflows_source_path()
    if not source or not source.exists():
        console.print(Messages.error("Workflow files not found in package"))
        console.print(
            f"{Styles.DIM}This might indicate a packaging issue or development mode setup[/{Styles.DIM}]"
        )
        return

    console.print(f"\n{Messages.header('Available AI Workflow Files:')}\n")

    # Get all task directories that have instructions.md
    try:
        workflows_list = sorted(
            [d for d in source.iterdir() if d.is_dir() and (d / "instructions.md").exists()]
        )
    except Exception as e:
        console.print(Messages.error(f"Error reading workflow directory: {e}"))
        return

    if not workflows_list:
        console.print(f"[{Styles.WARNING}]No workflow files found[/{Styles.WARNING}]")
        return

    # Display each workflow with its title
    for task_dir in workflows_list:
        instructions_file = task_dir / "instructions.md"
        try:
            # Read first line (title) from instructions.md
            with open(instructions_file, encoding="utf-8") as f:
                lines = f.readlines()
                title = None
                in_frontmatter = False

                for line in lines:
                    stripped = line.strip()
                    # Handle YAML frontmatter
                    if stripped == "---":
                        in_frontmatter = not in_frontmatter
                        continue
                    # Find first heading outside frontmatter
                    if not in_frontmatter and line.startswith("#"):
                        title = line.lstrip("#").strip()
                        break

                # Display with title or just task name
                display_name = f"{task_dir.name}.md"
                if title:
                    console.print(
                        f"  [{Styles.SUCCESS}]•[/{Styles.SUCCESS}] {display_name:45} {title}"
                    )
                else:
                    console.print(f"  [{Styles.SUCCESS}]•[/{Styles.SUCCESS}] {display_name}")

        except Exception:
            # Fallback: just show task name
            console.print(
                f"  [{Styles.SUCCESS}]•[/{Styles.SUCCESS}] {task_dir.name}.md [{Styles.DIM}](read error)[/{Styles.DIM}]"
            )

    console.print(f"\n[{Styles.DIM}]Total: {len(workflows_list)} workflows[/{Styles.DIM}]")
    console.print(
        f"\n{Messages.info('Export these files:')} {Messages.command('osprey workflows export')}\n"
    )


@workflows.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    default="./osprey-workflows",
    help="Target directory for exported workflows",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files without prompting")
def export(output, force):
    """Export workflow files to a local directory.

    Copies all workflow markdown files from the installed Osprey package
    to a local directory, making them accessible for @-mentions in AI
    coding assistants (Cursor, Claude Code, etc.).

    The exported files include:
    - testing-workflow.md: Test type selection guide
    - commit-organization.md: Atomic commit organization
    - pre-merge-cleanup.md: Pre-commit cleanup checks
    - docstrings.md: Docstring generation guide
    - comments.md: Code commenting strategy
    - update-documentation.md: Documentation update guide
    - ai-code-review.md: AI code review checklist
    - channel-finder-pipeline-selection.md: Pipeline selection
    - channel-finder-database-builder.md: Database building
    - release-workflow.md: Release process guide
    - And more...

    Examples:

    \b
      # Export to default location (./osprey-workflows/)
      osprey workflows export

      # Export to custom location
      osprey workflows export --output ~/Documents/workflows

      # Overwrite existing files
      osprey workflows export --force
    """
    source = get_workflows_source_path()
    if not source or not source.exists():
        console.print(Messages.error("Workflow files not found in installed package"))
        console.print(
            f"[{Styles.DIM}]This might indicate a packaging issue. "
            f"Try reinstalling: pip install --force-reinstall osprey-framework[/{Styles.DIM}]"
        )
        return

    target = Path(output).resolve()

    # Check if target exists and handle accordingly
    if target.exists() and not force:
        # Check if directory is non-empty
        if any(target.iterdir()):
            console.print(f"\n{Messages.warning('Directory already exists:')} {target}")

            if not click.confirm("Overwrite existing files?", default=False):
                console.print(f"[{Styles.DIM}]Export cancelled[/{Styles.DIM}]")
                return

    # Create target directory
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        console.print(Messages.error(f"Failed to create directory: {e}"))
        return

    # Copy workflow files (from assist/tasks/*/instructions.md to {task-name}.md)
    console.print(f"\n{Messages.header('Exporting workflows to:')} {target}\n")

    copied = 0
    errors = []

    try:
        for task_dir in source.iterdir():
            if task_dir.is_dir():
                instructions_file = task_dir / "instructions.md"
                if instructions_file.exists():
                    try:
                        # Export as {task-name}.md for backward compatibility
                        dest_file = target / f"{task_dir.name}.md"
                        shutil.copy2(instructions_file, dest_file)
                        console.print(
                            f"  [{Styles.SUCCESS}]✓[/{Styles.SUCCESS}] {task_dir.name}.md"
                        )
                        copied += 1
                    except Exception as e:
                        errors.append((f"{task_dir.name}.md", str(e)))
                        console.print(
                            f"  [{Styles.ERROR}]✗[/{Styles.ERROR}] {task_dir.name}.md - {e}"
                        )
    except Exception as e:
        console.print(Messages.error(f"Error during export: {e}"))
        return

    # Summary
    console.print(f"\n{Messages.success(f'✓ Exported {copied} workflow files')}")

    if errors:
        console.print(f"\n{Messages.warning(f'Failed to copy {len(errors)} files:')}")
        for filename, error in errors:
            console.print(f"  - {filename}: {error}")

    # Usage instructions
    console.print(f"\n{Messages.header('Usage in AI coding assistants:')}")

    # Use relative path if under current directory, otherwise absolute
    try:
        rel_path = target.relative_to(Path.cwd())
        display_path = rel_path
    except ValueError:
        display_path = target

    console.print(
        f"  {Messages.command(f'@{display_path}/testing-workflow.md What type of test should I write?')}"
    )
    console.print(
        f"  {Messages.command(f'@{display_path}/pre-merge-cleanup.md Scan my uncommitted changes')}"
    )

    console.print(
        f"\n[{Styles.DIM}]Learn more: https://als-apg.github.io/osprey/contributing/03_ai-assisted-development.html[/{Styles.DIM}]\n"
    )
