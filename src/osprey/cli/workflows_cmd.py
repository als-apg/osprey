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

    Returns:
        Path to the workflows directory in the installed package,
        or None if not found.
    """
    try:
        # Python 3.11+ compatible way to access package resources
        # This works for both installed packages and development mode
        from importlib.resources import files

        workflows_ref = files("osprey").joinpath("workflows")

        # Convert to Path - handle both Traversable and Path objects
        if hasattr(workflows_ref, "__fspath__"):
            # It's a real Path
            return Path(workflows_ref)
        else:
            # It's a Traversable, convert via str
            return Path(str(workflows_ref))
    except Exception as e:
        console.print(f"{Messages.error('Error locating workflow files:')} {e}", style=Styles.ERROR)
        return None


@click.group(name="workflows", invoke_without_command=True)
@click.pass_context
def workflows(ctx):
    """Manage AI workflow documentation files.

    Export workflow files to your project for easy access by AI coding assistants.
    These markdown files guide assistants through common development tasks.

    Examples:

    \b
      # Export to current directory (creates osprey-workflows/)
      osprey workflows export

      # Export to custom location
      osprey workflows export --output ~/my-workflows

      # List available workflows
      osprey workflows list
    """
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

    # Get all markdown files except README
    try:
        workflows_list = sorted(
            [f for f in source.iterdir() if f.suffix == ".md" and f.name != "README.md"]
        )
    except Exception as e:
        console.print(Messages.error(f"Error reading workflow directory: {e}"))
        return

    if not workflows_list:
        console.print(f"[{Styles.WARNING}]No workflow files found[/{Styles.WARNING}]")
        return

    # Display each workflow with its title
    for wf in workflows_list:
        try:
            # Read first line (title) from each workflow
            with open(wf, encoding="utf-8") as f:
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

                # Display with title or just filename
                if title:
                    console.print(f"  [{Styles.SUCCESS}]•[/{Styles.SUCCESS}] {wf.name:45} {title}")
                else:
                    console.print(f"  [{Styles.SUCCESS}]•[/{Styles.SUCCESS}] {wf.name}")

        except Exception:
            # Fallback: just show filename
            console.print(
                f"  [{Styles.SUCCESS}]•[/{Styles.SUCCESS}] {wf.name} [{Styles.DIM}](read error)[/{Styles.DIM}]"
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

    # Copy workflow files
    console.print(f"\n{Messages.header('Exporting workflows to:')} {target}\n")

    copied = 0
    errors = []

    try:
        for wf_file in source.iterdir():
            if wf_file.suffix == ".md":
                try:
                    dest_file = target / wf_file.name
                    shutil.copy2(wf_file, dest_file)
                    console.print(f"  [{Styles.SUCCESS}]✓[/{Styles.SUCCESS}] {wf_file.name}")
                    copied += 1
                except Exception as e:
                    errors.append((wf_file.name, str(e)))
                    console.print(f"  [{Styles.ERROR}]✗[/{Styles.ERROR}] {wf_file.name} - {e}")
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
