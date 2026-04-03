"""Migration command for OSPREY projects.

This module provides the 'osprey migrate' command which helps facilities:
- Detect when a project needs migration
- Retroactively create manifests for existing projects
- Perform three-way diffs between old vanilla, new vanilla, and facility customizations
- Generate merge guidance and AI-assisted prompts

Business logic lives in osprey.services.migration; this module handles
CLI presentation and user interaction.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import click

from osprey.services.migration import (
    detect_project_settings,
    generate_migration_directory,
    load_manifest,
    migrate_claude_code_config,
    perform_migration_analysis,
)
from osprey.services.migration.engine import MANIFEST_FILENAME

from .styles import console
from .templates.manager import TemplateManager


def _recreate_vanilla_with_version(
    manifest: dict[str, Any],
    output_dir: Path,
    use_temp_venv: bool = True,
) -> Path | None:
    """Recreate vanilla project using exact OSPREY version from manifest.

    Args:
        manifest: Project manifest with version and init_args
        output_dir: Directory where vanilla project should be created
        use_temp_venv: If True, create temp virtualenv with exact version

    Returns:
        Path to recreated vanilla project, or None if failed
    """
    version = manifest["creation"]["osprey_version"]
    init_args = manifest["init_args"]
    project_name = init_args["project_name"]

    if use_temp_venv:
        # Create isolated virtualenv with exact version
        venv_dir = Path(tempfile.mkdtemp(prefix="osprey-migrate-"))
        console.print(f"  [dim]Creating temp environment at {venv_dir}...[/dim]")

        try:
            # Create virtualenv
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=True,
                capture_output=True,
            )

            # Get pip path
            if sys.platform == "win32":
                pip = venv_dir / "Scripts" / "pip"
                osprey_bin = venv_dir / "Scripts" / "osprey"
            else:
                pip = venv_dir / "bin" / "pip"
                osprey_bin = venv_dir / "bin" / "osprey"

            # Install exact version
            console.print(f"  [dim]Installing osprey-framework=={version}...[/dim]")
            result = subprocess.run(
                [str(pip), "install", f"osprey-framework=={version}"],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                console.print(f"[warning]Could not install osprey-framework=={version}[/warning]")
                console.print(f"[dim]{result.stderr}[/dim]")
                return None

            # Build init command
            cmd = [str(osprey_bin), "init", project_name]

            if init_args.get("template"):
                cmd.extend(["--template", init_args["template"]])

            if init_args.get("registry_style") and init_args["registry_style"] != "extend":
                cmd.extend(["--registry-style", init_args["registry_style"]])

            if init_args.get("provider"):
                cmd.extend(["--provider", init_args["provider"]])

            if init_args.get("model"):
                cmd.extend(["--model", init_args["model"]])

            # Run init in output directory
            console.print(f"  [dim]Running: {' '.join(cmd)}[/dim]")
            result = subprocess.run(
                cmd,
                cwd=output_dir,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                console.print("[warning]Failed to recreate vanilla project[/warning]")
                console.print(f"[dim]{result.stderr}[/dim]")
                return None

            return output_dir / project_name

        except Exception as e:
            console.print(f"[warning]Error creating vanilla project: {e}[/warning]")
            return None
        finally:
            # Clean up temp venv (but not the output project)
            try:
                shutil.rmtree(venv_dir)
            except Exception as e:
                console.print(f"[dim]Warning: Could not remove temp venv {venv_dir}: {e}[/dim]")
    else:
        # Use current OSPREY version (approximate)
        manager = TemplateManager()

        # Build context from init_args
        context = {}
        if init_args.get("provider"):
            context["default_provider"] = init_args["provider"]
        if init_args.get("model"):
            context["default_model"] = init_args["model"]
        if init_args.get("channel_finder_mode"):
            context["channel_finder_mode"] = init_args["channel_finder_mode"]
        if init_args.get("code_generator"):
            context["code_generator"] = init_args["code_generator"]

        try:
            project_path = manager.create_project(
                project_name=project_name,
                output_dir=output_dir,
                template_name=init_args.get("template", "control_assistant"),
                registry_style=init_args.get("registry_style", "extend"),
                context=context if context else None,
            )
            return project_path
        except Exception as e:
            console.print(f"[warning]Error creating vanilla project: {e}[/warning]")
            return None


@click.group()
def migrate():
    """Migrate OSPREY projects between versions.

    The migrate command helps facilities upgrade their OSPREY projects
    while preserving customizations. It uses three-way diffs to identify
    what changed in both the template and your project.

    \b
    Subcommands:
      init     Create manifest for existing project (retroactive)
      check    Check if migration is needed
      run      Perform migration analysis and generate prompts
      config   Migrate claude_code config from legacy to new format

    \b
    Examples:
      # Check if project needs migration
      $ osprey migrate check

      # Create manifest for existing project
      $ osprey migrate init

      # Run migration (dry-run by default)
      $ osprey migrate run

      # Run migration and apply safe changes
      $ osprey migrate run --apply
    """
    pass


@migrate.command("init")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Project directory (default: current directory)",
)
@click.option(
    "--version",
    "-v",
    "osprey_version",
    help="OSPREY version used to create project (will prompt if not provided)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing manifest",
)
def migrate_init(project: Path, osprey_version: str | None, force: bool):
    """Create manifest for existing project.

    For projects created before manifest support, this command
    detects settings from existing files and creates a manifest
    to enable future migrations.

    \b
    Example:
      $ cd my-project
      $ osprey migrate init

      # With explicit version
      $ osprey migrate init --version 0.10.2
    """
    console.print("[bold]OSPREY Migration: Initialize Manifest[/bold]\n")

    # Check for existing manifest
    manifest_path = project / MANIFEST_FILENAME
    if manifest_path.exists() and not force:
        console.print(f"[warning]Manifest already exists at {manifest_path}[/warning]")
        console.print("Use --force to overwrite.")
        raise click.Abort()

    # Detect project settings
    console.print("1. Detecting project configuration...")
    settings = detect_project_settings(project)

    # Display any warnings from detection
    for warning in settings.get("warnings", []):
        console.print(f"[dim]Warning: {warning}[/dim]")

    if not settings.get("package_name"):
        console.print("[error]Could not detect project structure[/error]")
        console.print("Make sure you're in an OSPREY project directory.")
        raise click.Abort()

    # Show detected settings
    console.print(f"   [success]✓[/success] Package: {settings.get('package_name')}")

    if settings.get("template"):
        confidence = settings.get("confidence", {}).get("template", "unknown")
        console.print(
            f"   [success]✓[/success] Template: {settings['template']} (confidence: {confidence})"
        )

    if settings.get("registry_style"):
        confidence = settings.get("confidence", {}).get("registry_style", "unknown")
        console.print(
            f"   [success]✓[/success] Registry style: {settings['registry_style']} (confidence: {confidence})"
        )

    if settings.get("provider"):
        console.print(f"   [success]✓[/success] Provider: {settings['provider']}")

    if settings.get("model"):
        console.print(f"   [success]✓[/success] Model: {settings['model']}")

    if settings.get("code_generator"):
        console.print(f"   [success]✓[/success] Code generator: {settings['code_generator']}")

    # Get or prompt for OSPREY version
    console.print("\n2. OSPREY version...")

    if osprey_version:
        version = osprey_version
    elif settings.get("estimated_osprey_version"):
        version = settings["estimated_osprey_version"]
        confidence = settings.get("confidence", {}).get("osprey_version", "unknown")
        console.print(
            f"   [dim]Detected from pyproject.toml: {version} (confidence: {confidence})[/dim]"
        )
        confirm = click.prompt(
            "   Confirm version (or enter correct version)",
            default=version,
        )
        version = confirm
    else:
        console.print("   [warning]Could not detect OSPREY version[/warning]")
        version = click.prompt(
            "   Enter OSPREY version used to create this project",
            default="0.10.0",
        )

    # Build project name from package name
    package_name = settings["package_name"]
    project_name = package_name.replace("_", "-")

    # Build context
    context: dict[str, Any] = {}
    if settings.get("provider"):
        context["default_provider"] = settings["provider"]
    if settings.get("model"):
        context["default_model"] = settings["model"]
    if settings.get("channel_finder_mode"):
        context["channel_finder_mode"] = settings["channel_finder_mode"]
    if settings.get("code_generator"):
        context["code_generator"] = settings["code_generator"]

    # Generate manifest
    console.print("\n3. Creating manifest...")

    manager = TemplateManager()

    # Override framework version for retroactive manifest
    from .templates import manifest as manifest_mod

    original_get_version = manifest_mod.get_framework_version
    manifest_mod.get_framework_version = lambda: version  # type: ignore[assignment]

    try:
        manifest = manager.generate_manifest(
            project_dir=project,
            project_name=project_name,
            template_name=settings.get("template", "minimal"),
            registry_style=settings.get("registry_style", "extend"),
            context=context,
        )
    finally:
        manifest_mod.get_framework_version = original_get_version  # type: ignore[assignment]

    console.print(f"   [success]✓[/success] Written {MANIFEST_FILENAME}")
    console.print(
        f"   [success]✓[/success] Calculated checksums for {len(manifest['file_checksums'])} files"
    )

    console.print("\n[success]Manifest created successfully![/success]")
    console.print(f"\nReproducible command: [accent]{manifest['reproducible_command']}[/accent]")


@migrate.command("check")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Project directory (default: current directory)",
)
def migrate_check(project: Path):
    """Check if project needs migration.

    Compares the project's manifest version against the currently
    installed OSPREY version to determine if migration is needed.

    \b
    Example:
      $ osprey migrate check
    """
    console.print("[bold]OSPREY Migration: Version Check[/bold]\n")

    # Load manifest
    manifest = load_manifest(project)

    if not manifest:
        console.print("[warning]No manifest found[/warning]")
        console.print("Run 'osprey migrate init' to create one for this project.")
        return

    # Get versions
    project_version = manifest["creation"]["osprey_version"]
    from .templates.manifest import get_framework_version

    current_version = get_framework_version()

    console.print(f"Project OSPREY version: [accent]{project_version}[/accent]")
    console.print(f"Installed OSPREY version: [accent]{current_version}[/accent]")

    # Simple version comparison (could be more sophisticated)
    if project_version == current_version:
        console.print("\n[success]Project is up to date![/success]")
    else:
        console.print(
            f"\n[warning]Migration may be needed: {project_version} -> {current_version}[/warning]"
        )
        console.print("\nRun 'osprey migrate run' to analyze changes and generate merge guidance.")


@migrate.command("run")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Project directory (default: current directory)",
)
@click.option(
    "--dry-run/--apply",
    default=True,
    help="Dry run (default) or apply safe changes",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for migration files (default: project/_migration)",
)
@click.option(
    "--use-current-version",
    is_flag=True,
    help="Use current OSPREY for old vanilla (skip exact version recreation)",
)
def migrate_run(project: Path, dry_run: bool, output: Path | None, use_current_version: bool):
    """Run migration analysis and generate merge guidance.

    This command:
    1. Recreates the old vanilla project (from manifest version)
    2. Creates new vanilla project (current OSPREY version)
    3. Performs three-way diff analysis
    4. Generates merge prompts for files needing manual attention

    \b
    Example:
      # Analyze changes (dry run)
      $ osprey migrate run

      # Apply safe changes automatically
      $ osprey migrate run --apply

      # Skip exact version recreation (faster but less accurate)
      $ osprey migrate run --use-current-version
    """
    console.print("[bold]OSPREY Migration[/bold]\n")

    # Load manifest
    manifest = load_manifest(project)

    if not manifest:
        console.print("[error]No manifest found[/error]")
        console.print("Run 'osprey migrate init' first to create a manifest.")
        raise click.Abort()

    old_version = manifest["creation"]["osprey_version"]
    from .templates.manifest import get_framework_version

    new_version = get_framework_version()

    console.print(f"Migration: [accent]{old_version}[/accent] -> [accent]{new_version}[/accent]\n")

    # Create temp directory for vanilla projects
    temp_dir = Path(tempfile.mkdtemp(prefix="osprey-migrate-"))

    try:
        # Step 1: Recreate old vanilla
        console.print("1. Recreating vanilla projects...")

        old_vanilla_dir = None
        if not use_current_version:
            console.print(f"   [dim]Creating old vanilla (OSPREY {old_version})...[/dim]")
            old_vanilla_dir = _recreate_vanilla_with_version(
                manifest,
                temp_dir / "old",
                use_temp_venv=True,
            )

            if old_vanilla_dir:
                console.print(f"   [success]✓[/success] Old vanilla: {old_vanilla_dir}")
            else:
                console.print(
                    "   [warning]Could not create old vanilla with exact version[/warning]"
                )
                console.print("   [dim]Continuing without old vanilla (less accurate)...[/dim]")

        # Step 2: Create new vanilla
        console.print(f"   [dim]Creating new vanilla (OSPREY {new_version})...[/dim]")
        new_vanilla_dir = _recreate_vanilla_with_version(
            manifest,
            temp_dir / "new",
            use_temp_venv=False,  # Use current version
        )

        if not new_vanilla_dir:
            console.print("[error]Failed to create new vanilla project[/error]")
            raise click.Abort()

        console.print(f"   [success]✓[/success] New vanilla: {new_vanilla_dir}")

        # Step 3: Perform analysis
        console.print("\n2. Analyzing file changes...")
        analysis = perform_migration_analysis(project, old_vanilla_dir, new_vanilla_dir)

        # Display summary
        console.print("\n3. File Classification\n")

        if analysis["auto_copy"]:
            console.print(
                f"   [bold]AUTO-COPY[/bold] ({len(analysis['auto_copy'])} files)"
                " - Template changed, you didn't"
            )
            for file_info in analysis["auto_copy"][:5]:
                console.print(f"     - {file_info['path']}")
            if len(analysis["auto_copy"]) > 5:
                console.print(f"     ... and {len(analysis['auto_copy']) - 5} more")
            console.print()

        if analysis["preserve"]:
            console.print(
                f"   [bold]PRESERVE[/bold] ({len(analysis['preserve'])} files)"
                " - You modified, template unchanged"
            )
            for file_info in analysis["preserve"][:5]:
                console.print(f"     - {file_info['path']}")
            if len(analysis["preserve"]) > 5:
                console.print(f"     ... and {len(analysis['preserve']) - 5} more")
            console.print()

        if analysis["merge"]:
            console.print(
                f"   [bold yellow]MERGE REQUIRED[/bold yellow] ({len(analysis['merge'])} files)"
                " - Both changed"
            )
            for file_info in analysis["merge"]:
                console.print(f"     - {file_info['path']}")
            console.print()

        if analysis["new"]:
            console.print(
                f"   [bold green]NEW FILES[/bold green] ({len(analysis['new'])} files)"
                " - Added in new template"
            )
            for file_info in analysis["new"][:5]:
                console.print(f"     - {file_info['path']}")
            if len(analysis["new"]) > 5:
                console.print(f"     ... and {len(analysis['new']) - 5} more")
            console.print()

        # Step 4: Generate migration directory
        if dry_run:
            console.print("\n4. Generating migration guidance (dry run)...")
        else:
            console.print("\n4. Applying migration...")

        output_dir = output or project
        migration_dir = generate_migration_directory(
            output_dir,
            analysis,
            project,
            old_vanilla_dir,
            new_vanilla_dir,
            old_version,
            new_version,
        )

        console.print(f"   [success]✓[/success] Created {migration_dir}")

        # Apply changes if not dry run
        if not dry_run:
            # Auto-copy files
            for file_info in analysis["auto_copy"]:
                src = new_vanilla_dir / file_info["path"]
                dst = project / file_info["path"]
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            console.print(f"   [success]✓[/success] Auto-copied {len(analysis['auto_copy'])} files")

            # Copy new files
            for file_info in analysis["new"]:
                src = new_vanilla_dir / file_info["path"]
                dst = project / file_info["path"]
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            console.print(f"   [success]✓[/success] Added {len(analysis['new'])} new files")

        # Summary
        console.print("\n[bold]Migration Summary[/bold]")
        console.print(f"  Auto-copy: {len(analysis['auto_copy'])} files")
        console.print(f"  Preserve: {len(analysis['preserve'])} files")
        console.print(f"  Merge required: {len(analysis['merge'])} files")
        console.print(f"  New files: {len(analysis['new'])} files")

        if analysis["merge"]:
            console.print("\n[bold]Next Steps[/bold]")
            console.print(f"  1. Review merge prompts in: {migration_dir / 'merge_required'}")
            console.print("  2. Merge your customizations with template updates")
            console.print("  3. Run 'osprey health' to verify configuration")
            console.print(f"  4. Delete {migration_dir} when complete")

    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            console.print(f"[dim]Warning: Could not remove temp directory {temp_dir}: {e}[/dim]")


@migrate.command("config")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path),
    default=Path.cwd(),
    help="Project directory",
)
@click.option("--dry-run/--apply", default=True, help="Preview vs apply changes")
def migrate_config(project: Path, dry_run: bool):
    """Migrate claude_code config from legacy to new format.

    Converts legacy keys to the new extensibility format:

    \b
      disable_servers: [x]  ->  servers: {x: {enabled: false}}
      extra_servers: {n: s} ->  servers: {n: s}
      disable_agents: [x]  ->  agents: {x: {enabled: false}}
    """
    from ruamel.yaml import YAML

    config_path = project / "config.yml"
    if not config_path.exists():
        console.print(f"[error]No config.yml found at {project}[/error]")
        raise click.Abort()

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 200

    with open(config_path, encoding="utf-8") as f:
        data = yaml.load(f)

    if data is None:
        console.print("[warning]config.yml is empty[/warning]")
        raise click.Abort()

    claude_code = data.get("claude_code")
    if claude_code is None:
        console.print("[info]No claude_code section in config.yml — nothing to migrate.[/info]")
        return

    legacy_keys = {"disable_servers", "extra_servers", "disable_agents"}
    found = legacy_keys & set(claude_code.keys())
    if not found:
        console.print(
            "[info]No legacy keys found in claude_code section — already using new format.[/info]"
        )
        return

    console.print("[bold]OSPREY Config Migration[/bold]\n")
    console.print(f"Found legacy keys: {', '.join(sorted(found))}\n")

    servers, agents, changes = migrate_claude_code_config(claude_code)

    for change in changes:
        console.print(f"  {change}")

    if dry_run:
        console.print("\n[dim]Dry run — no changes written. Use --apply to write.[/dim]")
        return

    # Apply changes: remove legacy keys, insert new ones
    for key in ("disable_servers", "extra_servers", "disable_agents"):
        if key in claude_code:
            del claude_code[key]

    if servers:
        claude_code["servers"] = servers
    if agents:
        claude_code["agents"] = agents

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    console.print(f"\n[success]Migration applied to {config_path}[/success]")
