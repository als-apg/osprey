"""Migration command for OSPREY projects.

This module provides the 'osprey migrate' command which helps facilities:
- Detect when a project needs migration
- Retroactively create manifests for existing projects
- Perform three-way diffs between old vanilla, new vanilla, and facility customizations
- Generate merge guidance and AI-assisted prompts
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import click

from .styles import console
from .templates import MANIFEST_FILENAME, TemplateManager


class FileCategory(Enum):
    """Classification categories for migration files."""

    DATA = "data"  # User data directories - always preserve
    AUTO_COPY = "auto_copy"  # Template changed, facility didn't - copy from new
    PRESERVE = "preserve"  # Facility modified, template unchanged - keep facility
    MERGE = "merge"  # Both changed - needs manual/AI merge
    NEW = "new"  # Only exists in new template - copy from new
    REMOVED = "removed"  # Only exists in old template - may need cleanup


def _load_manifest(project_dir: Path) -> dict[str, Any] | None:
    """Load manifest from project directory.

    Args:
        project_dir: Path to the project root

    Returns:
        Manifest dict if found, None otherwise
    """
    manifest_path = project_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"[warning]Warning: Could not read manifest: {e}[/warning]")
        return None


def _detect_project_settings(project_dir: Path) -> dict[str, Any]:
    """Detect settings from an existing project without a manifest.

    Examines config.yml, registry.py, pyproject.toml etc. to infer
    the original init settings.

    Args:
        project_dir: Path to the project root

    Returns:
        Dictionary of detected settings
    """
    settings: dict[str, Any] = {
        "detected": True,
        "confidence": {},
    }

    # Try to detect from config.yml
    config_path = project_dir / "config.yml"
    if config_path.exists():
        try:
            import yaml

            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if config:
                # Detect provider and model
                if "llm" in config:
                    settings["provider"] = config["llm"].get("default_provider")
                    settings["model"] = config["llm"].get("default_model")
                    settings["confidence"]["provider"] = "high"
                    settings["confidence"]["model"] = "high"

                # Detect channel finder settings
                if "channel_finder" in config:
                    cf = config["channel_finder"]
                    settings["channel_finder_mode"] = cf.get("default_pipeline")
                    settings["confidence"]["channel_finder_mode"] = "medium"

                # Detect template from config structure
                if "channel_finder" in config:
                    settings["template"] = "control_assistant"
                    settings["confidence"]["template"] = "high"
                elif "capabilities" in config:
                    settings["template"] = "minimal"
                    settings["confidence"]["template"] = "medium"
        except Exception as e:
            console.print(f"[dim]Warning: Could not parse config.yml: {e}[/dim]")

    # Try to detect from pyproject.toml
    pyproject_path = project_dir / "pyproject.toml"
    if pyproject_path.exists():
        try:
            import tomllib

            with open(pyproject_path, "rb") as f:
                pyproject = tomllib.load(f)

            # Look for osprey-framework dependency
            deps = pyproject.get("project", {}).get("dependencies", [])
            for dep in deps:
                if "osprey-framework" in dep:
                    # Try to extract version constraint
                    if ">=" in dep:
                        version = dep.split(">=")[1].split(",")[0].strip()
                        settings["estimated_osprey_version"] = version
                        settings["confidence"]["osprey_version"] = "medium"
                    elif "==" in dep:
                        version = dep.split("==")[1].strip()
                        settings["estimated_osprey_version"] = version
                        settings["confidence"]["osprey_version"] = "high"
                    break
        except Exception as e:
            console.print(f"[dim]Warning: Could not parse pyproject.toml: {e}[/dim]")

    # Try to detect registry style from registry.py
    src_dir = project_dir / "src"
    if src_dir.exists():
        for pkg_dir in src_dir.iterdir():
            if pkg_dir.is_dir() and not pkg_dir.name.startswith("_"):
                registry_path = pkg_dir / "registry.py"
                if registry_path.exists():
                    try:
                        content = registry_path.read_text(encoding="utf-8")
                        if "OspreyFrameworkRegistry" in content and "extend" in content.lower():
                            settings["registry_style"] = "extend"
                            settings["confidence"]["registry_style"] = "high"
                        elif "explicit" in content.lower() or (
                            "CapabilityRegistration" in content
                            and content.count("CapabilityRegistration") > 5
                        ):
                            settings["registry_style"] = "standalone"
                            settings["confidence"]["registry_style"] = "medium"
                        else:
                            settings["registry_style"] = "extend"
                            settings["confidence"]["registry_style"] = "low"

                        # Detect package name
                        settings["package_name"] = pkg_dir.name
                    except Exception as e:
                        console.print(
                            f"[dim]Warning: Could not analyze registry file '{registry_path}': {e}[/dim]"
                        )
                break

    # Try to detect code generator from config file presence
    if (project_dir / "claude_generator_config.yml").exists():
        settings["code_generator"] = "claude_code"
        settings["confidence"]["code_generator"] = "high"
    elif (project_dir / "basic_generator_config.yml").exists():
        settings["code_generator"] = "basic"
        settings["confidence"]["code_generator"] = "high"

    return settings


def _calculate_file_hash(file_path: Path) -> str | None:
    """Calculate SHA256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hex-encoded SHA256 hash, or None if file can't be read
    """
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except OSError:
        return None


def _read_file_content(file_path: Path) -> str | None:
    """Read file content, returning None if not readable.

    Args:
        file_path: Path to the file

    Returns:
        File content as string, or None if not readable
    """
    try:
        return file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _classify_file(
    rel_path: str,
    facility_hash: str | None,
    old_vanilla_hash: str | None,
    new_vanilla_hash: str | None,
) -> FileCategory:
    """Classify a file for migration action.

    Args:
        rel_path: Relative path of the file
        facility_hash: Hash of facility's version (None if doesn't exist)
        old_vanilla_hash: Hash in old vanilla project (None if doesn't exist)
        new_vanilla_hash: Hash in new vanilla project (None if doesn't exist)

    Returns:
        FileCategory indicating what action to take
    """
    # Data directories are always preserved
    if rel_path.startswith(("data/", "_agent_data/")):
        return FileCategory.DATA

    # File only in new template
    if new_vanilla_hash and not old_vanilla_hash and not facility_hash:
        return FileCategory.NEW

    # File only in old template (removed in new)
    if old_vanilla_hash and not new_vanilla_hash:
        return FileCategory.REMOVED

    # File exists in facility but not in templates - preserve
    if facility_hash and not old_vanilla_hash and not new_vanilla_hash:
        return FileCategory.PRESERVE

    # Compare hashes for three-way diff
    if facility_hash and old_vanilla_hash and new_vanilla_hash:
        facility_unchanged = facility_hash == old_vanilla_hash
        template_unchanged = old_vanilla_hash == new_vanilla_hash

        if facility_unchanged and not template_unchanged:
            # Template changed, facility didn't - auto-copy new template
            return FileCategory.AUTO_COPY
        elif not facility_unchanged and template_unchanged:
            # Facility changed, template didn't - preserve facility
            return FileCategory.PRESERVE
        elif not facility_unchanged and not template_unchanged:
            # Both changed - needs merge
            return FileCategory.MERGE
        else:
            # Neither changed - preserve (no action needed)
            return FileCategory.PRESERVE

    # Default to preserve for safety
    return FileCategory.PRESERVE


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

            if init_args.get("template") and init_args["template"] != "minimal":
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
                template_name=init_args.get("template", "minimal"),
                registry_style=init_args.get("registry_style", "extend"),
                context=context if context else None,
            )
            return project_path
        except Exception as e:
            console.print(f"[warning]Error creating vanilla project: {e}[/warning]")
            return None


def _perform_migration_analysis(
    facility_dir: Path,
    old_vanilla_dir: Path | None,
    new_vanilla_dir: Path,
) -> dict[str, Any]:
    """Perform three-way diff analysis for migration.

    Args:
        facility_dir: Path to facility's current project
        old_vanilla_dir: Path to old vanilla project (None if not available)
        new_vanilla_dir: Path to new vanilla project

    Returns:
        Dictionary with classified files by category
    """
    results: dict[str, list[dict[str, Any]]] = {
        "auto_copy": [],
        "preserve": [],
        "merge": [],
        "new": [],
        "data": [],
        "removed": [],
    }

    # Collect all unique file paths across all directories
    all_files: set[str] = set()

    for directory in [facility_dir, old_vanilla_dir, new_vanilla_dir]:
        if directory and directory.exists():
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(directory))
                    # Skip manifest and hidden git files
                    if rel_path == MANIFEST_FILENAME or ".git" in rel_path:
                        continue
                    all_files.add(rel_path)

    # Classify each file
    for rel_path in sorted(all_files):
        facility_path = facility_dir / rel_path
        old_vanilla_path = old_vanilla_dir / rel_path if old_vanilla_dir else None
        new_vanilla_path = new_vanilla_dir / rel_path

        facility_hash = _calculate_file_hash(facility_path) if facility_path.exists() else None
        old_vanilla_hash = (
            _calculate_file_hash(old_vanilla_path)
            if old_vanilla_path and old_vanilla_path.exists()
            else None
        )
        new_vanilla_hash = (
            _calculate_file_hash(new_vanilla_path) if new_vanilla_path.exists() else None
        )

        category = _classify_file(rel_path, facility_hash, old_vanilla_hash, new_vanilla_hash)

        file_info = {
            "path": rel_path,
            "facility_exists": facility_path.exists(),
            "old_vanilla_exists": old_vanilla_path.exists() if old_vanilla_path else False,
            "new_vanilla_exists": new_vanilla_path.exists(),
        }

        results[category.value].append(file_info)

    return results


def _generate_merge_prompt(
    rel_path: str,
    facility_content: str,
    old_vanilla_content: str | None,
    new_vanilla_content: str,
    old_version: str,
    new_version: str,
) -> str:
    """Generate a markdown merge prompt for a file requiring manual merge.

    Args:
        rel_path: Relative path of the file
        facility_content: Content from facility's version
        old_vanilla_content: Content from old template (may be None)
        new_vanilla_content: Content from new template
        old_version: Old OSPREY version
        new_version: New OSPREY version

    Returns:
        Markdown content for the merge prompt file
    """
    prompt = f"""# OSPREY Migration: Merge Required

**File**: `{rel_path}`
**Migration**: {old_version} -> {new_version}

## Facility's Current Version

```
{facility_content}
```

"""

    if old_vanilla_content:
        prompt += f"""## Original Template ({old_version})

```
{old_vanilla_content}
```

"""

    prompt += f"""## New Template ({new_version})

```
{new_vanilla_content}
```

## Your Task

1. **Preserve facility customizations** - Keep any facility-specific configurations, paths, or settings
2. **Apply template updates** - Incorporate new fields, fixes, or structural changes from the new template
3. **Resolve conflicts** - When in doubt, prioritize facility values for business logic

## Guidelines

- Look for new configuration options that should be added
- Check for renamed or restructured fields
- Preserve comments that explain facility-specific choices
- Test the merged configuration before committing

## Output

Please provide:
1. The merged file content
2. A brief summary of changes made
"""

    return prompt


def _generate_migration_directory(
    project_dir: Path,
    analysis: dict[str, list[dict[str, Any]]],
    facility_dir: Path,
    old_vanilla_dir: Path | None,
    new_vanilla_dir: Path,
    old_version: str,
    new_version: str,
) -> Path:
    """Generate _migration/ directory with merge prompts and summaries.

    Args:
        project_dir: Directory where to create _migration/
        analysis: File classification from _perform_migration_analysis
        facility_dir: Path to facility's project
        old_vanilla_dir: Path to old vanilla (may be None)
        new_vanilla_dir: Path to new vanilla
        old_version: Old OSPREY version
        new_version: New OSPREY version

    Returns:
        Path to the created _migration directory
    """
    migration_dir = project_dir / "_migration"
    migration_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (migration_dir / "merge_required").mkdir(exist_ok=True)
    (migration_dir / "auto_applied").mkdir(exist_ok=True)
    (migration_dir / "preserved").mkdir(exist_ok=True)

    # Generate README
    readme_content = f"""# OSPREY Migration: {old_version} -> {new_version}

Generated: {datetime.now(UTC).isoformat()}

## Summary

| Category | Count | Action |
|----------|-------|--------|
| Auto-copy | {len(analysis["auto_copy"])} | Template changed, you didn't - safe to update |
| Preserve | {len(analysis["preserve"])} | You customized, template unchanged - keep yours |
| Merge Required | {len(analysis["merge"])} | Both changed - needs manual review |
| New Files | {len(analysis["new"])} | Added in new template - copy to project |
| Data | {len(analysis["data"])} | User data - always preserved |

## Files Requiring Merge

These files have been modified in both your project and the template.
See `merge_required/` for detailed merge prompts.

"""

    for file_info in analysis["merge"]:
        readme_content += f"- `{file_info['path']}`\n"

    readme_content += """

## Next Steps

1. Review files in `merge_required/` directory
2. For each file, merge your customizations with template updates
3. Copy merged files to your project
4. Run `osprey health` to verify configuration
5. Delete `_migration/` directory when complete
"""

    (migration_dir / "README.md").write_text(readme_content, encoding="utf-8")

    # Generate merge prompts for each file needing merge
    for file_info in analysis["merge"]:
        rel_path = file_info["path"]

        facility_content = _read_file_content(facility_dir / rel_path) or "[File not readable]"
        old_vanilla_content = (
            _read_file_content(old_vanilla_dir / rel_path) if old_vanilla_dir else None
        )
        new_vanilla_content = (
            _read_file_content(new_vanilla_dir / rel_path) or "[File not readable]"
        )

        prompt = _generate_merge_prompt(
            rel_path,
            facility_content,
            old_vanilla_content,
            new_vanilla_content,
            old_version,
            new_version,
        )

        # Create prompt file with safe filename
        safe_name = rel_path.replace("/", "_").replace("\\", "_")
        prompt_path = migration_dir / "merge_required" / f"{safe_name}.md"
        prompt_path.write_text(prompt, encoding="utf-8")

    # Generate auto-applied summary
    auto_summary = "# Auto-Applied Changes\n\n"
    auto_summary += (
        "These files were updated from the new template because you hadn't modified them.\n\n"
    )
    for file_info in analysis["auto_copy"]:
        auto_summary += f"- `{file_info['path']}`\n"
    (migration_dir / "auto_applied" / "summary.md").write_text(auto_summary, encoding="utf-8")

    # Generate preserved summary
    preserved_summary = "# Preserved Files\n\n"
    preserved_summary += "These files were kept unchanged because you customized them.\n\n"
    for file_info in analysis["preserve"]:
        preserved_summary += f"- `{file_info['path']}`\n"
    (migration_dir / "preserved" / "summary.md").write_text(preserved_summary, encoding="utf-8")

    return migration_dir


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
    settings = _detect_project_settings(project)

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
    original_get_version = manager._get_framework_version
    manager._get_framework_version = lambda: version  # type: ignore[method-assign]

    try:
        manifest = manager.generate_manifest(
            project_dir=project,
            project_name=project_name,
            template_name=settings.get("template", "minimal"),
            registry_style=settings.get("registry_style", "extend"),
            context=context,
        )
    finally:
        manager._get_framework_version = original_get_version  # type: ignore[method-assign]

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
    manifest = _load_manifest(project)

    if not manifest:
        console.print("[warning]No manifest found[/warning]")
        console.print("Run 'osprey migrate init' to create one for this project.")
        return

    # Get versions
    project_version = manifest["creation"]["osprey_version"]
    manager = TemplateManager()
    current_version = manager._get_framework_version()

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
    manifest = _load_manifest(project)

    if not manifest:
        console.print("[error]No manifest found[/error]")
        console.print("Run 'osprey migrate init' first to create a manifest.")
        raise click.Abort()

    old_version = manifest["creation"]["osprey_version"]
    manager = TemplateManager()
    new_version = manager._get_framework_version()

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
        analysis = _perform_migration_analysis(project, old_vanilla_dir, new_vanilla_dir)

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
        migration_dir = _generate_migration_directory(
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
