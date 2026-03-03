"""Migration engine — pure business logic for OSPREY project migrations.

Handles file classification, migration analysis, merge prompt generation,
and project settings detection. No CLI or console dependencies.
"""

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = ".osprey-manifest.json"


class FileCategory(Enum):
    """Classification categories for migration files."""

    DATA = "data"  # User data directories - always preserve
    AUTO_COPY = "auto_copy"  # Template changed, facility didn't - copy from new
    PRESERVE = "preserve"  # Facility modified, template unchanged - keep facility
    MERGE = "merge"  # Both changed - needs manual/AI merge
    NEW = "new"  # Only exists in new template - copy from new
    REMOVED = "removed"  # Only exists in old template - may need cleanup


def load_manifest(project_dir: Path) -> dict[str, Any] | None:
    """Load manifest from project directory.

    Args:
        project_dir: Path to the project root

    Returns:
        Manifest dict if found, None otherwise (caller handles messaging)
    """
    manifest_path = project_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def detect_project_settings(project_dir: Path) -> dict[str, Any]:
    """Detect settings from an existing project without a manifest.

    Examines config.yml, registry.py, pyproject.toml etc. to infer
    the original init settings.

    Args:
        project_dir: Path to the project root

    Returns:
        Dictionary of detected settings with a "warnings" key containing
        any warning messages (caller handles display)
    """
    settings: dict[str, Any] = {
        "detected": True,
        "confidence": {},
        "warnings": [],
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
                    settings["template"] = "control_assistant"
                    settings["confidence"]["template"] = "medium"
        except Exception as e:
            settings["warnings"].append(f"Could not parse config.yml: {e}")

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
            settings["warnings"].append(f"Could not parse pyproject.toml: {e}")

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
                        settings["warnings"].append(
                            f"Could not analyze registry file '{registry_path}': {e}"
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


def calculate_file_hash(file_path: Path) -> str | None:
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


def read_file_content(file_path: Path) -> str | None:
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


def classify_file(
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


def perform_migration_analysis(
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

        facility_hash = calculate_file_hash(facility_path) if facility_path.exists() else None
        old_vanilla_hash = (
            calculate_file_hash(old_vanilla_path)
            if old_vanilla_path and old_vanilla_path.exists()
            else None
        )
        new_vanilla_hash = (
            calculate_file_hash(new_vanilla_path) if new_vanilla_path.exists() else None
        )

        category = classify_file(rel_path, facility_hash, old_vanilla_hash, new_vanilla_hash)

        file_info = {
            "path": rel_path,
            "facility_exists": facility_path.exists(),
            "old_vanilla_exists": old_vanilla_path.exists() if old_vanilla_path else False,
            "new_vanilla_exists": new_vanilla_path.exists(),
        }

        results[category.value].append(file_info)

    return results


def generate_merge_prompt(
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


def generate_migration_directory(
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
        analysis: File classification from perform_migration_analysis
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

        facility_content = read_file_content(facility_dir / rel_path) or "[File not readable]"
        old_vanilla_content = (
            read_file_content(old_vanilla_dir / rel_path) if old_vanilla_dir else None
        )
        new_vanilla_content = (
            read_file_content(new_vanilla_dir / rel_path) or "[File not readable]"
        )

        prompt = generate_merge_prompt(
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


def migrate_claude_code_config(
    claude_code: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Transform legacy claude_code config keys to new extensibility format.

    Converts:
        disable_servers: [x]  ->  servers: {x: {enabled: false}}
        extra_servers: {n: s} ->  servers: {n: s}
        disable_agents: [x]  ->  agents: {x: {enabled: false}}

    Args:
        claude_code: The claude_code section from config.yml

    Returns:
        Tuple of (servers_dict, agents_dict, change_descriptions)
    """
    changes: list[str] = []

    # Build servers dict
    servers = dict(claude_code.get("servers", {}))

    disable_servers = claude_code.get("disable_servers", [])
    for name in disable_servers:
        servers[name] = {"enabled": False}
        changes.append(f"disable_servers: {name} -> servers.{name}.enabled: false")

    extra_servers = claude_code.get("extra_servers", {})
    for name, spec in extra_servers.items():
        servers[name] = dict(spec) if hasattr(spec, "items") else spec
        changes.append(f"extra_servers: {name} -> servers.{name}: {{...}}")

    # Build agents dict
    agents = dict(claude_code.get("agents", {}))

    disable_agents = claude_code.get("disable_agents", [])
    for name in disable_agents:
        agents[name] = {"enabled": False}
        changes.append(f"disable_agents: {name} -> agents.{name}.enabled: false")

    return servers, agents, changes
