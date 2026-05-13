"""Project creation helpers: directory structure, services, data files.

Includes :func:`materialize_tier_dbs`, the build-time step that picks the
tier-routed channel-database file(s) for the selected paradigm(s), copies
them into the flat ``data/channel_databases/<paradigm>.json`` locations,
and prunes the now-redundant ``tiers/`` subtree.
"""

import json
import logging
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from osprey.cli.styles import console
from osprey.cli.templates._rendering import render_template

logger = logging.getLogger("osprey.cli.templates")


def _detect_system_timezone() -> str | None:
    """Detect the system IANA timezone name (e.g., 'America/New_York').

    Uses /etc/localtime symlink (macOS/Linux) or /etc/timezone (Linux).
    Returns None if detection fails.
    """
    import pathlib

    # macOS / Linux: /etc/localtime is usually a symlink into zoneinfo
    localtime = pathlib.Path("/etc/localtime")
    if localtime.is_symlink():
        target = str(localtime.resolve())
        if "zoneinfo/" in target:
            tz_name = target.split("zoneinfo/", 1)[1]
            try:
                from zoneinfo import ZoneInfo

                ZoneInfo(tz_name)
                return tz_name
            except (KeyError, Exception):
                pass

    # Linux: /etc/timezone contains the IANA name directly
    etc_tz = pathlib.Path("/etc/timezone")
    if etc_tz.exists():
        tz_name = etc_tz.read_text().strip()
        if "/" in tz_name:
            return tz_name

    return None


def detect_environment_variables() -> dict[str, str]:
    """Detect environment variables from the system for use in templates.

    This checks for common environment variables that are typically
    needed in .env files (API keys, paths, etc.) and returns those that are
    currently set in the system.

    Sources are checked in priority order (highest priority last, so it wins):
    1. Shell environment (os.environ)
    2. Project root .env file (if running from within an osprey project)

    The .env file takes precedence because it represents the user's explicitly
    configured project values, which may be more current than stale shell exports.

    Returns:
        Dictionary of detected environment variables with their values.
        Only includes variables that are actually set (non-empty).
    """
    # API key env vars from canonical registry + non-API env vars
    from osprey.models.provider_registry import PROVIDER_API_KEYS

    env_vars_to_check = [v for v in PROVIDER_API_KEYS.values() if v is not None] + [
        "PROJECT_ROOT",
        "LOCAL_PYTHON_VENV",
        "CONFLUENCE_ACCESS_TOKEN",
        "TZ",
    ]

    # Load .env file from current directory if it exists (project root values
    # take precedence over shell environment for API keys)
    dotenv_values = {}
    env_file = Path.cwd() / ".env"
    if env_file.is_file():
        try:
            from dotenv import dotenv_values as _dotenv_values

            dotenv_values = _dotenv_values(env_file)
        except ImportError:
            pass

    detected = {}
    for var in env_vars_to_check:
        # Shell environment first, then .env file overrides
        value = os.environ.get(var)
        if value:
            detected[var] = value
        env_file_value = dotenv_values.get(var)
        if env_file_value:
            detected[var] = env_file_value

    # If TZ not in environment, try to detect system timezone
    if "TZ" not in detected:
        detected_tz = _detect_system_timezone()
        if detected_tz:
            detected["TZ"] = detected_tz

    return detected


def create_project_structure(
    template_root: Path,
    jinja_env,
    project_dir: Path,
    data_bundle: str,
    ctx: dict,
):
    """Create base project files (config, README, pyproject.toml, etc.).

    Args:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
        project_dir: Root directory of the project
        data_bundle: Name of the data bundle (apps/ subdirectory) to use
        ctx: Template context variables
    """
    project_template_dir = template_root / "project"
    app_template_dir = template_root / "apps" / data_bundle

    # Render template files (no pyproject.toml or requirements.txt -- no src/ package)
    files_to_render = [
        ("config.yml.j2", "config.yml"),
        ("env.example.j2", ".env.example"),
        ("README.md.j2", "README.md"),
    ]

    # Copy static files
    static_files = [
        # requirements.txt moved to rendered templates to handle {{ framework_version }}
    ]

    for template_file, output_file in files_to_render:
        # Check if app template has its own version first (e.g., requirements.txt.j2)
        app_specific_template = app_template_dir / (
            template_file + ".j2" if not template_file.endswith(".j2") else template_file
        )
        default_template = project_template_dir / template_file

        if app_specific_template.exists():
            # Use app-specific template
            render_template(
                jinja_env,
                f"apps/{data_bundle}/{app_specific_template.name}",
                ctx,
                project_dir / output_file,
            )
        elif default_template.exists():
            # Use default project template
            render_template(jinja_env, f"project/{template_file}", ctx, project_dir / output_file)

    # Create .env file only if API keys are detected
    from osprey.models.provider_registry import PROVIDER_API_KEYS

    detected_env_vars = ctx.get("env", {})
    api_key_names = {v for v in PROVIDER_API_KEYS.values() if v is not None}
    has_api_keys = any(key in detected_env_vars for key in api_key_names)

    if has_api_keys:
        env_template = project_template_dir / "env.j2"
        if env_template.exists():
            render_template(jinja_env, "project/env.j2", ctx, project_dir / ".env")
            # Set proper permissions (owner read/write only)
            os.chmod(project_dir / ".env", 0o600)

    # Copy static files
    for src_name, dst_name in static_files:
        src_file = project_template_dir / src_name
        if src_file.exists():
            shutil.copy(src_file, project_dir / dst_name)

    # Copy gitignore (renamed from 'gitignore' to '.gitignore')
    gitignore_source = project_template_dir / "gitignore"
    if gitignore_source.exists():
        shutil.copy(gitignore_source, project_dir / ".gitignore")


def copy_services(template_root: Path, project_dir: Path):
    """Copy service configurations to project (flattened structure).

    Services are copied with a flattened structure (not nested under osprey/).
    This makes the user's project structure cleaner.

    Args:
        template_root: Path to osprey's bundled templates directory
        project_dir: Root directory of the project
    """
    src_services = template_root / "services"
    dst_services = project_dir / "services"

    if not src_services.exists():
        return

    dst_services.mkdir(parents=True, exist_ok=True)

    # Copy each service directory individually (flattened)
    for item in src_services.iterdir():
        if item.is_dir():
            shutil.copytree(item, dst_services / item.name, dirs_exist_ok=True)
        elif item.is_file() and item.suffix in [".j2", ".yml", ".yaml"]:
            # Copy docker-compose template/config files
            shutil.copy(item, dst_services / item.name)


def copy_services_selective(template_root: Path, project_dir: Path, service_names: list[str]):
    """Copy only specified service directories to project.

    Args:
        template_root: Path to osprey's bundled templates directory
        project_dir: Root directory of the project
        service_names: List of service directory names to copy (e.g., ["postgresql"])
    """
    src_services = template_root / "services"
    dst_services = project_dir / "services"

    if not src_services.exists():
        return

    dst_services.mkdir(parents=True, exist_ok=True)

    for name in service_names:
        src_dir = src_services / name
        if src_dir.is_dir():
            shutil.copytree(src_dir, dst_services / name, dirs_exist_ok=True)

    # Also copy docker-compose template if any services were copied
    if service_names:
        for item in src_services.iterdir():
            if item.is_file() and item.suffix in [".j2", ".yml", ".yaml"]:
                shutil.copy(item, dst_services / item.name)


def _copy_data_tree(src_dir: Path, dst_dir: Path, template_root: Path, jinja_env, ctx: dict):
    """Copy a data directory, rendering .j2 files and copying the rest as-is.

    Files ending in .j2 are rendered through Jinja2 (with the extension stripped).
    All other files are copied verbatim.
    """
    for item in src_dir.iterdir():
        if item.is_dir():
            _copy_data_tree(item, dst_dir / item.name, template_root, jinja_env, ctx)
        elif item.suffix == ".j2":
            # Render through Jinja2 and strip the .j2 extension
            dst_file = dst_dir / item.stem  # e.g. foo.json.j2 → foo.json
            template_path = str(item.relative_to(template_root))
            render_template(jinja_env, template_path, ctx, dst_file)
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst_dir / item.name)


def copy_template_data(
    template_root: Path,
    project_dir: Path,
    package_name: str,
    data_bundle: str,
    ctx: dict,
    jinja_env=None,
):
    """Copy data files from template to project root (no src/ package).

    Data files (channel databases, channel_limits.json, logbook seeds,
    benchmark datasets) are placed at project_dir/data/.  Files with a
    ``.j2`` extension are rendered through Jinja2 (extension stripped);
    all other files are copied as-is.

    Args:
        template_root: Path to osprey's bundled templates directory
        project_dir: Root directory of the project
        package_name: Python package name (used to locate template data dirs)
        data_bundle: Name of the data bundle (apps/ subdirectory) to use
        ctx: Template context variables
        jinja_env: Optional Jinja2 environment for rendering .j2 data files
    """
    app_template_dir = template_root / "apps" / data_bundle

    # Look for data/ subdirectory in the template
    template_data_dir = app_template_dir / "data"
    if template_data_dir.exists() and template_data_dir.is_dir():
        dst_data = project_dir / "data"
        if jinja_env is not None:
            _copy_data_tree(template_data_dir, dst_data, template_root, jinja_env, ctx)
        else:
            shutil.copytree(template_data_dir, dst_data, dirs_exist_ok=True)
        console.print(
            f"  [success]✓[/success] Copied template data files to [path]{dst_data}[/path]"
        )
        return

    # Fallback: scan for data/ directories inside template subdirectories
    # (some templates put data inside package-level dirs)
    for template_file in app_template_dir.rglob("*"):
        if not template_file.is_dir():
            continue
        if template_file.name == "data":
            # Copy to project root data/ (flatten from template structure)
            dst_data = project_dir / "data"
            if jinja_env is not None:
                _copy_data_tree(template_file, dst_data, template_root, jinja_env, ctx)
            else:
                if not dst_data.exists():
                    shutil.copytree(template_file, dst_data, dirs_exist_ok=True)
                else:
                    # Merge into existing data/
                    for item in template_file.iterdir():
                        dst_item = dst_data / item.name
                        if item.is_dir():
                            shutil.copytree(item, dst_item, dirs_exist_ok=True)
                        elif item.is_file():
                            dst_item.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dst_item)
            console.print(
                f"  [success]✓[/success] Copied template data files to [path]{dst_data}[/path]"
            )
            return


_ALL_PARADIGMS: tuple[str, ...] = ("in_context", "hierarchical", "middle_layer")


def materialize_tier_dbs(
    project_dir: Path, tier: int, channel_finder_mode: str | None
) -> None:
    """Materialize tier-routed channel databases into flat destinations.

    The preset ships channel databases under
    ``data/channel_databases/tiers/tier{1,2,3}/<paradigm>.json``.  After
    ``osprey build``, this helper picks the requested ``tier`` and copies
    the relevant paradigm DB(s) up to the flat
    ``data/channel_databases/<paradigm>.json`` location, then removes the
    ``tiers/`` subtree so only the active DBs remain.
    Facility profiles overlaying their own DB files don't care which tier
    was selected — their overlay overwrites the preset DB after this step.
    Tier itself is build-time only and is NOT written into ``config.yml``.
    Paradigm mapping (matches the rest of the codebase, see
    :mod:`osprey.cli.templates.manager`):

    * ``"in_context"`` → ``{"in_context"}``
    * ``"hierarchical"`` → ``{"hierarchical"}``
    * ``"middle_layer"`` → ``{"middle_layer"}``
    * ``"all"`` or ``None`` → all three paradigms

    Args:
        project_dir: Root directory of the rendered project.
        tier: Tier number (1, 2, or 3) selecting the source subdirectory.
        channel_finder_mode: Paradigm selector from the build profile.

    Raises:
        FileNotFoundError: If a required tier source DB is missing. Raised
            BEFORE any destination file is overwritten or any directory is
            removed, so the project tree is left untouched on failure.

    No-ops (returns silently) when the rendered project carries no
    ``data/channel_databases/tiers/`` subtree — bundles that don't ship
    channel-finder DBs (e.g. ``hello_world``) have nothing to materialize.
    """
    tiers_root = project_dir / "data" / "channel_databases" / "tiers"
    if not tiers_root.exists():
        return

    if channel_finder_mode in (None, "all"):
        paradigms: set[str] = set(_ALL_PARADIGMS)
    elif channel_finder_mode in _ALL_PARADIGMS:
        paradigms = {channel_finder_mode}
    else:
        # Unknown mode — be strict; the build profile validator should have
        # already caught this, but don't silently materialize the wrong DBs.
        raise ValueError(
            f"Unknown channel_finder_mode {channel_finder_mode!r}; "
            f"expected one of {sorted(_ALL_PARADIGMS)!r}, 'all', or None"
        )

    tier_dir = tiers_root / f"tier{tier}"
    flat_root = project_dir / "data" / "channel_databases"

    # Resolve every (src, dst) pair up front, validate existence, then copy.
    # This keeps the destination tree consistent on FileNotFoundError.
    pairs: list[tuple[Path, Path]] = []
    for paradigm in sorted(paradigms):
        src = tier_dir / f"{paradigm}.json"
        dst = flat_root / f"{paradigm}.json"
        if not src.exists():
            raise FileNotFoundError(
                f"Tier-routed channel database not found: {src} "
                f"(tier={tier}, paradigm={paradigm!r})"
            )
        pairs.append((src, dst))

    for src, dst in pairs:
        shutil.copy2(src, dst)

    # All copies succeeded — safe to prune the tiers/ subtree now.
    shutil.rmtree(tiers_root)

    console.print(
        f"  [success]✓[/success] Materialized tier{tier} channel database(s) "
        f"for {sorted(paradigms)!r} to [path]{flat_root}[/path]"
    )


def rebase_logbook_timestamps(project_dir: Path) -> None:
    """Shift demo logbook timestamps so the most recent entry is near 'now'.

    The bundled demo logbook has fixed timestamps (e.g. March 2024).  When a
    user runs ``osprey build`` months or years later, the logbook entries look
    stale and won't align with mock archiver data (which is generated relative
    to the current time).

    This loads the copied demo logbook, finds the latest entry timestamp,
    computes the offset needed to place it 2 days before the current time,
    and shifts every entry by that offset.  Relative gaps between entries
    are preserved.
    """
    logbook_path = project_dir / "data" / "logbook_seed" / "demo_logbook.json"
    if not logbook_path.exists():
        return

    try:
        data = json.loads(logbook_path.read_text())
        entries = data.get("entries", [])
        if not entries:
            return

        timestamps = []
        for entry in entries:
            ts_str = entry.get("timestamp", "")
            if ts_str:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                timestamps.append(ts)

        if not timestamps:
            return

        latest = max(timestamps)
        target = datetime.now(UTC) - timedelta(days=2)
        # Round to whole days so time-of-day is preserved -- entry text
        # contains hardcoded clock times (e.g. "tripped at 03:15") that
        # we can't programmatically adjust.
        offset = timedelta(days=(target - latest).days)

        for entry in entries:
            ts_str = entry.get("timestamp", "")
            if ts_str:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                new_ts = ts + offset
                entry["timestamp"] = new_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        logbook_path.write_text(json.dumps(data, indent=2) + "\n")
        span_days = (max(timestamps) - min(timestamps)).days
        console.print(
            f"  [success]✓[/success] Rebased {len(entries)} logbook entries "
            f"(spanning {span_days} days) to current date"
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        pass  # Non-fatal -- leave the file as-is if anything goes wrong


def create_application_code(
    template_root: Path,
    jinja_env,
    src_dir: Path,
    package_name: str,
    data_bundle: str,
    ctx: dict,
    project_root: Path = None,
):
    """Create application code from template.

    Args:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
        src_dir: src/ directory where package will be created
        package_name: Python package name (e.g., "my_assistant")
        data_bundle: Name of the data bundle (apps/ subdirectory) to use
        ctx: Template context variables
        project_root: Actual project root (for placing scripts/ at root)

    Note:
        Special handling: Files in scripts/ directory are placed at project root
        instead of inside the package to provide convenient CLI access.
    """
    app_template_dir = template_root / "apps" / data_bundle
    app_dir = src_dir / package_name
    app_dir.mkdir(parents=True)

    # Use src_dir's parent as project_root if not provided
    if project_root is None:
        project_root = src_dir.parent

    # Project-level files that should only live at project root, not in src/
    # These are handled by create_project_structure() and should be skipped here
    PROJECT_LEVEL_FILES = {
        "config.yml.j2",
        "config.yml",
        "README.md.j2",
        "README.md",
        "env.example.j2",
        "env.example",
        "env.j2",
        ".env",
        "requirements.txt.j2",
        "requirements.txt",
        "pyproject.toml.j2",
        "pyproject.toml",
    }

    # Process all files in the template
    for template_file in app_template_dir.rglob("*"):
        if not template_file.is_file():
            continue

        rel_path = template_file.relative_to(app_template_dir)

        # Skip project-level files at template root (handled by create_project_structure)
        if len(rel_path.parts) == 1 and rel_path.name in PROJECT_LEVEL_FILES:
            continue

        # Special handling for scripts/ directory - place at project root
        if rel_path.parts[0] == "scripts":
            base_output_dir = project_root
            output_rel_path = rel_path
        else:
            base_output_dir = app_dir
            output_rel_path = rel_path

        # Determine output path
        if template_file.suffix == ".j2":
            # Template file - render it
            output_name = template_file.stem  # Remove .j2 extension
            output_path = base_output_dir / output_rel_path.parent / output_name
            # Convert Windows backslashes to forward slashes for Jinja2
            # (harmless on Linux/macOS where paths already use forward slashes)
            template_path_str = f"apps/{data_bundle}/{rel_path}".replace("\\", "/")
            render_template(jinja_env, template_path_str, ctx, output_path)
        else:
            # Static file - copy directly
            output_path = base_output_dir / output_rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(template_file, output_path)


def create_agent_data_structure(template_root: Path, project_dir: Path, ctx: dict):
    """Create _agent_data directory structure for the project.

    This creates the agent data directory and all standard subdirectories
    based on osprey's default configuration. This ensures that container
    deployments won't fail due to missing mount points.

    Args:
        template_root: Path to osprey's bundled templates directory
        project_dir: Root directory of the project
        ctx: Template context variables (used for conditional directory creation)
    """
    # Create main _agent_data directory
    agent_data_dir = project_dir / "_agent_data"
    agent_data_dir.mkdir(parents=True, exist_ok=True)

    # Create standard subdirectories
    subdirs = [
        "executed_scripts",
        "user_memory",
        "api_calls",
    ]

    for subdir in subdirs:
        subdir_path = agent_data_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)

    console.print(
        f"  [success]✓[/success] Created agent data structure at [path]{agent_data_dir}[/path]"
    )

    # Create a README to explain the directory structure
    readme_content = """# Agent Data Directory

This directory contains runtime data for the Claude Code project:

- `executed_scripts/`: Python scripts executed via MCP tools
- `user_memory/`: User memory data
- `api_calls/`: Raw LLM API inputs/outputs (when API logging enabled)
"""

    readme_content += """
This directory is excluded from git (see .gitignore) but is required for
proper framework operation, especially when using containerized services.
"""

    readme_path = agent_data_dir / "README.md"
    # Use UTF-8 encoding explicitly to support Unicode characters on Windows
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
