"""Project creation helpers: directory structure, services, data files.

Includes :func:`materialize_tier_artifacts`, the build-time step that picks
the tier-routed channel-database file(s) and the matching tier-routed
benchmark query file for the selected paradigm, copies them into the canonical
flat locations (``data/channel_databases/<paradigm>.json`` and
``data/benchmarks/queries.json``), and prunes the now-redundant ``tiers/`` and
``benchmarks/cross_paradigm/`` subtrees.
"""

import logging
import os
import shutil
from pathlib import Path

from osprey.cli.styles import console
from osprey.cli.templates._rendering import render_template

logger = logging.getLogger("osprey.cli.templates")

# Fallback default for the Dockerfile.j2 CLAUDE_CLI_VERSION build ARG when
# the project's config.yml doesn't set claude_code.cli_version (the common
# case — it's an opt-in pin). Matches the dispatch-worker image's pinned
# version (src/osprey/templates/services/event_dispatcher/Dockerfile) so a
# freshly-built project image ships a deliberately-chosen CLI version rather
# than silently tracking whatever `npm install -g @anthropic-ai/claude-code`
# resolves to at build time. Bump deliberately, alongside the dispatch-worker
# pin.
_DEFAULT_CLAUDE_CLI_VERSION = "2.1.146"


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


def _render_env_preserving_existing(jinja_env, project_dir: Path, ctx: dict) -> None:
    """Render ``project/env.j2`` to ``.env``, merging with an existing file.

    On a fresh build this is a plain render. On a --force re-render an
    existing ``.env`` is authoritative: its values win over freshly detected
    environment values and keys it alone carries are appended — the project's
    service tokens and volume-pinned passwords must survive a re-render (see
    :func:`osprey.utils.dotenv.merge_env_preserving_existing`).
    """
    env_path = project_dir / ".env"
    rendered = jinja_env.get_template("project/env.j2").render(**ctx)
    if env_path.exists():
        from osprey.utils.dotenv import merge_env_preserving_existing

        content = merge_env_preserving_existing(rendered, env_path.read_text(encoding="utf-8"))
    else:
        content = rendered if rendered.endswith("\n") else rendered + "\n"
    env_path.write_text(content, encoding="utf-8")
    # Owner read/write only: the file carries secrets.
    os.chmod(env_path, 0o600)


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

    # Expose claude_code.cli_version to Dockerfile.j2's CLAUDE_CLI_VERSION ARG
    # default, so the same version pin that `osprey claude chat`/`osprey web`
    # honor at runtime (osprey.utils.claude_launcher) also pins the image's
    # build-time CLI install. Callers may pre-populate ctx["claude_code_cli_version"]
    # (flat) or ctx["claude_code"]["cli_version"] (nested, mirroring config.yml's
    # shape); absent either, fall back to the framework's last verified pin.
    if "claude_code_cli_version" not in ctx:
        ctx["claude_code_cli_version"] = (
            ctx.get("claude_code", {}).get("cli_version") or _DEFAULT_CLAUDE_CLI_VERSION
        )

    # Render template files (no pyproject.toml or requirements.txt -- no src/ package)
    files_to_render = [
        ("config.yml.j2", "config.yml"),
        ("env.example.j2", ".env.example"),
        ("README.md.j2", "README.md"),
        # Reference container image — rendered once at build; regen never touches it
        ("Dockerfile.j2", "Dockerfile"),
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
            _render_env_preserving_existing(jinja_env, project_dir, ctx)

    # Copy static files
    for src_name, dst_name in static_files:
        src_file = project_template_dir / src_name
        if src_file.exists():
            shutil.copy(src_file, project_dir / dst_name)

    # Copy gitignore (renamed from 'gitignore' to '.gitignore')
    gitignore_source = project_template_dir / "gitignore"
    if gitignore_source.exists():
        shutil.copy(gitignore_source, project_dir / ".gitignore")

    # Copy dockerignore (renamed to '.dockerignore') — keeps .env/.venv/.git
    # out of the image built from the generated Dockerfile
    dockerignore_source = project_template_dir / "dockerignore"
    if dockerignore_source.exists():
        shutil.copy(dockerignore_source, project_dir / ".dockerignore")


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


def materialize_tier_artifacts(project_dir: Path, tier: int, channel_finder_mode: str) -> None:
    """Materialize tier-routed channel databases AND benchmark queries.

    The preset ships:
    - channel databases under
      ``data/channel_databases/tiers/tier{1,3}/<paradigm>.json``
    - benchmark query files under
      ``data/benchmarks/cross_paradigm/queries/tier{1,3}_queries.json``

    After ``osprey build``, this helper picks the requested ``tier`` and:
    - copies the active paradigm's DB to the flat
      ``data/channel_databases/<paradigm>.json``
    - copies the tier-matching query file to the flat
      ``data/benchmarks/queries.json``
    - prunes both the ``tiers/`` and ``benchmarks/cross_paradigm/`` subtrees
      so only the active artifacts remain.

    Facility profiles overlaying their own DB files don't care which tier
    was selected — their overlay overwrites the preset DB after this step.
    Tier itself is build-time only and is NOT written into ``config.yml``.

    Args:
        project_dir: Root directory of the rendered project.
        tier: Tier number (1 or 3) selecting the source subdirectories. Tier 1
            ships only the ``in_context`` paradigm; the build-profile validator
            rejects tier 1 paired with a non-in_context channel_finder_mode
            before this step, so a missing tier1/<paradigm>.json here is a bug.
        channel_finder_mode: Paradigm selector from the build profile. Must
            be one of ``"in_context"``, ``"hierarchical"``, ``"middle_layer"``.

    Raises:
        ValueError: If ``channel_finder_mode`` is not one of the three valid
            paradigms (the build-profile validator and ``manager.py`` should
            catch this earlier, but this is a defensive guard).
        FileNotFoundError: If a required source artifact is missing. Raised
            BEFORE any destination file is overwritten or any directory is
            removed, so the project tree is left untouched on failure.

    No-ops (returns silently) when the rendered project carries no
    ``data/channel_databases/tiers/`` subtree — bundles that don't ship
    channel-finder DBs (e.g. ``hello_world``) have nothing to materialize.
    """
    tiers_root = project_dir / "data" / "channel_databases" / "tiers"
    if not tiers_root.exists():
        return

    if channel_finder_mode not in _ALL_PARADIGMS:
        raise ValueError(
            f"Unknown channel_finder_mode {channel_finder_mode!r}; "
            f"expected one of {sorted(_ALL_PARADIGMS)!r}"
        )
    paradigms: set[str] = {channel_finder_mode}

    tier_dir = tiers_root / f"tier{tier}"
    flat_root = project_dir / "data" / "channel_databases"
    queries_src_root = project_dir / "data" / "benchmarks" / "cross_paradigm"

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

    # The unified query file lives under the preset's
    # data/benchmarks/cross_paradigm/queries/ subtree, which copy_template_data
    # wholesale-copies into the project. Pick the tier-matching file and
    # materialize it as the canonical data/benchmarks/queries.json.
    queries_src = queries_src_root / "queries" / f"tier{tier}_queries.json"
    queries_dst = project_dir / "data" / "benchmarks" / "queries.json"
    if not queries_src.exists():
        raise FileNotFoundError(
            f"Tier-routed benchmark queries file not found: {queries_src} (tier={tier})"
        )
    pairs.append((queries_src, queries_dst))

    for src, dst in pairs:
        shutil.copy2(src, dst)

    # All copies succeeded — safe to prune the preset's staging subtrees.
    shutil.rmtree(tiers_root)
    if queries_src_root.exists():
        shutil.rmtree(queries_src_root)

    console.print(
        f"  [success]✓[/success] Materialized tier{tier} artifacts "
        f"(channel DB + queries) for {sorted(paradigms)!r} to "
        f"[path]{project_dir / 'data'}[/path]"
    )


def prune_csv_build_artifacts(project_dir: Path, channel_finder_mode: str) -> None:
    """Remove ``data/raw/`` for paradigms that have no CSV → DB build path.

    The bundled ``osprey channel-finder build-database`` tool consumes a flat
    CSV (``data/raw/address_list.csv``) and emits a flat in_context-format
    JSON. Hierarchical and middle_layer databases have a nested structure
    that the CSV format cannot express, so the ``raw/`` directory is dead
    weight in those projects.

    Args:
        project_dir: Root directory of the rendered project.
        channel_finder_mode: Paradigm selector from the build profile.

    No-op when ``channel_finder_mode == "in_context"`` or when the rendered
    project carries no ``data/raw/`` subtree.
    """
    if channel_finder_mode == "in_context":
        return

    raw_dir = project_dir / "data" / "raw"
    if not raw_dir.exists():
        return

    shutil.rmtree(raw_dir)
    console.print(
        f"  [success]✓[/success] Removed [path]{raw_dir}[/path] "
        f"(no CSV build path for {channel_finder_mode!r} paradigm)"
    )


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
