"""Project creation helpers: directory structure, services, data files.

Includes :func:`materialize_tier_artifacts`, the build-time step that picks
the tier-routed channel-database file(s) and the matching tier-routed
benchmark query file for the selected paradigm, copies them into the canonical
flat locations (``data/channel_databases/<paradigm>.json`` and
``data/benchmarks/queries.json``), and prunes the now-redundant ``tiers/`` and
``benchmarks/cross_paradigm/`` subtrees.
"""

import logging
import shutil
from pathlib import Path

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
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


def provider_api_key_entries() -> list[dict[str, str]]:
    """Provider API-key env vars for env-file templates, in registry order.

    Derived from :data:`osprey.models.provider_registry.PROVIDER_API_KEYS`
    (the single source of truth for the provider list) so that
    ``env.example.j2`` cannot drift from the real provider set. Key-less
    providers (ollama, vllm, ds4, asksage) are excluded — they have no API-key
    env var to scaffold.

    Returns:
        Ordered list of ``{"provider": <name>, "var": <ENV_VAR>}`` dicts.
    """
    from osprey.models.provider_registry import PROVIDER_API_KEYS

    return [
        {"provider": provider, "var": var}
        for provider, var in PROVIDER_API_KEYS.items()
        if var is not None
    ]


# Human-readable blurbs for the deploy-minted variables, keyed by var name.
# Prose only: the *list* of variables comes from ``_SERVICE_TOKEN_VARS``, so a
# newly minted var still reaches ``.env.example`` (named by the services that
# declare it) even with no entry here. Nothing silently drops out.
_SERVICE_TOKEN_VAR_NOTES: dict[str, str] = {
    "EVENT_DISPATCHER_TOKEN": "authenticates callers to the event-dispatcher API",
    "DISPATCH_WORKER_TOKEN": "authenticates the dispatch worker back to the dispatcher",
    "BLUESKY_LAUNCH_TOKEN": "arms the Bluesky bridge's plan-launch endpoint",
    "BLUESKY_TILED_API_KEY": "the key the bridge presents to the co-deployed Tiled catalog",
    "ZO_ROOT_USER_PASSWORD": "OpenObserve root/ingest credential",
    "ARIEL_DB_PASSWORD": "ARIEL Postgres password (also fills the agent's derived DSN)",
    "MONGO_ROOT_PASSWORD": "archiver store root password (the seeder, recorder and agent all authenticate with it)",
    "GRAPHDB_PASSWORD": "graph store password (the seeder, health check and deploy staging all authenticate with it)",
}


def service_token_var_entries() -> list[dict[str, str]]:
    """Every variable ``osprey up`` mints, for env-file templates.

    Derived from :data:`osprey.deployment.container_lifecycle._SERVICE_TOKEN_VARS`
    — the map the deploy path actually mints from — so the documented set
    cannot fall behind the minted set. A variable declared by more than one
    service (``EVENT_DISPATCHER_TOKEN``) appears once, naming both.

    Returns:
        Ordered list of ``{"var": <ENV_VAR>, "services": "<a, b>", "note":
        <blurb or "">}`` dicts, in declaration order.
    """
    from osprey.deployment.container_lifecycle import _SERVICE_TOKEN_VARS

    services_by_var: dict[str, list[str]] = {}
    for service, token_vars in _SERVICE_TOKEN_VARS.items():
        for var in token_vars:
            services_by_var.setdefault(var, []).append(service)

    return [
        {
            "var": var,
            "services": ", ".join(services),
            "note": _SERVICE_TOKEN_VAR_NOTES.get(var, ""),
        }
        for var, services in services_by_var.items()
    ]


def create_project_structure(
    template_root: Path,
    jinja_env,
    project_dir: Path,
    data_bundle: str,
    ctx: dict,
):
    """Create base project files (config, README, Dockerfile, etc.).

    No ``.env`` is written. The render is ``build/``, and the deployment's one
    secret store is the ``.env`` at the repo root — the only file compose is
    pointed at
    (``--project-directory <repo>`` + ``--env-file <repo>/.env``), the file
    ``osprey up`` mints service tokens into, and the file a ``rm -rf build/`` is
    documented not to touch. A second copy inside the render would be a second
    thing to keep in step, and one that a build could silently rewrite.

    ``.env.example`` is rendered here: it carries no values, documents
    what the repo's ``.env`` may hold, and is safe to commit.

    No ``.env.shared`` either, for the same reason as ``.env``. It is committed
    rather than secret, but it is still one of the two files the deployment's
    env chain is read from at the repo root, and a copy inside the render would
    be a second one to keep in step. ``osprey init`` authors the repo's, once.

    Args:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
        project_dir: Root directory of the rendered project
        data_bundle: Name of the data bundle (apps/ subdirectory) to use
        ctx: Template context variables
    """
    project_template_dir = template_root / "project"
    app_template_dir = template_root / "apps" / data_bundle

    # Expose claude_code.cli_version to Dockerfile.j2's CLAUDE_CLI_VERSION ARG
    # default, so the same version pin that `osprey chat`/`osprey web`
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
    data_root: Path | None = None,
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
        data_root: Resolved data tree carried by the build profile (its ``data:``
            key). When given it fully replaces the bundle's data tree — see the
            profile-mode branch below. Symlinks inside the tree are
            dereferenced into real files, matching the bundle branch: a built
            project is self-contained and must not depend on paths under the
            profile directory surviving.
    """
    # Profile-sourced data is a full replacement, not a layer: neither the
    # apps/<bundle>/data derivation nor the rglob fallback below runs, so no
    # bundle file can leak into the project alongside the facility's own tree.
    # It is content, not templates — a plain copytree, so a stray ``.j2`` lands
    # byte-identical. Rendering is not merely skipped but impossible here:
    # _copy_data_tree addresses templates by their path relative to
    # ``template_root`` through a package-rooted Jinja loader, which cannot
    # reach a tree outside the osprey package at all.
    if data_root is not None:
        dst_data = project_dir / "data"
        # dirs_exist_ok is defensive — no build path reaches here with data/ present.
        shutil.copytree(data_root, dst_data, dirs_exist_ok=True)
        logger.debug("Copied profile data files from %s to %s", data_root, dst_data)
        return

    app_template_dir = template_root / "apps" / data_bundle

    # Look for data/ subdirectory in the template
    template_data_dir = app_template_dir / "data"
    if template_data_dir.exists() and template_data_dir.is_dir():
        dst_data = project_dir / "data"
        if jinja_env is not None:
            _copy_data_tree(template_data_dir, dst_data, template_root, jinja_env, ctx)
        else:
            shutil.copytree(template_data_dir, dst_data, dirs_exist_ok=True)
        logger.debug("Copied template data files to %s", dst_data)
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
            logger.debug("Copied template data files to %s", dst_data)
            return


#: Alias of the paradigm registry in :mod:`osprey.build.build_tiers`, kept
#: under the local name this module's guard reads. Adding a paradigm to the
#: registry admits it here with no edit.
_ALL_PARADIGMS: tuple[str, ...] = VALID_CHANNEL_FINDER_MODES


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

    ``graph`` takes the query file and nothing else: its store is a seeded
    graph service, so the preset ships no ``tiers/tier{N}/graph.json`` and the
    build materializes no channel database for it. The queries still land,
    because the graph benchmark lane scores the same tier-3 ground truth as
    the file-database paradigms. Graph only ever reaches this function at the
    derived tier 3 — :func:`osprey.build.build_tiers.tier_mode_conflict`
    rejects an explicit ``tier`` paired with it.

    Facility profiles overlaying their own DB files don't care which tier
    was selected — their overlay overwrites the preset DB after this step.
    Tier itself is build-time only and is NOT written into ``config.yml``.

    Args:
        project_dir: Root directory of the rendered project.
        tier: Tier number (1 or 3) selecting the source subdirectories. Tier 1
            ships only the ``in_context`` paradigm; the build-profile validator
            rejects tier 1 paired with a non-in_context channel_finder_mode
            before this step, so a missing tier1/<paradigm>.json here is a bug.
            For ``graph`` the tier selects the query file only.
        channel_finder_mode: Paradigm selector from the build profile. Must
            be one of the paradigms in
            :data:`osprey.build.build_tiers.VALID_CHANNEL_FINDER_MODES`.

    Raises:
        ValueError: If ``channel_finder_mode`` is not a registered paradigm
            (the build-profile validator and ``manager.py`` should catch this
            earlier, but this is a defensive guard).
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
    # The paradigms that have a channel database to flatten. ``graph`` is
    # backed by a seeded graph service rather than a database file, so it
    # contributes no (src, dst) DB pair — only the query file below.
    paradigms: set[str] = set() if channel_finder_mode == "graph" else {channel_finder_mode}

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

    logger.debug(
        "Materialized tier%s artifacts for %r (channel DBs: %r) to %s",
        tier,
        channel_finder_mode,
        sorted(paradigms),
        project_dir / "data",
    )


def prune_csv_build_artifacts(project_dir: Path, channel_finder_mode: str) -> None:
    """Remove ``data/raw/`` for paradigms that have no CSV → DB build path.

    The bundled ``osprey channel-finder build-database`` tool consumes a flat
    CSV (``data/raw/address_list.csv``) and emits a flat in_context-format
    JSON. Hierarchical and middle_layer databases have a nested structure
    that the CSV format cannot express, and a graph build has no database file
    at all, so the ``raw/`` directory is dead weight in those projects.

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
    logger.debug("Removed %s (no CSV build path for %r paradigm)", raw_dir, channel_finder_mode)
