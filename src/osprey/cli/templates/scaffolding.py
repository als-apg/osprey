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
_DEFAULT_CLAUDE_CLI_VERSION = "2.1.258"

CONFIG_TEMPLATE = "config.yml.j2"
"""The project-file template that renders ``config.yml``."""


def _default_cli_version(ctx: dict) -> None:
    """Expose ``claude_code.cli_version`` to Dockerfile.j2's CLAUDE_CLI_VERSION
    ARG default, so the same version pin that ``osprey chat``/``osprey web``
    honor at runtime (osprey.utils.claude_launcher) also pins the image's
    build-time CLI install. Callers may pre-populate
    ``ctx["claude_code_cli_version"]`` (flat) or ``ctx["claude_code"]["cli_version"]``
    (nested, mirroring config.yml's shape); absent either, fall back to the
    framework's last verified pin."""
    if "claude_code_cli_version" not in ctx:
        ctx["claude_code_cli_version"] = (
            ctx.get("claude_code", {}).get("cli_version") or _DEFAULT_CLAUDE_CLI_VERSION
        )


def project_template_for(template_root: Path, template_file: str) -> str | None:
    """The template that renders project file *template_file*.

    Every project file renders from the shared ``project/`` directory: the
    framework template is the only ``config.yml.j2``, and what a project
    deploys is spelled by its profile's ``config:`` block, not by a per-app
    copy of the template.

    Returns:
        The template's path relative to the templates root, in the spelling the
        Jinja environment loads by, or ``None`` when ``project/`` does not ship
        the file.
    """
    if (template_root / "project" / template_file).exists():
        return f"project/{template_file}"
    return None


def render_project_config(
    template_root: Path,
    jinja_env,
    output_path: Path,
    ctx: dict,
) -> None:
    """Render ``config.yml`` alone — the one file of a project render that
    says what the project deploys and where — with the same template and
    context :func:`create_project_structure` uses.

    Args:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
        output_path: Where the rendered ``config.yml`` is written
        ctx: Template context variables

    Raises:
        ValueError: If ``project/`` ships no ``config.yml.j2``.
    """
    _default_cli_version(ctx)
    template_path = project_template_for(template_root, CONFIG_TEMPLATE)
    if template_path is None:
        raise ValueError(f"{template_root / 'project'} renders no config.yml")
    render_template(jinja_env, template_path, ctx, output_path)


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


def provider_base_url_entries() -> list[dict[str, str]]:
    """Env vars naming an endpoint the deployment has to supply, in registry order.

    A provider that requires a ``base_url`` and ships no ``default_base_url``
    has no host to fall back to: the gateway it fronts is the deployment's own,
    and a launch that names none is refused. Those are the variables whose
    absence stops a deployment, so ``.env.example`` lists them beside the keys
    rather than leaving them to the documentation.

    Derived from the provider classes themselves — the same three attributes
    the launch paths read — so the file cannot drift from what is actually
    required. Providers with a working default (cborg, argo, ollama, …) are
    excluded: setting their variable redirects them, it does not enable them.

    Returns:
        List of ``{"provider": <name>, "var": <ENV_VAR>}`` dicts, in the
        registry's own (alphabetical) provider order. Empty when every provider
        ships an endpoint.
    """
    from osprey.models.provider_registry import get_provider_registry

    registry = get_provider_registry()
    entries: list[dict[str, str]] = []
    for provider in registry.list_providers():
        # A provider whose module will not import is not this function's
        # problem — the registry reports that where it is actionable.
        try:
            cls = registry.get_provider(provider)
        except Exception:  # noqa: BLE001 - see comment above
            continue
        if cls is None:
            continue
        if not cls.requires_base_url or cls.default_base_url:
            continue
        if not cls.base_url_env_var:
            continue
        entries.append({"provider": provider, "var": cls.base_url_env_var})
    return entries


# Human-readable blurbs for the deploy-minted variables, keyed by var name.
# Prose only: the *list* of variables comes from ``_SERVICE_TOKEN_VARS``, so a
# newly minted var still reaches ``.env.example`` (named by the services that
# declare it) even with no entry here. Nothing silently drops out.
_SERVICE_TOKEN_VAR_NOTES: dict[str, str] = {
    "EVENT_DISPATCHER_TOKEN": "authenticates callers to the event-dispatcher API",
    "DISPATCH_WORKER_TOKEN": "authenticates the dispatch worker back to the dispatcher",
    "BLUESKY_LAUNCH_TOKEN": "arms the Bluesky bridge's plan-launch endpoint",
    "BLUESKY_TILED_API_KEY": "the key the bridge presents to the co-deployed Tiled catalog",
    # The opt-in SECOND plan lane's own launch token. One per control-system
    # target a lane can serve, so three names — but a deployment renders at most
    # one second lane, so at most one of them exists in any repo's `.env`, and
    # none at all without `bluesky.second_lane`: a lane is named for the target
    # it serves, and which target the second lane serves depends on which
    # machine the deployment baseline is. All three are documented anyway,
    # because this file lists what a repo's `.env` MAY hold rather than what
    # this one deployment does.
    "BLUESKY_VA_LAUNCH_TOKEN": (
        "arms the plan-launch endpoint of the second Bluesky lane, the one serving the "
        "virtual accelerator (only on a deployment with `bluesky.second_lane`)"
    ),
    "BLUESKY_LIVE_LAUNCH_TOKEN": (
        "arms the plan-launch endpoint of the second Bluesky lane, the one serving the live "
        "machine (only on a deployment with `bluesky.second_lane`)"
    ),
    "BLUESKY_STANDIN_LAUNCH_TOKEN": (
        "arms the plan-launch endpoint of the Bluesky lane serving the live stand-in soft "
        "IOC (only on a deployment with `bluesky.second_lane`)"
    ),
    "OSPREY_TERMINAL_SECRET": "the operator login secret for the bluesky-web panel's web gate",
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
        ctx: Template context variables
    """
    project_template_dir = template_root / "project"

    _default_cli_version(ctx)

    # Render template files (no pyproject.toml or requirements.txt -- no src/ package)
    files_to_render = [
        (CONFIG_TEMPLATE, "config.yml"),
        ("env.example.j2", ".env.example"),
        ("README.md.j2", "README.md"),
        # Reference container image — rendered once at build; regen never touches it
        ("Dockerfile.j2", "Dockerfile"),
    ]

    # Copy static files
    static_files = [
        # requirements.txt moved to rendered templates to handle {{ framework_version }}
        #
        # The container entrypoint. Copied rather than rendered: it derives the
        # render it maintains from its own location, so there is nothing
        # project-specific to substitute — and a file with no Jinja in it is one
        # fewer thing that can be broken by a context key going missing. It
        # lands in the render, beside the Dockerfile that installs it, because
        # that is what makes it part of the deployment the image copies in
        # rather than a sidecar the build has to remember to carry.
        ("entrypoint.sh", "entrypoint.sh"),
    ]

    for template_file, output_file in files_to_render:
        template_path = project_template_for(template_root, template_file)
        if template_path is not None:
            render_template(jinja_env, template_path, ctx, project_dir / output_file)

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


def copy_template_data(
    template_root: Path,
    project_dir: Path,
    package_name: str,
    data_bundle: str,
    ctx: dict,
    jinja_env=None,
    data_root: Path | None = None,
):
    """Copy the profile's data tree to the project root (no src/ package).

    Data files (channel databases, channel_limits.json, logbook seeds,
    benchmark datasets) are placed at ``project_dir/data/``. The profile's
    ``data:`` tree is the only source: it is copied verbatim, so a stray
    ``.j2`` file lands byte-identical rather than being rendered.

    Args:
        template_root: Path to osprey's bundled templates directory
        project_dir: Root directory of the project
        package_name: Python package name (unused; kept for the caller's shape)
        data_bundle: Name of the data bundle the project's other packaged
            trees come from (unused here)
        ctx: Template context variables
        jinja_env: Optional Jinja2 environment (unused here)
        data_root: Resolved data tree carried by the build profile (its
            ``data:`` key), required. Symlinks inside the tree are
            dereferenced into real files: a built project is self-contained
            and must not depend on paths under the profile directory
            surviving.

    Raises:
        ValueError: If ``data_root`` is None. Every profile declares ``data:``
            (``BuildProfile.validate`` requires it), so there is no packaged
            tree to fall back to.
    """
    # The facility's tree is the whole of the project's data/: nothing reads
    # `apps/<bundle>/data` here any more, so no packaged file can land beside
    # what the profile ships. It is content, not templates — a plain copytree,
    # so a stray `.j2` lands byte-identical. Rendering is not merely skipped
    # but impossible: the package-rooted Jinja loader addresses templates by
    # their path relative to ``template_root`` and cannot reach a tree outside
    # the osprey package at all.
    if data_root is None:
        raise ValueError(
            "copy_template_data requires the profile's resolved `data:` tree; "
            "there is no packaged data bundle to fall back to. Pass "
            "data_root=BuildProfile.resolved_data_root(profile_dir)."
        )

    from osprey.utils.workspace import RUNTIME_DATA_DIR_NAME

    dst_data = project_dir / "data"

    def _drop_runtime_output(directory: str, names: list[str]) -> set[str]:
        # data/.runtime/ is runtime-minted material (`osprey up`'s CURVE
        # certificates) — private keys that must not be staged into
        # build/ or into the images built from it. Top level only, the
        # same anchoring the fingerprint fold applies to the same name.
        if Path(directory) == Path(data_root) and RUNTIME_DATA_DIR_NAME in names:
            return {RUNTIME_DATA_DIR_NAME}
        return set()

    # dirs_exist_ok is defensive — no build path reaches here with data/ present.
    shutil.copytree(data_root, dst_data, dirs_exist_ok=True, ignore=_drop_runtime_output)
    logger.debug("Copied profile data files from %s to %s", data_root, dst_data)


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
