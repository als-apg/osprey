"""Project creation helpers: directory structure, services, data files.

Includes :func:`materialize_benchmark_queries`, the build-time step that copies
the selected mode's benchmark query file to ``data/benchmarks/queries.json``
and removes the facility tree's channel-finder staging subtrees from the
render.
"""

import logging
import shutil
from pathlib import Path

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
#
# The value is npm's `stable` dist-tag for @anthropic-ai/claude-code, never
# `latest` or `next`; read the channel with
# `npm view @anthropic-ai/claude-code dist-tags`.
_DEFAULT_CLAUDE_CLI_VERSION = "2.1.267"

CONFIG_TEMPLATE = "config.yml.j2"
"""The project-file template that renders ``config.yml``."""


def _default_cli_version(ctx: dict) -> None:
    """Expose ``claude_code.cli_version`` to Dockerfile.j2's CLAUDE_CLI_VERSION
    ARG default, so the same version pin that ``osprey chat``/``osprey web``
    honor at runtime (osprey.agent_runner.launcher) also pins the image's
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

    Derived from the provider registry, where a built-in's entry names its
    variable and a registered provider's class does, so ``env.example.j2``
    cannot drift from the providers the registry holds. Built-ins come first in
    table order, then registrations. Keyless providers (ollama, vllm, ds4) are
    left out: they have no API-key env var to scaffold.

    Returns:
        Ordered list of ``{"provider": <name>, "var": <ENV_VAR>}`` dicts.
    """
    from osprey.models.provider_registry import get_provider_registry

    return [
        {"provider": provider, "var": var}
        for provider, var in get_provider_registry().api_key_env_vars().items()
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
        except Exception:  # see comment above
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
    "ARIEL_DB_READONLY_PASSWORD": (
        "password of the SELECT-only Postgres role the agent's SQL tool queries through"
    ),
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
    template_root: Path,  # noqa: ARG001 - template-copier signature; a profile's own data tree is copied verbatim
    project_dir: Path,
    package_name: str,  # noqa: ARG001 - template-copier signature; a profile's own data tree is copied verbatim
    data_bundle: str,  # noqa: ARG001 - template-copier signature; a profile's own data tree is copied verbatim
    ctx: dict,  # noqa: ARG001 - template-copier signature; a profile's own data tree is copied verbatim
    jinja_env=None,  # noqa: ARG001 - template-copier signature; a profile's own data tree is copied verbatim
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


#: The staging subtrees a facility data tree may carry that no render needs:
#: the per-tier channel databases and benchmark query sources, and the raw
#: inputs they were generated from. Relative to the render's ``data/``.
_RENDER_EXCLUDED_DATA_DIRS: tuple[tuple[str, ...], ...] = (
    ("benchmarks", "cross_paradigm"),
    ("channel_databases", "tiers"),
    ("raw",),
)


def materialize_benchmark_queries(project_dir: Path, channel_finder_mode: str) -> None:
    """Copy the mode's benchmark query file into place and prune the staging trees.

    The facility tree ships its benchmark query sources under
    ``data/benchmarks/cross_paradigm/queries/``: ``tier1_queries.json`` for
    ``in_context`` and ``tier3_queries.json`` for every other mode. The selected
    one is copied to ``data/benchmarks/queries.json``. Then
    ``data/benchmarks/cross_paradigm/``, ``data/channel_databases/tiers/`` and
    ``data/raw/`` are removed from the render; the facility tree they were
    copied from is never touched. Each channel-finder index is the view the
    build writes at its own path, so nothing is flattened here.

    A render whose tree ships no ``data/benchmarks/cross_paradigm/`` subtree
    gets no ``queries.json``.

    Args:
        project_dir: Root directory of the rendered project.
        channel_finder_mode: Channel-finder mode from the build profile.

    Raises:
        FileNotFoundError: If the tree ships query sources but none for this
            mode. Raised before anything is copied or removed.
    """
    data_dir = project_dir / "data"
    queries_root = data_dir / "benchmarks" / "cross_paradigm" / "queries"
    if queries_root.exists():
        source_name = (
            "tier1_queries.json" if channel_finder_mode == "in_context" else "tier3_queries.json"
        )
        queries_src = queries_root / source_name
        if not queries_src.exists():
            raise FileNotFoundError(
                f"Benchmark queries file not found: {queries_src} "
                f"(channel_finder_mode={channel_finder_mode!r})"
            )
        queries_dst = data_dir / "benchmarks" / "queries.json"
        shutil.copy2(queries_src, queries_dst)
        logger.debug("Copied benchmark queries %s to %s", queries_src, queries_dst)

    for parts in _RENDER_EXCLUDED_DATA_DIRS:
        excluded = data_dir.joinpath(*parts)
        if excluded.exists():
            shutil.rmtree(excluded)
            logger.debug("Removed %s from the render", excluded)
