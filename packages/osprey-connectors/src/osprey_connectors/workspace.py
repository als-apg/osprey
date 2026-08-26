"""Cross-layer workspace and config resolution utilities.

These functions are used across the ``interfaces/``, ``cli/``, ``services/``,
and ``mcp_server/`` layers, so they live in ``utils/`` — below all of them —
rather than inside any one layer.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey_connectors.dotenv import ENV_CHAIN_FILENAMES

logger = logging.getLogger("osprey_connectors.workspace")

#: Zone directory names inside a deployment repo. ``build/`` is the render zone,
#: re-created wholesale by every ``osprey build``; ``var/`` is the durable state
#: zone that survives it. Spelled here, in ``utils/``, because the runtime layers
#: resolve paths against them and must not import ``cli``.
BUILD_DIR_NAME = "build"
STATE_DIR_NAME = "var"

#: Where the build stages the container-destined copy of a project, under the
#: render zone. Each is a deployment REPO — ``profile.yml`` and the source zone
#: at its root, its own render below in ``build/`` — rendered against the
#: ``/app/<project>`` path a container sees rather than the building host's, and
#: it is what an image is built from. Spelled here beside the zone names because
#: the build writes it and the deploy layer builds from it, and the two must not
#: disagree about where it is.
IMAGE_DIR_NAME = ".image"

#: Reserved runtime-output directory directly under a repo's ``data/`` tree:
#: ``data/.runtime/``. The one place ``osprey up`` may write inside the
#: otherwise build-owned data tree — today the per-lane Bluesky CURVE
#: certificate sets, ``data/.runtime/<lane>_curve/``. The build treats it as
#: not existing: the profile-material fold never hashes it (a successful start
#: must not mark its own build OUT OF DATE and strand the scaffolded boot
#: unit at the drift gate, #716) and the data copy never stages it (secrets
#: must not land in ``build/`` or in images). Spelled here beside the zone
#: names because the writer lives in ``deployment/`` and the fold in ``cli/``,
#: and the two must agree on these exact bytes.
RUNTIME_DATA_DIR_NAME = ".runtime"

#: The rendered config, relative to the repo root. It is an *output*: the source
#: of truth is ``profile.yml`` at the repo root, and this file is what a build
#: derives from it. Every producer and consumer of that path spells it through
#: this constant so the two cannot drift.
RENDERED_CONFIG_RELPATH = f"{BUILD_DIR_NAME}/config.yml"

#: Agent-data root used when a config declares no ``agent_data.base_dir``.
#: Under ``var/`` rather than beside the render: memory, sessions, and artifacts
#: have to survive ``rm -rf build/``, which is the one guarantee the build zone
#: does not offer.
DEFAULT_AGENT_DATA_BASE_DIR = f"{STATE_DIR_NAME}/agent_data"

#: The safety audit log, relative to the repo root: what the agent was asked to
#: do and what it did. Durable like the agent-data root beside it, and kept by
#: ``osprey reset`` unless ``--purge-audit`` is passed. Named here rather than
#: spelled at each end because the directory the reset command promises to keep
#: has to be the directory the refusal recorder writes into.
AUDIT_DIR_RELPATH = f"{STATE_DIR_NAME}/audit"

#: The two directories that make up the state zone, created empty and otherwise
#: the agent's to write. Both ``osprey init`` and ``osprey build`` guarantee they
#: exist — a fresh clone carries no git-ignored directory, so whichever command
#: runs first has to make them, and a repo that was reset must end up looking
#: like one that was freshly cloned. Spelled once for exactly that reason: two
#: commands agreeing by coincidence is how a reset and a clone start to differ.
STATE_ZONE_DIRS: tuple[str, ...] = (DEFAULT_AGENT_DATA_BASE_DIR, AUDIT_DIR_RELPATH)

#: Subdirectory of the agent-data root holding the simulation's mutable
#: ``active_scenarios`` state file.
SIMULATION_STATE_DIR_NAME = "simulation"

#: Config key naming that directory explicitly (relative paths resolve against
#: the project root). Unset — the normal case — puts it under the agent-data
#: root. It lives here, rather than in the simulation package, so
#: :data:`osprey_connectors.config.RUNTIME_WRITE_PATH_KEYS` and
#: :func:`osprey_connectors.simulation.engine.resolve_state_dir` share one spelling without
#: ``config`` having to import the engine (which would pull numpy into every
#: config load).
SIMULATION_STATE_DIR_CONFIG_KEY = "simulation.state_dir"


def dotted_config_str(config: Mapping[str, Any] | None, key: str) -> str | None:
    """Read a dotted config key, or ``None`` if it does not name a real string.

    ``None`` is returned for every way the key can fail to answer — a missing
    segment, a segment that is not a mapping, a non-string value, or an empty
    one — because each of those means the same thing to every caller: the key
    is unset, use the default. A mistyped key therefore falls through to the
    default rather than raising.

    The single reader for the path-shaped config keys, so the checks that
    *reject* a value and the resolvers that *use* it cannot disagree about what
    the config says — including on ``dict`` versus ``Mapping``, a difference
    that stays invisible only while every config arrives straight from the YAML
    loader.

    Args:
        config: Loaded config mapping, or ``None``.
        key: Dotted path, e.g. ``simulation.state_dir``.
    """
    value: Any = config or {}
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value if isinstance(value, str) and value else None


def anchored_path(value: str, project_root: Path) -> Path:
    """Expand ``~`` in a configured path and anchor a relative one at the project.

    The other half of :func:`dotted_config_str`: a configured path is absolute,
    home-relative, or project-relative, and every consumer has to make the same
    three-way choice before comparing paths or creating directories.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(project_root) / path


def rendered_config_path(repo_root: Path | str) -> Path:
    """Path to the rendered ``config.yml`` of the repo rooted at *repo_root*.

    The one place the ``build/config.yml`` spelling is assembled, so that a
    caller holding a repo root never has to know which zone the render lands in.
    """
    return Path(repo_root) / RENDERED_CONFIG_RELPATH


def container_image_context(repo_root: Path | str, project_name: str) -> Path:
    """The directory an image for *project_name* is built from.

    The build writes it (``build_cmd._render_container_project``) and the deploy
    layer builds from it, so the path is assembled here rather than spelled at
    both ends. It is a deployment repo in its own right: the ``Dockerfile`` a
    build uses lives at :data:`RENDERED_CONFIG_RELPATH`'s sibling inside it, not
    at its root, because the root is the repo and the render is one level down.
    """
    return Path(repo_root) / BUILD_DIR_NAME / IMAGE_DIR_NAME / project_name


def agent_data_base_dir(config: Mapping[str, Any] | None) -> str:
    """Read the agent-data root out of an already-loaded config mapping.

    ``agent_data.base_dir`` is the only key that names this directory. Callers
    holding a config dict read it through here and anchor the result themselves
    (the health checks and the compose generator anchor on ``project_root``);
    callers with no config in hand use :func:`resolve_agent_data_root`, which
    loads the config and anchors on the same ``project_root``.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.

    Returns:
        The configured base directory, possibly relative to the caller's anchor.
    """
    section = (config or {}).get("agent_data") or {}
    if not isinstance(section, Mapping):
        return DEFAULT_AGENT_DATA_BASE_DIR
    return str(section.get("base_dir") or DEFAULT_AGENT_DATA_BASE_DIR)


def resolve_simulation_state_dir(config: Mapping[str, Any] | None, project_root: Path) -> Path:
    """Resolve the directory holding the mutable ``active_scenarios`` state file.

    The state file is the one piece of simulation state that changes after a
    build, so it lives under the agent-data root rather than next to
    ``machine.json``: everything in a project's ``data/`` tree is build-owned
    and checksummed (see
    :func:`osprey.cli.templates.manifest.calculate_file_checksums`), and a
    scenario switch is not project drift.

    Lives here rather than in the simulation package so the three sides that
    must agree — the engine that writes the file, the compose generator that
    renders the container's bind-mount source, and the build injector that
    pre-creates it — resolve it identically without ``deployment`` importing
    numpy through :mod:`osprey_connectors.simulation.engine`. Re-exported there as
    ``resolve_state_dir``.

    A mistyped :data:`SIMULATION_STATE_DIR_CONFIG_KEY` (anything but a non-empty
    string) falls through to the default rather than raising, matching how
    :func:`osprey_connectors.config.find_runtime_write_paths_under_data` reads the
    same key.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.
        project_root: Root the relative paths above resolve against.

    Returns:
        Absolute path to the state directory (not created here).
    """
    config = config or {}
    configured = dotted_config_str(config, SIMULATION_STATE_DIR_CONFIG_KEY)
    relative = configured or f"{agent_data_base_dir(config)}/{SIMULATION_STATE_DIR_NAME}"
    return anchored_path(relative, project_root)


def resolve_config_path() -> Path:
    """Resolve the path to the rendered ``config.yml``.

    Resolution order:
      1. ``OSPREY_CONFIG`` environment variable (with shell variable expansion) —
         what every deployed process actually gets, set by the CLI on the host
         and by the compose templates in every container.
      2. ``build/config.yml`` under the current working directory — standing in
         a repo root with a render in it.
      3. ``config.yml`` under the current working directory — a container
         project directory, whose root *is* the render, and the path reported
         when nothing exists at all.

    ``CONFIG_FILE`` is deliberately NOT read here, even though
    :func:`osprey.agent_runner.artifact_resolve.deployed_config_path` reads it
    ahead of ``OSPREY_CONFIG``. That helper answers for one deployed process
    whose compose service sets the variable; this one answers for the framework
    everywhere, and ``CONFIG_FILE`` is a name generic enough that honoring it
    globally would let an unrelated export redirect every OSPREY process on the
    host. Where both apply — inside the dispatch worker — they agree by
    construction rather than by precedence: the compose sets ``CONFIG_FILE`` to
    ``/app/<project>/build/config.yml``, and the image's ``WORKDIR`` is
    ``/app/<project>``, so step 2 above finds that same file.
    """
    import os

    configured = os.environ.get("OSPREY_CONFIG")
    if configured:
        return Path(os.path.expandvars(configured))

    cwd = Path.cwd()
    rendered = rendered_config_path(cwd)
    return rendered if rendered.is_file() else cwd / "config.yml"


def load_osprey_config() -> dict:
    """Load OSPREY configuration (delegates to ConfigBuilder singleton).

    Delegates to the framework's ``ConfigBuilder`` so that ``${VAR:-default}``
    environment-variable placeholders are resolved consistently.

    Resolution order:
      1. ``OSPREY_CONFIG`` environment variable
      2. ``./config.yml`` relative to the current working directory

    Returns:
        Parsed YAML dict (with env vars resolved), or empty dict if the file is missing.
    """
    config_path = resolve_config_path()
    try:
        from osprey_connectors.config import get_config_builder

        builder = get_config_builder(config_path=str(config_path), set_as_default=True)
        return builder.raw_config
    except (FileNotFoundError, Exception):
        return {}


def reset_config_cache() -> None:
    """Clear all config caches — used between tests.

    The warn-once latch on the facility-timezone fallback is cleared with them.
    It is process state in exactly the way a cache is: left set, whichever test
    happened to run first would consume the single warning and every later test
    asserting on it would see none — an order-dependent pass that says nothing
    about the code.
    """
    from osprey_connectors import config as config_module

    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()
    config_module._tz_fallback_warned = False


def repo_root_for_config(config_path: Path | str) -> Path:
    """The repo root that a config file at *config_path* belongs to.

    The rendered config lives at ``<repo>/build/config.yml``, so the repo is
    two levels up — except in a container, whose project directory *is* the
    render and holds ``config.yml`` at its root. One helper for both, so the
    "is this the build zone?" question is asked in exactly one place.

    Known non-goal: a FLAT project whose own directory is literally named
    ``build`` is read as a render, and its parent is returned. Accepted
    — every runtime path derives the root through this helper, so a regen or a
    status check agreeing with the runtime beats a special case that would make
    the two disagree.
    """
    parent = Path(config_path).parent
    return parent.parent if parent.name == BUILD_DIR_NAME else parent


def deployment_env_path(config_path: Path | str) -> Path:
    """The ``.env`` that belongs to the deployment whose config is *config_path*.

    There is one ``.env`` per deployment and it sits at the repo ROOT, beside
    ``profile.yml`` — never in the render, which every ``osprey build``
    re-creates from scratch. So the config's own directory is the wrong answer
    on a host, where the config is ``<repo>/build/config.yml``, and the right
    one in a container, whose project directory *is* the render.

    Both are covered by one rule: prefer the repo root, fall back to the
    config's directory when nothing is there. The fallback is what keeps the
    container case working without a second rule.

    This answers for the SINGLE local file only. A consumer that reads or
    watches the deployment's whole environment wants
    :func:`deployment_env_chain`, which applies the same anchoring to every
    chain member — ``.env.shared`` included.
    """
    root_env = repo_root_for_config(config_path) / ".env"
    if root_env.is_file():
        return root_env
    return Path(config_path).parent / ".env"


def deployment_env_chain(config_path: Path | str) -> list[Path]:
    """Every env-chain file of the deployment whose config is *config_path*.

    The chain counterpart of :func:`deployment_env_path`, with the same
    root-then-container anchoring: the repo root's chain when any member exists
    there, else the config's own directory (the container case, whose project
    directory *is* the render). Paths come back in ascending precedence —
    ``.env.shared`` before ``.env`` — so an ``override=True`` loader can walk
    the list as-is and land on local-wins.

    Every member path is returned whether or not the file exists. A watcher
    has to stat the places a member could *appear*, or a ``.env`` created
    after start-up never invalidates anything — the same reason
    :func:`deployment_env_path` is spelled once for its readers and watchers.

    Spelled here rather than at each caller because the callers that ask this
    have to agree exactly — a reader that loads one set of files and a
    signature that stats another produces a cache which never invalidates, and
    reports nothing at all while doing it.
    """
    root = repo_root_for_config(config_path)
    if any((root / name).is_file() for name in ENV_CHAIN_FILENAMES):
        base = root
    else:
        base = Path(config_path).parent
    return [base / name for name in ENV_CHAIN_FILENAMES]


def repo_root_for_agent_data(
    agent_data_root: Path | str, base_dir: str = DEFAULT_AGENT_DATA_BASE_DIR
) -> Path:
    """The repo root an agent-data root sits under.

    Stores hand agent code *relative* file pointers, and agent code runs with
    its working directory at the repo root (see the python executor's
    ``_resolve_project_root``), so the two have to be anchored at the same
    place. Taking the agent-data root's parent was that anchor only while the
    directory sat exactly one level down; ``var/agent_data`` is two, and a
    pointer anchored one level short resolves to nothing from the agent's cwd.

    So: strip the configured base directory off the tail when it is there, and
    fall back to the parent when it is not — which is what a project that
    relocated its data root to a single-segment directory still wants.

    An ABSOLUTE ``base_dir`` is the case the tail rule cannot serve, and the
    guard below is what keeps it from being an exception instead of an answer.
    Such a root is not under any repo (``anchored_path`` returns it verbatim),
    so no repo-relative pointer to it exists; its parts also begin with the
    filesystem root, which makes ``tail`` match the whole of ``root.parts`` and
    ``parents[len(tail) - 1]`` index past the end. Requiring a strictly longer
    root sends that case to the parent fallback — the answer this helper gave
    before it learned about ``base_dir`` at all.

    Be precise about what that fallback does, because the obvious guess is
    wrong: the parent is still an ANCESTOR of everything under the root, so a
    caller's ``relative_to`` SUCCEEDS against it and yields a wrong-but-plausible
    pointer (``agent/artifacts/x.json`` for a root of ``/data/agent``). It does
    not raise, and no defensive bare-filename branch fires. That is the
    committed baseline's behavior, preserved deliberately — the alternative
    would be inventing an answer for a layout that has none.

    Args:
        agent_data_root: Resolved agent-data root, e.g. ``<repo>/var/agent_data``.
        base_dir: The ``agent_data.base_dir`` that produced it.
    """
    root = Path(agent_data_root)
    tail = Path(base_dir).parts
    if tail and len(root.parts) > len(tail) and root.parts[-len(tail) :] == tail:
        return root.parents[len(tail) - 1]
    return root.parent


def resolve_project_root(config: Mapping[str, Any] | None = None) -> Path:
    """Resolve the repo root that relative configured paths anchor against.

    ``project_root`` names the deployment repo — the directory holding
    ``profile.yml`` — and not the directory holding ``config.yml``: the render
    lives one level down, in ``build/``, so anchoring on the config file's own
    parent would put every relative path inside the disposable zone.

    Resolution order:
      1. The config's own ``project_root`` key, when it names a directory that
         exists here. That qualifier is what makes it right inside a container,
         where the repo was built at one path and runs at another: a deployed
         service reads a config STAGED on the host — the compose generator
         flattens the host render into ``build/services/<svc>/config.yml``, and
         the service's compose file bind-mounts that over the one in the image —
         so its ``project_root`` is a host path that names nothing in here.
         Anchoring on it would put the
         agent-data root, and every other relative path, outside the mounted
         volume. Same rule and same reason as
         :func:`osprey.deployment.compose_generator.resolve_repo_root`.
      2. The repo the resolved config path sits in — its parent, or that
         parent's parent when the config is in the ``build/`` zone. Filesystem
         truth, and correct in every container layout: the mount target names
         the render, so unwrapping it names the repo.
      3. The configured value after all, when there is no config file to derive
         from — a build rendered with ``--runtime-root`` records a path that
         exists only on the machine it will run on, and has nothing better.
      4. The current working directory, when there is neither.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None`` to consult only the
            config *path* (callers that have not loaded one).
    """
    configured = dotted_config_str(config, "project_root")
    candidate = Path(configured).expanduser() if configured else None
    if candidate is not None and candidate.is_dir():
        return candidate

    config_path = resolve_config_path()
    if config_path.exists():
        return repo_root_for_config(config_path)
    return candidate if candidate is not None else Path.cwd()


def resolve_agent_data_root() -> Path:
    """Resolve the agent data root directory from config.

    Uses ``agent_data.base_dir`` from config.yml, anchored on the repo root that
    :func:`resolve_project_root` resolves. Falls back to
    :data:`DEFAULT_AGENT_DATA_BASE_DIR` relative to that root when the config
    declares no base directory.
    """
    config = load_osprey_config()
    resolved = anchored_path(agent_data_base_dir(config), resolve_project_root(config)).resolve()

    import os

    session_id = os.environ.get("OSPREY_SESSION_ID")
    if session_id:
        resolved = resolved / "sessions" / session_id

    logger.debug("Agent data root resolved to %s", resolved)
    return resolved


def resolve_shared_data_root() -> Path:
    """Resolve the agent data root WITHOUT session-path isolation.

    Use for stores whose data must be visible to long-lived daemons
    (gallery, ARIEL) that run outside any specific session.  Logical
    session isolation is handled at the index level via entry metadata
    (e.g. ``ArtifactEntry.session_id``).
    """
    config = load_osprey_config()
    resolved = anchored_path(agent_data_base_dir(config), resolve_project_root(config)).resolve()
    logger.debug("Shared data root resolved to %s", resolved)
    return resolved


# Backward-compatible alias
resolve_workspace_root = resolve_agent_data_root


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the project root from config.

    Absolute paths are returned as-is. Relative paths are resolved
    against ``project_root`` from the active configuration.

    Args:
        path_str: Path string (absolute or relative to project root)

    Returns:
        Resolved absolute Path object
    """
    from osprey_connectors.config import get_config_builder

    config_builder = get_config_builder()
    project_root = Path(config_builder.get("project_root"))
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path
