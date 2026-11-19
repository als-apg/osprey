"""Configuration: one loader, the config it loads, and the process default.

:func:`load_config` reads one rendered ``config.yml`` into a
:class:`LoadedConfig`: the file's tree with ``${VAR}`` placeholders resolved,
the tree as written, the repo root it belongs to, the placeholders no variable
resolved, and the facility zone. It reads the file and nothing else: it neither
reads ``.env`` nor writes ``os.environ``, and a failure is a
:class:`~osprey_connectors.errors.ConfigLoadError` naming the path.

The process default is one explicit object. :func:`current_config` answers
every unqualified lookup: the config a :func:`using_config` block names, else
the one an entry point installed with :func:`install_config`, else the file
:func:`locate_config` finds, loaded and kept until the file changes.
:func:`active_config` asks the same question without loading anything.

**Importing this module does not read ``.env`` and does not touch
``os.environ``.** The one lookup that does is the lazy default, which applies
the working directory's env chain once before it loads, as every config load
did before there was an explicit one: an application asks for a config, never
the mere act of importing an ``osprey`` module. Loading ``.env`` at import
time made a library consumer's environment depend on which directory the
process happened to start in, gave non-credential keys ``override=True``
semantics they were never scoped for, and silently undid callers' own
``os.environ`` writes depending on import order.

Applications load ``.env`` explicitly at startup: the CLI in
``osprey.cli.main``, MCP servers via :func:`osprey.mcp_env.load_dotenv_from_project`,
and the Claude Code launch paths via
:func:`osprey.agent_runner.provider_env.inject_provider_env`.
"""

import copy
import logging
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import NoReturn, overload
from zoneinfo import ZoneInfo

import yaml
from pydantic import PrivateAttr
from typing_extensions import override

# Safe at module level despite this module's no-osprey-imports rule below:
# ``workspace`` and ``yaml_loader`` import nothing from osprey itself, so
# there is no cycle.
from osprey_connectors import yaml_loader
from osprey_connectors._config_read import config_flag, is_unresolved_placeholder
from osprey_connectors._contract_types import ContractAny
from osprey_connectors.config_sections import (
    AgentDataSection,
    ArchiverSection,
    ControlSystemSection,
    LoggingSection,
    PvDirectorySection,
    SimulationSection,
    SystemSection,
)
from osprey_connectors.errors import (
    ConfigLoadError,
    ConfigNotFoundError,
    ConfigParseError,
    ConfigurationError,
)
from osprey_connectors.model import FrozenModel, FrozenTuple
from osprey_connectors.workspace import (
    DEFAULT_AGENT_DATA_BASE_DIR,
    SIMULATION_STATE_DIR_CONFIG_KEY,
    agent_data_base_dir,
    anchored_path,
    rendered_config_path,
    repo_root_for_config,
    resolve_simulation_state_dir,
)

__all__ = [
    "LoadedConfig",
    "UnresolvedRef",
    "active_config",
    "config_flag",
    "config_section",
    "connector_block",
    "current_config",
    "facility_timezone",
    "install_config",
    "is_unresolved_placeholder",
    "load_config",
    "localize_facility",
    "locate_config",
    "mapping_or_empty",
    "resolve_env_vars",
    "subsection",
    "unresolved_placeholders",
    "using_config",
]

# Use standard logging (not get_logger) to avoid circular imports with logger.py
# The short name 'CONFIG' enables easy filtering: quiet_logger(['registry', 'CONFIG'])
logger = logging.getLogger("CONFIG")


# The data comes back in the shape it went in, a YAML parse's untyped tree as a
# rule, and every caller reads the shape its own config has.
def resolve_env_vars(data: object, *, environ: "Mapping[str, str] | None" = None) -> ContractAny:
    """Recursively resolve environment variables in configuration data.

    Supports both simple and bash-style default value syntax:
    - ``${VAR_NAME}`` — simple substitution
    - ``${VAR_NAME:-default_value}`` — with default value
    - ``$VAR_NAME`` — simple substitution without braces

    Placeholders inside ``claude_code.servers.*.env`` blocks are preserved
    verbatim: Claude Code expands ``${VAR}`` in ``.mcp.json`` env at MCP
    server *launch* time, so expanding here would bake either secrets or
    empty strings (when vars are unset, e.g. in CI image builds) into build
    artifacts.

    :func:`load_config` resolves a config file's tree with it; use it directly
    only for values that do not come from a config file (a provider spec, an
    overlay of ``os.environ`` and ``.env``).

    Args:
        data: The config data (dict/list/str/scalar) to resolve.
        environ: Optional mapping to resolve ``${VAR}`` against instead of the
            process environment. Lets callers expand against an
            ``os.environ`` + ``.env`` overlay without mutating global state
            (e.g. the Claude Code provider loader). Defaults to ``os.environ``.
    """
    return _resolved(data, os.environ if environ is None else environ)


def _resolved(data: object, lookup: "Mapping[str, str]") -> object:
    """:func:`resolve_env_vars` against *lookup*: a dict, a list, a string, or *data* itself."""
    if isinstance(data, dict):
        return _resolved_dict(data, lookup)
    if isinstance(data, list):
        return [_resolved(item, lookup) for item in data]
    if not isinstance(data, str):
        return data

    def replace_env_var(match: re.Match[str]) -> str:
        if match.group(1):  # ${VAR_NAME:-default} or ${VAR_NAME}
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else None
        else:  # $VAR_NAME (simple form)
            var_name = match.group(3)
            default_value = None

        env_value = lookup.get(var_name)
        # Match bash :- semantics: empty string triggers default too
        if env_value is None or (not env_value and default_value is not None):
            if default_value is not None:
                return default_value
            # DEBUG, not INFO: this resolver is generic over the whole
            # config tree and knows nothing about which of the values it
            # walks actually matter to the caller. Reported at INFO it
            # rendered every build as a wall of misses for providers the
            # project never uses, while saying nothing about the keys
            # that *did* resolve. User-facing credential reporting lives
            # in osprey.cli.build_environment.report_provider_credentials,
            # which knows the selected provider.
            if not os.environ.get("OSPREY_QUIET"):
                logger.debug(
                    f"Environment variable '{var_name}' not found, keeping placeholder verbatim"
                )
            return match.group(0)
        return env_value

    pattern = r"\$\{([^}:]+)(?::-(.*?))?\}|\$([A-Za-z_][A-Za-z0-9_]*)"
    return re.sub(pattern, replace_env_var, data)


def _resolved_dict(data: dict[str, object], lookup: "Mapping[str, str]") -> dict[str, object]:
    """A dict's values resolved by :func:`_resolved`, with the MCP server env blocks kept raw.

    The keys come back as they went in; typed ``str``, as a config's are, because
    only the ``claude_code`` key is ever looked up.
    """
    raw_server_envs: dict[object, object] = {}
    cc = data.get("claude_code")
    servers = cc.get("servers") if isinstance(cc, dict) else None
    if isinstance(servers, dict):
        for name, spec in servers.items():
            env = spec.get("env") if isinstance(spec, dict) else None
            if isinstance(env, dict):
                raw_server_envs[name] = copy.deepcopy(env)

    resolved = {key: _resolved(value, lookup) for key, value in data.items()}

    # Restore preserved MCP server env blocks (only meaningful at the top of a
    # config dict; sub-recursions never hit this path because they have no
    # ``claude_code`` key and therefore no snapshot to restore).
    if raw_server_envs:
        cc = resolved.setdefault("claude_code", {})
        servers = cc.setdefault("servers", {}) if isinstance(cc, dict) else None
        if isinstance(servers, dict):
            for name, env in raw_server_envs.items():
                spec = servers.get(name)
                if isinstance(spec, dict):
                    spec["env"] = env
    return resolved


#: The placeholder shapes :func:`resolve_env_vars` substitutes; one still
#: present after resolution named a variable the environment does not have.
_PLACEHOLDER = re.compile(r"\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*")


def unresolved_placeholders(value: object) -> list[str]:
    """Every placeholder left in *value*, depth-first.

    Where :func:`is_unresolved_placeholder` asks whether a value is exactly one
    reference, this finds every reference anywhere in a string, and in every
    string nested in mappings, lists and tuples — so a block a supervisor is
    about to hand a connector can be refused whole when any part of it is still
    unresolved.

    Args:
        value: A config value, of any type, after :func:`resolve_env_vars`.

    Returns:
        The placeholders found, in order, repeats included; empty when none.
    """
    if isinstance(value, str):
        return _PLACEHOLDER.findall(value)
    if isinstance(value, Mapping):
        return [found for item in value.values() for found in unresolved_placeholders(item)]
    if isinstance(value, list | tuple):
        return [found for item in value for found in unresolved_placeholders(item)]
    return []


#: Directory the build owns end to end: every file under it is rendered from the
#: profile/preset and checksummed into the manifest.
_BUILD_OWNED_DATA_DIR = "data"

#: Directory runtime state belongs in — the durable ``var/`` zone, outside the
#: render entirely, so it survives every rebuild of ``build/``. Named in the
#: advisory that fires when a runtime writer is pointed at build-owned ``data/``.
_RUNTIME_STATE_DIR = DEFAULT_AGENT_DATA_BASE_DIR

#: Config keys this package reads whose value names a path something writes to
#: *at run time* (the framework's own, its services' among them, are in
#: ``osprey.utils.workspace.RUNTIME_WRITE_PATH_KEYS``). These must stay out of
#: ``data/``: that tree is build-owned and checksummed by
#: :func:`osprey.cli.templates.manifest.calculate_file_checksums`, so a runtime
#: write landing there reads as project drift and is erased by the next
#: ``osprey build``.
_RUNTIME_WRITE_PATH_KEYS = (SIMULATION_STATE_DIR_CONFIG_KEY,)

#: The sections the load reads whole, each of which has to be a mapping (or
#: absent, or empty): a scalar or a list there is refused when the file is read,
#: by key, rather than by every reader that walks into it later.
_MAPPING_SECTIONS = ("models", "api.providers", "services", "execution", "python_executor")


def _runtime_write_paths_under_data(
    config: "Mapping[str, object]", project_root: Path
) -> list[tuple[str, str]]:
    """Find configured runtime-write paths that resolve inside ``<root>/data/``.

    Args:
        config: Loaded ``config.yml`` mapping.
        project_root: Directory relative paths in the config resolve against.

    Returns:
        ``(config_key, configured_value)`` pairs for every offending key, in
        :data:`_RUNTIME_WRITE_PATH_KEYS` order. Empty when the config is clean.
    """
    data_root = (project_root / _BUILD_OWNED_DATA_DIR).resolve()
    offenders: list[tuple[str, str]] = []

    for key in _RUNTIME_WRITE_PATH_KEYS:
        found, value = _walk(config, key)
        if not found or not isinstance(value, str) or not value:
            continue

        resolved = anchored_path(value, project_root).resolve()
        if resolved == data_root or data_root in resolved.parents:
            offenders.append((key, value))

    return offenders


def _warn_on_runtime_write_paths_under_data(tree: "Mapping[str, object]", path: Path) -> None:
    """Warn when a runtime writer is pointed at the build-owned ``data/`` tree.

    Advisory only: the misconfiguration still works until the next
    ``osprey build`` wipes ``data/`` and takes the runtime state
    with it, so this warns rather than raising.
    """
    configured_root = _configured_root(tree)
    project_root = (
        Path(configured_root) if configured_root else repo_root_for_config(path)
    ).expanduser()

    for key, value in _runtime_write_paths_under_data(tree, project_root):
        logger.warning(
            "Runtime-write path '%s' = %r resolves inside the build-owned "
            "'%s/' tree (%s). That directory is re-rendered and checksummed on "
            "every build, so runtime writes there show up as project drift and "
            "are erased by the next 'osprey build'. Point it at '%s/' instead.",
            key,
            value,
            _BUILD_OWNED_DATA_DIR,
            path,
            _RUNTIME_STATE_DIR,
        )


def _walk(tree: object, key: str) -> tuple[bool, object]:
    """``(found, value)`` for the dotted *key* in *tree*, walked as nested mappings.

    A missing key, or a segment that is not a mapping (a list, a scalar,
    ``None``), is not found. The value is the tree's own object, not a copy.
    """
    value: object = tree
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _configured_root(tree: "Mapping[str, object]") -> str | None:
    """The ``project_root`` *tree* names, when it is a non-empty string."""
    value = tree.get("project_root")
    return value if isinstance(value, str) and value else None


class UnresolvedRef(FrozenModel):
    """One ``${VAR}`` reference the load left in the config, because ``VAR`` is unset.

    The value stays as written (a lone ``${VAR}`` is data, never read as
    absent); this is the diagnostic a refusal or a report names.

    Attributes:
        key: The dotted key of the value that holds it (a list item's index is a
            segment, as in ``a.b.0``).
        var: The variable it names.
    """

    key: str
    var: str


def _unresolved_refs(node: object, prefix: str = "") -> Iterator[UnresolvedRef]:
    """Every reference :func:`resolve_env_vars` left in *node*, depth-first, keyed by path.

    The ``claude_code.servers.*.env`` blocks are skipped: their placeholders are
    kept on purpose, for Claude Code to expand when it launches the server.
    """
    if isinstance(node, str):
        for found in _PLACEHOLDER.findall(node):
            yield UnresolvedRef(key=prefix, var=found.strip("${}"))
        return
    if isinstance(node, Mapping):
        items: list[tuple[object, object]] = list(node.items())
        for name, value in items:
            path = f"{prefix}.{name}" if prefix else str(name)
            if prefix.startswith("claude_code.servers.") and name == "env":
                continue
            yield from _unresolved_refs(value, path)
        return
    if isinstance(node, list):
        for index, item in enumerate(node):
            yield from _unresolved_refs(item, f"{prefix}.{index}" if prefix else str(index))


class LoadedConfig(FrozenModel):
    """One rendered ``config.yml``, loaded.

    Built by :func:`load_config` (and, for a test, by
    :func:`osprey_connectors.testing.config_of`); nothing else makes one. The
    tree itself is held privately as the plain ``dict`` the YAML parse made and
    is handed out only as copies (:meth:`get`, :meth:`section`,
    :meth:`raw_tree`), so no caller can change the config another reads.

    Attributes:
        path: The file it was loaded from, as it was named.
        repo_root: The deployment repo the config belongs to: its
            ``project_root`` when that names a directory here, else the repo
            the file sits in (``<repo>/build/config.yml``, or a container's
            project directory, which is its own render).
        unresolved: Every ``${VAR}`` reference left verbatim because ``VAR`` is
            unset, for diagnostics; the values themselves stay as written.
        control_system: The ``control_system`` section, read once at load
            (never refused there: what a reader refuses, it refuses where it
            reads). The reference monitor and the write-path readers judge by it.
        archiver: The ``archiver`` section, read once at load: the selected
            archiver and its settings block (a block the archive cannot be
            built from is refused where it is resolved, never here).
        pv_directory: The ``pv_directory`` section, read once at load: one
            block per service kind.
        system: The ``system`` section the package reads: the facility clock
            ``system.timezone`` names.
        logging: The ``logging`` section's three switches, read once at load.
        simulation: The ``simulation`` section's periods and state directory,
            read once at load.
        agent_data: The ``agent_data`` section's ``base_dir``, read once at load.
    """

    path: Path
    repo_root: Path
    unresolved: FrozenTuple[UnresolvedRef] = ()
    control_system: ControlSystemSection
    archiver: ArchiverSection
    pv_directory: PvDirectorySection
    system: SystemSection
    logging: LoggingSection
    simulation: SimulationSection
    agent_data: AgentDataSection

    _tree: dict[str, object] = PrivateAttr()
    _unexpanded: dict[str, object] = PrivateAttr()
    # Whether facility_timezone has reported this load's zone problem already.
    _zone_reported: bool = PrivateAttr(default=False)

    def get(self, key: str, default: object = None) -> object:
        """The value at the dotted *key*, as a copy; *default* when it is not there.

        A segment that is missing, or that is not a mapping, is not there. A
        key written with a YAML null is there, and answers ``None``.

        Raises:
            ConfigurationError: *key* is empty.
        """
        if not key:
            raise ConfigurationError("Configuration path cannot be empty or None")
        found, value = _walk(self._tree, key)
        return copy.deepcopy(value) if found else default

    def section(self, key: str) -> dict[str, object]:
        """The mapping at the dotted *key*, as a copy; ``{}`` when absent or null.

        Raises:
            ConfigurationError: The value there is not a mapping, naming *key*.
        """
        value = self.get(key)
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        raise ConfigurationError(
            f"'{key}' in {self.path} must be a mapping, got {type(value).__name__}", key=key
        )

    def flag(self, key: str, *, default: bool) -> bool:
        """The on/off switch at the dotted *key*, read by :func:`config_flag`'s rule.

        An absent key, or one written as null, is *default*.

        Raises:
            ConfigurationError: The value spells neither on nor off, naming *key*.
        """
        value = self.get(key)
        return default if value is None else config_flag(value, key=key)

    def repo_path(self, value: "str | os.PathLike[str]") -> Path:
        """A path the config names, as a path here.

        ``~`` is expanded; an absolute path is returned as written, and a
        relative one is anchored at :attr:`repo_root`, as every relative path
        a deployment configures is.
        """
        return anchored_path(os.fspath(value), self.repo_root)

    @property
    def agent_data_root(self) -> Path:
        """The shared agent-data root: ``agent_data.base_dir`` (else ``var/agent_data``), resolved.

        Shared by every session: the stores long-lived daemons read (the
        gallery, ARIEL, the control-context record) live here.
        """
        return self.repo_path(agent_data_base_dir(self._tree)).resolve()

    def session_data_root(self, session_id: str) -> Path:
        """One session's own agent-data directory, under :attr:`agent_data_root`.

        Raises:
            ValueError: *session_id* is blank.
        """
        if not session_id.strip():
            raise ValueError("a session's data root needs a non-blank session id")
        return self.agent_data_root / "sessions" / session_id

    @property
    def simulation_state_dir(self) -> Path:
        """The directory holding the simulation's ``active_scenarios`` state file.

        ``simulation.state_dir`` when it names one (relative to
        :attr:`repo_root`), else ``simulation`` under the agent-data base
        directory; not created here.
        """
        return resolve_simulation_state_dir(self._tree, self.repo_root)

    # The trees are the YAML parse's, handed out as such: every framework reader
    # reads the shape its own keys have.
    def raw_tree(self) -> dict[str, ContractAny]:
        """The whole resolved tree, as a copy: for framework code that reads its own keys."""
        return copy.deepcopy(self._tree)

    def unexpanded_tree(self) -> dict[str, ContractAny]:
        """The whole tree as written, ``${VAR}`` placeholders kept, as a copy.

        For deployment code that writes the config out again and must not bake
        a secret, or an empty string for an unset variable, into the file.
        """
        return copy.deepcopy(self._unexpanded)

    @override
    @classmethod
    def model_validate(cls, *args: object, **kwargs: object) -> NoReturn:
        """Refused: a loaded config is built from its file, never from its fields.

        Its tree is held privately, so a copy validated from the fields alone
        (``model_validate``, and :func:`~osprey_connectors.model.evolve`, which
        goes through it) would have none, and its sections would no longer be
        the tree's. Load the changed file again, or build the changed tree with
        :func:`osprey_connectors.testing.config_of`.

        Raises:
            TypeError: Always.
        """
        raise TypeError(
            "a LoadedConfig is built by load_config (or testing.config_of), not from its "
            "fields: evolve and model_validate would drop the tree it was loaded from"
        )


def _loaded(
    tree: dict[str, object],
    path: Path,
    *,
    environ: "Mapping[str, str]",
) -> LoadedConfig:
    """A YAML document's tree as a :class:`LoadedConfig`: resolved, checked, and read.

    The part of a load after the file is parsed, shared by :func:`load_config`
    and :func:`osprey_connectors.testing.config_of`.

    Raises:
        ConfigurationError: A connector block still spells its call bound
            ``timeout`` (see
            :func:`osprey_connectors.call_timeout.refuse_renamed_timeout_keys`).
        ConfigParseError: One of the sections the load reads whole is not a
            mapping, naming its key.
    """
    from osprey_connectors.call_timeout import refuse_renamed_timeout_keys

    unexpanded = copy.deepcopy(tree)
    resolved = _resolved_dict(tree, environ)

    refuse_renamed_timeout_keys(resolved)
    for key in _MAPPING_SECTIONS:
        _, value = _walk(resolved, key)
        if value is not None and not isinstance(value, dict):
            raise ConfigParseError(
                f"'{key}' in {path} must be a mapping, got {type(value).__name__}",
                path=path,
                key=key,
            )

    _warn_on_runtime_write_paths_under_data(resolved, path)

    configured_root = _configured_root(resolved)
    candidate = Path(configured_root).expanduser() if configured_root else None
    repo_root = (
        candidate if candidate is not None and candidate.is_dir() else repo_root_for_config(path)
    )

    loaded = LoadedConfig(
        path=path,
        repo_root=repo_root,
        unresolved=tuple(_unresolved_refs(resolved)),
        control_system=ControlSystemSection.read(resolved.get("control_system")),
        archiver=ArchiverSection.read(resolved.get("archiver")),
        pv_directory=PvDirectorySection.read(resolved.get("pv_directory")),
        system=SystemSection.read(resolved.get("system")),
        logging=LoggingSection.read(resolved.get("logging")),
        simulation=SimulationSection.read(resolved.get("simulation")),
        agent_data=AgentDataSection.read(resolved.get("agent_data")),
    )
    loaded._tree = resolved
    loaded._unexpanded = unexpanded
    return loaded


def _candidates(environ: "Mapping[str, str]", cwd: Path) -> tuple[Path, ...]:
    """The paths :func:`locate_config` looks at, in order."""
    configured = environ.get("OSPREY_CONFIG")
    if configured:
        return (Path(str(_resolved(configured, environ))),)
    return (rendered_config_path(cwd), cwd / "config.yml")


def locate_config(*, environ: "Mapping[str, str] | None" = None, cwd: Path | None = None) -> Path:
    """The ``config.yml`` a lookup that names no file reads.

    Resolution order:
      1. ``OSPREY_CONFIG``, with ``$VAR`` references in it expanded. The CLI
         sets it on the host, every framework MCP server definition sets it,
         and so does the compose service of every container whose working
         directory does not hold the render.
      2. ``build/config.yml`` under *cwd*: standing in a repo root with a
         render in it.
      3. ``config.yml`` under *cwd*: a container project directory, whose root
         *is* the render, and the path reported when nothing exists at all.

    ``CONFIG_FILE`` is not read: it is a name generic enough that honoring it
    would let an unrelated export redirect every OSPREY process on the host.
    Every compose service that sets it sets ``OSPREY_CONFIG`` to the same path.

    Args:
        environ: The environment to read ``OSPREY_CONFIG`` from; the process's
            own by default.
        cwd: The directory steps 2 and 3 look in; the working directory by
            default.

    Returns:
        The path, which exists unless nothing at all was found.
    """
    env = os.environ if environ is None else environ
    candidates = _candidates(env, Path.cwd() if cwd is None else cwd)
    return next((path for path in candidates[:-1] if path.is_file()), candidates[-1])


def _not_found_message(path: Path, *, named: bool, environ: "Mapping[str, str]") -> str:
    """Why a load found no config at *path*, naming where it looked."""
    if named:
        return (
            f"Config file not found: {path}\n\n"
            f"Set the OSPREY_CONFIG environment variable to a valid config.yml, "
            f"or run from a project directory that contains one."
        )
    if environ.get("OSPREY_CONFIG"):
        return (
            f"No config.yml found: OSPREY_CONFIG names {path}, which does not exist.\n\n"
            f"Point OSPREY_CONFIG at a rendered config.yml, or unset it to read the one "
            f"under the current directory."
        )
    cwd = path.parent
    return (
        f"No config.yml found in current directory: {cwd}\n\n"
        f"Looked for {rendered_config_path(cwd)} and {cwd / 'config.yml'}.\n\n"
        f"Run from a project directory (`osprey build` renders its config to "
        f"build/config.yml),\n"
        f"or set the OSPREY_CONFIG environment variable to point to your config file.\n\n"
        f"Example: export OSPREY_CONFIG=/path/to/your/config.yml"
    )


def load_config(
    path: Path | str | None = None, *, environ: "Mapping[str, str] | None" = None
) -> LoadedConfig:
    """Load one ``config.yml``: read it, resolve its ``${VAR}`` placeholders, and check it.

    Reads the file and nothing else: no ``.env`` is applied and ``os.environ``
    is never written, so the caller decides what environment the placeholders
    resolve against. Nothing is cached and no process default is installed
    (see :func:`install_config`).

    The load refuses only what no reader could use: a file that is not YAML or
    whose document is not a mapping, a connector block that still spells its
    call bound ``timeout``, and a ``models``, ``api.providers``, ``services``,
    ``execution`` or ``python_executor`` section that is not a mapping. Every
    other value is read where it is used, so a typo in one section does not
    stop a process that never reads it.

    Args:
        path: The file to load; ``None`` loads the one :func:`locate_config`
            finds in *environ*.
        environ: What ``${VAR}`` placeholders (and, with no *path*,
            ``OSPREY_CONFIG``) resolve against; the process environment by
            default.

    Returns:
        The loaded config.

    Raises:
        ConfigNotFoundError: No file is there.
        ConfigLoadError: The path is a directory, or the file cannot be read.
        ConfigParseError: The file is not YAML, its document is not a mapping,
            or a section read whole is not one.
        ConfigurationError: A connector block spells ``timeout`` for
            ``timeout_s``.
    """
    env = os.environ if environ is None else environ
    if path is None:
        searched = _candidates(env, Path.cwd())
        config_path = next((p for p in searched[:-1] if p.is_file()), searched[-1])
    else:
        config_path = Path(path)
        searched = (config_path,)

    # A container bind-mount whose host source did not exist at create time is
    # materialized as an empty directory on both ends, so the path names one;
    # the YAML loader would otherwise surface a bare "[Errno 21] Is a directory".
    if config_path.is_dir():
        raise ConfigLoadError(
            f"Config path is a directory, not a file: {config_path}\n\n"
            f"config.yml is expected to be a file here. This usually means it "
            f"was meant to be provided at runtime (e.g. a bind-mount whose "
            f"source was missing, so the container runtime created an empty "
            f"directory) or was not baked into the image. Ensure config.yml "
            f"exists as a file at this path.",
            path=config_path,
        )
    if not config_path.exists():
        raise ConfigNotFoundError(
            _not_found_message(config_path, named=path is not None, environ=env),
            path=config_path,
            searched=searched,
        )

    try:
        with open(config_path, encoding="utf-8") as f:
            document: object = yaml_loader.safe_load(f)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        line = mark.line + 1 if mark is not None else None
        raise ConfigParseError(
            f"Error parsing YAML configuration {config_path}: {e}", path=config_path, line=line
        ) from e
    except UnicodeDecodeError as e:
        raise ConfigParseError(
            f"Configuration file {config_path} is not UTF-8 text: {e}", path=config_path
        ) from e
    except OSError as e:
        raise ConfigLoadError(
            f"Could not read configuration file {config_path}: {e}", path=config_path
        ) from e

    if document is None:
        logger.warning(f"Configuration file is empty: {config_path}")
        document = {}
    if not isinstance(document, dict):
        raise ConfigParseError(
            f"Configuration file must contain a dictionary/mapping: {config_path}",
            path=config_path,
        )

    loaded = _loaded(document, config_path, environ=env)
    logger.debug(f"Loaded configuration from {config_path}")
    return loaded


# --- the process default ------------------------------------------------------

#: The config :func:`install_config` made this process's default, or ``None``.
_installed: LoadedConfig | None = None

#: The config the innermost :func:`using_config` block of this context names.
#: Layered over :data:`_installed` rather than replacing it: a context variable
#: is invisible to a thread started outside ``copy_context`` (an executor, a
#: server's worker threads), so the process-wide default has to stay a global.
_scoped: ContextVar[LoadedConfig | None] = ContextVar("osprey_connectors_config", default=None)

#: The lazy default's loads, by resolved path, with the file's ``st_mtime_ns``
#: when it was read: a re-rendered file is loaded again without a reset.
_loads: dict[Path, tuple[int, LoadedConfig]] = {}

#: The config the lazy default answered with last, for :func:`active_config`.
_last_lazy: LoadedConfig | None = None

#: Whether :func:`facility_timezone` has reported that no config could be loaded.
_no_zone_reported = False


def _mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _cache_hit(path: Path) -> LoadedConfig | None:
    """The lazy rule's load of *path*, if the file has not changed since it was read."""
    key = path.resolve()
    cached = _loads.get(key)
    return cached[1] if cached is not None and cached[0] == _mtime_ns(key) else None


def _load_and_cache(path: Path | None) -> LoadedConfig:
    """*path* loaded, or the located file when ``None``, and kept with its ``st_mtime_ns``.

    The stamp is taken before the read, so a rewrite between the two shows as
    a change on the next lookup.
    """
    located = locate_config() if path is None else path
    key = located.resolve()
    mtime = _mtime_ns(key)
    loaded = load_config(path)
    if mtime is not None:
        _loads[key] = (mtime, loaded)
    return loaded


def _lazy_default() -> LoadedConfig:
    """The file :func:`locate_config` finds, cached until it changes, the env chain applied first.

    The working directory's env chain is applied before each load this makes,
    as every config load applied it before loads were explicit, so the
    ``${VAR}`` placeholders resolve against the deployment's ``.env``.
    """
    global _last_lazy

    from osprey_connectors.dotenv import load_project_dotenv

    loaded = _cache_hit(locate_config())
    if loaded is None:
        # The env chain may itself set OSPREY_CONFIG, so look again once it is
        # applied, as a load always did: the chain first, then the file.
        load_project_dotenv()
        loaded = _load_and_cache(None)
    _last_lazy = loaded
    return loaded


def current_config() -> LoadedConfig:
    """The config this process's unqualified lookups read.

    The config of the innermost :func:`using_config` block, else the one
    :func:`install_config` installed, else the file :func:`locate_config`
    finds, loaded and kept until it changes (the working directory's env chain
    applied before each such load).

    Raises:
        ConfigLoadError: No config is installed and the located file does not
            load (:class:`~osprey_connectors.errors.ConfigNotFoundError` when
            there is none); the refusals of :func:`load_config` as well.
    """
    scoped = _scoped.get()
    if scoped is not None:
        return scoped
    if _installed is not None:
        return _installed
    return _lazy_default()


def active_config() -> LoadedConfig | None:
    """The config :func:`current_config` would answer with, if it has one already.

    Side-effect free: nothing is located or loaded, and no ``.env`` is read.
    ``None`` until something has installed, scoped or lazily loaded a config.
    """
    scoped = _scoped.get()
    if scoped is not None:
        return scoped
    return _installed if _installed is not None else _last_lazy


def install_config(config: LoadedConfig | None) -> LoadedConfig | None:
    """Make *config* the process default every unqualified lookup reads; ``None`` removes it.

    Process-wide, so every thread sees it: what an entry point that knows its
    deployment calls once it has loaded that deployment's config. A verb that
    acts on one deployment for a while puts the previous default back with the
    value this returns.

    Returns:
        The default installed before, or ``None``.
    """
    global _installed

    previous, _installed = _installed, config
    return previous


@contextmanager
def using_config(config: "LoadedConfig | Path | str") -> Iterator[LoadedConfig]:
    """Answer this context's unqualified lookups from *config* for the block.

    Scoped to the current context (a task, a thread, a ``copy_context`` run):
    blocks nest, concurrent tasks each see their own, and the previous answer
    is back on exit. A thread started without the context does not see it;
    :func:`install_config` is the process-wide default.

    Args:
        config: The config, or the path of one to load with :func:`load_config`.

    Yields:
        The config in force.

    Raises:
        ConfigLoadError: *config* is a path that does not load (see
            :func:`load_config`).
    """
    loaded = config if isinstance(config, LoadedConfig) else load_config(config)
    token = _scoped.set(loaded)
    try:
        yield loaded
    finally:
        _scoped.reset(token)


def _clear_process_state() -> None:
    """Forget the process default, every lazy load and every warn-once latch.

    For :func:`osprey_connectors.testing.clear_config_state`, between tests; the current
    context's :func:`using_config` scope is cleared with them.
    """
    global _installed, _last_lazy, _no_zone_reported

    _installed = None
    _last_lazy = None
    _loads.clear()
    _no_zone_reported = False
    _scoped.set(None)


def mapping_or_empty(value: object, path: str, source: object) -> dict[str, object]:
    """Return a config section as a mapping: ``None`` reads as ``{}``, a non-mapping is refused."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise ConfigurationError(
        f"'{path}' in {source} must be a mapping, got {type(value).__name__}", key=path
    )


def config_section(config: object, name: str) -> dict[str, object]:
    """One top-level section of a rendered config, or an empty mapping.

    Nested sections only, the way ``MCPServerConfig`` and :class:`LoadedConfig`
    read this file: a top-level dotted line in ``config.yml`` configures nothing.
    """
    if not isinstance(config, dict):
        return {}
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def subsection(section: Mapping[str, object], name: str) -> dict[str, object]:
    """One nested section of *section*, or an empty mapping when it is absent or not one."""
    value = section.get(name)
    return value if isinstance(value, dict) else {}


def connector_block(config: object, connector_type: str) -> object:
    """The ``control_system.connector.<type>`` block, as written."""
    return subsection(config_section(config, "control_system"), "connector").get(connector_type)


def facility_timezone(config: LoadedConfig | None = None) -> ZoneInfo:
    """The facility clock: the zone *config* names, else the process config's.

    The one lenient reader of the facility zone, so it never raises: every
    simulation synthesis primitive and timestamp render site calls it. A zone
    that is not one (see :class:`~osprey_connectors.config_sections.FacilityZone`) is UTC, reported once per load;
    no config at all (none installed and none to find) is UTC too, reported
    once per process. Once rather than per call, because synthesis calls this
    per channel per chunk, and once rather than never, because a process
    reading the wrong clock still has to be diagnosable from its own log.

    When the zone is configured and the container's ``$TZ`` disagrees, that
    is reported once per load as well: agent-facing timestamps key off
    ``system.timezone``, so a divergence only mis-stamps OS-level logs, but it
    signals a misconfigured deploy.

    Args:
        config: The config whose zone to read; :func:`current_config`'s by
            default.

    Returns:
        The facility zone.
    """
    global _no_zone_reported

    if config is None:
        try:
            config = current_config()
        except (ConfigLoadError, ConfigurationError) as exc:
            if not _no_zone_reported:
                _no_zone_reported = True
                logger.warning(
                    f"Falling back to UTC facility timezone ({type(exc).__name__}: {exc})"
                )
            return ZoneInfo("UTC")

    zone = config.system.timezone
    if not config._zone_reported:
        config._zone_reported = True
        if zone.source == "fallback":
            logger.warning(f"Falling back to UTC facility timezone ({zone.problem})")
        elif zone.source == "configured":
            _warn_on_tz_drift(zone.name)
    return zone.zone


def _warn_on_tz_drift(tz_name: str) -> None:
    """Warn if the container ``$TZ`` disagrees with an explicit ``system.timezone``.

    ``system.timezone`` stays the single source of truth: this never changes
    the zone read and never raises.
    """
    host_tz = os.environ.get("TZ")
    if not host_tz or host_tz == tz_name:
        return
    logger.warning(
        f"Container $TZ={host_tz!r} differs from the facility system.timezone={tz_name!r}; "
        f"OS-level log timestamps will differ from agent-reported times. Set both to "
        f"the same zone (system.timezone is authoritative for what the agent reports)."
    )


@overload
def localize_facility(dt: datetime, tz: ZoneInfo) -> datetime: ...


@overload
def localize_facility(dt: None, tz: ZoneInfo) -> None: ...


def localize_facility(dt: datetime | None, tz: ZoneInfo) -> datetime | None:
    """Read a naive datetime as wall-clock in *tz*; pass an aware one through.

    A naive datetime is stamped with the facility zone the caller names
    (normally :func:`facility_timezone`'s) -- never the box ``$TZ``, which
    ``astimezone`` would silently impose.

    Args:
        dt: A datetime in either state, or ``None``.
        tz: The facility zone.

    Returns:
        The same instant, timezone-aware, or ``None`` for ``None`` input.
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt


# No eager load here, deliberately. Building the default config at import time
# loaded `.env` into os.environ as a side effect of importing any osprey module
# (osprey/utils/__init__.py imports this one), which is the behaviour the
# module docstring rules out. The first current_config() call builds it
# instead; entry points that need `.env` in the environment load it themselves.
