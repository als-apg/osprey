"""The deployment repo's layout, as the framework spells it.

``osprey build`` lays a deployment repo out in zones — ``profile.yml`` and the
source layers at the root, the render in ``build/``, durable state in ``var/`` —
and the CLI, the deployment layer, the runtime and the services all resolve
paths against that layout. The names live here, below all of them, so a writer
and a reader of one directory cannot spell it two ways.

The connector package resolves the rendered config and the agent-data root on
its own (:mod:`osprey_connectors.workspace`), and only the names it needs to
say where those are stay public there: :data:`~osprey_connectors.workspace.BUILD_DIR_NAME`,
:data:`~osprey_connectors.workspace.RENDERED_CONFIG_RELPATH` and
:data:`~osprey_connectors.workspace.DEFAULT_AGENT_DATA_BASE_DIR`. Everything
else about the layout is the framework's. A deployment-root entry is spelled in
:mod:`osprey.repo_layout`, and its name here is an alias of that row; the names
inside the zones are spelled here. Two names here
are also spelled privately inside the connector package, because its resolvers
need them: :data:`PROFILE_FILENAME` and :data:`STATE_DIR_NAME`. A test holds
each pair equal.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from osprey.repo_layout import child_display_path, entry, layer_rows
from osprey_connectors.dotenv import ENV_CHAIN_FILENAMES
from osprey_connectors.workspace import (
    BUILD_DIR_NAME,
    DEFAULT_AGENT_DATA_BASE_DIR,
    SIMULATION_STATE_DIR_CONFIG_KEY,
    anchored_path,
    repo_root_for_config,
)

__all__ = [
    "APPS_DIRNAME",
    "ARCHIVE_DIR_RELPATH",
    "CHAT_UPLOADS_SUBDIR",
    "IMAGE_DIR_NAME",
    "LAYER_DIRNAMES",
    "LAYER_DIR_KINDS",
    "PRESET_TRACKING_DIRNAME",
    "PRE_UPGRADE_AUDIT_RELPATH",
    "PROFILE_FILENAME",
    "NO_PROTECTED_PATHS_CONSEQUENCE",
    "PROTECTED_PATHS_CONFIG_KEY",
    "RECORD_DIR_RELPATH",
    "RETIRED_LAYER_DIRS",
    "RUNTIME_DIR_RELPATH",
    "RUNTIME_WRITE_PATH_KEYS",
    "SIMULATOR_STANDIN_LOG_SUBDIR",
    "SPOOL_DIR_RELPATH",
    "STATE_DIR_NAME",
    "STATE_DIR_RELPATH",
    "STATE_ZONE_DIRS",
    "TIERS_DIRNAME",
    "AnchoredEntry",
    "ProtectedPathsReading",
    "ProtectedPathsState",
    "anchor_protected_entry",
    "container_image_context",
    "deployment_env_chain",
    "layer_dir_kind",
    "load_protected_paths",
    "protected_roots_from_entries",
    "read_protected_paths",
    "repo_root_for_agent_data",
    "session_hold_dir",
]

#: The durable state zone inside a deployment repo: ``var/`` survives every
#: ``osprey build``, which re-creates ``build/`` wholesale.
STATE_DIR_NAME = entry("var").path

#: The file that makes a directory a deployment repo root — the marker the
#: CLI's repo walk and the hook library's anchor both look for.
PROFILE_FILENAME = entry("profile.yml").path

#: Directory beside ``profile.yml`` (or in the bundled presets) holding the
#: logins: one file per login of the deployment's agent.
TIERS_DIRNAME: Final = entry("tiers").path

#: Directory beside ``profile.yml`` holding the applications, one
#: ``apps/<name>/`` directory each. Not a layer directory: no file in it is a
#: login.
APPS_DIRNAME: Final = entry("apps").path

#: Directory beside ``profile.yml`` holding what OSPREY records about the repo
#: for itself: the saved copy of what the presets produced, under
#: ``preset-base/`` (:mod:`osprey.cli.build_profile_updates`). Tracked in git.
PRESET_TRACKING_DIRNAME: Final = entry(".osprey").path

#: Each live layer directory -> the kind word its files are read as. The
#: listing order is the order every layer listing follows. Read from the
#: layer rows, below ``cli``, because the hook library and the runtime read the
#: layout too.
LAYER_DIR_KINDS: Mapping[str, str] = MappingProxyType(
    {row.path: row.layer_kind for row in layer_rows() if row.layer_kind is not None}
)

#: Each retired layer directory -> the live directory its files are read as,
#: with one warning naming the move. Closing the deprecation window is deleting
#: the retired path from its row.
RETIRED_LAYER_DIRS: Mapping[str, str] = MappingProxyType(
    {retired: row.path for row in layer_rows() for retired in row.retired_paths}
)

#: Every directory beside ``profile.yml`` whose files are layers: the live ones
#: in listing order, then the retired ones.
LAYER_DIRNAMES: tuple[str, ...] = (*LAYER_DIR_KINDS, *RETIRED_LAYER_DIRS)


def layer_dir_kind(dirname: str) -> str | None:
    """The kind word (``"tier"``) a layer directory's files are read as.

    A retired directory answers with its successor's kind; any other name is
    not a layer directory and answers ``None``.
    """
    return LAYER_DIR_KINDS.get(RETIRED_LAYER_DIRS.get(dirname, dirname))


#: Where the build stages the container-destined copy of a project, under the
#: render zone. Each is a deployment REPO — ``profile.yml`` and the source zone
#: at its root, its own render below in ``build/`` — rendered against the
#: ``/app/<project>`` path a container sees rather than the building host's, and
#: it is what an image is built from. Spelled here beside the zone names because
#: the build writes it and the deploy layer builds from it, and the two must not
#: disagree about where it is.
IMAGE_DIR_NAME = ".image"

#: Material ``osprey up`` mints when it starts a deployment, relative to the
#: repo root: the per-lane Bluesky CURVE certificate sets,
#: ``<lane>_curve/``. It lies in the state zone, so the build never hashes it,
#: never stages it and never copies it into an image, and a start never marks
#: its own build out of date. The ``var`` row's ``runtime`` child
#: (:mod:`osprey.repo_layout`), named here beside the other state-zone paths
#: because the executor's protected-write roots and the reset listing have to
#: name the same directory.
RUNTIME_DIR_RELPATH = child_display_path("var", "runtime")

#: The audit tree a deployment wrote before the agent record existed, relative
#: to the repo root. Nothing writes or reads it: it stays on disk
#: unchanged, executed code is still refused writing it, and only
#: ``osprey reset --purge-audit`` deletes it. Named here because those two
#: keepers — the executor's protected-write roots and the reset purge — have to
#: name the same directory.
PRE_UPGRADE_AUDIT_RELPATH = f"{STATE_DIR_NAME}/audit"

#: The archive, relative to the repo root: each closed day's export of the agent
#: record beside the artifact and Bluesky copies, with an append-only manifest per
#: day. Only ``osprey reset --purge-audit`` deletes under it, and only the
#: record's exports (``<day>/agent_record/``), marking each such day purged.
ARCHIVE_DIR_RELPATH = f"{STATE_DIR_NAME}/archive"

#: The spool root, relative to the repo root: one directory per writer identity
#: (``<identity>/``) plus agent-record's receiver-owned ``.otlp/<login>/``. The
#: writer client appends here and agent-record ingests and then deletes; it is
#: transport, never history.
SPOOL_DIR_RELPATH = f"{STATE_DIR_NAME}/spool"

#: The embedded agent-record store's directory, relative to the repo root,
#: written by agent-record alone.
RECORD_DIR_RELPATH = f"{STATE_DIR_NAME}/agent_record"

#: Per-container state that is neither the record nor the spool, relative to the
#: repo root: one ``<identity>/`` directory per container (the preflight-refused
#: marker, the sidecar's revocations).
STATE_DIR_RELPATH = f"{STATE_DIR_NAME}/state"

#: The subdirectory of
#: :data:`osprey_connectors.workspace.SIMULATOR_LOG_DIR_RELPATH` the live
#: stand-in instance writes, so a record's directory names the machine that
#: wrote it.
SIMULATOR_STANDIN_LOG_SUBDIR = "standin"

#: The directories that make up the state zone, created empty and otherwise
#: the agent's to write. Both ``osprey init`` and ``osprey build`` guarantee they
#: exist — a fresh clone carries no git-ignored directory, so whichever command
#: runs first has to make them, and a repo that was reset must end up looking
#: like one that was freshly cloned. Spelled once for exactly that reason: two
#: commands agreeing by coincidence is how a reset and a clone start to differ.
STATE_ZONE_DIRS: tuple[str, ...] = (DEFAULT_AGENT_DATA_BASE_DIR,)

#: Subdirectory of the agent-data root holding the files operators attach in the
#: operator chat. The web terminal stores them there and hands the agent
#: their absolute paths; the rendered permissions allow the agent to read exactly
#: that directory. Spelled once so the store and the rule cannot name two places.
CHAT_UPLOADS_SUBDIR = "uploads"

#: Config keys whose value names a path something writes to *at run time*.
#: These must stay out of ``data/``: that tree is build-owned and checksummed by
#: :func:`osprey.cli.templates.manifest.calculate_file_checksums`, so a runtime
#: write landing there reads as project drift and is erased by the next
#: ``osprey build``. The protected-key table unpacks this tuple. The connector
#: package warns about the same keys when it loads a config, from its own
#: private copy; a test holds the two equal.
RUNTIME_WRITE_PATH_KEYS = (SIMULATION_STATE_DIR_CONFIG_KEY,)

#: The rendered config key carrying the paths the profile names: every input
#: the deployment's agent reads its instructions, code or data from. It is the
#: opposite of :data:`RUNTIME_WRITE_PATH_KEYS` — nothing writes these at run
#: time, so the python executor adds them to the roots agent code may not
#: write. The value is a list of strings, each absolute or relative to the
#: rendered ``project_root``; ``~`` and ``${VAR}`` placeholders are written
#: verbatim and expanded by :func:`anchor_protected_entry`, because the build's
#: environment is not the run-time one. An empty list is a valid set. Only the
#: build writes it, every config file one build writes carries the same set,
#: and only :func:`read_protected_paths` interprets it. Spelled here, below
#: ``osprey.cli``, because the executor reads it and must not import the CLI.
PROTECTED_PATHS_CONFIG_KEY: Final = "protected_paths"

#: What a render without the protected-path list costs, in the words every
#: place that reports it repeats: the executor's refusal, and the start verbs,
#: ``status``, ``chat`` and ``query`` that predict it, so an operator meets the
#: refusal already explained.
NO_PROTECTED_PATHS_CONSEQUENCE: Final = (
    "the python executor refuses agent code until `osprey build`"
)

_logger = logging.getLogger(__name__)


class ProtectedPathsState(enum.Enum):
    """What a config says about :data:`PROTECTED_PATHS_CONFIG_KEY`."""

    #: The key holds a list of strings — possibly empty, which is a valid set.
    PRESENT = "present"
    #: The key is not in the config at all: a build that predates it.
    ABSENT = "absent"
    #: The key is there but is not a list of non-empty strings.
    MALFORMED = "malformed"


@dataclass(frozen=True)
class ProtectedPathsReading:
    """The verdict of :func:`read_protected_paths`.

    Attributes:
        state: Whether the key is present, absent or malformed.
        entries: The entries as written, in order; empty unless ``state`` is
            :attr:`ProtectedPathsState.PRESENT`.
    """

    state: ProtectedPathsState
    entries: tuple[str, ...] = ()

    @property
    def present(self) -> bool:
        """Whether the config carries a well-formed set."""
        return self.state is ProtectedPathsState.PRESENT


def read_protected_paths(config: object) -> ProtectedPathsReading:
    """Read :data:`PROTECTED_PATHS_CONFIG_KEY` from a loaded config.

    The one interpreter of the key: the executor and the drift check both ask
    it, so "carries the set" means one thing. Anything but a list of strings
    is malformed — ``null``, a bare string, a mapping, a list holding a
    non-string. An empty string is malformed too: anchored, it would name the
    project root itself, which the build never emits.

    Args:
        config: The parsed config document. ``None`` is a config with no keys;
            any other non-mapping cannot carry the key and is malformed.

    Returns:
        The state, and the entries when the state is present.
    """
    if config is None:
        return ProtectedPathsReading(ProtectedPathsState.ABSENT)
    if not isinstance(config, Mapping):
        return ProtectedPathsReading(ProtectedPathsState.MALFORMED)
    if PROTECTED_PATHS_CONFIG_KEY not in config:
        return ProtectedPathsReading(ProtectedPathsState.ABSENT)
    value = config[PROTECTED_PATHS_CONFIG_KEY]
    if not isinstance(value, list) or not all(isinstance(e, str) and e for e in value):
        return ProtectedPathsReading(ProtectedPathsState.MALFORMED)
    return ProtectedPathsReading(ProtectedPathsState.PRESENT, tuple(value))


def load_protected_paths(config_path: Path) -> tuple[tuple[str, ...], str | None]:
    """Read :data:`PROTECTED_PATHS_CONFIG_KEY` from a rendered config file.

    The file is read and parsed directly, never through a configuration cache,
    and judged by :func:`read_protected_paths`.

    Args:
        config_path: The rendered ``config.yml``.

    Returns:
        ``(entries, None)`` when the file carries a well-formed list, possibly
        empty; ``((), finding)`` otherwise, *finding* a clause naming the file
        and what is wrong: it does not exist, cannot be read or parsed, lacks
        the key, or carries it malformed.
    """
    import yaml

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return (), f"{config_path} does not exist"
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return (), f"{config_path} could not be read"
    reading = read_protected_paths(config)
    if reading.present:
        return reading.entries, None
    if reading.state is ProtectedPathsState.MALFORMED:
        return (), (
            f"{config_path} has a `{PROTECTED_PATHS_CONFIG_KEY}` that is not a list of paths"
        )
    return (), f"{config_path} was rendered without the protected-path list"


@dataclass(frozen=True)
class AnchoredEntry:
    """One protected-path entry, anchored.

    Attributes:
        entry: The entry as the config spells it.
        path: The resolved location it names, or ``None`` when a placeholder
            in it names a variable the environment does not have.
        unresolved: The placeholders left after expansion; empty when none.
        inside_permitted: The permitted root the location lies in, when it
            lies in one; ``None`` otherwise.
    """

    entry: str
    path: Path | None
    unresolved: tuple[str, ...] = ()
    inside_permitted: Path | None = None

    @property
    def protects(self) -> bool:
        """Whether the entry adds a protected root.

        An entry with an unresolved placeholder names no location, and one
        inside a permitted root names a place agent code is meant to write.
        """
        return self.path is not None and self.inside_permitted is None


def anchor_protected_entry(
    entry: str,
    project_root: Path | str,
    permitted_roots: Iterable[Path | str] = (),
    *,
    environ: Mapping[str, str] | None = None,
) -> AnchoredEntry:
    """Anchor one :data:`PROTECTED_PATHS_CONFIG_KEY` entry at a deployment.

    Expands environment placeholders with the configuration loader's own
    function, then ``~``, then anchors a relative entry at *project_root*
    (:func:`~osprey_connectors.workspace.anchored_path`). There is no
    existence test: a location that does not exist yet is still protected,
    so creating it is refused too. Used at run time by the executor and at
    build time by the emitter, which is why the permitted roots are passed
    in rather than computed here.

    Args:
        entry: One entry, as :func:`read_protected_paths` returned it.
        project_root: The rendered ``project_root`` relative entries anchor at.
        permitted_roots: The roots agent code may write; an entry that lands
            in one (or on one) protects nothing.
        environ: The environment placeholders expand against; the process
            environment when ``None``.

    Returns:
        The anchored entry, saying whether and why it protects nothing.
    """
    # Deferred: the loader module pulls in yaml and the config builder, and
    # this module is imported by the executor at start-up.
    from osprey_connectors.config import resolve_env_vars, unresolved_placeholders

    expanded = resolve_env_vars(entry, environ=environ)
    left = tuple(unresolved_placeholders(expanded))
    if left:
        return AnchoredEntry(entry=entry, path=None, unresolved=left)
    path = anchored_path(expanded, Path(project_root)).resolve()
    for root in permitted_roots:
        resolved_root = Path(root).resolve()
        if path == resolved_root or resolved_root in path.parents:
            return AnchoredEntry(entry=entry, path=path, inside_permitted=resolved_root)
    return AnchoredEntry(entry=entry, path=path)


#: Entries already reported as protecting nothing, so a server process says
#: so once per entry rather than once per execution request.
_reported_unprotecting: set[tuple[str, str]] = set()


def protected_roots_from_entries(
    entries: Iterable[str],
    project_root: Path | str,
    permitted_roots: Iterable[Path | str] = (),
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """The protected roots a set of entries adds, in entry order.

    Each entry goes through :func:`anchor_protected_entry`. One that protects
    nothing raises no error: it is logged as a warning, once per entry per
    process, and left out.

    Args:
        entries: The entries of a present :class:`ProtectedPathsReading`.
        project_root: The rendered ``project_root``.
        permitted_roots: The roots agent code may write.
        environ: The environment placeholders expand against; the process
            environment when ``None``.

    Returns:
        The anchored location of every entry that protects something.
    """
    permitted = tuple(permitted_roots)
    roots: list[Path] = []
    for stated in entries:
        anchored = anchor_protected_entry(stated, project_root, permitted, environ=environ)
        if anchored.path is not None and anchored.inside_permitted is None:
            roots.append(anchored.path)
            continue
        if anchored.path is None:
            reason = f"it holds an unset placeholder ({', '.join(anchored.unresolved)})"
        else:
            reason = f"it lies inside the writable root {anchored.inside_permitted}"
        marker = (stated, reason)
        if marker not in _reported_unprotecting:
            _reported_unprotecting.add(marker)
            _logger.warning(
                "%s entry %r protects nothing: %s.", PROTECTED_PATHS_CONFIG_KEY, stated, reason
            )
    return tuple(roots)


def session_hold_dir(project_root: Path | str) -> Path:
    """The root of a deployment's session holds, one ``<key>/record.json`` per session.

    Under the state zone, a root executed code may not write.
    """
    return Path(project_root) / STATE_DIR_RELPATH / "session_holds"


def container_image_context(repo_root: Path | str, project_name: str) -> Path:
    """The directory an image for *project_name* is built from.

    The build writes it (``build_cmd._render_container_project``) and the deploy
    layer builds from it, so the path is assembled here rather than spelled at
    both ends. It is a deployment repo in its own right: the ``Dockerfile`` a
    build uses lives at
    :data:`~osprey_connectors.workspace.RENDERED_CONFIG_RELPATH`'s sibling
    inside it, not at its root, because the root is the repo and the render is
    one level down.
    """
    return Path(repo_root) / BUILD_DIR_NAME / IMAGE_DIR_NAME / project_name


def deployment_env_chain(config_path: Path | str) -> list[Path]:
    """Every env-chain file of the deployment whose config is *config_path*.

    There is one env chain per deployment and it sits at the repo ROOT, beside
    ``profile.yml`` — never in the render, which every ``osprey build``
    re-creates from scratch. So the config's own directory is the wrong answer
    on a host, where the config is ``<repo>/build/config.yml``, and the right
    one in a container, whose project directory *is* the render: the repo
    root's chain when any member exists there, else the config's own
    directory. Paths come back in ascending precedence — ``.env.shared`` before
    ``.env`` — so an ``override=True`` loader can walk the list as-is and land
    on local-wins.

    Every member path is returned whether or not the file exists. A watcher
    has to stat the places a member could *appear*, or a ``.env`` created
    after start-up never invalidates anything.

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
