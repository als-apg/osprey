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
else about the layout is the framework's, and is spelled here. Two names here
are also spelled privately inside the connector package, because its resolvers
need them: :data:`PROFILE_FILENAME` and :data:`STATE_DIR_NAME`. A test holds
each pair equal.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

from osprey_connectors.dotenv import ENV_CHAIN_FILENAMES
from osprey_connectors.workspace import (
    BUILD_DIR_NAME,
    DEFAULT_AGENT_DATA_BASE_DIR,
    SIMULATION_STATE_DIR_CONFIG_KEY,
    repo_root_for_config,
)

__all__ = [
    "APPS_DIRNAME",
    "ARCHIVE_DIR_RELPATH",
    "CHAT_UPLOADS_SUBDIR",
    "IMAGE_DIR_NAME",
    "LAYER_DIRNAMES",
    "LAYER_DIR_KINDS",
    "PRE_UPGRADE_AUDIT_RELPATH",
    "PROFILE_FILENAME",
    "RECORD_DIR_RELPATH",
    "RETIRED_LAYER_DIRS",
    "RUNTIME_DATA_DIR_NAME",
    "RUNTIME_WRITE_PATH_KEYS",
    "SIMULATOR_STANDIN_LOG_SUBDIR",
    "SPOOL_DIR_RELPATH",
    "STATE_DIR_NAME",
    "STATE_DIR_RELPATH",
    "STATE_ZONE_DIRS",
    "TIERS_DIRNAME",
    "container_image_context",
    "deployment_env_chain",
    "layer_dir_kind",
    "repo_root_for_agent_data",
]

#: The durable state zone inside a deployment repo: ``var/`` survives every
#: ``osprey build``, which re-creates ``build/`` wholesale.
STATE_DIR_NAME = "var"

#: The file that makes a directory a deployment repo root — the marker the
#: CLI's repo walk and the hook library's anchor both look for.
PROFILE_FILENAME = "profile.yml"

#: Directory beside ``profile.yml`` (or in the bundled presets) holding the
#: logins: one file per login of the deployment's agent.
TIERS_DIRNAME: Final = "tiers"

#: Directory beside ``profile.yml`` holding the applications, one
#: ``apps/<name>/`` directory each. Not a layer directory: no file in it is a
#: login.
APPS_DIRNAME: Final = "apps"

#: Each live layer directory -> the kind word its files are read as. The
#: listing order is the order every layer listing follows. Spelled here, below
#: ``cli``, because the hook library and the runtime read the layout too.
LAYER_DIR_KINDS: Mapping[str, str] = MappingProxyType({TIERS_DIRNAME: "tier"})

#: Each retired layer directory -> the live directory its files are read as,
#: with one warning naming the move. Closing the deprecation window is deleting
#: the entry.
RETIRED_LAYER_DIRS: Mapping[str, str] = MappingProxyType(
    {"personas": TIERS_DIRNAME}  # retired-spelling
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
#: Simple view's chat. The web terminal stores them there and hands the agent
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
