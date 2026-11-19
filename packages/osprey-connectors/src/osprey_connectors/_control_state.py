"""Where the control state is, and what this process's environment says about it.

The write switch (:mod:`osprey_connectors.write_switch`) and the record it
reads (:mod:`osprey_connectors.control_context`) both need to know where the
control state lives and what the environment a process was started in says:
which directory is this identity's, whether a read-only tree of every
identity's directory is bound in, whether the process is a dispatch job, and
what launch pin an executor sandbox carries. This module answers those
questions, from the environment and the config root alone, and holds the file
primitives every file in the state directory is written and swept with.

It is private. Its names are the package's own seam between the record, the
switch and the write authorization, so none of them reaches into another's
private names; the public ones are re-exported by
:mod:`osprey_connectors.write_switch` (the paths, the binds, the dispatch mark
and the launch pin) and :mod:`osprey_connectors.control_context` (the file
primitives), and are documented there.

Rule numbers here are the switch's (rule 1: where the state is) and the
record's (rules 3 and 4: how a file in the state directory is written and
when a change to one is seen).

It sits below the record: it imports nothing from
:mod:`~osprey_connectors.control_context` or the switch, and nothing from
``osprey``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, Literal, get_args

from osprey_connectors.connector_types import CONTROL_TARGETS
from osprey_connectors.identity import acting_identity
from osprey_connectors.safety_file import SHARED_FILE_MODE, atomic_replace, ensure_shared_dir
from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR

logger = logging.getLogger("osprey_connectors._control_state")

#: The anchor stamp. Read by NAME rather than imported from
#: ``osprey.audit.posture``, which declares it for the stamping side: this
#: package is the lean connector chain and must not grow an ``osprey`` import
#: to learn one string.
_AGENT_DATA_ROOT_ENV_VAR = "OSPREY_AGENT_DATA_ROOT"


#: Subdirectory of the agent-data root holding one directory per identity, each
#: with that identity's control-context record, per-server reports and request
#: files. The same tree ``target_state.state_dir()`` resolves, spelled here
#: because a record the writer puts in one directory and a reader looks for in
#: another is a switch that silently never applies.
CONTROL_STATE_DIR_NAME = "control_target"


#: The container's OWN state directory, bound in by compose and named in the
#: environment of every container that holds a chip: the terminal's
#: ``web-<user>``. Top rung of rule 1, and already the per-identity directory —
#: a container's bind target has nothing to do with the host tree's shape, so
#: the two fixed hops are the host derivation's business and never appended to
#: this. Distinct from :data:`CONTROL_CONTEXT_TREE_ENV_VAR`, the read-only bind
#: of every identity's directory, which names no single one of them.
#:
#: It belongs in ``NON_PINNABLE_AUDIT_MARKERS`` for the same reason the audit
#: identity does: a spec that could pin it could point one user's writes at
#: another user's switch.
CONTROL_CONTEXT_DIR_ENV_VAR = "OSPREY_CONTROL_CONTEXT_DIR"


#: The mark the dispatch worker's ``sdk_runner`` sets on every job it runs: a
#: scheduled or triggered job, started by no login at a terminal. Only the value
#: ``"1"`` marks a dispatch run; anything else, ``"0"`` included, is no dispatch.
#: A dispatch run that names no owner resolves
#: :data:`~osprey_connectors.write_switch.NO_OWNER` and is refused
#: ``owner_unscheduled`` (:func:`~osprey_connectors.write_switch.write_verdict`):
#: the process account it runs as is no login whose switch could cover it.
DISPATCH_RUN_ENV_VAR = "OSPREY_DISPATCH_RUN"


#: The read-only bind of the whole per-user control-state tree, named in the
#: environment of a container that holds no chip of its own: a lane's
#: queueserver, the dispatch worker. Only its PRESENCE is read here; the tree is
#: read by the verdict.
#:
#: Presence is what makes such a container owned-or-nothing. The account it runs
#: as names nobody whose chip anyone ever set, so falling through to
#: :func:`~osprey_connectors.identity.acting_identity` there would read a record
#: directory that does not exist, so a plan would be decided by nobody's
#: switch rather than its owner's.
CONTROL_CONTEXT_TREE_ENV_VAR = "OSPREY_CONTROL_CONTEXT_TREE"

#: The marker file at the root of the control-context tree, written by
#: ``osprey build``. Its PRESENCE is the tree reader's first question, because a
#: read-only bind whose host source does not exist is created empty by the
#: container runtime — after which an unprovisioned tree is indistinguishable
#: from a tree in which nobody has switched anything on, and a provisioning
#: fault reads as an operator's answer on exactly the deployment that never
#: provisioned it.
#:
#: Restated here rather than imported from
#: ``osprey.deployment.compose_generator``, which declares it for the writing
#: side: this package is the lean connector chain and may not import the
#: deployment layer. The two spellings are pinned together by a test that
#: imports both — a drift between them is a tree that reads as unprovisioned in
#: every container.
CONTROL_TREE_MARKER_NAME = ".osprey-control-tree"


# -- path resolution --------------------------------------------------------


def absolute_bind(variable: str) -> Path | None:
    """The bind *variable* names as an absolute path, or ``None`` when it names none.

    The one normalisation every bind in rule 1 goes through, so the container's
    own state directory and the read-only tree cannot read their environment two
    different ways.

    Three answers are "no bind": unset, blank or whitespace-only, and a value
    that is not absolute after expansion. A bind names a path in the container's
    own filesystem — the compose file is where it is written — so a relative one
    would resolve against whatever directory the process happened to start in,
    which is a directory nothing binds and no writer creates. It is a deployment
    mistake rather than a degraded state, and it gets the one warning line that
    says which variable to fix; falling back to the derivation below it is what
    keeps a mistake here from raising into a write path.

    Expansion is :func:`os.path.expanduser` rather than
    :meth:`pathlib.Path.expanduser`, which raises ``RuntimeError`` for a ``~``
    naming a user this container has no passwd entry for — a container binds
    paths from another host's layout, so that is an ordinary value here and must
    degrade to "no bind" rather than to an exception. The stdlib hook restates
    this rule with the same call.
    """
    raw = (os.environ.get(variable) or "").strip()
    if not raw:
        return None
    expanded = os.path.expanduser(raw)
    if not os.path.isabs(expanded):
        logger.warning(
            "%s=%s is not an absolute path, so it names no bind: a bind is a path "
            "in this container's own filesystem. Set it to an absolute path or "
            "unset it.",
            variable,
            raw,
        )
        return None
    return Path(expanded)


def stamped_agent_data_root() -> Path | None:
    """The :data:`_AGENT_DATA_ROOT_ENV_VAR` stamp as a path, or ``None`` unstamped.

    The stamp half of rule 1 on its own, so that the files under this root
    cannot read one environment variable two different ways. A blank or
    whitespace-only stamp is no stamp, and a ``~`` in it is expanded: a reader
    that took either literally would look in a directory no writer ever
    creates, which is the silently-ignored-switch failure rule 1 exists to prevent.
    The expansion is :func:`os.path.expanduser`, for the reason
    :func:`absolute_bind` uses it: the ``Path`` method raises for a ``~``
    naming a user this process cannot resolve, and this is read on a write path.

    Unlike a bind, a stamp is not required to be absolute: it is written by a
    process that shares this one's working directory rather than by a compose
    file describing another host's layout.

    ``osprey.mcp_server.control_system.target_state.state_dir`` calls this
    rather than restating it. The CONFIG half below is deliberately not shared:
    that module raises where this one answers ``None``, and a route turns its
    raise into a ``store_unavailable`` 503.
    """
    stamped = (os.environ.get(_AGENT_DATA_ROOT_ENV_VAR) or "").strip()
    return Path(os.path.expanduser(stamped)) if stamped else None


def _agent_data_root() -> Path | None:
    """The agent-data root the control state lives under.

    Rule 1: the :data:`_AGENT_DATA_ROOT_ENV_VAR` stamp
    when it names a non-blank path, else the config derivation. ``None`` when
    neither answers — a deployment whose project root cannot be resolved has
    no record, which is a different thing from having one with every switch off.
    """
    stamped = stamped_agent_data_root()
    if stamped is not None:
        return stamped
    try:
        return _config_agent_data_root()
    except Exception:  # an unresolvable root is "no record", not a crash
        logger.debug("Could not resolve the shared data root for the control state", exc_info=True)
        return None


def _config_agent_data_root() -> Path:
    """The process config's shared agent-data root; without a config, the default under the repo.

    A config that does not load still has a root: ``var/agent_data`` under the
    repo the process stands in (:func:`~osprey_connectors.workspace.locate_repo_root`).
    """
    # Looked up at the call, so a suite that diverts the no-config ladder
    # reaches this caller too.
    from osprey_connectors import workspace
    from osprey_connectors.config import current_config

    try:
        config = current_config()
    except Exception:  # a config that does not load, for any reason, names no root of its own
        return (workspace.locate_repo_root() / DEFAULT_AGENT_DATA_BASE_DIR).resolve()
    return config.agent_data_root


def bound_state_dir() -> Path | None:
    """The :data:`CONTROL_CONTEXT_DIR_ENV_VAR` bind as a path, or ``None`` unset.

    The top rung of rule 1 on its own, so that the three seams resolving this
    directory cannot read one environment variable three different ways. Blank,
    whitespace-only and non-absolute values all name no bind, and a ``~`` is
    expanded — :func:`absolute_bind` is that rule, shared with the tree bind
    beside it.

    ``osprey.mcp_server.control_system.target_state.state_dir`` calls this
    rather than restating it, exactly as it calls
    :func:`stamped_agent_data_root`.
    """
    return absolute_bind(CONTROL_CONTEXT_DIR_ENV_VAR)


def state_dir() -> Path | None:
    """This identity's control-state directory, or ``None``.

    Rule 1: the :func:`bound_state_dir` bind when a
    container carries one, else the acting identity under
    :data:`CONTROL_STATE_DIR_NAME` under :func:`_agent_data_root`.

    The identity is resolved on every call rather than once at import, because
    :func:`~osprey_connectors.identity.acting_identity` reads an environment
    that compose and the entrypoint set per process; a value frozen at import
    would be whatever the first importer happened to see. It never fails —
    an unresolvable identity is :data:`~osprey_connectors.identity.UNKNOWN_IDENTITY`,
    a real directory holding the state of whoever could not be named, which is
    a switch that applies rather than one that silently does not.

    A ``None`` here is what a route reports as ``store_unavailable``: there is
    nowhere to record a switch that any reader would find.
    """
    bound = bound_state_dir()
    if bound is not None:
        return bound
    root = _agent_data_root()
    return None if root is None else root / CONTROL_STATE_DIR_NAME / acting_identity()


def tree_is_bound() -> bool:
    """Whether this process was handed the read-only tree at all — usable or not.

    The ONE spelling of that question, because two readings of it are a
    split-brain on the write path:
    :func:`~osprey_connectors.write_switch.current_owner` answers
    :data:`~osprey_connectors.write_switch.NO_OWNER` for a container that holds
    the tree (it holds no chip of its own), while
    :func:`~osprey_connectors.write_switch.write_verdict` picks the reader that
    governs the write. A variable that made one of them say "tree" and the other
    say "host" would send a container with no host record to the host reader,
    which answers for this host's login and not for the owner of the work — the
    split the tree reader exists to close.

    So this asks only whether a value is THERE. Whether the value is usable is
    :func:`control_context_tree`'s question, and a bind that is set but
    unusable is a deployment mistake that refuses rather than falls back.
    """
    return bool((os.environ.get(CONTROL_CONTEXT_TREE_ENV_VAR) or "").strip())


def control_context_tree() -> Path | None:
    """The :data:`CONTROL_CONTEXT_TREE_ENV_VAR` bind as a path, or ``None`` unset.

    The read-only bind of EVERY identity's directory, carried by a container
    that holds no chip of its own. It is not a rung of :func:`state_dir`: it
    names no single identity's directory, so nothing reads a record through it
    without first being told whose record to read — which is what
    :func:`~osprey_connectors.control_context.read_tree_context` is for.

    Normalised by :func:`absolute_bind`, the same rule the container's own
    state directory goes through, so the two binds cannot disagree about what a
    ``~`` or a relative path means.
    """
    return absolute_bind(CONTROL_CONTEXT_TREE_ENV_VAR)


def tree_bind_unusable() -> str:
    """The remedy for a bind that is set and names no directory to open.

    Its own wording rather than the unprovisioned one: nothing is missing on
    the deployment's disk, the variable simply does not name a path this
    container can resolve, and the operator repairs the compose file rather
    than building anything.
    """
    return (
        f"{CONTROL_CONTEXT_TREE_ENV_VAR} names no usable path, so the read-only "
        "control-context tree could not be opened — the bind must name an "
        "absolute path in this container"
    )


# -- the dispatch mark -------------------------------------------------------


def is_dispatch_run() -> bool:
    """Whether this process is a dispatch job: :data:`DISPATCH_RUN_ENV_VAR` is ``"1"``."""
    return os.environ.get(DISPATCH_RUN_ENV_VAR) == "1"


# -- the launch-time pin ----------------------------------------------------

#: The two values a launch pin may spell. Everything else is unreadable.
LaunchPin = Literal["sandbox", "writes"]

#: The launch-pin value: the run started with the machine's writes off. The
#: only pin value that ever refuses anything.
LAUNCH_PIN_SANDBOX: Final[LaunchPin] = "sandbox"

#: The run started with the machine's writes on. Recorded so a stamp states
#: what the run launched under; it never grants anything.
LAUNCH_PIN_WRITES: Final[LaunchPin] = "writes"

_VALID_PINS: frozenset[LaunchPin] = frozenset(get_args(LaunchPin))


#: The launch-time pin, stamped into an executor sandbox's environment by
#: ``osprey.mcp_server.python_executor.executor._apply_target_stamp`` and read
#: back HERE, inside that sandbox, by :func:`launch_pin_term`.
#:
#: The wire format is one ``"<target>=<posture>"`` pair and nothing else:
#:
#: * ``<target>`` is the control target the run was stamped against — a member
#:   of :data:`~osprey_connectors.connector_types.CONTROL_TARGETS` — or
#:   :data:`_LAUNCH_PIN_ALL_TARGETS` when the executor could not identify
#:   one, in which case the pin covers every target;
#: * ``<posture>`` is :data:`LAUNCH_PIN_SANDBOX` or :data:`LAUNCH_PIN_WRITES`.
#:
#: ``sandbox`` is the only value that does anything: ``writes`` is recorded so
#: the in-flight marker states what the run launched under, and is inert here:
#: a stamp can refuse and never grant. A missing or blank stamp is inert too,
#: which is what makes every process that is not an executor sandbox
#: unaffected by this term; an unparseable one refuses (:func:`launch_pin_term`).
#:
#: Spelled here rather than imported because this is the reading side and the
#: reading side runs in the lean connector chain;
#: ``registry/mcp.py`` and the executor import the name from this module, and
#: it is stripped from every rendered ``.mcp.json`` env block
#: (``NON_PINNABLE_AUDIT_MARKERS``) — a spec that could pin it could hand a
#: run launched with writes off the writes it was denied.
LAUNCH_PIN_ENV_VAR = "OSPREY_LAUNCH_POSTURE"

#: The launch stamp's target when the executor could not name one. Covers every
#: target: a run that could not say which machine it is about must not be the
#: one run a switched-off machine fails to reach.
_LAUNCH_PIN_ALL_TARGETS = "*"


def launch_pin_stamp(target: str | None, pin: LaunchPin) -> str:
    """Compose the :data:`LAUNCH_PIN_ENV_VAR` value for one launch.

    The one place the wire format is written, so the executor that stamps it
    and :func:`launch_pin_term` which reads it cannot disagree about the
    separator or about how "no target" is spelled.

    Args:
        target: The control target the run is stamped against, or ``None`` when
            the executor could not identify one — spelled
            :data:`_LAUNCH_PIN_ALL_TARGETS`, which covers every target.
        pin: :data:`LAUNCH_PIN_SANDBOX` or :data:`LAUNCH_PIN_WRITES` — the
            recorded answer for that target at the moment of launch.
    """
    return f"{target or _LAUNCH_PIN_ALL_TARGETS}={pin}"


def launch_pin_term(target: str | None) -> Literal["none", "pinned", "unreadable"]:
    """The launch pin as a term of the write verdict — fail-closed on a bad stamp.

    Reads :data:`LAUNCH_PIN_ENV_VAR` on every call and answers, for *target*:

    * ``"none"`` — no stamp (absent or blank), or a ``<t>=writes`` stamp whose
      ``<t>`` is a member of :data:`~osprey_connectors.connector_types.CONTROL_TARGETS` or
      :data:`_LAUNCH_PIN_ALL_TARGETS`, or a ``<t>=sandbox`` stamp naming a
      different target than *target*;
    * ``"pinned"`` — a ``<t>=sandbox`` stamp with ``<t>`` equal to *target*, or
      ``<t>`` :data:`_LAUNCH_PIN_ALL_TARGETS`. A caller that names no target
      (``None``) is pinned by any sandbox stamp: the most restrictive answer
      wins;
    * ``"unreadable"`` — anything else, for every target: no separator, an empty
      or unknown target, a posture that is neither :data:`LAUNCH_PIN_SANDBOX` nor
      :data:`LAUNCH_PIN_WRITES`.

    ``pinned`` and ``unreadable`` both refuse as ``launch_pinned``; a stamp is
    only ever written by the executor, so one this module cannot read is a run
    whose launch posture is unknown, and that must not widen to writable.
    Surrounding whitespace on the stamp and on each half is ignored.
    """
    text = (os.environ.get(LAUNCH_PIN_ENV_VAR) or "").strip()
    if not text:
        return "none"
    stamped, separator, launch_posture = text.partition("=")
    stamped = stamped.strip()
    launch_posture = launch_posture.strip()
    if (
        not separator
        or launch_posture not in _VALID_PINS
        or (stamped != _LAUNCH_PIN_ALL_TARGETS and stamped not in CONTROL_TARGETS)
    ):
        return "unreadable"
    if launch_posture != LAUNCH_PIN_SANDBOX:
        return "none"
    if stamped in (_LAUNCH_PIN_ALL_TARGETS, target) or not target:
        return "pinned"
    return "none"


# -- files in the state directory -------------------------------------------


def file_signature(path: Path | None) -> tuple[int, int, int] | None:
    """``(mtime_ns, size, inode)`` for *path*, or ``None`` when it is absent.

    Rule 4 of the record's contract, and the same answer for everything that
    watches one of these files: the record's cache re-parses on it, and the
    reconciler wakes on it. The inode is the third element because
    :func:`write_json_atomic` replaces the file rather than rewriting it, so
    two switches inside one filesystem clock tick differ there and nowhere
    else. An unresolvable path has no signature, which reads as "no file".
    """
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write *payload* to *path* as JSON, replacing any existing file. Rule 3.

    The file is replaced whole through ``safety_file.atomic_replace``, so a
    failed write leaves neither half a file nor litter beside it.

    Every file in the state directory is written this way, the record and the
    per-server reports and requests alike: they are read by processes that
    never coordinate with the writer, so a reader must never be able to see
    half of one.

    The mode is load-bearing, not a preference: those readers run under other
    uids — a dispatch worker at uid 1000 reading what the host account wrote —
    and reach the file through the setgid group of the state directory, so the
    file is given 0640 whatever the umask: under umask 077 a record only the
    owner can read makes every owned lane write refuse
    ``control_context_unavailable``, while CI at umask 022 stays green and never
    sees it. Every directory this writer creates — the
    levels above the leaf as much as the leaf — is set to exactly 0o2770 rather
    than merely widened, because a directory left at the umask 0755 would let
    ``other`` list the identity names (``safety_file.ensure_shared_dir``).

    Raises:
        OSError: The write itself failed.
    """
    ensure_shared_dir(path.parent)
    atomic_replace(
        path,
        json.dumps(payload, indent=2).encode("utf-8"),
        mode=SHARED_FILE_MODE,
        sync="best_effort",
    )


_PROC_ROOT = "/proc"


def _is_thread_id(pid: int) -> bool:
    """Whether the kernel knows *pid* as a thread of some other process.

    Linux draws PIDs and thread ids from one number space, and ``/proc/<tid>``
    opens by number though no listing shows it. A process is its own thread
    group leader, so a ``Tgid`` that differs from *pid* means a thread. ``False``
    when ``/proc`` says nothing, which is every platform without one.
    """
    try:
        with open(f"{_PROC_ROOT}/{pid}/status", "rb") as status:
            for line in status:
                if line.startswith(b"Tgid:"):
                    return int(line.split()[1]) != pid
    except (OSError, ValueError, IndexError):
        return False  # no /proc, or one that does not read like Linux's: no claim either way
    return False


def is_process_alive(pid: object) -> bool:
    """Whether *pid* names a running process.

    ``os.kill(pid, 0)`` sends no signal and only asks the kernel whether the
    process exists — the ``osprey web`` PID-file check does the same. A
    ``PermissionError`` means the process exists but belongs to another user, so
    it counts as ALIVE: sweeping a file whose owner is merely unreachable would
    delete live state. Non-positive PIDs are rejected without calling
    ``os.kill`` at all, because 0 and negatives address process *groups*.

    ``os.kill`` also answers for a thread id, which is no process: a PID kept
    in a record that outlived its container can land on a thread of the
    process reading it, which would then follow itself for ever
    (:func:`_is_thread_id`). A real process on that PID is not told apart.

    Anything that is not an ``int`` names no process and is ``False`` without
    a call either, so a field read off a file can be handed over unconverted.
    That includes ``bool``: ``True`` is ``1`` to ``os.kill``, and PID 1 is
    always alive, so a report carrying ``"server_pid": true`` would otherwise
    read as a running server forever.
    """
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # another user's process still exists; the thread check below applies to it too
    except OSError:  # pragma: no cover - platform oddity; assume alive
        return True
    return not _is_thread_id(pid)


def _pid_from_filename(name: str, prefix: str, suffix: str) -> int | None:
    """The PID *name* encodes, or ``None`` when it encodes none.

    Shared by every family of file in the state directory: each is named for the
    process that owns it precisely so a sweeper can judge it without opening
    it, and a name that does not follow the rule is judged the same way in all
    of them — no PID, therefore no live owner.
    """
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    stem = name[len(prefix) : -len(suffix)]
    try:
        return int(stem)
    except ValueError:
        return None


def sweep_dead(
    directory: Path,
    *,
    prefix: str,
    suffix: str,
    keep: str | None = None,
    is_alive: Callable[[object], bool] = is_process_alive,
) -> list[Path]:
    """Unlink every ``prefix<pid>suffix`` file in *directory* whose PID is gone.

    A process that is killed rather than asked to stop leaves its file behind,
    and a later process that happens to reuse the PID would otherwise inherit
    it. The name is the whole judgement — a file whose name encodes no PID
    encodes no owner either, and goes the same way.

    Args:
        directory: Where to look. A missing or unreadable one sweeps nothing.
        prefix: Filename prefix of the family to sweep.
        suffix: Filename suffix of the family to sweep.
        keep: A filename to leave alone without probing it. This is how a
            process spares its own file: asking whether oneself is alive is a
            way to get it wrong.
        is_alive: The liveness predicate.

    Returns:
        The paths removed, in discovery order.
    """
    try:
        entries = sorted(directory.glob(f"{prefix}*{suffix}"))
    except OSError:
        return []

    removed: list[Path] = []
    for entry in entries:
        if keep is not None and entry.name == keep:
            continue
        pid = _pid_from_filename(entry.name, prefix, suffix)
        if pid is not None and is_alive(pid):
            continue
        try:
            entry.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - unwritable state dir
            logger.warning("Could not remove stale file %s: %s", entry, exc)
            continue
        logger.info("Swept stale file %s (owner pid %s gone)", entry.name, pid)
        removed.append(entry)
    return removed
