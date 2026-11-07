"""Which artifacts a session owns: one scope type and one predicate.

An entry is owned by a scope when the scope resolved a transcript, the entry is
operator-visible (no dispatch run, no watchdog judge) and :func:`owner_of` over
the scope's ledger names that transcript. Every surface asks :func:`owned_by`
instead of re-deriving the rule. The predicate is pure: it reads no
environment, file or clock.

The resolver answers which scope a process acts for. :func:`resolve_scope`
applies the rules to inputs already in hand; :func:`current_scope` gathers them
from an environment mapping and the CLI transcript pointer. Nothing here picks
the newest transcript file: a process that names no session owns nothing.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from osprey.stores.artifact_store import ArtifactEntry, Ledger, operator_visible, owner_of
from osprey.stores.session_ledger import EPOCH, LedgerRow, load_checked
from osprey.stores.session_ledger import Ledger as LedgerRows

__all__ = [
    "CLI_POINTER_DIR",
    "ScopeKind",
    "SessionScope",
    "current_scope",
    "export_cli_launch",
    "owned_by",
    "read_ledger",
    "resolve_scope",
    "scope_for",
    "session_tag",
]

logger = logging.getLogger(__name__)

ScopeKind = Literal["web", "cli", "dispatch", "none"]


@dataclass(frozen=True)
class SessionScope:
    """The session a process acts for, as far as artifact ownership goes."""

    kind: ScopeKind
    transcript: str | None
    """The current transcript; ``None`` when unresolved."""
    ledger: Ledger | None
    """The ledger :func:`owner_of` reads for this scope."""
    reason: str = ""
    """Why ``transcript`` is ``None``; ``""`` when resolved."""
    ledger_ok: bool = True
    """``False`` when the hub ledger could not be read."""


def owned_by(entry: ArtifactEntry, scope: SessionScope) -> bool:
    """Whether *scope* owns *entry*."""
    return (
        scope.transcript is not None
        and operator_visible(entry)
        and owner_of(entry, scope.ledger) == scope.transcript
    )


#: The directory under the agent-data root holding one CLI pointer per launch.
CLI_POINTER_DIR = "cli-transcripts"

#: The web terminal's session key.
_SESSION_KEY_ENV = "OSPREY_SESSION_ID"
#: A dispatched run's id; non-empty in every process of the run.
_DISPATCH_RUN_ID_ENV = "OSPREY_DISPATCH_RUN_ID"
#: ``"1"`` in a headless dispatch run.
_DISPATCH_RUN_ENV = "OSPREY_DISPATCH_RUN"
#: The terminal session; ``chat-<uuid>`` for an ``osprey chat``/``query`` launch.
_TERMINAL_SESSION_ENV = "OSPREY_TERMINAL_SESSION"
#: The launch id naming this launch's CLI pointer file.
_CLI_LAUNCH_ENV = "OSPREY_CLI_LAUNCH"

_REASON_DISPATCH = "dispatch run: its artifacts belong to the run, not to a session"
_REASON_LEDGER = "the session ledger could not be read"
_REASON_NO_ROW = "this session has no transcript recorded yet"
_REASON_NO_SESSION = "this process belongs to no web terminal session or osprey chat launch"
_REASON_NO_TRANSCRIPT = "no transcript given"


def resolve_scope(
    *,
    key: str | None,
    ledger: LedgerRows | None,
    ledger_ok: bool,
    cli_conversation: str | None = None,
    dispatch: bool = False,
) -> SessionScope:
    """The scope these inputs name; the first matching rule wins.

    1. *dispatch* gives kind ``dispatch`` with no transcript.
    2. A non-empty *key* gives kind ``web`` with ``ledger.current(key)``,
       or no transcript when the ledger is unreadable or has no row for it.
    3. A non-empty *cli_conversation* gives kind ``cli`` with that transcript
       and a one-row ledger ``("cli:" + cid, cid, EPOCH)``.
    4. Otherwise kind ``none``.

    An unresolved scope carries the hub ledger it was given, its *ledger_ok*
    and a non-empty reason.
    """
    if dispatch:
        return SessionScope("dispatch", None, ledger, _REASON_DISPATCH, ledger_ok)
    if key:
        if not ledger_ok or ledger is None:
            return SessionScope("web", None, ledger, _REASON_LEDGER, ledger_ok)
        transcript = ledger.current(key)
        if transcript is None:
            return SessionScope("web", None, ledger, _REASON_NO_ROW, ledger_ok)
        return SessionScope("web", transcript, ledger, "", ledger_ok)
    if cli_conversation:
        cli_ledger = LedgerRows(
            rows=(LedgerRow("cli:" + cli_conversation, cli_conversation, EPOCH),)
        )
        return SessionScope("cli", cli_conversation, cli_ledger)
    return SessionScope("none", None, ledger, _REASON_NO_SESSION, ledger_ok)


def scope_for(transcript: str | None, ledger: Ledger | None, ledger_ok: bool) -> SessionScope:
    """The web scope of a transcript the caller already knows.

    ``None`` gives kind ``none`` with a reason.
    """
    if not transcript:
        return SessionScope("none", None, ledger, _REASON_NO_TRANSCRIPT, ledger_ok)
    return SessionScope("web", transcript, ledger, "", ledger_ok)


def read_ledger() -> tuple[LedgerRows, bool]:
    """The hub's session ledger and whether it could be read. Never raises."""
    try:
        from osprey_connectors.posture_store import state_dir

        location = state_dir()
    except Exception:
        logger.debug("No state directory for the session ledger", exc_info=True)
        return LedgerRows(), False
    return load_checked(location)


def current_scope(
    environ: Mapping[str, str] | None = None, *, root: Path | None = None
) -> SessionScope:
    """The scope of the process *environ* describes (default ``os.environ``).

    The key is ``OSPREY_SESSION_ID``; a dispatch run sets
    ``OSPREY_DISPATCH_RUN_ID`` or ``OSPREY_DISPATCH_RUN=1``. With neither, a
    launch whose ``OSPREY_TERMINAL_SESSION`` carries the host prefix is a CLI
    conversation: the transcript is the CLI pointer's content, else the harness
    id (which does not follow a ``/clear``). The hub ledger is read only for a
    key.

    :param root: The agent-data root holding the pointer; ``None`` derives it
        the way the turn-state hook does (``agent_data.base_dir`` on the repo).
    """
    env = os.environ if environ is None else environ
    key = env.get(_SESSION_KEY_ENV) or None
    if _is_dispatch(env):
        return resolve_scope(key=key, ledger=None, ledger_ok=True, dispatch=True)
    if key:
        ledger, ledger_ok = read_ledger()
        return resolve_scope(key=key, ledger=ledger, ledger_ok=ledger_ok)
    return resolve_scope(
        key=None, ledger=None, ledger_ok=True, cli_conversation=_cli_conversation(env, root)
    )


def session_tag(environ: Mapping[str, str] | None = None, *, root: Path | None = None) -> str:
    """The ``session_id`` an agent-process save defaults to.

    ``web`` gives the key, ``cli`` gives ``"cli:" + transcript`` (the pointer's
    content, so a ``/clear`` moves it), ``dispatch`` and ``none`` give ``""``.
    Reads only *environ* (default ``os.environ``) and the CLI pointer, never the
    ledger, so the store calls it under its index lock. A ``cli:`` tag is never a
    ledger key, so the web ledger backfill and the hub ledger never adopt it.

    :param root: The agent-data root holding the pointer, as for
        :func:`current_scope`; the store passes its workspace root.
    """
    env = os.environ if environ is None else environ
    if _is_dispatch(env):
        return ""
    key = env.get(_SESSION_KEY_ENV)
    if key:
        return key
    conversation = _cli_conversation(env, root)
    return "cli:" + conversation if conversation else ""


def export_cli_launch(environ: MutableMapping[str, str]) -> None:
    """Give an ``osprey chat``/``query`` launch a fresh ``OSPREY_CLI_LAUNCH``.

    Never an inherited one: a launch nested in another's Bash keeps a CLI
    transcript pointer of its own (``osprey_turn_state.py``).
    """
    environ[_CLI_LAUNCH_ENV] = str(uuid.uuid4())


def _is_dispatch(env: Mapping[str, str]) -> bool:
    """Whether *env* is a process of a dispatch run."""
    return bool(env.get(_DISPATCH_RUN_ID_ENV)) or env.get(_DISPATCH_RUN_ENV) == "1"


def _cli_conversation(env: Mapping[str, str], root: Path | None) -> str | None:
    """The CLI conversation of a host launch, or ``None`` for any other process."""
    from osprey.build.claude_code_telemetry import HOST_SESSION_PREFIX

    if not env.get(_TERMINAL_SESSION_ENV, "").startswith(HOST_SESSION_PREFIX):
        return None
    pointed = _read_pointer(env.get(_CLI_LAUNCH_ENV, ""), root)
    if pointed:
        return pointed
    from osprey.audit.call import harness_session_id

    return harness_session_id(env)


def _read_pointer(launch: str, root: Path | None) -> str | None:
    """The transcript id the launch's pointer names; missing/empty/unreadable is ``None``."""
    if not launch or launch in {".", ".."} or "/" in launch or "\\" in launch:
        return None
    try:
        if root is None:
            from osprey_connectors.workspace import resolve_shared_data_root

            root = resolve_shared_data_root()
        text = (Path(root) / CLI_POINTER_DIR / launch).read_text(encoding="utf-8")
    except Exception:
        logger.debug("No readable CLI transcript pointer for launch %s", launch, exc_info=True)
        return None
    first = text.strip().split("\n", 1)[0].strip()
    return first or None
