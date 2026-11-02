"""The registry of write-verdict surfaces: one named projection per surface.

Every src module that asks :func:`osprey_connectors.posture_store.write_verdict`,
:func:`~osprey_connectors.posture_store.execution_verdict` or
:func:`~osprey_connectors.posture_store.backstop_verdict` (or the hook twin's
``write_verdict`` or ``execution_verdict``) "may this write?" is a *surface*. A surface rarely
returns the verdict itself: the connector host answers a bool, the executor a
launch stamp, the kernel a hint line, the chip a turn-on lock. A
*projection* is the rule that maps one case-table row's expected verdict onto
what that surface returns, together with the rows the surface can be asked
(its domain): a surface that names the owner by login cannot be asked a row
that leaves the owner to the ladder, a surface that ignores the launch pin
cannot be asked a pin row that honours it.

Projections are only ever narrowed by a domain the surface's own signature
imposes, never by what the surface happens to answer: a row in the domain
whose answer disagrees is a bug in the surface, fixed in src.

This module is ADD-ONLY (D6): an existing entry is never edited, a new
surface gets a new entry. ``freeze_check.sh`` in the plan directory enforces
that against the ``wp/table-frozen`` tag.

The one edit the switch's three positions forced: the two entries that spelled
the retired switch word ``on`` — ``chip-post-lock``'s request state and
``control-context-hook``'s expected word — spell ``asked``, the narrowest
arming word, so each keeps its meaning. With the word went the rows the old
word could not tell apart: ``chip-post-lock`` asks only a request that widens
(the route's locks guard widenings and nothing else), and
``control-context-hook`` only the rows whose write runs at ``asked``. The
positions themselves are new entries: ``approval-position``,
``control-context-word``, ``chip-row-position`` and ``status-position``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey_connectors_tests._harness._write_verdict_cases import Case, expected_position

#: What a projection returns for a row outside its surface's domain.
OUT_OF_DOMAIN: Any = object()

#: The trailer every refusal phrase ends with (``refusals.refusal_text``).
_TRAILER = re.compile(r"\[osprey-refusal: ([a-z_]+)\]")


@dataclass
class Staged:
    """One row staged for one surface: the row and where its config lives."""

    case: Case
    tmp_path: Path
    monkeypatch: pytest.MonkeyPatch
    config_path: Path


@dataclass(frozen=True)
class Surface:
    """One surface that answers "may this write?", and its projection.

    Attributes:
        name: The surface's name in test ids.
        module: The dotted src module whose verdict call this surface projects.
        verdict: ``write`` (write rows) or ``backstop`` (backstop rows).
        stage1: Whether the surface decides at approval stage 1 — before a
            prompt, the tool, or the bridge — so a switch that is off must
            read ``switch_off`` there.
        reports_reason: Whether the surface's answer carries the reason, so
            the stage-1 row can be checked for ``switch_off`` by name.
        project: ``(observed, expected)`` for a staged row, or
            :data:`OUT_OF_DOMAIN`.
    """

    name: str
    module: str
    verdict: str
    stage1: bool
    reports_reason: bool
    project: Callable[[Staged], Any]


_HOOKS = "osprey.templates.claude_code.claude.hooks"
_HOOKS_DIR = Path(__file__).resolve().parents[2] / "src" / Path(*_HOOKS.split("."))


# -- domain and expectation helpers -----------------------------------------


def _pin_env(case: Case) -> bool:
    from osprey_connectors import posture_store

    return posture_store.LAUNCH_POSTURE_ENV_VAR in case.env


def _ladder_pin_honoured(case: Case) -> bool:
    """A surface that passes no owner and honours the pin."""
    return not case.has_owner and not case.ignore_launch_pin


def _ladder_pin_ignored(case: Case) -> bool:
    """A surface that passes no owner and ignores the pin."""
    return not case.has_owner and (case.ignore_launch_pin or not _pin_env(case))


def _login_pin_ignored(case: Case) -> bool:
    """A surface that names the owner by login and ignores the pin."""
    return (
        case.has_owner
        and isinstance(case.owner, str)
        and (case.ignore_launch_pin or not _pin_env(case))
    )


def _str_target(case: Case) -> bool:
    return isinstance(case.target, str) and bool(case.target)


def _expected(case: Case) -> tuple[bool, str | None]:
    return case.expect["allowed"], case.expect["reason"]


def _section(staged: Staged) -> Any:
    return staged.case.section


def _refusal_reason(error: BaseException) -> str | None:
    match = _TRAILER.search(str(error))
    return match.group(1) if match else None


def _import_hook(name: str) -> Any:
    """Import a hook in-process, stripping the ``sys.path`` entry it adds.

    The same seam as ``tests/hooks/conftest.import_hook``: a hook prepends its
    own directory so it can import its siblings, and leaving that entry would
    let a later ``import osprey_hook_log`` anywhere resolve to the template copy.
    """
    import importlib
    import os
    import sys

    hooks_dir = os.path.abspath(str(_HOOKS_DIR))
    try:
        return importlib.import_module(f"{_HOOKS}.{name}")
    finally:
        sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry) != hooks_dir]


def _reset_hook_config_memo(staged: Staged, hook_function: Callable[..., Any]) -> None:
    """Clear ``osprey_hook_log``'s config memo, in the copy *hook_function* reads."""
    loader = hook_function.__globals__["load_osprey_config"]
    staged.monkeypatch.setitem(loader.__globals__, "_osprey_config_cache", None)


# -- the package and the twin ------------------------------------------------


def _package(staged: Staged) -> Any:
    from osprey_connectors.posture_store import write_verdict
    from osprey_connectors_tests._harness._write_verdict_cases import owner_argument

    case = staged.case
    verdict = write_verdict(
        case.section,
        case.target,
        ignore_launch_pin=case.ignore_launch_pin,
        **owner_argument(case),
    )
    return (verdict.allowed, verdict.reason), _expected(case)


def _package_backstop(staged: Staged) -> Any:
    from osprey_connectors.posture_store import backstop_verdict

    verdict = backstop_verdict(staged.case.section)
    return (verdict.allowed, verdict.reason), _expected(staged.case)


def _twin(staged: Staged) -> Any:
    twin = _import_hook("osprey_target_state")
    from osprey_connectors_tests._harness._write_verdict_cases import owner_argument

    case = staged.case
    verdict = twin.write_verdict(
        case.section,
        case.target,
        ignore_launch_pin=case.ignore_launch_pin,
        **owner_argument(case, no_owner=twin.NO_OWNER),
    )
    return (verdict.allowed, verdict.reason), _expected(case)


# -- connector package surfaces ----------------------------------------------


def _monitor(staged: Staged) -> Any:
    """The reference monitor: config section, the stamped target, the ladder."""
    case = staged.case
    if not _ladder_pin_honoured(case) or not (case.target is None or _str_target(case)):
        return OUT_OF_DOMAIN
    from osprey_connectors.control_system._base import _monitor_verdict

    verdict = _monitor_verdict(case.target)
    return (verdict.allowed, verdict.reason), _expected(case)


def _ipc_pool(staged: Staged) -> Any:
    """The connector-host pool's gateway role: allowed only, section passed in."""
    case = staged.case
    if not _ladder_pin_honoured(case) or not _str_target(case):
        return OUT_OF_DOMAIN
    if not isinstance(case.section, dict):
        return OUT_OF_DOMAIN  # the pool's signature takes a mapping
    from osprey_connectors.ipc._pool import ConnectorHostPool

    return ConnectorHostPool._writes_allowed(case.section, case.target), case.expect["allowed"]


# -- framework surfaces ------------------------------------------------------


def _audit_posture(staged: Staged) -> Any:
    """``audit.posture.verdict``: config section, the recorded target."""
    case = staged.case
    if not _ladder_pin_honoured(case):
        return OUT_OF_DOMAIN
    from osprey.audit import posture

    if posture.recorded_control_target() != case.target:
        return OUT_OF_DOMAIN
    verdict = posture.verdict()
    return (verdict.allowed, verdict.reason), _expected(case)


def _middleware_backstop(staged: Staged) -> Any:
    from osprey.mcp_server.audit_middleware import _backstop

    verdict = _backstop()
    return (verdict.allowed, verdict.reason), _expected(staged.case)


def _qserver_lane(staged: Staged) -> Any:
    """The qserver worker's startup half: the backstop, allowed only."""
    from osprey.services.bluesky_bridge.queue_backend import lane_tier_allows

    return lane_tier_allows(staged.case.section), staged.case.expect["allowed"]


def _bluesky_lane(staged: Staged) -> Any:
    """The queue's arming gate for one lane's target."""
    case = staged.case
    if not _ladder_pin_honoured(case) or not _str_target(case):
        return OUT_OF_DOMAIN
    from osprey.mcp_server.bluesky.tools.queue import _lane_verdict

    verdict = _lane_verdict(case.target)
    return (verdict.allowed, verdict.reason), _expected(case)


def _connector_host_manager(staged: Staged) -> Any:
    case = staged.case
    if not _ladder_pin_honoured(case) or not _str_target(case):
        return OUT_OF_DOMAIN
    from osprey.mcp_server.control_system.connector_host_manager import _writes_allowed

    return _writes_allowed(case.section, case.target), case.expect["allowed"]


def _target_eligibility(staged: Staged) -> Any:
    case = staged.case
    if not _ladder_pin_honoured(case) or not _str_target(case):
        return OUT_OF_DOMAIN
    from osprey.mcp_server.control_system.target_eligibility import target_write_verdict

    verdict = target_write_verdict(case.section, case.target)
    return (verdict.allowed, verdict.reason), _expected(case)


def _channel_write_entry(staged: Staged) -> Any:
    """channel_write's entry check, for a session with no record (the baseline)."""
    case = staged.case
    if not _ladder_pin_honoured(case):
        return OUT_OF_DOMAIN
    from osprey.mcp_server.control_system.tools import channel_write

    config = {"control_system": case.section}
    if channel_write._write_target(None, case.section) != case.target:
        return OUT_OF_DOMAIN
    try:
        channel_write._check_write_verdict(None, config)
    except Exception as error:  # the refusal envelope
        return (False, _refusal_reason(error)), _expected(case)
    return (True, None), _expected(case)


def _app_tool_call(staged: Staged) -> Any:
    """An application tool server's approved-call check, for an offered tool that asks nobody."""
    case = staged.case
    if not _ladder_pin_honoured(case):
        return OUT_OF_DOMAIN
    from osprey.apps.sdk import ApprovalRefused, _approval, check_approved_call
    from osprey.audit.posture import recorded_control_target

    if recorded_control_target() != case.target:
        return OUT_OF_DOMAIN
    config = {
        "control_system": case.section,
        "approval": {"enabled": True, "tools": {"mcp__surface__act": "skip"}},
        "apps": {"surface": {"approval": {"act": {"app": "surface"}}}},
    }
    staged.monkeypatch.setenv(_approval.APP_NAME_ENV_VAR, "surface")
    staged.monkeypatch.setattr(_approval, "load_osprey_config", lambda: config)
    try:
        check_approved_call("act", {})
    except ApprovalRefused as error:
        return (False, _refusal_reason(error)), _expected(case)
    return (True, None), _expected(case)


def _phoebus_drive(staged: Staged) -> Any:
    """phoebus_drive server-side: config section, the deployment baseline."""
    case = staged.case
    if not _ladder_pin_honoured(case) or not isinstance(case.section, dict):
        return OUT_OF_DOMAIN
    from osprey.mcp_server.phoebus.tools.bridge_tools import _check_drive_switch
    from osprey_connectors.types import baseline_target

    try:
        if baseline_target(case.section) != case.target:
            return OUT_OF_DOMAIN
    except Exception:
        return OUT_OF_DOMAIN
    try:
        _check_drive_switch()
    except Exception as error:
        return (False, _refusal_reason(error)), _expected(case)
    return (True, None), _expected(case)


def _executor_pin(staged: Staged) -> Any:
    """The executor's launch pin: the verdict at launch, pin ignored, as a stamp."""
    case = staged.case
    if not _ladder_pin_ignored(case) or not _str_target(case):
        return OUT_OF_DOMAIN
    from osprey.mcp_server.python_executor.executor import _launch_posture
    from osprey_connectors import posture_store

    expected = posture_store.launch_posture_stamp(
        case.target,
        posture_store.POSTURE_WRITES if case.expect["allowed"] else posture_store.POSTURE_SANDBOX,
    )
    return _launch_posture(case.target), expected


def _executor_gate(staged: Staged) -> Any:
    """The executor's launch gate: config section, the target the run would write to."""
    case = staged.case
    if not _ladder_pin_honoured(case):
        return OUT_OF_DOMAIN
    from osprey.audit import posture
    from osprey.mcp_server.python_executor.tools._execution_gates import (
        _launch_target,
        _launch_verdict,
    )

    if _launch_target(posture.control_system_section()) != case.target:
        return OUT_OF_DOMAIN
    verdict = _launch_verdict()
    return (verdict.allowed, verdict.reason), _expected(case)


def _runtime_pool(staged: Staged) -> Any:
    case = staged.case
    if not _ladder_pin_honoured(case) or not _str_target(case):
        return OUT_OF_DOMAIN
    if not isinstance(case.section, dict):
        return OUT_OF_DOMAIN  # the runtime reads a non-mapping section as empty
    from osprey.runtime import _pooled_writes_allowed

    return _pooled_writes_allowed(case.target), case.expect["allowed"]


def _kernel_hint(staged: Staged) -> Any:
    """The kernel's hint line for a blocked write, on the stamped target."""
    case = staged.case
    if not _ladder_pin_honoured(case) or not _str_target(case):
        return OUT_OF_DOMAIN
    from osprey import jupyter_kernel
    from osprey.mcp_server.python_executor import executor
    from osprey_connectors.errors import ChannelWriteBlockedError

    staged.monkeypatch.setenv(executor.ENV_CONTROL_TARGET, case.target)
    hint = jupyter_kernel._hint_for(ChannelWriteBlockedError("SR:TEST", "WRITES_DISABLED"))
    reason = case.expect["reason"]
    expected = (
        jupyter_kernel.HINT_WRITES_OFF if reason in jupyter_kernel._WRITES_OFF_REASONS else None
    )
    return hint, expected


def _chip_row(staged: Staged) -> Any:
    """The chip's GET row for a login (C-GET): pin ignored."""
    case = staged.case
    if not _login_pin_ignored(case) or not _str_target(case):
        return OUT_OF_DOMAIN
    from osprey.interfaces.web_terminal.routes.websocket import _row_verdict

    verdict = _row_verdict(case.section, case.target, case.owner)
    return (verdict.allowed, verdict.reason), _expected(case)


def _chip_post_lock(staged: Staged) -> Any:
    """The chip's POST turn-on lock (C-POST): only the turn-on locks refuse."""
    case = staged.case
    if not _login_pin_ignored(case) or not _str_target(case):
        return OUT_OF_DOMAIN
    from osprey.interfaces.web_terminal.routes import websocket

    facts = websocket._posture_post_facts(staged.config_path, case.target, "asked", case.owner)
    if case.target not in facts.configured or not websocket._widens("asked", facts.before):
        return OUT_OF_DOMAIN
    observed = facts.lock.reason if facts.lock is not None else None
    reason = case.expect["reason"]
    return observed, reason if reason in websocket._TURN_ON_LOCKS else None


# -- hook surfaces -----------------------------------------------------------


def _place(staged: Staged, hook_module: Any) -> None:
    """Place the hooked write on the row's target: placement is the twin's own test."""
    target = staged.case.target
    staged.monkeypatch.setattr(
        hook_module._target_state, "hook_write_target", lambda *_args, **_kw: target
    )


def _writes_check(staged: Staged) -> Any:
    """The writes-check hook's stage 2, for a framework write placed on the row's target."""
    case = staged.case
    if not _ladder_pin_honoured(case) or not _str_target(case):
        return OUT_OF_DOMAIN
    hook = _import_hook("osprey_writes_check")

    _reset_hook_config_memo(staged, hook._stage_two)
    _place(staged, hook)
    answer = hook._stage_two({"tool_input": {}}, "mcp__controls__channel_write", "channel_write")
    return (answer[0], answer[1]), _expected(case)


def _approval_defer(staged: Staged) -> Any:
    """The approval hook's defer: the twin's ``allowed`` for a placed write."""
    case = staged.case
    if not _ladder_pin_honoured(case) or not _str_target(case):
        return OUT_OF_DOMAIN
    hook = _import_hook("osprey_approval")

    _place(staged, hook)
    staged.monkeypatch.setitem(hook._call_write_posture.__globals__, "is_write_tool", _yes)
    staged.monkeypatch.setitem(hook._call_write_posture.__globals__, "is_write_call", _yes)
    answer = hook._call_write_posture(
        {"control_system": case.section},
        "mcp__controls__channel_write",
        "channel_write",
        {},
        {"tool_input": {}},
    )
    return answer, case.expect["allowed"]


def _yes(*_args: Any, **_kwargs: Any) -> bool:
    return True


def _control_context_hook(staged: Staged) -> Any:
    """The control-context hook's ``writes`` word for the acting login."""
    case = staged.case
    if not case.has_owner or not isinstance(case.owner, str) or case.ignore_launch_pin:
        return OUT_OF_DOMAIN  # the hook asks for the acting login, the pin honoured
    if not _str_target(case) or case.record is None or case.record.get("status") != "ok":
        return OUT_OF_DOMAIN  # the hook reads the host record it was rendered beside
    if case.tree is not None:
        return OUT_OF_DOMAIN
    hook = _import_hook("osprey_control_context")

    if case.owner != hook._target_state.acting_identity():
        return OUT_OF_DOMAIN
    _reset_hook_config_memo(staged, hook.current_state)
    state = hook.current_state({})
    if state is None or case.target not in state["writes"]:
        return OUT_OF_DOMAIN
    allowed, reason = _expected(case)
    if allowed and expected_position(case) != "asked":
        return OUT_OF_DOMAIN  # the word at another position is control-context-word's
    if allowed:
        expected = "asked"
    elif reason == "switch_off":
        expected = hook.WRITES_OFF
    else:
        expected = f"{hook.WRITES_REFUSED}:{reason}"
    return state["writes"][case.target], expected


# -- a held session ----------------------------------------------------------
#
# A surface that executes a write answers for the session's hold too. Each row
# is asked again with a flag open: the backstop's two reasons still come first,
# and every later term gives way to ``session_held``.

_HELD_SESSION = "11111111-2222-3333-4444-555555555555"


def _hold(staged: Staged) -> None:
    """Open one flag on the session this process belongs to."""
    from osprey_connectors import session_hold

    root = staged.tmp_path / "session_holds"
    staged.monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(root))
    staged.monkeypatch.setenv("OSPREY_SESSION_ID", _HELD_SESSION)
    session_hold.ensure_session(_HELD_SESSION)
    session_hold.open_flag(
        _HELD_SESSION,
        flag_id="f1",
        source="watchdog:scope",
        reason="off the plan",
        prompt_ref="record:f1",
        prompt_excerpt="move the corrector",
    )


def _held(project: Callable[[Staged], Any]) -> Callable[[Staged], Any]:
    """*project*, asked with the session held: the row's answer gives way to the hold."""

    def held(staged: Staged) -> Any:
        from osprey_connectors.posture_store import backstop_verdict

        _hold(staged)
        answer = project(staged)
        if answer is OUT_OF_DOMAIN:
            return answer
        backstop = backstop_verdict(staged.case.section)
        expected = (False, backstop.reason if not backstop.allowed else "session_held")
        return answer[0], expected

    return held


def _twin_execution(staged: Staged) -> Any:
    twin = _import_hook("osprey_target_state")
    from osprey_connectors_tests._harness._write_verdict_cases import owner_argument

    case = staged.case
    verdict = twin.execution_verdict(
        case.section,
        case.target,
        ignore_launch_pin=case.ignore_launch_pin,
        **owner_argument(case, no_owner=twin.NO_OWNER),
    )
    return (verdict.allowed, verdict.reason), _expected(case)


def _middleware_clamp(staged: Staged) -> Any:
    """The clamp's answer for a clamped write tool: the backstop, then the hold."""
    from osprey.mcp_server.audit_middleware import _backstop, _clamp_refusal

    refusal = _clamp_refusal(_backstop(), "channel_write")
    observed = (True, None) if refusal is None else (False, refusal.reason)
    return observed, _expected(staged.case)


# -- position surfaces (added with the switch's three positions) -------------


def _approval_position(staged: Staged) -> Any:
    """The approval hook's position for a placed governed write (``_governed_position``)."""
    case = staged.case
    if not _ladder_pin_honoured(case) or not _str_target(case):
        return OUT_OF_DOMAIN
    hook = _import_hook("osprey_approval")

    _place(staged, hook)
    staged.monkeypatch.setitem(hook._governed_position.__globals__, "is_write_call", _yes)
    answer = hook._governed_position(
        {"control_system": case.section},
        "mcp__controls__channel_write",
        "channel_write",
        {},
        {"tool_input": {}},
    )
    return answer, expected_position(case) if case.expect["allowed"] else None


def _control_context_word(staged: Staged) -> Any:
    """The control-context hook's ``writes`` word at every position: the word is the position."""
    case = staged.case
    if not case.has_owner or not isinstance(case.owner, str) or case.ignore_launch_pin:
        return OUT_OF_DOMAIN  # the hook asks for the acting login, the pin honoured
    if not _str_target(case) or case.record is None or case.record.get("status") != "ok":
        return OUT_OF_DOMAIN  # the hook reads the host record it was rendered beside
    if case.tree is not None:
        return OUT_OF_DOMAIN
    hook = _import_hook("osprey_control_context")

    if case.owner != hook._target_state.acting_identity():
        return OUT_OF_DOMAIN
    _reset_hook_config_memo(staged, hook.current_state)
    state = hook.current_state({})
    if state is None or case.target not in state["writes"]:
        return OUT_OF_DOMAIN
    allowed, reason = _expected(case)
    if allowed:
        expected = expected_position(case)
    elif reason == "switch_off":
        expected = hook.WRITES_OFF
    else:
        expected = f"{hook.WRITES_REFUSED}:{reason}"
    return state["writes"][case.target], expected


#: The stored switch words a reader keeps; the retired ``on`` reads ``asked`` (D1).
_STORED_POSITIONS = {"asked": "asked", "allowed": "allowed", "on": "asked"}


def _stored_position(case: Case) -> str | None:
    """The position the host record stores for the row's target, ``None`` when unreadable."""
    record = case.record
    if record is None:
        return "off"
    if record.get("status") != "ok":
        return None
    return _STORED_POSITIONS.get((record.get("switch") or {}).get(case.target), "off")


def _chip_row_position(staged: Staged) -> Any:
    """The chip's GET row (C-GET): the verdict, ``prompt`` and the stored ``switch``."""
    case = staged.case
    if not _login_pin_ignored(case) or not _str_target(case) or case.tree is not None:
        return OUT_OF_DOMAIN
    from types import SimpleNamespace

    from osprey.interfaces.web_terminal.routes import websocket

    if case.owner != websocket.acting_identity():
        return OUT_OF_DOMAIN  # the row is the signed-in login's
    view = websocket._posture_view(SimpleNamespace(state=SimpleNamespace()), staged.config_path)
    rows = [row for row in view["targets"] if row["target"] == case.target]
    if not rows:
        return OUT_OF_DOMAIN
    (row,) = rows
    allowed, reason = _expected(case)
    observed = (row["write_allowed"], row["write_reason"], row["prompt"], row["switch"])
    expected = (allowed, reason, expected_position(case) == "asked", _stored_position(case))
    return observed, expected


#: The verdict reasons ``osprey status`` can tell: it reads the switch and the tier only.
_STATUS_REASONS = frozenset({None, "switch_off", "tier_never"})


def _status_position(staged: Staged) -> Any:
    """``osprey status``'s Writes cell for the host login, read back as a verdict."""
    case = staged.case
    if case.has_owner or case.env or case.tree is not None or not _str_target(case):
        return OUT_OF_DOMAIN  # status reads the host's own records, at the deployment's tier
    if not isinstance(case.section, dict) or "login_tiers" in case.section:
        return OUT_OF_DOMAIN
    if case.record is None or case.record.get("status") != "ok":
        return OUT_OF_DOMAIN
    if case.expect["reason"] not in _STATUS_REASONS:
        return OUT_OF_DOMAIN
    from osprey.deployment import compose_generator, status_display
    from osprey_connectors.control_context import _record_path_under
    from osprey_connectors_tests._harness._write_verdict_cases import host_root

    record_path = _record_path_under(host_root(staged.tmp_path))
    tree, login = record_path.parent.parent, record_path.parent.name
    staged.monkeypatch.setattr(
        compose_generator, "control_target_tree_dir", lambda *_args, **_kw: tree
    )
    staged.monkeypatch.setattr(status_display, "_staged_login_tiers", lambda *_args: None)
    rows = status_display.writes_rows({"control_system": case.section}, staged.tmp_path)
    (row,) = [row for row in rows if row[0] == login]
    _login, tier, _target, cell = row
    if tier == "never":
        observed: tuple[Any, ...] = (False, "tier_never", None)
    else:
        entries = [entry for entry in cell.split(", ") if entry.startswith(f"{case.target}:")]
        if not entries:
            observed = (False, "switch_off", None)
        else:
            (entry,) = entries
            position = entry[len(case.target) + 1 :].split(" ")[0]
            if status_display.CAPPED_AT_ASKED in entry:
                position = "asked"
            observed = (True, None, position)
    allowed, reason = _expected(case)
    return observed, (allowed, reason, expected_position(case))


# -- the registry -------------------------------------------------------------

SURFACES: tuple[Surface, ...] = (
    Surface(
        "package",
        "osprey_connectors.posture_store",
        "write",
        stage1=True,
        reports_reason=True,
        project=_package,
    ),
    Surface(
        "package-backstop",
        "osprey_connectors.posture_store",
        "backstop",
        stage1=False,
        reports_reason=True,
        project=_package_backstop,
    ),
    Surface(
        "twin",
        f"{_HOOKS}.osprey_target_state",
        "write",
        stage1=True,
        reports_reason=True,
        project=_twin,
    ),
    Surface(
        "monitor",
        "osprey_connectors.control_system._base",
        "write",
        stage1=True,
        reports_reason=True,
        project=_monitor,
    ),
    Surface(
        "ipc-pool-role",
        "osprey_connectors.ipc._pool",
        "write",
        stage1=False,
        reports_reason=False,
        project=_ipc_pool,
    ),
    Surface(
        "audit-posture",
        "osprey.audit.posture",
        "write",
        stage1=False,
        reports_reason=True,
        project=_audit_posture,
    ),
    Surface(
        "middleware-backstop",
        "osprey.mcp_server.audit_middleware",
        "backstop",
        stage1=False,
        reports_reason=True,
        project=_middleware_backstop,
    ),
    Surface(
        "qserver-lane-tier",
        "osprey.services.bluesky_bridge.queue_backend",
        "backstop",
        stage1=False,
        reports_reason=False,
        project=_qserver_lane,
    ),
    Surface(
        "bluesky-lane",
        "osprey.mcp_server.bluesky.tools.queue",
        "write",
        stage1=True,
        reports_reason=True,
        project=_bluesky_lane,
    ),
    Surface(
        "connector-host-manager",
        "osprey.mcp_server.control_system.connector_host_manager",
        "write",
        stage1=False,
        reports_reason=False,
        project=_connector_host_manager,
    ),
    Surface(
        "target-eligibility",
        "osprey.mcp_server.control_system.target_eligibility",
        "write",
        stage1=False,
        reports_reason=True,
        project=_target_eligibility,
    ),
    Surface(
        "channel-write-entry",
        "osprey.mcp_server.control_system.tools.channel_write",
        "write",
        stage1=True,
        reports_reason=True,
        project=_channel_write_entry,
    ),
    Surface(
        "phoebus-drive",
        "osprey.mcp_server.phoebus.tools.bridge_tools",
        "write",
        stage1=True,
        reports_reason=True,
        project=_phoebus_drive,
    ),
    Surface(
        "executor-pin",
        "osprey.mcp_server.python_executor.executor",
        "write",
        stage1=False,
        reports_reason=False,
        project=_executor_pin,
    ),
    Surface(
        "executor-gate",
        "osprey.mcp_server.python_executor.tools._execution_gates",
        "write",
        stage1=True,
        reports_reason=True,
        project=_executor_gate,
    ),
    Surface(
        "runtime-pool",
        "osprey.runtime",
        "write",
        stage1=False,
        reports_reason=False,
        project=_runtime_pool,
    ),
    Surface(
        "kernel-hint",
        "osprey.jupyter_kernel",
        "write",
        stage1=False,
        reports_reason=False,
        project=_kernel_hint,
    ),
    Surface(
        "chip-get-row",
        "osprey.interfaces.web_terminal.routes.websocket",
        "write",
        stage1=False,
        reports_reason=True,
        project=_chip_row,
    ),
    Surface(
        "chip-post-lock",
        "osprey.interfaces.web_terminal.routes.websocket",
        "write",
        stage1=False,
        reports_reason=False,
        project=_chip_post_lock,
    ),
    Surface(
        "writes-check-hook",
        f"{_HOOKS}.osprey_writes_check",
        "write",
        stage1=True,
        reports_reason=True,
        project=_writes_check,
    ),
    Surface(
        "approval-defer",
        f"{_HOOKS}.osprey_approval",
        "write",
        stage1=True,
        reports_reason=False,
        project=_approval_defer,
    ),
    Surface(
        "control-context-hook",
        f"{_HOOKS}.osprey_control_context",
        "write",
        stage1=False,
        reports_reason=True,
        project=_control_context_hook,
    ),
    Surface(
        "app-tool-call",
        "osprey.apps.sdk._approval",
        "write",
        stage1=True,
        reports_reason=True,
        project=_app_tool_call,
    ),
    Surface(
        "twin-held",
        f"{_HOOKS}.osprey_target_state",
        "write",
        stage1=False,
        reports_reason=True,
        project=_held(_twin_execution),
    ),
    Surface(
        "bluesky-lane-held",
        "osprey.mcp_server.bluesky.tools.queue",
        "write",
        stage1=False,
        reports_reason=True,
        project=_held(_bluesky_lane),
    ),
    Surface(
        "channel-write-entry-held",
        "osprey.mcp_server.control_system.tools.channel_write",
        "write",
        stage1=False,
        reports_reason=True,
        project=_held(_channel_write_entry),
    ),
    Surface(
        "phoebus-drive-held",
        "osprey.mcp_server.phoebus.tools.bridge_tools",
        "write",
        stage1=False,
        reports_reason=True,
        project=_held(_phoebus_drive),
    ),
    Surface(
        "executor-gate-held",
        "osprey.mcp_server.python_executor.tools._execution_gates",
        "write",
        stage1=False,
        reports_reason=True,
        project=_held(_executor_gate),
    ),
    Surface(
        "writes-check-hook-held",
        f"{_HOOKS}.osprey_writes_check",
        "write",
        stage1=False,
        reports_reason=True,
        project=_held(_writes_check),
    ),
    Surface(
        "middleware-clamp-held",
        "osprey.mcp_server.audit_middleware",
        "backstop",
        stage1=False,
        reports_reason=True,
        project=_held(_middleware_clamp),
    ),
    Surface(
        "approval-position",
        f"{_HOOKS}.osprey_approval",
        "write",
        stage1=False,
        reports_reason=False,
        project=_approval_position,
    ),
    Surface(
        "control-context-word",
        f"{_HOOKS}.osprey_control_context",
        "write",
        stage1=False,
        reports_reason=True,
        project=_control_context_word,
    ),
    Surface(
        "chip-row-position",
        "osprey.interfaces.web_terminal.routes.websocket",
        "write",
        stage1=False,
        reports_reason=True,
        project=_chip_row_position,
    ),
    Surface(
        "status-position",
        "osprey.deployment.status_display",
        "write",
        stage1=False,
        reports_reason=True,
        project=_status_position,
    ),
)

#: Surfaces by name.
BY_NAME: Mapping[str, Surface] = {surface.name: surface for surface in SURFACES}


def write_config(path: Path, section: Any) -> Path:
    """Render a ``config.yml`` whose ``control_system`` block is *section*."""
    path.write_text(yaml.safe_dump({"control_system": section}), encoding="utf-8")
    return path
