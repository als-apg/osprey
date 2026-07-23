"""The Bluesky bridge's write-safety validation gates — the whole set, isolated.

This module holds every gate that stands between a launch request and real
hardware motion, deliberately gathered in one small, reviewable place. Nothing
here builds a runner, touches a connector, or mutates process state; each
function only *reads* config/disk and either returns cleanly or raises. Keep it
that way — a reviewer must be able to audit the bridge's entire write-safety
posture by reading this one file.

Two independent gates, each defense-in-depth against a different failure:

- :func:`_assert_limits_readable_if_writable` — a STARTUP guard. Refuses to
  bring the bridge up in the one unsafe posture: writes enabled + limits
  checking enabled + the limits database unreadable. Fail-OPEN by design (see
  its docstring): every other combination starts normally.
- :func:`_validate_launchable_request` / :func:`_launch_validation_gate` — a
  per-LAUNCH gate. Refuses to launch a session/unreviewed plan whose CURRENT
  on-disk content has no passing validation record, re-reading and re-hashing
  the file at launch time rather than trusting any earlier snapshot.

Neither gate is a containment boundary on its own — the plan validator has a
documented, accepted obfuscation residual (see ``plan_validation.py``), and the
real backstop for a malicious plan body is human approval at launch (the MCP
``launch_run`` PreToolUse prompt). These gates keep the *honest* mistakes and
stale-record races out; do not weaken either, and do not let the launch gate's
freshness (re-read + re-hash every time) regress into a cached lookup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from .plan_validation import hash_plan_body
from .session_dir import resolve_session_plan_dir
from .validation_record import validation_records

if TYPE_CHECKING:
    from .runs import Run


def _request_field(request: Any, field: str, default: Any = None) -> Any:
    """Read ``field`` off a stored launch intent, which may be a plain dict
    (re-hydrated JSON) or a ``RunRequest``-shaped object — the lifecycle core
    (``runs.py``) treats ``request`` as opaque, so both shapes occur."""
    if isinstance(request, dict):
        return request.get(field, default)
    return getattr(request, field, default)


def _assert_limits_readable_if_writable() -> None:
    """Refuse startup if writes are enabled but the limits database can't be read.

    Fail-OPEN by design (task 3.1): this is the ONLY combination that refuses
    startup — ``control_system.writes_enabled`` AND
    ``control_system.limits_checking.enabled`` both true, AND the limits
    database is missing, unreadable, or unparseable. Every other combination
    starts normally: writes disabled (read-only posture) never even probes
    the database; writes enabled with limits checking disabled needs no
    database at all; writes enabled with a readable database is the healthy
    case. A writable deploy with no working limits enforcement is the one
    unsafe posture this guard exists to catch before any connector/CA work
    begins.

    Mirrors `LimitsValidator.from_config`'s ``database_path`` resolution
    (a relative path resolved against the ``CONFIG_FILE`` env var's directory
    when set, falling back to ``project_root`` otherwise — container-correct,
    since the deploy flattens ``project_root`` in as the HOST build path,
    while ``CONFIG_FILE`` points at the config actually mounted in-container),
    but probes readability via `LimitsValidator._load_limits_database`
    directly rather than calling `from_config` — `from_config` swallows every
    load failure to `None`, which would hide the exact failure this guard
    must detect and raise on.

    No project config context at all (e.g. running outside a configured
    OSPREY project — most unit-test environments) is treated the same way
    `LimitsValidator.from_config` treats it: nothing to probe, so this
    returns without blocking startup, rather than raising on the config
    lookup itself.

    Raises:
        RuntimeError: naming which condition failed (config keys, and
            whether the database path was configured/found/parseable) —
            never the database's file contents or any other secret value.
    """
    from osprey.utils.config import get_config_value

    try:
        writes_enabled = get_config_value("control_system.writes_enabled", False)
        limits_enabled = get_config_value("control_system.limits_checking.enabled", False)
        db_path = get_config_value("control_system.limits_checking.database_path", None)
        project_root = get_config_value("project_root", None)
    except (FileNotFoundError, KeyError, RuntimeError):
        return

    if not writes_enabled:
        return
    if not limits_enabled:
        return

    if not db_path or not isinstance(db_path, str):
        raise RuntimeError(
            "refusing to start writable: control_system.writes_enabled and "
            "control_system.limits_checking.enabled are both set, but "
            "control_system.limits_checking.database_path is not configured"
        )

    from osprey.connectors.control_system.limits_validator import LimitsValidator

    # Same relative-path resolution as `LimitsValidator.from_config`.
    db_path = LimitsValidator.resolve_database_path(db_path, project_root)

    try:
        LimitsValidator._load_limits_database(db_path)
    except Exception as exc:
        raise RuntimeError(
            "refusing to start writable: control_system.writes_enabled and "
            "control_system.limits_checking.enabled are both set, but the "
            "configured control_system.limits_checking.database_path could "
            "not be read or parsed"
        ) from exc


def _launch_validation_gate(run: Run) -> None:
    """Refuse to launch a session/unreviewed plan with no CURRENT passing validation record.

    Thin `Run`-shaped adapter over :func:`_validate_launchable_request` — the
    signature `runs.do_launch` expects for its dependency-injected
    ``validator``. `POST /draft/run` calls the request-level helper directly,
    *before* minting a run record, so a gate rejection there leaves nothing
    behind in the registry at all.
    """
    _validate_launchable_request(run.request)


def _validate_launchable_request(request: Any) -> None:
    """Refuse to launch a session/unreviewed plan with no CURRENT passing validation record.

    Defense-in-depth alongside task 2.4's session-layer LOAD gate
    (`plan_loader.py`'s `_load_plan_file`): that gate already keeps an
    unvalidated session/unreviewed file out of `get_facility_plans().plans`
    entirely, so in the common case this validator finds nothing to reject.
    It exists for the narrow race the load gate can't close on its own — the
    `PlanSpec` `get_facility_plans()` returned to resolve this run's
    `plan_name` moments earlier could be stale by the time launch runs (e.g.
    the session file was edited in between) — so this independently re-reads
    the file straight from `resolve_session_plan_dir()` and re-hashes its
    CURRENT content with `hash_plan_body`, the same normalization the record
    was keyed on, rather than trusting the earlier snapshot.

    Raises `HTTPException(409, ...)` for any plan name backed by a file in
    `resolve_session_plan_dir()` whose current content has no passing
    record — whether or not `get_facility_plans()` currently registers it.
    A name the load gate is quarantining *right now* for lacking a record
    resolves to no `PlanSpec` at all, but its file still exists under the
    session directory; treating that as `session` provenance too (rather than
    "not found") is what turns an already-quarantined plan's launch attempt
    into this clear 409 instead of a confusing "unknown plan" failure further
    downstream. A non-session provenance (`shipped`/`preset`/`facility`), or a
    name with neither a `PlanSpec` nor a session-dir file at all, is left
    alone — `PlanRunner.reinitialize`'s own "unknown plan" handling is the right
    place for the latter.
    """
    plan_name = _request_field(request, "plan_name")
    if not plan_name:
        return

    from .plan_loader import get_facility_plans

    spec = get_facility_plans().plans.get(plan_name)
    plan_path = resolve_session_plan_dir() / f"{plan_name}.py"
    if spec is not None:
        is_session = spec.provenance in ("session", "unreviewed")
    else:
        is_session = plan_path.is_file()

    if not is_session or not plan_path.is_file():
        # Not a session-tier plan at all, or its file has since vanished —
        # either way there is nothing here to re-hash; `PlanRunner.reinitialize`
        # will hit its own "unknown plan" path if the name doesn't resolve.
        return

    content = plan_path.read_text(encoding="utf-8")
    if not validation_records.has_passing_record(hash_plan_body(content)):
        raise HTTPException(
            status_code=409,
            detail=(
                f"session plan {plan_name!r} has no passing validation record; "
                "validate it before launching"
            ),
        )
