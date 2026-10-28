"""The Bluesky bridge's write-safety validation gates — the whole set, isolated.

This module holds every gate that stands between a launch request and real
hardware motion, deliberately gathered in one small, reviewable place. Nothing
here builds a runner, touches a connector, or mutates process state; each
function only *reads* config/disk and either returns cleanly or raises. Keep it
that way — a reviewer must be able to audit the bridge's entire write-safety
posture by reading this one file.

Two independent gates, each defense-in-depth against a different failure:

- :func:`_assert_limits_readable_if_writable` — a STARTUP guard. Refuses to
  bring the bridge up in the one unsafe posture: a write tier that allows
  writes on a lane that is not degraded + a limits database required for the
  target THIS LANE serves (:func:`osprey_connectors.types.limits_database_required`)
  + the limits database unreadable. Fail-OPEN by design (see its docstring):
  every other combination starts normally.
- :func:`_validate_launchable_request` — a per-ENQUEUE gate. Refuses to enqueue
  a session/unreviewed plan whose CURRENT on-disk content has no passing
  validation record, re-reading and re-hashing the file at enqueue time rather
  than trusting any earlier snapshot.

Neither gate is a containment boundary on its own — the plan validator has a
documented, accepted obfuscation residual (see ``plan_validation.py``), and the
real backstop for a malicious plan body is human approval at launch (the MCP
PreToolUse prompt) plus the launch token that arms the queue. These gates keep
the *honest* mistakes and stale-record races out; do not weaken either, and do
not let the enqueue gate's freshness (re-read + re-hash every time) regress
into a cached lookup.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from .plan_validation import hash_plan_body
from .session_dir import resolve_session_plan_dir
from .validation_record import validation_records


def _request_field(request: Any, field: str, default: Any = None) -> Any:
    """Read ``field`` off a launch request, which may be a plain dict
    (re-hydrated JSON) or an object carrying the field as an attribute (the
    enqueue path passes a ``draft.LaunchSnapshot``) — both shapes occur, and
    this gate treats the request as opaque either way."""
    if isinstance(request, dict):
        return request.get(field, default)
    return getattr(request, field, default)


def _assert_limits_readable_if_writable() -> None:
    """Refuse startup if this lane could write but the limits database can't be read.

    "Could write" is the startup half of the write verdict, asked through
    :func:`~osprey.services.bluesky_bridge.queue_backend.lane_tier_allows`:
    :func:`osprey_connectors.posture_store.backstop_verdict` allows (the run
    is not read-only and the login tier allows writes), and the lane is not
    degraded. The tier is target-blind — a deployment that
    allows writes allows them on every machine, and the person's switch picks
    which one, at run time — so a lane whose tier allows writes may be switched
    on at any moment after it starts, and it needs working limits enforcement
    from the start. A degraded lane addresses a machine this deployment cannot
    resolve, the connector refuses each of its writes, and it needs no limits
    database.

    The switch and the owner are deliberately not read here. Startup is nobody's
    work: no item has been dequeued and nothing has bound an owner, and a
    switch is a record that moves while the process runs. The gate that stands
    between a plan and hardware is the per-write verdict the connector asks
    once the plan wrapper has bound the owner the item was enqueued under.

    Limits checking, unlike the tier, is resolved per lane: a deployment may
    leave its virtual accelerator unchecked while its live machine enforces
    limits, so the condition is
    ``control_system.connector.<type>.limits_checking.enabled`` for this
    lane's type, inheriting the deployment-wide block where no per-type block
    says anything. Only ``database_path`` stays deployment-wide — the
    deployment mounts one limits database, and a per-type block that omits the
    path is complete rather than incomplete.

    Whether this lane's limits posture needs the database at all is
    :func:`osprey_connectors.types.limits_database_required`, the predicate
    the connector's validator, the compose mount and the build
    also ask: limits checking on, and a mode other than ``optional``
    or a path configured. ``optional`` with no path configured needs none —
    every channel is unlisted and written with no limits — while a configured
    path must load whatever the mode.

    Fail-OPEN by design: this is the ONLY combination that refuses
    startup — this lane could write AND this lane's limits posture requires a
    database, AND the limits database is missing, unreadable, or unparseable.
    Every other combination starts normally: a lane that cannot write never
    even probes the database; writes enabled with limits checking disabled
    needs no database at all, nor does ``optional`` with no path; writes
    enabled with a readable database is the healthy case. A writable deploy
    with no working limits enforcement is the one
    unsafe posture this guard exists to catch before any connector/CA work
    begins.

    Resolves and loads the database through
    `LimitsValidator.load_configured_database`, the same call
    `LimitsValidator.from_config` makes (a relative path anchors on the
    directory of the config actually loaded, falling back to the
    ``CONFIG_FILE`` env var's directory and then ``project_root`` —
    container-correct, since the deploy flattens ``project_root`` in as the
    HOST build path while the loaded config is the one mounted in-container),
    rather than calling `from_config` itself — `from_config` swallows every
    load failure into a fail-safe validator, which would hide the exact
    failure this guard must detect and raise on.

    No project config context at all (e.g. running outside a configured
    OSPREY project — most unit-test environments) is treated the same way
    `LimitsValidator.from_config` treats it: nothing to probe, so this
    returns without blocking startup, rather than raising on the config
    lookup itself.

    Raises:
        RuntimeError: naming the lane, its target, and which condition failed
            (the resolved limits key, so an operator is sent to the line that
            actually answered rather than to a deployment-wide one some
            per-type block overrides, whether the limits posture could be read
            at all — a leaf missing from a per-type block or written as
            anything but a literal boolean in either scope names every such
            leaf — and whether the database path was configured/found/
            parseable) — never the database's file contents or any other
            secret value.
    """
    from osprey_connectors.config import get_config_value
    from osprey_connectors.types import target_limits_posture

    from .queue_backend import lane_tier_allows, resolve_lane_identity

    lane, lane_target = resolve_lane_identity()

    try:
        section = get_config_value("control_system", {})
    except (FileNotFoundError, KeyError, RuntimeError):
        return

    if not lane_tier_allows(section):
        return
    posture = target_limits_posture(section, lane_target)
    if posture.incomplete:
        # The connector and the hook both build the blocking failsafe from
        # this same posture, so no write would get through anyway; the gate
        # exists to turn that into a startup refusal that names the lines.
        unreadable = ", ".join(posture.key(leaf) for leaf in posture.incomplete)
        raise RuntimeError(
            f"refusing to start writable: lane {lane} serves target {lane_target}, "
            f"where the write tier allows writes, but its limits posture cannot be read: "
            f"{unreadable} must each be a literal true or false"
        )
    if posture.enabled is not True:
        return

    limits_key = posture.key("enabled")

    from osprey_connectors.control_system import (
        LIMITS_DATABASE_CONFIG_KEY,
        LimitsValidator,
    )
    from osprey_connectors.types import limits_database_required

    try:
        configured = LimitsValidator.configured_database_path()
    except (FileNotFoundError, KeyError, RuntimeError):
        return
    if not limits_database_required(posture, path_configured=configured is not None):
        return
    if configured is None:
        raise RuntimeError(
            f"refusing to start writable: lane {lane} serves target {lane_target}, "
            f"where the write tier allows writes and {limits_key} is set, but "
            f"{LIMITS_DATABASE_CONFIG_KEY} is not configured"
        )

    loaded, reason = LimitsValidator.load_configured_database()
    if loaded is None:
        raise RuntimeError(
            f"refusing to start writable: lane {lane} serves target {lane_target}, "
            f"where the write tier allows writes and {limits_key} is set, but the configured "
            f"{LIMITS_DATABASE_CONFIG_KEY} could not be read or parsed: {reason}"
        )


def _validate_launchable_request(request: Any) -> None:
    """Refuse to enqueue a session/unreviewed plan with no CURRENT passing validation record.

    Defense-in-depth alongside the session-layer LOAD gate
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
    record — whether or not `get_facility_plans()` currently registers it. The
    refusal body is the queue surface's contract shape, NOT a bare string: this
    gate runs on the enqueue path ahead of `session_upload`'s own
    admissibility check, so for an edited-after-pass session plan it is the
    refusal a caller actually receives, and it must carry the same
    `session_plan_unvalidated` code that check would have carried. Anything
    less and `detail.code` is unreadable for exactly the case the code exists
    to name.
    A name the load gate is quarantining *right now* for lacking a record
    resolves to no `PlanSpec` at all, but its file still exists under the
    session directory; treating that as `session` provenance too (rather than
    "not found") is what turns an already-quarantined plan's launch attempt
    into this clear 409 instead of a confusing "unknown plan" failure further
    downstream. A non-session provenance (`shipped`/`preset`/`facility`), or a
    name with neither a `PlanSpec` nor a session-dir file at all, is left
    alone — the queue server's own "plan not in the allowed namespace"
    rejection is the right place for the latter.
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
        # either way there is nothing here to re-hash, and the manager rejects
        # the item outright if the name doesn't resolve in its namespace.
        return

    content = plan_path.read_text(encoding="utf-8")
    if not validation_records.has_passing_record(hash_plan_body(content)):
        # `reason` and `code` carry the same value, mirroring
        # `SessionPlanError.to_dict()` plus the `code` key `queue.py`'s
        # `_session_refusal` adds — this refusal and that one are the same
        # answer to the same question, so they must look identical on the wire.
        from .session_upload import REASON_UNVALIDATED

        raise HTTPException(
            status_code=409,
            detail={
                "code": REASON_UNVALIDATED,
                "reason": REASON_UNVALIDATED,
                "detail": (
                    f"session plan {plan_name!r} has no passing validation record; "
                    "validate it before enqueuing"
                ),
                "plan": plan_name,
            },
        )
