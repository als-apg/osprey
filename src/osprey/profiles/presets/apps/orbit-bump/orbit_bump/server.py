"""The application's backend: the panel, and the state, approval and plan routes behind its door.

The application has no machine of its own. A state read answers from the machine the
request's owner is on, which the owner's record says; nobody tells the backend. Whether a
write may go ahead is the connector's verdict for that owner. The models, their BPMs and
correctors and each model's response matrix come from the SDK; this module adds no facility
knowledge.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import math
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from orbit_bump import APP_NAME
from orbit_bump.orm import READY as ORM_READY
from orbit_bump.orm import OrmCache, OrmSnapshot
from orbit_bump.plans import (
    APPLIED,
    APPLYING,
    DISCARDED,
    NOT_APPLICABLE,
    PARTIALLY_APPLIED,
    READY,
    PlanConflict,
    PlanNotCurrent,
    PlanNotFound,
    PlanStore,
)
from orbit_bump.ring import PlaneView, RingView, build_ring
from orbit_bump.solver import (
    CLOSURE,
    MANUAL,
    SelectResult,
    select_correctors,
    solve_bump,
)
from osprey.apps.sdk import (
    LAUNCH_TOKEN_HEADER,
    OWNER_STATE_KEY,
    ModelMembers,
    bind_owner,
    configure_app,
    owner_from_header,
    owner_target,
    prewrite_verdict,
    refusal_phrase,
    verify_launch_token,
    write_verdict,
)
from osprey_connectors.call_timeout import DEFAULT_TIMEOUT_S, call_timeout_s
from osprey_connectors.config import get_config_value
from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError
from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors
from osprey_connectors.posture_store import RESERVED_OWNER_KWARG
from osprey_connectors.types import resolve_target

_log = logging.getLogger(__name__)

#: Where the deployment hands this application its launch token.
LAUNCH_TOKEN_ENV = APP_NAME.upper().replace("-", "_") + "_LAUNCH_TOKEN"

#: The panel's state.
STATE_ROUTE = "/api/state"
#: The approval hook's state: the current ready plan's id, digest and display lines.
APPROVAL_ROUTE = "/api/approval"
#: Start a model's response-matrix build.
RECOMPUTE_ROUTE = "/api/recompute"
#: Discard the owner's current plan.
DISCARD_ROUTE = "/api/discard"
#: The panel's draft: what the owner has set, computed or not, for the agent to read.
DRAFT_ROUTE = "/api/draft"
#: Solve a bump and store it as the owner's current plan.
COMPUTE_ROUTE = "/api/compute"
#: Write the owner's current plan to the machine.
APPLY_ROUTE = "/api/apply"
#: The prefix the panel's MML signed-index rules carry; the solver takes the list bare.
MML_RULE_PREFIX = "mml:"
#: The other BPMs (every BPM of the plane not in the bump) are left out of the solve.
FREE = "free"
#: The other BPMs aim at their orbit at compute: a local bump.
HOLD = "hold"
#: The other BPMs aim at their golden orbit.
ZERO = "zero"
#: A bump BPM is missed when its predicted position is off its aim by more than this
#: (in the BPM unit) or by ``MISS_FRACTION`` of the largest target offset, the larger.
MISS_FLOOR = 0.001
MISS_FRACTION = 0.05

#: Seconds an apply waits for the written correctors' readbacks to settle.
SETTLE_TIMEOUT_S = 5.0
#: Seconds between two settle reads.
SETTLE_POLL_S = 0.1

#: A row's write outcomes (the connector's words) and the one for a row never written.
CONFIRMED = "confirmed"
REFUSED = "refused"
UNCONFIRMED = "unconfirmed"
NOT_ATTEMPTED = "not_attempted"

#: The approval hook's caps: characters per display line, lines in all.
MAX_DISPLAY_LINE_CHARS = 200
MAX_DISPLAY_LINES = 20
#: What a row without a tolerance says on the approval card.
NO_TOLERANCE_NOTE = "no tolerance: not staleness-checked"
#: Between two corrector rows packed onto one display line.
ROW_SEPARATOR = " | "

#: The environment a container keeps when compose names its Channel Access name server.
NAME_SERVER_ENV = "EPICS_CA_NAME_SERVERS"
_KEPT_EPICS_ENV = frozenset({NAME_SERVER_ENV, "EPICS_CA_AUTO_ADDR_LIST"})

ConnectorFor = Callable[[str], Awaitable[Any]]


class DisplayOverflow(ValueError):
    """A plan's missed targets and closure do not fit the approval hook's display lines."""


def panel_dir() -> Path:
    """The panel's files: inside the installed package, else beside it in the source tree."""
    package = Path(__file__).resolve().parent
    installed = package / "panel"
    return installed if installed.is_dir() else package.parent / "panel"


def _section() -> dict[str, Any]:
    section = get_config_value("control_system", {})
    return section if isinstance(section, dict) else {}


def _keep_name_server_pair() -> None:
    """Drop every inherited EPICS variable but the name-server pair compose sets."""
    for name in [n for n in os.environ if n.startswith("EPICS_")]:
        if name not in _KEPT_EPICS_ENV:
            del os.environ[name]


async def build_connector(target: str) -> Any:
    """A connector for the control target *target*, connected."""
    section = _section()
    kind = resolve_target(section, target)
    config = {**section, "type": kind}
    if os.environ.get(NAME_SERVER_ENV):
        # In its container: compose names the name server it reaches. The config's
        # gateways are the host's view, and connecting through them would replace it.
        _keep_name_server_pair()
        connectors = dict(section.get("connector") or {})
        block = {k: v for k, v in (connectors.get(kind) or {}).items() if k != "gateways"}
        config["connector"] = {**connectors, kind: block}
    register_builtin_connectors()  # idempotent; a fresh process has none registered
    return await ConnectorFactory.create_control_system_connector(config, control_target=target)


_READ_MARGIN_S = 1.0
"""Seconds the whole batch read may outlast the connector's per-channel bound."""


def _call_bound(section: Mapping[str, Any], target: str) -> float:
    """The target connector's call bound, in seconds."""
    try:
        kind = resolve_target(section, target)
        block = (section.get("connector") or {}).get(kind) or {}
        return call_timeout_s(block, kind)
    except (ValueError, TypeError, AttributeError):
        return DEFAULT_TIMEOUT_S


def setpoint_digest(correctors: Iterable[Mapping[str, Any]]) -> str:
    """The sha256 of the canonical JSON ``[[setpoint_addr, new], ...]``, in plan order."""
    pairs = [[row["setpoint_addr"], row["new"]] for row in correctors]
    text = json.dumps(
        pairs, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _width(line: str) -> int:
    return len(html.escape(line, quote=True))


def _row_text(row: Mapping[str, Any]) -> str:
    unit = f" {row['unit']}" if row.get("unit") else ""
    text = (
        f"{row['element']} {row['setpoint_addr']}: {row['now']:.6g} -> {row['new']:.6g}{unit}"
        f" (delta {row['delta']:+.6g})"
    )
    if row.get("tolerance") is None:
        text += f", {NO_TOLERANCE_NOTE}"
    return text


def _pack(pieces: Iterable[str], separator: str) -> list[str]:
    """Greedy-pack *pieces* into lines that fit the display cap."""
    lines: list[str] = []
    for piece in pieces:
        if _width(piece) > MAX_DISPLAY_LINE_CHARS:
            raise DisplayOverflow(f"{piece[:40]}... is longer than one display line")
        if lines and _width(lines[-1] + separator + piece) <= MAX_DISPLAY_LINE_CHARS:
            lines[-1] += separator + piece
        else:
            lines.append(piece)
    return lines


def miss_text(plan: Mapping[str, Any]) -> str | None:
    """The line naming the bump BPMs the plan misses; ``None`` when it misses none."""
    missed = list(plan.get("missed") or [])
    if not missed:
        return None
    unit = f" {plan['bpm_unit']}" if plan.get("bpm_unit") else ""
    worst = plan.get("target_miss_max") or 0.0
    return f"targets missed: up to {worst:.3f}{unit} at {', '.join(missed)}"


ALSO_WRITES = "also writes: "


def _summary_text(
    correctors: list[Mapping[str, Any]], plan: Mapping[str, Any], listed: int, unchecked: int
) -> str:
    unit = next((f" {row['unit']}" for row in correctors if row.get("unit")), "")
    largest = max((abs(row["delta"]) for row in correctors), default=0.0)
    return (
        f"{len(correctors)} correctors, max |delta| {largest:.6g}{unit}, "
        f"{len(plan.get('bounded') or [])} bounded, {unchecked} not staleness-checked, "
        f"setpoint digest {setpoint_digest(correctors)[:12]}; {listed} rows in full, "
        f"the rest named below"
    )


def _run_tokens(hidden: list[tuple[int, Mapping[str, Any]]], ranges: bool) -> list[str]:
    """The hidden rows as runs of consecutive plan positions: ``HCM1-HCM23 (0.1-0.5)``."""
    runs: list[list[tuple[int, Mapping[str, Any]]]] = []
    for item in hidden:
        if runs and runs[-1][-1][0] + 1 == item[0]:
            runs[-1].append(item)
        else:
            runs.append([item])
    tokens = []
    for run in runs:
        first, last = run[0][1]["element"], run[-1][1]["element"]
        token = first if first == last and len(run) == 1 else f"{first}\u2013{last}"
        if ranges:
            sizes = [abs(row["delta"]) for _, row in run]
            low, high = min(sizes), max(sizes)
            token += f" ({low:.3g})" if low == high else f" ({low:.3g}\u2013{high:.3g})"
        tokens.append(token)
    return tokens


def _pack_names(tokens: list[str]) -> list[str]:
    """Greedy-pack *tokens* into ``also writes:`` lines that fit the display cap."""
    lines: list[str] = []
    for token in tokens:
        if _width(ALSO_WRITES + token) > MAX_DISPLAY_LINE_CHARS:
            raise DisplayOverflow(f"{token[:40]}... is longer than one display line")
        if lines and _width(lines[-1] + ", " + token) <= MAX_DISPLAY_LINE_CHARS:
            lines[-1] += ", " + token
        else:
            lines.append(ALSO_WRITES + token)
    return lines


def _summary_block(
    correctors: list[Mapping[str, Any]], plan: Mapping[str, Any], room: int
) -> list[str]:
    """The summary line, the rows in full and the lines naming every hidden row.

    The rows without a tolerance are always in full; the rest by falling |delta| as long as
    the names of the others still fit in *room* lines.

    Raises:
        DisplayOverflow: No split leaves one full row and every hidden name in *room* lines.
    """
    unchecked = [i for i, row in enumerate(correctors) if row.get("tolerance") is None]
    pinned = set(unchecked)
    rest = sorted(
        (i for i in range(len(correctors)) if i not in pinned),
        key=lambda i: -abs(correctors[i]["delta"]),
    )
    cause = "no split of the plan's rows fits"
    for count in range(len(rest), -1, -1):
        shown = pinned | set(rest[:count])
        if not shown:
            cause = "the summary would list no row in full"
            continue
        order = [i for i in range(len(correctors)) if i in shown]
        by_size = sorted(order, key=lambda i: (i not in pinned, -abs(correctors[i]["delta"])))
        try:
            row_lines = _pack((_row_text(correctors[i]) for i in by_size), ROW_SEPARATOR)
            hidden = [(i, correctors[i]) for i in range(len(correctors)) if i not in shown]
            for ranges in (True, False):
                names = _pack_names(_run_tokens(hidden, ranges)) if hidden else []
                if len(row_lines) + len(names) <= room:
                    summary = _summary_text(correctors, plan, len(shown), len(unchecked))
                    if _width(summary) > MAX_DISPLAY_LINE_CHARS:
                        raise DisplayOverflow("the summary line is longer than one display line")
                    return [summary, *row_lines, *names]
        except DisplayOverflow as exc:
            cause = str(exc)
    raise DisplayOverflow(
        f"the {len(unchecked)} rows without a tolerance and the names of the other rows "
        f"do not fit the approval card ({cause})"
    )


def display_lines(plan: Mapping[str, Any]) -> list[str]:
    """The approval card's lines for *plan*: its corrector rows, the targets it misses,
    then the closure.

    When every row does not fit, a summary line (corrector count, largest |delta|,
    bounded and not-staleness-checked counts, short setpoint digest), the rows without a
    tolerance and the largest-|delta| rows in full, and ``also writes:`` lines naming
    every other corrector take the rows' place. Each line is at most ``MAX_DISPLAY_LINE_CHARS`` once HTML-escaped,
    and there are at most ``MAX_DISPLAY_LINES`` in all.

    Raises:
        DisplayOverflow: The summary, the missed targets and the closure do not fit.
    """
    correctors = list(plan.get("correctors") or [])
    dropped = list(plan.get("dropped_bpms") or []) or ["none"]
    skipped = list(plan.get("skipped_unsolved") or []) or ["none"]
    closure_text = f"dropped_bpms: {', '.join(dropped)}; skipped_unsolved: {', '.join(skipped)}"
    missed = miss_text(plan)
    tail = (_pack(missed.split(" "), " ") if missed else []) + _pack(closure_text.split(" "), " ")
    try:
        rows = _pack((_row_text(row) for row in correctors), ROW_SEPARATOR)
    except DisplayOverflow:
        rows = None
    if rows is not None and len(rows) + len(tail) <= MAX_DISPLAY_LINES:
        return rows + tail
    room = MAX_DISPLAY_LINES - 1 - len(tail)
    if room < 0:
        raise DisplayOverflow(
            f"the missed targets and the closure need {len(tail)} display lines; the "
            f"approval card shows {MAX_DISPLAY_LINES}"
        )
    return [*_summary_block(correctors, plan, room), *tail]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class RecomputeRequest(BaseModel):
    """Which model's response matrix to build again."""

    model_config = ConfigDict(extra="forbid")

    model: str


class ComputeRequest(BaseModel):
    """The bump to solve: BPM offsets from golden, and how the correctors are picked."""

    model_config = ConfigDict(extra="forbid")

    model: str
    plane: Literal["x", "y"]
    targets: dict[str, float]
    rule: str
    correctors: list[str] | None = None
    priority: Literal["closure", "targets"] = CLOSURE
    others: Literal["free", "hold", "zero"] = HOLD
    all_correctors: bool = False


class DraftRequest(BaseModel):
    """What the owner has set in the panel, computed or not."""

    model_config = ConfigDict(extra="forbid")

    plane: Literal["x", "y"]
    model: str | None = None
    targets: dict[str, float] = {}
    rule: str = "auto"
    correctors: list[str] = []
    priority: Literal["closure", "targets"] = CLOSURE
    others: Literal["free", "hold", "zero"] = HOLD
    all_correctors: bool = False
    shape: Literal["flat", "peak", "cos"] = "flat"
    amplitude: float = 0.0


class ApplyRequest(BaseModel):
    """The approved pair: the plan's id and the digest of its setpoints."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    setpoint_digest: str


class ComputeRefused(Exception):
    """A Compute the route answers with ``status`` and ``detail`` instead of a plan."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _solver_rule(rule: str) -> str:
    """The panel's rule as the solver names it: ``mml:[-1 1]`` -> ``[-1 1]``."""
    return rule.removeprefix(MML_RULE_PREFIX)


def _positions_of(members: Iterable[Any], elements: Iterable[str], kind: str) -> list[int]:
    """The ring positions of ``elements`` among ``members``; 422 naming an unknown one."""
    at = {member.element: position for position, member in enumerate(members)}
    found = []
    for element in elements:
        if element not in at:
            raise ComputeRefused(
                422, f"{element} is not a {kind} of this plane; the {kind}s: {', '.join(at)}"
            )
        found.append(at[element])
    return found


def _bump_reads(
    plane: PlaneView, bump: list[int], values: Mapping[str, float | None]
) -> list[float]:
    """Each bump BPM's golden value; 409 naming a bump BPM that blocks the Compute."""
    golden = []
    for b in bump:
        bpm = plane.bpms[b]
        if values.get(bpm.position_addr) is None:
            raise ComputeRefused(
                409, f"bump BPM {bpm.element} ({bpm.position_addr}) is unreadable or not finite"
            )
        if bpm.golden_addr is None:
            raise ComputeRefused(409, f"bump BPM {bpm.element} has no golden orbit")
        value = values.get(bpm.golden_addr)
        if value is None:
            raise ComputeRefused(
                409,
                f"the golden orbit of bump BPM {bpm.element} ({bpm.golden_addr}) is "
                "unreadable or not finite",
            )
        golden.append(value)
    return golden


def _delta_bounds(
    plane: PlaneView, selected: list[int], values: Mapping[str, float | None], target: str
) -> tuple[np.ndarray, np.ndarray]:
    """Each picked corrector's delta bounds, from the limits its pre-write verdict judges by.

    ``[min − now, max − now]``, narrowed to ``±max_step``; open where the record says
    nothing, limits are off, or the record cannot be met from ``now`` (its verdict refuses).
    """
    lower = np.full(len(selected), -np.inf)
    upper = np.full(len(selected), np.inf)
    for i, c in enumerate(selected):
        cor = plane.correctors[c]
        now = values[cor.setpoint_addr]
        if now is None:
            continue
        limits = prewrite_verdict(cor.setpoint_addr, now, target=target, current=now).limits
        if not limits:
            continue
        lo, hi = -np.inf, np.inf
        if limits.get("min") is not None:
            lo = limits["min"] - now
        if limits.get("max") is not None:
            hi = limits["max"] - now
        if limits.get("max_step") is not None:
            lo, hi = max(lo, -limits["max_step"]), min(hi, limits["max_step"])
        if lo <= hi:
            lower[i], upper[i] = lo, hi
    return lower, upper


def solve_plan(
    body: ComputeRequest,
    *,
    target: str,
    view: RingView,
    snapshot: OrmSnapshot,
    values: Mapping[str, float | None],
    created_at: float,
) -> dict[str, Any]:
    """The plan for ``body`` on one ORM snapshot and one read of the machine.

    ``values`` holds every address :func:`compute_reads` names. The plan is ``ready``,
    or ``not_applicable`` with its ``reason`` when a row's pre-write verdict refuses or
    the missed targets and the closure do not fit the approval card.

    Raises:
        ComputeRefused: 409 or 422, naming the cause.
    """
    plane = view.plane(body.plane)
    bump, picked = compute_pick(body, view, snapshot)
    golden = _bump_reads(plane, bump, values)
    for c in picked.selected:
        cor = plane.correctors[c]
        if values.get(cor.setpoint_addr) is None:
            raise ComputeRefused(
                409, f"the setpoint of corrector {cor.element} ({cor.setpoint_addr}) is unreadable"
            )
    orbit = {bpm.element: values.get(bpm.position_addr) for bpm in plane.bpms}
    # The orbit each other BPM aims at: its orbit at compute, or its golden orbit.
    aim: dict[str, float | None] = dict(orbit)
    if body.others == ZERO:
        aim = {
            bpm.element: values.get(bpm.golden_addr) if bpm.golden_addr else None
            for bpm in plane.bpms
        }
    dropped = [
        b
        for b, bpm in enumerate(plane.bpms)
        if b not in bump and (orbit[bpm.element] is None or aim[bpm.element] is None)
    ]
    used = [b for b in range(len(plane.bpms)) if b not in dropped]
    row_of = {b: r for r, b in enumerate(used)}
    if not picked.selected:
        raise ComputeRefused(409, "no solved corrector is left to make the bump")
    # The held rows: every other BPM, inside the corrector span and outside it.
    leak = [] if body.others == FREE else [b for b in used if b not in bump]
    if body.others != FREE and not leak:
        verb = "hold" if body.others == HOLD else "steer to golden"
        raise ComputeRefused(409, f"no other BPM is readable to {verb}; set other BPMs free")
    try:
        matrix = snapshot.slice(
            [plane.bpms[b].position_addr for b in used],
            [plane.correctors[c].setpoint_addr for c in picked.selected],
        )
    except KeyError as exc:
        raise ComputeRefused(409, str(exc.args[0])) from None
    now_orbit = np.array([orbit[plane.bpms[b].element] for b in used], dtype=float)
    goal = now_orbit.copy()
    for b in leak:
        goal[row_of[b]] = aim[plane.bpms[b].element]
    for b, gold in zip(bump, golden, strict=True):
        goal[row_of[b]] = gold + body.targets[plane.bpms[b].element]
    lower, upper = _delta_bounds(plane, picked.selected, values, target)
    try:
        solved = solve_bump(
            matrix,
            now_orbit,
            goal,
            [row_of[b] for b in bump],
            [row_of[b] for b in leak],
            lower=lower,
            upper=upper,
            priority=body.priority,
        )
    except ValueError as exc:
        raise ComputeRefused(409, str(exc)) from None

    rows = []
    for c, delta in zip(picked.selected, solved.delta, strict=True):
        cor = plane.correctors[c]
        now = values[cor.setpoint_addr]
        new = float(now + delta)
        verdict = prewrite_verdict(cor.setpoint_addr, new, target=target, current=now)
        rows.append(
            {
                "element": cor.element,
                "setpoint_addr": cor.setpoint_addr,
                "readback_addr": cor.readback_addr,
                "now": now,
                "readback_at_compute": values.get(cor.readback_addr) if cor.readback_addr else None,
                "new": new,
                "delta": float(delta),
                "unit": cor.unit,
                "verdict": asdict(verdict),
                "tolerance": cor.tolerance,
            }
        )
    predicted: dict[str, float | None] = {bpm.element: None for bpm in plane.bpms}
    for b in used:
        predicted[plane.bpms[b].element] = float(solved.predicted[row_of[b]])
    leakage = []
    for b in leak:
        element = plane.bpms[b].element
        residual = predicted[element] - aim[element]  # type: ignore[operator]
        leakage.append({"element": element, "predicted_residual": residual})
    miss = {
        plane.bpms[b].element: float(solved.predicted[row_of[b]] - goal[row_of[b]]) for b in bump
    }
    allowed_miss = max(MISS_FLOOR, MISS_FRACTION * max(abs(v) for v in body.targets.values()))
    plan: dict[str, Any] = {
        "target": target,
        "model": view.model,
        "plane": body.plane,
        "rule": body.rule,
        "priority": body.priority,
        "others": body.others,
        "all_correctors": body.all_correctors,
        "orm_version": snapshot.version,
        "created_at": created_at,
        "status": READY,
        "reason": None,
        "targets": dict(body.targets),
        "bpm_unit": next((bpm.unit for bpm in plane.bpms if bpm.unit), None),
        "miss": miss,
        "target_miss_max": max(abs(v) for v in miss.values()),
        "missed": [element for element, v in miss.items() if abs(v) > allowed_miss],
        "setpoint_digest": setpoint_digest(rows),
        "correctors": rows,
        "orbit_at_compute": orbit,
        "predicted": predicted,
        "leakage": leakage,
        "predicted_closure_max": max(
            (abs(entry["predicted_residual"]) for entry in leakage), default=0.0
        ),
        "sigma_kept": solved.sigma_kept,
        "sigma_total": solved.sigma_total,
        "bounded": [plane.correctors[picked.selected[c]].element for c in solved.bounded],
        "skipped_unsolved": [plane.correctors[c].element for c in picked.skipped_unsolved],
        "dropped_bpms": [plane.bpms[b].element for b in dropped],
    }
    refused_rows = [row for row in rows if not row["verdict"]["allowed"]]
    if refused_rows:
        plan["status"] = NOT_APPLICABLE
        plan["reason"] = "; ".join(
            f"{row['element']} ({row['setpoint_addr']}) refused: {row['verdict']['reason']}"
            for row in refused_rows
        )
        return plan
    try:
        display_lines(plan)
    except DisplayOverflow as exc:
        plan["status"] = NOT_APPLICABLE
        plan["reason"] = str(exc)
    return plan


def compute_pick(
    body: ComputeRequest, view: RingView, snapshot: OrmSnapshot
) -> tuple[list[int], Any]:
    """The bump BPM positions and the rule's corrector pick (unsolved ones skipped).

    Raises:
        ComputeRefused: 422 on an unknown BPM, corrector or rule, or a non-finite offset.
    """
    plane = view.plane(body.plane)
    if not body.targets:
        raise ComputeRefused(422, "a bump needs at least one target BPM")
    for element, offset in body.targets.items():
        if not math.isfinite(offset):
            raise ComputeRefused(422, f"the offset of {element} is not finite")
    bump = _positions_of(plane.bpms, body.targets, "BPM")
    if body.all_correctors:
        # Every solved corrector of the plane, rule aside.
        unsolved = view.unsolved_positions(snapshot.unsolved, body.plane)
        every = range(len(plane.correctors))
        return bump, SelectResult(
            selected=[c for c in every if c not in unsolved],
            skipped_unsolved=[c for c in every if c in unsolved],
        )
    manual = None
    if body.rule == MANUAL:
        if not body.correctors:
            raise ComputeRefused(422, "the manual rule needs the ticked correctors")
        manual = _positions_of(plane.correctors, body.correctors, "corrector")
    try:
        picked = select_correctors(
            bump,
            plane.next_cor,
            len(plane.correctors),
            _solver_rule(body.rule),
            manual,
            view.unsolved_positions(snapshot.unsolved, body.plane),
        )
    except ValueError as exc:
        raise ComputeRefused(422, str(exc)) from None
    return bump, picked


def compute_reads(body: ComputeRequest, view: RingView, snapshot: OrmSnapshot) -> list[str]:
    """Every address one Compute reads: the plane's BPM positions, the bump BPMs'
    golden (every BPM's when the other BPMs aim at golden), and the picked correctors'
    setpoints and readbacks."""
    plane = view.plane(body.plane)
    bump, picked = compute_pick(body, view, snapshot)
    addrs = [bpm.position_addr for bpm in plane.bpms]
    gold_of = range(len(plane.bpms)) if body.others == ZERO else bump
    addrs += [plane.bpms[b].golden_addr for b in gold_of]
    for c in picked.selected:
        addrs += [plane.correctors[c].setpoint_addr, plane.correctors[c].readback_addr]
    return list(dict.fromkeys(a for a in addrs if a))


def stale_rows(plan: Mapping[str, Any], values: Mapping[str, float | None]) -> list[str]:
    """What moved since Compute, one sentence per corrector that is past its tolerance.

    A row's setpoint is held to ``now`` and its readback to ``readback_at_compute``; an
    unreadable one counts as moved. A row without tolerance is not checked, and a readback
    nobody read at Compute has nothing to be compared with.
    """
    moved = []
    for row in plan.get("correctors") or []:
        tolerance = row.get("tolerance")
        if tolerance is None:
            continue
        name = f"corrector {row['element']}"
        setpoint = values.get(row["setpoint_addr"])
        if setpoint is None:
            moved.append(f"the setpoint of {name} ({row['setpoint_addr']}) is unreadable")
        elif abs(setpoint - row["now"]) > tolerance:
            moved.append(
                f"the setpoint of {name} is {setpoint:.6g}, {row['now']:.6g} at compute "
                f"(tolerance {tolerance:g})"
            )
        before = row.get("readback_at_compute")
        if before is None or not row.get("readback_addr"):
            continue
        readback = values.get(row["readback_addr"])
        if readback is None:
            moved.append(f"the readback of {name} ({row['readback_addr']}) is unreadable")
        elif abs(readback - before) > tolerance:
            moved.append(
                f"the readback of {name} is {readback:.6g}, {before:.6g} at compute "
                f"(tolerance {tolerance:g})"
            )
    return moved


async def _write_row(live: Any, row: dict[str, Any]) -> None:
    """Write one row with a confirmed write and record its outcome on it (R5 CC-1)."""
    try:
        result = await live.write_channel(row["setpoint_addr"], row["new"], confirm=True)
    except ChannelLimitsViolationError as exc:
        row["outcome"] = REFUSED
        row["reason"] = f"{exc.violation_type}: {exc.violation_reason}"
        return
    except ChannelWriteBlockedError as exc:
        row["outcome"] = REFUSED
        row["reason"] = str(exc)
        return
    except Exception as exc:  # what the channel holds is unknown
        row["outcome"] = UNCONFIRMED
        row["reason"] = f"the write raised: {exc}"
        return
    row["outcome"] = str(result.outcome)
    message = getattr(result, "error_message", None)
    if message:
        row["reason"] = message


async def _settle(
    rows: list[dict[str, Any]],
    read: Callable[[list[str]], Awaitable[dict[str, float | None]]],
) -> None:
    """Wait until every written row's readback is within its tolerance of ``new``.

    Bounded by ``SETTLE_TIMEOUT_S``; a row without tolerance (or readback) waits it out
    and is never settled. Each row records its last readback and ``settled``.
    """
    for row in rows:
        row["settled"] = False
        row.setdefault("readback", None)
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    while True:
        open_rows = [row for row in rows if not row["settled"] and row.get("readback_addr")]
        values = await read([row["readback_addr"] for row in open_rows])
        for row in open_rows:
            readback = values.get(row["readback_addr"])
            if readback is not None:
                row["readback"] = readback
            tolerance = row.get("tolerance")
            if (
                tolerance is not None
                and readback is not None
                and abs(readback - row["new"]) <= tolerance
            ):
                row["settled"] = True
        waiting = [row for row in rows if not row["settled"]]
        if not waiting or time.monotonic() >= deadline:
            return
        await asyncio.sleep(min(SETTLE_POLL_S, max(0.0, deadline - time.monotonic())))


def residual_after(
    plan: Mapping[str, Any], plane: PlaneView, values: Mapping[str, float | None]
) -> dict[str, float]:
    """Each bump BPM's position less its aim, each held BPM's less its orbit at compute
    (its golden orbit when the plan aims the other BPMs at golden).

    An unreadable BPM is left out.
    """
    bpms = {bpm.element: bpm for bpm in plane.bpms}
    residual: dict[str, float] = {}
    for element, offset in (plan.get("targets") or {}).items():
        bpm = bpms.get(element)
        if bpm is None or bpm.golden_addr is None:
            continue
        position = values.get(bpm.position_addr)
        golden = values.get(bpm.golden_addr)
        if position is not None and golden is not None:
            residual[element] = position - (golden + offset)
    orbit = plan.get("orbit_at_compute") or {}
    for entry in plan.get("leakage") or []:
        element = entry["element"]
        bpm = bpms.get(element)
        if bpm is None:
            continue
        position = values.get(bpm.position_addr)
        if plan.get("others") == ZERO:
            aim = values.get(bpm.golden_addr) if bpm.golden_addr else None
        else:
            aim = orbit.get(element)
        if position is not None and aim is not None:
            residual[element] = position - aim
    return residual


def create_app(
    connector_for: ConnectorFor = build_connector,
    *,
    sdk: Any = None,
    clock: Callable[[], float] = time.time,
) -> FastAPI:
    """Build the backend.

    Args:
        connector_for: Builds the connector of one control target.
        sdk: The object holding ``closed_orbit_models``, ``model_members`` and
            ``model_response_matrix``; the application SDK when omitted.
        clock: The wall clock the ORM's age is measured on.
    """
    if sdk is None:
        from osprey.apps import sdk as default_sdk

        sdk = default_sdk
    connectors: dict[str, Any] = {}
    # One bring-up per target, shielded so a poll that stops waiting leaves it running
    # for the next poll to reuse instead of abandoning a half-connected connector.
    bringing_up: dict[str, asyncio.Task[Any]] = {}
    store = PlanStore()
    # A successful build discards the plans solved on the old matrix (R4 CC-2).
    orm = OrmCache(sdk=sdk, clock=clock, on_ready=store.discard_for_orm_change)
    members_of: dict[str, ModelMembers] = {}
    rings: dict[str, RingView] = {}
    refused: dict[str, ValueError] = {}
    # Each owner's panel draft, in memory only: a restart starts every panel afresh.
    drafts: dict[str | None, dict[str, Any]] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            orm.start()
        except Exception:  # no models to read is not a reason to refuse the panel
            _log.exception("the first response-matrix build did not start")
        yield
        for task in bringing_up.values():
            task.cancel()
        for target, each in list(connectors.items()):
            try:
                await each.disconnect()
            except Exception:  # one bad close must not leave the others open
                _log.warning("the connector of %s did not close cleanly", target, exc_info=True)

    async def connector(target: str, bound: float | None = None) -> Any:
        """The target's connector; waits at most ``bound`` seconds for its bring-up."""
        if target in connectors:
            return connectors[target]
        task = bringing_up.get(target)
        if task is None:
            task = asyncio.ensure_future(connector_for(target))
            bringing_up[target] = task

            def settle(done: asyncio.Task[Any], target: str = target) -> None:
                bringing_up.pop(target, None)
                if not done.cancelled() and done.exception() is None:
                    connectors[target] = done.result()

            task.add_done_callback(settle)
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=bound)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def offered() -> list[str]:
        return list(sdk.closed_orbit_models())

    def ring(model: str) -> tuple[ModelMembers, RingView]:
        """The model's members and ring view, read once; raises ValueError."""
        if model in refused:
            raise refused[model]
        if model not in rings:
            try:
                members = sdk.model_members(model)
            except ValueError as exc:
                refused[model] = exc
                _log.warning("model %s is not shown: %s", model, exc)
                raise
            members_of[model] = members
            rings[model] = build_ring(members)
        return members_of[model], rings[model]

    def owner_of(request: Request) -> str | None:
        return owner_from_header(getattr(request.state, OWNER_STATE_KEY, None))

    def target_of(owner: str | None) -> str:
        target = owner_target(_section(), owner)
        if target is None:
            raise HTTPException(503, f"the control target of {owner} could not be read")
        return target

    def model_entry(model: str) -> dict[str, Any] | None:
        try:
            members, view = ring(model)
        except ValueError:  # logged once, when the refusal was cached
            return None
        state = orm.state(model)
        return {
            "name": model,
            "length_m": members.length_m,
            "orm": {
                "status": state.status,
                "age_s": None if state.built_at is None else clock() - state.built_at,
                "version": state.version,
                "error": state.error,
                "unsolved": view.unsolved_elements(state.unsolved),
            },
            "members": {
                "bpms": [asdict(m) for m in members.bpms],
                "correctors": [asdict(m) for m in members.correctors],
            },
        }

    def posture(target: str, owner: str | None) -> dict[str, Any]:
        verdict = write_verdict(_section(), target, owner=owner)
        phrase = None
        if verdict.reason is not None:
            try:
                phrase = refusal_phrase(verdict.reason, verdict.kind)
            except ValueError:
                phrase = verdict.reason
        return {"allowed": verdict.allowed, "reason": verdict.reason, "phrase": phrase}

    async def read_values(target: str, addrs: list[str]) -> dict[str, float | None]:
        """Each address's finite value, or None; bounded by the connector's call bound.

        The connector bounds each channel by ``bound`` and leaves out the silent ones;
        the outer guard waits a margin longer, so it fires only when the whole read
        hangs and one silent channel never nulls the healthy ones.
        """
        values: dict[str, float | None] = dict.fromkeys(addrs)
        if not addrs:
            return values
        bound = _call_bound(_section(), target)
        try:
            live = await connector(target, bound)
            read_back = await asyncio.wait_for(
                live.read_multiple_channels(addrs, timeout=bound),
                timeout=bound + max(_READ_MARGIN_S, 0.5 * bound),
            )
        except Exception as exc:  # an unreachable machine reads null, never a 5xx
            _log.warning("the orbit of %s could not be read: %s", target, exc)
            return values
        for addr in addrs:
            got = read_back.get(addr) if isinstance(read_back, Mapping) else None
            values[addr] = _finite(getattr(got, "value", None))
        return values

    app = FastAPI(title=APP_NAME, lifespan=lifespan)
    app.state.orm = orm
    app.state.plans = store
    app.state.connectors = connectors

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(panel_dir() / "index.html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(STATE_ROUTE)
    async def state(request: Request) -> dict[str, Any]:
        owner = owner_of(request)
        target = target_of(owner)
        models = [entry for entry in map(model_entry, offered()) if entry is not None]
        positions: list[str] = []
        golden: list[str] = []
        for model in models:
            for bpm in model["members"]["bpms"]:
                if bpm["position_addr"] and bpm["position_addr"] not in positions:
                    positions.append(bpm["position_addr"])
                if bpm["golden_addr"] and bpm["golden_addr"] not in golden:
                    golden.append(bpm["golden_addr"])
        values = await read_values(target, list(dict.fromkeys(positions + golden)))
        return {
            "models": models,
            "posture": posture(target, owner),
            "orbit": {
                "position": {addr: values[addr] for addr in positions},
                "golden": {addr: values[addr] for addr in golden},
            },
            "plan": store.current(owner),
            "draft": drafts.get(owner),
        }

    @app.get(APPROVAL_ROUTE)
    async def approval(request: Request) -> dict[str, Any] | None:
        plan = store.current(owner_of(request))
        if plan is None or plan.get("status") != READY:
            return None
        try:
            lines = display_lines(plan)
        except DisplayOverflow:  # nothing an approver cannot read is approvable
            _log.warning("plan %s does not fit the approval card", plan.get("id"))
            return None
        return {
            "plan_id": plan["id"],
            "setpoint_digest": setpoint_digest(plan.get("correctors") or []),
            "display_lines": lines,
        }

    @app.post(RECOMPUTE_ROUTE, status_code=202)
    async def recompute(body: RecomputeRequest) -> dict[str, Any]:
        models = offered()
        if body.model not in models:
            raise HTTPException(
                409,
                f"model {body.model} is not offered; the offered models: "
                f"{', '.join(models) or 'none'}",
            )
        started = orm.recompute(body.model)
        return {"model": body.model, "started": started}

    @app.post(DRAFT_ROUTE, status_code=204)
    async def draft(body: DraftRequest, request: Request) -> Response:
        # A mirror of the panel's input for the agent to read; nothing is solved or written.
        drafts[owner_of(request)] = {**body.model_dump(), "updated_at": clock()}
        return Response(status_code=204)

    @app.post(DISCARD_ROUTE, status_code=204)
    async def discard(request: Request) -> Response:
        try:
            store.discard(owner_of(request))
        except PlanConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(status_code=204)

    @app.post(COMPUTE_ROUTE, status_code=201)
    async def compute(body: ComputeRequest, request: Request) -> dict[str, Any]:
        # Compute writes nothing, so it plans under any posture; apply is the gated step.
        owner = owner_of(request)
        target = target_of(owner)
        models = offered()
        if body.model not in models:
            raise HTTPException(
                409,
                f"model {body.model} is not offered; the offered models: "
                f"{', '.join(models) or 'none'}",
            )
        try:
            _members, view = ring(body.model)
        except ValueError as exc:
            raise HTTPException(409, f"model {body.model} is refused: {exc}") from exc
        # One immutable snapshot, taken once; a build still running serves no plan.
        entry = orm.ensure(body.model)
        snapshot = orm.snapshot(body.model)
        if entry.status != ORM_READY or snapshot is None or snapshot.version != entry.version:
            detail = f": {entry.error}" if entry.error else ""
            raise HTTPException(
                409,
                f"the response matrix of {body.model} is {entry.status}{detail}; "
                "compute once it is ready",
            )
        try:
            addrs = compute_reads(body, view, snapshot)
            values = await read_values(target, addrs)
            plan = solve_plan(
                body,
                target=target,
                view=view,
                snapshot=snapshot,
                values=values,
                created_at=clock(),
            )
        except ComputeRefused as exc:
            raise HTTPException(exc.status, exc.detail) from exc
        try:
            return store.set_current(owner, plan)
        except PlanConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    def refuse_apply(plan_id: str, status: int, detail: str) -> HTTPException:
        _log.info("apply of plan %s refused (%s): %s", plan_id, status, detail)
        return HTTPException(status, detail)

    @app.post(APPLY_ROUTE)
    async def apply(body: ApplyRequest, request: Request) -> dict[str, Any]:
        verify_launch_token(request.headers.get(LAUNCH_TOKEN_HEADER), LAUNCH_TOKEN_ENV)
        owner = owner_of(request)
        target = target_of(owner)
        stance = posture(target, owner)
        if not stance["allowed"]:
            raise HTTPException(403, stance["phrase"] or stance["reason"] or "writes are refused")
        # Refusals before the claim: the plan is left as it is.
        try:
            plan = store.get(body.plan_id)
        except PlanNotFound:
            current = store.current(owner)
            raise HTTPException(
                409, str(PlanNotCurrent(body.plan_id, owner, current and current["id"]))
            ) from None
        if plan.get("owner") != owner:
            raise HTTPException(403, f"plan {body.plan_id} is {plan.get('owner')}'s, not {owner}'s")
        if plan.get("target") != target:
            raise HTTPException(
                403,
                f"plan {body.plan_id} was computed on {plan.get('target')}; "
                f"{owner} is on {target} now",
            )
        if plan.get("model") not in offered():
            raise HTTPException(403, f"model {plan.get('model')} is no longer offered")
        if body.setpoint_digest != setpoint_digest(plan.get("correctors") or []):
            raise HTTPException(
                409, f"the setpoint digest is not plan {body.plan_id}'s; compute and approve again"
            )
        # The claim, under the store lock, before any read: a second apply gets 409.
        try:
            plan = store.claim(body.plan_id, owner)
        except (PlanNotCurrent, PlanConflict) as exc:
            raise HTTPException(409, str(exc)) from exc

        rows: list[dict[str, Any]] = [dict(row) for row in plan["correctors"]]
        writing = False
        try:
            entry = orm.state(plan["model"])
            if entry.version != plan["orm_version"]:
                store.transition(body.plan_id, DISCARDED)
                raise refuse_apply(
                    body.plan_id,
                    409,
                    f"the response matrix of {plan['model']} is version {entry.version}; "
                    f"plan {body.plan_id} was solved on version {plan['orm_version']} and is "
                    "discarded",
                )
            bound = _call_bound(_section(), target)
            addrs = [row["setpoint_addr"] for row in rows]
            addrs += [row["readback_addr"] for row in rows if row.get("readback_addr")]
            moved = stale_rows(plan, await read_values(target, list(dict.fromkeys(addrs))))
            if moved:
                raise refuse_apply(
                    body.plan_id, 409, "the machine moved since compute: " + "; ".join(moved)
                )
            try:
                live = await connector(target, bound)
            except HTTPException:
                raise
            except Exception as exc:
                raise refuse_apply(
                    body.plan_id, 503, f"the connector of {target} is not available: {exc}"
                ) from exc

            with bind_owner({RESERVED_OWNER_KWARG: owner}):
                for row in rows:
                    row["outcome"] = NOT_ATTEMPTED
                for row in rows:
                    writing = True
                    row["outcome"] = UNCONFIRMED  # in flight: unknown until it answers
                    await _write_row(live, row)
                    store.transition(body.plan_id, APPLYING, correctors=rows)
                    if row["outcome"] != CONFIRMED:
                        break

            written = [row for row in rows if row["outcome"] not in (NOT_ATTEMPTED, REFUSED)]
            store.transition(body.plan_id, APPLYING, phase="settling", correctors=rows)
            await _settle(written, lambda some: read_values(target, some))
            _members, view = ring(plan["model"])
            plane = view.plane(plan["plane"])
            orbit_addrs = [bpm.position_addr for bpm in plane.bpms]
            orbit_addrs += [bpm.golden_addr for bpm in plane.bpms if bpm.golden_addr]
            orbit = await read_values(target, list(dict.fromkeys(a for a in orbit_addrs if a)))
            every_confirmed = all(row["outcome"] == CONFIRMED for row in rows)
            expired = any(
                not row["settled"] and row.get("tolerance") is not None for row in written
            )
            status = APPLIED if every_confirmed and not expired else PARTIALLY_APPLIED
            return store.transition(
                body.plan_id,
                status,
                correctors=rows,
                residual_after=residual_after(plan, plane, orbit),
                applied_at=clock(),
            )
        finally:
            # A plan never stays applying once the request ends.
            if store.get(body.plan_id)["status"] == APPLYING:
                if writing:
                    store.transition(
                        body.plan_id, PARTIALLY_APPLIED, correctors=rows, applied_at=clock()
                    )
                else:
                    store.release(body.plan_id)

    configure_app(app, static_dir=panel_dir())
    return app


def main() -> None:
    """Serve the backend where the deployment says: ``OSPREY_APP_BIND``, ``OSPREY_APP_PORT``."""
    uvicorn.run(
        create_app(),
        host=os.environ.get("OSPREY_APP_BIND", "127.0.0.1"),
        port=int(os.environ["OSPREY_APP_PORT"]),
    )


if __name__ == "__main__":
    main()
