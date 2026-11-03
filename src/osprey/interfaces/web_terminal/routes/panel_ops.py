"""The agent's dock ops: the routes that move, open and close panels on screen.

An agent changes the operator's dock only through these routes. Each write
route checks what the server can check, relays one ``panel_op`` frame to the
browser whose dock report is current (the dock client), and answers with the
layout and ``rev`` that client reports back through ``/api/panel-op-result``,
or with a named refusal, in which case nothing changed — except
``no_dock_answer``: the client did not answer in time and may still apply the
op, so its outcome is unknown until the panels are read again.

One producer. The dock client's layout holder is the only thing that changes
the dock: it answers every op and every report, and the server stores what it
answers. Nothing here applies a layout itself or asks every page to apply one.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from osprey.interfaces.panel_urls import (
    member_navigation_url,
    navigation_url_refusal,
    normalise_navigation_url,
)
from osprey.interfaces.web_terminal.dock_layout import (
    REASON_TEXT,
    layout_keys,
    layout_refusal,
    normalize_layout,
)
from osprey.interfaces.web_terminal.presets import preset_refusal, prune_preset_layout
from osprey.interfaces.web_terminal.routes.panels import (
    _TERMINAL_PANEL_ID,
    _checked_dock_layout,
    _checked_popouts,
    _known_panel_ids,
    _mirror_agent_panel_activity,
    _prefix_path,
    _record_dock_report,
    agent_hop_refusal,
)
from osprey.interfaces.web_terminal.routes.proxy import _resolve_panel_url

router = APIRouter()


# The only agent dock-write path. A write route checks what the server can
# check (argument shapes, known panels, the terminal invariant, a pop-out, a
# stale ``base_rev``), then relays one ``panel_op`` frame addressed to the
# current dock client — the client whose report is current — and waits for that
# client to run the op against its layout holder and POST the outcome to
# ``/api/panel-op-result``. That route records the resulting layout as the
# client's report and mints the ``rev`` the write returns. A refusal at either
# end has a named reason and changes nothing. A client that does not answer in
# time is the one exception: the frame went out, so the write answers
# ``no_dock_answer`` (the outcome is unknown), and the server keeps the request
# id so that the client's late answer is still recorded as its report.

#: How long a write route waits for the dock client's answer. The MCP op helper's
#: socket timeout (8 s) is longer, so a timeout here is always the route's
#: ``no_dock_answer`` refusal and never a dropped connection.
OP_TIMEOUT_S = 5.0

#: How many requests the server remembers after it stopped waiting for their
#: answers (``no_dock_answer``), the oldest forgotten first. A dock client that
#: applies such an op late still answers it, and that answer is recorded.
UNANSWERED_CAP = 64

#: The agent-activity tool name each op is recorded under (the MCP verbs).
OP_ACTIVITY_TOOLS: dict[str, str] = {
    "drop": "drop_panel",
    "close_tab": "close_panel_tab",
    "close_tile": "close_panel_tile",
    "open": "open_panel",
    "open_set": "open_panel_set",
    "open_arrangement": "arrange_workspace",
}

#: Named refusals and the text the agent is shown. A dock client's own refusal
#: reasons (``too_many_for_side``, ``panel_unhealthy``, ...) pass through with
#: the client's text when they are not listed here.
OP_REFUSAL_TEXT: dict[str, str] = {
    "stale_base": "the workspace changed since base_rev was read; read the panels again",
    "no_dock_client": "no dock client is showing the workspace",
    "no_dock_answer": (
        "the dock client did not answer in time and may still apply the op; "
        "read the panels again to see whether it did"
    ),
    "terminal_not_agent_movable": (
        "the Agent pane is always its own tile: never an agent drop source or close "
        "target, and no tab is dropped into it"
    ),
    "terminal_in_layout": "a layout names service panels only, never the terminal",
    "popped_out": (
        "the panel is in a window of its own; the operator brings it back, "
        "an open brings its window forward"
    ),
    "unknown_member": "not a known panel id",
    "invalid_url": (
        "a url is a path under the panel's own mount (/panel/<id>/...) or an http(s) "
        "URL on the panel's own server"
    ),
    "agent_hop_panel_url": (
        "a panel driven by its own workspace tool takes no url, and no url may reach "
        "such a panel's mount or process"
    ),
    "unknown_key": "no such tab",
    "unknown_tile": "no tile with this id in the current layout",
    "unknown_arrangement": "no preset with this name",
    "duplicate_member": "a member is named twice",
    "invalid_source": "a drop source is {new: {panel}}, {tab: key} or {tile: id}",
    "invalid_target": (
        "a drop target is {tile, zone: 'centre'}, {tile, zone: 'tab', index}, "
        "{tile, edge}, {border: {tile, side}} or {outer}"
    ),
    "invalid_layout": "as: 'layout' needs a layout, and only as: 'layout' takes one",
    "duplicate_request": (
        "a request with this request_id is already pending, or still awaits its late answer"
    ),
    "invalid_result": "the dock client answered with a layout the server refused",
}

_SIDES = ("left", "right", "top", "bottom")
_REFUSAL_REASON_RE = re.compile(r"^[a-z][a-z_]{0,63}$")
_REQUEST_ID = Field(min_length=1, max_length=128)


class OpRefused(Exception):
    """An op refused for a named ``reason``; nothing changed.

    The one exception is ``no_dock_answer``: the dock client was sent the op and
    did not answer in time, so the outcome is unknown.

    ``status`` is 422 for a refused argument and 409 for a refusal that depends
    on the workspace's state (stale base, no dock client, a dock client that did
    not answer, a pop-out, the client's own).
    """

    def __init__(self, reason: str, detail: str | None = None, *, status: int = 422):
        self.reason = reason
        self.detail = detail or OP_REFUSAL_TEXT.get(reason, reason)
        self.status = status
        super().__init__(f"{reason}: {self.detail}")


@dataclass
class _PendingOp:
    target: str
    future: asyncio.Future


@dataclass
class _Prepared:
    """A checked op: the frame's ``args``, the history subject, the rail effect."""

    args: dict
    subject: str
    rail_add: list[str] = field(default_factory=list)
    rail_prune: bool = False


@dataclass(frozen=True)
class PresetArrangement:
    """A configured preset resolved for the dock client: its members and its layout over them.

    ``layout`` is the preset's collection-form layout with the members' indexes
    in its tabs: the form every arrangement frame carries.
    """

    members: list[str]
    layout: dict


def _pending_ops(state) -> dict[str, _PendingOp]:
    ops = getattr(state, "panel_ops", None)
    if ops is None:
        ops = {}
        state.panel_ops = ops
    return ops


def _unanswered_ops(state) -> dict[str, str]:
    """The requests the server stopped waiting for, each with its dock client, oldest first."""
    ops = getattr(state, "panel_ops_unanswered", None)
    if ops is None:
        ops = {}
        state.panel_ops_unanswered = ops
    return ops


def _remember_unanswered(state, request_id: str, target: str) -> None:
    """Keep a timed-out request for its late answer, forgetting the oldest past the cap."""
    unanswered = _unanswered_ops(state)
    unanswered[request_id] = target
    while len(unanswered) > UNANSWERED_CAP:
        del unanswered[next(iter(unanswered))]


def _popouts(request: Request) -> list[str]:
    return list(getattr(request.app.state, "dock_popouts", []))


def _tiles_of(layout: dict | None) -> dict[str, list[str]]:
    """Every tile of a dock-form layout: ``{tile id: its tab keys}``."""
    found: dict[str, list[str]] = {}

    def walk(node: dict) -> None:
        if "split" in node:
            for item in node["items"]:
                walk(item)
        else:
            found[node["tile"]] = list(node["tabs"])

    if layout is not None:
        walk(layout)
    return found


def _names_terminal(layout: object) -> bool:
    """Whether a layout of any shape names the terminal; a malformed tree names nothing."""
    try:
        return _TERMINAL_PANEL_ID in layout_keys(layout)  # type: ignore[arg-type]
    except (KeyError, TypeError, AttributeError):
        return False


def _current_tile(
    request: Request, tile: object, *, source: bool = False, zone: bool = False
) -> str:
    """``tile`` checked against the current layout.

    The Agent pane's tile is never a source and never takes a tab (a ``zone``
    target); an edge beside it is fine.
    """
    tiles = _tiles_of(getattr(request.app.state, "dock_layout", None))
    if not isinstance(tile, str) or not tile:
        raise OpRefused("invalid_source" if source else "invalid_target")
    if tile not in tiles:
        raise OpRefused("unknown_tile", f"no tile {tile!r} in the current layout")
    if (source or zone) and _TERMINAL_PANEL_ID in tiles[tile]:
        raise OpRefused("terminal_not_agent_movable")
    return tile


def _known_member(request: Request, member: object) -> str:
    """A panel an agent may open: known, and never the terminal."""
    if member == _TERMINAL_PANEL_ID:
        raise OpRefused("terminal_not_agent_movable")
    if not isinstance(member, str) or member not in _known_panel_ids(request):
        raise OpRefused("unknown_member", f"unknown panel {member!r}")
    return member


def _member_mount_and_upstream(request: Request, member: str) -> tuple[str, str | None]:
    """A known panel's browser mount (no URL prefix) and its upstream URL.

    The mount is what :func:`~osprey.interfaces.web_terminal.routes.panels._browser_panel_url`
    hands the browser: ``/panel/<id>`` for a built-in, a sidecar or a URL-backed
    custom panel, and a discovered bundle's own ``/panel-static/<id>/`` URL. The
    upstream is the backend the proxy forwards ``/panel/<id>`` to, ``None`` for
    a bundle served from disk.
    """
    for cp in getattr(request.app.state, "custom_panels", []):
        if cp.get("id") == member and cp.get("discovered"):
            return cp["url"], None
    return f"/panel/{member}", _resolve_panel_url(request, member)


def _not_popped_out(request: Request, key: str) -> str:
    """``key`` once it is not in a pop-out window: a popped panel is focused, never moved."""
    if key in _popouts(request):
        raise OpRefused("popped_out", status=409)
    return key


def _drop_source(request: Request, source: dict) -> tuple[dict, str, list[str]]:
    """A drop source: ``(source, history subject, panels it adds to the rail)``."""
    if len(source) != 1:
        raise OpRefused("invalid_source")
    ((kind, value),) = source.items()
    if kind == "new":
        if not isinstance(value, dict) or set(value) != {"panel"}:
            raise OpRefused("invalid_source")
        member = _not_popped_out(request, _known_member(request, value["panel"]))
        return {"new": {"panel": member}}, member, [member]
    if kind == "tab":
        if value == _TERMINAL_PANEL_ID:
            raise OpRefused("terminal_not_agent_movable")
        if isinstance(value, str):
            _not_popped_out(request, value)
        layout = getattr(request.app.state, "dock_layout", None)
        if not isinstance(value, str) or value not in layout_keys(layout):
            raise OpRefused("unknown_key", f"no tab {value!r} in the current layout")
        return {"tab": value}, value, []
    if kind == "tile":
        return {"tile": _current_tile(request, value, source=True)}, value, []
    raise OpRefused("invalid_source")


def _drop_target(request: Request, target: dict) -> dict:
    """A drop target, its tile ids checked against the current layout."""
    keys = set(target)
    if keys == {"outer"} and target["outer"] in _SIDES:
        return dict(target)
    if keys == {"border"}:
        border = target["border"]
        if (
            isinstance(border, dict)
            and set(border) == {"tile", "side"}
            and border["side"] in _SIDES
        ):
            _current_tile(request, border["tile"])
            return {"border": dict(border)}
        raise OpRefused("invalid_target")
    if keys == {"tile", "edge"} and target["edge"] in _SIDES:
        _current_tile(request, target["tile"])
        return dict(target)
    if keys == {"tile", "zone"} and target["zone"] == "centre":
        _current_tile(request, target["tile"], zone=True)
        return dict(target)
    if keys == {"tile", "zone", "index"} and target["zone"] == "tab":
        index = target["index"]
        if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
            _current_tile(request, target["tile"], zone=True)
            return dict(target)
    raise OpRefused("invalid_target")


def _set_layout(members: list[str], layout: dict | None) -> dict | None:
    """A collection-form layout over ``members``, refused by its named reason.

    The terminal is checked first: a layout naming it is ``terminal_in_layout``
    whatever else is wrong with it.
    """
    if layout is None:
        return None
    if _names_terminal(layout):
        raise OpRefused("terminal_in_layout")
    refusal = layout_refusal(layout, form="collection", members=len(members))
    if refusal is not None:
        reason, where = refusal
        raise OpRefused(reason, f"{where}: {reason}: {REASON_TEXT[reason]}")
    return normalize_layout(layout, form="collection")


def _set_members(request: Request, members: list[str]) -> list[str]:
    """The members of a set or preset: known service panels, each named once, none popped out."""
    if _TERMINAL_PANEL_ID in members:
        raise OpRefused("terminal_in_layout")
    if len(set(members)) != len(members):
        raise OpRefused("duplicate_member")
    known = _known_panel_ids(request)
    unknown = [member for member in members if member not in known]
    if unknown:
        raise OpRefused("unknown_member", f"unknown panel ids: {unknown}")
    for member in members:
        _not_popped_out(request, member)
    return list(members)


def _member_index_layout(node: Mapping, index: dict[str, int]) -> dict:
    """A panel-id preset layout rewritten with member indexes in its tabs."""
    if "split" in node:
        return {**node, "items": [_member_index_layout(item, index) for item in node["items"]]}
    out = {**node, "tabs": [index[key] for key in node["tabs"]]}
    if "showing" in out:
        out["showing"] = index[out["showing"]]
    return out


def resolve_preset(request: Request, name: str) -> PresetArrangement:
    """A configured preset by name, resolved against the panels known now.

    The one place a preset's configuration (``app.state.panel_presets``, the
    entries ``GET /api/panels`` serves) is read for an arrangement. An entry's
    ``layout`` names panel ids; a panel unknown now is pruned from it
    (:func:`~osprey.interfaces.web_terminal.presets.prune_preset_layout`), so a
    stale or mistyped member skips rather than breaks the layout, and the
    members are what is left, in reading order.

    Raises:
        OpRefused: ``unknown_arrangement`` for a name no preset carries,
            ``terminal_in_layout`` for a layout naming the terminal,
            ``invalid_layout`` for an entry without a valid layout,
            ``unknown_member`` for a preset left with no member.
    """
    presets: list[dict] = list(getattr(request.app.state, "panel_presets", []))
    match = next((p for p in presets if p.get("name") == name), None)
    if match is None:
        available = [p.get("name") for p in presets]
        raise OpRefused("unknown_arrangement", f"unknown preset {name!r}; presets: {available}")
    layout = match.get("layout")
    if _names_terminal(layout):
        raise OpRefused("terminal_in_layout")
    if preset_refusal(layout) is not None:
        raise OpRefused("invalid_layout", f"preset {name!r} has no valid layout")
    known = _known_panel_ids(request)
    pruned = prune_preset_layout(layout, lambda key: key in known)  # type: ignore[arg-type]
    if pruned is None:
        raise OpRefused("unknown_member", f"preset {name!r} has no known members")
    members = layout_keys(pruned)
    index = {pid: n for n, pid in enumerate(members)}
    return PresetArrangement(members, _member_index_layout(pruned, index))


def _refusal_response(request: Request, refused: OpRefused) -> JSONResponse:
    return JSONResponse(
        status_code=refused.status,
        content={
            "status": "refused",
            "reason": refused.reason,
            "detail": refused.detail,
            "rev": getattr(request.app.state, "dock_layout_rev", 0),
        },
    )


async def _relay_op(
    request: Request, op: str, args: dict, request_id: str, base_rev: int | None
) -> dict:
    """Send the op to the current dock client and wait for its outcome.

    The frame goes to that client's event streams only
    (:meth:`~osprey.interfaces.web_terminal.file_watcher.FileEventBroadcaster.send_to`),
    so no other page learns the ``request_id`` it would need to answer
    ``/api/panel-op-result`` in the dock client's place.

    Returns:
        ``{"layout": ..., "popouts": ..., "rev": ...}`` as recorded by
        ``/api/panel-op-result``.

    Raises:
        OpRefused: ``stale_base`` at once when ``base_rev`` is behind the
            current rev; ``no_dock_client`` when no report is current (nothing
            was sent) or the client's stream closes first;
            ``no_dock_answer`` when the client does not answer within
            :data:`OP_TIMEOUT_S` — it may still apply the op, so the request id
            is kept for its late answer; the client's own refusal otherwise.
    """
    state = request.app.state
    if base_rev is not None and base_rev < getattr(state, "dock_layout_rev", 0):
        raise OpRefused("stale_base", status=409)
    target = getattr(state, "dock_client_id", None)
    if target is None:
        raise OpRefused("no_dock_client", status=409)
    pending = _pending_ops(state)
    if request_id in pending or request_id in _unanswered_ops(state):
        raise OpRefused("duplicate_request", status=409)
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    pending[request_id] = _PendingOp(target, future)
    frame: dict = {
        "type": "panel_op",
        "request_id": request_id,
        "target_client": target,
        "op": op,
        "args": args,
        "source": "agent",
    }
    if base_rev is not None:
        frame["base_rev"] = base_rev
    try:
        state.broadcaster.send_to(target, frame, reliable=True)
        return await asyncio.wait_for(future, timeout=OP_TIMEOUT_S)
    except TimeoutError:
        _remember_unanswered(state, request_id, target)
        raise OpRefused("no_dock_answer", status=409) from None
    finally:
        pending.pop(request_id, None)


def _add_to_rail(request: Request, panels: list[str], *, prune: bool) -> None:
    """Put ``panels`` on the launcher rail; with ``prune``, the rail keeps only them.

    Every client learns each change from a ``panel_visibility`` frame, in the
    order clients apply them: removals first, then additions.
    """
    state = request.app.state
    before: list[str] = list(getattr(state, "visible_panels", []))
    after = [pid for pid in before if pid in panels] if prune else list(before)
    after += [pid for pid in panels if pid not in after]
    state.visible_panels = after
    for pid in before:
        if pid not in after:
            state.broadcaster.broadcast(
                {"type": "panel_visibility", "panel": pid, "visible": False, "source": "agent"}
            )
    for pid in after:
        if pid not in before:
            state.broadcaster.broadcast(
                {"type": "panel_visibility", "panel": pid, "visible": True, "source": "agent"}
            )
    active = getattr(state, "active_panel", None)
    if prune and active is not None and active not in after:
        state.active_panel = after[0] if after else None


async def _run_op(
    request: Request, op: str, request_id: str, base_rev: int | None, prepare
) -> JSONResponse | dict:
    """Check, relay and record one agent op; the shared body of every write route."""
    try:
        prepared: _Prepared = prepare()
        outcome = await _relay_op(request, op, prepared.args, request_id, base_rev)
    except OpRefused as refused:
        return _refusal_response(request, refused)
    shown = set(layout_keys(outcome["layout"]))
    rail = [pid for pid in prepared.rail_add if pid in shown]
    if rail or prepared.rail_prune:
        _add_to_rail(request, rail, prune=prepared.rail_prune)
    if op == "open" and prepared.subject in shown:
        request.app.state.active_panel = prepared.subject
    _mirror_agent_panel_activity(request, OP_ACTIVITY_TOOLS[op], prepared.subject)
    return {
        "status": "ok",
        "layout": outcome["layout"],
        "popouts": outcome["popouts"],
        "rev": outcome["rev"],
    }


class _OpRequest(BaseModel):
    request_id: str = _REQUEST_ID


class DropOpRequest(_OpRequest):
    base_rev: int = Field(ge=0)
    source: dict
    target: dict


class CloseTabOpRequest(_OpRequest):
    key: str = Field(min_length=1)


class CloseTileOpRequest(_OpRequest):
    base_rev: int = Field(ge=0)
    tile: str = Field(min_length=1)


class OpenOpRequest(_OpRequest):
    member: str = Field(min_length=1)
    url: str | None = None


class OpenSetOpRequest(_OpRequest):
    model_config = ConfigDict(populate_by_name=True)

    members: list[str] = Field(min_length=1)
    as_: Literal["tabs", "side", "layout"] = Field(alias="as")
    layout: dict | None = None


class OpenArrangementOpRequest(_OpRequest):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(min_length=1)
    as_: Literal["tabs", "side", "layout"] | None = Field(default=None, alias="as")


@router.post("/api/panel-op/drop")
async def op_drop(body: DropOpRequest, request: Request):
    """The ``drop {source, target}`` op: move a tab or tile, or open a panel at a target.

    A ``new`` source is ``{panel}``; the dock mints its key. Tile ids name the
    layout of ``base_rev``. The terminal tab, and a tile holding it, is never a
    source; a target may be the terminal's tile. A panel in a pop-out window is
    never a source (``popped_out``).

    Returns:
        ``{"status": "ok", "layout", "popouts", "rev"}``, or a ``{"status":
        "refused", "reason", "detail", "rev"}`` body with 422 (argument) or 409
        (state).
    """

    def prepare() -> _Prepared:
        source, subject, rail = _drop_source(request, body.source)
        return _Prepared(
            {"source": source, "target": _drop_target(request, body.target)}, subject, rail
        )

    return await _run_op(request, "drop", body.request_id, body.base_rev, prepare)


@router.post("/api/panel-op/close-tab")
async def op_close_tab(body: CloseTabOpRequest, request: Request):
    """The ``close_tab {key}`` op; the terminal tab and a pop-out are never agent close targets."""

    def prepare() -> _Prepared:
        if body.key == _TERMINAL_PANEL_ID:
            raise OpRefused("terminal_not_agent_movable")
        if body.key not in _known_panel_ids(request):
            raise OpRefused("unknown_key", f"unknown panel {body.key!r}")
        _not_popped_out(request, body.key)
        return _Prepared({"key": body.key}, body.key)

    return await _run_op(request, "close_tab", body.request_id, None, prepare)


@router.post("/api/panel-op/close-tile")
async def op_close_tile(body: CloseTileOpRequest, request: Request):
    """The ``close_tile {tile}`` op; a tile holding the terminal is never closed by an agent."""

    def prepare() -> _Prepared:
        tile = _current_tile(request, body.tile, source=True)
        return _Prepared({"tile": tile}, tile)

    return await _run_op(request, "close_tile", body.request_id, body.base_rev, prepare)


@router.post("/api/panel-op/open")
async def op_open(body: OpenOpRequest, request: Request):
    """The ``open {member, url?}`` op; where the panel lands is the dock's open rule.

    ``url`` is read the way a browser's URL parser reads it
    (:func:`~osprey.interfaces.panel_urls.normalise_navigation_url`, the
    navigation guard) and must then stay with the panel
    (:func:`~osprey.interfaces.panel_urls.member_navigation_url`): a path
    under the panel's own mount, or an http(s) URL on the panel's own server,
    which is rewritten into the mount. Anything else is refused
    ``invalid_url`` and no frame is relayed. Before either, a url for an
    agent-hop panel (CANOPY), under one's mount or on one's process is refused
    ``agent_hop_panel_url`` (:func:`~osprey.interfaces.web_terminal.routes.panels.agent_hop_refusal`):
    the agent's ``open_panel`` judges it too, but only for a friendlier error,
    and a caller posting here directly meets the same rule. ``open_set``,
    ``open_arrangement`` and ``drop`` name panels by id and take no url. The admitted path is then
    prefixed into the user's mount. A panel that is
    not on the rail joins it once the dock shows it. A panel in a pop-out
    window is brought forward in its window and the layout comes back unchanged.
    """

    def prepare() -> _Prepared:
        member = _known_member(request, body.member)
        args: dict = {"member": member}
        if body.url is not None:
            url = normalise_navigation_url(body.url)
            hop_refusal = agent_hop_refusal(request, member, url)
            if hop_refusal is not None:
                raise OpRefused("agent_hop_panel_url", hop_refusal)
            refusal = navigation_url_refusal(url)
            if refusal is not None:
                raise OpRefused("invalid_url", refusal)
            mount, upstream = _member_mount_and_upstream(request, member)
            admitted, refusal = member_navigation_url(url, mount=mount, upstream=upstream)
            if admitted is None:
                raise OpRefused("invalid_url", refusal)
            args["url"] = _prefix_path(admitted)
        return _Prepared(args, member, [member])

    return await _run_op(request, "open", body.request_id, None, prepare)


@router.post("/api/panel-op/open-set")
async def op_open_set(body: OpenSetOpRequest, request: Request):
    """The ``open_set {members, as, layout?}`` op; ``layout`` keys are member indexes."""

    def prepare() -> _Prepared:
        members = _set_members(request, body.members)
        if (body.as_ == "layout") != (body.layout is not None):
            raise OpRefused("invalid_layout")
        args: dict = {"members": members, "as": body.as_}
        layout = _set_layout(members, body.layout)
        if layout is not None:
            args["layout"] = layout
        return _Prepared(args, members[0], members)

    return await _run_op(request, "open_set", body.request_id, None, prepare)


@router.post("/api/panel-op/open-arrangement")
async def op_open_arrangement(body: OpenArrangementOpRequest, request: Request):
    """The ``open_arrangement {name, as?}`` op for a configured preset.

    The members and layout are resolved here (:func:`resolve_preset`), so the
    dock client needs no preset table. A preset opens as its layout says unless
    ``as`` asks for its panels as tabs or side by side instead. Presets keep
    their prune-to-members rail behaviour: once the dock shows the arrangement,
    the rail holds its members.
    """

    def prepare() -> _Prepared:
        preset = resolve_preset(request, body.name)
        members = _set_members(request, preset.members)
        args: dict = {"name": body.name, "members": members}
        args["layout"] = _set_layout(members, preset.layout)
        if body.as_ is not None:
            args["as"] = body.as_
        return _Prepared(args, body.name, members, rail_prune=True)

    return await _run_op(request, "open_arrangement", body.request_id, None, prepare)


class PanelOpResultRequest(BaseModel):
    """The dock client's answer to one ``panel_op`` frame: a layout or a refusal.

    ``detail`` is browser-supplied text; it reaches the agent as the refusal's
    wording, bounded to 500 characters, and is never interpreted.
    """

    request_id: str = _REQUEST_ID
    client_id: str = Field(min_length=1, max_length=128)
    layout: dict | None = None
    popouts: list[str] = Field(default_factory=list)
    refusal: str | None = None
    detail: str | None = Field(default=None, max_length=500)


def _settle(pending: _PendingOp | None, outcome: dict | OpRefused) -> None:
    """Hand an answer to the write still waiting on it; a late answer has none."""
    if pending is None or pending.future.done():
        return
    if isinstance(outcome, OpRefused):
        pending.future.set_exception(outcome)
    else:
        pending.future.set_result(outcome)


@router.post("/api/panel-op-result")
async def panel_op_result(body: PanelOpResultRequest, request: Request):
    """Record a dock client's outcome for a pending op (browser-session tier).

    Only a pending ``request_id``, answered by the client the op was addressed
    to, is accepted — or, once, the late answer to a request whose write gave
    up waiting (``no_dock_answer``) and that the server still remembers
    (:data:`UNANSWERED_CAP`). A layout is recorded as that client's current
    report (:func:`_record_dock_report`), which mints the ``rev`` this response
    and a waiting write return; the client takes it as its ack baseline. A late
    layout passes the same checks and is recorded the same way, as a layout
    report would be: the write has already returned, so nothing else follows
    from it. A refusal is passed to a waiting write unchanged and records
    nothing.

    Returns:
        ``{"status": "ok", "layout": <dict|None>, "popouts": [...], "rev": <int>}``

    Raises:
        HTTPException: 404 for a request id that is neither pending nor awaiting
            a late answer, 403 for a client the op was not addressed to, 422 for
            an answer that is not exactly one of ``layout`` and ``refusal`` or
            whose layout is refused.
    """
    state = request.app.state
    pending = _pending_ops(state).get(body.request_id)
    target = pending.target if pending is not None else _unanswered_ops(state).get(body.request_id)
    if target is None:
        raise HTTPException(status_code=404, detail="No pending op with this request_id")
    if body.client_id != target:
        raise HTTPException(status_code=403, detail="The op was addressed to another client")
    if pending is None:
        del _unanswered_ops(state)[body.request_id]
    if (body.layout is None) == (body.refusal is None):
        raise HTTPException(status_code=422, detail="Answer with exactly one of layout or refusal")
    if body.refusal is not None:
        if not _REFUSAL_REASON_RE.match(body.refusal):
            raise HTTPException(status_code=422, detail="A refusal is a snake_case reason")
        _settle(pending, OpRefused(body.refusal, body.detail, status=409))
        return {
            "status": "ok",
            "layout": None,
            "popouts": _popouts(request),
            "rev": getattr(state, "dock_layout_rev", 0),
        }
    try:
        layout = _checked_dock_layout(request, body.layout)
        popouts = _checked_popouts(request, body.popouts, layout)
    except HTTPException as exc:
        _settle(pending, OpRefused("invalid_result", str(exc.detail), status=409))
        raise
    rev, _ = _record_dock_report(state, body.client_id, layout, True, popouts)
    _settle(pending, {"layout": layout, "popouts": popouts, "rev": rev})
    return {"status": "ok", "layout": layout, "popouts": popouts, "rev": rev}


def dock_stream_closed(state, client_id: str) -> None:
    """React to a page's event stream closing.

    When the page was the current dock client and has no other stream open, it
    stops being current: ops waiting on it are refused ``no_dock_client`` at
    once, and every remaining client is asked to report (a reliable
    ``panel_report_request`` frame) so a surviving dock becomes current.
    """
    if state.broadcaster.has_client(client_id):
        return
    if getattr(state, "dock_client_id", None) != client_id:
        return
    state.dock_client_id = None
    for op in list(_pending_ops(state).values()):
        if op.target == client_id and not op.future.done():
            op.future.set_exception(OpRefused("no_dock_client", status=409))
    state.broadcaster.broadcast({"type": "panel_report_request"}, reliable=True)
