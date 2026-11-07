"""The agent dock-op transport, server half (``routes/panel_ops.py``).

A write route under ``/api/panel-op/`` checks what the server can check, relays a
``panel_op`` frame addressed to the current dock client on a channel that never
drops frames, and waits for that client's ``/api/panel-op-result``. The result
route records the layout as the client's report and mints the ``rev`` the write
returns. Every refusal is named and changes nothing, except ``no_dock_answer``:
the client did not answer in time and may still apply the op, so its outcome
is unknown, and an answer that comes later is recorded as the client's report.

The dock client is played by the test: it reads the relayed frame from a real
:class:`FileEventBroadcaster` subscription and answers over the same app.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal import file_watcher
from osprey.interfaces.web_terminal.dock_layout import layout_keys, normalize_layout
from osprey.interfaces.web_terminal.file_watcher import FileEventBroadcaster
from osprey.interfaces.web_terminal.presets import ONE_PANEL_PER_TILE_TEXT
from osprey.interfaces.web_terminal.routes import files, panel_ops
from osprey.interfaces.web_terminal.routes.panel_ops import dock_stream_closed
from osprey.interfaces.web_terminal.routes.panel_ops import router as ops_router
from osprey.interfaces.web_terminal.routes.panels import router

from .conftest import bare_route_app

CLIENT = "page-1"

#: The current layout: ARIEL left of the terminal.
LAYOUT = normalize_layout(
    {
        "split": "row",
        "items": [
            {"tile": "t1", "tabs": ["ariel"], "showing": "ariel", "size": 0.6},
            {"tile": "t2", "tabs": ["terminal"], "showing": "terminal", "size": 0.4},
        ],
    }
)

#: The layout a dock answers with after opening LATTICE: it takes ARIEL's tile,
#: and ARIEL leaves the dock (it keeps its rail entry).
AFTER = {
    "split": "row",
    "items": [
        {"tile": "t1", "tabs": ["lattice"], "showing": "lattice", "size": 0.6},
        {"tile": "t2", "tabs": ["terminal"], "showing": "terminal", "size": 0.4},
    ],
}

#: The layout a dock answers with after closing ARIEL's tile: the terminal alone.
TERMINAL_ONLY = {"tile": "t2", "tabs": ["terminal"], "showing": "terminal"}

#: A tree the dock grammar allows but OSPREY's dock never shows: two panels in one tile.
CROWDED = {
    "split": "row",
    "items": [
        {"tile": "t1", "tabs": ["ariel", "lattice"], "showing": "lattice", "size": 0.6},
        {"tile": "t2", "tabs": ["terminal"], "showing": "terminal", "size": 0.4},
    ],
}


def _app(**state) -> FastAPI:
    defaults = {
        "enabled_panels": {"ariel", "lattice", "artifacts"},
        "custom_panels": [],
        "visible_panels": ["ariel"],
        "panel_presets": [],
        "broadcaster": FileEventBroadcaster(),
        "dock_layout": LAYOUT,
        "dock_layout_rev": 3,
        "dock_client_id": CLIENT,
        "layout_dock": True,
    }
    return bare_route_app(router, ops_router, **{**defaults, **state})


def _layout_answer(layout: dict = AFTER) -> Callable[[dict], dict]:
    return lambda frame: {"request_id": frame["request_id"], "client_id": CLIENT, "layout": layout}


def _refusal_answer(reason: str) -> Callable[[dict], dict]:
    return lambda frame: {"request_id": frame["request_id"], "client_id": CLIENT, "refusal": reason}


async def _write(
    app: FastAPI, path: str, body: dict, answer: Callable[[dict], dict] | None = None
) -> tuple[httpx.Response, dict | None, httpx.Response | None]:
    """POST a write; when ``answer`` is given, play the dock client answering its frame.

    Returns the write's response, the relayed frame (or ``None``) and the
    result route's response (or ``None``).
    """
    page = app.state.broadcaster.subscribe(CLIENT)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://wt") as client:
        task = asyncio.create_task(client.post(path, json={"request_id": "r1", **body}))
        frame = result = None
        if answer is not None:
            frame = await asyncio.wait_for(page.get(), timeout=2)
            result = await client.post("/api/panel-op-result", json=answer(frame))
        resp = await task
    if answer is None and not page.empty():
        frame = page.get_nowait()
    app.state.broadcaster.unsubscribe(page)
    return resp, frame, result


def _activity(app: FastAPI) -> list[tuple[str, str]]:
    return [(row["tool"], row["target"]["panel"]) for row in app.state.agent_activity_buffer]


# ---- open ---- #


async def test_the_op_frame_reaches_only_the_dock_clients_stream():
    """Another page on the session never sees an op's ``request_id``."""
    app = _app()
    other = app.state.broadcaster.subscribe("page-2")
    resp, frame, _ = await _write(
        app, "/api/panel-op/open", {"member": "lattice"}, _layout_answer()
    )
    assert resp.status_code == 200
    assert frame is not None and frame["type"] == "panel_op"
    seen = [other.get_nowait() for _ in range(other.qsize())]
    assert [f["type"] for f in seen] == ["panel_visibility"]
    assert not any("request_id" in f for f in seen)


async def test_open_relays_a_frame_to_the_current_client_and_returns_the_new_rev():
    app = _app()
    resp, frame, result = await _write(
        app, "/api/panel-op/open", {"member": "lattice"}, _layout_answer()
    )

    assert frame == {
        "type": "panel_op",
        "request_id": "r1",
        "target_client": CLIENT,
        "op": "open",
        "args": {"member": "lattice"},
        "source": "agent",
    }
    assert result.status_code == 200
    assert result.json()["rev"] == 4
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "layout": normalize_layout(AFTER),
        "popouts": [],
        "rev": 4,
    }


async def test_the_result_is_recorded_as_the_clients_current_report():
    app = _app(dock_client_id=CLIENT)
    await _write(app, "/api/panel-op/open", {"member": "lattice"}, _layout_answer())

    state = app.state
    assert state.dock_layout == normalize_layout(AFTER)
    assert state.dock_layout_rev == 4
    assert state.dock_client_id == CLIENT
    assert [key for key in layout_keys(state.dock_layout) if key != "terminal"] == ["lattice"]
    assert state.layout_dock is True
    assert isinstance(state.layout_ts, float)
    with TestClient(app) as client:
        body = client.get("/api/panels").json()
    assert (body["layout"], body["rev"]) == (normalize_layout(AFTER), 4)


async def test_open_records_one_open_panel_row_and_puts_the_panel_on_the_rail():
    """The panel the open replaced keeps its rail entry: it is one click away."""
    app = _app()
    page = app.state.broadcaster.subscribe("other-page")
    await _write(app, "/api/panel-op/open", {"member": "lattice"}, _layout_answer())

    assert _activity(app) == [("open_panel", "lattice")]
    assert app.state.visible_panels == ["ariel", "lattice"]
    assert app.state.active_panel == "lattice"
    frames = [page.get_nowait() for _ in range(page.qsize())]
    assert frames[-1] == {
        "type": "panel_visibility",
        "panel": "lattice",
        "visible": True,
        "source": "agent",
    }


async def test_a_member_the_dock_did_not_show_does_not_join_the_rail():
    app = _app()
    await _write(app, "/api/panel-op/open", {"member": "lattice"}, _layout_answer(LAYOUT))

    assert app.state.visible_panels == ["ariel"]
    assert getattr(app.state, "active_panel", None) is None


async def test_open_with_a_path_url_carries_it_on_the_frame():
    app = _app()
    _, frame, _ = await _write(
        app, "/api/panel-op/open", {"member": "ariel", "url": "/panel/ariel/#x"}, _layout_answer()
    )
    assert frame["args"] == {"member": "ariel", "url": "/panel/ariel/#x"}


# ---- pop-outs ---- #


def _popped_answer(layout: dict, popouts: list[str]) -> Callable[[dict], dict]:
    return lambda frame: {
        "request_id": frame["request_id"],
        "client_id": CLIENT,
        "layout": layout,
        "popouts": popouts,
    }


async def test_a_result_may_name_pop_outs_and_the_server_holds_them():
    app = _app()
    resp, _, _ = await _write(
        app, "/api/panel-op/open", {"member": "lattice"}, _popped_answer(LAYOUT, ["lattice"])
    )
    assert resp.status_code == 200
    assert resp.json()["popouts"] == ["lattice"]
    assert resp.json()["rev"] == 4
    assert app.state.dock_popouts == ["lattice"]
    with TestClient(app) as client:
        assert client.get("/api/panels").json()["popouts"] == ["lattice"]


async def test_a_result_with_a_null_layout_and_the_agent_detached_is_recorded():
    """The Agent pane alone left for its window: the dock is empty, and says so."""
    app = _app(dock_layout=TERMINAL_ONLY)
    resp, _, result = await _write(
        app, "/api/panel-op/open", {"member": "lattice"}, _popped_answer(None, ["terminal"])
    )
    assert result.status_code == 200
    assert result.json()["layout"] is None
    assert result.json()["popouts"] == ["terminal"]
    assert resp.status_code == 200
    assert app.state.dock_layout is None
    assert app.state.dock_popouts == ["terminal"]
    assert app.state.layout_dock is True


async def test_a_result_with_a_null_layout_and_no_pop_outs_is_an_empty_dock():
    """An op that emptied the dock settles at once, and empty is what the server holds."""
    app = _app(dock_layout=TERMINAL_ONLY)
    resp, _, result = await _write(
        app, "/api/panel-op/open", {"member": "lattice"}, _popped_answer(None, [])
    )
    assert result.status_code == 200
    assert result.json()["layout"] is None
    assert result.json()["popouts"] == []
    assert resp.status_code == 200
    assert app.state.dock_layout is None
    assert app.state.dock_popouts == []
    assert app.state.layout_dock is True


async def test_a_result_with_neither_a_layout_key_nor_a_refusal_is_refused(monkeypatch):
    monkeypatch.setattr(panel_ops, "OP_TIMEOUT_S", 0.05)
    app = _app()
    _, _, result = await _write(
        app,
        "/api/panel-op/open",
        {"member": "lattice"},
        lambda frame: {"request_id": frame["request_id"], "client_id": CLIENT},
    )
    assert result.status_code == 422


async def test_a_result_with_both_a_null_layout_and_a_refusal_is_refused(monkeypatch):
    monkeypatch.setattr(panel_ops, "OP_TIMEOUT_S", 0.05)
    app = _app()
    _, _, result = await _write(
        app,
        "/api/panel-op/open",
        {"member": "lattice"},
        lambda frame: {
            "request_id": frame["request_id"],
            "client_id": CLIENT,
            "layout": None,
            "refusal": "popped_out",
        },
    )
    assert result.status_code == 422


async def test_a_pop_out_that_is_also_in_the_layout_is_a_refused_result():
    app = _app()
    resp, _, result = await _write(
        app, "/api/panel-op/open", {"member": "lattice"}, _popped_answer(AFTER, ["lattice"])
    )
    assert result.status_code == 422
    assert resp.json()["reason"] == "invalid_result"
    assert getattr(app.state, "dock_popouts", []) == []


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("close", {"key": "lattice"}),
        ("drop", {"base_rev": 3, "source": {"tab": "lattice"}, "target": {"outer": "left"}}),
        (
            "drop",
            {"base_rev": 3, "source": {"new": {"panel": "lattice"}}, "target": {"outer": "left"}},
        ),
        ("open-set", {"members": ["lattice", "ariel"], "as": "side"}),
    ],
    ids=["close", "drop-tab", "drop-new", "open-set"],
)
async def test_a_popped_out_panel_is_never_moved_or_closed(path, body):
    app = _app(dock_popouts=["lattice"])
    resp, frame, _ = await _write(app, f"/api/panel-op/{path}", body)
    assert resp.status_code == 409
    assert resp.json()["reason"] == "popped_out"
    assert frame is None


async def test_an_open_of_a_popped_out_panel_is_relayed_and_may_leave_the_layout_alone():
    app = _app(dock_popouts=["lattice"])
    resp, frame, _ = await _write(
        app, "/api/panel-op/open", {"member": "lattice"}, _popped_answer(LAYOUT, ["lattice"])
    )
    assert frame["op"] == "open"
    assert resp.status_code == 200
    assert resp.json()["rev"] == 3
    assert resp.json()["popouts"] == ["lattice"]


async def test_open_of_an_unknown_panel_is_refused():
    app = _app()
    resp, frame, _ = await _write(app, "/api/panel-op/open", {"member": "nope"})
    assert resp.status_code == 422
    assert resp.json() == {
        "status": "refused",
        "reason": "unknown_member",
        "detail": "unknown panel 'nope'",
        "rev": 3,
    }
    assert frame is None


async def test_the_terminal_is_refused_even_when_a_custom_panel_squats_its_id():
    app = _app(custom_panels=[{"id": "terminal", "label": "T", "url": "http://10.0.0.5:1"}])
    resp, frame, _ = await _write(app, "/api/panel-op/open", {"member": "terminal"})
    assert resp.status_code == 422
    assert resp.json()["reason"] == "terminal_not_agent_movable"
    assert frame is None


# ---- state refusals ---- #


async def test_no_dock_client_is_refused_at_once():
    app = _app(dock_client_id=None)
    resp, frame, _ = await _write(app, "/api/panel-op/open", {"member": "lattice"})
    assert resp.status_code == 409
    assert resp.json()["reason"] == "no_dock_client"
    assert frame is None


async def test_a_dock_that_does_not_answer_in_time_leaves_the_outcome_unknown(monkeypatch):
    """The frame went out, so the dock may still apply the op: the write never says it did not."""
    monkeypatch.setattr(panel_ops, "OP_TIMEOUT_S", 0.05)
    app = _app()
    resp, frame, _ = await _write(app, "/api/panel-op/open", {"member": "lattice"})

    assert resp.status_code == 409
    assert resp.json()["reason"] == "no_dock_answer"
    assert resp.json()["rev"] == 3
    assert frame["type"] == "panel_op"
    assert app.state.panel_ops == {}
    assert app.state.dock_layout_rev == 3
    assert _activity(app) == []


def test_the_route_waits_five_seconds():
    assert panel_ops.OP_TIMEOUT_S == 5.0


async def test_a_clients_refusal_is_passed_on_and_changes_nothing():
    app = _app()
    resp, _, result = await _write(
        app, "/api/panel-op/open", {"member": "lattice"}, _refusal_answer("panel_unhealthy")
    )

    assert result.status_code == 200
    assert resp.status_code == 409
    assert resp.json()["reason"] == "panel_unhealthy"
    assert app.state.dock_layout == LAYOUT
    assert app.state.dock_layout_rev == 3
    assert app.state.visible_panels == ["ariel"]
    assert _activity(app) == []


async def test_a_pending_request_id_cannot_be_reused():
    app = _app()
    app.state.panel_ops = {
        "r1": panel_ops._PendingOp(CLIENT, asyncio.get_running_loop().create_future())
    }
    resp, frame, _ = await _write(app, "/api/panel-op/open", {"member": "lattice"})
    assert resp.status_code == 409
    assert resp.json()["reason"] == "duplicate_request"
    assert frame is None


# ---- a dock that answers after the write gave up ---- #


async def _late(
    app: FastAPI, body: dict, answer: Callable[[dict], dict], request_id: str = "r1"
) -> tuple[httpx.Response, httpx.Response]:
    """POST an ``open`` the dock client does not answer in time, then play its late answer.

    Returns the write's response and the late answer's.
    """
    page = app.state.broadcaster.subscribe(CLIENT)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://wt") as client:
        resp = await client.post("/api/panel-op/open", json={"request_id": request_id, **body})
        late = await client.post("/api/panel-op-result", json=answer(page.get_nowait()))
    app.state.broadcaster.unsubscribe(page)
    return resp, late


async def test_a_late_answer_is_recorded_as_the_clients_report(monkeypatch):
    """A dock that applies the op after the write gave up still answers, and the server keeps it.

    The write has already returned, so the answer changes what the panels read
    says (layout, rev, current client) and nothing else: no rail entry and no
    activity row for an op the agent was told may not have happened.
    """
    monkeypatch.setattr(panel_ops, "OP_TIMEOUT_S", 0.05)
    app = _app(dock_client_id=CLIENT)
    resp, late = await _late(app, {"member": "lattice"}, _layout_answer())

    assert resp.json()["reason"] == "no_dock_answer"
    assert late.status_code == 200
    assert late.json()["rev"] == 4
    assert late.json()["layout"] == normalize_layout(AFTER)
    assert app.state.dock_layout == normalize_layout(AFTER)
    assert app.state.dock_layout_rev == 4
    assert app.state.dock_client_id == CLIENT
    assert app.state.visible_panels == ["ariel"]
    assert _activity(app) == []


async def test_a_late_answer_is_checked_like_a_report(monkeypatch):
    monkeypatch.setattr(panel_ops, "OP_TIMEOUT_S", 0.05)
    app = _app()
    bad = {"tile": "t1", "tabs": ["nope"], "showing": "nope"}
    _, late = await _late(app, {"member": "lattice"}, _layout_answer(bad))

    assert late.status_code == 422
    assert app.state.dock_layout == LAYOUT
    assert app.state.dock_layout_rev == 3


async def test_a_late_refusal_records_nothing(monkeypatch):
    monkeypatch.setattr(panel_ops, "OP_TIMEOUT_S", 0.05)
    app = _app()
    _, late = await _late(app, {"member": "lattice"}, _refusal_answer("stale_base"))

    assert late.status_code == 200
    assert app.state.dock_layout == LAYOUT
    assert app.state.dock_layout_rev == 3


async def test_a_late_answer_from_another_client_is_refused_and_the_dock_can_still_answer(
    monkeypatch,
):
    monkeypatch.setattr(panel_ops, "OP_TIMEOUT_S", 0.05)
    app = _app()

    def other(frame: dict) -> dict:
        return {"request_id": frame["request_id"], "client_id": "page-2", "layout": AFTER}

    _, late = await _late(app, {"member": "lattice"}, other)
    assert late.status_code == 403
    assert app.state.dock_layout == LAYOUT

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://wt") as client:
        answer = {"request_id": "r1", "client_id": CLIENT, "layout": AFTER}
        assert (await client.post("/api/panel-op-result", json=answer)).status_code == 200
    assert app.state.dock_layout == normalize_layout(AFTER)


async def test_a_late_answer_counts_once(monkeypatch):
    monkeypatch.setattr(panel_ops, "OP_TIMEOUT_S", 0.05)
    app = _app()
    _, late = await _late(app, {"member": "lattice"}, _layout_answer())
    assert late.status_code == 200

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://wt") as client:
        again = {"request_id": "r1", "client_id": CLIENT, "layout": LAYOUT}
        assert (await client.post("/api/panel-op-result", json=again)).status_code == 404
    assert app.state.dock_layout == normalize_layout(AFTER)


async def test_an_unanswered_request_id_cannot_be_reused(monkeypatch):
    """A second write under its id could otherwise take the first op's late answer as its own."""
    monkeypatch.setattr(panel_ops, "OP_TIMEOUT_S", 0.05)
    app = _app()
    first, _, _ = await _write(app, "/api/panel-op/open", {"member": "lattice"})
    assert first.json()["reason"] == "no_dock_answer"

    again, frame, _ = await _write(app, "/api/panel-op/open", {"member": "lattice"})
    assert again.json()["reason"] == "duplicate_request"
    assert frame is None


async def test_only_the_newest_unanswered_requests_are_kept(monkeypatch):
    monkeypatch.setattr(panel_ops, "OP_TIMEOUT_S", 0.05)
    monkeypatch.setattr(panel_ops, "UNANSWERED_CAP", 1)
    app = _app()
    page = app.state.broadcaster.subscribe(CLIENT)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://wt") as client:
        for request_id in ("r1", "r2"):
            resp = await client.post(
                "/api/panel-op/open", json={"request_id": request_id, "member": "lattice"}
            )
            assert resp.json()["reason"] == "no_dock_answer"
        oldest = {"request_id": "r1", "client_id": CLIENT, "layout": AFTER}
        newest = {"request_id": "r2", "client_id": CLIENT, "layout": AFTER}
        assert (await client.post("/api/panel-op-result", json=oldest)).status_code == 404
        assert (await client.post("/api/panel-op-result", json=newest)).status_code == 200
    app.state.broadcaster.unsubscribe(page)


# ---- drop ---- #


async def test_drop_behind_the_current_rev_is_stale_at_once():
    app = _app()
    body = {"base_rev": 2, "source": {"tab": "ariel"}, "target": {"outer": "right"}}
    resp, frame, _ = await _write(app, "/api/panel-op/drop", body)
    assert resp.status_code == 409
    assert resp.json()["reason"] == "stale_base"
    assert resp.json()["rev"] == 3
    assert frame is None


async def test_drop_requires_base_rev():
    app = _app()
    resp, frame, _ = await _write(
        app, "/api/panel-op/drop", {"source": {"tab": "ariel"}, "target": {"outer": "right"}}
    )
    assert resp.status_code == 422
    assert frame is None


async def test_drop_relays_source_target_and_base_rev():
    app = _app()
    body = {
        "base_rev": 3,
        "source": {"new": {"panel": "lattice"}},
        "target": {"tile": "t1", "edge": "left"},
    }
    resp, frame, _ = await _write(app, "/api/panel-op/drop", body, _layout_answer())

    assert frame["op"] == "drop"
    assert frame["base_rev"] == 3
    assert frame["args"] == {
        "source": {"new": {"panel": "lattice"}},
        "target": {"tile": "t1", "edge": "left"},
    }
    assert resp.status_code == 200
    assert _activity(app) == [("drop_panel", "lattice")]


@pytest.mark.parametrize(
    "source",
    [{"tab": "terminal"}, {"tile": "t2"}, {"new": {"panel": "terminal"}}],
    ids=["terminal-tab", "terminal-tile", "new-terminal"],
)
async def test_the_terminal_is_never_a_drop_source(source):
    app = _app()
    body = {"base_rev": 3, "source": source, "target": {"outer": "left"}}
    resp, frame, _ = await _write(app, "/api/panel-op/drop", body)
    assert resp.status_code == 422
    assert resp.json()["reason"] == "terminal_not_agent_movable"
    assert frame is None


@pytest.mark.parametrize(
    "target",
    [
        {"tile": "t2", "edge": "bottom"},
        {"border": {"tile": "t1", "side": "right"}},
        {"outer": "top"},
    ],
    ids=["edge", "border", "outer"],
)
async def test_each_target_form_is_relayed(target):
    app = _app()
    body = {"base_rev": 3, "source": {"tab": "ariel"}, "target": target}
    resp, frame, _ = await _write(app, "/api/panel-op/drop", body, _layout_answer())
    assert resp.status_code == 200
    assert frame["args"]["target"] == target


@pytest.mark.parametrize(
    ("source", "target", "reason"),
    [
        ({"tab": "ariel", "tile": "t1"}, {"outer": "left"}, "invalid_source"),
        ({"new": "lattice"}, {"outer": "left"}, "invalid_source"),
        ({"new": {"ref": "x"}}, {"outer": "left"}, "invalid_source"),
        ({"slot": 1}, {"outer": "left"}, "invalid_source"),
        ({"tab": "lattice"}, {"outer": "left"}, "unknown_key"),
        ({"tile": "t9"}, {"outer": "left"}, "unknown_tile"),
        ({"tab": "ariel"}, {"outer": "middle"}, "invalid_target"),
        ({"tab": "ariel"}, {"tile": "t1", "zone": "tab"}, "invalid_target"),
        ({"tab": "ariel"}, {"tile": "t1", "zone": "tab", "index": -1}, "invalid_target"),
        ({"tab": "ariel"}, {"tile": "t1", "zone": "middle"}, "invalid_target"),
        ({"tab": "ariel"}, {"border": {"tile": "t1"}}, "invalid_target"),
        ({"tab": "ariel"}, {"tile": "t9", "edge": "left"}, "unknown_tile"),
        ({"tab": "ariel"}, {"border": {"tile": "t9", "side": "left"}}, "unknown_tile"),
    ],
)
async def test_bad_drops_are_refused_by_name(source, target, reason):
    app = _app()
    body = {"base_rev": 3, "source": source, "target": target}
    resp, frame, _ = await _write(app, "/api/panel-op/drop", body)
    assert resp.status_code == 422
    assert resp.json()["reason"] == reason
    assert frame is None


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ({"tab": "ariel"}, {"tile": "t1", "zone": "centre"}),
        ({"new": {"panel": "lattice"}}, {"tile": "t1", "zone": "tab", "index": 0}),
        ({"tab": "ariel"}, {"tile": "t2", "zone": "centre"}),
        ({"new": {"panel": "lattice"}}, {"tile": "t9", "zone": "tab", "index": 1}),
    ],
    ids=["centre", "header", "agent-pane", "unknown-tile"],
)
async def test_a_drop_into_a_tile_is_refused_one_panel_per_tile(source, target):
    """A tile shows one panel: a ``zone`` target, whatever tile it names, is no target."""
    app = _app()
    body = {"base_rev": 3, "source": source, "target": target}
    resp, frame, _ = await _write(app, "/api/panel-op/drop", body)
    assert resp.status_code == 422
    assert resp.json()["reason"] == "one_panel_per_tile"
    assert resp.json()["detail"].startswith(ONE_PANEL_PER_TILE_TEXT)
    assert frame is None


# ---- close ---- #


async def test_close_relays_the_key_without_a_base_rev():
    app = _app()
    resp, frame, _ = await _write(
        app, "/api/panel-op/close", {"key": "ariel"}, _layout_answer(TERMINAL_ONLY)
    )
    assert (frame["op"], frame["args"]) == ("close", {"key": "ariel"})
    assert "base_rev" not in frame
    assert resp.status_code == 200
    assert resp.json()["layout"] == normalize_layout(TERMINAL_ONLY)
    assert _activity(app) == [("close_panel", "ariel")]
    assert app.state.visible_panels == ["ariel"]


async def test_the_terminal_is_never_an_agent_close_target():
    app = _app()
    resp, frame, _ = await _write(app, "/api/panel-op/close", {"key": "terminal"})
    assert resp.status_code == 422
    assert resp.json()["reason"] == "terminal_not_agent_movable"
    assert frame is None


async def test_close_of_an_unknown_panel_is_refused():
    app = _app()
    resp, frame, _ = await _write(app, "/api/panel-op/close", {"key": "nope"})
    assert resp.status_code == 422
    assert resp.json()["reason"] == "unknown_key"
    assert frame is None


def test_close_is_the_one_close_route():
    """One close verb: the tile goes with the one panel it shows."""
    paths = {route.path for route in ops_router.routes}
    assert {path for path in paths if path.startswith("/api/panel-op/close")} == {
        "/api/panel-op/close"
    }


# ---- /api/panel-op-result ---- #


async def test_a_result_for_no_pending_request_is_refused():
    app = _app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://wt") as client:
        resp = await client.post(
            "/api/panel-op-result", json={"request_id": "r9", "client_id": CLIENT, "layout": AFTER}
        )
    assert resp.status_code == 404
    assert app.state.dock_layout == LAYOUT


async def test_a_result_from_another_client_is_refused(monkeypatch):
    monkeypatch.setattr(panel_ops, "OP_TIMEOUT_S", 0.5)
    app = _app()

    def other(frame: dict) -> dict:
        return {"request_id": frame["request_id"], "client_id": "page-2", "layout": AFTER}

    resp, _, result = await _write(app, "/api/panel-op/open", {"member": "lattice"}, other)
    assert result.status_code == 403
    assert resp.json()["reason"] == "no_dock_answer"
    assert app.state.dock_layout == LAYOUT


@pytest.mark.parametrize(
    "extra",
    [{}, {"layout": AFTER, "refusal": "stale_base"}, {"refusal": "Not A Reason"}],
    ids=["neither", "both", "bad-reason"],
)
async def test_a_result_must_be_one_layout_or_one_named_refusal(monkeypatch, extra):
    monkeypatch.setattr(panel_ops, "OP_TIMEOUT_S", 0.5)
    app = _app()

    def answer(frame: dict) -> dict:
        return {"request_id": frame["request_id"], "client_id": CLIENT, **extra}

    _, _, result = await _write(app, "/api/panel-op/open", {"member": "lattice"}, answer)
    assert result.status_code == 422


async def test_a_refused_result_layout_refuses_the_write():
    app = _app()
    bad = {"tile": "t1", "tabs": ["nope"], "showing": "nope"}
    resp, _, result = await _write(
        app, "/api/panel-op/open", {"member": "lattice"}, _layout_answer(bad)
    )
    assert result.status_code == 422
    assert resp.status_code == 409
    assert resp.json()["reason"] == "invalid_result"
    assert app.state.dock_layout == LAYOUT


async def test_a_result_with_two_panels_in_a_tile_refuses_the_write():
    """The dock's holder never answers with a crowded tile; the server refuses one anyway."""
    app = _app()
    resp, _, result = await _write(
        app, "/api/panel-op/open", {"member": "lattice"}, _layout_answer(CROWDED)
    )
    assert result.status_code == 422
    assert result.json()["detail"] == (
        f"layout.items[0].tabs: one_panel_per_tile: {ONE_PANEL_PER_TILE_TEXT}"
    )
    assert resp.status_code == 409
    assert resp.json()["reason"] == "invalid_result"
    assert "one_panel_per_tile" in resp.json()["detail"]
    assert (app.state.dock_layout, app.state.dock_layout_rev) == (LAYOUT, 3)
    assert app.state.visible_panels == ["ariel"]
    assert _activity(app) == []


async def test_an_unchanged_result_keeps_the_rev():
    app = _app()
    resp, _, _ = await _write(
        app, "/api/panel-op/open", {"member": "ariel"}, _layout_answer(LAYOUT)
    )
    assert resp.json()["rev"] == 3


# ---- the event stream ---- #


async def test_the_current_clients_stream_closing_clears_it_and_asks_for_reports():
    app = _app()
    other = app.state.broadcaster.subscribe("page-2")
    loop = asyncio.get_running_loop()
    waiting = panel_ops._PendingOp(CLIENT, loop.create_future())
    app.state.panel_ops = {"r1": waiting}

    dock_stream_closed(app.state, CLIENT)

    assert app.state.dock_client_id is None
    assert other.get_nowait() == {"type": "panel_report_request"}
    with pytest.raises(panel_ops.OpRefused) as refused:
        waiting.future.result()
    assert refused.value.reason == "no_dock_client"


async def test_a_second_stream_of_the_same_page_keeps_it_current():
    app = _app()
    app.state.broadcaster.subscribe(CLIENT)
    dock_stream_closed(app.state, CLIENT)
    assert app.state.dock_client_id == CLIENT


async def test_another_clients_stream_closing_changes_nothing():
    app = _app()
    watcher = app.state.broadcaster.subscribe(None)
    dock_stream_closed(app.state, "page-2")
    assert app.state.dock_client_id == CLIENT
    assert watcher.empty()


def test_the_event_stream_refuses_an_empty_or_overlong_client_id():
    app = bare_route_app(files.router, broadcaster=FileEventBroadcaster())
    with TestClient(app) as client:
        assert client.get("/api/files/events", params={"client_id": ""}).status_code == 400
        assert client.get("/api/files/events", params={"client_id": "x" * 129}).status_code == 400


async def test_op_frames_are_never_dropped_for_a_slow_stream():
    broadcaster = FileEventBroadcaster()
    slow = broadcaster.subscribe(CLIENT)
    for n in range(file_watcher.QUEUE_CAP + 5):
        broadcaster.broadcast({"type": "file", "n": n})
    assert slow.qsize() == file_watcher.QUEUE_CAP

    broadcaster.broadcast({"type": "panel_op"}, reliable=True)
    assert slow.qsize() == file_watcher.QUEUE_CAP + 1
    assert broadcaster.has_client(CLIENT)
    broadcaster.unsubscribe(slow)
    assert not broadcaster.has_client(CLIENT)
