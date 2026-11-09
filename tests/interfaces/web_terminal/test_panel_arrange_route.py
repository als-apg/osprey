"""Tests for the arrangement ops, ``open_set`` and ``open_arrangement`` (`routes/panel_ops.py`).

An arrangement is a dock op relayed to the current dock client like every other
agent write (see ``test_panel_op_route.py`` for the transport). These tests pin
what the server itself decides for the two arrangement ops: the members and
layout it relays, the terminal invariant, one panel per tile, and the rail
effect once the dock shows the result. A preset in ``app.state.panel_presets``
is what the app's loader resolves from ``web.presets``: ``{name, panels,
layout}``, the layout over panel ids it opens.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from osprey.interfaces.web_terminal.file_watcher import FileEventBroadcaster
from osprey.interfaces.web_terminal.presets import ONE_PANEL_PER_TILE_TEXT
from osprey.interfaces.web_terminal.routes.panel_ops import router as ops_router
from osprey.interfaces.web_terminal.routes.panels import router

from .conftest import bare_route_app

CLIENT = "page-1"


def _tile(tile: str, tabs: list[str], size: float) -> dict:
    return {"tile": tile, "tabs": tabs, "showing": tabs[0], "size": size}


TERMINAL_ONLY = {"tile": "t0", "tabs": ["terminal"], "showing": "terminal"}


def _side_by_side(*panels: str) -> dict:
    """The layout a dock answers with: ``panels`` as one tile each, left of the terminal."""
    share = 0.6 / len(panels)
    tiles = [_tile(f"s{n}", [pid], share) for n, pid in enumerate(panels)]
    return {"split": "row", "items": [*tiles, _tile("t0", ["terminal"], 0.4)]}


def _app(**state) -> FastAPI:
    defaults = {
        "enabled_panels": {"ariel", "lattice", "artifacts"},
        "custom_panels": [{"id": "grafana", "label": "GRAFANA", "url": "http://10.0.0.5:3000"}],
        "visible_panels": [],
        "panel_presets": [],
        "broadcaster": FileEventBroadcaster(),
        "dock_layout": TERMINAL_ONLY,
        "dock_layout_rev": 1,
        "dock_client_id": CLIENT,
        "layout_dock": True,
    }
    return bare_route_app(router, ops_router, **{**defaults, **state})


async def _arrange(
    app: FastAPI, path: str, body: dict, answer: dict | None = None
) -> tuple[httpx.Response, dict | None, list[dict]]:
    """POST an arrangement; with ``answer``, the dock answers its frame with that layout.

    Returns the response, the relayed op frame (or ``None``) and the rail frames
    a second page received.
    """
    page = app.state.broadcaster.subscribe(CLIENT)
    watcher = app.state.broadcaster.subscribe("page-2")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://wt") as client:
        task = asyncio.create_task(client.post(path, json={"request_id": "r1", **body}))
        frame = None
        if answer is not None:
            frame = await asyncio.wait_for(page.get(), timeout=2)
            await client.post(
                "/api/panel-op-result",
                json={"request_id": frame["request_id"], "client_id": CLIENT, "layout": answer},
            )
        resp = await task
    if answer is None and not page.empty():
        frame = page.get_nowait()
    after = [watcher.get_nowait() for _ in range(watcher.qsize())]
    return resp, frame, [f for f in after if f.get("type") != "panel_op"]


class TestOpenSet:
    async def test_relays_members_and_as(self):
        app = _app()
        resp, frame, _ = await _arrange(
            app,
            "/api/panel-op/open-set",
            {"members": ["lattice", "ariel"], "as": "side"},
            _side_by_side("lattice", "ariel"),
        )
        assert frame["op"] == "open_set"
        assert frame["args"] == {"members": ["lattice", "ariel"], "as": "side"}
        assert "base_rev" not in frame
        assert resp.status_code == 200
        assert resp.json()["rev"] == 2

    async def test_a_layout_set_carries_its_normalised_member_index_layout(self):
        app = _app()
        layout = {"split": "column", "items": [{"tabs": [0]}, {"tabs": [1], "size": 3}]}
        _, frame, _ = await _arrange(
            app,
            "/api/panel-op/open-set",
            {"members": ["lattice", "ariel"], "as": "layout", "layout": layout},
            _side_by_side("lattice", "ariel"),
        )
        assert frame["args"]["as"] == "layout"
        assert [item["size"] for item in frame["args"]["layout"]["items"]] == [0.25, 0.75]

    async def test_shown_members_join_the_rail_and_nothing_is_pruned(self):
        app = _app(visible_panels=["artifacts"])
        _, _, frames = await _arrange(
            app,
            "/api/panel-op/open-set",
            {"members": ["grafana", "ariel"], "as": "side"},
            _side_by_side("grafana", "ariel"),
        )
        assert app.state.visible_panels == ["artifacts", "grafana", "ariel"]
        assert [(f["panel"], f["visible"]) for f in frames] == [("grafana", True), ("ariel", True)]
        assert [row["tool"] for row in app.state.agent_activity_buffer] == ["open_panel_set"]

    @pytest.mark.parametrize(
        ("body", "reason"),
        [
            ({"members": ["terminal"], "as": "side"}, "terminal_in_layout"),
            ({"members": ["ariel", "nope"], "as": "side"}, "unknown_member"),
            ({"members": ["ariel", "ariel"], "as": "side"}, "duplicate_member"),
            ({"members": ["ariel"], "as": "layout"}, "invalid_layout"),
            ({"members": ["ariel"], "as": "side", "layout": {"tabs": [0]}}, "invalid_layout"),
            (
                {"members": ["ariel"], "as": "layout", "layout": {"tabs": ["terminal", 0]}},
                "terminal_in_layout",
            ),
            (
                {"members": ["ariel", "lattice"], "as": "layout", "layout": {"tabs": [0]}},
                "member_missing",
            ),
            ({"members": ["ariel"], "as": "layout", "layout": {"slot": 0}}, "not_layout"),
            (
                {"members": ["ariel", "lattice"], "as": "layout", "layout": {"tabs": [0, 1]}},
                "one_panel_per_tile",
            ),
        ],
        ids=[
            "terminal-member",
            "unknown",
            "duplicate",
            "layout-missing",
            "layout-without-as-layout",
            "terminal-in-layout",
            "member-unplaced",
            "old-shape",
            "two-panels-in-a-tile",
        ],
    )
    async def test_refusals_are_named_and_relay_nothing(self, body, reason):
        app = _app()
        resp, frame, _ = await _arrange(app, "/api/panel-op/open-set", body)
        assert resp.status_code == 422
        assert resp.json()["reason"] == reason
        assert frame is None
        assert app.state.visible_panels == []

    async def test_two_panels_in_one_tile_are_refused_with_where_they_are(self):
        """Every member is placed, and the layout keeps the grammar: only the policy refuses it."""
        app = _app()
        layout = {"split": "row", "items": [{"tabs": [0]}, {"tabs": [1, 2], "showing": 2}]}
        resp, frame, _ = await _arrange(
            app,
            "/api/panel-op/open-set",
            {"members": ["ariel", "lattice", "grafana"], "as": "layout", "layout": layout},
        )
        assert resp.status_code == 422
        assert resp.json()["reason"] == "one_panel_per_tile"
        assert resp.json()["detail"] == (
            f"layout.items[1].tabs: one_panel_per_tile: {ONE_PANEL_PER_TILE_TEXT}"
        )
        assert frame is None

    @pytest.mark.parametrize("as_", ["side", "tabs"], ids=["empty", "tabs-mode"])
    async def test_an_empty_set_or_a_tabs_mode_is_a_bad_request(self, as_):
        """``as`` is ``side`` or ``layout``: there is no mode that stacks panels in one tile."""
        app = _app()
        members = [] if as_ == "side" else ["ariel", "lattice"]
        resp, frame, _ = await _arrange(
            app, "/api/panel-op/open-set", {"members": members, "as": as_}
        )
        assert resp.status_code == 422
        assert "reason" not in resp.json()
        assert frame is None


def _preset(name: str, *panels: str, layout: dict | None = None) -> dict:
    """A preset as the app's loader resolves it: its name, panels and layout (one tile each)."""
    if layout is None:
        tiles = [{"tabs": [pid], "showing": pid} for pid in panels]
        layout = tiles[0] if len(tiles) == 1 else {"split": "row", "items": tiles}
    return {"name": name, "panels": list(panels), "layout": layout}


class TestOpenArrangement:
    async def test_relays_the_known_members_and_the_layout_over_them(self):
        """A panel the deployment does not serve now is pruned from the layout; the rest open as configured."""
        app = _app(panel_presets=[_preset("L1", "lattice", "nope", "ariel", "grafana")])
        resp, frame, _ = await _arrange(
            app,
            "/api/panel-op/open-arrangement",
            {"name": "L1"},
            _side_by_side("lattice", "ariel", "grafana"),
        )
        assert frame["op"] == "open_arrangement"
        assert frame["args"] == {
            "name": "L1",
            "members": ["lattice", "ariel", "grafana"],
            "layout": {
                "split": "row",
                "items": [
                    {"tabs": [0], "showing": 0, "size": pytest.approx(1 / 3)},
                    {"tabs": [1], "showing": 1, "size": pytest.approx(1 / 3)},
                    {"tabs": [2], "showing": 2, "size": pytest.approx(1 / 3)},
                ],
            },
        }
        assert resp.status_code == 200
        assert [(r["tool"], r["target"]["panel"]) for r in app.state.agent_activity_buffer] == [
            ("arrange_workspace", "L1")
        ]

    async def test_a_layout_keeps_its_showing_and_weights(self):
        """The members are the layout's panels in reading order; showing and sizes carry over by index."""
        layout = {
            "split": "column",
            "items": [
                {"tabs": ["lattice"], "showing": "lattice", "size": 3},
                {"tabs": ["grafana"]},
            ],
        }
        app = _app(panel_presets=[_preset("L1", "lattice", "grafana", layout=layout)])
        _, frame, _ = await _arrange(
            app,
            "/api/panel-op/open-arrangement",
            {"name": "L1"},
            _side_by_side("lattice", "grafana"),
        )
        assert frame["args"]["members"] == ["lattice", "grafana"]
        assert frame["args"]["layout"] == {
            "split": "column",
            "items": [
                {"tabs": [0], "showing": 0, "size": 0.75},
                {"tabs": [1], "showing": 1, "size": 0.25},
            ],
        }

    async def test_a_preset_with_two_panels_in_a_tile_is_refused(self):
        """``resolve_preset`` refuses it by name, whatever the startup loader let through."""
        layout = {
            "split": "column",
            "items": [
                {"tabs": ["ariel", "lattice"], "showing": "lattice", "size": 3},
                {"tabs": ["grafana"]},
            ],
        }
        app = _app(
            panel_presets=[_preset("L1", "ariel", "lattice", "grafana", layout=layout)],
            visible_panels=["artifacts"],
        )
        resp, frame, _ = await _arrange(app, "/api/panel-op/open-arrangement", {"name": "L1"})
        assert resp.status_code == 422
        assert resp.json()["reason"] == "one_panel_per_tile"
        assert resp.json()["detail"] == (
            f"layout.items[0].tabs: one_panel_per_tile: {ONE_PANEL_PER_TILE_TEXT}"
        )
        assert frame is None
        assert app.state.visible_panels == ["artifacts"]

    async def test_a_layout_naming_the_terminal_is_refused(self):
        app = _app(panel_presets=[_preset("L1", "terminal", "ariel")])
        resp, frame, _ = await _arrange(app, "/api/panel-op/open-arrangement", {"name": "L1"})
        assert resp.status_code == 422
        assert resp.json()["reason"] == "terminal_in_layout"
        assert frame is None

    async def test_an_entry_without_a_layout_is_refused(self):
        """An entry in the retired ``{name, panels}`` shape opens nothing."""
        app = _app(panel_presets=[{"name": "L1", "panels": ["ariel"]}])
        resp, frame, _ = await _arrange(app, "/api/panel-op/open-arrangement", {"name": "L1"})
        assert resp.status_code == 422
        assert resp.json()["reason"] == "invalid_layout"
        assert frame is None

    async def test_a_preset_left_with_no_known_member_is_refused(self):
        app = _app(panel_presets=[_preset("L1", "nope")])
        resp, frame, _ = await _arrange(app, "/api/panel-op/open-arrangement", {"name": "L1"})
        assert resp.status_code == 422
        assert resp.json()["reason"] == "unknown_member"
        assert frame is None

    async def test_as_is_relayed_when_given(self):
        app = _app(panel_presets=[_preset("L1", "ariel")])
        _, frame, _ = await _arrange(
            app,
            "/api/panel-op/open-arrangement",
            {"name": "L1", "as": "side"},
            _side_by_side("ariel"),
        )
        assert frame["args"]["as"] == "side"

    async def test_as_tabs_is_a_bad_request(self):
        """A preset opens by its layout or side by side, never stacked in one tile."""
        app = _app(panel_presets=[_preset("L1", "ariel", "lattice")])
        resp, frame, _ = await _arrange(
            app, "/api/panel-op/open-arrangement", {"name": "L1", "as": "tabs"}
        )
        assert resp.status_code == 422
        assert "reason" not in resp.json()
        assert frame is None

    async def test_a_preset_prunes_the_rail_to_its_members(self):
        app = _app(
            visible_panels=["artifacts", "ariel"],
            active_panel="artifacts",
            panel_presets=[_preset("Machine setup", "lattice", "ariel")],
        )
        _, _, frames = await _arrange(
            app,
            "/api/panel-op/open-arrangement",
            {"name": "Machine setup"},
            _side_by_side("lattice", "ariel"),
        )
        assert app.state.visible_panels == ["ariel", "lattice"]  # survivor keeps its slot
        assert [(f["panel"], f["visible"]) for f in frames] == [
            ("artifacts", False),
            ("lattice", True),
        ]
        assert app.state.active_panel in app.state.visible_panels

    @pytest.mark.parametrize(
        ("presets", "reason"),
        [
            ([], "unknown_arrangement"),
            ([_preset("L1", "nope")], "unknown_member"),
            ([_preset("L1", "terminal")], "terminal_in_layout"),
        ],
        ids=["unknown-name", "no-known-member", "terminal-only"],
    )
    async def test_refusals_are_named_and_change_nothing(self, presets, reason):
        app = _app(panel_presets=presets, visible_panels=["artifacts"])
        resp, frame, _ = await _arrange(app, "/api/panel-op/open-arrangement", {"name": "L1"})
        assert resp.status_code == 422
        assert resp.json()["reason"] == reason
        assert frame is None
        assert app.state.visible_panels == ["artifacts"]

    async def test_a_popped_out_member_is_refused(self):
        app = _app(panel_presets=[_preset("L1", "ariel")], dock_popouts=["ariel"])
        resp, frame, _ = await _arrange(app, "/api/panel-op/open-arrangement", {"name": "L1"})
        assert resp.status_code == 409
        assert resp.json()["reason"] == "popped_out"
        assert frame is None


def test_the_arrange_and_close_routes_are_the_op_routes_only():
    paths = {route.path for route in [*router.routes, *ops_router.routes]}
    assert "/api/panel-arrange" not in paths
    assert "/api/panel-close" not in paths
