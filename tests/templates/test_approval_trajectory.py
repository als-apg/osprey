"""The approval hook's trajectory describers: what a shot or an apply prompt shows and binds.

A shot fires the injection chain and an apply writes correctors, so the hook
puts both to a human and binds the call to the trajectory service's own target,
read from one snapshot fetched through the web terminal's proxy with the panel
token and no session. An apply of the pending step renders that step and binds
its digest; a put-back renders from the call body and binds the values digest,
which the hook restates from the shared stamp module and which is pinned here.
A shot and a step apply also state the SolveConfig in force, read from the
same snapshot's published solve; the ``solve.config`` names are pinned to the
service's model here. A prompt that cannot show what the call would act on is
not asked.

The hook ships standalone, so it is imported in-process and pointed at a real
stdlib HTTP server standing in for the terminal.
"""

from __future__ import annotations

import http.server
import importlib
import json
import math
import socket
import sys
import threading
import time
from contextlib import contextmanager

import pytest

from osprey.mcp_server import approval_stamp
from osprey.mcp_server.trajectory import client as trajectory_client
from osprey.services.trajectory.state import Catalog, Snapshot, Solve, SolveConfig

TOKEN = "panel-token-123"

_STEP = {
    "digest": "d" * 64,
    "deltas": {"HCM1": 1.5e-4, "VCM2": -2.0e-5},
    "factor": 0.8,
    "reason": "ok",
    "status": "solved within bounds",
    "all_zero": False,
    "zeroed_h": ["HCM9"],
    "zeroed_v": [],
    "no_response_names": [],
    "unsolved": ["VCM7"],
}


def _import_approval():
    """Import the approval hook in-process, undoing its ``sys.path`` shim."""
    before = list(sys.path)
    try:
        return importlib.import_module("osprey.templates.claude_code.claude.hooks.osprey_approval")
    finally:
        sys.path[:] = before


@pytest.fixture
def hook(monkeypatch):
    mod = _import_approval()
    monkeypatch.setattr(mod, "_HOOK_ENTERED_AT", time.monotonic())
    monkeypatch.setattr(mod, "_PREVIEW_BUDGET_S", 5.0)
    monkeypatch.setenv("OSPREY_PANEL_TOKEN", TOKEN)
    return mod


class _Terminal(http.server.BaseHTTPRequestHandler):
    routes: dict = {}
    seen: list = []

    def do_GET(self):  # stdlib signature
        self.seen.append((self.path, dict(self.headers)))
        status, body = self.routes.get(self.path, (404, {"detail": "missing"}))
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):  # stdlib signature
        pass


@contextmanager
def fake_terminal(monkeypatch, routes):
    """A terminal answering *routes*; yields the list of (path, headers) it saw."""
    seen: list = []
    handler = type("_H", (_Terminal,), {"routes": routes, "seen": seen})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OSPREY_WEB_PORT", str(server.server_address[1]))
    try:
        yield seen
    finally:
        server.shutdown()
        thread.join()


_SOLVE = {
    "sha": "5" * 64,
    "singular_values": [3.0, 1.0],
    "inverse_weights": [0.3, 0.0],
    "kind": ["LIVE", "DISCARDED"],
    "L": 2,
    "live_rank": 2,
    "rm": {"source": "model", "rows": 6, "h_cols": 1, "v_cols": 1, "computed_at": 1.0},
    "config": {
        "alpha": 0.002,
        "keep_live": 1,
        "n_discard": 1,
        "gain": 0.5,
        "excluded_bpms": ["BPM4"],
        "excluded_cms": ["HCM2", "VCM9"],
        "loss_cutoff": 0.25,
        "overridden": ["alpha", "excluded_cms"],
    },
}


def _state(step=_STEP, target="va", solve=_SOLVE):
    body = {"target": target, "step": step, "solve": solve}
    return {f"{trajectory_client.BASE_PATH}/state": (200, body)}


def _lines(hook, short_name, tool_input):
    return hook._trajectory_call_lines({}, short_name, tool_input, hook._preview_deadline())


# -- restated names and digest ------------------------------------------------


def test_restated_path_and_field_names_match_the_server(hook) -> None:
    """The proxy path and the approved field names are the server's own."""
    assert hook._TRAJECTORY_BASE_PATH == trajectory_client.BASE_PATH
    assert hook._APPROVED_TARGET_KEY in approval_stamp.APPROVED_FIELDS
    assert hook._APPROVED_STEP_DIGEST_KEY in approval_stamp.APPROVED_FIELDS
    assert hook._APPROVED_VALUES_SHA256_KEY in approval_stamp.APPROVED_FIELDS
    assert {hook._TRAJECTORY_SHOT, hook._TRAJECTORY_APPLY} == {
        name for name in approval_stamp.TOOL_REQUIRED_FIELDS if name.startswith("trajectory_")
    }
    assert set(approval_stamp.required_fields("trajectory_apply").exactly_one_of[0]) == {
        hook._APPROVED_STEP_DIGEST_KEY,
        hook._APPROVED_VALUES_SHA256_KEY,
    }


@pytest.mark.parametrize(
    ("target", "model", "values"),
    [
        ("va", "ring", [{"name": "HCM1", "before": 0.5}]),
        ("va", "ring", [{"name": "b", "before": 1}, {"name": "a", "before": -2.25e-7}]),
        ("tgt-ü", "mödel", [{"name": "Ωcorr", "before": 0.0, "extra": "ignored"}]),
        ("x", "m", [{"name": 7, "before": 1e300}]),
    ],
)
def test_put_back_digest_equals_the_shared_module(hook, target, model, values) -> None:
    """The hook's restated digest is the stamp module's, byte for byte."""
    assert hook._approved_values_sha256(
        target, model, values
    ) == approval_stamp.approved_values_sha256(target, model, values)


def test_put_back_digest_rejects_what_the_shared_module_rejects(hook) -> None:
    for values in ([{"name": "a"}], [{"name": "a", "before": math.nan}]):
        with pytest.raises((KeyError, ValueError)):
            approval_stamp.approved_values_sha256("t", "m", values)
        with pytest.raises((KeyError, ValueError)):
            hook._approved_values_sha256("t", "m", values)


def test_restated_solve_config_names_are_the_services() -> None:
    """The prompt states every field of the service's SolveConfig, by its own name."""
    hook = _import_approval()
    assert set(hook._SOLVE_CONFIG_FIELDS) == set(SolveConfig.model_fields)
    assert "solve" in Snapshot.model_fields
    assert {"config", "L", "live_rank"} <= set(Solve.model_fields)


def test_the_fixture_solve_is_a_valid_published_solve() -> None:
    Solve.model_validate(_SOLVE)


# -- the SolveConfig lines -------------------------------------------------------


def _solve_text(lines) -> str:
    return "\n".join(lines)


def test_the_solve_lines_state_every_setting_in_force(hook) -> None:
    text = _solve_text(hook._trajectory_solve_lines(_SOLVE))

    assert "SolveConfig in force:" in text
    assert "alpha: 0.002, gain: 0.5" in text
    assert "keep_live: 1 (n_discard 1 of L 2, live rank 2)" in text
    assert "loss_cutoff: 0.25" in text
    assert 'Excluded BPMs: "BPM4"' in text
    assert 'Excluded correctors: "HCM2", "VCM9"' in text
    assert 'Operator overrides: "alpha", "excluded_cms"' in text


def test_file_defaults_state_no_override(hook) -> None:
    solve = {**_SOLVE, "config": {**_SOLVE["config"], "overridden": [], "excluded_cms": []}}

    text = _solve_text(hook._trajectory_solve_lines(solve))

    assert "Operator overrides: none" in text
    assert "Excluded correctors: none" in text


@pytest.mark.parametrize("solve", [None, {}, {"config": None}, "x"])
def test_no_published_solve_is_stated(hook, solve) -> None:
    assert hook._trajectory_solve_lines(solve) == [
        "SolveConfig: none published (the service solves with no model matrix)"
    ]


def test_solve_names_are_quoted_as_prompt_text(hook) -> None:
    solve = {**_SOLVE, "config": {**_SOLVE["config"], "excluded_bpms": ["B\nPM"]}}

    text = _solve_text(hook._trajectory_solve_lines(solve))

    assert 'Excluded BPMs: "B\\nPM"' in text


_CATALOG = {
    "bpms": [
        {
            "name": "BPM4",
            "device": "bpm-04",
            "label": "BPM 04",
            "places": [{"id": "SR", "level": "ring"}, {"id": "SR/SECT2", "level": "sector"}],
            "s": 41.2,
            "plane": None,
            "groups": [],
        }
    ],
    "hcms": [{"name": "HCM2", "device": "hcm-02", "label": "HCM 02", "places": [], "plane": "x"}],
    "vcms": [],
    "groups": [],
}


def test_the_fixture_catalog_is_a_valid_catalog() -> None:
    Catalog.model_validate(_CATALOG)


def test_exclusions_are_named_by_label_and_place(hook) -> None:
    """A catalogued device reads as its label at its own (leaf) place, its matrix name after."""
    text = _solve_text(hook._trajectory_solve_lines(_SOLVE, _CATALOG))

    assert 'Excluded BPMs: "BPM 04" at "SR/SECT2" ("BPM4")' in text
    # No place: the label alone; not in the catalog: the matrix name alone.
    assert 'Excluded correctors: "HCM 02" ("HCM2"), "VCM9"' in text


@pytest.mark.parametrize("catalog", [None, {}, "x", {"bpms": "x"}, {"bpms": [{"label": "L"}]}])
def test_a_missing_or_malformed_catalog_names_exclusions_by_matrix_name(hook, catalog) -> None:
    text = _solve_text(hook._trajectory_solve_lines(_SOLVE, catalog))

    assert 'Excluded BPMs: "BPM4"' in text
    assert 'Excluded correctors: "HCM2", "VCM9"' in text


def test_catalog_labels_and_places_are_quoted_as_prompt_text(hook) -> None:
    entry = {**_CATALOG["bpms"][0], "label": "BPM\n04", "places": [{"id": "S\nR", "level": None}]}

    text = _solve_text(hook._trajectory_solve_lines(_SOLVE, {**_CATALOG, "bpms": [entry]}))

    assert 'Excluded BPMs: "BPM\\n04" at "S\\nR" ("BPM4")' in text


# -- the step apply -----------------------------------------------------------


def test_apply_renders_the_step_and_binds_target_and_digest(hook, monkeypatch) -> None:
    with fake_terminal(monkeypatch, _state()) as seen:
        lines, binding = _lines(hook, "trajectory_apply", {"step_digest": _STEP["digest"]})
    text = "\n".join(lines)
    assert 'Trajectory service target: "va"' in text
    assert '"ok"' in text and '"solved within bounds"' in text
    assert "Factor: 0.8" in text
    assert '"HCM1": +0.00015' in text
    assert '"VCM2": -2e-05' in text
    assert 'Zeroed horizontal correctors: "HCM9"' in text
    assert "Zeroed vertical correctors: none" in text
    assert 'Unsolved correctors: "VCM7"' in text
    assert "SolveConfig in force:" in text and "alpha: 0.002, gain: 0.5" in text
    assert binding == {"approved_target": "va", "approved_step_digest": _STEP["digest"]}

    # One GET of the snapshot, with the panel token and no session header.
    assert [path for path, _ in seen] == [f"{trajectory_client.BASE_PATH}/state"]
    headers = {k.lower(): v for k, v in seen[0][1].items()}
    assert headers["authorization"] == f"Bearer {TOKEN}"
    assert trajectory_client.CLIENT_HEADER.lower() not in headers


def test_apply_refuses_a_digest_that_is_not_the_pending_steps(hook, monkeypatch) -> None:
    with fake_terminal(monkeypatch, _state()):
        with pytest.raises(hook._TrajectoryRefused) as caught:
            _lines(hook, "trajectory_apply", {"step_digest": "e" * 64})
    assert caught.value.audit_reason == "trajectory_step_digest_mismatch"


def test_apply_refuses_when_no_step_is_pending(hook, monkeypatch) -> None:
    with fake_terminal(monkeypatch, _state(step=None)):
        with pytest.raises(hook._TrajectoryRefused) as caught:
            _lines(hook, "trajectory_apply", {"step_digest": _STEP["digest"]})
    assert caught.value.audit_reason == "trajectory_no_step"


@pytest.mark.parametrize("status", [401, 404, 502])
def test_apply_refuses_when_the_fetch_fails(hook, monkeypatch, status) -> None:
    routes = {f"{trajectory_client.BASE_PATH}/state": (status, {"detail": "no"})}
    with fake_terminal(monkeypatch, routes):
        with pytest.raises(hook._TrajectoryRefused) as caught:
            _lines(hook, "trajectory_apply", {"step_digest": _STEP["digest"]})
    assert caught.value.audit_reason.startswith("trajectory_state_unavailable")
    assert f"HTTP {status}" in caught.value.reason


def test_apply_refuses_when_the_terminal_is_unreachable(hook, monkeypatch) -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    monkeypatch.setenv("OSPREY_WEB_PORT", str(port))
    with pytest.raises(hook._TrajectoryRefused) as caught:
        _lines(hook, "trajectory_apply", {"step_digest": _STEP["digest"]})
    assert caught.value.audit_reason.startswith("trajectory_state_unavailable")


def test_apply_refuses_when_the_budget_is_spent(hook, monkeypatch) -> None:
    monkeypatch.setattr(hook, "_PREVIEW_BUDGET_S", -1.0)
    with fake_terminal(monkeypatch, _state()) as seen:
        with pytest.raises(hook._TrajectoryRefused):
            _lines(hook, "trajectory_apply", {"step_digest": _STEP["digest"]})
    assert seen == []


@pytest.mark.parametrize(
    "tool_input",
    [
        {},
        {"step_digest": "d" * 64, "model": "ring", "values": [{"name": "a", "before": 1}]},
        {"model": "ring"},
        {"values": [{"name": "a", "before": 1}]},
        {"model": "ring", "values": []},
    ],
)
def test_apply_refuses_a_call_that_is_neither_a_step_nor_a_put_back(hook, tool_input) -> None:
    with pytest.raises(hook._TrajectoryRefused) as caught:
        _lines(hook, "trajectory_apply", tool_input)
    assert caught.value.audit_reason == "trajectory_apply_bad_request"


# -- the put-back --------------------------------------------------------------


def test_put_back_renders_the_body_and_binds_the_values_digest(hook, monkeypatch) -> None:
    values = [{"name": "VCM2", "before": -1.0}, {"name": "HCM1", "before": 0.25}]
    with fake_terminal(monkeypatch, _state(target="ring-target")):
        lines, binding = _lines(hook, "trajectory_apply", {"model": "ring", "values": values})
    text = "\n".join(lines)
    assert 'Trajectory service target: "ring-target"' in text
    assert 'Put-back on model "ring": 2 correctors' in text
    assert '"HCM1" -> 0.25' in text and '"VCM2" -> -1.0' in text
    digest = approval_stamp.approved_values_sha256("ring-target", "ring", values)
    assert f"Values sha256: {digest}" in text
    assert binding == {"approved_target": "ring-target", "approved_values_sha256": digest}


def test_put_back_refuses_values_it_cannot_show(hook, monkeypatch) -> None:
    with fake_terminal(monkeypatch, _state()):
        with pytest.raises(hook._TrajectoryRefused) as caught:
            _lines(hook, "trajectory_apply", {"model": "ring", "values": [{"name": "a"}]})
    assert caught.value.audit_reason == "trajectory_put_back_unreadable"


# -- the shot ------------------------------------------------------------------


def test_shot_binds_the_services_target(hook, monkeypatch) -> None:
    with fake_terminal(monkeypatch, _state(step=None, target="va")):
        lines, binding = _lines(hook, "trajectory_shot", {"n_shots": 3, "average": "nanmean"})
    assert 'Trajectory service target: "va"' in lines
    assert 'Shots: 3, averaged by "nanmean"' in lines
    assert "SolveConfig in force:" in lines
    assert 'Excluded correctors: "HCM2", "VCM9"' in "\n".join(lines)
    assert binding == {"approved_target": "va"}


@pytest.mark.parametrize(
    ("short_name", "tool_input", "step"),
    [("trajectory_shot", {}, None), ("trajectory_apply", {"step_digest": "d" * 64}, _STEP)],
)
def test_shot_and_apply_name_exclusions_from_the_served_catalog(
    hook, monkeypatch, short_name, tool_input, step
) -> None:
    routes = _state(step=step)
    path = f"{trajectory_client.BASE_PATH}/state"
    routes[path] = (200, {**routes[path][1], "catalog": _CATALOG})
    with fake_terminal(monkeypatch, routes):
        lines, _binding = _lines(hook, short_name, tool_input)
    assert 'Excluded BPMs: "BPM 04" at "SR/SECT2" ("BPM4")' in "\n".join(lines)


def test_put_back_states_no_solve_config(hook, monkeypatch) -> None:
    """A put-back writes recorded values; no solve is involved."""
    values = [{"name": "HCM1", "before": 0.25}]
    with fake_terminal(monkeypatch, _state()):
        lines, _binding = _lines(hook, "trajectory_apply", {"model": "ring", "values": values})
    assert not any("SolveConfig" in line for line in lines)


def test_shot_refuses_when_the_fetch_fails(hook, monkeypatch) -> None:
    with fake_terminal(monkeypatch, {}):
        with pytest.raises(hook._TrajectoryRefused):
            _lines(hook, "trajectory_shot", {})


def test_bound_input_replaces_agent_sent_approvals(hook, monkeypatch) -> None:
    """An ``approved_*`` value the agent sent never survives into the call."""
    with fake_terminal(monkeypatch, _state()):
        lines, updated, binding = hook._trajectory_call(
            {},
            "mcp__trajectory__trajectory_apply",
            "trajectory_apply",
            {
                "step_digest": _STEP["digest"],
                "approved_target": "forged",
                "approved_values_sha256": "forged",
            },
            {},
        )
    assert updated == {"step_digest": _STEP["digest"], **binding}
    assert "approved_values_sha256" not in updated


def test_header_lists_the_trajectory_tools(hook) -> None:
    doc = hook.__doc__ or ""
    tools_line = next(line for line in doc.splitlines() if line.startswith("tools:"))
    tools = {name.strip() for name in tools_line.split(":", 1)[1].split(",")}
    assert {"trajectory_shot", "trajectory_apply"} <= tools
    assert "trajectory_step" not in tools
