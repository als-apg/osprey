"""Tests for the `osprey_approval` hook's `launch_run` enrichment (task 2.6).

`launch_run` is the sole promote path for a Bluesky scan (see
`mcp_server/bluesky/tools/launch.py`) — its own tool_input carries nothing but a
bare `run_id`. This enrichment resolves that id against the bridge's
`/runs/{id}`, `/plans`, and `/plans/{name}/source` routes (task 2.6's new
endpoint) to render the plan actually about to launch: its metadata,
provenance/trust tier, validation status, and a truncated source excerpt.

This is the human backstop for the plan validator's documented, accepted
obfuscation residual (a `getattr`/string-concat body that passes the
sandbox's AST import/pattern scan — see `plan_validation.py`'s module
docstring): an approver who can SEE the actual source has a chance to refuse
it even where the earlier automated stages could not. So the tests below
assert the rendered reason both surfaces that source text AND labels a
session/unreviewed plan unmistakably as agent-authored/unreviewed.

The hook runs as a real subprocess (see `hook_runner` in conftest.py) and
talks to the bridge over `urllib.request` — these tests stand up a tiny
real HTTP server (stdlib `http.server`) to play the bridge's part, so the
hook's actual network code path is exercised, not a mock.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
from contextlib import contextmanager

import pytest

SCAN_HOOK_CONFIG = {
    "server_prefixes": ["mcp__bluesky__"],
    "approval_prefixes": ["mcp__bluesky__"],
}


class _FakeBridgeHandler(http.server.BaseHTTPRequestHandler):
    """Serves canned JSON bodies for whatever paths `routes` maps."""

    routes: dict[str, object] = {}

    def do_GET(self):  # noqa: N802 (stdlib method name)
        body = self.routes.get(self.path)
        if body is None:
            payload = b'{"detail": "not found"}'
            self.send_response(404)
        else:
            payload = json.dumps(body).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):  # noqa: A002 (stdlib signature)
        pass  # keep test output quiet


@contextmanager
def fake_bridge(routes: dict[str, object]):
    """Runs a real threaded HTTP server for the duration of the `with` block."""
    handler_cls = type("_Handler", (_FakeBridgeHandler,), {"routes": routes})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join()


def _unused_port() -> int:
    """A port nothing is listening on, for the fail-open (unreachable) test."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


_SHIPPED_SOURCE = (
    'PLAN_METADATA = {"name": "response_matrix", "description": "orm", '
    '"category": "accelerator", "required_devices": ["correctors", "detectors"], '
    '"writes": True}\n\n'
    "def build_plan(devices, params):\n"
    "    yield from ()\n"
)

_OBFUSCATED_SESSION_SOURCE = (
    'PLAN_METADATA = {"name": "sneaky_plan", "description": "", '
    '"category": "accelerator", "required_devices": [], "writes": False}\n\n'
    "def build_plan(devices, params):\n"
    "    leak = ().__class__.__base__.__subclasses__()\n"
    "    yield from ()\n"
)


@pytest.mark.unit
def test_launch_run_renders_shipped_plan_metadata_and_source(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A shipped (operator-trusted) plan renders category/devices/hazard/
    provenance/validation-status/source without any UNTRUSTED labeling."""
    config = make_config(
        {
            "approval": {"enabled": True, "default_policy": "always"},
            "control_system": {"writes_enabled": True},
        }
    )
    routes = {
        "/runs/run-1": {
            "id": "run-1",
            "status": "intent",
            "plan_name": "response_matrix",
            "plan_args": {"num_points": 5},
        },
        "/plans": [
            {
                "name": "response_matrix",
                "description": "orm",
                "schema": {},
                "metadata": {
                    "name": "response_matrix",
                    "description": "orm",
                    "category": "accelerator",
                    "required_devices": ["correctors", "detectors"],
                    "writes": True,
                },
                "provenance": "shipped",
            }
        ],
        "/plans/response_matrix/source": {
            "name": "response_matrix",
            "provenance": "shipped",
            "validated": True,
            "truncated": False,
            "source": _SHIPPED_SOURCE,
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = hook_runner(
            "osprey_approval.py",
            "mcp__bluesky__launch_run",
            {"run_id": "run-1"},
            config_path=config,
            cwd=tmp_path,
            hook_config=SCAN_HOOK_CONFIG,
        )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    reason = output["permissionDecisionReason"]

    assert "Plan: response_matrix" in reason
    assert "Category: accelerator" in reason
    assert "correctors" in reason and "detectors" in reason
    assert "Hazard: writes to hardware" in reason
    assert "Provenance: shipped" in reason
    assert "Validation status: not applicable" in reason
    assert _SHIPPED_SOURCE in reason
    assert "num_points" in reason
    # A trusted tier must never be mislabeled as agent-authored/unreviewed.
    assert "AGENT-AUTHORED" not in reason


@pytest.mark.unit
def test_launch_run_labels_unvalidated_session_plan_as_untrusted(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A session-tier plan with NO passing validation record is clearly
    labelled agent-authored/unreviewed, and the obfuscated body itself is
    rendered legibly — the human backstop the plan validator's documented
    residual relies on."""
    config = make_config(
        {
            "approval": {"enabled": True, "default_policy": "always"},
            "control_system": {"writes_enabled": True},
        }
    )
    routes = {
        "/runs/run-2": {
            "id": "run-2",
            "status": "intent",
            "plan_name": "sneaky_plan",
            "plan_args": {},
        },
        "/plans": [],  # quarantined: absent from GET /plans entirely
        "/plans/sneaky_plan/source": {
            "name": "sneaky_plan",
            "provenance": "session",
            "validated": False,
            "truncated": False,
            "source": _OBFUSCATED_SESSION_SOURCE,
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = hook_runner(
            "osprey_approval.py",
            "mcp__bluesky__launch_run",
            {"run_id": "run-2"},
            config_path=config,
            cwd=tmp_path,
            hook_config=SCAN_HOOK_CONFIG,
        )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    reason = output["permissionDecisionReason"]

    assert "Plan: sneaky_plan" in reason
    assert "SESSION" in reason
    assert "AGENT-AUTHORED, NOT REVIEWED BY A HUMAN" in reason
    assert "NO PASSING RECORD" in reason
    # The obfuscated body itself must be visible to the approver.
    assert "__subclasses__" in reason


@pytest.mark.unit
def test_launch_run_reports_a_validated_session_plan_as_passed(
    tmp_path, hook_runner, make_config, monkeypatch
):
    config = make_config(
        {
            "approval": {"enabled": True, "default_policy": "always"},
            "control_system": {"writes_enabled": True},
        }
    )
    routes = {
        "/runs/run-3": {
            "id": "run-3",
            "status": "intent",
            "plan_name": "reviewed_ish_plan",
            "plan_args": {},
        },
        "/plans": [
            {
                "name": "reviewed_ish_plan",
                "description": "",
                "schema": {},
                "metadata": {
                    "name": "reviewed_ish_plan",
                    "description": "",
                    "category": "accelerator",
                    "required_devices": [],
                    "writes": False,
                },
                "provenance": "session",
            }
        ],
        "/plans/reviewed_ish_plan/source": {
            "name": "reviewed_ish_plan",
            "provenance": "session",
            "validated": True,
            "truncated": False,
            "source": "PLAN_METADATA = {}\n",
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = hook_runner(
            "osprey_approval.py",
            "mcp__bluesky__launch_run",
            {"run_id": "run-3"},
            config_path=config,
            cwd=tmp_path,
            hook_config=SCAN_HOOK_CONFIG,
        )

    reason = result["hookSpecificOutput"]["permissionDecisionReason"]
    assert "AGENT-AUTHORED, NOT REVIEWED BY A HUMAN" in reason
    assert "Validation status: PASSED" in reason


@pytest.mark.unit
def test_launch_run_fails_open_when_bridge_is_unreachable(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A dead bridge must never block the approval prompt: the hook still
    asks, just with the plain tool/policy reason instead of any plan detail.
    `hook_runner` itself asserts a zero exit code, so this also proves the
    enrichment code never raises out to the subprocess boundary.
    """
    config = make_config(
        {
            "approval": {"enabled": True, "default_policy": "always"},
            "control_system": {"writes_enabled": True},
        }
    )
    monkeypatch.setenv("BLUESKY_BRIDGE_URL", f"http://127.0.0.1:{_unused_port()}")

    result = hook_runner(
        "osprey_approval.py",
        "mcp__bluesky__launch_run",
        {"run_id": "run-unreachable"},
        config_path=config,
        cwd=tmp_path,
        hook_config=SCAN_HOOK_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    reason = output["permissionDecisionReason"]
    assert "Tool: launch_run" in reason
    assert "Approval policy: always" in reason
    assert "Plan:" not in reason


@pytest.mark.unit
def test_launch_run_without_a_run_id_degrades_to_the_plain_reason(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Malformed tool_input (no run_id) degrades gracefully — never a crash."""
    config = make_config(
        {
            "approval": {"enabled": True, "default_policy": "always"},
            "control_system": {"writes_enabled": True},
        }
    )
    monkeypatch.setenv("BLUESKY_BRIDGE_URL", f"http://127.0.0.1:{_unused_port()}")

    result = hook_runner(
        "osprey_approval.py",
        "mcp__bluesky__launch_run",
        {},
        config_path=config,
        cwd=tmp_path,
        hook_config=SCAN_HOOK_CONFIG,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    assert "Tool: launch_run" in output["permissionDecisionReason"]
