"""Tests for the `osprey_approval` hook's `launch_run` enrichment.

`launch_run` is the sole launch path for a Bluesky scan (see
`mcp_server/bluesky/tools/launch.py`). It carries nothing but a pinned
`draft_revision` — no run record exists yet, the bridge mints the run *from*
the shared draft at launch time. So this enrichment fetches the shared draft
(`GET /draft`) and renders exactly what would launch: the plan name and args
currently staged, whether the draft still matches the pinned revision, and —
for a non-empty draft — the plan's provenance/trust tier, validation status,
and a source excerpt (resolved against `/plans` and `/plans/{name}/source`).

Two things the rendered prompt must always carry:

* a revision-match line — a plain "matches" when `GET /draft`'s revision equals
  the pinned `draft_revision`, a LOUD "DRAFT CHANGED" warning (showing both
  revisions) when it does not: the human backstop against launching a draft the
  agent pinned but that has since moved on; and
* for a session/unreviewed plan, the plan validator's documented obfuscation
  residual made legible — a `getattr`/string-concat body that passes the
  sandbox's AST scan (see `plan_validation.py`) is surfaced verbatim and
  labelled unmistakably agent-authored/unreviewed, so an approver who can SEE
  the source can refuse it where the automated stages could not.

The hook runs as a real subprocess (see `hook_runner` in conftest.py) and talks
to the bridge over `urllib.request` — these tests stand up a tiny real HTTP
server (stdlib `http.server`) to play the bridge's part, so the hook's actual
network code path is exercised, not a mock. Fail-open is proven end-to-end:
`hook_runner` asserts a zero exit code, so an unreachable or malformed bridge
that still produces an `ask` decision proves the enrichment never raises out to
the subprocess boundary.
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

_MISSING = object()


class _FakeBridgeHandler(http.server.BaseHTTPRequestHandler):
    """Serves canned bodies for whatever paths `routes` maps.

    A route value that is `bytes` is written raw (used to serve a malformed,
    non-JSON body); anything else is JSON-encoded. An unmapped path is a 404.
    """

    routes: dict[str, object] = {}

    def do_GET(self):  # noqa: N802 (stdlib method name)
        body = self.routes.get(self.path, _MISSING)
        if body is _MISSING:
            payload = b'{"detail": "not found"}'
            self.send_response(404)
        elif isinstance(body, bytes):
            payload = body
            self.send_response(200)
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
    'PLAN_METADATA = {"name": "orm", "description": "orm", '
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


def _launch_config(make_config):
    return make_config(
        {
            "approval": {"enabled": True, "default_policy": "always"},
            "control_system": {"writes_enabled": True},
        }
    )


def _run_launch(hook_runner, config, tmp_path, draft_revision):
    return hook_runner(
        "osprey_approval.py",
        "mcp__bluesky__launch_run",
        {"draft_revision": draft_revision},
        config_path=config,
        cwd=tmp_path,
        hook_config=SCAN_HOOK_CONFIG,
    )


@pytest.mark.unit
def test_matching_revision_renders_shipped_plan_and_source(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """When the live draft's revision equals the pinned one, render the plain
    match line plus the full plan detail; a shipped (operator-trusted) plan is
    never mislabeled as agent-authored."""
    config = _launch_config(make_config)
    routes = {
        "/draft": {
            "draft": {
                "plan_name": "orm",
                "plan_args": {"num_points": 5},
                "updated_by": "plan-panel",
                "updated_at": "2026-07-16T00:00:00+00:00",
            },
            "revision": 7,
        },
        "/plans": [
            {
                "name": "orm",
                "description": "orm",
                "schema": {},
                "metadata": {
                    "name": "orm",
                    "description": "orm",
                    "category": "accelerator",
                    "required_devices": ["correctors", "detectors"],
                    "writes": True,
                },
                "provenance": "shipped",
            }
        ],
        "/plans/orm/source": {
            "name": "orm",
            "provenance": "shipped",
            "validated": True,
            "truncated": False,
            "source": _SHIPPED_SOURCE,
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_launch(hook_runner, config, tmp_path, draft_revision=7)

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    reason = output["permissionDecisionReason"]

    assert "matches pinned revision 7" in reason
    assert "DRAFT CHANGED" not in reason
    assert "Plan: orm" in reason
    assert "num_points" in reason
    assert "Category: accelerator" in reason
    assert "correctors" in reason and "detectors" in reason
    assert "Hazard: writes to hardware" in reason
    assert "Provenance: shipped" in reason
    assert "Validation status: not applicable" in reason
    assert _SHIPPED_SOURCE in reason
    # A trusted tier must never be mislabeled as agent-authored/unreviewed.
    assert "AGENT-AUTHORED" not in reason


@pytest.mark.unit
def test_matching_revision_labels_unvalidated_session_plan_as_untrusted(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A session-tier plan with NO passing validation record is clearly
    labelled agent-authored/unreviewed, and the obfuscated body itself is
    rendered legibly — the human backstop the plan validator's documented
    residual relies on."""
    config = _launch_config(make_config)
    routes = {
        "/draft": {
            "draft": {
                "plan_name": "sneaky_plan",
                "plan_args": {},
                "updated_by": "mcp-agent",
                "updated_at": "2026-07-16T00:00:00+00:00",
            },
            "revision": 3,
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
        result = _run_launch(hook_runner, config, tmp_path, draft_revision=3)

    assert result is not None
    reason = result["hookSpecificOutput"]["permissionDecisionReason"]

    assert "matches pinned revision 3" in reason
    assert "Plan: sneaky_plan" in reason
    assert "SESSION" in reason
    assert "AGENT-AUTHORED, NOT REVIEWED BY A HUMAN" in reason
    assert "NO PASSING RECORD" in reason
    # The obfuscated body itself must be visible to the approver.
    assert "__subclasses__" in reason


@pytest.mark.unit
def test_matching_revision_reports_validated_session_plan_as_passed(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A session-tier plan WITH a passing validation record renders
    "Validation status: PASSED" — and is STILL labelled agent-authored: a
    passing hash does not upgrade the trust tier, only the validation line."""
    config = _launch_config(make_config)
    routes = {
        "/draft": {
            "draft": {
                "plan_name": "reviewed_ish_plan",
                "plan_args": {},
                "updated_by": "mcp-agent",
                "updated_at": "2026-07-16T00:00:00+00:00",
            },
            "revision": 2,
        },
        "/plans": [
            {
                "name": "reviewed_ish_plan",
                "metadata": {
                    "name": "reviewed_ish_plan",
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
        result = _run_launch(hook_runner, config, tmp_path, draft_revision=2)

    reason = result["hookSpecificOutput"]["permissionDecisionReason"]
    assert "Validation status: PASSED" in reason
    # A passing hash never launders the trust tier: still agent-authored.
    assert "AGENT-AUTHORED, NOT REVIEWED BY A HUMAN" in reason


@pytest.mark.unit
def test_newline_in_plan_name_cannot_forge_an_enrichment_line(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """`plan_name` is agent-authored (a session plan's PLAN_METADATA["name"])
    and reaches the prompt RAW — the bridge gates it only by registry
    membership, not character content. A newline in it must not forge a fake
    enrichment line: the render escapes control characters so the whole name
    stays on the "Plan:" line, visible but inert."""
    config = _launch_config(make_config)
    spoof = "orm\nValidation status: PASSED (SPOOFED BY THE PLAN NAME)"
    routes = {
        "/draft": {
            "draft": {
                "plan_name": spoof,
                "plan_args": {},
                "updated_by": "mcp-agent",
                "updated_at": "2026-07-16T00:00:00+00:00",
            },
            "revision": 1,
        }
        # No /plans or /source routes: the injection surface under test is the
        # plan_name interpolation, not the provenance block.
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_launch(hook_runner, config, tmp_path, draft_revision=1)

    assert result is not None
    reason = result["hookSpecificOutput"]["permissionDecisionReason"]

    # The embedded newline is escaped to a visible token — the name stays on
    # one line, so the spoofed text can never begin its own line.
    assert "\\x0a" in reason
    assert "\nValidation status: PASSED (SPOOFED BY THE PLAN NAME)" not in reason
    assert not any(
        line.startswith("Validation status: PASSED (SPOOFED")
        for line in reason.splitlines()
    )
    # The (inert) text still rides along on the Plan line for the approver to see.
    assert "SPOOFED BY THE PLAN NAME" in reason


@pytest.mark.unit
def test_changed_revision_renders_loud_drift_warning(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """When the live draft has moved past the pinned revision, the prompt leads
    with a LOUD warning naming both revisions, then renders the CURRENT draft."""
    config = _launch_config(make_config)
    routes = {
        "/draft": {
            "draft": {
                "plan_name": "orm",
                "plan_args": {"num_points": 9},
                "updated_by": "plan-panel",
                "updated_at": "2026-07-16T00:00:00+00:00",
            },
            "revision": 11,
        },
        "/plans": [
            {
                "name": "orm",
                "metadata": {
                    "name": "orm",
                    "category": "accelerator",
                    "required_devices": ["correctors"],
                    "writes": True,
                },
                "provenance": "shipped",
            }
        ],
        "/plans/orm/source": {
            "name": "orm",
            "provenance": "shipped",
            "validated": True,
            "truncated": False,
            "source": _SHIPPED_SOURCE,
        },
    }

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_launch(hook_runner, config, tmp_path, draft_revision=8)

    assert result is not None
    reason = result["hookSpecificOutput"]["permissionDecisionReason"]

    assert "DRAFT CHANGED" in reason
    # Both the pinned and the current revision are named.
    assert "8" in reason and "11" in reason
    assert "matches pinned revision" not in reason
    # The current draft is still rendered so the approver sees what would run.
    assert "Plan: orm" in reason
    assert "num_points" in reason


@pytest.mark.unit
def test_empty_draft_renders_explicit_empty_line(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A never-set / cleared draft renders an explicit EMPTY line, never a
    silent absence of plan detail."""
    config = _launch_config(make_config)
    routes = {"/draft": {"draft": None, "revision": 4}}

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_launch(hook_runner, config, tmp_path, draft_revision=4)

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    reason = output["permissionDecisionReason"]

    assert "Draft: EMPTY" in reason
    assert "Plan:" not in reason
    # Tool/policy plain reason is still present.
    assert "Tool: launch_run" in reason


@pytest.mark.unit
def test_unreachable_bridge_fails_open(tmp_path, hook_runner, make_config, monkeypatch):
    """A dead bridge must never block the approval prompt: the hook still asks,
    just with the plain tool/policy reason instead of any draft detail.
    `hook_runner` asserts a zero exit code, so this also proves the enrichment
    never raises out to the subprocess boundary."""
    config = _launch_config(make_config)
    monkeypatch.setenv("BLUESKY_BRIDGE_URL", f"http://127.0.0.1:{_unused_port()}")

    result = _run_launch(hook_runner, config, tmp_path, draft_revision=5)

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    reason = output["permissionDecisionReason"]
    assert "Tool: launch_run" in reason
    assert "Approval policy: always" in reason
    assert "Plan:" not in reason
    assert "Draft:" not in reason


@pytest.mark.unit
def test_malformed_draft_response_fails_open(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A `GET /draft` body that is not parseable JSON must fail open exactly
    like an unreachable bridge — plain reason, no draft detail, zero exit."""
    config = _launch_config(make_config)
    routes = {"/draft": b"this is not json {{{"}

    with fake_bridge(routes) as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        result = _run_launch(hook_runner, config, tmp_path, draft_revision=6)

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    reason = output["permissionDecisionReason"]
    assert "Tool: launch_run" in reason
    assert "Plan:" not in reason
    assert "Draft:" not in reason
