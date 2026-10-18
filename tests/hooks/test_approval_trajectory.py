"""The approval hook, run as the harness runs it, on a trajectory shot and apply.

The describer contract (lines, binding, refusals, the restated digest) is pinned
in-process by ``tests/templates/test_approval_trajectory.py``. These run the
whole hook as a subprocess against a stdlib HTTP server standing in for the web
terminal, so the wiring through ``main()`` — the prompt, ``updatedInput`` and
the deny — is what is under test.
"""

from __future__ import annotations

import http.server
import json
import shlex
import threading
from contextlib import contextmanager

import pytest

from osprey.registry.mcp import FRAMEWORK_SERVERS

HOOK = "osprey_approval.py"
PREFIXES = {"approval_prefixes": ["mcp__trajectory__"]}
STATE_PATH = "/panel/trajectory/api/state"
DIGEST = "a" * 64
STEP = {
    "digest": DIGEST,
    "deltas": {"HCM1": 1.5e-4, "VCM2": -2.0e-5},
    "factor": 0.5,
    "reason": "ok",
    "status": "solved",
    "all_zero": False,
    "zeroed_h": [],
    "zeroed_v": ["VCM3"],
    "unsolved": [],
}


@contextmanager
def fake_terminal(monkeypatch, routes):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # stdlib signature
            status, body = routes.get(self.path, (404, {"detail": "missing"}))
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):  # stdlib signature
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OSPREY_WEB_PORT", str(server.server_address[1]))
    try:
        yield
    finally:
        server.shutdown()
        thread.join()


def _rendered_args(tool_name: str) -> list[str]:
    """The arguments the registry renders after the hook script for ``tool_name``.

    An apply buys the pre-flight budget its preview needs; run without them the
    hook falls back to the default budget, which a loaded host spends on start-up.
    """
    (rule,) = [
        rule for rule in FRAMEWORK_SERVERS["trajectory"].hooks_pre if rule.matcher == tool_name
    ]
    (entry,) = rule.hooks
    words = shlex.split(entry.command)
    return words[[word.endswith(HOOK) for word in words].index(True) + 1 :]


@pytest.fixture
def run_hook(hook_runner):
    def run(tool_name, tool_input, **kwargs):
        return hook_runner(HOOK, tool_name, tool_input, args=_rendered_args(tool_name), **kwargs)

    return run


@pytest.fixture
def config(make_config):
    return make_config({"approval": {"enabled": True, "default_policy": "always"}})


def _specific(result):
    assert result is not None
    return result["hookSpecificOutput"]


def test_apply_asks_with_the_step_and_binds_it(run_hook, config, monkeypatch) -> None:
    with fake_terminal(monkeypatch, {STATE_PATH: (200, {"target": "va", "step": STEP})}):
        result = run_hook(
            "mcp__trajectory__trajectory_apply",
            {"step_digest": DIGEST, "approved_step_digest": "forged"},
            config_path=config,
            hook_config=PREFIXES,
        )
    out = _specific(result)
    assert out["permissionDecision"] == "ask"
    reason = out["permissionDecisionReason"]
    assert "Tool: trajectory_apply" in reason
    assert 'Trajectory service target: "va"' in reason
    assert "Factor: 0.5" in reason
    assert '"HCM1": +0.00015' in reason
    assert 'Zeroed vertical correctors: "VCM3"' in reason
    assert out["updatedInput"] == {
        "step_digest": DIGEST,
        "approved_target": "va",
        "approved_step_digest": DIGEST,
    }


def test_apply_is_denied_when_the_step_changed(run_hook, config, monkeypatch) -> None:
    with fake_terminal(monkeypatch, {STATE_PATH: (200, {"target": "va", "step": STEP})}):
        result = run_hook(
            "mcp__trajectory__trajectory_apply",
            {"step_digest": "b" * 64},
            config_path=config,
            hook_config=PREFIXES,
        )
    out = _specific(result)
    assert out["permissionDecision"] == "deny"
    assert "trajectory_step" in out["permissionDecisionReason"]


def test_apply_is_denied_when_the_service_cannot_be_read(run_hook, config, monkeypatch) -> None:
    with fake_terminal(monkeypatch, {STATE_PATH: (502, {"detail": "down"})}):
        result = run_hook(
            "mcp__trajectory__trajectory_apply",
            {"step_digest": DIGEST},
            config_path=config,
            hook_config=PREFIXES,
        )
    out = _specific(result)
    assert out["permissionDecision"] == "deny"
    assert "HTTP 502" in out["permissionDecisionReason"]


def test_put_back_asks_and_binds_the_values_digest(run_hook, config, monkeypatch) -> None:
    from osprey.mcp_server.approval_stamp import approved_values_sha256

    values = [{"name": "HCM1", "before": 0.25}]
    with fake_terminal(monkeypatch, {STATE_PATH: (200, {"target": "va", "step": None})}):
        result = run_hook(
            "mcp__trajectory__trajectory_apply",
            {"model": "ring", "values": values},
            config_path=config,
            hook_config=PREFIXES,
        )
    out = _specific(result)
    assert out["permissionDecision"] == "ask"
    assert out["updatedInput"]["approved_target"] == "va"
    assert out["updatedInput"]["approved_values_sha256"] == approved_values_sha256(
        "va", "ring", values
    )
    assert "approved_step_digest" not in out["updatedInput"]


def test_shot_asks_and_binds_the_services_target(run_hook, config, monkeypatch) -> None:
    with fake_terminal(monkeypatch, {STATE_PATH: (200, {"target": "va", "step": None})}):
        result = run_hook(
            "mcp__trajectory__trajectory_shot",
            {"n_shots": 2},
            config_path=config,
            hook_config=PREFIXES,
        )
    out = _specific(result)
    assert out["permissionDecision"] == "ask"
    assert out["updatedInput"] == {"n_shots": 2, "approved_target": "va"}


def test_skip_policy_allows_without_fetching(run_hook, make_config, monkeypatch) -> None:
    config = make_config({"approval": {"enabled": True, "tools": {"trajectory_shot": "skip"}}})
    with fake_terminal(monkeypatch, {}):
        result = run_hook(
            "mcp__trajectory__trajectory_shot",
            {},
            config_path=config,
            hook_config=PREFIXES,
        )
    assert _specific(result)["permissionDecision"] == "allow"
