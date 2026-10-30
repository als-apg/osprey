"""The launch's terminal session on the OTLP export and the ledger alike.

Every launch carries its terminal session as the ``terminal.session`` resource
attribute of its OTLP events — the same value the ledger writers file the
launch's entries under. A launch site that spawns the session later fills the
attribute in; a dispatched job carries ``dispatch-<run id>`` and no posture; the
sandbox child keeps the session and no server spec can pin it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from osprey.build import claude_code_telemetry as cct
from osprey.build.claude_code_telemetry import (
    export_host_terminal_session,
    record_telemetry_env,
    terminal_session_env,
)

_RECORDED_LAUNCH = {
    "OSPREY_AGENT_RECORD_URL": "http://127.0.0.1:10061",
    "OSPREY_AGENT_RECORD_LOGIN": "alice:s3cret",
}


def _record_env() -> dict[str, str]:
    return record_telemetry_env(_RECORDED_LAUNCH)


# --- the terminal session at the launch sites ------------------------------


def test_a_launch_site_fills_the_record_session_attribute():
    added = terminal_session_env("abc-123", _record_env())
    assert added == {
        "OSPREY_TERMINAL_SESSION": "abc-123",
        "OTEL_RESOURCE_ATTRIBUTES": "terminal.session=abc-123",
    }


def test_a_launch_site_keeps_other_resource_attributes_in_order():
    inherited = {"OTEL_RESOURCE_ATTRIBUTES": "service.name=osprey,terminal.session="}
    assert terminal_session_env("abc-123", inherited)["OTEL_RESOURCE_ATTRIBUTES"] == (
        "service.name=osprey,terminal.session=abc-123"
    )


def test_a_launch_site_leaves_a_collector_of_its_owns_attributes_alone():
    inherited = {"OTEL_RESOURCE_ATTRIBUTES": "service.name=osprey"}
    assert terminal_session_env("abc", inherited) == {"OSPREY_TERMINAL_SESSION": "abc"}
    assert terminal_session_env("abc", {}) == {"OSPREY_TERMINAL_SESSION": "abc"}


def test_a_host_launch_mints_a_chat_session_and_no_posture():
    environ: dict[str, str] = {}
    session = export_host_terminal_session(environ)
    assert session.startswith("chat-") and len(session) == len("chat-") + 36
    assert environ == {"OSPREY_TERMINAL_SESSION": session}


def test_a_host_launch_inside_a_terminal_keeps_that_terminals_session():
    environ = {"OSPREY_POSTURE_SESSION": "pty-key"}
    assert export_host_terminal_session(environ) == "pty-key"
    assert environ["OSPREY_TERMINAL_SESSION"] == "pty-key"
    environ = {"OSPREY_TERMINAL_SESSION": "dispatch-r1"}
    assert export_host_terminal_session(environ) == "dispatch-r1"


def test_the_session_literal_matches_its_owner():
    from osprey.audit.envelope import TERMINAL_SESSION_ENV

    assert cct.TERMINAL_SESSION_ENV == TERMINAL_SESSION_ENV


# --- dispatch: the job's terminal session, never a posture session -----------


def test_a_dispatch_job_carries_the_session_without_a_posture():
    """A dispatch job's ledger ``session`` and OTLP resource carry ``dispatch-<run id>``.

    Run in a child process so the posture and envelope modules read exactly the
    environment a dispatched job is spawned with: ``OSPREY_TERMINAL_SESSION`` set
    by the worker, no posture pair.
    """
    inherited = _record_env()
    job_env = terminal_session_env(f"{cct.DISPATCH_SESSION_PREFIX}run-7", inherited)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OSPREY_POSTURE_SESSION", "OSPREY_POSTURE_SOURCE")
    }
    env.update(job_env)
    probe = (
        "import json\n"
        "from osprey.audit import posture\n"
        "from osprey.audit.envelope import terminal_session\n"
        "print(json.dumps([posture.posture_session(), terminal_session()]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    posture_session, ledger_session = json.loads(out.stdout.strip().splitlines()[-1])
    assert posture_session is None
    assert ledger_session == "dispatch-run-7"
    assert job_env["OTEL_RESOURCE_ATTRIBUTES"] == "terminal.session=dispatch-run-7"


# --- the session is carried into the sandbox and never pinnable --------------


def test_the_sandbox_child_keeps_the_terminal_session_and_drops_its_neighbours():
    from osprey.mcp_server.sandbox_env import (
        SANDBOX_CHILD_ENV_ALLOW_NAMES,
        drop_sandbox_denied_env,
    )

    assert "OSPREY_TERMINAL_SESSION" in SANDBOX_CHILD_ENV_ALLOW_NAMES
    kept = drop_sandbox_denied_env(
        {
            "OSPREY_TERMINAL_SESSION": "s",
            "OSPREY_TERMINAL_SESSION_LIFETIME": "60",
            "OSPREY_TERMINAL_SESSION_STORE_DIR": "/x",
        }
    )
    assert kept == {"OSPREY_TERMINAL_SESSION": "s"}


def test_a_server_spec_cannot_pin_the_terminal_session():
    from osprey.registry.mcp import NON_PINNABLE_AUDIT_MARKERS

    assert "OSPREY_TERMINAL_SESSION" in NON_PINNABLE_AUDIT_MARKERS
