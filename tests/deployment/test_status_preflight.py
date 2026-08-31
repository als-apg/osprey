"""``osprey status`` reads the pre-flight refusal marker ``osprey web`` writes.

The container half of this contract lives in :mod:`osprey.cli.web_cmd`: on a
pre-flight refusal the web entrypoint writes ``$OSPREY_AUDIT_DIR/preflight-refused``
— line 1 an ISO-8601 UTC timestamp, the rest the findings, each prefixed ``- ``.
Under ``restart: unless-stopped`` the container is then restarted forever, and
the reason is invisible to anyone reading the status table. These tests pin the
read side: a FRESH marker annotates the user's state cell, and everything else
(no marker, a marker older than the container's current incarnation, an
unparseable one) leaves the row exactly as it was.

The container runtime is mocked entirely — ``subprocess.run`` returns canned
``ps``/``volume ls``/``inspect`` output — so nothing here touches a real runtime.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from rich.console import Console

from osprey.cli import styles
from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.cli.styles import osprey_theme
from osprey.cli.web_cmd import PREFLIGHT_REFUSED_MARKER
from osprey.deployment import status_display

_CONFIGS: dict[str, dict] = {}

_STARTED_AT = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)


def _fake_load_project_config(config_path, **kwargs):
    """Stand-in for the shared project-config loader keyed by the path passed in."""
    return _CONFIGS[config_path]


def _base_config(users):
    return {
        "project_name": "demo-project",
        "facility": {"prefix": "dls"},
        "modules": {"web_terminals": {"enabled": True, "users": users}},
    }


def _ps_container(name, state, *, started_at=_STARTED_AT):
    """One ``ps --format json`` record; ``started_at=None`` omits the field.

    Podman's ``ps`` carries ``StartedAt``, docker's does not — omitting it is
    how these tests reach the ``inspect`` fallback.
    """
    record = {
        "Names": [name],
        "Labels": {},
        "State": state,
        "Ports": [],
        "Image": "img",
    }
    if started_at is not None:
        record["StartedAt"] = int(started_at.timestamp())
    return record


def _write_marker(repo_root, user, written, findings):
    """Write a pre-flight refusal marker the way ``osprey web`` writes one."""
    audit_dir = repo_root / "var" / "audit" / user
    audit_dir.mkdir(parents=True, exist_ok=True)
    body = "\n".join([written, *(f"- {finding}" for finding in findings)]) + "\n"
    (audit_dir / PREFLIGHT_REFUSED_MARKER).write_text(body, encoding="utf-8")


@pytest.fixture(autouse=True)
def _patch_config_loader(monkeypatch):
    monkeypatch.setattr(status_display, "load_project_config", _fake_load_project_config)
    yield
    _CONFIGS.clear()


@pytest.fixture(autouse=True)
def rendered(monkeypatch):
    """Capture both renderer streams into one wide recording console."""
    console = Console(record=True, width=200, theme=osprey_theme)

    class _Recording(PhaseReporter):
        # Untyped on purpose: rich's ``Console`` is Any to this repo's mypy
        # settings, so an annotated override trips ``no-any-return``.
        def out(self):
            return console

    previous = install_reporter(_Recording(color=False))
    monkeypatch.setattr(styles, "err_console", console)
    yield console
    install_reporter(previous)


@pytest.fixture
def runtime_calls(monkeypatch):
    """Patch the runtime command builders and ``subprocess.run``.

    ``ps`` answers ``ps_stdout``, ``volume ls`` answers ``""`` (this file is not
    about volumes), and ``inspect`` answers ``inspect_stdout`` — which is the
    fallback path for a runtime whose ``ps`` carries no ``StartedAt``.
    """
    calls: dict[str, object] = {"argvs": [], "ps_stdout": "", "inspect_stdout": ""}

    monkeypatch.setattr(
        status_display,
        "get_ps_command",
        lambda config, all_containers=False: ["docker", "ps", "-a", "--format", "json"],
    )
    monkeypatch.setattr(status_display, "get_runtime_command", lambda config=None: ["docker"])

    class _Result:
        def __init__(self, stdout, returncode=0):
            self.stdout = stdout
            self.returncode = returncode
            self.stderr = ""

    def _fake_run(cmd, capture_output=True, text=True, timeout=10):
        calls["argvs"].append(cmd)  # type: ignore[union-attr]
        if cmd[:2] == ["docker", "ps"]:
            return _Result(calls["ps_stdout"])
        if cmd[:2] == ["docker", "volume"]:
            return _Result("")
        if cmd[:2] == ["docker", "inspect"]:
            return _Result(calls["inspect_stdout"])
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(status_display.subprocess, "run", _fake_run)
    return calls


def _run_status(tmp_path, runtime_calls, containers, users=("alice",)):
    """Render ``show_status`` against a repo rooted at *tmp_path*.

    The legacy verb derives its repo root from the directory holding the config
    it was handed, which is what puts the audit zone at ``tmp_path/var/audit``.
    Output is read off the ``rendered`` console fixture.
    """
    config_path = str(tmp_path / "config.yml")
    _CONFIGS[config_path] = _base_config(list(users))
    runtime_calls["ps_stdout"] = json.dumps(list(containers))
    status_display.show_status(config_path)


# ---------------------------------------------------------------------------
# The annotated cell
# ---------------------------------------------------------------------------


def test_status_annotates_the_state_cell_from_a_fresh_preflight_marker(
    tmp_path, runtime_calls, rendered
):
    """A marker written AFTER the container started explains the restart loop."""
    _write_marker(
        tmp_path,
        "alice",
        (_STARTED_AT + timedelta(seconds=5)).isoformat(),
        ["control writes are armed but no write token was minted"],
    )

    _run_status(tmp_path, runtime_calls, [_ps_container("dls-web-alice", "restarting")])
    output = rendered.export_text()

    assert "(preflight:" in output
    assert "control writes are armed" in output


def test_status_truncates_a_long_preflight_reason_to_keep_the_table_shape(
    tmp_path, runtime_calls, rendered
):
    """A findings line is arbitrarily long; the Container column is not."""
    finding = "the roster names a persona that the rendered compose file does not declare anywhere"
    _write_marker(tmp_path, "alice", (_STARTED_AT + timedelta(seconds=5)).isoformat(), [finding])

    _run_status(tmp_path, runtime_calls, [_ps_container("dls-web-alice", "restarting")])
    output = rendered.export_text()

    assert "(preflight:" in output
    assert "..." in output
    assert finding not in output
    assert finding[:20] in output


def test_status_joins_multiple_preflight_findings_into_one_reason(
    tmp_path, runtime_calls, rendered
):
    """Several findings are one cell, joined with ``; `` and truncated as one."""
    _write_marker(
        tmp_path,
        "alice",
        (_STARTED_AT + timedelta(seconds=5)).isoformat(),
        ["control writes armed with no token", "no archiver URL", "no roster grant"],
    )

    _run_status(tmp_path, runtime_calls, [_ps_container("dls-web-alice", "restarting")])
    output = rendered.export_text()

    assert "control writes armed with no token; no" in output
    assert "..." in output  # the tail is past the cap
    assert "no roster grant" not in output


def test_status_reads_started_at_from_inspect_when_ps_omits_it(tmp_path, runtime_calls, rendered):
    """docker's ``ps`` carries no ``StartedAt``; one ``inspect`` supplies it.

    And only when a marker exists — a deployment with no refusals must not pay
    an inspect per container.
    """
    runtime_calls["inspect_stdout"] = "2026-08-30T12:00:00.123456789Z\n"
    _write_marker(
        tmp_path, "alice", (_STARTED_AT + timedelta(seconds=5)).isoformat(), ["no write token"]
    )

    _run_status(
        tmp_path, runtime_calls, [_ps_container("dls-web-alice", "restarting", started_at=None)]
    )
    output = rendered.export_text()

    assert "(preflight: no write token)" in output
    inspects = [c for c in runtime_calls["argvs"] if c[:2] == ["docker", "inspect"]]
    assert len(inspects) == 1


# ---------------------------------------------------------------------------
# Every other case leaves the row exactly as it was
# ---------------------------------------------------------------------------


def test_status_leaves_the_row_unchanged_when_there_is_no_preflight_marker(
    tmp_path, runtime_calls, rendered
):
    _run_status(tmp_path, runtime_calls, [_ps_container("dls-web-alice", "running")])
    output = rendered.export_text()

    assert "● Running" in output
    assert "preflight" not in output
    # No marker, no inspect: the read side costs nothing on a healthy deployment.
    assert [c for c in runtime_calls["argvs"] if c[:2] == ["docker", "inspect"]] == []


def test_status_ignores_a_preflight_marker_older_than_the_container(
    tmp_path, runtime_calls, rendered
):
    """A marker from a previous incarnation (or a ``--skip-preflight`` launch).

    ``osprey web --skip-preflight`` deliberately leaves a stale marker in place,
    so the container's own start time is what bounds its staleness here.
    """
    _write_marker(
        tmp_path, "alice", (_STARTED_AT - timedelta(hours=3)).isoformat(), ["no write token"]
    )

    _run_status(tmp_path, runtime_calls, [_ps_container("dls-web-alice", "running")])
    output = rendered.export_text()

    assert "● Running" in output
    assert "preflight" not in output


@pytest.mark.parametrize(
    "first_line",
    ["", "   ", "not-a-timestamp", "2026-13-45T99:99:99"],
    ids=["empty", "blank", "garbage", "impossible"],
)
def test_status_ignores_an_unparseable_preflight_marker(
    tmp_path, runtime_calls, rendered, first_line
):
    """A torn or hand-edited marker is treated as no marker, never as a crash."""
    _write_marker(tmp_path, "alice", first_line, ["no write token"])

    _run_status(tmp_path, runtime_calls, [_ps_container("dls-web-alice", "running")])
    output = rendered.export_text()

    assert "● Running" in output
    assert "preflight" not in output


def test_status_leaves_an_uncreated_container_unchanged_despite_a_preflight_marker(
    tmp_path, runtime_calls, rendered
):
    """No container, no incarnation to date the marker against — say nothing."""
    _write_marker(
        tmp_path, "alice", (_STARTED_AT + timedelta(seconds=5)).isoformat(), ["no write token"]
    )

    _run_status(tmp_path, runtime_calls, [])
    output = rendered.export_text()

    assert "Not created" in output
    assert "preflight" not in output


def test_status_survives_an_unreadable_preflight_marker_directory(
    tmp_path, runtime_calls, rendered, monkeypatch
):
    """The marker path must never be able to take ``osprey status`` down."""

    def _boom(*args, **kwargs):
        raise OSError("audit zone is not readable")

    monkeypatch.setattr(status_display, "audit_identity_dir", _boom)

    _run_status(tmp_path, runtime_calls, [_ps_container("dls-web-alice", "running")])
    output = rendered.export_text()

    assert "● Running" in output
    assert "preflight" not in output
