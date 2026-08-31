"""The pre-flight refusal marker ``osprey web`` leaves in the per-user audit dir.

Under container supervision (``restart: unless-stopped``) a pre-flight refusal
is invisible: the container exits 1, the runtime restarts it, it refuses again,
and from the outside all anyone sees is a service flapping. The marker is the
only surface that says *why* — ``osprey status`` reads it to render a
"restarting (pre-flight: ...)" row instead of a bare restart count.

The marker is deliberately advisory: an absent or unwritable audit directory
must never turn a pre-flight refusal into a second, different failure, and a
bare laptop launch (no ``OSPREY_AUDIT_DIR``, no supervisor) writes nothing at
all.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from osprey.cli.web_cmd import AUDIT_DIR_ENV, PREFLIGHT_REFUSED_MARKER, web
from tests.cli._lifecycle_build import stub_build

#: One refusal finding, in the shape ``_preflight`` actually returns them.
FINDING = "port 8081 is already in use by another process (artifact server)"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def deployment(lifecycle_repo: Path) -> Path:
    """An exemplar repo with a render — the shape every launch test needs."""
    stub_build(lifecycle_repo, config="web: {}\n")
    return lifecycle_repo


@pytest.fixture(autouse=True)
def _isolate_launch_env(monkeypatch):
    """Roll back every env key a launch writes directly into ``os.environ``.

    ``web()`` writes ``OSPREY_CONFIG``, ``OSPREY_WEB_PORT`` and the operator
    secret itself, not through monkeypatch, so a plain ``delenv`` on an
    already-absent key would record no undo entry and let those writes leak
    into later tests. setenv-then-delenv forces the undo entry (same trick as
    ``test_web_cmd.py``); ``AUDIT_DIR_ENV`` joins the list so no ambient
    export from the developer's shell decides where a marker lands.
    """
    from osprey.interfaces.web_auth import OPERATOR_SECRET_ENV, reset_web_credentials

    for key in ("OSPREY_CONFIG", "OSPREY_WEB_PORT", OPERATOR_SECRET_ENV, AUDIT_DIR_ENV):
        monkeypatch.setenv(key, "__unset_by_test_fixture__")
        monkeypatch.delenv(key)
    reset_web_credentials()
    yield
    reset_web_credentials()


def _stub_launch(monkeypatch) -> None:
    """Stop a passing pre-flight from starting a real server."""
    monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", lambda **_kw: None)
    monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)


def _invoke(runner, deployment, *extra):
    return runner.invoke(
        web,
        ["--repo", str(deployment), "--shell", "true", *extra],
        catch_exceptions=False,
    )


def _seed_marker(audit_dir: Path) -> Path:
    """Leave a marker from a previous, already-answered refusal."""
    marker = audit_dir / PREFLIGHT_REFUSED_MARKER
    marker.write_text("2000-01-01T00:00:00+00:00\n- a stale finding\n", encoding="utf-8")
    return marker


class TestPreflightRefusalWritesTheMarker:
    """A refusal records itself where the supervisor's reader can find it."""

    @patch("osprey.cli.web_cmd._preflight", return_value=([FINDING], []))
    def test_marker_carries_a_utc_timestamp_then_the_refusal_findings(
        self, _mock_preflight, runner, monkeypatch, deployment, tmp_path
    ):
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        monkeypatch.setenv(AUDIT_DIR_ENV, str(audit_dir))

        result = _invoke(runner, deployment)

        assert result.exit_code == 1
        assert "Pre-flight checks failed" in result.output

        marker = audit_dir / PREFLIGHT_REFUSED_MARKER
        first_line, _, body = marker.read_text(encoding="utf-8").partition("\n")
        stamp = datetime.fromisoformat(first_line)
        assert stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0
        assert FINDING in body

    @patch("osprey.cli.web_cmd._preflight", return_value=([FINDING], []))
    def test_a_second_refusal_refreshes_the_marker_rather_than_appending(
        self, _mock_preflight, runner, monkeypatch, deployment, tmp_path
    ):
        """Each supervised restart re-runs pre-flight, so the marker must read
        as *this* attempt, not as a growing pile of every attempt."""
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        monkeypatch.setenv(AUDIT_DIR_ENV, str(audit_dir))
        stale = _seed_marker(audit_dir)

        assert _invoke(runner, deployment).exit_code == 1

        text = stale.read_text(encoding="utf-8")
        assert "2000-01-01" not in text
        assert text.count(FINDING) == 1


class TestPreflightMarkerLifecycle:
    """Who clears the marker, and — deliberately — who does not."""

    @patch("osprey.cli.web_cmd._preflight", return_value=([], []))
    def test_a_passing_preflight_clears_a_stale_marker(
        self, _mock_preflight, runner, monkeypatch, deployment, tmp_path
    ):
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        monkeypatch.setenv(AUDIT_DIR_ENV, str(audit_dir))
        stale = _seed_marker(audit_dir)
        _stub_launch(monkeypatch)

        assert _invoke(runner, deployment).exit_code == 0
        assert not stale.exists()

    def test_skip_preflight_leaves_the_marker_alone(
        self, runner, monkeypatch, deployment, tmp_path
    ):
        """``--skip-preflight`` forces past a refusal it never re-ran. Clearing
        the marker there would erase the evidence of the very refusal the
        operator is overriding, so the record stands until a real pre-flight
        passes."""
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        monkeypatch.setenv(AUDIT_DIR_ENV, str(audit_dir))
        stale = _seed_marker(audit_dir)
        before = stale.read_text(encoding="utf-8")
        _stub_launch(monkeypatch)

        assert _invoke(runner, deployment, "--skip-preflight").exit_code == 0
        assert stale.read_text(encoding="utf-8") == before


class TestPreflightMarkerNeverBecomesASecondFailure:
    """The marker is advisory: it may be skipped, never escalated."""

    @patch("osprey.cli.web_cmd._preflight", return_value=([FINDING], []))
    def test_unwritable_audit_dir_still_reports_the_refusal_and_exits_one(
        self, _mock_preflight, runner, monkeypatch, deployment, tmp_path
    ):
        """A regular file where the audit directory should be — the cheapest
        deterministic stand-in for a root-owned or read-only mount."""
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("", encoding="utf-8")
        monkeypatch.setenv(AUDIT_DIR_ENV, str(blocked / "alice"))

        result = _invoke(runner, deployment)

        assert result.exit_code == 1
        assert "Pre-flight checks failed" in result.output
        assert FINDING in result.output

    @patch("osprey.cli.web_cmd._preflight", return_value=([FINDING], []))
    def test_no_audit_dir_env_writes_no_marker_anywhere(
        self, _mock_preflight, runner, monkeypatch, deployment, tmp_path
    ):
        """A bare ``osprey web`` on a laptop has no supervisor to explain the
        refusal to, and no per-user audit directory to explain it in."""
        monkeypatch.chdir(tmp_path)

        result = _invoke(runner, deployment)

        assert result.exit_code == 1
        assert "Pre-flight checks failed" in result.output
        assert list(tmp_path.rglob(PREFLIGHT_REFUSED_MARKER)) == []
        assert list(deployment.rglob(PREFLIGHT_REFUSED_MARKER)) == []
