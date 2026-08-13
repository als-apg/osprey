"""Unit tests for the ``osprey sim`` CLI, focused on the ``--now`` anchor.

The ``--now`` / ``OSPREY_SIM_NOW`` passthrough freezes the apply-time anchor T0
so seeded logbook dates are reproducible for documentation screenshots. These
tests assert the CLI threads the parsed anchor into
:func:`osprey.simulation.apply.apply_scenarios` without touching a real DB.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from osprey.cli.sim import sim_group
from tests.cli._lifecycle_build import stub_build

# ``archiver`` is None here for the same reason ``load_config`` returns {}: this
# project declares no stored archive, so the rewrite has nothing to report.
_FAKE_RESULT = SimpleNamespace(active=["nominal"], logbook_seeded=0, archiver=None)


@pytest.fixture
def deployment(lifecycle_repo: Path) -> Path:
    """An exemplar repo with a render, ready for ``sim`` to discover.

    ``sim`` resolves its deployment before it parses anything, so the anchor
    passthrough under test is only reachable from inside a real repo — a bare
    directory with a ``config.yml`` in it is no longer a deployment.
    """
    stub_build(lifecycle_repo)
    return lifecycle_repo


def _invoke(deployment: Path, args, env=None):
    """Run ``sim`` against *deployment* with the parts under test isolated.

    ``load_config`` is stubbed to ``{}`` (no ariel → no purge prompt) and
    ``apply_scenarios`` mocked, so what reaches the assertions is the CLI's own
    anchor handling and nothing else. ``--repo`` rather than a chdir: these
    tests care about the anchor, and naming the repo keeps the discovery step
    from being a second thing that can fail here.
    """
    with (
        patch("osprey.cli.sim.load_config", return_value={}),
        patch("osprey.simulation.apply.apply_scenarios", return_value=_FAKE_RESULT) as apply_mock,
    ):
        result = CliRunner().invoke(sim_group, [*args, "--repo", str(deployment)], env=env or {})
    return result, apply_mock


def test_apply_now_threads_aware_anchor(deployment):
    result, apply_mock = _invoke(
        deployment, ["apply", "nominal", "--yes", "--now", "2024-03-18T12:00:00"]
    )
    assert result.exit_code == 0, result.output
    now = apply_mock.call_args.kwargs["now"]
    assert isinstance(now, datetime)
    assert now.tzinfo is not None, "naive --now must be given the facility timezone"
    assert (now.year, now.month, now.day, now.hour) == (2024, 3, 18, 12)


def test_apply_now_from_env(deployment):
    result, apply_mock = _invoke(
        deployment, ["apply", "nominal", "--yes"], env={"OSPREY_SIM_NOW": "2024-03-18T12:00:00"}
    )
    assert result.exit_code == 0, result.output
    now = apply_mock.call_args.kwargs["now"]
    assert now is not None
    assert (now.year, now.month, now.day, now.hour) == (2024, 3, 18, 12)


def test_apply_flag_overrides_env(deployment):
    result, apply_mock = _invoke(
        deployment,
        ["apply", "nominal", "--yes", "--now", "2025-01-02T03:00:00"],
        env={"OSPREY_SIM_NOW": "2024-03-18T12:00:00"},
    )
    assert result.exit_code == 0, result.output
    now = apply_mock.call_args.kwargs["now"]
    assert (now.year, now.month, now.day, now.hour) == (2025, 1, 2, 3)


def test_apply_without_now_passes_none(deployment):
    result, apply_mock = _invoke(deployment, ["apply", "nominal", "--yes"])
    assert result.exit_code == 0, result.output
    assert apply_mock.call_args.kwargs["now"] is None


def test_apply_now_invalid_iso_errors(deployment):
    result, apply_mock = _invoke(deployment, ["apply", "nominal", "--yes", "--now", "not-a-date"])
    assert result.exit_code != 0
    assert "not valid ISO-8601" in result.output
    apply_mock.assert_not_called()


def test_apply_help_shows_now():
    result = CliRunner().invoke(sim_group, ["apply", "--help"])
    assert result.exit_code == 0
    assert "--now" in result.output
    assert "OSPREY_SIM_NOW" in result.output


class TestPerStoreOptOuts:
    """Each store can be left alone on its own, and ``--no-seed`` covers both.

    Applying a scenario writes to two places — the logbook database and the
    stored archive — and there are real reasons to skip either one: no database
    running, or an archive somebody is mid-way through inspecting. The flags are
    only useful if they actually reach ``apply_scenarios``, which is what these
    pin.
    """

    def test_by_default_both_stores_are_seeded(self, deployment):
        _, apply_mock = _invoke(deployment, ["apply", "nominal", "--yes"])
        assert apply_mock.call_args.kwargs["seed_logbook"] is True
        assert apply_mock.call_args.kwargs["seed_archive"] is True

    def test_no_seed_logbook_leaves_the_archive_alone_too(self, deployment):
        _, apply_mock = _invoke(deployment, ["apply", "nominal", "--yes", "--no-seed-logbook"])
        assert apply_mock.call_args.kwargs["seed_logbook"] is False
        assert apply_mock.call_args.kwargs["seed_archive"] is True

    def test_no_seed_archiver_leaves_the_logbook_alone(self, deployment):
        _, apply_mock = _invoke(deployment, ["apply", "nominal", "--yes", "--no-seed-archiver"])
        assert apply_mock.call_args.kwargs["seed_logbook"] is True
        assert apply_mock.call_args.kwargs["seed_archive"] is False

    def test_no_seed_still_means_neither(self, deployment):
        """The pre-existing flag kept its meaning: telemetry only."""
        _, apply_mock = _invoke(deployment, ["apply", "nominal", "--yes", "--no-seed"])
        assert apply_mock.call_args.kwargs["seed_logbook"] is False
        assert apply_mock.call_args.kwargs["seed_archive"] is False

    def test_the_flags_are_documented(self):
        result = CliRunner().invoke(sim_group, ["apply", "--help"])
        assert "--no-seed-logbook" in result.output
        assert "--no-seed-archiver" in result.output
