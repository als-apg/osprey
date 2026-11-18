"""A start that came up short of one step does not end as a success.

Seeding the graph store is deliberately not a reason to deny an operator the
rest of the stack, so a failed seed lets the start go on. Before this, that was
also where it ended: one log record, exit code 0, and a store with nothing in
it. The unit half lives in ``tests/deployment/test_stage_graphdb_store.py``;
this pins the verbs: the stack is reported as started, the summary card still
prints, the last thing printed names the failed step with its cause and remedy,
and the exit code is not zero.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.deployment import runtime_helper
from osprey.deployment.errors import DeployStepFailedError

pytestmark = pytest.mark.usefixtures("container_runtime_is_up", "repo")

STEP = "Seeding the graph store"
CAUSE = "importing data/graph/facility.ttl failed (KO: Java heap space)"
REMEDY = "Fix the cause, then run `osprey build && osprey up`."
CARD = "the summary card"


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo with a render, every runtime seam stubbed out."""
    from osprey.cli import deploy_cmd, repo_resolver, summary_card
    from osprey.cli import output as cli_output
    from osprey.deployment import container_lifecycle

    build = tmp_path / "build"
    build.mkdir()
    (build / "config.yml").write_text("project_name: demo\n", encoding="utf-8")

    monkeypatch.setattr(repo_resolver, "find_repo_root", lambda start=None: tmp_path)
    monkeypatch.setattr(deploy_cmd, "gate_start_from_build", lambda *a, **k: None)
    monkeypatch.setattr(deploy_cmd, "ensure_repo_env", lambda *a, **k: None)
    monkeypatch.setattr(deploy_cmd, "load_project_config", lambda *a, **k: {})
    monkeypatch.setattr(
        container_lifecycle, "as_built_config_path", lambda root: tmp_path / "build" / "config.yml"
    )
    monkeypatch.setattr(
        runtime_helper, "get_runtime_command", lambda config=None: ["docker", "compose"]
    )
    monkeypatch.setattr(
        summary_card, "print_summary_card", lambda root, state: cli_output.report(CARD)
    )
    return tmp_path


def _comes_up_short(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make the start function ``name`` bring the stack up and raise the step."""
    from osprey.deployment import container_lifecycle

    def start(*args, **kwargs):
        raise DeployStepFailedError(STEP, CAUSE, REMEDY)

    monkeypatch.setattr(container_lifecycle, name, start)


@pytest.mark.parametrize(
    ("verb", "start", "phase"),
    [
        ("up", "up_as_built", "Starting"),
        ("restart", "restart_deployment", "Restarting"),
    ],
)
def test_the_run_ends_on_the_failed_step_and_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch, verb: str, start: str, phase: str
) -> None:
    _comes_up_short(monkeypatch, start)

    result = CliRunner().invoke(cli, [verb, "-d"])

    assert result.exit_code == 1
    lines = [line for line in result.output.splitlines() if line.strip()]
    # The stack did start: the phase closes as done, and the card still prints.
    assert any(line.lstrip().startswith(f"✓ {phase}") for line in lines)
    assert CARD in lines
    # The run ends on the step: what failed, why, and what to do about it.
    assert lines[-3:] == [
        f"✗ {STEP} failed. The rest of the deployment is running.",
        f"  {CAUSE}",
        f"  → {REMEDY}",
    ]


def test_a_failed_step_is_not_reported_as_a_failed_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing was refused and nothing was torn down, so neither the refusal
    block nor click's own abort line belongs in the output."""
    _comes_up_short(monkeypatch, "up_as_built")

    result = CliRunner().invoke(cli, ["up", "-d"])

    assert "Deployment failed" not in result.output
    assert "Aborted" not in result.output
    assert "✗ Starting" not in result.output


def test_a_start_with_no_failed_step_still_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osprey.deployment import container_lifecycle

    monkeypatch.setattr(container_lifecycle, "up_as_built", lambda *a, **k: None)

    result = CliRunner().invoke(cli, ["up", "-d"])

    assert result.exit_code == 0
    assert "✗" not in result.output
