"""``osprey up`` and ``osprey restart`` over another copy's containers print the refusal.

The decision is pinned in ``tests/deployment/test_container_lifecycle.py``; this
pins what the operator reads: the shared refusal named for the verb they typed,
the way to stop the other copy, the overlay that gives this copy its own name,
and what was left unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.deployment.compose_generator import REPO_ID_LABEL
from osprey.deployment.container_ownership import (
    COMPOSE_PROJECT_LABEL,
    START_REFUSAL,
    Resource,
    host_claim,
)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo whose start verbs are stubbed down to their refusal."""
    from osprey.cli import deploy_cmd, repo_resolver

    monkeypatch.setattr(repo_resolver, "find_repo_root", lambda start=None: tmp_path / "here")
    (tmp_path / "here").mkdir()
    monkeypatch.setattr(deploy_cmd, "gate_start_from_build", lambda *a, **k: None)
    monkeypatch.setattr(deploy_cmd, "_preflight", lambda *a, **k: None)
    return tmp_path


def _refusal(other: Path):
    """The error ``_start_stack`` raises when *other* runs two containers of the name."""
    from osprey.deployment.container_lifecycle import _own_name_recipe

    labels = {
        COMPOSE_PROJECT_LABEL: "uitf",
        REPO_ID_LABEL: "0123456789ab",
        "com.docker.compose.project.working_dir": str(other),
    }

    class _Probe:
        runtime = "docker"

        def containers_for_project(self, project, *, include_stopped=True):  # noqa: ARG002 - RuntimeProbe's signature
            return [Resource("container", f"uitf-{n}", labels) for n in ("a", "b")]

        def volumes_for_project(self, project):  # noqa: ARG002 - RuntimeProbe's signature
            return [Resource("volume", "uitf_data", {COMPOSE_PROJECT_LABEL: "uitf"})]

    claim = host_claim("uitf", "aaaaaaaaaaaa", probe=_Probe())
    rename = None if claim.other_copy_gone else _own_name_recipe("uitf")
    return claim.refusal(START_REFUSAL, extra_remedy=rename)


def _flowed(text: str) -> str:
    return " ".join(text.split())


def test_up_prints_the_refusal_and_says_nothing_was_deployed(repo, monkeypatch):
    from osprey.deployment import container_lifecycle

    other = repo / "uitf"
    other.mkdir()

    def refuse(*args, **kwargs):
        raise _refusal(other)

    monkeypatch.setattr(container_lifecycle, "up_as_built", refuse)

    result = CliRunner().invoke(cli, ["up", "-d"])

    assert result.exit_code != 0
    flowed = _flowed(result.output)
    assert "osprey up will not start over containers from another copy of this repo" in flowed
    assert "2 containers are named 'uitf'" in flowed
    assert f"{other} (still on disk)" in flowed
    assert f"stop that deployment where it lives: `osprey down --repo {other}`" in flowed
    assert "or give this copy its own name:" in flowed
    assert "profiles/scratch.yml: project_name: uitf-scratch" in flowed
    assert ".env.variant: OSPREY_PROFILE_VARIANT=scratch" in flowed
    assert "then `osprey build` and start again. Nothing was deployed." in flowed
    assert "Traceback" not in result.output


def test_restart_prints_the_refusal_and_says_nothing_was_stopped(repo, monkeypatch):
    from osprey.deployment import container_lifecycle

    def refuse(*args, **kwargs):
        raise _refusal(repo / "gone")

    monkeypatch.setattr(container_lifecycle, "restart_deployment", refuse)

    result = CliRunner().invoke(cli, ["restart", "-d"])

    assert result.exit_code != 0
    flowed = _flowed(result.output)
    assert "osprey restart will not start over containers from another copy" in flowed
    assert "no such directory on this host now" in flowed
    assert (
        "→ the copy they came from is no longer on this host, so remove its containers: "
        "`docker rm -f uitf-a uitf-b`. The project's volumes are untouched and keep their "
        "data. Nothing was stopped."
    ) in flowed
    assert "give this copy its own name" not in flowed
