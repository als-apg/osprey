"""The execution backend is resolved at build, on the rendered config (issue #465).

``execution.execution_method`` is read by every container's python executor
through :func:`osprey.utils.config.resolve_execution_method` — at its FIRST
``execute`` call. A profile's ``config:`` overlay can write any value into the
render, and nothing on the build path used to read it back, so a typo
survived ``osprey build``, ``osprey up`` and MCP startup and surfaced as a
failed tool call inside a deployed container. The build now runs the same
resolver on each render: an unknown backend is refused with the resolver's
message, and a legacy spelling (``local``, ``container``) is written into the
render as the backend that runs, with ``container``'s deprecation warning
raised where the operator is.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.utils import config as config_module

CI_FLAGS = ["--skip-deps", "--skip-lifecycle"]

PROFILE = """\
name: Execution Method
app_template: hello_world
provider: anthropic
config:
{override}"""


@pytest.fixture(autouse=True)
def _reset_container_warning():
    """The resolver warns about ``container`` once per process; start clean."""
    config_module._container_method_warned = False
    yield
    config_module._container_method_warned = False


def _build_repo(tmp_path: Path, monkeypatch, override: str):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "profile.yml").write_text(PROFILE.format(override=override))
    monkeypatch.chdir(repo)
    result = CliRunner().invoke(build, CI_FLAGS)
    return repo, result


def _rendered_method(repo: Path) -> str:
    config = yaml.safe_load((repo / "build" / "config.yml").read_text())
    return config["execution"]["execution_method"]


def test_an_unknown_backend_is_refused_at_build(tmp_path: Path, monkeypatch, caplog):
    with caplog.at_level(logging.ERROR):
        repo, result = _build_repo(tmp_path, monkeypatch, "  execution.execution_method: docker\n")

    assert result.exit_code != 0
    reported = result.output + caplog.text + str(result.exception or "")
    assert "execution.execution_method: 'docker'" in reported
    assert "Expected 'subprocess'" in reported
    assert "Unexpected error" not in reported


def test_container_is_warned_at_build_and_rendered_as_subprocess(
    tmp_path: Path, monkeypatch, caplog
):
    with caplog.at_level(logging.WARNING, logger="CONFIG"):
        repo, result = _build_repo(
            tmp_path, monkeypatch, "  execution.execution_method: container\n"
        )

    assert result.exit_code == 0, result.output
    assert _rendered_method(repo) == "subprocess"
    deprecations = [r for r in caplog.records if "is deprecated" in r.getMessage()]
    assert len(deprecations) == 1, [r.getMessage() for r in deprecations]
    assert "config.yml" in deprecations[0].getMessage()


def test_local_is_rendered_as_subprocess(tmp_path: Path, monkeypatch):
    repo, result = _build_repo(tmp_path, monkeypatch, "  execution.execution_method: local\n")

    assert result.exit_code == 0, result.output
    assert _rendered_method(repo) == "subprocess"
