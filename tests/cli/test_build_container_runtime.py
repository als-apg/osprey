"""The container runtime is resolved at build, on the rendered config.

``container_runtime: auto`` is a *question*, and every deployment template
ships it as the default. Copying the question into ``build/config.yml`` — the
as-built config every lifecycle verb reads — meant that ``down``, ``reset``,
``restart`` and the port preflight each asked it again, from scratch, on every
invocation. Detection is docker-then-podman with a short probe timeout, so on
a host with both installed one slow ``docker ps`` was enough for a ``down`` to
run under podman, find nothing, exit 0, and report the still-running docker
stack as stopped.

The build now answers the question once, with the runtime that served the
build, and writes the answer into the render. A verb that later reads
``docker`` there is pinned to docker, and a docker that does not answer is a
refusal — never a silent switch to the other runtime. An explicitly pinned
value is left as written, and a build on a host where no runtime answers keeps
``auto``: there is nothing to pin to, and such a host cannot start anything
anyway.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.deployment import runtime_helper

CI_FLAGS = ["--skip-deps", "--skip-lifecycle"]

PROFILE = """\
name: Runtime Pin
app_template: hello_world
provider: anthropic
config:
{override}"""


def _build_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: str = ""):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "profile.yml").write_text(PROFILE.format(override=override))
    monkeypatch.chdir(repo)
    result = CliRunner().invoke(build, CI_FLAGS)
    return repo, result


def _rendered_runtime(repo: Path) -> str:
    config = yaml.safe_load((repo / "build" / "config.yml").read_text())
    return config["container_runtime"]


def _fake_detection(monkeypatch: pytest.MonkeyPatch, runtime: str) -> None:
    """Answer every detection with ``runtime``."""
    monkeypatch.setattr(
        runtime_helper, "get_runtime_command", lambda config=None: [runtime, "compose"]
    )


def test_auto_is_resolved_into_the_runtime_that_answered(tmp_path: Path, monkeypatch):
    _fake_detection(monkeypatch, "podman")

    repo, result = _build_repo(tmp_path, monkeypatch)

    assert result.exit_code == 0, result.output
    assert _rendered_runtime(repo) == "podman"


def test_a_pinned_runtime_is_written_as_pinned(tmp_path: Path, monkeypatch):
    """An explicit pin is the operator's answer; detection cannot overrule it."""
    _fake_detection(monkeypatch, "podman")

    repo, result = _build_repo(tmp_path, monkeypatch, "  container_runtime: docker\n")

    assert result.exit_code == 0, result.output
    assert _rendered_runtime(repo) == "docker"


def test_no_runtime_answering_leaves_auto_in_place(tmp_path: Path, monkeypatch):
    """A host with no runtime can still build; the question stays open."""

    def _nothing(config=None):
        raise RuntimeError("No container runtime found.")

    monkeypatch.setattr(runtime_helper, "get_runtime_command", _nothing)

    repo, result = _build_repo(tmp_path, monkeypatch)

    assert result.exit_code == 0, result.output
    assert _rendered_runtime(repo) == "auto"
