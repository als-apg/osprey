"""The container runtime is resolved at build, on the rendered config.

``container_runtime: auto`` is a *question*, and every deployment template
ships it as the default. Copying the question into ``build/config.yml`` — the
as-built config every lifecycle verb reads — meant that ``down``, ``reset``,
``restart`` and the port preflight each asked it again, from scratch, on every
invocation. Detection is docker-then-podman with a short probe timeout, so on
a host with both installed one slow ``docker ps`` was enough for a ``down`` to
run under podman, find nothing, exit 0, and report the still-running docker
stack as stopped.

The build answers the question once, with the runtime that served the
build, and writes the answer into the render. A verb that later reads
``docker`` there is pinned to docker, and a docker that does not answer is a
refusal — never a silent switch to the other runtime. An explicitly pinned
value is left as written, and a build on a host where no runtime answers keeps
``auto``: there is nothing to pin to, and such a host cannot start anything
anyway. Every render of one build — the deployment's own, each tier and app,
each image copy — states that same answer, asked once by the deployment's
render.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.cli import build_cmd
from osprey.deployment import runtime_helper
from tests.cli._lifecycle_build import invoke_build, rendered_config

PROFILE = """\
extends: hello-world
name: Runtime Pin
project_name: repo
data: data
provider: anthropic
"""


def _build_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: str = ""):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "data").mkdir(exist_ok=True)
    body = PROFILE + (f"config:\n{override}" if override else "")
    (repo / "profile.yml").write_text(body)
    monkeypatch.chdir(repo)
    result = invoke_build()
    return repo, result


def _rendered_runtime(repo: Path) -> str:
    config = rendered_config(repo)
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

    def _nothing(_config=None):
        raise RuntimeError("No container runtime found.")

    monkeypatch.setattr(runtime_helper, "get_runtime_command", _nothing)

    repo, result = _build_repo(tmp_path, monkeypatch)

    assert result.exit_code == 0, result.output
    assert _rendered_runtime(repo) == "auto"


def _stated_runtimes(repo: Path) -> dict[str, str]:
    """``container_runtime`` of every render under ``build/`` that states one."""
    stated: dict[str, str] = {}
    for path in sorted((repo / "build").rglob("config.yml")):
        config = yaml.safe_load(path.read_text())
        if isinstance(config, dict) and "container_runtime" in config:
            stated[path.relative_to(repo).as_posix()] = config["container_runtime"]
    return stated


@pytest.mark.parametrize(
    ("first", "later"),
    [
        (RuntimeError("No container runtime found."), "docker"),
        ("podman", "docker"),
    ],
    ids=["no-runtime-answers", "podman-answers-first"],
)
def test_every_render_of_one_build_states_the_one_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: object, later: str
):
    """One build asks once; every render it writes states the deployment's answer."""
    calls: list[object] = []

    def _detect(config=None):
        calls.append(config)
        if len(calls) == 1:
            if isinstance(first, Exception):
                raise first
            return [first, "compose"]
        return [later, "compose"]

    monkeypatch.setattr(runtime_helper, "get_runtime_command", _detect)
    monkeypatch.setattr(build_cmd, "_warn_if_deployment_running", lambda *a, **k: None)

    repo, result = _build_repo(tmp_path, monkeypatch)

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    stated = _stated_runtimes(repo)
    assert "build/.image/repo/build/config.yml" in stated
    answer = _rendered_runtime(repo)
    assert answer == ("auto" if isinstance(first, Exception) else first)
    assert stated == dict.fromkeys(stated, answer)
