"""Tests for the pre-build check that Docker can parse the service Dockerfiles.

Every service Dockerfile uses ``RUN --mount=type=cache``, which only BuildKit
parses. A Docker host whose ``compose build`` falls back to the legacy builder
stops mid-build on a Dockerfile parse error that names neither BuildKit nor the
missing buildx plugin. The check asks the host once, before anything is built,
and refuses with the piece that is missing and how to install it.
"""

from __future__ import annotations

import subprocess

import pytest

from osprey.deployment import container_lifecycle, runtime_helper
from osprey.deployment.runtime_helper import buildkit_missing


def _answer(returncode: int):
    """A stand-in for the runtime probe that records what it was asked."""
    calls: list[list[str]] = []

    def probe(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(list(argv), returncode, stdout="", stderr="")

    return probe, calls


def test_docker_without_buildx_is_refused_with_the_plugin_named(monkeypatch) -> None:
    probe, calls = _answer(1)
    monkeypatch.setattr(runtime_helper, "_await_runtime_answer", probe)

    text = buildkit_missing("docker", {})

    assert calls == [["docker", "buildx", "version"]]
    assert text is not None
    assert "docker-buildx-plugin" in text
    assert "DOCKER_BUILDKIT" in text


def test_docker_with_buildx_passes(monkeypatch) -> None:
    probe, calls = _answer(0)
    monkeypatch.setattr(runtime_helper, "_await_runtime_answer", probe)

    assert buildkit_missing("docker", {}) is None
    assert calls == [["docker", "buildx", "version"]]


def test_podman_needs_no_buildkit_probe(monkeypatch) -> None:
    probe, calls = _answer(1)
    monkeypatch.setattr(runtime_helper, "_await_runtime_answer", probe)

    assert buildkit_missing("podman", {"DOCKER_BUILDKIT": "0"}) is None
    assert calls == []


def test_docker_buildkit_zero_is_refused_by_name(monkeypatch) -> None:
    probe, calls = _answer(0)
    monkeypatch.setattr(runtime_helper, "_await_runtime_answer", probe)

    text = buildkit_missing("docker", {"DOCKER_BUILDKIT": "0"})

    assert text is not None
    assert "DOCKER_BUILDKIT=0" in text
    assert calls == []


def test_docker_buildkit_one_skips_the_probe(monkeypatch) -> None:
    probe, calls = _answer(1)
    monkeypatch.setattr(runtime_helper, "_await_runtime_answer", probe)

    assert buildkit_missing("docker", {"DOCKER_BUILDKIT": "1"}) is None
    assert calls == []


@pytest.mark.parametrize(
    "failure", [subprocess.TimeoutExpired(["docker", "buildx", "version"], 120), OSError("gone")]
)
def test_a_probe_that_cannot_answer_refuses_nothing(monkeypatch, failure) -> None:
    def probe(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(runtime_helper, "_await_runtime_answer", probe)

    assert buildkit_missing("docker", {}) is None


def test_a_building_docker_host_without_buildkit_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(container_lifecycle, "_resolve_prebuilt_images", lambda config: False)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle, "buildkit_missing", lambda runtime, env: f"no buildx on {runtime}"
    )

    with pytest.raises(RuntimeError, match="no buildx on docker"):
        container_lifecycle._preflight_buildkit({})


def test_a_prebuilt_host_is_never_refused(monkeypatch) -> None:
    def never(*args, **kwargs):
        raise AssertionError("a host that builds nothing must not be probed")

    monkeypatch.setattr(container_lifecycle, "_resolve_prebuilt_images", lambda config: True)
    monkeypatch.setattr(container_lifecycle, "get_runtime_command", never)
    monkeypatch.setattr(container_lifecycle, "buildkit_missing", never, raising=False)

    container_lifecycle._preflight_buildkit({})


def test_a_host_with_no_usable_runtime_is_not_this_checks_to_refuse(monkeypatch) -> None:
    def no_runtime(_config):
        raise RuntimeError("no container runtime")

    monkeypatch.setattr(container_lifecycle, "_resolve_prebuilt_images", lambda config: False)
    monkeypatch.setattr(container_lifecycle, "get_runtime_command", no_runtime)

    container_lifecycle._preflight_buildkit({})
