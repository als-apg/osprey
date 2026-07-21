"""Tests for the core ``containers`` health category (task 4.4).

Composes the container probe with ``get_runtime_command`` and the ``--version``
subprocess patched, and injects a fake probe so per-service rows need no real
runtime:

- full mode → ``<runtime>_available`` plus one ``container_<svc>`` row per
  ``deployed_services`` entry (composed via the probe);
- no runtime (``RuntimeError``/``FileNotFoundError``) → single ``skip`` row;
- broken runtime (``--version`` fails) → lone ``error`` availability row, no
  per-service probing;
- degraded mode (``config is None``) → availability row plus one
  ``config unavailable`` skip row.
"""

from __future__ import annotations

import asyncio

import pytest

import osprey.health.core.containers as containers_mod
from osprey.health.core.containers import containers
from osprey.health.models import CheckResult, Status


class _FakeProc:
    """Minimal ``asyncio`` subprocess stand-in for the ``--version`` call."""

    def __init__(self, stdout: bytes = b"", returncode: int = 0) -> None:
        self._stdout = stdout
        self.returncode: int | None = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, b""

    def kill(self) -> None:  # pragma: no cover - only hit on timeout
        pass

    async def wait(self) -> int:  # pragma: no cover - only hit on timeout
        return self.returncode or 0


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    runtime: list[str] | None = None,
    version_proc: _FakeProc | None = None,
    version_error: Exception | None = None,
    runtime_error: Exception | None = None,
) -> None:
    """Patch ``get_runtime_command`` and the ``--version`` subprocess."""
    if runtime_error is not None:

        def _raise(*_a: object, **_k: object) -> list[str]:
            raise runtime_error

        monkeypatch.setattr(containers_mod, "get_runtime_command", _raise)
        return

    monkeypatch.setattr(
        containers_mod, "get_runtime_command", lambda *_a, **_k: runtime or ["docker", "compose"]
    )

    async def _fake_exec(*_a: object, **_k: object) -> _FakeProc:
        if version_error is not None:
            raise version_error
        assert version_proc is not None
        return version_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)


def _echo_probe(captured: list[dict] | None = None):
    """A fake container probe that echoes its spec into an ``ok`` result."""

    async def probe(spec, ctx):  # noqa: ANN001, ANN202 - test double
        if captured is not None:
            captured.append(dict(spec))
        return CheckResult(
            spec["name"],
            spec["category"],
            Status.OK,
            f"{spec['container']}: running",
            value="running",
        )

    return probe


async def _boom_probe(spec, ctx):  # noqa: ANN001, ANN202 - test double
    raise AssertionError("container probe should not be called")


async def _rows(callable_) -> dict[str, CheckResult]:
    results = await callable_()
    assert isinstance(results, list)
    return {r.name: r for r in results}


async def test_factory_returns_awaitable_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, version_proc=_FakeProc(b"Docker version 27.0.0"))
    category = containers({"deployed_services": []}, probe=_boom_probe)
    results = await category()
    assert isinstance(results, list)


async def test_full_mode_emits_available_and_per_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, version_proc=_FakeProc(b"Docker version 27.0.0"))
    config = {"deployed_services": ["openobserve", "web"]}
    results = await containers(config, probe=_echo_probe())()

    names = [r.name for r in results]
    assert names == ["docker_available", "container_openobserve", "container_web"]
    assert results[0].status is Status.OK
    assert "Docker version 27.0.0" in results[0].message
    assert all(r.category == "containers" for r in results)
    assert results[1].status is Status.OK


async def test_probe_receives_expected_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, version_proc=_FakeProc(b"Docker version 27.0.0"))
    captured: list[dict] = []
    await containers({"deployed_services": ["openobserve"]}, probe=_echo_probe(captured))()

    assert captured == [
        {"container": "openobserve", "name": "container_openobserve", "category": "containers"}
    ]


async def test_no_runtime_single_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, runtime_error=RuntimeError("No container runtime found"))
    by_name = await _rows(containers({"deployed_services": ["web"]}, probe=_boom_probe))

    assert set(by_name) == {"container_runtime"}
    assert by_name["container_runtime"].status is Status.SKIP
    assert by_name["container_runtime"].message == "no container runtime available"


async def test_no_runtime_filenotfound_single_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, runtime_error=FileNotFoundError("docker"))
    by_name = await _rows(containers({"deployed_services": ["web"]}, probe=_boom_probe))

    assert set(by_name) == {"container_runtime"}
    assert by_name["container_runtime"].status is Status.SKIP


async def test_broken_runtime_version_nonzero_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, version_proc=_FakeProc(b"", returncode=1))
    results = await containers({"deployed_services": ["web"]}, probe=_boom_probe)()

    assert len(results) == 1
    assert results[0].name == "docker_available"
    assert results[0].status is Status.ERROR


async def test_broken_runtime_version_missing_binary_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, version_error=FileNotFoundError("docker"))
    results = await containers({"deployed_services": ["web"]}, probe=_boom_probe)()

    assert len(results) == 1
    assert results[0].status is Status.ERROR


async def test_degraded_mode_config_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, version_proc=_FakeProc(b"Docker version 27.0.0"))
    by_name = await _rows(containers(None, probe=_boom_probe))

    assert set(by_name) == {"docker_available", "container_services"}
    assert by_name["docker_available"].status is Status.OK
    assert by_name["container_services"].status is Status.SKIP
    assert by_name["container_services"].message == "config unavailable"


async def test_empty_deployed_services_only_availability_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, version_proc=_FakeProc(b"Docker version 27.0.0"))
    results = await containers({"deployed_services": []}, probe=_boom_probe)()

    assert [r.name for r in results] == ["docker_available"]


async def test_podman_runtime_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(
        monkeypatch,
        runtime=["podman", "compose"],
        version_proc=_FakeProc(b"podman version 5.0.0"),
    )
    by_name = await _rows(containers({"deployed_services": []}, probe=_boom_probe))

    assert set(by_name) == {"podman_available"}
    assert by_name["podman_available"].status is Status.OK
