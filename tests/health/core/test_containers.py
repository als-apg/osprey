"""Tests for the core ``containers`` health category.

The category reads the deployment record once and reports on the runtime it
names. The ``--version`` subprocess is patched and a fake probe is injected,
so per-service rows need no real runtime:

- full mode → ``<recorded runtime>_available`` plus one ``container_<svc>``
  row per ``deployed_services`` entry (composed via the probe);
- no record → one warning row carrying :data:`NOTHING_DEPLOYED`, no
  availability row, no probing;
- unreadable record → one warning row naming it unreadable;
- broken runtime (``--version`` fails) → lone ``error`` availability row, no
  per-service probing;
- degraded mode (``config is None``) → availability row plus one
  ``config unavailable`` skip row.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from osprey.deployment.deployment_record import (
    DEPLOYMENT_RECORD_RELPATH,
    NOTHING_DEPLOYED,
    DeploymentRecord,
    write_deployment_record,
)
from osprey.health.core.containers import containers
from osprey.health.models import CheckResult, Status
from tests.health._fakes import FakeProc


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo the category locates: the working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _record(repo: Path, runtime: str = "docker") -> None:
    write_deployment_record(repo, DeploymentRecord(runtime=runtime, compose_project="demo"))


def _config(repo: Path, services: list[str]) -> dict:
    return {"project_root": str(repo), "deployed_services": services}


def _patch_version(
    monkeypatch: pytest.MonkeyPatch,
    *,
    version_proc: FakeProc | None = None,
    version_error: Exception | None = None,
    argv: list[tuple] | None = None,
) -> None:
    """Patch the ``--version`` subprocess, recording each argv it is given."""

    async def _fake_exec(*a: object, **_k: object) -> FakeProc:
        if argv is not None:
            argv.append(a)
        if version_error is not None:
            raise version_error
        assert version_proc is not None
        return version_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)


def _echo_probe(captured: list[dict] | None = None):
    """A fake container probe that echoes its spec into an ``ok`` result."""

    async def probe(spec, _ctx):  # test double
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


async def _boom_probe(_spec, _ctx):  # test double
    raise AssertionError("container probe should not be called")


async def _rows(callable_) -> dict[str, CheckResult]:
    results = await callable_()
    assert isinstance(results, list)
    return {r.name: r for r in results}


async def test_factory_returns_awaitable_callable(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record(repo)
    _patch_version(monkeypatch, version_proc=FakeProc(b"Docker version 27.0.0"))
    category = containers(_config(repo, []), probe=_boom_probe)
    results = await category()
    assert isinstance(results, list)


async def test_full_mode_emits_available_and_per_service(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record(repo)
    _patch_version(monkeypatch, version_proc=FakeProc(b"Docker version 27.0.0"))
    results = await containers(_config(repo, ["postgresql", "web"]), probe=_echo_probe())()

    names = [r.name for r in results]
    assert names == ["docker_available", "container_postgresql", "container_web"]
    assert results[0].status is Status.OK
    assert "Docker version 27.0.0" in results[0].message
    assert all(r.category == "containers" for r in results)
    assert results[1].status is Status.OK


async def test_probe_receives_expected_spec(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _record(repo)
    _patch_version(monkeypatch, version_proc=FakeProc(b"Docker version 27.0.0"))
    captured: list[dict] = []
    await containers(_config(repo, ["postgresql"]), probe=_echo_probe(captured))()

    assert captured == [
        {
            "container": "postgresql",
            "name": "container_postgresql",
            "category": "containers",
        }
    ]


async def test_the_recorded_runtime_names_the_availability_row(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The setting selects docker; the record says podman, and podman is asked."""
    _record(repo, runtime="podman")
    monkeypatch.setenv("CONTAINER_RUNTIME", "docker")
    argv: list[tuple] = []
    _patch_version(monkeypatch, version_proc=FakeProc(b"podman version 5.0.0"), argv=argv)
    config = {**_config(repo, []), "container_runtime": "docker"}
    by_name = await _rows(containers(config, probe=_boom_probe))

    assert set(by_name) == {"podman_available"}
    assert by_name["podman_available"].status is Status.OK
    assert [a[0] for a in argv] == ["podman"]


async def test_no_record_is_one_warning_row_and_probes_nothing(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv: list[tuple] = []
    _patch_version(monkeypatch, version_proc=FakeProc(b"Docker version 27.0.0"), argv=argv)
    results = await containers(_config(repo, ["web"]), probe=_boom_probe)()

    assert len(results) == 1
    assert results[0].status is Status.WARNING
    assert results[0].message == NOTHING_DEPLOYED
    assert argv == []


async def test_an_unreadable_record_is_one_row_naming_it(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = repo / DEPLOYMENT_RECORD_RELPATH
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    argv: list[tuple] = []
    _patch_version(monkeypatch, version_proc=FakeProc(b"Docker version 27.0.0"), argv=argv)
    results = await containers(_config(repo, ["web"]), probe=_boom_probe)()

    assert len(results) == 1
    assert results[0].status is Status.WARNING
    assert results[0].message == "the deployment record is unreadable"
    assert str(path) in results[0].details
    assert argv == []


async def test_broken_runtime_version_nonzero_is_error(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record(repo)
    _patch_version(monkeypatch, version_proc=FakeProc(b"", returncode=1))
    results = await containers(_config(repo, ["web"]), probe=_boom_probe)()

    assert len(results) == 1
    assert results[0].name == "docker_available"
    assert results[0].status is Status.ERROR


async def test_broken_runtime_version_missing_binary_is_error(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record(repo)
    _patch_version(monkeypatch, version_error=FileNotFoundError("docker"))
    results = await containers(_config(repo, ["web"]), probe=_boom_probe)()

    assert len(results) == 1
    assert results[0].status is Status.ERROR


async def test_degraded_mode_config_none(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _record(repo)
    _patch_version(monkeypatch, version_proc=FakeProc(b"Docker version 27.0.0"))
    by_name = await _rows(containers(None, probe=_boom_probe))

    assert set(by_name) == {"docker_available", "container_services"}
    assert by_name["docker_available"].status is Status.OK
    assert by_name["container_services"].status is Status.SKIP
    assert by_name["container_services"].message == "config unavailable"


async def test_empty_deployed_services_only_availability_row(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record(repo)
    _patch_version(monkeypatch, version_proc=FakeProc(b"Docker version 27.0.0"))
    results = await containers(_config(repo, []), probe=_boom_probe)()

    assert [r.name for r in results] == ["docker_available"]
