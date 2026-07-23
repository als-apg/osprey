"""Tests for the ``container`` health probe (task 2.3).

Drives :func:`osprey.health.probes.container.run` with
:mod:`osprey.deployment.runtime_helper.get_ps_command` and
:func:`asyncio.create_subprocess_exec` patched, so no container runtime is
touched:

- a matching ``running`` container → ``ok`` (healthy note carried through);
- ``running`` but ``unhealthy`` → ``warning``;
- any other state (e.g. ``exited``) → ``warning`` carrying the state;
- no matching container → ``warning`` "not deployed";
- no container runtime (``RuntimeError``/``FileNotFoundError``) → ``skip``;
- fuzzy underscore/hyphen short-name matching, Docker NDJSON + Podman array
  ``ps`` output, non-zero ``ps`` exit, and query timeout.

Also exercises the lazy registry (:func:`osprey.health.probes.get_probe`).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from osprey.health.models import Status
from osprey.health.probes import ProbeContext, get_probe
from osprey.health.probes import container as container_mod
from osprey.health.probes.container import run
from osprey.health.runtime import HealthRuntime


def _ctx() -> ProbeContext:
    """A probe context with a never-constructed runtime; container ignores it."""
    return ProbeContext(runtime=HealthRuntime({}))


class _FakeProc:
    """Minimal stand-in for an ``asyncio`` subprocess used by the probe."""

    def __init__(self, stdout: bytes = b"", returncode: int = 0, hang: bool = False) -> None:
        self._stdout = stdout
        self.returncode: int | None = returncode
        self._hang = hang
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            await asyncio.sleep(10)
        return self._stdout, b""

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self.returncode or 0


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    proc: _FakeProc | None = None,
    ps_error: Exception | None = None,
) -> None:
    """Patch ``get_ps_command`` and ``create_subprocess_exec`` for one test."""
    if ps_error is not None:

        def _raise(*_a: object, **_k: object) -> list[str]:
            raise ps_error

        monkeypatch.setattr(container_mod, "get_ps_command", _raise)
        return

    monkeypatch.setattr(
        container_mod,
        "get_ps_command",
        lambda *_a, **_k: ["docker", "ps", "-a", "--format", "json"],
    )

    async def _fake_exec(*_a: object, **_k: object) -> _FakeProc:
        assert proc is not None
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)


def _ndjson(*containers: dict) -> bytes:
    """Encode containers as Docker-style newline-delimited JSON."""
    return ("\n".join(json.dumps(c) for c in containers) + "\n").encode()


async def test_running_container_is_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc(
        stdout=json.dumps(
            [{"Names": ["openobserve"], "State": "running", "Status": "Up 2 hours (healthy)"}]
        ).encode()
    )
    _patch_runtime(monkeypatch, proc=proc)

    result = await run({"container": "openobserve"}, _ctx())

    assert result.status is Status.OK
    assert result.value == "running"
    assert "healthy" in result.message
    assert result.latency_ms > 0
    assert result.name == "container.openobserve"
    assert result.category == "containers"


async def test_running_but_unhealthy_is_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc(
        stdout=json.dumps(
            [{"Names": ["web"], "State": "running", "Status": "Up 5 minutes (unhealthy)"}]
        ).encode()
    )
    _patch_runtime(monkeypatch, proc=proc)

    result = await run({"container": "web"}, _ctx())

    assert result.status is Status.WARNING
    assert result.value == "running"
    assert "unhealthy" in result.message


async def test_stopped_container_is_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc(stdout=json.dumps([{"Names": ["openobserve"], "State": "exited"}]).encode())
    _patch_runtime(monkeypatch, proc=proc)

    result = await run({"container": "openobserve"}, _ctx())

    assert result.status is Status.WARNING
    assert result.value == "exited"
    assert "exited" in result.message


async def test_missing_container_is_warning_not_deployed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _FakeProc(
        stdout=json.dumps([{"Names": ["something-else"], "State": "running"}]).encode()
    )
    _patch_runtime(monkeypatch, proc=proc)

    result = await run({"container": "openobserve"}, _ctx())

    assert result.status is Status.WARNING
    assert result.value == "not found"
    assert "not deployed" in result.message


async def test_no_runtime_runtimeerror_is_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime(monkeypatch, ps_error=RuntimeError("No container runtime found"))

    result = await run({"container": "openobserve"}, _ctx())

    assert result.status is Status.SKIP
    assert result.message == "no container runtime available"


async def test_no_runtime_filenotfound_is_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime(monkeypatch, ps_error=FileNotFoundError("docker"))

    result = await run({"container": "openobserve"}, _ctx())

    assert result.status is Status.SKIP
    assert result.message == "no container runtime available"


async def test_docker_ndjson_output_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    # Docker emits one JSON object per line (not a JSON array). Names is a string.
    proc = _FakeProc(
        stdout=_ndjson(
            {"Names": "proj-postgres-1", "State": "exited"},
            {"Names": "proj-openobserve-1", "State": "running", "Status": "Up (healthy)"},
        )
    )
    _patch_runtime(monkeypatch, proc=proc)

    result = await run({"container": "openobserve"}, _ctx())

    assert result.status is Status.OK
    assert result.value == "running"


async def test_fuzzy_underscore_hyphen_match(monkeypatch: pytest.MonkeyPatch) -> None:
    # Dotted service name, hyphen in target, underscore in the container name.
    proc = _FakeProc(stdout=_ndjson({"Names": "myproj_python_executor_1", "State": "running"}))
    _patch_runtime(monkeypatch, proc=proc)

    result = await run({"container": "osprey.python-executor"}, _ctx())

    assert result.status is Status.OK
    assert result.name == "container.python-executor"


async def test_ps_nonzero_exit_is_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc(stdout=b"", returncode=1)
    _patch_runtime(monkeypatch, proc=proc)

    result = await run({"container": "openobserve"}, _ctx())

    assert result.status is Status.WARNING
    assert "failed" in result.message


async def test_query_timeout_is_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc(hang=True)
    _patch_runtime(monkeypatch, proc=proc)

    result = await run({"container": "openobserve", "timeout_s": 0.05}, _ctx())

    assert result.status is Status.WARNING
    assert "timed out" in result.message
    assert proc.killed is True


async def test_custom_name_and_category_flow_through(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc(stdout=json.dumps([{"Names": ["foo"], "State": "running"}]).encode())
    _patch_runtime(monkeypatch, proc=proc)

    result = await run({"container": "foo", "name": "infra.foo", "category": "infra"}, _ctx())

    assert result.name == "infra.foo"
    assert result.category == "infra"


async def test_service_alias_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc(stdout=json.dumps([{"Names": ["openobserve"], "State": "running"}]).encode())
    _patch_runtime(monkeypatch, proc=proc)

    result = await run({"service": "openobserve"}, _ctx())

    assert result.status is Status.OK


async def test_missing_container_name_is_error() -> None:
    result = await run({}, _ctx())

    assert result.status is Status.ERROR
    assert "container" in result.message


async def test_get_probe_resolves_container_lazily() -> None:
    probe = get_probe("container")
    assert probe is run
