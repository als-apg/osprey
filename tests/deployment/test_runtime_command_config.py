"""``container_runtime`` must survive memoization and reach the health surface.

``get_runtime_command`` memoizes its docker-vs-podman detection. Memoizing a
single answer per process made the first caller authoritative: a config-less
call (health used to make two of them) cached ``docker``, and every later call
— including one explicitly configured for podman — was served that cached
answer. Health then reported on a runtime the project never selected.

These tests pin both halves of the fix:

* the memo is keyed on the inputs that determine it (``CONTAINER_RUNTIME`` +
  ``config["container_runtime"]``), so no call can poison another — while calls
  with *identical* inputs are still served from cache;
* both health call sites thread the loaded config through — the ``containers``
  category into ``get_runtime_command``, and the ``container`` probe into
  ``get_ps_command`` via ``ProbeContext.config``.

No test here touches a real docker/podman: ``shutil.which`` and
``subprocess.run`` are faked, and the health tests patch the runtime helpers out
entirely.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any

import pytest

import osprey.health.core.containers as containers_mod
import osprey.health.probes.container as container_probe_mod
from osprey.deployment import runtime_helper
from osprey.deployment.runtime_helper import get_runtime_command, reset_runtime_cache
from osprey.health.core.containers import containers
from osprey.health.models import CheckResult, Status
from osprey.health.probes import ProbeContext
from osprey.health.runtime import HealthRuntime


@pytest.fixture(autouse=True)
def clear_container_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop any host ``CONTAINER_RUNTIME`` override so detection is deterministic."""
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)


def _fake_probe(monkeypatch: pytest.MonkeyPatch, healthy: set[str]) -> list[list[str]]:
    """Fake ``which``/``run`` so only ``healthy`` runtimes detect; record probes.

    Every runtime is on PATH; a runtime in ``healthy`` exits 0 for both its
    ``compose version`` and ``ps`` probe, and any other exits 1.
    """
    probed: list[list[str]] = []

    def _run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        probed.append(list(cmd))
        rc = 0 if cmd[0] in healthy else 1
        return subprocess.CompletedProcess(cmd, rc, stdout=b"", stderr=b"")

    monkeypatch.setattr(runtime_helper.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(runtime_helper.subprocess, "run", _run)
    return probed


class TestMemoIsKeyedOnItsInputs:
    """The poison sequence that made health lie, plus the memo's remaining value."""

    def test_configless_call_does_not_poison_a_configured_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_probe(monkeypatch, healthy={"docker", "podman"})

        # Auto-detection picks docker first, and used to pin it process-wide.
        assert get_runtime_command() == ["docker", "compose"]

        assert get_runtime_command({"container_runtime": "podman"}) == ["podman", "compose"]

    def test_configured_call_does_not_poison_a_configless_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_probe(monkeypatch, healthy={"docker", "podman"})

        assert get_runtime_command({"container_runtime": "podman"}) == ["podman", "compose"]

        # The reverse direction: auto-detection must not inherit podman.
        assert get_runtime_command() == ["docker", "compose"]

    def test_env_override_is_part_of_the_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_probe(monkeypatch, healthy={"docker", "podman"})

        monkeypatch.setenv("CONTAINER_RUNTIME", "podman")
        assert get_runtime_command() == ["podman", "compose"]

        monkeypatch.delenv("CONTAINER_RUNTIME")
        assert get_runtime_command() == ["docker", "compose"]

    def test_auto_and_absent_share_one_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``auto`` resolves to the same probe order as no config — one cache entry."""
        probed = _fake_probe(monkeypatch, healthy={"docker"})

        assert get_runtime_command({"container_runtime": "auto"}) == ["docker", "compose"]
        after_first = len(probed)
        assert get_runtime_command() == ["docker", "compose"]

        assert len(probed) == after_first

    def test_repeated_identical_inputs_are_still_memoized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probed = _fake_probe(monkeypatch, healthy={"podman"})
        config = {"container_runtime": "podman"}

        assert get_runtime_command(config) == ["podman", "compose"]
        after_first = len(probed)
        assert get_runtime_command(config) == ["podman", "compose"]

        assert len(probed) == after_first

    def test_cached_entry_is_not_corrupted_by_a_mutated_return(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_probe(monkeypatch, healthy={"podman"})

        cmd = get_runtime_command({"container_runtime": "podman"})
        cmd.append("mutated")

        assert get_runtime_command({"container_runtime": "podman"}) == ["podman", "compose"]

    def test_reset_clears_every_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probed = _fake_probe(monkeypatch, healthy={"docker", "podman"})
        get_runtime_command()
        get_runtime_command({"container_runtime": "podman"})
        before = len(probed)

        reset_runtime_cache()

        get_runtime_command()
        get_runtime_command({"container_runtime": "podman"})
        assert len(probed) > before


class _FakeProc:
    """Minimal ``asyncio`` subprocess stand-in for the ``--version`` call."""

    def __init__(self, stdout: bytes = b"podman version 5.0.0") -> None:
        self._stdout = stdout
        self.returncode: int | None = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, b""


class TestHealthThreadsConfigThrough:
    """Both health call sites must measure the runtime the config selects."""

    async def test_category_passes_config_to_runtime_detection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []

        def _record(config: Any = None) -> list[str]:
            seen.append(config)
            raise RuntimeError("no container runtime")  # short-circuits to a skip row

        monkeypatch.setattr(containers_mod, "get_runtime_command", _record)
        config = {"container_runtime": "podman", "deployed_services": ["web"]}

        rows = await containers(config)()

        assert seen == [config]
        assert [r.status for r in rows] == [Status.SKIP]

    async def test_category_reports_the_configured_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With config threaded, a podman project gets a ``podman_available`` row."""

        def _detect(config: Any = None) -> list[str]:
            runtime = (config or {}).get("container_runtime", "docker")
            return [runtime, "compose"]

        async def _fake_exec(*_a: object, **_k: object) -> _FakeProc:
            return _FakeProc()

        monkeypatch.setattr(containers_mod, "get_runtime_command", _detect)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

        rows = await containers({"container_runtime": "podman", "deployed_services": []})()

        assert [r.name for r in rows] == ["podman_available"]
        assert rows[0].status is Status.OK

    async def test_category_hands_the_config_to_the_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []

        async def _capturing_probe(spec: Any, ctx: ProbeContext) -> CheckResult:
            seen.append(ctx.config)
            return CheckResult(spec["name"], spec["category"], Status.OK, "running")

        async def _fake_exec(*_a: object, **_k: object) -> _FakeProc:
            return _FakeProc()

        monkeypatch.setattr(containers_mod, "get_runtime_command", lambda *_a, **_k: ["podman"])
        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
        config = {"container_runtime": "podman", "deployed_services": ["web"]}

        await containers(config, probe=_capturing_probe)()

        assert seen == [config]

    async def test_probe_passes_ctx_config_to_ps_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []

        def _record(config: Any = None, all_containers: bool = False) -> list[str]:
            seen.append(config)
            raise RuntimeError("no container runtime")  # short-circuits to a skip row

        monkeypatch.setattr(container_probe_mod, "get_ps_command", _record)
        config = {"container_runtime": "podman"}
        ctx = ProbeContext(runtime=HealthRuntime({}), config=config)

        result = await container_probe_mod.run({"container": "web"}, ctx)

        assert seen == [config]
        assert result.status is Status.SKIP
