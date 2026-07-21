"""Tests for the ``provider_canary`` health probe (task 2.5).

Drives :func:`osprey.health.probes.provider_canary.run` with fake provider
classes injected through a stub registry (the ``registry=`` DI hook), so no
provider module is imported and no network is touched:

- ``check_health`` ``(True, msg)`` → ``ok``; ``(False, msg)`` → ``warning``;
- unknown provider name → ``warning`` ``"Unknown provider"`` (never a crash);
- ``${VAR}`` api-key/base-url resolution against ``os.environ``, including an
  unset variable collapsing to an empty key;
- a raised ``check_health`` and a hard timeout both map to ``warning``;
- the provider config block is read from ``api.providers.<name>`` when the key
  is absent from ``spec``.

Also exercises the lazy registry (:func:`osprey.health.probes.get_probe`).
"""

from __future__ import annotations

import time
from typing import Any

from osprey.health.models import Status
from osprey.health.probes import ProbeContext, get_probe, provider_canary
from osprey.health.probes.provider_canary import run
from osprey.health.runtime import HealthRuntime


def _ctx() -> ProbeContext:
    """A probe context with a real (never-constructed) runtime; canary ignores it."""
    return ProbeContext(runtime=HealthRuntime({}))


class _StubRegistry:
    """Duck-typed stand-in for ``ProviderRegistry`` with a name→class map."""

    def __init__(self, mapping: dict[str, type]) -> None:
        self._mapping = mapping

    def get_provider(self, name: str) -> type | None:
        return self._mapping.get(name)


def _capturing_provider(captured: list[dict[str, Any]], result: tuple[bool, str]) -> type:
    """Build a fake provider class recording each ``check_health`` call."""

    class _Fake:
        def check_health(
            self,
            api_key: str | None,
            base_url: str | None,
            timeout: float = 5.0,
            model_id: str | None = None,
        ) -> tuple[bool, str]:
            captured.append(
                {
                    "api_key": api_key,
                    "base_url": base_url,
                    "timeout": timeout,
                    "model_id": model_id,
                }
            )
            return result

    return _Fake


def _registry(name: str, provider_cls: type) -> _StubRegistry:
    return _StubRegistry({name: provider_cls})


async def test_success_is_ok() -> None:
    cls = _capturing_provider([], (True, "reachable"))
    result = await run(
        {"name": "cborg", "api_key": "sk-live"},
        _ctx(),
        registry=_registry("cborg", cls),
    )

    assert result.status is Status.OK
    assert result.name == "cborg"
    assert result.category == "providers"
    assert result.message == "reachable"
    assert result.latency_ms > 0


async def test_failure_is_warning() -> None:
    cls = _capturing_provider([], (False, "auth failed"))
    result = await run(
        {"name": "cborg", "api_key": "sk-bad"},
        _ctx(),
        registry=_registry("cborg", cls),
    )

    assert result.status is Status.WARNING
    assert result.message == "auth failed"
    assert result.latency_ms > 0


async def test_unknown_provider_is_warning() -> None:
    result = await run(
        {"name": "made_up"},
        _ctx(),
        registry=_StubRegistry({}),
    )

    assert result.status is Status.WARNING
    assert result.name == "made_up"
    assert result.message == "Unknown provider"


async def test_env_var_resolution() -> None:
    import os

    os.environ["OSPREY_TEST_CANARY_KEY"] = "secret-123"
    try:
        captured: list[dict[str, Any]] = []
        cls = _capturing_provider(captured, (True, "ok"))
        result = await run(
            {
                "name": "cborg",
                "api_key": "${OSPREY_TEST_CANARY_KEY}",
                "base_url": "https://api.example/${OSPREY_TEST_CANARY_KEY}",
                "model_id": "tiny",
            },
            _ctx(),
            registry=_registry("cborg", cls),
        )
    finally:
        del os.environ["OSPREY_TEST_CANARY_KEY"]

    assert result.status is Status.OK
    assert captured[0]["api_key"] == "secret-123"
    assert captured[0]["base_url"] == "https://api.example/secret-123"
    assert captured[0]["model_id"] == "tiny"


async def test_missing_env_var_collapses_to_empty() -> None:
    import os

    os.environ.pop("OSPREY_TEST_CANARY_MISSING", None)
    captured: list[dict[str, Any]] = []
    cls = _capturing_provider(captured, (False, "no key"))
    result = await run(
        {"name": "cborg", "api_key": "${OSPREY_TEST_CANARY_MISSING}"},
        _ctx(),
        registry=_registry("cborg", cls),
    )

    assert result.status is Status.WARNING
    assert captured[0]["api_key"] == ""


async def test_timeout_forwarded_to_check_health() -> None:
    captured: list[dict[str, Any]] = []
    cls = _capturing_provider(captured, (True, "ok"))
    await run(
        {"name": "cborg", "api_key": "k", "timeout_s": 3.0},
        _ctx(),
        registry=_registry("cborg", cls),
    )

    assert captured[0]["timeout"] == 3.0


async def test_check_health_exception_is_warning() -> None:
    class _Boom:
        def check_health(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
            raise RuntimeError("connection reset")

    result = await run(
        {"name": "cborg", "api_key": "k"},
        _ctx(),
        registry=_registry("cborg", _Boom),
    )

    assert result.status is Status.WARNING
    assert "connection reset" in result.details


async def test_hard_timeout_is_warning(monkeypatch: Any) -> None:
    monkeypatch.setattr(provider_canary, "_OFFLOAD_MARGIN_S", 0.05)

    class _Hang:
        def check_health(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
            time.sleep(2.0)  # ignores its own timeout; the bridge must abandon it
            return (True, "late")

    result = await run(
        {"name": "cborg", "api_key": "k", "timeout_s": 0.05},
        _ctx(),
        registry=_registry("cborg", _Hang),
    )

    assert result.status is Status.WARNING
    assert "timed out" in result.message


async def test_api_key_falls_back_to_config_block(monkeypatch: Any) -> None:
    def _fake_config_value(path: str, default: Any = None) -> Any:
        assert path == "api.providers.cborg"
        return {"api_key": "${OSPREY_TEST_CANARY_CFG}", "base_url": "https://cfg"}

    monkeypatch.setattr(provider_canary, "get_config_value", _fake_config_value)
    monkeypatch.setenv("OSPREY_TEST_CANARY_CFG", "cfg-secret")

    captured: list[dict[str, Any]] = []
    cls = _capturing_provider(captured, (True, "ok"))
    result = await run(
        {"name": "cborg"},  # no api_key/base_url → config fallback
        _ctx(),
        registry=_registry("cborg", cls),
    )

    assert result.status is Status.OK
    assert captured[0]["api_key"] == "cfg-secret"
    assert captured[0]["base_url"] == "https://cfg"


async def test_provider_param_distinct_from_result_name() -> None:
    """Declarative shape: ``provider`` resolves the class, ``name`` is the row id.

    The runner injects ``name`` as the check's identity and leaves the provider
    to a probe param, so a check named ``my_cborg_check`` targeting provider
    ``cborg`` must resolve ``cborg`` yet report under ``my_cborg_check``.
    """
    cls = _capturing_provider([], (True, "reachable"))
    result = await run(
        {"name": "my_cborg_check", "provider": "cborg", "api_key": "k"},
        _ctx(),
        registry=_registry("cborg", cls),
    )

    assert result.status is Status.OK
    assert result.name == "my_cborg_check"


async def test_unknown_via_provider_param_uses_result_name() -> None:
    result = await run(
        {"name": "cborg_health", "provider": "made_up"},
        _ctx(),
        registry=_StubRegistry({}),
    )

    assert result.status is Status.WARNING
    assert result.name == "cborg_health"
    assert result.message == "Unknown provider"


async def test_lazy_registry_resolves_to_run() -> None:
    probe = get_probe("provider_canary")
    assert probe is run
