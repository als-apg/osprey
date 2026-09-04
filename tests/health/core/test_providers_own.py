"""The deployment's own agent provider is not an advisory.

``claude_code.provider`` names the provider every web terminal and the dispatch
worker authenticate with. When that one fails, the deployment is down; the
``providers`` row for it is an error (``osprey health`` exits 2), while every
other configured provider stays the warning it always was.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from osprey.health.core.providers import providers
from osprey.health.models import CheckResult, Status


class _StubRegistry:
    def __init__(self, mapping: dict[str, type]) -> None:
        self._mapping = mapping

    def get_provider(self, name: str) -> type | None:
        return self._mapping.get(name)


def _provider(result: tuple[bool, str], sleep: float = 0.0) -> type:
    class _Fake:
        def check_health(
            self,
            api_key: str | None,
            base_url: str | None,
            timeout: float = 5.0,
            model_id: str | None = None,
        ) -> tuple[bool, str]:
            if sleep:
                time.sleep(sleep)
            return result

    return _Fake


async def _run(config: Mapping[str, Any] | None, registry: _StubRegistry) -> list[CheckResult]:
    return await providers(config, registry=registry)()


_AUTH_FAILED = (False, "Authentication failed (invalid API key)")


async def test_the_deployments_own_provider_failing_is_an_error() -> None:
    registry = _StubRegistry({"own": _provider(_AUTH_FAILED), "other": _provider(_AUTH_FAILED)})
    config = {
        "claude_code": {"provider": "own"},
        "api": {"providers": {"own": {"api_key": "k"}, "other": {"api_key": "k"}}},
    }
    by_name = {r.name: r for r in await _run(config, registry)}

    assert by_name["own"].status is Status.ERROR
    assert "Authentication failed" in by_name["own"].message
    assert "claude_code.provider" in by_name["own"].message
    assert by_name["other"].status is Status.WARNING


async def test_the_deployments_own_provider_passing_is_plain_ok() -> None:
    registry = _StubRegistry({"own": _provider((True, "API accessible and authenticated"))})
    config = {"claude_code": {"provider": "own"}, "api": {"providers": {"own": {"api_key": "k"}}}}
    rows = await _run(config, registry)

    assert rows[0].status is Status.OK
    assert rows[0].message == "API accessible and authenticated"


async def test_without_a_claude_code_provider_every_failure_stays_advisory() -> None:
    registry = _StubRegistry({"p": _provider((False, "auth failed"))})
    rows = await _run({"api": {"providers": {"p": {"api_key": "k"}}}}, registry)

    assert rows[0].status is Status.WARNING


async def test_an_own_provider_absent_from_api_providers_is_reported() -> None:
    """Named as the agent's provider but configured nowhere: there is no key to
    probe, and that is itself the error."""
    config = {"claude_code": {"provider": "own"}, "api": {"providers": {"p": {"api_key": "k"}}}}
    by_name = {r.name: r for r in await _run(config, _StubRegistry({"p": _provider((True, "ok"))}))}

    assert by_name["own"].status is Status.ERROR
    assert "api.providers" in by_name["own"].message
