"""The deployment's own agent provider is not an advisory.

``claude_code.provider`` names the provider every web terminal and the dispatch
worker authenticate with. When that one fails, the deployment is down; the
``providers`` row for it is an error (``osprey health`` exits 2), while every
other configured provider stays the warning it always was.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osprey.health.core.providers import providers
from osprey.health.models import CheckResult, Status
from tests.health._fakes import StubRegistry, fake_provider


async def _run(config: Mapping[str, Any] | None, registry: StubRegistry) -> list[CheckResult]:
    return await providers(config, registry=registry)()


_AUTH_FAILED = (False, "Authentication failed (invalid API key)")


async def test_the_deployments_own_provider_failing_is_an_error() -> None:
    registry = StubRegistry(
        {"own": fake_provider(_AUTH_FAILED), "other": fake_provider(_AUTH_FAILED)}
    )
    config = {
        "claude_code": {"provider": "own"},
        "api": {"providers": {"own": {"api_key": "k"}, "other": {"api_key": "k"}}},
    }
    by_name = {r.name: r for r in await _run(config, registry)}

    assert by_name["own"].status is Status.ERROR
    assert "Authentication failed" in by_name["own"].message
    assert "claude_code.provider" in by_name["own"].message
    assert by_name["other"].status is Status.WARNING


async def test_with_personal_logins_allowed_the_own_provider_failing_is_a_warning() -> None:
    """A terminal on its user's own login never touches the provider (#925).

    Which terminals are login-routed is decided inside each container and is
    not visible to health, so the row stays the probe's warning and says who
    is affected instead of declaring the deployment down.
    """
    registry = StubRegistry({"own": fake_provider(_AUTH_FAILED)})
    config = {
        "claude_code": {"provider": "own", "allow_personal_login": True},
        "api": {"providers": {"own": {"api_key": "k"}}},
    }
    (row,) = await _run(config, registry)
    assert row.status is Status.WARNING
    assert "Authentication failed" in row.message
    assert "allow_personal_login" in row.message
    assert "without a personal login" in row.message


async def test_the_deployments_own_provider_passing_is_plain_ok() -> None:
    registry = StubRegistry({"own": fake_provider((True, "API accessible and authenticated"))})
    config = {"claude_code": {"provider": "own"}, "api": {"providers": {"own": {"api_key": "k"}}}}
    rows = await _run(config, registry)

    assert rows[0].status is Status.OK
    assert rows[0].message == "API accessible and authenticated"


async def test_without_a_claude_code_provider_every_failure_stays_advisory() -> None:
    registry = StubRegistry({"p": fake_provider((False, "auth failed"))})
    rows = await _run({"api": {"providers": {"p": {"api_key": "k"}}}}, registry)

    assert rows[0].status is Status.WARNING


async def test_the_deployments_own_config_only_provider_is_skipped_not_failed() -> None:
    """No adapter class for the agent's provider is not a broken deployment.

    ``osprey health`` exits 2 on an error row, so grading "there is no Python
    class for this name" as a failure takes down the exit status of a
    deployment whose agent is running fine on a config-only endpoint.
    """
    config = {
        "claude_code": {"provider": "house_gateway"},
        "api": {"providers": {"house_gateway": {"base_url": "https://gw.example/v1"}}},
    }
    by_name = {r.name: r for r in await _run(config, StubRegistry({}))}

    assert by_name["house_gateway"].status is Status.SKIP
    assert "no code adapter" in by_name["house_gateway"].message
    assert "claude_code.provider" in by_name["house_gateway"].message


async def test_an_own_provider_absent_from_api_providers_is_reported() -> None:
    """Named as the agent's provider but configured nowhere: there is no key to
    probe, and that is itself the error."""
    config = {"claude_code": {"provider": "own"}, "api": {"providers": {"p": {"api_key": "k"}}}}
    by_name = {
        r.name: r for r in await _run(config, StubRegistry({"p": fake_provider((True, "ok"))}))
    }

    assert by_name["own"].status is Status.ERROR
    assert "api.providers" in by_name["own"].message
