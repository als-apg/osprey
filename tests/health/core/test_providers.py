"""Tests for the core ``providers`` health category (task 4.6).

Composes the ``provider_canary`` probe once per ``api.providers`` entry, running
every item concurrently. Fake provider classes are injected through the
factory's ``registry=`` DI hook, so no real provider module is imported and no
network is touched. Covers:

- zero providers (and ``config=None``) → no rows;
- mixed ok/warning results, statuses never ``error``;
- api_key/base_url flow from the config block into ``check_health``;
- an unknown provider name → a warning row (never a crash);
- 7 slow providers complete within the suite deadline with 7 rows, proving the
  items run concurrently rather than serially.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from osprey.health.core.providers import providers
from osprey.health.models import CheckResult, Status


class _StubRegistry:
    """Duck-typed stand-in for ``ProviderRegistry`` with a name→class map."""

    def __init__(self, mapping: dict[str, type]) -> None:
        self._mapping = mapping

    def get_provider(self, name: str) -> type | None:
        return self._mapping.get(name)


def _provider(result: tuple[bool, str] = (True, "ok"), sleep: float = 0.0) -> type:
    """Build a fake provider class returning *result* after an optional sleep."""

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
    results = await providers(config, registry=registry)()
    assert isinstance(results, list)
    return results


async def test_zero_providers_no_rows() -> None:
    rows = await _run({"api": {"providers": {}}}, _StubRegistry({}))
    assert rows == []


async def test_config_none_no_rows() -> None:
    rows = await _run(None, _StubRegistry({}))
    assert rows == []


async def test_mixed_results_ok_and_warning() -> None:
    registry = _StubRegistry(
        {
            "good": _provider((True, "reachable")),
            "bad": _provider((False, "auth failed")),
        }
    )
    config = {"api": {"providers": {"good": {"api_key": "k"}, "bad": {"api_key": "k"}}}}
    by_name = {r.name: r for r in await _run(config, registry)}

    assert set(by_name) == {"good", "bad"}
    assert by_name["good"].status is Status.OK
    assert by_name["bad"].status is Status.WARNING
    assert all(r.category == "providers" for r in by_name.values())
    assert all(r.status in (Status.OK, Status.WARNING) for r in by_name.values())


async def test_unknown_provider_is_warning() -> None:
    config = {"api": {"providers": {"made_up": {"api_key": "k"}}}}
    rows = await _run(config, _StubRegistry({}))

    assert len(rows) == 1
    assert rows[0].name == "made_up"
    assert rows[0].status is Status.WARNING
    assert rows[0].message == "Unknown provider"


async def test_api_key_and_base_url_flow_from_config() -> None:
    import os

    os.environ["OSPREY_TEST_CORE_PROV_KEY"] = "secret-xyz"
    captured: list[dict[str, Any]] = []

    class _Recording:
        def check_health(
            self,
            api_key: str | None,
            base_url: str | None,
            timeout: float = 5.0,
            model_id: str | None = None,
        ) -> tuple[bool, str]:
            captured.append({"api_key": api_key, "base_url": base_url})
            return (True, "ok")

    config = {
        "api": {
            "providers": {
                "cborg": {
                    "api_key": "${OSPREY_TEST_CORE_PROV_KEY}",
                    "base_url": "https://cborg.example",
                }
            }
        }
    }
    try:
        rows = await _run(config, _StubRegistry({"cborg": _Recording}))
    finally:
        del os.environ["OSPREY_TEST_CORE_PROV_KEY"]

    assert rows[0].status is Status.OK
    assert captured[0]["api_key"] == "secret-xyz"
    assert captured[0]["base_url"] == "https://cborg.example"


async def test_seven_slow_providers_run_concurrently() -> None:
    n = 7
    per_item_sleep = 1.0
    registry = _StubRegistry(
        {f"p{i}": _provider((True, "ok"), sleep=per_item_sleep) for i in range(n)}
    )
    config = {"api": {"providers": {f"p{i}": {"api_key": "k"} for i in range(n)}}}

    t0 = perf_counter()
    rows = await _run(config, registry)
    elapsed = perf_counter() - t0

    assert len(rows) == n
    assert all(r.status is Status.OK for r in rows)
    # Serial execution would take n × per_item_sleep = 7s; concurrent ≈ 1s. A
    # generous ceiling well under the serial time proves the items overlap.
    assert elapsed < 4.0
