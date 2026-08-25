"""Tests for the core ``reach`` health category.

Drives the category through an injected ``knock`` so no socket is opened,
exercising: no live consumer ⇒ no rows; one row per service naming every
consumer that dials it; the address in the row is the one the client's own
resolver produces; and the three outcomes — reachable, unreachable, nothing
to dial.
"""

from __future__ import annotations

import pytest

from osprey.health.core.reach import CATEGORY, reach
from osprey.health.models import CheckResult, Status

HYBRID_ON = {"ariel": {"search_modules": {"hybrid": {"enabled": True}}}}


async def _run(config, *, knock=None) -> dict[str, CheckResult]:
    """Run the category with a knock that answers, keyed by row name."""

    async def _answers(host: str, port: int) -> None:
        return None

    results = await reach(config, knock=knock or _answers)()
    assert isinstance(results, list)
    assert all(r.category == CATEGORY for r in results)
    return {r.name: r for r in results}


class TestRows:
    async def test_no_live_consumer_contributes_no_rows(self):
        """A render that switches nothing on has nothing to reach."""
        assert await _run({}) == {}

    async def test_one_row_per_service_naming_every_consumer(self):
        """Hybrid search and the OKF panel both dial the qmd sidecar: one knock,
        one row, both names — a service is not reported twice because two
        clients depend on it."""
        config = {
            **HYBRID_ON,
            "services": {"qmd": {"port": 8180}},
            "web": {"panels": {"okf": {"enabled": True}}},
            "facility_knowledge": {"bundle_path": "data/facility_knowledge"},
        }
        rows = await _run(config)
        assert "reach.qmd" in rows
        assert "ARIEL hybrid search" in rows["reach.qmd"].message
        assert "OKF panel ranked search" in rows["reach.qmd"].message

    async def test_knocks_on_what_the_client_dials(self):
        """The address comes from the client's resolver, not from a literal:
        a moved sidecar port is the port knocked on."""
        seen: list[tuple[str, int]] = []

        async def knock(host: str, port: int) -> None:
            seen.append((host, port))

        rows = await _run({**HYBRID_ON, "services": {"qmd": {"port": 9180}}}, knock=knock)
        # The `ariel:` section also makes the ARIEL database a live consumer;
        # the sidecar's knock is the one this test is about.
        assert ("127.0.0.1", 9180) in seen
        row = rows["reach.qmd"]
        assert row.status is Status.OK
        assert row.value == "up"
        assert "127.0.0.1:9180" in row.message


class TestOutcomes:
    async def test_unreachable_is_a_warning_naming_the_address(self):
        async def refused(host: str, port: int) -> None:
            raise ConnectionRefusedError(111, "Connection refused")

        rows = await _run({**HYBRID_ON, "services": {"qmd": {"port": 8180}}}, knock=refused)
        row = rows["reach.qmd"]
        assert row.status is Status.WARNING
        assert row.value == "offline"
        assert "127.0.0.1:8180" in row.message
        assert "services.qmd.port" in row.details

    async def test_timeout_is_unreachable_too(self):
        async def hangs(host: str, port: int) -> None:
            raise TimeoutError()

        rows = await _run({**HYBRID_ON, "services": {"qmd": {"port": 8180}}}, knock=hangs)
        assert rows["reach.qmd"].value == "offline"

    async def test_nothing_to_dial_is_a_warning_naming_the_key(self):
        """The state the build refuses, met at run time (a hand-edited render,
        an older build): switched on, no endpoint. Never a knock."""
        from osprey.deployment.qmd_service import DEFAULT_PORT

        seen: list[tuple[str, int]] = []

        async def knock(host: str, port: int) -> None:
            seen.append((host, port))

        rows = await _run(HYBRID_ON, knock=knock)
        row = rows["reach.qmd"]
        # No knock at the sidecar's default: nothing was resolved, nothing is
        # guessed. (The ARIEL database, live from the same `ariel:` section,
        # is knocked on as usual.)
        assert all(port != DEFAULT_PORT for _host, port in seen)
        assert row.status is Status.WARNING
        assert row.value == "unresolved"
        assert "ariel.search_modules.hybrid.enabled" in row.details
        assert "services.qmd.port" in row.details
        assert "Every use will fail" in row.details

    async def test_a_degrading_consumer_says_so(self):
        """The OKF panel alone falls back to substring search without a
        sidecar; the row reports the missing endpoint without crying wolf."""
        config = {
            "web": {"panels": {"okf": {"enabled": True}}},
            "facility_knowledge": {"bundle_path": "data/facility_knowledge"},
        }
        rows = await _run(config)
        assert rows["reach.qmd"].value == "unresolved"
        assert "degrades" in rows["reach.qmd"].details


@pytest.mark.parametrize("config", [None, {}])
async def test_missing_config_is_no_rows(config):
    assert await reach(config)() == []
