"""Which connector a lane's queueserver worker builds, and whether it may write.

Two lanes mount ONE config.yml, so ``control_system.type`` cannot answer "which
machine does this worker drive" on its own. The resolution ladder in
``queue_backend.resolve_lane_connector_type`` answers it from the lane's own
``services.<lane>.target`` block, and this module pins the two deployment
baselines a switchable project can have — plus the fallbacks that keep every
single-lane project rendered so far building exactly what it built before.

The write posture rides along, because the type the ladder lands on is also the
key of the block that arms it. The case worth being careful about is the one
where the lane declares a target this deployment cannot resolve: it is built as
the baseline type while addressing another machine, so its posture comes from
``control_system.writes_enabled`` alone and NOT from the baseline type's block.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.services.bluesky_bridge import qserver_startup
from osprey.services.bluesky_bridge import queue_backend as qb

pytestmark = pytest.mark.unit

# A live baseline that arms its simulator and nothing else: the deployment a
# facility runs when it wants an agent that may move the virtual accelerator
# while the real machine stays read-only.
LIVE_BASELINE = {
    "type": "epics",
    "writes_enabled": False,
    "connector": {
        "epics": {"timeout": 5.0},
        "virtual_accelerator": {"writes_enabled": True},
    },
}

# A virtual-accelerator baseline whose second lane serves `live`. It ships no
# live connector block on purpose: that lane's gateway is a facility address no
# build can verify, supplied at `osprey up` as EPICS_CA_NAME_SERVERS.
VA_BASELINE = {
    "type": "virtual_accelerator",
    "writes_enabled": False,
    "connector": {"virtual_accelerator": {"writes_enabled": True}},
}


@pytest.fixture
def deployment(monkeypatch: pytest.MonkeyPatch):
    """Stage the config.yml a lane's containers mount, and the lane env var."""

    def _stage(
        section: dict[str, Any],
        *,
        lane: str | None = None,
        targets: dict[str, str] | None = None,
    ) -> None:
        if lane is None:
            monkeypatch.delenv(qb.LANE_ENV, raising=False)
        else:
            monkeypatch.setenv(qb.LANE_ENV, lane)
        monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)

        declared = {f"services.{key}.target": value for key, value in (targets or {}).items()}

        def fake_get_config_value(key: str, default: Any = None, config_path: Any = None) -> Any:
            if key == "control_system":
                return section
            if key == "control_system.type":
                return section.get("type", default)
            if key == "control_system.writes_enabled":
                return section.get("writes_enabled", default)
            return declared.get(key, default)

        monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)

    return _stage


# =========================================================================
# Rung 1 — the lane declares no target
# =========================================================================


@pytest.mark.parametrize("control_system_type", ["mock", "epics", "virtual_accelerator", "doocs"])
def test_a_lane_with_no_declared_target_builds_control_system_type(
    deployment, control_system_type
) -> None:
    """Every single-lane project rendered so far, byte for byte what it was."""
    deployment({"type": control_system_type}, targets=None)

    assert qserver_startup.resolve_control_system_type() == control_system_type
    assert qb.resolve_lane_connector_type()[1] is None


def test_an_unreadable_config_still_builds_the_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented fail-safe: the mock connector never touches Channel Access."""

    def _raise(*_: Any, **__: Any) -> Any:
        raise FileNotFoundError("no project config context")

    monkeypatch.delenv(qb.LANE_ENV, raising=False)
    monkeypatch.setattr("osprey.utils.config.get_config_value", _raise)

    assert qserver_startup.resolve_control_system_type() == "mock"
    assert qserver_startup.worker_writes_enabled() is False


# =========================================================================
# Rung 2 — the declared target resolves
# =========================================================================


def test_the_va_lane_of_a_live_baseline_builds_the_virtual_accelerator(deployment) -> None:
    """The lane axis's whole point: two lanes, one config, two connectors."""
    deployment(
        LIVE_BASELINE,
        lane="bluesky_va",
        targets={"bluesky": "live", "bluesky_va": "va"},
    )

    assert qserver_startup.resolve_control_system_type() == "virtual_accelerator"
    assert qb.resolve_lane_connector_type()[1] is None


def test_the_va_lane_of_a_live_baseline_may_write(deployment) -> None:
    """Armed by `control_system.connector.virtual_accelerator.writes_enabled`.

    The deployment-wide key is false here, and that is the point: a facility
    whose real machine is a live one arms its simulator alone.
    """
    deployment(
        LIVE_BASELINE,
        lane="bluesky_va",
        targets={"bluesky": "live", "bluesky_va": "va"},
    )

    assert qserver_startup.worker_writes_enabled() is True


def test_the_live_lane_of_a_live_baseline_may_not_write(deployment) -> None:
    """The `epics` block says nothing, so it inherits the deployment-wide false."""
    deployment(
        LIVE_BASELINE,
        lane="bluesky",
        targets={"bluesky": "live", "bluesky_va": "va"},
    )

    assert qserver_startup.resolve_control_system_type() == "epics"
    assert qserver_startup.worker_writes_enabled() is False


def test_a_readonly_run_disarms_a_lane_its_config_armed(deployment, monkeypatch) -> None:
    """The deployment posture is one half; the run's own mode is the other."""
    deployment(
        LIVE_BASELINE,
        lane="bluesky_va",
        targets={"bluesky": "live", "bluesky_va": "va"},
    )
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    assert qserver_startup.worker_writes_enabled() is False


# =========================================================================
# Rung 3 — the declared target does not resolve
# =========================================================================


def test_the_live_lane_of_a_va_baseline_keeps_building_the_baseline_type(deployment) -> None:
    """Today's answer, kept: `live` resolves to nothing here, and the lane runs.

    Refusing instead would refuse the deployments that are correct — this lane's
    gateway arrives at `osprey up` as EPICS_CA_NAME_SERVERS, which no build can
    verify and no config block has to describe.
    """
    deployment(
        VA_BASELINE,
        lane="bluesky_live",
        targets={"bluesky": "va", "bluesky_live": "live"},
    )

    assert qserver_startup.resolve_control_system_type() == "virtual_accelerator"


def test_a_degraded_lane_names_what_it_declared_against_what_it_built(deployment) -> None:
    """The mismatch is reported rather than absorbed."""
    deployment(
        VA_BASELINE,
        lane="bluesky_live",
        targets={"bluesky": "va", "bluesky_live": "live"},
    )

    _, lane_degraded = qb.resolve_lane_connector_type()

    assert lane_degraded is not None
    assert "bluesky_live" in lane_degraded
    assert "'live'" in lane_degraded
    assert "'virtual_accelerator'" in lane_degraded
    assert "control_system.connector.epics" in lane_degraded


def test_a_degraded_lane_may_not_write_on_the_baseline_types_arming(deployment) -> None:
    """The safety property of rung 3, and the reason it is not the type's block.

    This deployment armed `virtual_accelerator`, and this lane was BUILT as a
    virtual accelerator — but it addresses the facility's live gateway. Reading
    the VA block's arming here would point "you may write to the simulator" at
    real hardware, so the only posture that applies is the deployment-wide one,
    which is false.
    """
    deployment(
        VA_BASELINE,
        lane="bluesky_live",
        targets={"bluesky": "va", "bluesky_live": "live"},
    )

    assert qserver_startup.worker_writes_enabled() is False


def test_the_va_lane_of_a_va_baseline_is_not_degraded_and_may_write(deployment) -> None:
    """The other lane of the same deployment resolves, and its block arms it."""
    deployment(
        VA_BASELINE,
        lane="bluesky",
        targets={"bluesky": "va", "bluesky_live": "live"},
    )

    assert qserver_startup.resolve_control_system_type() == "virtual_accelerator"
    assert qb.resolve_lane_connector_type()[1] is None
    assert qserver_startup.worker_writes_enabled() is True


def test_a_target_that_names_nothing_degrades_rather_than_raising(deployment) -> None:
    """The build refuses this spelling; a bridge that meets it anyway still answers.

    Every lane record is fail-closed, so it is also never half-built: a typo
    that reached a deployed config.yml lands on the same rung an underivable
    target does.
    """
    deployment(LIVE_BASELINE, lane="bluesky", targets={"bluesky": "prod"})

    connector_type, lane_degraded = qb.resolve_lane_connector_type()

    assert connector_type == "epics"
    assert lane_degraded is not None


# =========================================================================
# The capability record the bridge publishes
# =========================================================================


async def test_the_capability_record_carries_the_degradation(deployment) -> None:
    """`/health` is where an operator finds out a lane is not fully described."""
    deployment(
        VA_BASELINE,
        lane="bluesky_live",
        targets={"bluesky": "va", "bluesky_live": "live"},
    )

    capability = await qb.QueueBackend(None).capability()

    assert capability.lane == "bluesky_live"
    assert capability.lane_target == "live"
    assert capability.lane_degraded is not None
    assert capability.to_dict()["lane_degraded"] == capability.lane_degraded


async def test_the_capability_record_judges_the_lanes_connector_not_the_baselines(
    deployment,
) -> None:
    """The record answers for the connector plans will actually run against.

    A `doocs` deployment cannot execute plans, but its VA lane's worker builds a
    virtual accelerator and can — so judging the lane by `control_system.type`
    would have this bridge advertise a refusal that belongs to the other lane.
    """
    deployment(
        {"type": "doocs", "connector": {"virtual_accelerator": {}}},
        lane="bluesky_va",
        targets={"bluesky": "live", "bluesky_va": "va"},
    )

    capability = await qb.QueueBackend(None).capability()

    assert capability.reason == qb.REASON_MANAGER_NOT_CONFIGURED
    assert capability.lane_degraded is None
