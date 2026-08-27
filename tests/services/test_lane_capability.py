"""The lane identity a bridge publishes on its capability record.

A plan lane is a whole Bluesky stack wired, at render time, to ONE control
target. Two lanes mount the same project config, so the target a lane serves
cannot be read out of ``control_system.type`` alone — it comes from the lane's
own ``services.<lane>.target`` block, and which block that is comes from the
container's :data:`~osprey.services.bluesky_bridge.queue_backend.LANE_ENV`.

What these tests pin, in the order the record is built:

1. **The declared target wins.** A lane whose config block says ``target: va``
   reports ``va`` even when the deployment baseline says something else — that
   disagreement is the normal shape of the second lane, not an error.
2. **A legacy deployment still gets an honest answer.** No ``target:`` key is
   every single-lane deployment shipped so far; those derive the baseline by
   the SAME rule the host side uses (``target_banner.resolve_baseline_target``),
   and the two are asserted equal here rather than assumed.
3. **The record is STATIC.** The whole point of the producer split: the bridge
   reports its lane, the host composes the active/inactive view. A fabricated
   session switch — a real state file, written and then switched, exactly as
   the controls server would — must not move the bridge's record by one field.
   The bridge cannot even see that file from its container; this test is what
   keeps the code from quietly growing the ability to.
4. **Every consumer of the shape sees the fields**, including the hand-built
   fallback dict `/health` answers with when the capability probe itself
   explodes — the one path that does not go through ``Capability.to_dict``.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.mcp_server.control_system import target_banner, target_state
from osprey.services.bluesky_bridge import app as bridge_app_module
from osprey.services.bluesky_bridge import queue_backend as qb
from osprey.services.bluesky_bridge.queue_backend import Capability, QueueBackend

pytestmark = pytest.mark.unit

# The wire keys the capability object carries. Pinned as a set because every
# consumer branches on these names: the JS panel client, the MCP queue tools,
# and `/health` itself.
_WIRE_KEYS = {"can_execute", "reason", "detail", "lane", "lane_target", "lane_degraded"}


class _ExplodingBackend:
    """A backend whose capability probe fails in a way nothing anticipated."""

    async def capability(self) -> Any:
        raise RuntimeError("the probe itself broke")


@pytest.fixture(autouse=True)
def _isolated_backend():
    """`get_queue_backend` memoizes process-wide; clear it around every test."""
    bridge_app_module.set_queue_backend(None)
    yield
    bridge_app_module.set_queue_backend(None)


@pytest.fixture
def deployment(monkeypatch: pytest.MonkeyPatch):
    """Stage the config a bridge container would read, and its lane env var.

    `services.<lane>.target`, `control_system.type` and the `control_system`
    section itself are the only keys this fixture answers; everything else
    falls through to its default, the way the surrounding suite's `connector`
    fixture does — a bridge reads more keys than these and answering them all
    with one value would stage nonsense.
    """

    def _stage(
        *,
        control_system_type: str = "epics",
        lane: str | None = None,
        targets: dict[str, str] | None = None,
        connector: dict[str, Any] | None = None,
    ) -> None:
        if lane is None:
            monkeypatch.delenv(qb.LANE_ENV, raising=False)
        else:
            monkeypatch.setenv(qb.LANE_ENV, lane)

        declared = {f"services.{key}.target": value for key, value in (targets or {}).items()}
        control_system: dict[str, Any] = {"type": control_system_type}
        if connector is not None:
            control_system["connector"] = connector

        def fake_get_config_value(key: str, default: Any = None, config_path: Any = None) -> Any:
            if key == "control_system.type":
                return control_system_type
            if key == "control_system":
                return control_system
            return declared.get(key, default)

        monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)
        # The host-side resolver this module's fallback is pinned equal to
        # reads the section, not a dotted key.
        monkeypatch.setattr(
            target_banner, "load_osprey_config", lambda: {"control_system": control_system}
        )

    return _stage


async def _capability() -> Capability:
    """The record a bridge with no queue server publishes.

    `manager_not_configured` is the shortest path through `capability()` that
    still runs the whole lane resolution, and the lane fields are the same on
    every path — which the reason-code sweep below pins.
    """
    return await QueueBackend(None).capability()


# =========================================================================
# The lane's own config block
# =========================================================================


async def test_a_lane_reports_the_target_its_own_config_block_declares(deployment) -> None:
    """The declared target wins over the deployment baseline.

    Staged with a live baseline and a VA lane on purpose: on a two-lane deploy
    exactly one lane disagrees with the baseline, and a bridge that fell back
    to `control_system.type` would mislabel it as the one that serves the real
    machine.
    """
    deployment(control_system_type="epics", targets={"bluesky": "va"})

    capability = await _capability()

    assert capability.lane == qb.DEFAULT_LANE
    assert capability.lane_target == "va"


async def test_the_second_lane_reads_its_own_service_block(deployment) -> None:
    """Two lanes, one mounted config: the env var is what tells them apart."""
    deployment(
        control_system_type="epics",
        lane="bluesky_va",
        targets={"bluesky": "live", "bluesky_va": "va"},
    )

    capability = await _capability()

    assert capability.lane == "bluesky_va"
    assert capability.lane_target == "va"


async def test_an_unset_lane_env_is_the_historical_service_key(deployment) -> None:
    """Every deployment shipped so far renders exactly one `services.bluesky`."""
    deployment(control_system_type="epics", lane=None, targets={"bluesky": "live"})

    assert (await _capability()).lane == "bluesky"


# =========================================================================
# The legacy deployment: no `target:` key at all
# =========================================================================


@pytest.mark.parametrize(
    ("control_system_type", "expected"),
    [("virtual_accelerator", "va"), ("epics", "live"), ("mock", "live")],
)
async def test_a_legacy_lane_derives_the_deployment_baseline(
    deployment, control_system_type, expected
) -> None:
    """No `target:` key means a single-lane deploy, which serves the baseline.

    `mock` resolving to `live` is not an oversight: a mock deployment has no
    virtual accelerator to serve, and the host side answers the same way, which
    the next test pins rather than restates.
    """
    deployment(control_system_type=control_system_type, targets=None)

    assert (await _capability()).lane_target == expected


@pytest.mark.parametrize("control_system_type", ["virtual_accelerator", "epics", "mock", "doocs"])
async def test_the_legacy_fallback_agrees_with_the_host_side_baseline(
    deployment, control_system_type
) -> None:
    """Bridge and host must not hold two opinions about one lane's target.

    The host refuses `queue_add` while the session target differs from the
    lane's; if this module derived the baseline by its own rule, a deployment
    could be refused for a mismatch neither side actually has.
    """
    deployment(control_system_type=control_system_type, targets=None)

    assert (await _capability()).lane_target == target_banner.resolve_baseline_target()


async def test_an_unreadable_config_still_yields_a_lane_identity(monkeypatch) -> None:
    """The capability record is fail-closed, so it is also never half-built.

    `config_unreadable` is a real reason code with a real record; a consumer
    reading `lane_target` off it must find a string, not a hole.
    """

    def raising_get_config_value(key: str, default: Any = None, config_path: Any = None) -> Any:
        raise FileNotFoundError("no config.yml found")

    monkeypatch.delenv(qb.LANE_ENV, raising=False)
    monkeypatch.setattr("osprey.utils.config.get_config_value", raising_get_config_value)

    capability = await _capability()

    assert capability.reason == qb.REASON_CONFIG_UNREADABLE
    assert capability.lane == qb.DEFAULT_LANE
    assert capability.lane_target in target_state.TARGET_NAMES


# =========================================================================
# The wire shape
# =========================================================================


def test_to_dict_round_trips_the_lane_fields() -> None:
    payload = Capability(
        can_execute=False,
        reason=qb.REASON_BROWSE_ONLY_CONNECTOR,
        detail="why",
        lane="bluesky_va",
        lane_target="va",
    ).to_dict()

    assert payload == {
        "can_execute": False,
        "reason": qb.REASON_BROWSE_ONLY_CONNECTOR,
        "detail": "why",
        "lane": "bluesky_va",
        "lane_target": "va",
        "lane_degraded": None,
    }


@pytest.mark.parametrize(
    ("control_system_type", "reason"),
    [
        ("mock", qb.REASON_BROWSE_ONLY_CONNECTOR),
        ("doocs", qb.REASON_UNSUPPORTED_CONNECTOR),
        ("epics", qb.REASON_MANAGER_NOT_CONFIGURED),
    ],
)
async def test_every_refusal_carries_the_lane_identity(
    deployment, control_system_type, reason
) -> None:
    """A refusal is exactly when a consumer needs to know WHICH lane refused."""
    deployment(control_system_type=control_system_type, targets={"bluesky": "live"})

    capability = await _capability()

    assert capability.reason == reason
    assert set(capability.to_dict()) == _WIRE_KEYS
    assert capability.lane == "bluesky"
    assert capability.lane_target == "live"


def test_the_health_fallback_dict_matches_the_record(deployment) -> None:
    """`/health`'s hand-built fallback is the one path that skips `to_dict`.

    It exists because a capability probe that raises must still answer 200 with
    a cannot-execute record — and that record has to carry the same keys as
    every other, or a consumer parsing the fallback finds a different object
    than the one it was written against.
    """
    from fastapi.testclient import TestClient

    from osprey.services.bluesky_bridge.app import app as bridge_app

    deployment(control_system_type="virtual_accelerator", lane="bluesky_va")
    bridge_app_module.set_queue_backend(_ExplodingBackend())

    with TestClient(bridge_app) as client:
        capability = client.get("/health").json()["capability"]

    assert set(capability) == _WIRE_KEYS
    assert capability["can_execute"] is False
    assert capability["reason"] == qb.REASON_MANAGER_UNREACHABLE
    assert capability["lane"] == "bluesky_va"
    assert capability["lane_target"] == "va"


# =========================================================================
# Static across a switch
# =========================================================================


async def test_the_record_does_not_move_when_the_session_switches(
    deployment, tmp_path, monkeypatch
) -> None:
    """The producer split, asserted end to end on a REAL state file.

    A controls server publishes the session target to a state file on the host;
    a bridge runs in a container that cannot see it. So the fabricated switch
    below is invisible to the bridge by construction — and this test fails the
    moment the bridge grows a read of session state, which is the failure mode
    the split exists to prevent.
    """
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path)
    deployment(control_system_type="epics", targets={"bluesky": "live"})

    target_state.write_on_start(target_state.TARGET_LIVE)
    before = await _capability()

    target_state.publish_switch(target_state.TARGET_VA, generation=1)
    after = await _capability()

    # The switch really did happen — otherwise this test proves nothing.
    assert target_state.read()["target"] == target_state.TARGET_VA
    assert before == after
    assert after.lane_target == target_state.TARGET_LIVE
