"""The write posture of a container that drives the target."""

from __future__ import annotations

import pytest

from osprey.bluesky_bridge_connection import LANE_KEYS
from osprey.deployment.compose_generator import drives_target_write_posture


@pytest.mark.parametrize(
    ("writes", "armed"), [("allowed", True), ("asked", True), ("never", False)]
)
def test_write_posture_is_the_login_tier_on_every_service_with_the_role(
    writes: str, armed: bool
) -> None:
    services = {
        **{lane: {} for lane in LANE_KEYS},
        "dispatch_worker": {},
        "projected": {"drives_target": True},
        "plain": {"drives_target": False},
    }

    posture = drives_target_write_posture(services, {"writes": writes})

    expected = dict.fromkeys((*LANE_KEYS, "dispatch_worker", "projected"), armed)
    assert posture == expected


def test_a_lane_key_with_no_block_is_no_lane() -> None:
    posture = drives_target_write_posture({"bluesky": None}, {"writes": "allowed"})

    assert posture == {}
