"""Tests for the ``trajectory:`` block of the build-profile schema.

The block's PRESENCE is the switch that deploys the trajectory service: there is
no ``enabled`` key. It carries four optional knobs (``target``, ``model``,
``network``, ``bind``) and no port — the port is the layout's ``trajectory``
slot, moved only by ``services.trajectory.port`` like every other slot. Covers
parsing, the closed key set, validation, the port ledger's slot claim, the
url-less ``web_panels`` carve-out, and the emitter's commented template.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_emit import (
    _COMMENTED_TEMPLATE_KEYS,
    _COMMENTED_TEMPLATE_ORDER,
    _COMMENTED_TEMPLATES,
    emit_standalone_profile_yaml,
)
from osprey.cli.build_profile_load import _KNOWN_PROFILE_KEYS, _parse_profile
from osprey.cli.build_profile_model import BuildProfile
from osprey.cli.build_profile_schema import DEFAULT_NETWORK_MODE, TrajectoryConfig
from osprey.errors import BuildProfileError
from osprey.port_layout import SLOTS_BY_NAME, default_port


@pytest.fixture(autouse=True)
def _facility_data_tree(tmp_path: Path) -> None:
    """The tree every profile's ``data:`` key names, beside the profile."""
    (tmp_path / "data").mkdir(exist_ok=True)


def _profile(**raw: Any) -> BuildProfile:
    return _parse_profile({"name": "traj", "data": "data", **raw})


def _validation_errors(profile: BuildProfile, profile_dir: Path) -> str:
    try:
        profile.validate(profile_dir)
    except BuildProfileError as exc:
        return str(exc)
    return ""


# ── the port slot ────────────────────────────────────────────────────────────


def test_trajectory_slot_is_offset_82_in_the_services_tier() -> None:
    slot = SLOTS_BY_NAME["trajectory"]
    assert slot.offset == 82
    assert slot.tier == "services"
    assert slot.config_key == "services.trajectory.port"
    assert default_port("trajectory", base=20000) == 20082


# ── parsing ──────────────────────────────────────────────────────────────────


def test_trajectory_is_a_known_profile_key() -> None:
    assert "trajectory" in _KNOWN_PROFILE_KEYS


def test_absent_block_leaves_the_field_none() -> None:
    assert _profile().trajectory is None
    assert BuildProfile(name="x").trajectory is None


def test_empty_block_switches_the_service_on_with_defaults() -> None:
    profile = _profile(trajectory={})
    assert profile.trajectory == TrajectoryConfig()
    assert profile.trajectory.target is None
    assert profile.trajectory.model is None
    assert profile.trajectory.network == DEFAULT_NETWORK_MODE
    assert profile.trajectory.bind is None


def test_every_knob_round_trips() -> None:
    profile = _profile(
        trajectory={"target": "va", "model": "sr", "network": "host", "bind": "127.0.0.1"}
    )
    assert profile.trajectory == TrajectoryConfig(
        target="va", model="sr", network="host", bind="127.0.0.1"
    )


def test_the_block_must_be_a_mapping() -> None:
    with pytest.raises(BuildProfileError, match="'trajectory' must be a mapping"):
        _profile(trajectory=["host"])


@pytest.mark.parametrize("key", ["enabled", "port", "netwrok"])
def test_unknown_keys_are_refused(key: str) -> None:
    """``enabled`` and ``port`` are refused by name: presence is the switch, and
    the port belongs to the layout slot."""
    with pytest.raises(BuildProfileError, match=rf"'{key}'"):
        _profile(trajectory={key: True})


def test_the_key_set_is_exactly_the_four_knobs() -> None:
    from osprey.cli.build_profile_load import _KNOWN_TRAJECTORY_KEYS

    assert _KNOWN_TRAJECTORY_KEYS == {"target", "model", "network", "bind"}


# ── validation ───────────────────────────────────────────────────────────────


def test_a_well_formed_block_validates(tmp_path: Path) -> None:
    profile = _profile(trajectory={"target": "va", "network": "host", "bind": "0.0.0.0"})
    assert "trajectory" not in _validation_errors(profile, tmp_path)


@pytest.mark.parametrize(
    ("block", "fragment"),
    [
        ({"network": "overlay"}, "trajectory.network must be one of bridge, host"),
        ({"network": True}, "trajectory.network must be one of"),
        ({"target": ""}, "trajectory.target"),
        ({"target": 3}, "trajectory.target"),
        ({"model": "  "}, "trajectory.model"),
        ({"model": ["sr"]}, "trajectory.model"),
        ({"bind": ""}, "trajectory.bind"),
        ({"bind": "0.0.0.0 "}, "trajectory.bind"),
        ({"bind": 8080}, "trajectory.bind"),
    ],
)
def test_malformed_knobs_are_reported(tmp_path: Path, block: dict, fragment: str) -> None:
    errors = _validation_errors(_profile(trajectory=block), tmp_path)
    assert fragment in errors


@pytest.mark.parametrize("port", [0, 70000, "10082", True])
def test_an_out_of_range_port_override_is_reported(tmp_path: Path, port: object) -> None:
    profile = _profile(trajectory={}, config={"services.trajectory.port": port})
    assert "services.trajectory.port must be a port in 1..65535" in _validation_errors(
        profile, tmp_path
    )


def test_a_port_override_without_the_block_is_not_this_blocks_concern(tmp_path: Path) -> None:
    profile = _profile(config={"services.trajectory.port": 0})
    assert "services.trajectory.port must be" not in _validation_errors(profile, tmp_path)


# ── the port ledger ──────────────────────────────────────────────────────────


def test_the_block_claims_the_trajectory_slot_at_the_resolved_base() -> None:
    claimed = _profile(trajectory={}, config={"deployment.port_base": 20000})._claimed_ports()
    assert claimed["services.trajectory.port"] == 20082


def test_no_block_claims_no_trajectory_port() -> None:
    claimed = _profile()._claimed_ports()
    assert "services.trajectory.port" not in claimed
    assert default_port("trajectory") not in claimed.values()


def test_an_override_moves_the_claim_and_the_slot_is_not_also_seeded() -> None:
    claimed = _profile(trajectory={}, config={"services.trajectory.port": 12345})._claimed_ports()
    assert claimed["services.trajectory.port"] == 12345
    assert default_port("trajectory") not in claimed.values()


def test_an_attached_project_publishes_no_trajectory_port() -> None:
    claimed = _profile(trajectory={}, deploy_services=False)._claimed_ports()
    assert "services.trajectory.port" not in claimed


# ── the web_panels carve-out ─────────────────────────────────────────────────


def test_a_url_less_trajectory_panel_is_accepted_with_the_block(tmp_path: Path) -> None:
    profile = _profile(trajectory={}, web_panels=["trajectory"])
    assert "Unknown web_panel 'trajectory'" not in _validation_errors(profile, tmp_path)


def test_a_url_less_trajectory_panel_is_refused_without_the_block(tmp_path: Path) -> None:
    profile = _profile(web_panels=["trajectory"])
    assert "Unknown web_panel 'trajectory'" in _validation_errors(profile, tmp_path)


# ── emission ─────────────────────────────────────────────────────────────────


def test_the_block_is_an_opt_in_commented_template() -> None:
    assert "trajectory" in _COMMENTED_TEMPLATE_KEYS
    assert "trajectory" in _COMMENTED_TEMPLATE_ORDER
    template = _COMMENTED_TEMPLATES["trajectory"]
    assert "\n# trajectory:" in template
    for knob in ("target", "model", "network", "bind"):
        assert f"#   {knob}:" in template
    assert "enabled" not in template


def test_a_profile_without_the_block_emits_the_commented_template() -> None:
    text = emit_standalone_profile_yaml("hello-world", (), "Emitted", project_name="emitted")
    assert "trajectory" not in (yaml.safe_load(text) or {})
    assert _COMMENTED_TEMPLATES["trajectory"] in text
