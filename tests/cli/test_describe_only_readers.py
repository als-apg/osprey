"""Describe-only readers answer from the login tier alone (PR-2).

Each reader here describes or entitles; none decides a write. They read
``control_system.writes`` — the tier, target-blind — rather than the retired
``writes_enabled`` leaves, so a normalized document with no retired leaf at all
answers the same as the spelling it was rewritten from, and a tier that allows
writes allows them for every machine the render reaches.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from osprey.cli.profile_card import _write_rights
from osprey.deployment.web_terminals.artifacts import BashLaunchTokenConflictError
from osprey.deployment.web_terminals.lint import Finding, lint_profile_config
from osprey.deployment.web_terminals.roster import (
    config_needs_launch_token,
    config_needs_launch_token_for,
    launch_token_writes_key,
)
from osprey.utils.workspace import TIERS_DIRNAME
from osprey_connectors.types import WRITES_KEY
from tests._renders import tier_render

# ---------------------------------------------------------------------------
# Launch tokens (roster.py)
# ---------------------------------------------------------------------------


def _two_lane_render(control_system: dict, *, bluesky: bool = True) -> dict:
    """A live-baseline render that also carries the VA lane."""
    return {
        "control_system": {
            "type": "epics",
            "connector": {"epics": {}, "virtual_accelerator": {}},
            **control_system,
        },
        "services": {"bluesky_va": {"target": "va", "port": 10081}},
        "claude_code": {"servers": {"bluesky": {"enabled": bluesky}}},
    }


def test_an_allowed_tier_entitles_every_lane_the_render_carries() -> None:
    config = _two_lane_render({"writes": "allowed"})

    assert config_needs_launch_token_for(config, "bluesky") is True
    assert config_needs_launch_token_for(config, "bluesky_va") is True
    assert config_needs_launch_token(config) is True


def test_a_lane_the_render_does_not_carry_entitles_nothing_under_an_allowed_tier() -> None:
    config = _two_lane_render({"writes": "allowed"})

    assert config_needs_launch_token_for(config, "bluesky_live") is False


@pytest.mark.parametrize("value", ["never", "yes", True, None])
def test_only_the_literal_allowed_entitles_a_lane(value: object) -> None:
    config = _two_lane_render({"writes": value})

    assert config_needs_launch_token_for(config, "bluesky") is False
    assert config_needs_launch_token_for(config, "bluesky_va") is False


def test_a_silent_tier_entitles_nothing() -> None:
    config = _two_lane_render({})

    assert config_needs_launch_token_for(config, "bluesky") is False
    assert config_needs_launch_token_for(config, "bluesky_va") is False


@pytest.mark.parametrize(
    "control_system",
    [
        {"writes_enabled": True, "connector": {"epics": {}, "virtual_accelerator": {}}},
        {
            "writes_enabled": False,
            "connector": {"epics": {}, "virtual_accelerator": {"writes_enabled": True}},
        },
    ],
    ids=["flat-armed", "connector-armed"],
)
def test_a_retired_spelling_alone_entitles_nothing(control_system: dict) -> None:
    # A render reads the tier key only: an armed retired leaf with no tier key
    # is `never` (rows tier-absent-old-armed-flat / -connector, ruling D2). The
    # build rewrites the leaf; a render that skipped the rebuild entitles no lane.
    config = _two_lane_render(control_system)

    assert config_needs_launch_token_for(config, "bluesky") is False
    assert config_needs_launch_token_for(config, "bluesky_va") is False


def test_an_allowed_tier_without_the_bluesky_server_entitles_nothing() -> None:
    config = _two_lane_render({"writes": "allowed"}, bluesky=False)

    assert config_needs_launch_token_for(config, "bluesky") is False
    assert config_needs_launch_token_for(config, "bluesky_va") is False


@pytest.mark.parametrize("lane", ["bluesky", "bluesky_va"])
def test_every_lane_names_the_tier_key(lane: str) -> None:
    assert launch_token_writes_key(_two_lane_render({"writes": "allowed"}), lane) == WRITES_KEY


def test_the_bash_refusal_names_the_tier_key_and_no_retired_leaf() -> None:
    for error in (
        BashLaunchTokenConflictError(
            ["rw"], by_lane={"bluesky_va": ["rw"]}, writes_keys={"bluesky_va": WRITES_KEY}
        ),
        BashLaunchTokenConflictError(["rw"]),
    ):
        message = str(error)
        assert f"{WRITES_KEY}: never" in message
        assert "writes_enabled" not in message


# ---------------------------------------------------------------------------
# The profile card (profile_card.py)
# ---------------------------------------------------------------------------


def _card_profile(config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        config={
            "control_system.type": "epics",
            "control_system.connector.epics.gateways": ["gw.example:5064"],
            "control_system.connector.virtual_accelerator.port": 5064,
            **config,
        }
    )


def test_a_switch_capable_render_prints_one_tier_wide_right() -> None:
    profile = _card_profile({})
    deltas = {"rw": {"config": {"control_system.writes": "allowed"}}}

    assert _write_rights(profile, deltas, "rw") == ["rights allowed"]


def test_a_never_layer_over_an_allowed_host_prints_no_right() -> None:
    profile = _card_profile({"control_system.writes": "allowed"})
    deltas = {"ro": {"config": {"control_system.writes": "never"}}}

    assert _write_rights(profile, deltas, "ro") == []


def test_a_silent_layer_inherits_the_host_tier() -> None:
    profile = _card_profile({"control_system.writes": "allowed"})

    assert _write_rights(profile, {"plain": {"config": {"web.theme": "main"}}}, "plain") == [
        "rights allowed"
    ]
    assert _write_rights(_card_profile({}), {}, "plain") == []


def test_a_retired_layer_is_read_as_its_rewrite_over_a_normalized_host() -> None:
    # The layer is a raw parse: its flat `false` is `never`, and it overrides the
    # host's `allowed` — a merge of the raw section would answer `allowed`.
    profile = _card_profile({"control_system.writes": "allowed"})
    readonly = {"ro": {"config": {"control_system.writes_enabled": False}}}
    va_write = {
        "va": {
            "config": {
                "control_system.writes_enabled": False,
                "control_system.connector.virtual_accelerator.writes_enabled": True,
            }
        }
    }

    assert _write_rights(profile, readonly, "ro") == []
    assert _write_rights(_card_profile({}), va_write, "va") == ["rights allowed"]


# ---------------------------------------------------------------------------
# The live-writer lint (web_terminals/lint.py)
# ---------------------------------------------------------------------------

_LIVE_WRITER_CODE = "web_terminals.live_writer_without_control_identity"


def _walled_config(tmp_path, delta_config: dict, **profile_keys: object) -> dict:
    (tmp_path / TIERS_DIRNAME).mkdir(exist_ok=True)
    (tmp_path / TIERS_DIRNAME / "tier.yml").write_text(
        yaml.safe_dump({"name": "Tier", "config": delta_config})
    )
    return {
        "deploy.fqdn": "127.0.0.1",
        "modules.web_terminals": {
            "enabled": True,
            "image_source": "local",
            "auth": {"method": "password"},
            "default_tier": "tier",
            "users": [{"name": "alice", "index": 0, "tier": "tier"}],
            "renders": {
                "tier": tier_render(
                    "tier",
                    "ca-tier",
                    project_path="../ca-tier",
                    build_profile=f"{TIERS_DIRNAME}/tier.yml",
                ),
            },
        },
        **profile_keys,
    }


def _live_writers(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.code == _LIVE_WRITER_CODE]


def test_an_allowed_tier_reaching_live_is_a_live_writer(tmp_path) -> None:
    config = _walled_config(
        tmp_path, {"control_system.type": "epics", "control_system.writes": "allowed"}
    )

    findings = _live_writers(lint_profile_config(config, profile_root=tmp_path))

    assert [f.severity for f in findings] == ["warn"], findings
    assert WRITES_KEY in findings[0].message


def test_a_never_tier_reaching_live_is_not_a_live_writer(tmp_path) -> None:
    config = _walled_config(
        tmp_path, {"control_system.type": "epics", "control_system.writes": "never"}
    )

    assert _live_writers(lint_profile_config(config, profile_root=tmp_path)) == []


def test_a_retired_false_tier_over_an_allowed_profile_is_not_a_live_writer(tmp_path) -> None:
    # Each layer is read as the build rewrites it, so the tier's flat `false`
    # is `never` and overrides the profile's `allowed`.
    config = _walled_config(
        tmp_path,
        {"control_system.writes_enabled": False},
        **{"control_system.type": "epics", "control_system.writes": "allowed"},
    )

    assert _live_writers(lint_profile_config(config, profile_root=tmp_path)) == []


def test_an_allowed_tier_on_a_simulator_only_deployment_is_not_a_live_writer(tmp_path) -> None:
    config = _walled_config(
        tmp_path,
        {
            "control_system.type": "virtual_accelerator",
            "control_system.connector.virtual_accelerator.serving": "in_process",
            "control_system.writes": "allowed",
        },
    )

    assert _live_writers(lint_profile_config(config, profile_root=tmp_path)) == []
