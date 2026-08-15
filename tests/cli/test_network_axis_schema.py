"""Tests for the ``network:`` axis on the build-profile surface.

Covers the knob itself (``dispatch.network``, ``services.<name>.config.network``)
and the two rules :meth:`BuildProfile.validate` enforces on it: the value must
name a mode in the closed vocabulary, and the dispatch pair keeps exactly ONE
knob — a ``network:`` authored on ``event_dispatcher`` or ``dispatch_worker``
is rejected whatever its value, because the build writes ``dispatch.network``
into both halves itself.

Both authoring surfaces are exercised for every rule: the ``services:`` block
and the dotted ``config:`` overrides reach the same key in the rendered
``config.yml``, so a rule that caught only one of them would be trivially
bypassable. Validation runs on the profile as authored, BEFORE injection, so
these checks see what a person wrote rather than what the build is about to
write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile import BuildProfile, DispatchConfig, ServiceDef, _parse_profile
from osprey.cli.build_profile_schema import (
    DEFAULT_NETWORK_MODE,
    VALID_NETWORK_MODES,
    network_mode_errors,
)
from osprey.errors import BuildProfileError


def _errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """Validate ``profile`` and return the individual accumulated failures."""
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    header, _, body = str(exc.value).partition(":\n  - ")
    assert header == "Build profile validation failed"
    return body.split("\n  - ")


def _network_errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """Return only the failures that mention a ``network`` key."""
    return [e for e in _errors(profile, profile_dir) if "network" in e]


def _triggers(tmp_path: Path, name: str = "trig.yml") -> str:
    """Write a minimal triggers file into ``tmp_path`` and return its name."""
    (tmp_path / name).write_text("triggers: []", encoding="utf-8")
    return name


def _profile(**raw: Any) -> BuildProfile:
    """Parse a profile from the YAML surface, filling in the required name."""
    return _parse_profile({"name": "netaxis", **raw})


# --- the vocabulary -------------------------------------------------------


def test_vocabulary_is_bridge_and_host_with_bridge_the_default() -> None:
    """The closed set is exactly the two modes, defaulting to the old behaviour."""
    assert VALID_NETWORK_MODES == ("bridge", "host")
    assert DEFAULT_NETWORK_MODE == "bridge"


def test_dispatch_network_defaults_to_bridge() -> None:
    """A dispatch block that declares nothing keeps the compose network."""
    assert DispatchConfig(triggers="t.yml").network == "bridge"


def test_service_network_mode_applies_the_default() -> None:
    """``network_mode()`` is the single place the per-service default is spelled."""
    assert ServiceDef(template="osprey.openobserve").network_mode() == "bridge"
    assert ServiceDef(template="osprey.openobserve", config={}).network_mode() == "bridge"
    assert (
        ServiceDef(template="osprey.openobserve", config={"network": "host"}).network_mode()
        == "host"
    )


@pytest.mark.parametrize("mode", VALID_NETWORK_MODES)
def test_every_valid_mode_passes_the_shared_checker(mode: str) -> None:
    """Each member of the vocabulary is accepted by the shared helper."""
    assert network_mode_errors(mode, "services.x.network") == []


def test_checker_names_the_key_and_the_allowed_set() -> None:
    """An invalid value reports where it was written and what was allowed."""
    assert network_mode_errors("hostnet", "services.x.network") == [
        "services.x.network must be one of bridge, host (got 'hostnet')"
    ]


def test_checker_names_the_yaml_resolver_for_a_boolean() -> None:
    """`network: on` arrives as a bool; the message says why, not just 'True'."""
    (message,) = network_mode_errors(True, "services.x.network")
    assert message.startswith("services.x.network must be one of bridge, host (got True)")
    assert "parses as a boolean" in message
    assert "quote the value" in message


# --- per-service values, both authoring surfaces --------------------------


def test_service_block_accepts_host(tmp_path: Path) -> None:
    """A declared service may move to the host network."""
    profile = _profile(
        services={
            "gchat_bridge": {"template": "osprey.gchat_bridge", "config": {"network": "host"}}
        }
    )
    profile.validate(tmp_path)


def test_service_block_rejects_an_invalid_value(tmp_path: Path) -> None:
    """An invented mode fails naming the key and the allowed set."""
    profile = _profile(
        services={
            "gchat_bridge": {"template": "osprey.gchat_bridge", "config": {"network": "hostnet"}}
        }
    )
    assert _network_errors(profile, tmp_path) == [
        "services.gchat_bridge.network must be one of bridge, host (got 'hostnet')"
    ]


def test_config_override_surface_rejects_an_invalid_value(tmp_path: Path) -> None:
    """The dotted `config:` spelling is checked the same way as the block."""
    profile = _profile(config={"services.gchat_bridge.network": "hostnet"})
    assert _network_errors(profile, tmp_path) == [
        "services.gchat_bridge.network must be one of bridge, host (got 'hostnet')"
    ]


def test_config_override_surface_accepts_host(tmp_path: Path) -> None:
    """A valid dotted override is not a failure."""
    _profile(config={"services.gchat_bridge.network": "host"}).validate(tmp_path)


def test_yaml_boolean_value_is_reported_not_crashed(tmp_path: Path) -> None:
    """`network: on` is read as a bool by the YAML 1.1 resolver, and reported."""
    raw = yaml.safe_load(
        "name: netaxis\n"
        "services:\n"
        "  gchat_bridge:\n"
        "    template: osprey.gchat_bridge\n"
        "    config:\n"
        "      network: on\n"
    )
    assert raw["services"]["gchat_bridge"]["config"]["network"] is True
    (message,) = _network_errors(_parse_profile(raw), tmp_path)
    assert "services.gchat_bridge.network must be one of bridge, host (got True)" in message
    assert "parses as a boolean" in message


def test_unrelated_dotted_config_keys_are_left_alone(tmp_path: Path) -> None:
    """Only `services.<name>.network` is claimed by the axis."""
    _profile(
        config={
            "services.gchat_bridge.port": 8080,
            "network": "hostnet",
            "services.gchat_bridge.network.mode": "hostnet",
        }
    ).validate(tmp_path)


# --- the dispatch pair's single knob --------------------------------------


def test_dispatch_network_reaches_the_config_from_the_yaml_surface(tmp_path: Path) -> None:
    """A profile-authored `dispatch.network:` must survive parsing.

    The parser pulls each dispatch field by name, so a field the dataclass
    grows but the parser never reads would be dropped in silence: the axis
    would look declared, validate clean, and render bridge anyway. This pins
    the value all the way from the YAML surface to the parsed block.
    """
    profile = _profile(dispatch={"triggers": _triggers(tmp_path), "network": "host"})
    assert profile.dispatch is not None
    assert profile.dispatch.network == "host"
    profile.validate(tmp_path)


def test_dispatch_without_a_network_key_parses_as_bridge(tmp_path: Path) -> None:
    """An untouched dispatch block keeps the compose network through parsing."""
    profile = _profile(dispatch={"triggers": _triggers(tmp_path)})
    assert profile.dispatch is not None
    assert profile.dispatch.network == "bridge"


def test_dispatch_network_invalid_value_survives_parsing_to_be_reported(tmp_path: Path) -> None:
    """A typo reaches the validator rather than being dropped by the parser."""
    profile = _profile(dispatch={"triggers": _triggers(tmp_path), "network": "hostnet"})
    assert _network_errors(profile, tmp_path) == [
        "dispatch.network must be one of bridge, host (got 'hostnet')"
    ]


def test_dispatch_network_accepts_host(tmp_path: Path) -> None:
    """The pair moves via its own knob."""
    profile = BuildProfile(
        name="netaxis",
        dispatch=DispatchConfig(triggers=_triggers(tmp_path), network="host"),
    )
    profile.validate(tmp_path)


def test_dispatch_network_rejects_an_invalid_value(tmp_path: Path) -> None:
    """An invented mode on the pair knob names the key and the allowed set."""
    profile = BuildProfile(
        name="netaxis",
        dispatch=DispatchConfig(triggers=_triggers(tmp_path), network="hostnet"),  # type: ignore[arg-type]
    )
    assert _network_errors(profile, tmp_path) == [
        "dispatch.network must be one of bridge, host (got 'hostnet')"
    ]


@pytest.mark.parametrize("half", ["event_dispatcher", "dispatch_worker"])
def test_per_half_network_in_services_block_is_rejected(tmp_path: Path, half: str) -> None:
    """Either half authoring its own `network:` fails, naming the pair's knob."""
    profile = _profile(
        services={half: {"template": f"osprey.{half}", "config": {"network": "host"}}}
    )
    (message,) = _network_errors(profile, tmp_path)
    assert message == (
        f"services.{half}.network is not a per-service knob: the event dispatcher "
        f"and its workers share one network, set by dispatch.network: (got 'host'). "
        f"Remove services.{half}.network and set dispatch.network instead."
    )


@pytest.mark.parametrize("half", ["event_dispatcher", "dispatch_worker"])
def test_per_half_network_in_config_overrides_is_rejected(tmp_path: Path, half: str) -> None:
    """The dotted spelling is not a way around the pair's single knob."""
    profile = _profile(config={f"services.{half}.network": "host"})
    (message,) = _network_errors(profile, tmp_path)
    assert message.startswith(f"services.{half}.network is not a per-service knob")
    assert "set by dispatch.network:" in message


def test_per_half_network_is_rejected_even_when_it_agrees(tmp_path: Path) -> None:
    """A second spelling that happens to agree today drifts tomorrow — reject it."""
    profile = BuildProfile(
        name="netaxis",
        dispatch=DispatchConfig(triggers=_triggers(tmp_path), network="host"),
        services={
            "event_dispatcher": ServiceDef(
                template="osprey.event_dispatcher", config={"network": "host"}
            )
        },
    )
    (message,) = _network_errors(profile, tmp_path)
    assert message.startswith("services.event_dispatcher.network is not a per-service knob")


def test_per_half_network_is_rejected_with_no_dispatch_block(tmp_path: Path) -> None:
    """The pair is dispatch-managed whether or not this profile declares it."""
    profile = _profile(
        services={
            "dispatch_worker": {"template": "osprey.dispatch_worker", "config": {"network": "host"}}
        }
    )
    assert len(_network_errors(profile, tmp_path)) == 1


def test_per_half_invalid_value_reports_the_pair_rule_only(tmp_path: Path) -> None:
    """One root cause, one error: the key is wrong, not just its value."""
    profile = _profile(
        services={
            "event_dispatcher": {
                "template": "osprey.event_dispatcher",
                "config": {"network": "hostnet"},
            }
        }
    )
    (message,) = _network_errors(profile, tmp_path)
    assert "is not a per-service knob" in message
    assert "must be one of" not in message


def test_a_half_authored_on_both_surfaces_is_reported_once(tmp_path: Path) -> None:
    """Two spellings of one mistake are one message, not two."""
    profile = _profile(
        services={
            "event_dispatcher": {
                "template": "osprey.event_dispatcher",
                "config": {"network": "host"},
            }
        },
        config={"services.event_dispatcher.network": "host"},
    )
    assert len(_network_errors(profile, tmp_path)) == 1


def test_both_halves_are_reported_separately(tmp_path: Path) -> None:
    """Each half names its own key, so both get fixed in one pass."""
    profile = _profile(
        services={
            "event_dispatcher": {
                "template": "osprey.event_dispatcher",
                "config": {"network": "host"},
            },
            "dispatch_worker": {
                "template": "osprey.dispatch_worker",
                "config": {"network": "host"},
            },
        }
    )
    messages = _network_errors(profile, tmp_path)
    assert len(messages) == 2
    assert any("services.event_dispatcher.network" in m for m in messages)
    assert any("services.dispatch_worker.network" in m for m in messages)


# --- YAML 1.1 key guards --------------------------------------------------


def test_boolean_service_name_is_reported_not_crashed(tmp_path: Path) -> None:
    """A bare `on:` service name arrives as a bool key and must not crash the scan."""
    raw = yaml.safe_load("name: netaxis\nservices:\n  on:\n    template: osprey.openobserve\n")
    assert True in raw["services"]
    messages = _errors(_parse_profile(raw), tmp_path)
    assert any(
        m.startswith("services keys must be strings (got True)") and "quote the service name" in m
        for m in messages
    )


def test_boolean_config_key_is_reported_not_crashed(tmp_path: Path) -> None:
    """A bare `off:` config key arrives as a bool key and must not crash the scan."""
    raw = yaml.safe_load("name: netaxis\nconfig:\n  off: 1\n")
    assert False in raw["config"]
    messages = _errors(_parse_profile(raw), tmp_path)
    assert any(
        m.startswith("config keys must be strings (got False)") and "quote the key" in m
        for m in messages
    )


# --- the accumulation contract --------------------------------------------


def test_network_failures_accumulate_with_unrelated_ones(tmp_path: Path) -> None:
    """The axis appends like every other check — no fail-fast, no short-circuit."""
    profile = BuildProfile(
        name="",
        tier=2,
        services={
            "gchat_bridge": ServiceDef(
                template="osprey.gchat_bridge", config={"network": "hostnet"}
            )
        },
    )
    messages = _errors(profile, tmp_path)
    assert "Profile 'name' is required" in messages
    assert "tier must be 1 or 3 (got 2)" in messages
    assert "services.gchat_bridge.network must be one of bridge, host (got 'hostnet')" in messages


def test_a_profile_declaring_no_network_anywhere_stays_clean(tmp_path: Path) -> None:
    """The axis is inert when unset — an untouched profile validates as before."""
    profile = BuildProfile(
        name="netaxis",
        dispatch=DispatchConfig(triggers=_triggers(tmp_path)),
        services={
            "gchat_bridge": ServiceDef(template="osprey.gchat_bridge", config={"port": 8080})
        },
    )
    profile.validate(tmp_path)
