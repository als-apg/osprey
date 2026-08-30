"""Tests for the ``env:`` axis on the build-profile surface.

Covers the per-service name list (``services.<name>.config.env``) and the two
rules :meth:`BuildProfile.validate` enforces on it: the declaration must be a
list of environment variable NAMES, and the dispatch pair carries no ``env:``
of its own — the ``dispatch:`` block rewrites both halves' service config, so
a name authored there would be dropped before it reached a container.

Both authoring surfaces are exercised for every rule: the ``services:`` block
and the dotted ``config:`` overrides reach the same key in the rendered
``config.yml``, so a rule that caught only one of them would be trivially
bypassable. Validation runs on the profile as authored, BEFORE injection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile import BuildProfile, DispatchConfig, ServiceDef, _parse_profile
from osprey.cli.build_profile_schema import env_names_errors
from osprey.errors import BuildProfileError


def _errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """Validate ``profile`` and return the individual accumulated failures."""
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    header, _, body = str(exc.value).partition(":\n  - ")
    assert header == "Build profile validation failed"
    return body.split("\n  - ")


def _env_errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """Return only the failures that mention an ``env`` key."""
    return [e for e in _errors(profile, profile_dir) if ".env" in e]


def _triggers(tmp_path: Path, name: str = "trig.yml") -> str:
    """Write a minimal triggers file into ``tmp_path`` and return its name."""
    (tmp_path / name).write_text("triggers: []", encoding="utf-8")
    return name


def _profile(**raw: Any) -> BuildProfile:
    """Parse a profile from the YAML surface, filling in the required name."""
    return _parse_profile({"name": "envaxis", **raw})


# --- the shared checker ---------------------------------------------------


def test_a_list_of_names_passes_the_shared_checker() -> None:
    """Names matching the env-var pattern are the whole of a valid declaration."""
    assert env_names_errors(["EPICS_CA_ADDR_LIST", "_PRIVATE", "X2"], "services.x.env") == []


def test_an_empty_list_is_valid() -> None:
    """Declaring the axis and passing nothing through is not an error."""
    assert env_names_errors([], "services.x.env") == []


def test_checker_rejects_a_non_list_naming_the_type() -> None:
    """A mapping where a list belongs is reported by name, not by iteration."""
    assert env_names_errors({"FOO": "bar"}, "services.x.env") == [
        "services.x.env must be a list of environment variable names (got dict)"
    ]


def test_checker_tells_a_bare_string_to_become_a_one_item_list() -> None:
    """A scalar would iterate as characters; the message says what to write."""
    (message,) = env_names_errors("EPICS_CA_ADDR_LIST", "services.x.env")
    assert message.startswith(
        "services.x.env must be a list of environment variable names (got str)"
    )
    assert "write it as [EPICS_CA_ADDR_LIST]" in message


def test_checker_names_the_yaml_resolver_for_a_boolean_list() -> None:
    """`env: on` arrives as a bool; the message says why, not just 'bool'."""
    (message,) = env_names_errors(True, "services.x.env")
    assert message.startswith(
        "services.x.env must be a list of environment variable names (got bool)"
    )
    assert "parses as a boolean" in message


@pytest.mark.parametrize("name", ["2LEADING", "WITH-DASH", "WITH SPACE", "", "no-proxy"])
def test_checker_rejects_names_outside_the_pattern(name: str) -> None:
    """The same pattern ``env.required`` and ``env.pinned`` are held to."""
    assert env_names_errors([name], "services.x.env") == [
        f"services.x.env[0] must be an environment variable name matching "
        f"[A-Za-z_][A-Za-z0-9_]* (got {name!r})"
    ]


@pytest.mark.parametrize("name", ["no_proxy", "http_proxy", "https_proxy", "mixedCase_1"])
def test_checker_admits_lowercase_names(name: str) -> None:
    """The proxy family is conventionally lowercase and must be passable and pinnable (#783).

    ``urllib``'s ``getproxies()`` takes whichever spelling comes LAST in the
    environment, and a container runtime that injects the host's lowercase
    ``no_proxy`` after the compose ``environment:`` block silently overrules an
    uppercase-only passthrough — so a deployment has to be able to spell both.
    """
    assert env_names_errors([name], "services.x.env") == []


def test_checker_reports_every_bad_entry_with_its_index() -> None:
    """Per-item messages, so a whole list is fixed in one pass."""
    messages = env_names_errors(["GOOD", "b-ad", "ALSO_GOOD", 7], "services.x.env")
    assert len(messages) == 2
    assert messages[0].startswith("services.x.env[1] must be an environment variable name")
    assert messages[1].startswith("services.x.env[3] must be an environment variable name")


def test_checker_names_the_yaml_resolver_for_a_boolean_entry() -> None:
    """A bare `- on` entry arrives as a bool and is explained, not just echoed."""
    (message,) = env_names_errors([True], "services.x.env")
    assert message.startswith("services.x.env[0] must be an environment variable name")
    assert "parses as a boolean" in message


# --- the accessor ---------------------------------------------------------


def test_service_env_names_applies_the_default() -> None:
    """``env_names()`` is the single place the empty case is spelled."""
    assert ServiceDef(template="osprey.openobserve").env_names() == []
    assert ServiceDef(template="osprey.openobserve", config={}).env_names() == []


def test_service_env_names_keeps_the_authored_order() -> None:
    """Order is the author's — it is what the rendered block shows."""
    svc = ServiceDef(template="osprey.openobserve", config={"env": ["B_VAR", "A_VAR"]})
    assert svc.env_names() == ["B_VAR", "A_VAR"]


def test_service_env_names_hands_back_nothing_for_an_unvalidated_shape() -> None:
    """A scalar reaches a consumer as nothing rather than as its characters."""
    assert ServiceDef(template="osprey.x", config={"env": "EPICS"}).env_names() == []
    assert ServiceDef(template="osprey.x", config={"env": ["OK", 7]}).env_names() == ["OK"]


# --- per-service values, both authoring surfaces --------------------------


def test_service_block_accepts_a_name_list(tmp_path: Path) -> None:
    """A declared service may name host variables to pass through."""
    profile = _profile(
        services={
            "gchat_bridge": {
                "template": "osprey.gchat_bridge",
                "config": {"env": ["EPICS_CA_ADDR_LIST"]},
            }
        }
    )
    profile.validate(tmp_path)


def test_service_block_rejects_an_invalid_name(tmp_path: Path) -> None:
    """A dashed name fails naming the key, the index, and the pattern."""
    profile = _profile(
        services={
            "gchat_bridge": {"template": "osprey.gchat_bridge", "config": {"env": ["ep-ics"]}}
        }
    )
    assert _env_errors(profile, tmp_path) == [
        "services.gchat_bridge.env[0] must be an environment variable name matching "
        "[A-Za-z_][A-Za-z0-9_]* (got 'ep-ics')"
    ]


def test_service_block_rejects_a_non_list(tmp_path: Path) -> None:
    """The shape is checked before the names are."""
    profile = _profile(
        services={"gchat_bridge": {"template": "osprey.gchat_bridge", "config": {"env": "EPICS"}}}
    )
    (message,) = _env_errors(profile, tmp_path)
    assert message.startswith(
        "services.gchat_bridge.env must be a list of environment variable names (got str)"
    )


def test_config_override_surface_rejects_an_invalid_name(tmp_path: Path) -> None:
    """The dotted `config:` spelling is checked the same way as the block."""
    profile = _profile(config={"services.gchat_bridge.env": ["ep-ics"]})
    assert _env_errors(profile, tmp_path) == [
        "services.gchat_bridge.env[0] must be an environment variable name matching "
        "[A-Za-z_][A-Za-z0-9_]* (got 'ep-ics')"
    ]


def test_config_override_surface_accepts_a_name_list(tmp_path: Path) -> None:
    """A valid dotted override is not a failure."""
    _profile(config={"services.gchat_bridge.env": ["EPICS_CA_ADDR_LIST"]}).validate(tmp_path)


def test_yaml_boolean_value_is_reported_not_crashed(tmp_path: Path) -> None:
    """`env: on` is read as a bool by the YAML 1.1 resolver, and reported."""
    raw = yaml.safe_load(
        "name: envaxis\n"
        "services:\n"
        "  gchat_bridge:\n"
        "    template: osprey.gchat_bridge\n"
        "    config:\n"
        "      env: on\n"
    )
    assert raw["services"]["gchat_bridge"]["config"]["env"] is True
    (message,) = _env_errors(_parse_profile(raw), tmp_path)
    assert "services.gchat_bridge.env must be a list of environment variable names" in message
    assert "parses as a boolean" in message


def test_yaml_boolean_entry_is_reported_not_crashed(tmp_path: Path) -> None:
    """A bare `- on` list entry arrives as a bool and is reported per item."""
    raw = yaml.safe_load(
        "name: envaxis\n"
        "services:\n"
        "  gchat_bridge:\n"
        "    template: osprey.gchat_bridge\n"
        "    config:\n"
        "      env:\n"
        "        - HOME\n"
        "        - on\n"
    )
    assert raw["services"]["gchat_bridge"]["config"]["env"] == ["HOME", True]
    (message,) = _env_errors(_parse_profile(raw), tmp_path)
    assert message.startswith("services.gchat_bridge.env[1] must be an environment variable name")


def test_unrelated_dotted_config_keys_are_left_alone(tmp_path: Path) -> None:
    """Only `services.<name>.env` is claimed by the axis."""
    _profile(
        config={
            "services.gchat_bridge.port": 8080,
            "env": "not-a-list",
            "services.gchat_bridge.env.extra": "not-a-list",
            "env.required": ["fine-here"],
        }
    ).validate(tmp_path)


def test_the_axis_does_not_claim_the_profile_level_env_block(tmp_path: Path) -> None:
    """``env:`` at profile level is EnvConfig, a different key with its own rules."""
    profile = _profile(env={"required": ["EPICS_CA_ADDR_LIST"]})
    assert profile.env.required == ["EPICS_CA_ADDR_LIST"]
    profile.validate(tmp_path)


# --- the dispatch pair carries no env of its own --------------------------


@pytest.mark.parametrize("half", ["event_dispatcher", "dispatch_worker"])
def test_per_half_env_in_services_block_is_rejected(tmp_path: Path, half: str) -> None:
    """Either half authoring its own `env:` fails, naming the env chain instead."""
    profile = _profile(
        services={half: {"template": f"osprey.{half}", "config": {"env": ["EPICS_CA_ADDR_LIST"]}}}
    )
    (message,) = _env_errors(profile, tmp_path)
    assert message == (
        f"services.{half}.env is not a per-service knob: the dispatch: block rewrites "
        f"the event dispatcher's and the workers' service config, so services.{half}.env "
        f"would be dropped before it reached a container (got ['EPICS_CA_ADDR_LIST']). "
        f"Both halves already read the project's env chain — put the variable in .env "
        f"and declare it under env.required instead."
    )


@pytest.mark.parametrize("half", ["event_dispatcher", "dispatch_worker"])
def test_per_half_env_in_config_overrides_is_rejected(tmp_path: Path, half: str) -> None:
    """The dotted spelling is not a way around the pair rule."""
    profile = _profile(config={f"services.{half}.env": ["EPICS_CA_ADDR_LIST"]})
    (message,) = _env_errors(profile, tmp_path)
    assert message.startswith(f"services.{half}.env is not a per-service knob")
    assert "env.required" in message


def test_per_half_env_is_rejected_with_no_dispatch_block(tmp_path: Path) -> None:
    """The pair is dispatch-managed whether or not this profile declares it."""
    profile = _profile(
        services={
            "dispatch_worker": {
                "template": "osprey.dispatch_worker",
                "config": {"env": ["EPICS_CA_ADDR_LIST"]},
            }
        }
    )
    assert len(_env_errors(profile, tmp_path)) == 1


def test_per_half_invalid_name_reports_the_pair_rule_only(tmp_path: Path) -> None:
    """One root cause, one error: the key is wrong, not just its contents."""
    profile = _profile(
        services={
            "event_dispatcher": {"template": "osprey.event_dispatcher", "config": {"env": "epics"}}
        }
    )
    (message,) = _env_errors(profile, tmp_path)
    assert "is not a per-service knob" in message
    assert "must be a list of environment variable names" not in message


def test_a_half_authored_on_both_surfaces_is_reported_once(tmp_path: Path) -> None:
    """Two spellings of one mistake are one message, not two."""
    profile = _profile(
        services={
            "event_dispatcher": {
                "template": "osprey.event_dispatcher",
                "config": {"env": ["EPICS_CA_ADDR_LIST"]},
            }
        },
        config={"services.event_dispatcher.env": ["EPICS_CA_ADDR_LIST"]},
    )
    assert len(_env_errors(profile, tmp_path)) == 1


def test_both_halves_are_reported_separately(tmp_path: Path) -> None:
    """Each half names its own key, so both get fixed in one pass."""
    profile = _profile(
        services={
            "event_dispatcher": {
                "template": "osprey.event_dispatcher",
                "config": {"env": ["EPICS_CA_ADDR_LIST"]},
            },
            "dispatch_worker": {
                "template": "osprey.dispatch_worker",
                "config": {"env": ["EPICS_CA_ADDR_LIST"]},
            },
        }
    )
    messages = _env_errors(profile, tmp_path)
    assert len(messages) == 2
    assert any("services.event_dispatcher.env" in m for m in messages)
    assert any("services.dispatch_worker.env" in m for m in messages)


# --- the accumulation contract --------------------------------------------


def test_env_failures_accumulate_with_the_network_axis_and_unrelated_ones(
    tmp_path: Path,
) -> None:
    """The axis appends like every other check — no fail-fast, no short-circuit."""
    profile = BuildProfile(
        name="",
        tier=2,
        services={
            "gchat_bridge": ServiceDef(
                template="osprey.gchat_bridge",
                config={"network": "hostnet", "env": ["ep-ics"]},
            )
        },
    )
    messages = _errors(profile, tmp_path)
    assert "Profile 'name' is required" in messages
    assert "tier must be 1 or 3 (got 2)" in messages
    assert "services.gchat_bridge.network must be one of bridge, host (got 'hostnet')" in messages
    assert (
        "services.gchat_bridge.env[0] must be an environment variable name matching "
        "[A-Za-z_][A-Za-z0-9_]* (got 'ep-ics')" in messages
    )


def test_a_malformed_service_key_is_reported_once_for_all_axes(tmp_path: Path) -> None:
    """Two axes walk the services mapping; a bool key is still one message."""
    raw = yaml.safe_load("name: envaxis\nservices:\n  on:\n    template: osprey.openobserve\n")
    assert True in raw["services"]
    messages = _errors(_parse_profile(raw), tmp_path)
    assert len([m for m in messages if m.startswith("services keys must be strings")]) == 1


def test_a_malformed_config_key_is_reported_once_for_all_axes(tmp_path: Path) -> None:
    """Same for the dotted override surface."""
    raw = yaml.safe_load("name: envaxis\nconfig:\n  off: 1\n")
    assert False in raw["config"]
    messages = _errors(_parse_profile(raw), tmp_path)
    assert len([m for m in messages if m.startswith("config keys must be strings")]) == 1


def test_a_profile_declaring_no_env_anywhere_stays_clean(tmp_path: Path) -> None:
    """The axis is inert when unset — an untouched profile validates as before."""
    profile = BuildProfile(
        name="envaxis",
        dispatch=DispatchConfig(triggers=_triggers(tmp_path)),
        services={
            "gchat_bridge": ServiceDef(template="osprey.gchat_bridge", config={"port": 8080})
        },
    )
    profile.validate(tmp_path)
