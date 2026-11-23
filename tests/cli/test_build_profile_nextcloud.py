"""Tests for the chat-bridge blocks of the build-profile schema.

Every chat bridge (``nextcloud_bridge:``, ``gchat_bridge:``, ``slack_bridge:``,
``teams_bridge:``) keeps the same contract, so this file pins it once,
parametrized over the four in ``BRIDGES``. Covered: each
``*BridgeProfileConfig`` dataclass (including its ``<bridge>-question`` trigger
default, which lives here and nowhere else — the runtime config's ``from_env``
deliberately applies no trigger default), its raw parsing, and the
:meth:`BuildProfile.validate` checks that keep a declared bridge deployable: it
needs a ``dispatch:`` block to post to, and its trigger must actually be
declared in the resolved source triggers file.

What only one bridge has stays in that bridge's own file
(``test_build_profile_{google_chat,slack,teams}.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from osprey.cli import build_profile as bp
from osprey.cli.build_profile import (
    BuildProfile,
    DispatchConfig,
    GChatBridgeProfileConfig,
    NextcloudBridgeProfileConfig,
    SlackBridgeProfileConfig,
    TeamsBridgeProfileConfig,
    _parse_profile,
)
from osprey_connectors.errors import BuildProfileError


@dataclass(frozen=True)
class Bridge:
    """One chat bridge: its profile key (also the BuildProfile field), its
    dataclass, and the trigger name that dataclass defaults to."""

    key: str
    config_cls: type
    default_trigger: str


BRIDGES = [
    Bridge("nextcloud_bridge", NextcloudBridgeProfileConfig, "nextcloud-question"),
    Bridge("gchat_bridge", GChatBridgeProfileConfig, "gchat-question"),
    Bridge("slack_bridge", SlackBridgeProfileConfig, "slack-question"),
    Bridge("teams_bridge", TeamsBridgeProfileConfig, "teams-question"),
]


@pytest.fixture(params=BRIDGES, ids=[b.key.removesuffix("_bridge") for b in BRIDGES])
def bridge(request: pytest.FixtureRequest) -> Bridge:
    """The bridge under test; every test in this file runs once per bridge."""
    return request.param


@pytest.fixture
def profile_dir(tmp_path: Path, bridge: Bridge) -> Path:
    """A profile dir holding a ``triggers.yml`` that declares the bridge trigger."""
    (tmp_path / "triggers.yml").write_text(
        f"""\
dispatcher:
  dispatch_target: http://dispatch-worker-1:9190
triggers:
  - name: {bridge.default_trigger}
    source: webhook
    action:
      prompt: Answer the operator's question.
  - name: other-trigger
    source: webhook
    action:
      prompt: Something else.
""",
        encoding="utf-8",
    )
    return tmp_path


pytestmark = pytest.mark.usefixtures("facility_data_tree")


def _bridge_profile(bridge: Bridge, **kwargs: object) -> BuildProfile:
    """A profile declaring ``bridge`` plus a dispatch block, overridable per test."""
    fields: dict = {
        "name": "x",
        "data": "data",
        "dispatch": DispatchConfig(triggers="triggers.yml"),
        bridge.key: bridge.config_cls(),
    }
    fields.update(kwargs)
    return BuildProfile(**fields)


# ── dataclass + parsing ──────────────────────────────────────────────────────


def test_bridge_default_trigger(bridge: Bridge) -> None:
    """The profile block is the single source of the '<bridge>-question' default.

    The runtime config's from_env applies no trigger default on purpose, so this
    assertion is what pins the name the compose template renders as
    DISPATCH_TRIGGER.
    """
    assert bridge.config_cls().trigger == bridge.default_trigger


def test_bridge_is_known_key(bridge: Bridge) -> None:
    """The bridge key is a recognized top-level profile key (no unknown-key warning)."""
    assert bridge.key in bp._KNOWN_PROFILE_KEYS


def test_bridge_parse_defaults_when_empty_mapping(bridge: Bridge) -> None:
    """An empty block (`<bridge>: {}`) parses to the default trigger."""
    block = getattr(_parse_profile({"name": "x", bridge.key: {}}), bridge.key)
    assert block is not None
    assert block.trigger == bridge.default_trigger


def test_bridge_parse_round_trip(bridge: Bridge) -> None:
    """An explicit trigger overrides the default through _parse_profile."""
    profile = _parse_profile({"name": "x", bridge.key: {"trigger": "desy-question"}})
    block = getattr(profile, bridge.key)
    assert block is not None
    assert block.trigger == "desy-question"


def test_bridge_mentions_default_on(bridge: Bridge) -> None:
    """A block without the key keeps @mentions on."""
    block = getattr(_parse_profile({"name": "x", bridge.key: {}}), bridge.key)
    assert block is not None
    assert block.mentions is True
    assert bridge.config_cls().mentions is True


def test_bridge_mentions_false_parses(bridge: Bridge) -> None:
    profile = _parse_profile({"name": "x", bridge.key: {"mentions": False}})
    block = getattr(profile, bridge.key)
    assert block is not None
    assert block.mentions is False


@pytest.mark.parametrize("value", ["no", 0, None])
def test_bridge_mentions_non_bool_raises(bridge: Bridge, value: object) -> None:
    with pytest.raises(BuildProfileError, match=rf"{bridge.key}\.mentions"):
        _parse_profile({"name": "x", bridge.key: {"mentions": value}})


def test_no_bridge_parses_to_none(bridge: Bridge) -> None:
    """A profile without the block leaves the field None through the loader too (opt-in key)."""
    assert getattr(_parse_profile({"name": "x"}), bridge.key) is None


def test_bridge_not_a_mapping_raises(bridge: Bridge) -> None:
    """A non-mapping bridge block raises during parsing."""
    with pytest.raises(BuildProfileError, match=bridge.key):
        _parse_profile({"name": "x", bridge.key: "not-a-mapping"})


# ── validate(): valid profile ────────────────────────────────────────────────


def test_bridge_with_dispatch_and_declared_trigger_validates(
    bridge: Bridge, profile_dir: Path
) -> None:
    """The bridge + a dispatch block whose triggers file declares the trigger validates."""
    _bridge_profile(bridge).validate(profile_dir)  # must not raise


def test_bridge_custom_declared_trigger_validates(bridge: Bridge, profile_dir: Path) -> None:
    """Any trigger declared in the triggers file is accepted, not just the default."""
    profile = _bridge_profile(bridge, **{bridge.key: bridge.config_cls(trigger="other-trigger")})
    profile.validate(profile_dir)  # must not raise


def test_bridge_resolves_bundled_triggers_file(bridge: Bridge, tmp_path: Path) -> None:
    """A bundled triggers name resolves against the packaged triggers dir.

    Mirrors the dispatch block's own two-candidate lookup (profile-relative
    first, then bundled) — 'hello-dispatch' is declared by the shipped
    tutorial_triggers.yml, which is not in the profile dir.
    """
    profile = _bridge_profile(
        bridge,
        dispatch=DispatchConfig(triggers="tutorial_triggers.yml"),
        **{bridge.key: bridge.config_cls(trigger="hello-dispatch")},
    )
    profile.validate(tmp_path)  # must not raise


# ── validate(): (a) missing dispatch block ───────────────────────────────────


def test_bridge_without_dispatch_raises(bridge: Bridge, profile_dir: Path) -> None:
    """A bridge declared with no dispatch block fails with a message naming the fix."""
    profile = BuildProfile(name="x", **{bridge.key: bridge.config_cls()})
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    message = str(exc.value)
    assert f"{bridge.key} requires a 'dispatch:' block" in message
    assert bridge.default_trigger in message


def test_bridge_without_dispatch_skips_trigger_check(bridge: Bridge, profile_dir: Path) -> None:
    """The missing-dispatch error stands alone — no trigger error piles on top.

    There is no triggers file to check against without a dispatch block, so the
    aggregate names one actionable problem rather than two.
    """
    profile = BuildProfile(name="x", **{bridge.key: bridge.config_cls()})
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    assert "is not declared in" not in str(exc.value)


# ── validate(): (b) trigger missing from the triggers file ───────────────────


def test_bridge_undeclared_trigger_raises(bridge: Bridge, profile_dir: Path) -> None:
    """A trigger absent from the triggers file fails, listing what is declared."""
    profile = _bridge_profile(bridge, **{bridge.key: bridge.config_cls(trigger="ghost")})
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    message = str(exc.value)
    assert f"{bridge.key}.trigger 'ghost' is not declared in" in message
    assert bridge.default_trigger in message  # the declared names are listed
    assert "other-trigger" in message


def test_bridge_default_trigger_undeclared_raises(bridge: Bridge, tmp_path: Path) -> None:
    """The default trigger is not exempt: the shipped tutorial triggers file does
    not declare '<bridge>-question', so a bridge left on the default there fails."""
    profile = _bridge_profile(bridge, dispatch=DispatchConfig(triggers="tutorial_triggers.yml"))
    with pytest.raises(BuildProfileError, match=bridge.default_trigger):
        profile.validate(tmp_path)


def test_bridge_empty_trigger_raises(bridge: Bridge, profile_dir: Path) -> None:
    """An explicitly blanked trigger is rejected with its own message."""
    profile = _bridge_profile(bridge, **{bridge.key: bridge.config_cls(trigger="")})
    with pytest.raises(BuildProfileError, match=f"{bridge.key}.trigger is required"):
        profile.validate(profile_dir)


def test_bridge_unresolvable_triggers_file_reports_dispatch_error_only(
    bridge: Bridge, tmp_path: Path
) -> None:
    """A missing triggers file is the dispatch block's error to report, not a
    second confusing trigger-not-declared error from the bridge check."""
    profile = _bridge_profile(bridge, dispatch=DispatchConfig(triggers="nope.yml"))
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(tmp_path)
    message = str(exc.value)
    assert "dispatch.triggers file not found" in message
    assert "is not declared in" not in message


def test_bridge_unparseable_triggers_file_reports_parse_failure(
    bridge: Bridge, tmp_path: Path
) -> None:
    """A triggers file that exists but cannot be parsed surfaces as a legible
    'fix that file first' error instead of an unhandled ValueError."""
    (tmp_path / "triggers.yml").write_text(
        "dispatcher:\n  dispatch_target: x\ntriggers:\n  - source: webhook\n", encoding="utf-8"
    )
    with pytest.raises(BuildProfileError) as exc:
        _bridge_profile(bridge).validate(tmp_path)
    assert f"{bridge.key}.trigger cannot be checked" in str(exc.value)
