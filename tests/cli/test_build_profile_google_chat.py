"""Tests for the ``gchat_bridge:`` block of the build-profile schema.

Covers the :class:`GChatBridgeProfileConfig` dataclass (including the
``gchat-question`` trigger default, which lives here and nowhere else — the
runtime config's ``from_env`` deliberately applies no trigger default), its raw
parsing, and the :meth:`BuildProfile.validate` checks that keep a declared
bridge deployable: it needs a ``dispatch:`` block to post to, and its trigger
must actually be declared in the resolved source triggers file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli import build_profile as bp
from osprey.cli.build_profile import (
    BuildProfile,
    DispatchConfig,
    GChatBridgeProfileConfig,
    NextcloudBridgeProfileConfig,
    _parse_profile,
)
from osprey.errors import BuildProfileError

_TRIGGERS_YAML = """\
dispatcher:
  dispatch_target: http://dispatch-worker-1:9190
triggers:
  - name: gchat-question
    source: webhook
    action:
      prompt: Answer the operator's question.
  - name: other-trigger
    source: webhook
    action:
      prompt: Something else.
"""


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    """A profile dir holding a ``triggers.yml`` that declares the bridge trigger."""
    (tmp_path / "triggers.yml").write_text(_TRIGGERS_YAML, encoding="utf-8")
    return tmp_path


def _bridge_profile(**kwargs: object) -> BuildProfile:
    """A profile declaring the bridge plus a dispatch block, overridable per test."""
    fields: dict = {
        "name": "x",
        "dispatch": DispatchConfig(triggers="triggers.yml"),
        "gchat_bridge": GChatBridgeProfileConfig(),
    }
    fields.update(kwargs)
    return BuildProfile(**fields)


# ── dataclass + parsing ──────────────────────────────────────────────────────


def test_gchat_bridge_default_trigger() -> None:
    """The profile block is the single source of the 'gchat-question' default.

    The runtime config's from_env applies no trigger default on purpose, so this
    assertion is what pins the name the compose template renders as
    DISPATCH_TRIGGER.
    """
    assert GChatBridgeProfileConfig().trigger == "gchat-question"


def test_gchat_bridge_is_known_key() -> None:
    """'gchat_bridge' is a recognized top-level profile key (no unknown-key warning)."""
    assert "gchat_bridge" in bp._KNOWN_PROFILE_KEYS


def test_gchat_bridge_parse_defaults_when_empty_mapping() -> None:
    """An empty block (`gchat_bridge: {}`) parses to the default trigger."""
    profile = _parse_profile({"name": "x", "gchat_bridge": {}})
    assert profile.gchat_bridge is not None
    assert profile.gchat_bridge.trigger == "gchat-question"


def test_gchat_bridge_parse_round_trip() -> None:
    """An explicit trigger overrides the default through _parse_profile."""
    profile = _parse_profile({"name": "x", "gchat_bridge": {"trigger": "desy-question"}})
    assert profile.gchat_bridge is not None
    assert profile.gchat_bridge.trigger == "desy-question"


def test_no_gchat_bridge_parses_to_none() -> None:
    """A profile without the block leaves the field None (opt-in key)."""
    assert _parse_profile({"name": "x"}).gchat_bridge is None


def test_gchat_bridge_not_a_mapping_raises() -> None:
    """A non-mapping 'gchat_bridge' block raises during parsing."""
    with pytest.raises(BuildProfileError, match="gchat_bridge"):
        _parse_profile({"name": "x", "gchat_bridge": "not-a-mapping"})


def test_gchat_bridge_parses_independently_of_nextcloud() -> None:
    """The two bridge keys are independent: declaring one leaves the other None."""
    profile = _parse_profile({"name": "x", "gchat_bridge": {}})
    assert profile.gchat_bridge is not None
    assert profile.nextcloud_bridge is None


# ── validate(): valid profile ────────────────────────────────────────────────


def test_gchat_bridge_with_dispatch_and_declared_trigger_validates(profile_dir: Path) -> None:
    """The bridge + a dispatch block whose triggers file declares the trigger validates."""
    _bridge_profile().validate(profile_dir)  # must not raise


def test_gchat_bridge_custom_declared_trigger_validates(profile_dir: Path) -> None:
    """Any trigger declared in the triggers file is accepted, not just the default."""
    profile = _bridge_profile(gchat_bridge=GChatBridgeProfileConfig(trigger="other-trigger"))
    profile.validate(profile_dir)  # must not raise


def test_gchat_bridge_resolves_bundled_triggers_file(tmp_path: Path) -> None:
    """A bundled triggers name resolves against the packaged triggers dir.

    Mirrors the dispatch block's own two-candidate lookup (profile-relative
    first, then bundled) — 'hello-dispatch' is declared by the shipped
    tutorial_triggers.yml, which is not in the profile dir.
    """
    profile = _bridge_profile(
        dispatch=DispatchConfig(triggers="tutorial_triggers.yml"),
        gchat_bridge=GChatBridgeProfileConfig(trigger="hello-dispatch"),
    )
    profile.validate(tmp_path)  # must not raise


# ── validate(): (a) missing dispatch block ───────────────────────────────────


def test_gchat_bridge_without_dispatch_raises(profile_dir: Path) -> None:
    """A bridge declared with no dispatch block fails with a message naming the fix."""
    profile = BuildProfile(name="x", gchat_bridge=GChatBridgeProfileConfig())
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    message = str(exc.value)
    assert "gchat_bridge requires a 'dispatch:' block" in message
    assert "gchat-question" in message


def test_gchat_bridge_without_dispatch_skips_trigger_check(profile_dir: Path) -> None:
    """The missing-dispatch error stands alone — no trigger error piles on top.

    There is no triggers file to check against without a dispatch block, so the
    aggregate names one actionable problem rather than two.
    """
    profile = BuildProfile(name="x", gchat_bridge=GChatBridgeProfileConfig())
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    assert "is not declared in" not in str(exc.value)


# ── validate(): (b) trigger missing from the triggers file ───────────────────


def test_gchat_bridge_undeclared_trigger_raises(profile_dir: Path) -> None:
    """A trigger absent from the triggers file fails, listing what is declared."""
    profile = _bridge_profile(gchat_bridge=GChatBridgeProfileConfig(trigger="ghost"))
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    message = str(exc.value)
    assert "gchat_bridge.trigger 'ghost' is not declared in" in message
    assert "gchat-question" in message  # the declared names are listed
    assert "other-trigger" in message


def test_gchat_bridge_default_trigger_undeclared_raises(tmp_path: Path) -> None:
    """The default trigger is not exempt: the shipped tutorial triggers file does
    not declare 'gchat-question', so a bridge left on the default there fails."""
    profile = _bridge_profile(dispatch=DispatchConfig(triggers="tutorial_triggers.yml"))
    with pytest.raises(BuildProfileError, match="gchat-question"):
        profile.validate(tmp_path)


def test_gchat_bridge_empty_trigger_raises(profile_dir: Path) -> None:
    """An explicitly blanked trigger is rejected with its own message."""
    profile = _bridge_profile(gchat_bridge=GChatBridgeProfileConfig(trigger=""))
    with pytest.raises(BuildProfileError, match="gchat_bridge.trigger is required"):
        profile.validate(profile_dir)


def test_gchat_bridge_unresolvable_triggers_file_reports_dispatch_error_only(
    tmp_path: Path,
) -> None:
    """A missing triggers file is the dispatch block's error to report, not a
    second confusing trigger-not-declared error from the bridge check."""
    profile = _bridge_profile(dispatch=DispatchConfig(triggers="nope.yml"))
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(tmp_path)
    message = str(exc.value)
    assert "dispatch.triggers file not found" in message
    assert "is not declared in" not in message


def test_gchat_bridge_unparseable_triggers_file_reports_parse_failure(
    tmp_path: Path,
) -> None:
    """A triggers file that exists but cannot be parsed surfaces as a legible
    'fix that file first' error instead of an unhandled ValueError."""
    (tmp_path / "triggers.yml").write_text(
        "dispatcher:\n  dispatch_target: x\ntriggers:\n  - source: webhook\n", encoding="utf-8"
    )
    with pytest.raises(BuildProfileError) as exc:
        _bridge_profile().validate(tmp_path)
    assert "gchat_bridge.trigger cannot be checked" in str(exc.value)


# ── both bridges at once ─────────────────────────────────────────────────────


def test_both_bridges_validate_side_by_side(profile_dir: Path) -> None:
    """A profile may declare both bridges; each is checked against the same file."""
    profile = _bridge_profile(
        nextcloud_bridge=NextcloudBridgeProfileConfig(trigger="other-trigger"),
    )
    profile.validate(profile_dir)  # must not raise


def test_both_bridges_report_their_own_undeclared_triggers(profile_dir: Path) -> None:
    """Two broken bridges produce two separately-named errors in one raise, so an
    operator fixes both in a single pass rather than one build at a time."""
    profile = _bridge_profile(
        gchat_bridge=GChatBridgeProfileConfig(trigger="ghost-a"),
        nextcloud_bridge=NextcloudBridgeProfileConfig(trigger="ghost-b"),
    )
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    message = str(exc.value)
    assert "gchat_bridge.trigger 'ghost-a' is not declared in" in message
    assert "nextcloud_bridge.trigger 'ghost-b' is not declared in" in message
