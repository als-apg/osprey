"""Tests for the ``slack_bridge:`` block of the build-profile schema.

Covers the :class:`SlackBridgeProfileConfig` dataclass (including the
``slack-question`` trigger default, which lives here and nowhere else — the
runtime config's ``from_env`` deliberately applies no trigger default), its raw
parsing, and the :meth:`BuildProfile.validate` checks that keep a declared
bridge deployable: it needs a ``dispatch:`` block to post to, and its trigger
must actually be declared in the resolved source triggers file. Also covers
the commented stanza an emitted profile carries for the block and the
profile card's bridge row.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli import build_profile as bp
from osprey.cli import build_profile_emit as emit
from osprey.cli.build_profile import (
    BuildProfile,
    DispatchConfig,
    GChatBridgeProfileConfig,
    NextcloudBridgeProfileConfig,
    SlackBridgeProfileConfig,
    TeamsBridgeProfileConfig,
    _parse_profile,
)
from osprey.cli.profile_card import card_rows
from osprey_connectors.errors import BuildProfileError

_TRIGGERS_YAML = """\
dispatcher:
  dispatch_target: http://dispatch-worker-1:9190
triggers:
  - name: slack-question
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


pytestmark = pytest.mark.usefixtures("facility_data_tree")


def _bridge_profile(**kwargs: object) -> BuildProfile:
    """A profile declaring the bridge plus a dispatch block, overridable per test."""
    fields: dict = {
        "name": "x",
        "data": "data",
        "dispatch": DispatchConfig(triggers="triggers.yml"),
        "slack_bridge": SlackBridgeProfileConfig(),
    }
    fields.update(kwargs)
    return BuildProfile(**fields)


# ── dataclass + parsing ──────────────────────────────────────────────────────


def test_slack_bridge_default_trigger() -> None:
    """The profile block is the single source of the 'slack-question' default.

    The runtime config's from_env applies no trigger default on purpose, so this
    assertion is what pins the name the compose template renders as
    DISPATCH_TRIGGER.
    """
    assert SlackBridgeProfileConfig().trigger == "slack-question"


def test_slack_bridge_custom_trigger_overrides_the_default() -> None:
    """An explicit trigger replaces the default on the dataclass itself."""
    assert SlackBridgeProfileConfig(trigger="desy-question").trigger == "desy-question"


def test_no_slack_bridge_leaves_the_field_none() -> None:
    """A profile that declares no block leaves the field None (opt-in key)."""
    assert BuildProfile(name="x").slack_bridge is None


def test_slack_bridge_is_known_key() -> None:
    """'slack_bridge' is a recognized top-level profile key (no unknown-key warning)."""
    assert "slack_bridge" in bp._KNOWN_PROFILE_KEYS


def test_slack_bridge_parse_defaults_when_empty_mapping() -> None:
    """An empty block (`slack_bridge: {}`) parses to the default trigger."""
    profile = _parse_profile({"name": "x", "slack_bridge": {}})
    assert profile.slack_bridge is not None
    assert profile.slack_bridge.trigger == "slack-question"


def test_slack_bridge_parse_round_trip() -> None:
    """An explicit trigger overrides the default through _parse_profile."""
    profile = _parse_profile({"name": "x", "slack_bridge": {"trigger": "desy-question"}})
    assert profile.slack_bridge is not None
    assert profile.slack_bridge.trigger == "desy-question"


def test_slack_bridge_mentions_default_on() -> None:
    """A block without the key keeps @mentions on."""
    profile = _parse_profile({"name": "x", "slack_bridge": {}})
    assert profile.slack_bridge is not None
    assert profile.slack_bridge.mentions is True
    assert SlackBridgeProfileConfig().mentions is True


def test_slack_bridge_mentions_false_parses() -> None:
    profile = _parse_profile({"name": "x", "slack_bridge": {"mentions": False}})
    assert profile.slack_bridge is not None
    assert profile.slack_bridge.mentions is False


@pytest.mark.parametrize("value", ["no", 0, None])
def test_slack_bridge_mentions_non_bool_raises(value: object) -> None:
    with pytest.raises(BuildProfileError, match=r"slack_bridge\.mentions"):
        _parse_profile({"name": "x", "slack_bridge": {"mentions": value}})


def test_no_slack_bridge_parses_to_none() -> None:
    """A profile without the block leaves the field None through the loader too."""
    assert _parse_profile({"name": "x"}).slack_bridge is None


def test_slack_bridge_not_a_mapping_raises() -> None:
    """A non-mapping 'slack_bridge' block raises during parsing."""
    with pytest.raises(BuildProfileError, match="slack_bridge"):
        _parse_profile({"name": "x", "slack_bridge": "not-a-mapping"})


def test_slack_bridge_parses_independently_of_the_other_bridges() -> None:
    """The bridge keys are independent: declaring one leaves the others None."""
    profile = _parse_profile({"name": "x", "slack_bridge": {}})
    assert profile.slack_bridge is not None
    assert profile.teams_bridge is None
    assert profile.gchat_bridge is None
    assert profile.nextcloud_bridge is None


# ── validate(): valid profile ────────────────────────────────────────────────


def test_slack_bridge_with_dispatch_and_declared_trigger_validates(profile_dir: Path) -> None:
    """The bridge + a dispatch block whose triggers file declares the trigger validates."""
    _bridge_profile().validate(profile_dir)  # must not raise


def test_slack_bridge_custom_declared_trigger_validates(profile_dir: Path) -> None:
    """Any trigger declared in the triggers file is accepted, not just the default."""
    profile = _bridge_profile(slack_bridge=SlackBridgeProfileConfig(trigger="other-trigger"))
    profile.validate(profile_dir)  # must not raise


def test_slack_bridge_resolves_bundled_triggers_file(tmp_path: Path) -> None:
    """A bundled triggers name resolves against the packaged triggers dir.

    Mirrors the dispatch block's own two-candidate lookup (profile-relative
    first, then bundled) — 'hello-dispatch' is declared by the shipped
    tutorial_triggers.yml, which is not in the profile dir.
    """
    profile = _bridge_profile(
        dispatch=DispatchConfig(triggers="tutorial_triggers.yml"),
        slack_bridge=SlackBridgeProfileConfig(trigger="hello-dispatch"),
    )
    profile.validate(tmp_path)  # must not raise


# ── validate(): (a) missing dispatch block ───────────────────────────────────


def test_slack_bridge_without_dispatch_raises(profile_dir: Path) -> None:
    """A bridge declared with no dispatch block fails with a message naming the fix."""
    profile = BuildProfile(name="x", slack_bridge=SlackBridgeProfileConfig())
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    message = str(exc.value)
    assert "slack_bridge requires a 'dispatch:' block" in message
    assert "slack-question" in message


def test_slack_bridge_without_dispatch_skips_trigger_check(profile_dir: Path) -> None:
    """The missing-dispatch error stands alone — no trigger error piles on top.

    There is no triggers file to check against without a dispatch block, so the
    aggregate names one actionable problem rather than two.
    """
    profile = BuildProfile(name="x", slack_bridge=SlackBridgeProfileConfig())
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    assert "is not declared in" not in str(exc.value)


# ── validate(): (b) trigger missing from the triggers file ───────────────────


def test_slack_bridge_undeclared_trigger_raises(profile_dir: Path) -> None:
    """A trigger absent from the triggers file fails, listing what is declared."""
    profile = _bridge_profile(slack_bridge=SlackBridgeProfileConfig(trigger="ghost"))
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    message = str(exc.value)
    assert "slack_bridge.trigger 'ghost' is not declared in" in message
    assert "slack-question" in message  # the declared names are listed
    assert "other-trigger" in message


def test_slack_bridge_default_trigger_undeclared_raises(tmp_path: Path) -> None:
    """The default trigger is not exempt: the shipped tutorial triggers file does
    not declare 'slack-question', so a bridge left on the default there fails."""
    profile = _bridge_profile(dispatch=DispatchConfig(triggers="tutorial_triggers.yml"))
    with pytest.raises(BuildProfileError, match="slack-question"):
        profile.validate(tmp_path)


def test_slack_bridge_empty_trigger_raises(profile_dir: Path) -> None:
    """An explicitly blanked trigger is rejected with its own message."""
    profile = _bridge_profile(slack_bridge=SlackBridgeProfileConfig(trigger=""))
    with pytest.raises(BuildProfileError, match="slack_bridge.trigger is required"):
        profile.validate(profile_dir)


def test_slack_bridge_unresolvable_triggers_file_reports_dispatch_error_only(
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


def test_slack_bridge_unparseable_triggers_file_reports_parse_failure(
    tmp_path: Path,
) -> None:
    """A triggers file that exists but cannot be parsed surfaces as a legible
    'fix that file first' error instead of an unhandled ValueError."""
    (tmp_path / "triggers.yml").write_text(
        "dispatcher:\n  dispatch_target: x\ntriggers:\n  - source: webhook\n", encoding="utf-8"
    )
    with pytest.raises(BuildProfileError) as exc:
        _bridge_profile().validate(tmp_path)
    assert "slack_bridge.trigger cannot be checked" in str(exc.value)


# ── every bridge at once ─────────────────────────────────────────────────────


def test_all_four_bridges_validate_side_by_side(profile_dir: Path) -> None:
    """A profile may declare every bridge; each is checked against the same file."""
    profile = _bridge_profile(
        teams_bridge=TeamsBridgeProfileConfig(trigger="other-trigger"),
        gchat_bridge=GChatBridgeProfileConfig(trigger="other-trigger"),
        nextcloud_bridge=NextcloudBridgeProfileConfig(trigger="other-trigger"),
    )
    profile.validate(profile_dir)  # must not raise


def test_all_four_bridges_report_their_own_undeclared_triggers(profile_dir: Path) -> None:
    """Four broken bridges produce four separately-named errors in one raise, so
    an operator fixes them all in a single pass rather than one build at a time."""
    profile = _bridge_profile(
        slack_bridge=SlackBridgeProfileConfig(trigger="ghost-s"),
        teams_bridge=TeamsBridgeProfileConfig(trigger="ghost-a"),
        gchat_bridge=GChatBridgeProfileConfig(trigger="ghost-b"),
        nextcloud_bridge=NextcloudBridgeProfileConfig(trigger="ghost-c"),
    )
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    message = str(exc.value)
    assert "slack_bridge.trigger 'ghost-s' is not declared in" in message
    assert "teams_bridge.trigger 'ghost-a' is not declared in" in message
    assert "gchat_bridge.trigger 'ghost-b' is not declared in" in message
    assert "nextcloud_bridge.trigger 'ghost-c' is not declared in" in message


# ── emitted profile: the commented stanza ────────────────────────────────────


def test_slack_bridge_has_a_commented_template_ordered_after_teams() -> None:
    """The key is part of the commented opt-in surface and is appended right after
    the Teams stanza, so two emissions stay byte-identical."""
    assert "slack_bridge" in emit._COMMENTED_TEMPLATE_KEYS
    order = emit._COMMENTED_TEMPLATE_ORDER
    assert order.index("slack_bridge") == order.index("teams_bridge") + 1


def test_slack_bridge_stanza_names_both_tokens_as_runtime_env() -> None:
    """The stanza tells the operator the two Slack tokens are env.required runtime
    env, never profile keys, and shows the block with its defaults."""
    stanza = emit._COMMENTED_TEMPLATES["slack_bridge"]
    assert "SLACK_BOT_TOKEN" in stanza
    assert "SLACK_APP_TOKEN" in stanza
    assert "`env.required`" in stanza
    assert "# slack_bridge:\n#   trigger: slack-question\n#   mentions: true" in stanza


def test_slack_bridge_stanza_is_comment_only_and_its_header_is_column_79() -> None:
    """Every stanza line is a comment (the stanza never activates the block) and the
    section header rule ends at column 79 like its neighbours."""
    lines = emit._COMMENTED_TEMPLATES["slack_bridge"].strip("\n").splitlines()
    assert all(line.startswith("#") for line in lines)
    assert lines[0].startswith("# --- Slack bridge ")
    assert len(lines[0]) == 79


# ── profile card ─────────────────────────────────────────────────────────────


def test_profile_card_shows_a_slack_bridge_row() -> None:
    """A profile carrying the block lists Slack among its services."""
    rows = card_rows(BuildProfile(name="x", slack_bridge=SlackBridgeProfileConfig()), {})
    assert {"group": "services", "label": "bridge", "value": "Slack"} in rows


def test_profile_card_omits_slack_when_the_block_is_absent() -> None:
    rows = card_rows(BuildProfile(name="x"), {})
    assert not any(row["value"] == "Slack" for row in rows)
