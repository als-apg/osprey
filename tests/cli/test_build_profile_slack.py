"""Tests only the ``slack_bridge:`` block of the build-profile schema has.

The contract every chat bridge shares (trigger default, parsing, the
:meth:`BuildProfile.validate` dispatch and trigger checks) is parametrized over
all four bridges in ``test_build_profile_nextcloud.py``. This file covers the
Slack dataclass and parsing extras, every bridge declared side by side, the
commented stanza an emitted profile carries for the block, and the profile
card's bridge row.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


def _bridge_profile(**kwargs: object) -> BuildProfile:
    """A profile declaring the bridge plus a dispatch block, overridable per test."""
    fields: dict = {
        "name": "x",
        "dispatch": DispatchConfig(triggers="triggers.yml"),
        "slack_bridge": SlackBridgeProfileConfig(),
    }
    fields.update(kwargs)
    return BuildProfile(**fields)


# ── dataclass + parsing ──────────────────────────────────────────────────────


def test_slack_bridge_custom_trigger_overrides_the_default() -> None:
    """An explicit trigger replaces the default on the dataclass itself."""
    assert SlackBridgeProfileConfig(trigger="desy-question").trigger == "desy-question"


def test_no_slack_bridge_leaves_the_field_none() -> None:
    """A profile that declares no block leaves the field None (opt-in key)."""
    assert BuildProfile(name="x").slack_bridge is None


def test_slack_bridge_parses_independently_of_the_other_bridges() -> None:
    """The bridge keys are independent: declaring one leaves the others None."""
    profile = _parse_profile({"name": "x", "slack_bridge": {}})
    assert profile.slack_bridge is not None
    assert profile.teams_bridge is None
    assert profile.gchat_bridge is None
    assert profile.nextcloud_bridge is None


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
    stanza = emit._commented_template("slack_bridge")
    assert "SLACK_BOT_TOKEN" in stanza
    assert "SLACK_APP_TOKEN" in stanza
    assert "`env.required`" in stanza
    assert "# slack_bridge:\n#   trigger: slack-question\n#   mentions: true" in stanza


def test_slack_bridge_stanza_is_comment_only_and_its_header_is_column_79() -> None:
    """Every stanza line is a comment (the stanza never activates the block) and the
    section header rule ends at column 79 like its neighbours."""
    lines = emit._commented_template("slack_bridge").strip("\n").splitlines()
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
