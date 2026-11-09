"""Tests for the Slack mention renderer (:func:`render_mentions`)."""

from __future__ import annotations

import pytest

from osprey.bridges.slack.formatting import markdown_to_mrkdwn, render_mentions

ROSTER: dict[str, str | None] = {"U1": "Ada Lovelace", "W2": "Grace", "U3": None}


@pytest.mark.parametrize("ident", ["U1", "W2", "U3"])
def test_a_listed_member_renders_live_when_mentions_are_on(ident):
    assert render_mentions(f"hi <@{ident}>!", ROSTER, enabled=True) == f"hi <@{ident}>!"


def test_both_user_id_prefixes_render_live_in_one_message():
    out = render_mentions("<@U1> and <@W2>", ROSTER, enabled=True)
    assert out == "<@U1> and <@W2>"


def test_an_unlisted_id_becomes_plain_at_id():
    assert render_mentions("ping <@U9> and <@W9>", ROSTER, enabled=True) == "ping @U9 and @W9"


def test_mentions_off_renders_listed_members_as_their_name():
    assert render_mentions("<@U1>, <@W2>", ROSTER, enabled=False) == "@Ada Lovelace, @Grace"


def test_mentions_off_renders_a_listed_member_without_a_name_as_the_id():
    assert render_mentions("<@U3>", ROSTER, enabled=False) == "@U3"


def test_mentions_off_renders_an_unlisted_id_as_the_id():
    assert render_mentions("<@U9>", ROSTER, enabled=False) == "@U9"


def test_an_empty_roster_never_renders_live():
    assert render_mentions("<@U1>", {}, enabled=True) == "@U1"


def test_a_roster_name_cannot_inject_a_control_sequence():
    roster = {"U1": "<!channel> &\n<@U2>"}
    out = render_mentions("<@U1>", roster, enabled=False)
    assert "<" not in out and ">" not in out
    assert out == "@&lt;!channel&gt; &amp; &lt;@U2&gt;"


BROADCASTS = [
    "<!channel>",
    "<!here>",
    "<!everyone>",
    "<!channel|channel>",
    "<!here|here>",
    "<!everyone|everyone>",
    "<!subteam^S0123ABC>",
    "<!subteam^S0123ABC|@oncall>",
    "<!date^1392734382^{date_short}|Feb 18>",
    "<!something-else>",
    "<!>",
]


@pytest.mark.parametrize("form", BROADCASTS)
@pytest.mark.parametrize("enabled", [True, False])
def test_every_broadcast_form_is_neutralized(form, enabled):
    out = render_mentions(f"hey {form} now", ROSTER, enabled=enabled)
    assert "<!" not in out
    assert "<" not in out and ">" not in out
    assert out.startswith("hey &lt;!") and out.endswith(" now")


def test_a_broadcast_is_neutralized_to_its_escaped_text():
    assert render_mentions("<!here>", ROSTER, enabled=True) == "&lt;!here&gt;"
    assert (
        render_mentions("<!subteam^S1|@team>", ROSTER, enabled=True) == "&lt;!subteam^S1|@team&gt;"
    )


def test_an_unclosed_broadcast_opener_is_neutralized():
    out = render_mentions("x <!channel <@U1>>", ROSTER, enabled=True)
    assert "<!" not in out
    assert "<@U1>" in out


def test_a_broadcast_written_as_a_placeholder_id_is_not_live():
    out = render_mentions("<@!channel>", ROSTER, enabled=True)
    assert out == "@!channel"


def test_text_without_mentions_is_unchanged():
    text = "see <https://ex.com|docs> &amp; *bold*"
    assert render_mentions(text, ROSTER, enabled=True) == text


def test_empty_text_returns_empty():
    assert render_mentions("", ROSTER, enabled=True) == ""


def test_applied_after_the_formatter_the_chunk_has_one_live_mention_and_no_broadcast():
    chunk = markdown_to_mrkdwn("**hi** <@U1>, <@U9> `<!channel>` <!here>")
    out = render_mentions(chunk, ROSTER, enabled=True)
    assert out == "*hi* <@U1>, @U9 `&lt;!channel&gt;` &lt;!here&gt;"
