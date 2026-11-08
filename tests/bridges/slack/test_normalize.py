"""Inbound Slack text normalization and the ``sl_*`` entry-key contract."""

from __future__ import annotations

import pytest

from osprey.bridges.core import RESERVED_ENTRY_KEYS
from osprey.bridges.slack import events
from osprey.bridges.slack.events import normalize_text

BOT = "UBOT123"
NAMES = {"UALICE1": "Alice", "WBOB22": "Bob"}


def norm(text, names=NAMES, bot=BOT):
    return normalize_text(text, bot, names)


# --- the bot's own mention -------------------------------------------------


def test_bot_mention_removed_and_ends_stripped():
    assert norm("<@UBOT123> what is the beam current?") == "what is the beam current?"


def test_bot_mention_with_label_removed():
    assert norm("<@UBOT123|osprey> status?") == "status?"


def test_bot_mention_trailing_removed():
    assert norm("status? <@UBOT123>") == "status?"


def test_only_bot_mention_yields_empty():
    assert norm("  <@UBOT123>  ") == ""


def test_empty_bot_id_removes_nothing():
    assert norm("<@UBOT123> hi", bot="") == "@UBOT123 hi"


# --- other users' mentions -------------------------------------------------


def test_other_user_known_name():
    assert norm("<@UBOT123> ask <@UALICE1> about it") == "ask Alice about it"


def test_other_user_w_prefix_known_name():
    assert norm("ping <@WBOB22>") == "ping Bob"


def test_other_user_unknown_falls_back_to_id():
    assert norm("ask <@UCAROL9> now") == "ask @UCAROL9 now"


def test_other_user_with_label_uses_known_name():
    assert norm("ask <@UALICE1|alice.s> now") == "ask Alice now"


def test_other_user_unknown_with_label_falls_back_to_id():
    assert norm("ask <@UCAROL9|carol> now") == "ask @UCAROL9 now"


def test_blank_name_falls_back_to_id():
    assert norm("ask <@UALICE1>", names={"UALICE1": "  "}) == "ask @UALICE1"


def test_names_none_or_garbage_is_tolerated():
    assert normalize_text("hi <@UALICE1>", BOT, None) == "hi @UALICE1"  # type: ignore[arg-type]
    assert normalize_text("hi <@UALICE1>", BOT, ["x"]) == "hi @UALICE1"  # type: ignore[arg-type]
    assert normalize_text("hi <@UALICE1>", BOT, {"UALICE1": 5}) == "hi @UALICE1"  # type: ignore[dict-item]


# --- links -----------------------------------------------------------------


def test_labelled_link():
    assert norm("see <https://example.org/x|the docs>") == "see the docs (https://example.org/x)"


def test_bare_link():
    assert norm("see <https://example.org/x>") == "see https://example.org/x"


def test_link_with_escaped_query_is_unescaped():
    assert norm("<https://a.b/?x=1&amp;y=2>") == "https://a.b/?x=1&y=2"


def test_mailto_link():
    assert norm("<mailto:a@b.org|a@b.org>") == "a@b.org (mailto:a@b.org)"


# --- channels and broadcasts ----------------------------------------------


def test_channel_reference():
    assert norm("post in <#C0123ABC|ops-room>") == "post in #ops-room"


def test_channel_reference_without_name():
    assert norm("post in <#C0123ABC>") == "post in #C0123ABC"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<!here> beam down", "@here beam down"),
        ("<!channel> beam down", "@channel beam down"),
        ("<!everyone> beam down", "@everyone beam down"),
        ("<!here|here> beam down", "@here beam down"),
        ("<!subteam^S0123|@operators> help", "@operators help"),
        ("<!subteam^S0123> help", "@S0123 help"),
    ],
)
def test_broadcast_tokens(raw, expected):
    assert norm(raw) == expected


# --- escaping and whitespace ----------------------------------------------


def test_entities_unescaped():
    assert norm("a &lt; b &amp;&amp; c &gt; d") == "a < b && c > d"


def test_double_escaped_entity_unescapes_once():
    assert norm("literal &amp;lt; here") == "literal &lt; here"


def test_label_entities_unescaped():
    assert norm("<https://x.y|R&amp;D>") == "R&D (https://x.y)"


def test_display_name_not_unescaped():
    assert norm("hi <@UALICE1>", names={"UALICE1": "A&amp;B"}) == "hi A&amp;B"


def test_inner_whitespace_and_code_survive():
    text = "<@UBOT123>\n```\nfor i in x:\n    print(i)\n```\n  done"
    assert norm(text) == "```\nfor i in x:\n    print(i)\n```\n  done"


def test_plain_text_untouched():
    assert norm("  just text  ") == "just text"


# --- never raises ----------------------------------------------------------


@pytest.mark.parametrize("bad", [None, 42, b"<@UBOT123> hi", ["x"], {"a": 1}])
def test_non_string_text_yields_empty(bad):
    assert normalize_text(bad, BOT, NAMES) == ""  # type: ignore[arg-type]


def test_non_string_bot_id_tolerated():
    assert normalize_text("<@UBOT123> hi", None, NAMES) == "@UBOT123 hi"  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "weird",
    ["<>", "<@>", "<#>", "<!>", "<|>", "< unclosed", "closed >", "<<@UBOT123>>", "<@UBOT123|>"],
)
def test_malformed_tokens_do_not_raise(weird):
    assert isinstance(norm(weird), str)


# --- entry-key constants ---------------------------------------------------

SL_KEYS = {
    "SL_CHANNEL": "sl_channel",
    "SL_TS": "sl_ts",
    "SL_THREAD_TS": "sl_thread_ts",
    "SL_CHANNEL_TYPE": "sl_channel_type",
    "SL_TEAM": "sl_team",
    "SL_FILES": "sl_files",
    "SL_TEXT_HEAD": "sl_text_head",
    "SL_PARENT_FILES": "sl_parent_files",
}


def test_key_constants_match_contract():
    for name, value in SL_KEYS.items():
        assert getattr(events, name) == value


def test_key_constants_prefixed_unique_and_disjoint_from_reserved():
    values = list(SL_KEYS.values())
    assert all(v.startswith("sl_") for v in values)
    assert len(set(values)) == len(values)
    assert not set(values) & RESERVED_ENTRY_KEYS
