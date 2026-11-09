"""SlackOps' posting members: wording, placement, failure contracts, the answer path."""

from __future__ import annotations

import logging
import time

import pytest

from osprey.bridges.core.errors import UndeliverableError
from osprey.bridges.slack import ops as ops_module
from osprey.bridges.slack.client import SlackApiError
from osprey.bridges.slack.events import SL_CHANNEL, SL_CHANNEL_TYPE, SL_TEXT_HEAD, SL_THREAD_TS
from osprey.bridges.slack.ops import (
    ACK_TEXT,
    ANSWER_CHUNK_CHARS,
    EMPTY_ANSWER_TEXT,
    ERROR_TEXT,
    GIVEUP_TEXT,
    QUEUED_TEXT,
    QUOTE_ELLIPSIS,
    QUOTE_LINE_CHARS,
    SUPERSEDED_TEXT,
    ack_text,
    quote_prefix,
    skipped_files_note,
)
from tests.bridges.slack._posting import (
    CHAN,
    DM,
    MPIM,
    PARENT,
    QUESTION,
    SENDER_ID,
    TS,
    VERSION,
    RecordingClient,
    expected_chunks,
    long_answer,
    make_entry,
    make_ops,
)

DM_QUOTE = f"> {QUESTION}\n\n"


def completed(text: str = "The stored beam current is **500 mA**.") -> dict:
    return {"status": "completed", "text_output": text, "run_id": "R1", "artifacts": []}


# --- wording -----------------------------------------------------------------


def test_wording_mirrors_teams():
    assert ACK_TEXT == "On it — working on this now."
    assert ack_text(VERSION) == f"On it — working on this now.  (OSPREY {VERSION})"
    assert ack_text("") == ACK_TEXT
    assert skipped_files_note(["a.png", "b.csv"]) == "I couldn't attach: a.png, b.csv."
    assert ANSWER_CHUNK_CHARS == 4000


def test_skipped_files_note_escapes_names():
    assert skipped_files_note(["<!here>&.png"]) == "I couldn't attach: &lt;!here&gt;&amp;.png."


# --- placement ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("channel_type", "channel"), [("channel", CHAN), ("group", "G0PRIV"), ("mpim", MPIM)]
)
def test_room_ack_goes_into_the_thread_without_a_quote(channel_type, channel):
    ops, client = make_ops()
    ops.post_ack(make_entry(channel_type, channel=channel))
    assert client.calls == [(channel, ack_text(VERSION), TS)]


def test_room_reply_in_an_existing_thread_uses_that_thread():
    ops, client = make_ops()
    ops.post_ack(make_entry("channel", thread_ts=PARENT))
    assert client.calls[0][2] == PARENT


def test_top_level_dm_ack_is_top_level_and_quoted():
    ops, client = make_ops()
    ops.post_ack(make_entry("im"))
    assert client.calls == [(DM, DM_QUOTE + ack_text(VERSION), None)]


def test_threaded_dm_goes_into_its_thread_unquoted():
    ops, client = make_ops()
    ops.post_ack(make_entry("im", thread_ts=PARENT))
    assert client.calls == [(DM, ack_text(VERSION), PARENT)]


def test_non_dm_entry_without_thread_falls_back_to_its_ts():
    ops, client = make_ops()
    ops.post_ack(make_entry("channel", thread_ts=""))
    assert client.calls[0][2] == TS


def test_non_dm_entry_with_no_thread_at_all_is_not_posted_at_top_level():
    ops, client = make_ops()
    entry = make_entry("channel", thread_ts="")
    del entry["sl_ts"]
    with pytest.raises(ValueError):
        ops.post_giveup(entry)
    assert client.calls == []


# --- quote prefix --------------------------------------------------------------


def test_quote_prefix_only_for_top_level_dm():
    assert quote_prefix(make_entry("im")) == DM_QUOTE
    assert quote_prefix(make_entry("im", thread_ts=PARENT)) == ""
    assert quote_prefix(make_entry("channel")) == ""
    assert quote_prefix(make_entry("mpim")) == ""


def test_quote_prefix_caps_long_lines_with_an_ellipsis():
    line = "q" * (QUOTE_LINE_CHARS + 30)
    assert quote_prefix(make_entry("im", text=line)) == (
        f"> {'q' * QUOTE_LINE_CHARS}{QUOTE_ELLIPSIS}\n\n"
    )


def test_quote_prefix_uses_first_line_and_falls_back_to_text():
    entry = make_entry("im", text="first line\nsecond line")
    assert quote_prefix(entry) == "> first line\n\n"
    entry[SL_TEXT_HEAD] = ""
    assert quote_prefix(entry) == "> first line\n\n"
    del entry[SL_TEXT_HEAD]
    assert quote_prefix(entry) == "> first line\n\n"


def test_quote_prefix_is_escaped_so_it_pings_nobody():
    entry = make_entry("im", text="ping <!here> and <@U999> & co")
    assert quote_prefix(entry) == "> ping &lt;!here&gt; and &lt;@U999&gt; &amp; co\n\n"


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_quote_prefix_blank_question_is_empty(text):
    entry = make_entry("im", text=text)
    assert quote_prefix(entry) == ""


def test_quote_prefix_total_on_degenerate_entries():
    assert quote_prefix({}) == ""
    assert quote_prefix({SL_CHANNEL_TYPE: "im", "text": 7}) == ""


# --- swallow vs raise ---------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [SlackApiError("boom", method="chat.postMessage"), UndeliverableError("is_archived")],
)
def test_ack_swallows_client_failures(error, caplog):
    ops, client = make_ops(RecordingClient(fail_with=error))
    with caplog.at_level(logging.WARNING):
        ops.post_ack(make_entry())
    assert len(client.calls) == 1
    assert "ack post failed" in caplog.text


def test_ack_swallows_a_malformed_entry():
    ops, client = make_ops()
    ops.post_ack({})
    assert client.calls == []


def test_resumed_and_superseded_swallow():
    error = SlackApiError("boom", method="chat.postMessage")
    ops, client = make_ops(RecordingClient(fail_with=error))
    ops.post_resumed(make_entry(), {"status": "completed"})
    ops.post_superseded(make_entry())
    ops.post_resumed({}, {})
    ops.post_superseded({})
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "error",
    [SlackApiError("boom", method="chat.postMessage"), UndeliverableError("channel_not_found")],
)
def test_queued_and_giveup_raise(error):
    ops, _ = make_ops(RecordingClient(fail_with=error))
    with pytest.raises(type(error)):
        ops.post_queued(make_entry(), {"status": "failed"})
    with pytest.raises(type(error)):
        ops.post_giveup(make_entry())


def test_queued_and_giveup_raise_on_a_malformed_entry():
    ops, _ = make_ops()
    with pytest.raises(ValueError):
        ops.post_queued({}, {})
    with pytest.raises(ValueError):
        ops.post_giveup({})


# --- notices -------------------------------------------------------------------


def test_notices_carry_their_wording_and_dm_quote():
    ops, client = make_ops()
    entry = make_entry("im")
    ops.post_queued(entry, {"status": "failed", "error": "secret-detail"})
    ops.post_giveup({**entry, "give_up_reason": "secret-reason"})
    ops.post_superseded(entry)
    assert client.texts == [
        DM_QUOTE + QUEUED_TEXT,
        DM_QUOTE + GIVEUP_TEXT,
        DM_QUOTE + SUPERSEDED_TEXT,
    ]
    assert not any("secret" in text for text in client.texts)


def test_resumed_names_when_it_was_queued():
    ops, client = make_ops()
    entry = make_entry("channel")
    entry["first_queued_at"] = time.time() - 12 * 60
    ops.post_resumed(entry, {"status": "completed"})
    text = client.texts[0]
    assert text.startswith("Resuming your request queued at ")
    assert "12 min" in text or "minutes" in text
    assert text.endswith("— service is restored, answer follows.")


def test_resumed_without_a_time_says_earlier():
    ops, client = make_ops()
    ops.post_resumed(make_entry("im"), {"status": "completed"})
    assert client.texts == [
        DM_QUOTE + "Resuming your request queued earlier — service is restored, answer follows."
    ]


# --- answer --------------------------------------------------------------------


def test_answer_is_converted_to_mrkdwn_and_threaded():
    ops, client = make_ops()
    ops.post_answer(make_entry("channel"), completed())
    assert client.calls == [(CHAN, "The stored beam current is *500 mA*.", TS)]


def test_long_answer_posts_every_chunk_in_order_quote_on_first_only():
    text = long_answer()
    chunks = expected_chunks(text)
    assert len(chunks) > 2
    ops, client = make_ops()
    ops.post_answer(make_entry("im"), completed(text))
    assert client.texts == [DM_QUOTE + chunks[0], *chunks[1:]]
    assert all(len(chunk) <= ANSWER_CHUNK_CHARS for chunk in chunks)
    assert {thread for _, _, thread in client.calls} == {None}


def test_answer_stops_at_first_failure_and_raises():
    text = long_answer()
    error = SlackApiError("boom", method="chat.postMessage")
    ops, client = make_ops(RecordingClient(fail_with=error, fail_from=2))
    with pytest.raises(SlackApiError):
        ops.post_answer(make_entry(), completed(text))
    assert len(client.calls) == 2


def test_answer_propagates_undeliverable_untouched():
    error = UndeliverableError("not_in_channel")
    ops, _ = make_ops(RecordingClient(fail_with=error))
    with pytest.raises(UndeliverableError) as excinfo:
        ops.post_answer(make_entry(), completed())
    assert excinfo.value is error


def test_answer_raises_on_a_malformed_entry():
    ops, _ = make_ops()
    with pytest.raises(ValueError):
        ops.post_answer({}, completed())


@pytest.mark.parametrize("status", ["error", "timeout", "cancelled", None])
def test_failed_run_never_leaks_its_error(status):
    ops, client = make_ops()
    result = {"status": status, "error": "Traceback: token=xoxb-leak", "text_output": "partial"}
    ops.post_answer(make_entry(), result)
    assert client.texts == [ERROR_TEXT]


@pytest.mark.parametrize("text", [None, "", "   \n"])
def test_silent_run_posts_the_placeholder(text):
    ops, client = make_ops()
    ops.post_answer(make_entry(), {"status": "completed", "text_output": text})
    assert client.texts == [EMPTY_ANSWER_TEXT]


def test_answer_never_mutates_the_result():
    ops, _ = make_ops()
    result = completed("**bold** answer")
    snapshot = dict(result)
    ops.post_answer(make_entry(), result)
    assert result == snapshot


def test_mentions_post_as_plain_ids_without_a_roster():
    ops, client = make_ops(mentions=True)
    ops.post_answer(make_entry(), completed(f"Ask <@{SENDER_ID}> or <!channel>."))
    text = client.texts[0]
    assert f"@{SENDER_ID}" in text
    assert f"<@{SENDER_ID}>" not in text
    assert "<!channel>" not in text


def test_mention_placeholder_is_never_split_across_chunks():
    filler = "y" * (ANSWER_CHUNK_CHARS - 5)
    ops, client = make_ops()
    ops.post_answer(make_entry(), completed(f"{filler}<@UAAAAAAAAA> tail"))
    assert any("@UAAAAAAAAA" in text for text in client.texts)


def test_render_failure_posts_plain_ids(monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("render broke")

    monkeypatch.setattr(ops_module, "render_mentions", broken)
    ops, client = make_ops()
    ops.post_answer(make_entry(), completed(f"Ask <@{SENDER_ID}>."))
    assert client.texts == [f"Ask @{SENDER_ID}."]


def test_dm_quote_in_answer_is_never_mention_rendered():
    ops, client = make_ops()
    entry = make_entry("im", text="is <@UBOB> around?")
    ops.post_answer(entry, completed("yes"))
    assert client.texts[0].startswith("> is &lt;@UBOB&gt; around?\n\n")


# --- coalesce -------------------------------------------------------------------


def test_coalesce_key_is_history_and_sender():
    ops, _ = make_ops()
    entry = make_entry()
    assert ops.coalesce_key(entry) == [entry["history_key"], SENDER_ID]


@pytest.mark.parametrize(
    "entry",
    [
        {"sender_id": SENDER_ID},
        {"history_key": "C0CHAN:1"},
        {"history_key": "", "sender_id": SENDER_ID},
        {"history_key": "C0CHAN:1", "sender_id": 7},
        {},
    ],
)
def test_coalesce_key_stands_alone_when_incomplete(entry):
    ops, _ = make_ops()
    assert ops.coalesce_key(entry) == ()


def test_entries_read_only_sl_keys_for_placement():
    # A placement key renamed on either side would silently mis-address; pin the names.
    assert (SL_CHANNEL, SL_THREAD_TS, SL_CHANNEL_TYPE) == (
        "sl_channel",
        "sl_thread_ts",
        "sl_channel_type",
    )
