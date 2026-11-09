"""``SlackOps.resolve_reply_context``: the thread parent a reply quotes.

Pins the worker contract (``core/ports.py`` ``ReplyContext``): a thread reply
resolves to ``reply_to={"sender", "text"}`` plus ``meta={SL_PARENT_FILES: refs}``
from ONE ``conversations.replies`` lookup; a non-reply makes no call at all; every
failure — a lookup error, a missing parent, a parent with nothing in it — is
``None``, never a raise.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from osprey.bridges.core import ReplyContext
from osprey.bridges.slack.client import SlackApiError, SlackAuthError
from osprey.bridges.slack.events import SL_PARENT_FILES
from osprey.bridges.slack.ops import MAX_QUOTED_TEXT, QUOTE_ELLIPSIS
from tests.bridges.slack._posting import (
    BOT_USER,
    CHAN,
    DM,
    PARENT,
    TS,
    RecordingClient,
    dm_payload,
    make_ops,
    mention_payload,
)

PARENT_USER = "UBOB0001"


class RepliesClient(RecordingClient):
    """A recording client that also answers ``conversations_replies``."""

    def __init__(
        self, messages: list[dict[str, Any]] | None = None, *, fail: BaseException | None = None
    ) -> None:
        super().__init__()
        self.messages = messages if messages is not None else []
        self.fail = fail
        self.lookups: list[tuple[str, str, int | None]] = []

    def conversations_replies(
        self, channel: str, ts: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        self.lookups.append((channel, ts, limit))
        if self.fail is not None:
            raise self.fail
        return self.messages


def parent(text: str = "Beam dumped at 10:42, see plot", **extra: Any) -> dict[str, Any]:
    message: dict[str, Any] = {
        "type": "message",
        "user": PARENT_USER,
        "text": text,
        "ts": PARENT,
        "thread_ts": PARENT,
        "user_profile": {"display_name": "Bob", "real_name": "Bob Example"},
    }
    message.update(extra)
    return message


def slack_file(file_id: str = "F1", name: str = "orbit.png", **extra: Any) -> dict[str, Any]:
    file: dict[str, Any] = {
        "id": file_id,
        "name": name,
        "mimetype": "image/png",
        "size": 1234,
        "url_private_download": f"https://files.slack.com/files-pri/T0-{file_id}/download/{name}",
    }
    file.update(extra)
    return file


def resolve(client: RepliesClient, wire: dict[str, Any]) -> ReplyContext | None:
    ops, _ = make_ops(client)
    event = ops.parse_event(wire)
    assert event is not None
    return ops.resolve_reply_context(event)


def test_a_channel_thread_reply_quotes_its_parent() -> None:
    client = RepliesClient([parent()])
    context = resolve(client, mention_payload(thread_ts=PARENT))
    assert context == ReplyContext(
        reply_to={"sender": "Bob", "text": "Beam dumped at 10:42, see plot"},
        meta={SL_PARENT_FILES: []},
    )
    assert client.lookups == [(CHAN, PARENT, 1)]


def test_a_dm_thread_reply_looks_up_the_dm_thread() -> None:
    client = RepliesClient([parent()])
    context = resolve(client, dm_payload(thread_ts=PARENT))
    assert context is not None
    assert client.lookups == [(DM, PARENT, 1)]


def test_a_top_level_question_is_not_a_reply_and_makes_no_call() -> None:
    client = RepliesClient([parent()])
    assert resolve(client, mention_payload()) is None
    assert resolve(client, dm_payload()) is None
    assert client.lookups == []
    assert client.calls == []


def test_a_thread_root_whose_thread_ts_is_its_own_ts_is_not_a_reply() -> None:
    client = RepliesClient([parent()])
    assert resolve(client, mention_payload(thread_ts=TS)) is None
    assert client.lookups == []


def test_the_parent_files_ride_in_meta_as_refs() -> None:
    files = [slack_file("F1", "orbit.png"), slack_file("F2", "log.csv", mimetype="text/csv")]
    client = RepliesClient([parent(files=files)])
    context = resolve(client, mention_payload(thread_ts=PARENT))
    assert context is not None
    refs = context.meta[SL_PARENT_FILES]
    assert [ref["id"] for ref in refs] == ["F1", "F2"]
    assert refs[0] == {
        "id": "F1",
        "name": "orbit.png",
        "mimetype": "image/png",
        "size": 1234,
        "url": "https://files.slack.com/files-pri/T0-F1/download/orbit.png",
    }


def test_a_parent_with_only_files_still_resolves() -> None:
    client = RepliesClient([parent(text="", files=[slack_file()])])
    context = resolve(client, mention_payload(thread_ts=PARENT))
    assert context is not None
    assert context.reply_to["text"] == ""
    assert len(context.meta[SL_PARENT_FILES]) == 1


def test_a_parent_with_neither_text_nor_files_is_none() -> None:
    client = RepliesClient([parent(text="   ")])
    assert resolve(client, mention_payload(thread_ts=PARENT)) is None


def test_the_parent_text_is_normalized() -> None:
    raw = f"<@{BOT_USER}> check <https://example.org|the log> &amp; <!here>"
    client = RepliesClient([parent(text=raw)])
    context = resolve(client, mention_payload(thread_ts=PARENT))
    assert context is not None
    assert context.reply_to["text"] == "check the log (https://example.org) & @here"


def test_a_long_parent_is_cut_to_a_hard_bound_with_the_ellipsis_inside() -> None:
    client = RepliesClient([parent(text="x" * (MAX_QUOTED_TEXT * 2))])
    context = resolve(client, mention_payload(thread_ts=PARENT))
    assert context is not None
    text = context.reply_to["text"]
    assert len(text) == MAX_QUOTED_TEXT
    assert text.endswith(QUOTE_ELLIPSIS)


def test_a_parent_at_the_bound_is_not_cut() -> None:
    client = RepliesClient([parent(text="y" * MAX_QUOTED_TEXT)])
    context = resolve(client, mention_payload(thread_ts=PARENT))
    assert context is not None
    assert context.reply_to["text"] == "y" * MAX_QUOTED_TEXT


@pytest.mark.parametrize(
    ("extra", "sender"),
    [
        ({"user_profile": {"display_name": "", "real_name": "Bob Example"}}, "Bob Example"),
        ({"user_profile": None, "bot_profile": {"name": "beam-bot"}}, "beam-bot"),
        ({"user_profile": None, "username": "legacy-hook"}, "legacy-hook"),
        ({"user_profile": None}, PARENT_USER),
    ],
)
def test_the_sender_falls_back_through_profile_bot_username_and_id(
    extra: dict[str, Any], sender: str
) -> None:
    client = RepliesClient([parent(**extra)])
    context = resolve(client, mention_payload(thread_ts=PARENT))
    assert context is not None
    assert context.reply_to["sender"] == sender


def test_the_parent_is_the_message_whose_ts_is_the_thread_ts() -> None:
    other = parent(text="a later reply", ts="1700000000.999999")
    client = RepliesClient([other, parent(text="the root")])
    context = resolve(client, mention_payload(thread_ts=PARENT))
    assert context is not None
    assert context.reply_to["text"] == "the root"


def test_a_thread_without_its_parent_is_none() -> None:
    client = RepliesClient([parent(ts="1700000000.999999")])
    assert resolve(client, mention_payload(thread_ts=PARENT)) is None


def test_an_empty_thread_is_none() -> None:
    assert resolve(RepliesClient([]), mention_payload(thread_ts=PARENT)) is None


@pytest.mark.parametrize(
    "error",
    [
        SlackApiError("missing_scope", method="conversations.replies", error="missing_scope"),
        SlackAuthError("token revoked", error="token_revoked"),
        RuntimeError("boom"),
    ],
)
def test_a_failed_lookup_is_none_and_logged(
    error: BaseException, caplog: pytest.LogCaptureFixture
) -> None:
    client = RepliesClient(fail=error)
    with caplog.at_level("WARNING"):
        assert resolve(client, mention_payload(thread_ts=PARENT)) is None
    assert client.lookups == [(CHAN, PARENT, 1)]
    assert caplog.records


def test_a_malformed_event_is_none() -> None:
    ops, client = make_ops(RepliesClient([parent()]))
    good = ops.parse_event(mention_payload(thread_ts=PARENT))
    assert good is not None
    for raw in (None, {}, {"event": "nope"}, {"event": {"thread_ts": PARENT}}):
        broken = dataclasses.replace(good, raw=raw)
        assert ops.resolve_reply_context(broken) is None
