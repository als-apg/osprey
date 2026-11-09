"""The Slack posting suites' world: ids, payloads and entries, and a recording client."""

from __future__ import annotations

from typing import Any

from osprey.bridges.core.text import chunk_text
from osprey.bridges.slack.config import SlackBridgeConfig
from osprey.bridges.slack.events import (
    SL_CHANNEL,
    SL_CHANNEL_TYPE,
    SL_FILES,
    SL_TEAM,
    SL_TEXT_HEAD,
    SL_THREAD_TS,
    SL_TS,
    BotIdentity,
)
from osprey.bridges.slack.formatting import MENTION_PLACEHOLDER_RE, markdown_to_mrkdwn
from osprey.bridges.slack.ops import ANSWER_CHUNK_CHARS, SlackOps

BOT_USER = "UBOT123"
BOT_ID = "B0OSPREY"
TEAM = "T0HOME"
CTX = BotIdentity(bot_user_id=BOT_USER, bot_id=BOT_ID, team_id=TEAM)

SENDER_ID = "UALICE1"
SENDER_NAME = "Alice"
DM = "D0DIRECT"
CHAN = "C0CHAN"
MPIM = "G0MPIM"
TS = "1700000000.000100"
PARENT = "1699999999.000001"
QUESTION = "What is the stored beam current right now?"
VERSION = "2026.9.0"


def make_config(version_tag: str = VERSION, *, mentions: bool = True) -> SlackBridgeConfig:
    """A config carrying only what the posting members read."""
    return SlackBridgeConfig(
        bot_token="xoxb-1111-2222-bot-secret",
        app_token="xapp-1-A111-3333-app-secret",  # gitleaks:allow - a fake test token
        version_tag=version_tag,
        mentions=mentions,
    )


def make_entry(
    channel_type: str = "channel",
    *,
    text: str = QUESTION,
    thread_ts: str | None = None,
    channel: str | None = None,
) -> dict[str, Any]:
    """A persisted entry as the claim stamped it, for one conversation kind.

    ``thread_ts`` defaults to what parse writes: empty for a top-level DM, the
    question's own ``ts`` for every other kind.
    """
    is_dm = channel_type == "im"
    if channel is None:
        channel = {"im": DM, "mpim": MPIM}.get(channel_type, CHAN)
    if thread_ts is None:
        thread_ts = "" if is_dm else TS
    return {
        "message_id": f"{channel}:{TS}",
        "text": text,
        "sender_id": SENDER_ID,
        "sender_display": SENDER_NAME,
        "history_key": channel if is_dm else f"{channel}:{thread_ts or TS}",
        SL_CHANNEL: channel,
        SL_TS: TS,
        SL_THREAD_TS: thread_ts,
        SL_CHANNEL_TYPE: channel_type,
        SL_TEAM: TEAM,
        SL_FILES: [],
        SL_TEXT_HEAD: text.split("\n", 1)[0].strip(),
    }


def _event(
    type_: str,
    text: str,
    *,
    channel: str,
    channel_type: str | None,
    thread_ts: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": type_,
        "user": SENDER_ID,
        "text": text,
        "ts": TS,
        "channel": channel,
        "team": TEAM,
        "user_team": TEAM,
        "user_profile": {"display_name": SENDER_NAME, "real_name": "Alice Example"},
    }
    if channel_type is not None:
        event["channel_type"] = channel_type
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "event_callback",
        "team_id": TEAM,
        "event_id": "Ev0001",
        "event": event,
        "is_ext_shared_channel": False,
        "authorizations": [{"team_id": TEAM, "user_id": BOT_USER, "is_bot": True}],
    }


def dm_payload(text: str = QUESTION, *, thread_ts: str | None = None) -> dict[str, Any]:
    """A 1:1 DM question, as the inbox stored it."""
    return _payload(_event("message", text, channel=DM, channel_type="im", thread_ts=thread_ts))


def mention_payload(text: str = QUESTION, *, thread_ts: str | None = None) -> dict[str, Any]:
    """A channel question: ``app_mention`` without ``channel_type``."""
    return _payload(
        _event(
            "app_mention",
            f"<@{BOT_USER}> {text}",
            channel=CHAN,
            channel_type=None,
            thread_ts=thread_ts,
        )
    )


class RecordingClient:
    """A stand-in for :class:`~osprey.bridges.slack.client.SlackWebClient`.

    Records every ``post_message`` and can be told to fail — always
    (``fail_with``) or from the n-th call on (``fail_from``, counted from one).
    """

    def __init__(self, fail_with: BaseException | None = None, *, fail_from: int = 1) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self.fail_with = fail_with
        self.fail_from = fail_from

    def post_message(self, channel: str, text: str, thread_ts: str | None = None) -> dict[str, Any]:
        self.calls.append((channel, text, thread_ts))
        if self.fail_with is not None and len(self.calls) >= self.fail_from:
            raise self.fail_with
        return {"ok": True, "ts": f"1700000001.{len(self.calls):06d}"}

    @property
    def texts(self) -> list[str]:
        """The ``text`` of every message posted, in order."""
        return [text for _, text, _ in self.calls]


def long_answer(lines: int = 200, width: int = 99) -> str:
    """An answer long enough to be split, from numbered lines of known length."""
    return "\n".join(f"{index:04d}{'x' * (width - 4)}" for index in range(lines))


def expected_chunks(text: str) -> list[str]:
    """The chunks the answer path must produce, composed from the exported pieces."""
    return chunk_text(
        markdown_to_mrkdwn(text), ANSWER_CHUNK_CHARS, keep_whole=MENTION_PLACEHOLDER_RE
    )


def make_ops(
    client: RecordingClient | None = None,
    *,
    version_tag: str = VERSION,
    mentions: bool = True,
) -> tuple[SlackOps, RecordingClient]:
    """A ``SlackOps`` wired to a recording client, and that client."""
    client = client if client is not None else RecordingClient()
    return SlackOps(make_config(version_tag, mentions=mentions), client, CTX), client  # type: ignore[arg-type]
