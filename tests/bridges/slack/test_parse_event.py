"""Slack ``event_callback`` payloads in, :class:`InboundEvent` (or nothing) out."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from osprey.bridges.core import RESERVED_ENTRY_KEYS
from osprey.bridges.slack.events import (
    SL_CHANNEL,
    SL_CHANNEL_TYPE,
    SL_FILES,
    SL_TEAM,
    SL_TEXT_HEAD,
    SL_THREAD_TS,
    SL_TS,
    BotIdentity,
    parse_event,
    people_seen,
)

BOT_USER = "UBOT123"
BOT_ID = "B0OSPREY"
TEAM = "T0HOME"
FOREIGN = "T0OTHER"
CTX = BotIdentity(bot_user_id=BOT_USER, bot_id=BOT_ID, team_id=TEAM)

ALICE = "UALICE1"
DM = "D0DIRECT"
CHAN = "C0CHAN"
MPIM = "G0MPIM"
TS = "1700000000.000100"
PARENT = "1699999999.000001"


def _event(
    type_: str = "message",
    text: str = "what is the beam current?",
    *,
    channel: str = DM,
    channel_type: str | None = "im",
    user: str = ALICE,
    ts: str = TS,
    thread_ts: str | None = None,
    profile: dict | None = None,
    **extra,
) -> dict:
    event = {
        "type": type_,
        "user": user,
        "text": text,
        "ts": ts,
        "channel": channel,
        "team": TEAM,
        "user_team": TEAM,
        "user_profile": (
            {"display_name": "Alice", "real_name": "Alice Example", "team": TEAM}
            if profile is None
            else profile
        ),
    }
    if channel_type is not None:
        event["channel_type"] = channel_type
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    event.update(extra)
    return event


def _payload(event: dict, *, is_ext_shared_channel: bool = False, team_id: str = TEAM) -> dict:
    return {
        "type": "event_callback",
        "team_id": team_id,
        "event_id": "Ev0001",
        "event": event,
        "is_ext_shared_channel": is_ext_shared_channel,
        "authorizations": [{"team_id": team_id, "user_id": BOT_USER, "is_bot": True}],
    }


def dm(text: str = "what is the beam current?", **kw) -> dict:
    return _payload(_event("message", text, **kw))


def mention(question: str = "what is the beam current?", **kw) -> dict:
    """A channel question as Slack sends it: ``app_mention`` WITHOUT ``channel_type``."""
    kw.setdefault("channel", CHAN)
    kw.setdefault("channel_type", None)
    return _payload(_event("app_mention", f"<@{BOT_USER}> {question}".strip(), **kw))


def _file(fid: str = "F0001", **extra) -> dict:
    entry = {
        "id": fid,
        "name": "orbit.png",
        "title": "orbit",
        "mimetype": "image/png",
        "filetype": "png",
        "size": 1234,
        "url_private": f"https://files.slack.com/files-pri/{TEAM}-{fid}/orbit.png",
        "url_private_download": f"https://files.slack.com/files-pri/{TEAM}-{fid}/download/orbit.png",
    }
    entry.update(extra)
    return entry


# --- accepted shapes ---------------------------------------------------------


def test_a_dm_parses_with_identity_and_dm_placement():
    ev = parse_event(dm(), CTX)
    assert ev is not None
    assert ev.message_id == f"{DM}:{TS}"
    assert ev.text == "what is the beam current?"
    assert ev.sender_id == ALICE
    assert ev.sender_display == "Alice"
    assert ev.history_key == DM
    assert ev.claim_meta[SL_CHANNEL] == DM
    assert ev.claim_meta[SL_TS] == TS
    assert ev.claim_meta[SL_THREAD_TS] == ""
    assert ev.claim_meta[SL_CHANNEL_TYPE] == "im"
    assert ev.claim_meta[SL_TEAM] == TEAM
    assert ev.claim_meta[SL_FILES] == []
    assert ev.claim_meta[SL_TEXT_HEAD] == "what is the beam current?"


def test_a_threaded_dm_answers_into_its_thread_but_keys_history_on_the_channel():
    ev = parse_event(dm(thread_ts=PARENT), CTX)
    assert ev is not None
    assert ev.claim_meta[SL_THREAD_TS] == PARENT
    assert ev.history_key == DM


def test_a_channel_mention_without_channel_type_parses_as_a_channel_thread():
    payload = mention()
    assert "channel_type" not in payload["event"]
    ev = parse_event(payload, CTX)
    assert ev is not None
    assert ev.text == "what is the beam current?"
    assert ev.message_id == f"{CHAN}:{TS}"
    assert ev.history_key == f"{CHAN}:{TS}"
    assert ev.claim_meta[SL_THREAD_TS] == TS
    assert ev.claim_meta[SL_CHANNEL_TYPE] == "channel"


def test_a_channel_mention_in_a_thread_keys_on_the_thread_root():
    ev = parse_event(mention(thread_ts=PARENT), CTX)
    assert ev is not None
    assert ev.history_key == f"{CHAN}:{PARENT}"
    assert ev.claim_meta[SL_THREAD_TS] == PARENT
    assert ev.message_id == f"{CHAN}:{TS}"


def test_an_app_mention_in_a_d_channel_without_channel_type_is_a_dm():
    ev = parse_event(mention(channel=DM), CTX)
    assert ev is not None
    assert ev.claim_meta[SL_CHANNEL_TYPE] == "im"
    assert ev.history_key == DM
    assert ev.claim_meta[SL_THREAD_TS] == ""


def test_an_app_mention_in_a_private_channel_without_channel_type_is_a_non_dm_room():
    ev = parse_event(mention(channel="G0PRIV"), CTX)
    assert ev is not None
    assert ev.claim_meta[SL_CHANNEL_TYPE] != "im"
    assert ev.history_key == f"G0PRIV:{TS}"


def test_an_mpim_message_with_the_bot_mention_parses_as_a_threaded_room():
    payload = _payload(
        _event("message", f"<@{BOT_USER}> status?", channel=MPIM, channel_type="mpim")
    )
    ev = parse_event(payload, CTX)
    assert ev is not None
    assert ev.text == "status?"
    assert ev.claim_meta[SL_CHANNEL_TYPE] == "mpim"
    assert ev.history_key == f"{MPIM}:{TS}"
    assert ev.claim_meta[SL_THREAD_TS] == TS


def test_both_deliveries_of_one_mpim_post_share_a_message_id():
    text = f"<@{BOT_USER}> status?"
    as_message = _payload(_event("message", text, channel=MPIM, channel_type="mpim"))
    as_mention = _payload(_event("app_mention", text, channel=MPIM, channel_type=None))
    first, second = parse_event(as_message, CTX), parse_event(as_mention, CTX)
    assert first is not None and second is not None
    assert first.message_id == second.message_id
    assert first.history_key == second.history_key


def test_a_labelled_bot_mention_token_counts_as_addressing():
    payload = _payload(
        _event("message", f"<@{BOT_USER}|osprey> status?", channel=MPIM, channel_type="mpim")
    )
    ev = parse_event(payload, CTX)
    assert ev is not None and ev.text == "status?"


def test_other_mentions_are_kept_and_the_sender_is_named_from_its_profile():
    ev = parse_event(dm(f"ask <@{ALICE}> and <@UCAROL9>"), CTX)
    assert ev is not None
    assert ev.text == "ask Alice and @UCAROL9"


def test_raw_is_kept_on_the_event_but_never_in_claim_meta():
    payload = dm()
    ev = parse_event(payload, CTX)
    assert ev is not None
    assert ev.raw is payload
    assert "event" not in ev.claim_meta
    assert all(key.startswith("sl_") for key in ev.claim_meta)
    assert not set(ev.claim_meta) & RESERVED_ENTRY_KEYS


def test_the_text_head_is_the_first_line_only():
    ev = parse_event(dm("first line\nsecond line"), CTX)
    assert ev is not None
    assert ev.text == "first line\nsecond line"
    assert ev.claim_meta[SL_TEXT_HEAD] == "first line"


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ({"display_name": "Ali", "real_name": "Alice Example"}, "Ali"),
        ({"display_name": "", "real_name": "Alice Example"}, "Alice Example"),
        ({"display_name": "  ", "real_name": ""}, ALICE),
        ({}, ALICE),
        ("not a mapping", ALICE),
    ],
)
def test_sender_display_prefers_display_name_then_real_name_then_the_id(profile, expected):
    ev = parse_event(dm(profile=profile), CTX)
    assert ev is not None
    assert ev.sender_display == expected


def test_the_context_may_be_any_object_with_the_three_attributes():
    ctx = SimpleNamespace(bot_user_id=BOT_USER, bot_id=BOT_ID, team_id=TEAM)
    assert parse_event(mention(), ctx) is not None


# --- files ---------------------------------------------------------------------


def test_a_file_share_carries_file_refs_preferring_the_download_url():
    payload = dm("see attached", files=[_file()], subtype="file_share")
    ev = parse_event(payload, CTX)
    assert ev is not None
    assert ev.claim_meta[SL_FILES] == [
        {
            "id": "F0001",
            "name": "orbit.png",
            "mimetype": "image/png",
            "size": 1234,
            "url": f"https://files.slack.com/files-pri/{TEAM}-F0001/download/orbit.png",
        }
    ]


def test_a_file_without_a_download_url_falls_back_to_url_private():
    f = _file()
    del f["url_private_download"]
    ev = parse_event(dm("x", files=[f], subtype="file_share"), CTX)
    assert ev is not None
    assert ev.claim_meta[SL_FILES][0]["url"] == f["url_private"]


def test_a_file_share_with_no_text_still_parses():
    ev = parse_event(dm("", files=[_file()], subtype="file_share"), CTX)
    assert ev is not None
    assert ev.text == ""
    assert ev.claim_meta[SL_TEXT_HEAD] == ""
    assert len(ev.claim_meta[SL_FILES]) == 1


def test_malformed_file_entries_are_skipped():
    files = ["junk", {"name": "no-id.txt"}, _file("F0002")]
    ev = parse_event(dm("x", files=files, subtype="file_share"), CTX)
    assert ev is not None
    assert [f["id"] for f in ev.claim_meta[SL_FILES]] == ["F0002"]


def test_the_claim_meta_is_json_plain():
    import json

    ev = parse_event(dm("x", files=[_file()], subtype="file_share"), CTX)
    assert ev is not None
    assert json.loads(json.dumps(dict(ev.claim_meta))) == dict(ev.claim_meta)


# --- ignored events -----------------------------------------------------------


def _with(payload: dict, **changes) -> dict:
    payload = copy.deepcopy(payload)
    payload["event"].update(changes)
    return payload


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(_with(dm(), type="reaction_added"), id="other-event-type"),
        pytest.param(_with(dm(), type="app_home_opened"), id="app-home"),
        pytest.param(_with(mention(), edited={"user": ALICE, "ts": TS}), id="edited-app-mention"),
        pytest.param(_with(dm(), subtype="message_changed"), id="message-changed"),
        pytest.param(_with(dm(), subtype="message_deleted"), id="message-deleted"),
        pytest.param(_with(dm(), subtype="bot_message"), id="bot-message-subtype"),
        pytest.param(_with(dm(), subtype="channel_join"), id="channel-join"),
        pytest.param(_with(dm(), subtype="thread_broadcast"), id="thread-broadcast"),
        pytest.param(_with(dm(), bot_id="B0OTHER"), id="bot-id-present"),
        pytest.param(_with(dm(), bot_id=BOT_ID), id="own-bot-id"),
        pytest.param(_with(dm(), app_id="A0APP"), id="app-id-present"),
        pytest.param(_with(dm(), user=BOT_USER), id="own-user"),
        pytest.param(_with(mention(), user=BOT_USER), id="own-user-mention"),
        pytest.param(_with(dm(), hidden=True), id="hidden"),
        pytest.param(_with(mention(), user_team=FOREIGN), id="foreign-user-team"),
        pytest.param(_with(mention(), team=FOREIGN), id="foreign-team"),
        pytest.param(_with(dm(), user_team=FOREIGN), id="foreign-user-team-dm"),
        pytest.param(_with(dm(), text="   "), id="whitespace-only"),
        pytest.param(_with(dm(), text=""), id="empty"),
        pytest.param(mention(""), id="bare-mention"),
        pytest.param(_with(dm(), channel=""), id="no-channel"),
        pytest.param(_with(dm(), ts=""), id="no-ts"),
        pytest.param(_with(dm(), user=""), id="no-user"),
        pytest.param(
            _with(dm(), text="", subtype="file_share", files=[]), id="file-share-no-text-no-files"
        ),
    ],
)
def test_events_that_are_not_questions_for_us_are_ignored(payload):
    assert parse_event(payload, CTX) is None


def test_an_ext_shared_channel_payload_is_ignored_even_with_a_home_user():
    assert parse_event(mention(), CTX) is not None
    assert parse_event(_payload(mention()["event"], is_ext_shared_channel=True), CTX) is None


def test_an_mpim_message_without_the_bot_mention_is_ignored():
    payload = _payload(_event("message", "just chatting", channel=MPIM, channel_type="mpim"))
    assert parse_event(payload, CTX) is None


@pytest.mark.parametrize("channel_type", ["channel", "group"])
def test_a_room_message_without_the_bot_mention_is_ignored(channel_type):
    payload = _payload(_event("message", "hello all", channel=CHAN, channel_type=channel_type))
    assert parse_event(payload, CTX) is None


def test_a_mention_of_someone_whose_id_extends_the_bots_does_not_address_the_bot():
    text = f"<@{BOT_USER}X> hello"
    payload = _payload(_event("message", text, channel=MPIM, channel_type="mpim"))
    assert parse_event(payload, CTX) is None


def test_an_app_mention_is_exempt_from_the_mention_token_check():
    payload = _payload(_event("app_mention", "no token here", channel=CHAN, channel_type=None))
    ev = parse_event(payload, CTX)
    assert ev is not None and ev.text == "no token here"


def test_a_dm_needs_no_mention_token():
    assert parse_event(dm("plain question"), CTX) is not None


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "string",
        42,
        [],
        {},
        {"event": None},
        {"event": "x"},
        {"event": {"type": "message"}},
        {"event": {"type": "message", "text": 5, "user": ALICE, "channel": DM, "ts": TS}},
        {"event": {"type": ["message"], "text": "hi", "user": ALICE, "channel": DM, "ts": TS}},
        {"event": {"type": "message", "text": "hi", "user": 7, "channel": DM, "ts": TS}},
        {"event": {"type": "message", "text": "hi", "user": ALICE, "channel": DM, "ts": 1.5}},
    ],
)
def test_malformed_payloads_return_none_without_raising(raw):
    assert parse_event(raw, CTX) is None


@pytest.mark.parametrize("ctx", [None, "ctx", SimpleNamespace(), SimpleNamespace(bot_user_id=5)])
def test_a_broken_context_never_raises(ctx):
    # Whatever happens, no exception; a DM is still answerable with no bot identity.
    parse_event(dm(), ctx)
    parse_event(mention(), ctx)


def test_without_a_bot_user_id_a_room_message_is_never_addressed():
    ctx = BotIdentity(bot_user_id="", bot_id=BOT_ID, team_id=TEAM)
    payload = _payload(_event("message", "<@> hi", channel=MPIM, channel_type="mpim"))
    assert parse_event(payload, ctx) is None


# --- people_seen ----------------------------------------------------------------


def test_people_seen_reports_the_sender_from_its_profile():
    assert people_seen(mention()) == (CHAN, {ALICE: "Alice"})


def test_people_seen_falls_back_to_real_name():
    payload = mention(profile={"display_name": "", "real_name": "Alice Example"})
    assert people_seen(payload) == (CHAN, {ALICE: "Alice Example"})


def test_people_seen_skips_a_sender_without_a_name():
    assert people_seen(mention(profile={})) == (CHAN, {})


def test_people_seen_skips_bot_posts():
    payload = _with(dm(), bot_id="B0OTHER")
    assert people_seen(payload) == (DM, {})


def test_people_seen_skips_non_user_ids():
    payload = _with(dm(), user="B0NOTAUSER")
    assert people_seen(payload) == (DM, {})


@pytest.mark.parametrize("raw", [None, "x", {}, {"event": []}, {"event": {"channel": 3}}])
def test_people_seen_never_raises_on_malformed_input(raw):
    assert people_seen(raw) == ("", {})
