"""The posting half of NextcloudTalkOps: the raise/swallow contracts, answer chunking,
and the wording that reaches the room.

Every test drives a real :class:`TalkClient` over an :class:`httpx.MockTransport` Talk, so
the form encoding and the OCS envelope are exercised alongside the ops logic and nothing
opens a socket. Each member reads only the persisted ``entry`` — the fixtures below are
entry dicts, never wire events, which is the shape the retry drain and startup reconcile
hand in after a restart.

The most consequential assertions here are the **raise-vs-swallow matrix** (inverting one
member silently breaks delivery: a swallowed ``post_answer`` failure drops an answer the
run already paid for, while a raising ``post_ack`` costs a user their answer over a
courtesy message) and the **chunk-2 failure** pair: the raise reaches the engine, and the
retry re-posts from chunk 1 — duplicate earlier chunks are at-least-once by design.
"""

from dataclasses import dataclass
from urllib.parse import parse_qs

import httpx
import pytest

from osprey.bridges.core import RESERVED_ENTRY_KEYS
from osprey.bridges.core import coalesce_key as normalize_coalesce_key
from osprey.bridges.nextcloud_talk import NextcloudBridgeConfig, TalkClient
from osprey.bridges.nextcloud_talk.ops import (
    ACK_TEXT,
    EMPTY_ANSWER_TEXT,
    ERROR_TEXT,
    GIVEUP_TEXT,
    MAX_MESSAGE_CHARS,
    NC_MESSAGE_ID,
    NC_ROOM,
    QUEUED_TEXT,
    SUPERSEDED_TEXT,
    NextcloudTalkOps,
    _chunk,
)

CFG = NextcloudBridgeConfig(
    base_url="https://cloud.example.org",
    bot_account="osprey-bot",
    app_password="app-pw",
    rooms=("roomA",),
)

RESULT_OK = {"status": "completed", "text_output": "the orbit is flat", "error": None}
RESULT_ERROR = {
    "status": "error",
    "text_output": "",
    "error": "DispatchError: worker 500 at /dispatch/R1",
    "failure_class": "infrastructure",
}


def entry(**overrides):
    """A persisted entry as the engine hands it back after a claim."""
    base = {
        NC_ROOM: "roomA",
        NC_MESSAGE_ID: 41,
        "history_key": "nextcloud:roomA",
        "sender_id": "alice",
        "sender_display": "Alice",
        "text": "is the orbit flat?",
    }
    return {**base, **overrides}


@dataclass(frozen=True)
class Post:
    """One recorded chat post."""

    room: str
    text: str
    reply_to: int | None


class FakeTalk:
    """A Talk that records every post and can fail chosen ones.

    ``fail_on`` holds 1-based post indices that answer with ``status`` instead of
    succeeding. A failing post is still recorded *before* it fails, so a test can assert
    what was attempted as well as what landed.
    """

    def __init__(self, *, fail_on=(), status=503, exc=None):
        self.posts: list[Post] = []
        self.fail_on = set(fail_on)
        self.status = status
        self.exc = exc

    @property
    def texts(self) -> list[str]:
        """The text of every attempted post, in order."""
        return [post.text for post in self.posts]

    def handler(self, request: httpx.Request) -> httpx.Response:
        form = parse_qs(request.content.decode())
        reply_to = form["replyTo"][0] if "replyTo" in form else None
        self.posts.append(
            Post(
                room=request.url.path.rsplit("/", 1)[-1],
                text=form.get("message", [""])[0],
                reply_to=int(reply_to) if reply_to is not None else None,
            )
        )
        if len(self.posts) in self.fail_on:
            if self.exc is not None:
                raise self.exc
            return httpx.Response(self.status, text="boom")
        return httpx.Response(
            200, json={"ocs": {"meta": {"status": "ok"}, "data": {"id": 100 + len(self.posts)}}}
        )


def _ops(talk: FakeTalk | None = None) -> tuple[NextcloudTalkOps, FakeTalk]:
    """``(ops, talk)`` — real TalkClient over a MockTransport-backed FakeTalk."""
    talk = talk if talk is not None else FakeTalk()
    client = TalkClient(CFG, client=httpx.Client(transport=httpx.MockTransport(talk.handler)))
    return NextcloudTalkOps(CFG, client), talk


# Every posting member as a zero-arg call, so a property that must hold across the whole
# surface is asserted once per member.
def _calls(ops: NextcloudTalkOps, on_entry=None):
    e = entry() if on_entry is None else on_entry
    return {
        "post_ack": lambda: ops.post_ack(e),
        "post_answer": lambda: ops.post_answer(e, RESULT_OK),
        "post_queued": lambda: ops.post_queued(e, RESULT_ERROR),
        "post_giveup": lambda: ops.post_giveup(e),
        "post_superseded": lambda: ops.post_superseded(e),
    }


POSTING_MEMBERS = ["post_ack", "post_answer", "post_queued", "post_giveup", "post_superseded"]

# The seam's failure contract, as a table. Inverting one entry is the most likely defect
# in this module, and it is the kind that passes a happy-path test.
RAISES_ON_FAILURE = {
    "post_ack": False,
    "post_answer": True,
    "post_queued": True,
    "post_giveup": True,
    "post_superseded": False,
}


# ==========================================================================
# The raise/swallow matrix — the seam's load-bearing asymmetry
# ==========================================================================


@pytest.mark.parametrize("member", POSTING_MEMBERS)
def test_failure_contract_matches_the_seam(member):
    ops, talk = _ops(FakeTalk(fail_on={1}))
    call = _calls(ops)[member]
    if RAISES_ON_FAILURE[member]:
        with pytest.raises(httpx.HTTPStatusError):
            call()
    else:
        call()  # must not raise
    # Either way the post was attempted — a swallowing member still tries.
    assert len(talk.posts) == 1


@pytest.mark.parametrize("member", POSTING_MEMBERS)
def test_failure_contract_holds_for_transport_errors_too(member):
    ops, _ = _ops(FakeTalk(fail_on={1}, exc=httpx.ConnectError("no route")))
    call = _calls(ops)[member]
    if RAISES_ON_FAILURE[member]:
        with pytest.raises(httpx.ConnectError):
            call()
    else:
        call()


@pytest.mark.parametrize("member", POSTING_MEMBERS)
def test_an_entry_without_a_room_follows_the_same_contract(member):
    # A destination-less entry cannot be delivered by any later attempt either, so the
    # raising members raise (the give-up ceiling eventually stops it) and the
    # best-effort members stay quiet.
    ops, talk = _ops()
    call = _calls(ops, on_entry=entry(**{NC_ROOM: None}))[member]
    if RAISES_ON_FAILURE[member]:
        with pytest.raises(ValueError, match=f"no {NC_ROOM}"):
            call()
    else:
        call()
    assert talk.posts == []


# ==========================================================================
# post_ack
# ==========================================================================


def test_ack_replies_into_the_room_with_the_ack_wording():
    ops, talk = _ops()
    ops.post_ack(entry())
    assert talk.posts == [Post(room="roomA", text=ACK_TEXT, reply_to=41)]


def test_ack_posts_plainly_when_the_entry_has_no_reply_target():
    ops, talk = _ops()
    ops.post_ack(entry(**{NC_MESSAGE_ID: None}))
    assert talk.posts[0].reply_to is None


@pytest.mark.parametrize("bad_id", [None, "41", True, 4.5, {}])
def test_a_malformed_reply_target_degrades_to_a_plain_post(bad_id):
    # An un-anchored notice still reaches the user; raising would cost them the answer.
    ops, talk = _ops()
    ops.post_answer(entry(**{NC_MESSAGE_ID: bad_id}), RESULT_OK)
    assert talk.posts[0].reply_to is None


# ==========================================================================
# post_answer — content
# ==========================================================================


def test_answer_posts_the_completed_text_as_a_reply():
    ops, talk = _ops()
    ops.post_answer(entry(), RESULT_OK)
    assert talk.posts == [Post(room="roomA", text="the orbit is flat", reply_to=41)]


def test_answer_passes_markdown_through_untouched():
    markdown = "**Orbit** is flat.\n\n- `BPM1` 0.1 mm\n- `BPM2` -0.2 mm\n\n```\nraw\n```"
    ops, talk = _ops()
    ops.post_answer(entry(), {"status": "completed", "text_output": markdown})
    assert talk.texts == [markdown]


def test_answer_posts_the_channel_error_wording_for_a_failed_run():
    ops, talk = _ops()
    ops.post_answer(entry(), RESULT_ERROR)
    assert talk.texts == [ERROR_TEXT]


def test_answer_never_leaks_the_machine_error_into_the_room():
    ops, talk = _ops()
    ops.post_answer(entry(), RESULT_ERROR)
    posted = talk.texts[0]
    assert "DispatchError" not in posted
    assert "/dispatch/R1" not in posted
    assert "infrastructure" not in posted


@pytest.mark.parametrize("status", ["error", "timeout", "cancelled", "", None])
def test_any_non_completed_status_gets_the_error_wording(status):
    ops, talk = _ops()
    ops.post_answer(entry(), {"status": status, "text_output": "partial internals"})
    assert talk.texts == [ERROR_TEXT]


@pytest.mark.parametrize("text", ["", "   \n\t ", None, 42])
def test_a_completed_run_with_no_text_still_posts_something(text):
    # "It worked but said nothing" must not be indistinguishable from a dead bridge.
    ops, talk = _ops()
    ops.post_answer(entry(), {"status": "completed", "text_output": text})
    assert talk.texts == [EMPTY_ANSWER_TEXT]


# ==========================================================================
# post_answer — chunking at Talk's 32,000-char ceiling
# ==========================================================================


def test_an_answer_at_exactly_the_limit_is_one_message():
    ops, talk = _ops()
    text = "x" * MAX_MESSAGE_CHARS
    ops.post_answer(entry(), {"status": "completed", "text_output": text})
    assert talk.texts == [text]


def test_one_character_over_the_limit_splits_into_two_messages():
    ops, talk = _ops()
    text = "x" * (MAX_MESSAGE_CHARS + 1)
    ops.post_answer(entry(), {"status": "completed", "text_output": text})
    assert [len(chunk) for chunk in talk.texts] == [MAX_MESSAGE_CHARS, 1]
    assert "".join(talk.texts) == text


def test_chunks_are_posted_in_order_and_reassemble_exactly():
    ops, talk = _ops()
    text = "".join(f"line {index}: {'y' * 80}\n" for index in range(1200))
    ops.post_answer(entry(), {"status": "completed", "text_output": text})
    assert len(talk.texts) > 1
    assert all(len(chunk) <= MAX_MESSAGE_CHARS for chunk in talk.texts)
    assert "".join(talk.texts) == text


def test_only_the_first_chunk_quotes_the_question():
    # A multi-message answer stays anchored to the question without repeating the quote.
    ops, talk = _ops()
    ops.post_answer(entry(), {"status": "completed", "text_output": "z" * (2 * MAX_MESSAGE_CHARS)})
    assert [post.reply_to for post in talk.posts] == [41, None]


def test_every_chunk_goes_to_the_same_room():
    ops, talk = _ops()
    ops.post_answer(entry(), {"status": "completed", "text_output": "z" * (2 * MAX_MESSAGE_CHARS)})
    assert {post.room for post in talk.posts} == {"roomA"}


@pytest.mark.parametrize(
    "text",
    [
        "short",
        "x" * MAX_MESSAGE_CHARS,
        "x" * (MAX_MESSAGE_CHARS + 1),
        "x" * (3 * MAX_MESSAGE_CHARS + 7),
        ("line\n" * 20_000),
        ("a" * 40_000 + "\n" + "b" * 40_000),
        "\n" * (MAX_MESSAGE_CHARS + 5),
        "\n" + "q" * (MAX_MESSAGE_CHARS + 5),
    ],
)
def test_chunking_splits_without_reflowing(text):
    # _chunk splits and never rewrites: markdown survives byte-for-byte, so the
    # concatenation is the identity and no chunk is empty or oversized.
    chunks = _chunk(text)
    assert "".join(chunks) == text
    assert all(0 < len(chunk) <= MAX_MESSAGE_CHARS for chunk in chunks)


def test_chunking_prefers_a_line_boundary():
    text = ("x" * 99 + "\n") * 400
    chunks = _chunk(text)
    assert chunks[0].endswith("\n")
    assert len(chunks[0]) <= MAX_MESSAGE_CHARS
    # The boundary is the last newline that fits, not an arbitrary earlier one.
    assert len(chunks[0]) > MAX_MESSAGE_CHARS - 100


def test_chunking_hard_splits_a_line_with_no_boundary():
    text = "x" * (MAX_MESSAGE_CHARS + 10)
    assert _chunk(text) == [text[:MAX_MESSAGE_CHARS], text[MAX_MESSAGE_CHARS:]]


def test_chunking_empty_text_yields_no_chunks():
    assert _chunk("") == []


# ==========================================================================
# post_answer — failure mid-answer
# ==========================================================================


def _three_chunk_result():
    """A completed result whose text needs exactly three messages."""
    text = "c1" * (MAX_MESSAGE_CHARS // 2) + "c2" * (MAX_MESSAGE_CHARS // 2) + "tail"
    return {"status": "completed", "text_output": text}


def test_a_failing_second_chunk_raises_and_stops_the_answer():
    ops, talk = _ops(FakeTalk(fail_on={2}))
    with pytest.raises(httpx.HTTPStatusError):
        ops.post_answer(entry(), _three_chunk_result())
    # Chunk 1 landed, chunk 2 was attempted and failed, chunk 3 was never attempted:
    # the raise is what tells the engine to re-queue for redelivery.
    assert len(talk.posts) == 2


def test_a_retry_after_a_mid_answer_failure_re_posts_from_chunk_one():
    # At-least-once by design: the drain re-delivers the whole answer, so chunk 1 is
    # posted twice. A duplicated chunk is accepted; a silently dropped tail is not.
    result = _three_chunk_result()
    ops, talk = _ops(FakeTalk(fail_on={2}))
    with pytest.raises(httpx.HTTPStatusError):
        ops.post_answer(entry(), result)
    first_attempt = list(talk.texts)

    ops.post_answer(entry(), result)  # the drain's redelivery; succeeds this time

    redelivered = talk.texts[len(first_attempt) :]
    assert redelivered[0] == first_attempt[0]
    assert "".join(redelivered) == result["text_output"]


def test_a_failing_first_chunk_raises_before_anything_lands():
    ops, talk = _ops(FakeTalk(fail_on={1}))
    with pytest.raises(httpx.HTTPStatusError):
        ops.post_answer(entry(), _three_chunk_result())
    assert len(talk.posts) == 1


def test_a_failing_final_chunk_still_raises():
    # The tail is the easiest failure to swallow by accident and the worst to lose: the
    # user would read a truncated answer as a complete one.
    ops, talk = _ops(FakeTalk(fail_on={3}))
    with pytest.raises(httpx.HTTPStatusError):
        ops.post_answer(entry(), _three_chunk_result())
    assert len(talk.posts) == 3


# ==========================================================================
# post_queued / post_giveup / post_superseded wording
# ==========================================================================


def test_queued_notice_replies_with_the_queued_wording():
    ops, talk = _ops()
    ops.post_queued(entry(), RESULT_ERROR)
    assert talk.posts == [Post(room="roomA", text=QUEUED_TEXT, reply_to=41)]


def test_queued_notice_never_leaks_the_failure_that_caused_the_park():
    ops, talk = _ops()
    ops.post_queued(entry(), RESULT_ERROR)
    assert "DispatchError" not in talk.texts[0]
    assert "infrastructure" not in talk.texts[0]


def test_giveup_notice_replies_with_the_giveup_wording():
    ops, talk = _ops()
    ops.post_giveup(entry())
    assert talk.posts == [Post(room="roomA", text=GIVEUP_TEXT, reply_to=41)]


def test_giveup_notice_never_leaks_the_machine_reason():
    ops, talk = _ops()
    ops.post_giveup(entry(give_up_reason="worker unreachable for 48h", attempts=17))
    posted = talk.texts[0]
    assert "worker unreachable" not in posted
    assert "48h" not in posted
    assert "17" not in posted


def test_superseded_note_replies_to_the_superseded_message():
    # The note belongs on the question it is about, not on the newest one.
    ops, talk = _ops()
    ops.post_superseded(entry(**{NC_MESSAGE_ID: 7}))
    assert talk.posts == [Post(room="roomA", text=SUPERSEDED_TEXT, reply_to=7)]


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        ("post_ack", ACK_TEXT),
        ("post_answer", "the orbit is flat"),
        ("post_queued", QUEUED_TEXT),
        ("post_giveup", GIVEUP_TEXT),
        ("post_superseded", SUPERSEDED_TEXT),
    ],
)
def test_every_member_posts_into_the_entrys_room_as_a_reply(member, expected):
    ops, talk = _ops()
    _calls(ops, on_entry=entry(**{NC_ROOM: "roomB", NC_MESSAGE_ID: 5}))[member]()
    assert talk.posts == [Post(room="roomB", text=expected, reply_to=5)]


# ==========================================================================
# coalesce_key — pure
# ==========================================================================


def test_coalesce_key_groups_by_conversation_and_sender():
    ops, _ = _ops()
    assert ops.coalesce_key(entry()) == ["nextcloud:roomA", "alice"]


def test_coalesce_key_normalizes_to_the_engines_tuple():
    ops, _ = _ops()
    parked = {"coalesce_key": ops.coalesce_key(entry())}
    assert normalize_coalesce_key(parked) == ("nextcloud:roomA", "alice")


def test_two_senders_in_one_room_do_not_coalesce_together():
    ops, _ = _ops()
    assert ops.coalesce_key(entry()) != ops.coalesce_key(entry(sender_id="bob"))


def test_one_sender_across_two_rooms_does_not_coalesce_together():
    ops, _ = _ops()
    assert ops.coalesce_key(entry()) != ops.coalesce_key(entry(history_key="nextcloud:roomB"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"history_key": ""},
        {"history_key": None},
        {"sender_id": ""},
        {"sender_id": None},
        {"history_key": 7},
        {"sender_id": ["alice"]},
    ],
)
def test_coalesce_key_stands_alone_when_either_half_is_missing(overrides):
    # Standing alone is fail-closed: a wrong group silently discards a question.
    ops, _ = _ops()
    assert ops.coalesce_key(entry(**overrides)) == ()
    assert normalize_coalesce_key({"coalesce_key": ops.coalesce_key(entry(**overrides))}) == ()


def test_coalesce_key_does_no_io():
    ops, talk = _ops()
    ops.coalesce_key(entry())
    assert talk.posts == []


# ==========================================================================
# Entry-key hygiene and construction
# ==========================================================================


def test_adapter_entry_keys_do_not_collide_with_the_engines():
    # A collision here is a silent overwrite in one direction or the other, so the
    # nc_ prefix is the whole safety mechanism.
    assert {NC_ROOM, NC_MESSAGE_ID}.isdisjoint(RESERVED_ENTRY_KEYS)
    assert all(key.startswith("nc_") for key in (NC_ROOM, NC_MESSAGE_ID))


def test_ops_builds_its_own_talk_client_when_none_is_injected():
    ops = NextcloudTalkOps(CFG)
    try:
        assert isinstance(ops._client, TalkClient)
    finally:
        ops._client.close()
