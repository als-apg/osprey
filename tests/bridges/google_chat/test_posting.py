"""GoogleChatOps' outbound half: the ack, the answer, and the three retry-queue notices.

Every test drives the real :class:`ChatClient` over a ``MagicMock`` discovery service, so
what is asserted is the message body Chat would actually receive — text, threading and
``cardsV2`` together — rather than a stand-in's idea of one. Artifact publication is
injected as two fake seams, so no GCS call, no worker call, and no Google library is
involved anywhere in this file.

The properties that get the most attention are the ones a live bridge breaks quietly:

* the answer's ``text_output`` must come back **byte-identical** — the same string is
  replayed to the agent as history, so a presentation transform that followed it there
  would change what the agent sees per channel;
* the failure contracts are asymmetric on purpose (``post_answer`` and ``post_giveup``
  raise; the ack and the superseded note swallow), and inverting one silently drops a
  user's answer rather than failing loudly;
* an artifact is mapped to a URL only when the correspondence is certain, and an image
  the PNG publisher declines is re-offered as a document rather than dropped;
* no machine detail — above all ``result["error"]`` — ever reaches the space.
"""

from typing import Any
from unittest.mock import MagicMock, call

import pytest

from osprey.bridges.google_chat.client import MAX_CHARS, REPLY_OPTION, ChatClient
from osprey.bridges.google_chat.config import GoogleChatBridgeConfig
from osprey.bridges.google_chat.events import GC_SPACE, GC_THREAD
from osprey.bridges.google_chat.ops import (
    ACK_TEXT,
    EMPTY_ANSWER_TEXT,
    ERROR_TEXT,
    GIVEUP_TEXT,
    QUEUED_TEXT,
    SUPERSEDED_TEXT,
    GoogleChatOps,
)

CFG = GoogleChatBridgeConfig(
    sa_key="/etc/osprey/sa-key.json",
    subscription="projects/p/subscriptions/s",
    app_id="users/1234567890",
    gcs_bucket="test-bucket",
)

SPACE = "spaces/AAAA"
THREAD = "spaces/AAAA/threads/TTTT"
RUN_ID = "R1"

CREATED = {"name": "spaces/AAAA/messages/MMMM", "thread": {"name": THREAD}}

ENTRY = {GC_SPACE: SPACE, GC_THREAD: THREAD}

PLOT_URL = "https://storage.googleapis.com/b/plots/1.png"
OTHER_PLOT_URL = "https://storage.googleapis.com/b/plots/2.png"
DOC_URL = "https://storage.googleapis.com/b/docs/report.md"


# --- the Chat service stand-in ---------------------------------------------


def make_service() -> MagicMock:
    service = MagicMock()
    service.spaces.return_value.messages.return_value.create.return_value.execute.return_value = (
        CREATED
    )
    return service


def creates(service: MagicMock) -> Any:
    """The ``messages.create`` mock, whose call list is the posted-message record."""
    return service.spaces.return_value.messages.return_value.create


def bodies(service: MagicMock) -> list[dict[str, Any]]:
    """Every message body posted, in order."""
    return [c.kwargs["body"] for c in creates(service).call_args_list]


def only_body(service: MagicMock) -> dict[str, Any]:
    posted = bodies(service)
    assert len(posted) == 1, f"expected exactly one message, got {posted}"
    return posted[0]


def make_ops(
    cfg: GoogleChatBridgeConfig = CFG,
    *,
    publisher: Any = None,
    doc_publisher: Any = None,
) -> tuple[GoogleChatOps, MagicMock]:
    """An ops instance over a mock service, with both publishers faked to publish
    nothing unless a test says otherwise."""
    service = make_service()
    ops = GoogleChatOps(
        cfg,
        ChatClient(cfg, service),
        publisher if publisher is not None else MagicMock(return_value=[]),
        doc_publisher if doc_publisher is not None else MagicMock(return_value=[]),
    )
    return ops, service


def fail_every_create(service: MagicMock, error: Exception | None = None) -> None:
    creates(service).return_value.execute.side_effect = error or RuntimeError("chat api 500")


# --- result / descriptor builders ------------------------------------------


def image_descriptor(artifact_id: str, filename: str | None = None) -> dict[str, Any]:
    """A run-status descriptor the worker INTENDED to deliver as a PNG."""
    return {
        "artifact_id": artifact_id,
        "filename": filename or f"{artifact_id}.png",
        "source_mime": "image/png",
        "delivered_mime": "image/png",
        "convertible": True,
    }


def doc_descriptor(
    artifact_id: str, mime: str = "text/markdown", filename: str = "report.md"
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "filename": filename,
        "source_mime": mime,
        "delivered_mime": mime,
        "convertible": True,
    }


def published_document(url: str = DOC_URL, filename: str = "report.md") -> dict[str, str]:
    """One :func:`publish_documents` result."""
    return {"url": url, "filename": filename, "mime": "text/markdown"}


def completed(text: str = "42 mA", artifacts: list[Any] | None = None) -> dict[str, Any]:
    return {
        "status": "completed",
        "text_output": text,
        "run_id": RUN_ID,
        "error": None,
        "artifacts": artifacts if artifacts is not None else [],
    }


# --- post_ack ---------------------------------------------------------------


def test_the_ack_posts_the_fixed_wording_into_the_messages_thread():
    ops, service = make_ops()
    ops.post_ack(ENTRY)

    call_kwargs = creates(service).call_args.kwargs
    assert call_kwargs["parent"] == SPACE
    assert call_kwargs["messageReplyOption"] == REPLY_OPTION
    assert call_kwargs["body"] == {"text": ACK_TEXT, "thread": {"name": THREAD}}


def test_the_ack_appends_the_configured_version_tag():
    ops, service = make_ops(GoogleChatBridgeConfig(app_id=CFG.app_id, version_tag="v2026.7.1"))
    ops.post_ack(ENTRY)

    assert only_body(service)["text"] == f"{ACK_TEXT}  [v2026.7.1]"


def test_an_unset_version_tag_leaves_the_ack_untouched():
    # The suffix is opt-in: an untagged deployment posts the bare wording, with no
    # trailing whitespace or empty brackets left behind.
    ops, service = make_ops(GoogleChatBridgeConfig(app_id=CFG.app_id))
    ops.post_ack(ENTRY)

    assert only_body(service)["text"] == ACK_TEXT


def test_a_failing_ack_is_swallowed_so_the_dispatch_continues():
    ops, service = make_ops()
    fail_every_create(service)

    ops.post_ack(ENTRY)  # must not raise: the ack is a courtesy, the answer is not


def test_an_entry_naming_no_space_never_raises_out_of_the_ack():
    ops, service = make_ops()

    ops.post_ack({GC_THREAD: THREAD})

    assert creates(service).call_count == 0


# --- post_answer: the text ---------------------------------------------------


def test_the_answer_is_posted_through_the_markdown_to_chat_transform():
    ops, service = make_ops()
    ops.post_answer(ENTRY, completed("**bold** and *italic*"))

    assert only_body(service)["text"] == "*bold* and _italic_"


def test_the_results_text_output_is_left_byte_identical():
    # The identical-agent guardrail: this string is replayed to the agent as history,
    # so the Chat transform must run on a local copy and never on the result.
    ops, _ = make_ops()
    result = completed("**bold**")
    ops.post_answer(ENTRY, result)

    assert result["text_output"] == "**bold**"


def test_the_answer_is_threaded_onto_the_question():
    ops, service = make_ops()
    ops.post_answer(ENTRY, completed())

    call_kwargs = creates(service).call_args.kwargs
    assert call_kwargs["parent"] == SPACE
    assert call_kwargs["messageReplyOption"] == REPLY_OPTION
    assert call_kwargs["body"]["thread"] == {"name": THREAD}


def test_an_entry_with_no_thread_opens_a_new_one():
    ops, service = make_ops()
    ops.post_answer({GC_SPACE: SPACE, GC_THREAD: None}, completed())

    assert "thread" not in only_body(service)


def test_a_long_answer_is_split_into_threaded_chunks_within_the_limit():
    ops, service = make_ops()
    ops.post_answer(ENTRY, completed("y" * (MAX_CHARS + 500)))

    posted = bodies(service)
    assert len(posted) == 2
    for body in posted:
        assert len(body["text"]) <= MAX_CHARS
        assert body["thread"] == {"name": THREAD}


@pytest.mark.parametrize("text", ["", "   ", None])
def test_a_completed_run_with_no_text_still_posts_something(text):
    # A successful-but-silent run must not be indistinguishable from a dead bridge.
    ops, service = make_ops()
    ops.post_answer(ENTRY, completed(text))

    assert only_body(service)["text"] == EMPTY_ANSWER_TEXT


@pytest.mark.parametrize("status", ["error", "timeout", "failed", None])
def test_a_non_completed_status_posts_the_fixed_error_wording(status):
    ops, service = make_ops()
    ops.post_answer(ENTRY, {"status": status, "text_output": "half an answer", "error": "boom"})

    assert only_body(service)["text"] == ERROR_TEXT


def test_the_results_error_string_never_reaches_the_space():
    ops, service = make_ops()
    ops.post_answer(
        ENTRY,
        {
            "status": "error",
            "text_output": "",
            "run_id": RUN_ID,
            "error": "Traceback: token=sk-secret at provider.py:41",
        },
    )

    text = only_body(service)["text"]
    assert "Traceback" not in text
    assert "sk-secret" not in text


def test_a_failed_status_never_publishes_artifacts():
    publisher = MagicMock(return_value=[PLOT_URL])
    ops, _ = make_ops(publisher=publisher)
    ops.post_answer(
        ENTRY,
        {
            "status": "error",
            "text_output": "",
            "run_id": RUN_ID,
            "artifacts": [image_descriptor("a1")],
        },
    )

    publisher.assert_not_called()


def test_a_failing_chunk_post_raises_so_the_engine_redelivers():
    ops, service = make_ops()
    fail_every_create(service)

    with pytest.raises(RuntimeError):
        ops.post_answer(ENTRY, completed())


def test_an_entry_naming_no_space_raises_rather_than_dropping_the_answer():
    ops, service = make_ops()

    with pytest.raises(ValueError):
        ops.post_answer({GC_THREAD: THREAD}, completed())
    assert creates(service).call_count == 0


# --- post_answer: the cards --------------------------------------------------


def test_a_published_image_rides_the_answer_as_one_image_card():
    publisher = MagicMock(return_value=[PLOT_URL])
    ops, service = make_ops(publisher=publisher)
    ops.post_answer(ENTRY, completed("here's the plot", [image_descriptor("a1")]))

    body = only_body(service)
    assert body["text"] == "here's the plot"
    cards = body["cardsV2"]
    assert len(cards) == 1
    assert cards[0]["card"]["sections"][0]["widgets"] == [
        {"image": {"imageUrl": PLOT_URL, "onClick": {"openLink": {"url": PLOT_URL}}}}
    ]
    publisher.assert_called_once_with(CFG, RUN_ID, ["a1"])


def test_each_image_widget_links_to_its_own_full_resolution_url():
    # Chat downscales the inline preview, so the click target must be the image's own
    # object — not the card's, and not the first artifact's for all of them.
    publisher = MagicMock(side_effect=[[PLOT_URL], [OTHER_PLOT_URL]])
    ops, service = make_ops(publisher=publisher)
    ops.post_answer(ENTRY, completed("two plots", [image_descriptor("a1"), image_descriptor("a2")]))

    widgets = only_body(service)["cardsV2"][0]["card"]["sections"][0]["widgets"]
    for widget, url in zip(widgets, [PLOT_URL, OTHER_PLOT_URL], strict=True):
        assert widget["image"]["imageUrl"] == url
        assert widget["image"]["onClick"]["openLink"]["url"] == url


def test_each_artifact_is_offered_to_the_publisher_on_its_own_call():
    # One artifact per call is what makes a short result attributable: the publisher
    # drops what it could not publish, so a bucket-at-a-time call could not say which.
    publisher = MagicMock(side_effect=[[PLOT_URL], [OTHER_PLOT_URL]])
    ops, _ = make_ops(publisher=publisher)
    ops.post_answer(ENTRY, completed("two plots", [image_descriptor("a1"), image_descriptor("a2")]))

    assert publisher.call_args_list == [call(CFG, RUN_ID, ["a1"]), call(CFG, RUN_ID, ["a2"])]


def test_only_the_last_chunk_of_a_long_answer_carries_the_cards():
    # A card on its own message could land even if a text chunk failed, and a
    # redelivery would then duplicate just the card.
    publisher = MagicMock(return_value=[PLOT_URL])
    ops, service = make_ops(publisher=publisher)
    ops.post_answer(ENTRY, completed("z" * (MAX_CHARS + 500), [image_descriptor("a1")]))

    posted = bodies(service)
    assert len(posted) == 2
    assert "cardsV2" not in posted[0]
    assert len(posted[1]["cardsV2"]) == 1


def test_documents_ride_the_answer_as_one_button_card_in_order():
    docs = [
        published_document("https://storage.googleapis.com/b/docs/1.md", "one.md"),
        published_document("https://storage.googleapis.com/b/docs/2.csv", "two.csv"),
    ]
    doc_publisher = MagicMock(side_effect=[[docs[0]], [docs[1]]])
    ops, service = make_ops(doc_publisher=doc_publisher)
    ops.post_answer(
        ENTRY,
        completed("two files", [doc_descriptor("d1"), doc_descriptor("d2", "text/csv", "two.csv")]),
    )

    cards = only_body(service)["cardsV2"]
    assert len(cards) == 1
    assert cards[0]["card"]["sections"][0]["widgets"] == [
        {
            "buttonList": {
                "buttons": [
                    {"text": "📄 Open one.md", "onClick": {"openLink": {"url": docs[0]["url"]}}},
                    {"text": "📄 Open two.csv", "onClick": {"openLink": {"url": docs[1]["url"]}}},
                ]
            }
        }
    ]


def test_the_image_card_comes_before_the_document_card():
    # An image-only answer must keep the body it had before documents were deliverable
    # at all, which is what fixes this order rather than leaving it incidental.
    publisher = MagicMock(return_value=[PLOT_URL])
    doc_publisher = MagicMock(return_value=[published_document()])
    ops, service = make_ops(publisher=publisher, doc_publisher=doc_publisher)
    ops.post_answer(
        ENTRY, completed("plot and report", [image_descriptor("a1"), doc_descriptor("d1")])
    )

    cards = only_body(service)["cardsV2"]
    assert len(cards) == 2
    assert "image" in cards[0]["card"]["sections"][0]["widgets"][0]
    assert "buttonList" in cards[1]["card"]["sections"][0]["widgets"][0]


def test_the_two_buckets_are_offered_to_their_own_publishers():
    publisher = MagicMock(return_value=[PLOT_URL])
    doc_publisher = MagicMock(return_value=[published_document()])
    images = image_descriptor("a1")
    documents = doc_descriptor("d1")
    ops, _ = make_ops(publisher=publisher, doc_publisher=doc_publisher)
    ops.post_answer(ENTRY, completed("plot and report", [images, documents]))

    publisher.assert_called_once_with(CFG, RUN_ID, ["a1"])
    doc_publisher.assert_called_once_with(CFG, RUN_ID, [documents])


def test_a_run_with_no_artifacts_posts_the_text_only_body_and_calls_no_publisher():
    publisher = MagicMock(return_value=[PLOT_URL])
    doc_publisher = MagicMock(return_value=[published_document()])
    ops, service = make_ops(publisher=publisher, doc_publisher=doc_publisher)
    ops.post_answer(ENTRY, completed("no plots here"))

    publisher.assert_not_called()
    doc_publisher.assert_not_called()
    assert only_body(service) == {"text": "no plots here", "thread": {"name": THREAD}}


def test_a_result_with_no_run_id_publishes_nothing():
    publisher = MagicMock(return_value=[PLOT_URL])
    ops, service = make_ops(publisher=publisher)
    result = completed("answer", [image_descriptor("a1")])
    result["run_id"] = None
    ops.post_answer(ENTRY, result)

    publisher.assert_not_called()
    assert "cardsV2" not in only_body(service)


def test_the_run_id_persisted_on_the_entry_is_used_when_the_result_carries_none():
    # The drain's re-attached delivery answers a run the result no longer names.
    publisher = MagicMock(return_value=[PLOT_URL])
    ops, service = make_ops(publisher=publisher)
    result = completed("answer", [image_descriptor("a1")])
    result["run_id"] = None
    ops.post_answer({**ENTRY, "run_id": RUN_ID}, result)

    publisher.assert_called_once_with(CFG, RUN_ID, ["a1"])
    assert "cardsV2" in only_body(service)


def test_a_publisher_that_returns_nothing_falls_back_to_a_text_only_answer():
    ops, service = make_ops(
        publisher=MagicMock(return_value=[]), doc_publisher=MagicMock(return_value=[])
    )
    ops.post_answer(ENTRY, completed("answer", [doc_descriptor("d1")]))

    body = only_body(service)
    assert "cardsV2" not in body
    assert body["text"] == "answer"


def test_a_publisher_that_raises_still_delivers_the_answer_text(caplog):
    def boom(*args: Any, **kwargs: Any) -> list[str]:
        raise RuntimeError("gcs down")

    ops, service = make_ops(publisher=boom, doc_publisher=boom)
    with caplog.at_level("WARNING"):
        ops.post_answer(ENTRY, completed("answer", [image_descriptor("a1")]))

    body = only_body(service)
    assert "cardsV2" not in body
    assert body["text"] == "answer"
    assert any("publisher raised" in record.getMessage() for record in caplog.records)


def test_a_card_create_failure_on_the_final_chunk_is_retried_as_text_only():
    publisher = MagicMock(return_value=[PLOT_URL])
    ops, service = make_ops(publisher=publisher)
    creates(service).return_value.execute.side_effect = [RuntimeError("chat api 500"), CREATED]

    ops.post_answer(ENTRY, completed("answer", [image_descriptor("a1")]))

    posted = bodies(service)
    assert len(posted) == 2
    assert "cardsV2" in posted[0]
    assert "cardsV2" not in posted[1]
    assert posted[1]["text"] == "answer"


def test_a_text_only_retry_that_also_fails_raises():
    # Both attempts gone means the answer did not land, and only a raise re-delivers it.
    publisher = MagicMock(return_value=[PLOT_URL])
    ops, service = make_ops(publisher=publisher)
    fail_every_create(service)

    with pytest.raises(RuntimeError):
        ops.post_answer(ENTRY, completed("answer", [image_descriptor("a1")]))


# --- post_answer: the declined-image re-offer -------------------------------


def test_an_image_the_png_publisher_declines_is_re_offered_as_a_document():
    # The fallback-conversion case: the descriptor claims PNG, the worker served
    # something else. The document path reads no mime off the descriptor, so the
    # artifact is published under what it really is instead of being dropped.
    descriptor = image_descriptor("a1", "orbit.png")
    publisher = MagicMock(return_value=[])
    doc_publisher = MagicMock(return_value=[published_document(DOC_URL, "orbit.png")])
    ops, service = make_ops(publisher=publisher, doc_publisher=doc_publisher)
    ops.post_answer(ENTRY, completed("here it is", [descriptor]))

    doc_publisher.assert_called_once_with(CFG, RUN_ID, [descriptor])
    cards = only_body(service)["cardsV2"]
    assert len(cards) == 1
    buttons = cards[0]["card"]["sections"][0]["widgets"][0]["buttonList"]["buttons"]
    assert buttons == [{"text": "📄 Open orbit.png", "onClick": {"openLink": {"url": DOC_URL}}}]


def test_a_declined_image_never_appears_in_the_image_card():
    published = image_descriptor("a1")
    declined = image_descriptor("a2")
    publisher = MagicMock(side_effect=[[PLOT_URL], []])
    doc_publisher = MagicMock(return_value=[published_document(DOC_URL, "a2.png")])
    ops, service = make_ops(publisher=publisher, doc_publisher=doc_publisher)
    ops.post_answer(ENTRY, completed("two plots, one fallback", [published, declined]))

    cards = only_body(service)["cardsV2"]
    image_widgets = cards[0]["card"]["sections"][0]["widgets"]
    assert image_widgets == [
        {"image": {"imageUrl": PLOT_URL, "onClick": {"openLink": {"url": PLOT_URL}}}}
    ]
    doc_publisher.assert_called_once_with(CFG, RUN_ID, [declined])


def test_a_re_offered_image_follows_the_artifacts_that_were_documents_already():
    declined = image_descriptor("a1")
    document = doc_descriptor("d1")
    publisher = MagicMock(return_value=[])
    doc_publisher = MagicMock(
        side_effect=[
            [published_document(DOC_URL, "report.md")],
            [published_document(DOC_URL, "a1.png")],
        ]
    )
    ops, _ = make_ops(publisher=publisher, doc_publisher=doc_publisher)
    ops.post_answer(ENTRY, completed("both", [declined, document]))

    assert doc_publisher.call_args_list == [
        call(CFG, RUN_ID, [document]),
        call(CFG, RUN_ID, [declined]),
    ]


def test_an_image_publisher_that_raises_still_re_offers_the_artifact():
    # A raise is indistinguishable from a decline from here, and the cost of re-offering
    # an artifact that is genuinely unfetchable is one more failed fetch.
    descriptor = image_descriptor("a1")

    def boom(*args: Any, **kwargs: Any) -> list[str]:
        raise RuntimeError("gcs down")

    doc_publisher = MagicMock(return_value=[published_document()])
    ops, _ = make_ops(publisher=boom, doc_publisher=doc_publisher)
    ops.post_answer(ENTRY, completed("answer", [descriptor]))

    doc_publisher.assert_called_once_with(CFG, RUN_ID, [descriptor])


# --- post_answer: the published-URL map -------------------------------------
#
# Read off the write seam directly (``_stashed_urls``): the member that consumes the map
# is not on the class yet, and this is the state it will consume.


def test_published_image_urls_are_stashed_under_the_run_id():
    publisher = MagicMock(side_effect=[[PLOT_URL], [OTHER_PLOT_URL]])
    ops, _ = make_ops(publisher=publisher)
    ops.post_answer(ENTRY, completed("two plots", [image_descriptor("a1"), image_descriptor("a2")]))

    assert ops._stashed_urls == {RUN_ID: {"a1": PLOT_URL, "a2": OTHER_PLOT_URL}}


def test_a_document_is_stashed_under_the_id_of_the_descriptor_it_was_published_from():
    # The published result carries no artifact id of its own, so the id comes from the
    # input descriptor of the very call that produced it.
    doc_publisher = MagicMock(return_value=[published_document()])
    ops, _ = make_ops(doc_publisher=doc_publisher)
    ops.post_answer(ENTRY, completed("here is the report", [doc_descriptor("d1")]))

    assert ops._stashed_urls == {RUN_ID: {"d1": DOC_URL}}


def test_a_re_offered_image_is_stashed_under_its_document_url():
    publisher = MagicMock(return_value=[])
    doc_publisher = MagicMock(return_value=[published_document(DOC_URL, "orbit.png")])
    ops, _ = make_ops(publisher=publisher, doc_publisher=doc_publisher)
    ops.post_answer(ENTRY, completed("answer", [image_descriptor("a1", "orbit.png")]))

    assert ops._stashed_urls == {RUN_ID: {"a1": DOC_URL}}


def test_one_artifact_failing_never_costs_its_neighbour_its_url():
    publisher = MagicMock(side_effect=[[PLOT_URL], []])
    ops, _ = make_ops(publisher=publisher, doc_publisher=MagicMock(return_value=[]))
    ops.post_answer(ENTRY, completed("two plots", [image_descriptor("a1"), image_descriptor("a2")]))

    assert ops._stashed_urls == {RUN_ID: {"a1": PLOT_URL}}


def test_an_image_publisher_answering_more_urls_than_it_was_offered_maps_nothing():
    # The counts-match rule: one artifact in, one URL out, or no correspondence at all.
    # No URL beats a wrong one — a descriptor with no gcs_url simply re-fetches later,
    # while a wrong one points the agent at somebody else's plot.
    publisher = MagicMock(return_value=[PLOT_URL, OTHER_PLOT_URL])
    ops, _ = make_ops(publisher=publisher, doc_publisher=MagicMock(return_value=[]))
    ops.post_answer(ENTRY, completed("one plot", [image_descriptor("a1")]))

    assert ops._stashed_urls == {}


def test_a_document_publisher_answering_more_results_than_it_was_offered_maps_nothing():
    doc_publisher = MagicMock(return_value=[published_document(), published_document()])
    ops, service = make_ops(doc_publisher=doc_publisher)
    ops.post_answer(ENTRY, completed("one file", [doc_descriptor("d1")]))

    assert ops._stashed_urls == {}
    assert "cardsV2" not in only_body(service)


def test_a_text_only_answer_stashes_nothing():
    ops, _ = make_ops()
    ops.post_answer(ENTRY, completed("no artifacts"))

    assert ops._stashed_urls == {}


def test_a_run_that_published_nothing_stashes_nothing():
    ops, _ = make_ops(
        publisher=MagicMock(return_value=[]), doc_publisher=MagicMock(return_value=[])
    )
    ops.post_answer(ENTRY, completed("answer", [image_descriptor("a1")]))

    assert ops._stashed_urls == {}


# --- post_queued / post_giveup / post_superseded ----------------------------


def test_the_queued_notice_uses_the_fixed_wording_in_the_questions_thread():
    ops, service = make_ops()
    ops.post_queued(ENTRY, {"status": "error", "error": "infra down"})

    body = only_body(service)
    assert body["text"] == QUEUED_TEXT
    assert body["thread"] == {"name": THREAD}
    assert "infra down" not in body["text"]


def test_the_queued_wording_claims_only_what_the_code_guarantees():
    # Queued plus an automatic retry. Nothing is filed to send a notification, and the
    # user is not asked to re-send — either claim would be a promise the bridge cannot
    # keep.
    assert "queued" in QUEUED_TEXT
    assert "automatically" in QUEUED_TEXT
    assert "notified" not in QUEUED_TEXT
    assert "re-send" not in QUEUED_TEXT


def test_a_failing_queued_notice_propagates_to_the_engine():
    # Contract-legal: the engine parked the entry before calling, and logs the raise.
    ops, service = make_ops()
    fail_every_create(service)

    with pytest.raises(RuntimeError):
        ops.post_queued(ENTRY, {"status": "error"})


def test_the_give_up_notice_uses_the_fixed_wording_in_the_questions_thread():
    ops, service = make_ops()
    ops.post_giveup({**ENTRY, "give_up_reason": "ceiling reached"})

    body = only_body(service)
    assert body["text"] == GIVEUP_TEXT
    assert body["thread"] == {"name": THREAD}
    assert "ceiling" not in body["text"]


def test_the_give_up_wording_asks_the_user_to_re_send():
    # The only actionable thing left: nothing will retry this on their behalf.
    assert "re-send your question" in GIVEUP_TEXT


def test_a_failing_give_up_notice_raises_so_the_entry_stays_queued():
    # Silence is worse than a duplicate notice: a user never told their question was
    # abandoned waits for an answer that will never come.
    ops, service = make_ops()
    fail_every_create(service)

    with pytest.raises(RuntimeError):
        ops.post_giveup(ENTRY)


def test_the_superseded_note_lands_in_the_old_messages_own_thread():
    ops, service = make_ops()
    ops.post_superseded(ENTRY)

    body = only_body(service)
    assert body["text"] == SUPERSEDED_TEXT
    assert body["thread"] == {"name": THREAD}


def test_a_failing_superseded_note_is_swallowed():
    # The coalesce CAS has already committed; a failed note must not disturb the pass.
    ops, service = make_ops()
    fail_every_create(service)

    ops.post_superseded(ENTRY)


@pytest.mark.parametrize(
    "wording", [ACK_TEXT, EMPTY_ANSWER_TEXT, ERROR_TEXT, QUEUED_TEXT, GIVEUP_TEXT, SUPERSEDED_TEXT]
)
def test_no_space_facing_wording_is_empty(wording):
    # Chat rejects an empty body outright, so every constant that can become one has to
    # carry text.
    assert wording.strip()
