"""``deliver_files`` and the run-keyed stash it pops.

Chat's artifacts ride the answer's own ``cardsV2``, so this member posts nothing: it hands
the engine the ``{artifact_id: public URL}`` map ``post_answer`` produced, which the engine
stamps into history as ``public_url``. That makes the stash the whole story, and the
properties worth pinning are the ones a live bridge breaks quietly — the map must survive
the gap between the two calls, it must be gone after one read, and a run whose delivery
never arrives must not be able to grow it.

The publishers are faked, so nothing here touches GCS, the worker, or any Google library.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from osprey.bridges.google_chat.client import ChatClient
from osprey.bridges.google_chat.config import GoogleChatBridgeConfig
from osprey.bridges.google_chat.events import GC_SPACE, GC_THREAD
from osprey.bridges.google_chat.ops import STASH_LIMIT, GoogleChatOps

CFG = GoogleChatBridgeConfig(
    sa_key="/etc/osprey/sa-key.json",
    subscription="projects/p/subscriptions/s",
    app_id="users/1234567890",
    gcs_bucket="test-bucket",
)

SPACE = "spaces/AAAA"
THREAD = "spaces/AAAA/threads/TTTT"
RUN_ID = "R1"
ENTRY = {GC_SPACE: SPACE, GC_THREAD: THREAD}

PLOT_URL = "https://storage.googleapis.com/test-bucket/plots/1.png"
OTHER_PLOT_URL = "https://storage.googleapis.com/test-bucket/plots/2.png"


def make_ops(*, publisher: Any = None) -> tuple[GoogleChatOps, MagicMock]:
    """An ops instance over a mock Chat service, with the service returned alongside so a
    test can read what was posted. The image publisher answers with one URL per call
    unless a test injects its own; the document publisher publishes nothing.
    """
    service = MagicMock()
    service.spaces.return_value.messages.return_value.create.return_value.execute.return_value = {
        "name": "spaces/AAAA/messages/MMMM"
    }
    ops = GoogleChatOps(
        CFG,
        ChatClient(CFG, service),
        publisher if publisher is not None else MagicMock(return_value=[PLOT_URL]),
        MagicMock(return_value=[]),
    )
    return ops, service


def creates(service: MagicMock) -> Any:
    """The ``messages.create`` mock, whose call count is the posted-message record."""
    return service.spaces.return_value.messages.return_value.create


def image_descriptor(artifact_id: str) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "filename": f"{artifact_id}.png",
        "source_mime": "image/png",
        "delivered_mime": "image/png",
        "convertible": True,
    }


def completed(run_id: str = RUN_ID, artifacts: list[Any] | None = None) -> dict[str, Any]:
    return {
        "status": "completed",
        "text_output": "42 mA",
        "run_id": run_id,
        "error": None,
        "artifacts": artifacts if artifacts is not None else [],
    }


def answered(ops: GoogleChatOps, run_id: str = RUN_ID) -> dict[str, Any]:
    """Post one answer carrying a single image artifact, and return its result."""
    result = completed(run_id, [image_descriptor("a1")])
    ops.post_answer(ENTRY, result)
    return result


def abandon(ops: GoogleChatOps, service: MagicMock, run_id: str) -> None:
    """Publish a run's artifacts and then fail its post, exactly as a run the engine gives
    up on: the stash was written, and ``deliver_files`` will never be called for it."""
    creates(service).return_value.execute.side_effect = RuntimeError("chat api 500")
    with pytest.raises(RuntimeError):
        ops.post_answer(ENTRY, completed(run_id, [image_descriptor("a1")]))


# --- what deliver_files returns ---------------------------------------------


def test_the_map_post_answer_published_is_what_deliver_files_returns():
    ops, _ = make_ops()
    result = answered(ops)

    assert ops.deliver_files(ENTRY, result) == {"a1": PLOT_URL}


def test_deliver_files_posts_nothing_because_the_artifacts_rode_the_answer():
    # The engine calls this AFTER post_answer succeeded. A message posted here would be a
    # second, artifact-only message in the space — the card cannot be attached to the
    # answer that already landed, which is why the publish happens in post_answer.
    ops, service = make_ops()
    result = answered(ops)
    posted_by_the_answer = creates(service).call_count

    ops.deliver_files(ENTRY, result)

    assert creates(service).call_count == posted_by_the_answer


def test_a_text_only_answer_delivers_an_empty_map():
    ops, _ = make_ops()
    result = completed()
    ops.post_answer(ENTRY, result)

    assert ops.deliver_files(ENTRY, result) == {}


def test_a_run_whose_publish_failed_entirely_delivers_an_empty_map():
    # Nothing was republished, so no descriptor may be stamped with a public_url: the
    # engine's later re-injection would fetch a URL that never existed.
    ops, _ = make_ops(publisher=MagicMock(return_value=[]))
    result = answered(ops)

    assert ops.deliver_files(ENTRY, result) == {}


def test_a_delivery_for_an_unknown_run_delivers_an_empty_map():
    ops, _ = make_ops()
    answered(ops)

    assert ops.deliver_files(ENTRY, completed("some-other-run")) == {}


def test_a_result_with_no_run_id_falls_back_to_the_one_the_entry_carries():
    # The drain re-attaches a queued delivery off the persisted entry, where the run id
    # was stamped at claim time; its result need not carry one. post_answer resolves the
    # run id the same way, so both halves must agree or the map is stranded.
    ops, _ = make_ops()
    entry = {**ENTRY, "run_id": RUN_ID}
    result = {**completed(), "run_id": None, "artifacts": [image_descriptor("a1")]}
    ops.post_answer(entry, result)

    assert ops.deliver_files(entry, result) == {"a1": PLOT_URL}


def test_a_delivery_that_can_name_no_run_at_all_returns_an_empty_map():
    ops, _ = make_ops()
    answered(ops)

    assert ops.deliver_files({}, {**completed(), "run_id": None}) == {}


def test_a_malformed_run_id_returns_an_empty_map_rather_than_raising():
    ops, _ = make_ops()

    assert ops.deliver_files({}, {"run_id": 17}) == {}


# --- the stash is popped, not read ------------------------------------------


def test_a_delivered_run_leaves_nothing_behind_in_the_stash():
    ops, _ = make_ops()
    result = answered(ops)
    ops.deliver_files(ENTRY, result)

    assert ops._stashed_urls == {}


def test_a_second_delivery_of_the_same_run_returns_an_empty_map():
    # At-least-once redelivery is normal. The second call must not re-stamp URLs onto a
    # turn the first one already stamped.
    ops, _ = make_ops()
    result = answered(ops)
    ops.deliver_files(ENTRY, result)

    assert ops.deliver_files(ENTRY, result) == {}


def test_delivering_one_run_leaves_another_runs_map_untouched():
    ops, _ = make_ops()
    first = answered(ops, "R1")
    second = answered(ops, "R2")

    assert ops.deliver_files(ENTRY, first) == {"a1": PLOT_URL}
    assert ops.deliver_files(ENTRY, second) == {"a1": PLOT_URL}


# --- the bound ---------------------------------------------------------------


def test_a_run_that_is_given_up_on_cannot_grow_the_stash_without_limit():
    # The leak the bound exists for: post_answer publishes the artifacts and THEN fails to
    # post, so the engine never reaches deliver_files and the map is never popped. A
    # bridge that answers for months must not accumulate one map per such run.
    ops, service = make_ops()
    for index in range(STASH_LIMIT * 4):
        abandon(ops, service, f"abandoned-{index}")

    assert len(ops._stashed_urls) == STASH_LIMIT


def test_the_oldest_map_is_the_one_evicted():
    ops, _ = make_ops()
    for index in range(STASH_LIMIT + 1):
        ops.post_answer(ENTRY, completed(f"R{index}", [image_descriptor("a1")]))

    assert set(ops._stashed_urls) == {f"R{i}" for i in range(1, STASH_LIMIT + 1)}


def test_an_evicted_runs_delivery_degrades_to_an_empty_map():
    # Eviction costs the public_url stamps of a run nobody ever delivered — never an
    # exception, and never somebody else's URLs.
    ops, _ = make_ops()
    evicted = completed("R0", [image_descriptor("a1")])
    ops.post_answer(ENTRY, evicted)
    for index in range(1, STASH_LIMIT + 1):
        ops.post_answer(ENTRY, completed(f"R{index}", [image_descriptor("a1")]))

    assert ops.deliver_files(ENTRY, evicted) == {}


def test_re_answering_a_stashed_run_refreshes_its_place_in_the_queue():
    # A redelivery re-publishes under new object names. The re-stashed run must take the
    # newest position rather than keeping the one its first write had, or the very run
    # being retried is the one thrown away when the next run arrives.
    ops, _ = make_ops()
    for index in range(STASH_LIMIT):
        ops.post_answer(ENTRY, completed(f"R{index}", [image_descriptor("a1")]))
    ops.post_answer(ENTRY, completed("R0", [image_descriptor("a1")]))  # the redelivery

    ops.post_answer(ENTRY, completed("newest", [image_descriptor("a1")]))

    assert "R1" not in ops._stashed_urls  # the oldest write still held
    assert ops.deliver_files(ENTRY, completed("R0")) == {"a1": PLOT_URL}


def test_a_re_answered_run_stashes_the_urls_of_the_newer_publish():
    ops, _ = make_ops(publisher=MagicMock(side_effect=[[PLOT_URL], [OTHER_PLOT_URL]]))
    result = completed("R0", [image_descriptor("a1")])
    ops.post_answer(ENTRY, result)
    ops.post_answer(ENTRY, result)

    assert ops.deliver_files(ENTRY, result) == {"a1": OTHER_PLOT_URL}


# --- concurrency -------------------------------------------------------------


def test_answers_and_deliveries_racing_on_two_threads_lose_no_map():
    # The engine shares ONE ops instance across the subscriber thread and the drain
    # thread, so the stash is written and popped concurrently. Every run posted here is
    # delivered exactly once, and nothing is left behind — no map is lost to a racing
    # eviction, and none is stranded.
    ops, _ = make_ops()
    runs = [f"R{index}" for index in range(STASH_LIMIT // 2)]
    delivered: list[dict[str, str]] = []
    lock = threading.Lock()
    start = threading.Barrier(2)

    def deliver(subset: list[str]) -> None:
        start.wait(timeout=5)
        for run_id in subset:
            result = completed(run_id, [image_descriptor("a1")])
            ops.post_answer(ENTRY, result)
            delivery = dict(ops.deliver_files(ENTRY, result))
            with lock:
                delivered.append(delivery)

    threads = [
        threading.Thread(target=deliver, args=(runs[::2],)),
        threading.Thread(target=deliver, args=(runs[1::2],)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert delivered == [{"a1": PLOT_URL}] * len(runs)
    assert ops._stashed_urls == {}
