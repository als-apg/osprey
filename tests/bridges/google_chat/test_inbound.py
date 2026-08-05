"""Inbound Chat attachment download: the allowlist, the caps, and the skip contract.

Hermetic: the media GET runs through an ``httpx.Client`` over a ``MockTransport``
(the seam the Talk download tests use), so there is no network, no credential, and
no dependency on ``google-auth`` being installed. Attachment references are the
JSON-plain dicts ``events.parse_attachments`` builds, which is exactly what
``ChannelOps.download_inputs`` will read back off the persisted entry.

Two things are deliberately NOT tested here because this module does not do them:
the per-message file-count and total-bytes budget, and the ``input_files``
capability probe. Both belong to the engine (``osprey.bridges.core.pipeline``),
which applies one shared budget across the own and quoted buckets — and the two
tests under "the engine owns the budget" pin that this module leaves them alone.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator, Mapping
from typing import Any

import httpx
import pytest

from osprey.bridges.core.pipeline import IMAGE_MIMES as ENGINE_REPLAY_MIMES
from osprey.bridges.core.pipeline import MAX_FILES, MAX_TOTAL_BYTES
from osprey.bridges.google_chat.inbound import (
    DOWNLOAD_FAILED_REASON,
    DRIVE_SKIP_REASON,
    IMAGE_MAX_BYTES,
    IMAGE_MIMES,
    NO_REFERENCE_REASON,
    TEXT_MAX_BYTES,
    TEXT_MIMES,
    UNPROCESSABLE_REASON,
    download_attachments,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels"


def ref(**overrides: Any) -> dict[str, Any]:
    """One attachment reference, shaped as ``events.parse_attachments`` builds it."""
    return {
        "content_name": "plot.png",
        "content_type": "image/png",
        "resource_name": "attachments/AAA",
        "source": "UPLOADED_CONTENT",
        **overrides,
    }


def session_for(by_resource: Mapping[str, bytes], status: int = 404) -> httpx.Client:
    """A session serving ``resource_name -> bytes``; anything else gets ``status``.

    The handler asserts the request shape the Chat media API requires, so every test
    that downloads anything also pins ``?alt=media`` and the resource path.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("alt") == "media"
        resource = str(request.url.path).split("/v1/media/", 1)[-1]
        if resource not in by_resource:
            return httpx.Response(status)
        return httpx.Response(200, content=by_resource[resource])

    return httpx.Client(transport=httpx.MockTransport(handler))


def counting_session(content: bytes = PNG) -> tuple[httpx.Client, list[httpx.Request]]:
    """``(session, requests)`` — a session that records every request it is asked for.

    Serves ``content`` for anything, so a test that expects NO round trip fails loudly
    (an extra file in the result) rather than quietly on a 404 that looks like a skip.
    """
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=content)

    return httpx.Client(transport=httpx.MockTransport(handler)), requests


def decoded(payload_file: Mapping[str, Any]) -> bytes:
    """The bytes a worker-payload file actually carries."""
    content = payload_file["content_b64"]
    assert isinstance(content, str)
    return base64.b64decode(content)


# --- the happy path ---------------------------------------------------------


def test_an_uploaded_image_is_shaped_for_the_worker_payload():
    files, skipped = download_attachments(session_for({"attachments/AAA": PNG}), [ref()])

    assert skipped == []
    assert files == [
        {
            "filename": "plot.png",
            "mime": "image/png",
            "content_b64": base64.b64encode(PNG).decode("ascii"),
            "ingest": True,
        }
    ]


def test_the_media_get_addresses_the_named_resource_and_asks_for_raw_bytes():
    session, requests = counting_session()

    download_attachments(session, [ref(resource_name="attachments/XYZ/123")])

    assert len(requests) == 1
    assert str(requests[0].url.path) == "/v1/media/attachments/XYZ/123"
    assert requests[0].url.params.get("alt") == "media"
    assert requests[0].url.host == "chat.googleapis.com"


@pytest.mark.parametrize("mime", sorted(IMAGE_MIMES | TEXT_MIMES))
def test_every_allowed_type_is_downloaded(mime):
    files, skipped = download_attachments(
        session_for({"r0": PNG}), [ref(content_type=mime, resource_name="r0")]
    )

    assert skipped == []
    assert [f["mime"] for f in files] == [mime]


def test_several_attachments_keep_the_order_chat_listed_them():
    refs = [
        ref(content_name="data.csv", content_type="text/csv", resource_name="r1"),
        ref(content_name="notes.md", content_type="text/markdown", resource_name="r2"),
        ref(content_name="x.json", content_type="application/json", resource_name="r3"),
    ]
    session = session_for({"r1": b"a,b", "r2": b"# hi", "r3": b"{}"})

    files, skipped = download_attachments(session, refs)

    assert skipped == []
    assert [f["filename"] for f in files] == ["data.csv", "notes.md", "x.json"]
    assert [decoded(f) for f in files] == [b"a,b", b"# hi", b"{}"]


# --- the allowlist ----------------------------------------------------------


@pytest.mark.parametrize(
    "mime", ["application/pdf", "application/zip", "image/svg+xml", "video/mp4", "", None]
)
def test_a_type_off_the_allowlist_is_skipped_without_a_round_trip(mime):
    session, requests = counting_session(b"%PDF")

    files, skipped = download_attachments(session, [ref(content_name="doc", content_type=mime)])

    assert files == []
    assert requests == []
    assert skipped == [{"filename": "doc", "reason": f"unsupported type {mime or 'unknown'}"}]


def test_the_image_allowlist_is_the_set_the_engine_can_replay():
    """An image the engine would never re-inject must not be ingestible in the first
    place, or the agent could see it this turn and silently not the next."""
    assert IMAGE_MIMES == ENGINE_REPLAY_MIMES


def test_the_shipped_caps_are_five_mib_per_image_and_one_hundred_kib_per_text_file():
    assert IMAGE_MAX_BYTES == 5 * 1024 * 1024
    assert TEXT_MAX_BYTES == 100 * 1024


def test_the_inbound_caps_match_the_workers_own_per_file_caps():
    """Accepting a file the worker will reject turns a clean "too large" skip note into
    a fatal 400 mid-dispatch; accepting less than it would makes the bridge needlessly
    the stricter party. Pinned rather than imported, like the allowlist above."""
    from osprey.mcp_server.dispatch_worker.input_files_policy import (
        MAX_IMAGE_BYTES,
        MAX_TEXT_BYTES,
    )

    assert IMAGE_MAX_BYTES == MAX_IMAGE_BYTES
    assert TEXT_MAX_BYTES == MAX_TEXT_BYTES


# --- the per-type caps ------------------------------------------------------


def test_an_image_over_the_image_cap_is_skipped_naming_the_cap():
    oversize = b"x" * (IMAGE_MAX_BYTES + 1)

    files, skipped = download_attachments(session_for({"attachments/AAA": oversize}), [ref()])

    assert files == []
    assert len(skipped) == 1
    assert skipped[0]["reason"].startswith("too large (")
    assert "5MB limit" in skipped[0]["reason"]


def test_a_text_file_over_the_text_cap_is_skipped_naming_the_cap():
    oversize = b"x" * (TEXT_MAX_BYTES + 1)
    attachment = ref(content_name="big.csv", content_type="text/csv", resource_name="r1")

    files, skipped = download_attachments(session_for({"r1": oversize}), [attachment])

    assert files == []
    assert "100KB limit" in skipped[0]["reason"]


def test_a_text_file_is_held_to_the_text_cap_not_the_image_cap():
    """The caps are per TYPE: 200KB is fine for an image and far too big for a CSV."""
    payload = b"x" * (2 * TEXT_MAX_BYTES)
    session = session_for({"r1": payload, "r2": payload})

    text_files, text_skipped = download_attachments(
        session, [ref(content_name="a.csv", content_type="text/csv", resource_name="r1")]
    )
    image_files, image_skipped = download_attachments(
        session, [ref(content_name="a.png", content_type="image/png", resource_name="r2")]
    )

    assert text_files == [] and len(text_skipped) == 1
    assert image_skipped == [] and len(image_files) == 1


def test_a_file_exactly_at_its_cap_is_kept(monkeypatch):
    """The guard is ``>``, not ``>=`` — pin the boundary."""
    monkeypatch.setattr("osprey.bridges.google_chat.inbound.IMAGE_MAX_BYTES", 64)
    session = session_for({"r1": b"x" * 64, "r2": b"x" * 65})

    at_cap, at_cap_skipped = download_attachments(session, [ref(resource_name="r1")])
    over_cap, over_cap_skipped = download_attachments(session, [ref(resource_name="r2")])

    assert at_cap_skipped == [] and len(at_cap) == 1
    assert over_cap == [] and len(over_cap_skipped) == 1


# --- Drive files ------------------------------------------------------------


@pytest.mark.parametrize("source", ["DRIVE_FILE", "GOOGLE_DRIVE", "drive_file"])
def test_a_drive_backed_reference_is_skipped_without_a_round_trip(source):
    session, requests = counting_session()
    attachment = ref(content_name="shared.png", source=source, resource_name="")

    files, skipped = download_attachments(session, [attachment])

    assert files == []
    assert requests == []
    assert skipped == [{"filename": "shared.png", "reason": DRIVE_SKIP_REASON}]


# --- failures degrade to one skip note each ---------------------------------


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500])
def test_any_error_status_becomes_a_skip_note(status):
    files, skipped = download_attachments(session_for({}, status=status), [ref()])

    assert files == []
    assert skipped == [{"filename": "plot.png", "reason": DOWNLOAD_FAILED_REASON}]


def test_a_transport_failure_becomes_a_skip_note():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("chat.googleapis.com unreachable")

    session = httpx.Client(transport=httpx.MockTransport(handler))

    files, skipped = download_attachments(session, [ref()])

    assert files == []
    assert skipped == [{"filename": "plot.png", "reason": DOWNLOAD_FAILED_REASON}]


def test_a_reference_without_a_resource_name_is_skipped_without_a_round_trip():
    session, requests = counting_session()

    files, skipped = download_attachments(session, [ref(content_name="a.png", resource_name="")])

    assert files == []
    assert requests == []
    assert skipped == [{"filename": "a.png", "reason": NO_REFERENCE_REASON}]


def test_a_reference_that_cannot_even_be_read_costs_only_itself():
    """The never-raises backstop: an exploding reference is one skip note, not a
    lost dispatch — and its siblings are still delivered."""

    class ExplodingRef(Mapping):
        def __getitem__(self, key: str) -> Any:
            raise RuntimeError("unreadable reference")

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("unreadable reference")

    session = session_for({"good": PNG})

    files, skipped = download_attachments(
        session, [ExplodingRef(), ref(content_name="ok.png", resource_name="good")]
    )

    assert [f["filename"] for f in files] == ["ok.png"]
    assert skipped == [{"filename": "attachment", "reason": UNPROCESSABLE_REASON}]


def test_one_failing_attachment_does_not_cost_its_siblings():
    refs = [
        ref(content_name="ok.png", resource_name="good"),
        ref(content_name="gone.png", resource_name="missing"),
        ref(content_name="bad.pdf", content_type="application/pdf", resource_name="also-good"),
        ref(content_name="also-ok.csv", content_type="text/csv", resource_name="csv"),
    ]
    session = session_for({"good": PNG, "also-good": b"%PDF", "csv": b"a,b"})

    files, skipped = download_attachments(session, refs)

    assert [f["filename"] for f in files] == ["ok.png", "also-ok.csv"]
    assert [s["filename"] for s in skipped] == ["gone.png", "bad.pdf"]


def test_every_reference_is_accounted_for_exactly_once():
    refs = [
        ref(content_name="ok.png", resource_name="good"),
        ref(content_name="gone.png", resource_name="missing"),
        ref(content_name="drive.png", source="DRIVE_FILE"),
        ref(content_name="bad.pdf", content_type="application/pdf"),
        ref(content_name="unreferenced.png", resource_name=""),
    ]

    files, skipped = download_attachments(session_for({"good": PNG}), refs)

    assert len(files) + len(skipped) == len(refs)
    assert {f["filename"] for f in files} | {s["filename"] for s in skipped} == {
        "ok.png",
        "gone.png",
        "drive.png",
        "bad.pdf",
        "unreferenced.png",
    }


# --- filenames --------------------------------------------------------------


def test_a_filename_carrying_crlf_is_sanitized():
    files, _ = download_attachments(
        session_for({"attachments/AAA": PNG}), [ref(content_name="a\r\nb.png")]
    )

    assert files[0]["filename"] == "ab.png"


@pytest.mark.parametrize("name", [None, "", "   ", 17])
def test_an_unusable_filename_falls_back(name):
    files, _ = download_attachments(session_for({"attachments/AAA": PNG}), [ref(content_name=name)])

    assert files[0]["filename"] == "attachment"


# --- nothing to do ----------------------------------------------------------


@pytest.mark.parametrize("refs", [None, [], ()])
def test_no_references_is_a_strict_no_op(refs):
    session, requests = counting_session()

    assert download_attachments(session, refs) == ([], [])
    assert requests == []


def test_a_non_iterable_reference_list_is_a_strict_no_op():
    """A caller bug, but it may not raise: a dispatch is worth more than a batch."""
    session, requests = counting_session()

    assert download_attachments(session, 17) == ([], [])
    assert requests == []


@pytest.mark.parametrize("junk", ["a string", 17, None, ["nested"]])
def test_a_non_mapping_entry_is_dropped(junk):
    session = session_for({"good": PNG})

    files, skipped = download_attachments(
        session, [junk, ref(content_name="ok.png", resource_name="good")]
    )

    assert [f["filename"] for f in files] == ["ok.png"]
    assert skipped == []


# --- the engine owns the budget ---------------------------------------------


def test_no_file_count_budget_is_applied_here():
    """``MAX_FILES`` is the ENGINE's, applied across the own and quoted buckets
    together; trimming here as well would double-count it."""
    count = MAX_FILES + 3
    refs = [ref(content_name=f"f{i}.png", resource_name=f"r{i}") for i in range(count)]

    files, skipped = download_attachments(session_for({f"r{i}": PNG for i in range(count)}), refs)

    assert len(files) == count
    assert skipped == []


def test_no_total_bytes_budget_is_applied_here():
    """Same for ``MAX_TOTAL_BYTES``: three 4MiB images are each within the image cap
    and together past the engine's total, and all three come back."""
    each = 4 * 1024 * 1024
    payload = b"x" * each
    assert 3 * each > MAX_TOTAL_BYTES
    refs = [ref(content_name=f"f{i}.png", resource_name=f"r{i}") for i in range(3)]

    files, skipped = download_attachments(session_for({f"r{i}": payload for i in range(3)}), refs)

    assert len(files) == 3
    assert skipped == []
    assert sum(len(decoded(f)) for f in files) > MAX_TOTAL_BYTES
