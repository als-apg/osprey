"""What the Teams adapter does with a run's artifacts.

Every artifact meets one of three outcomes. A PNG that fits the inline budget is
posted as a message activity carrying the image inline as a ``data:`` URL. Every
other artifact — a document, or an image too large or too broken to inline — is
uploaded into the file library the deployment names, shared with the
conversation's members, and announced by one card per file. Whatever reaches
neither is named in one closing note. This suite asserts the activity bodies the
adapter handed the connector, the bytes inside them, and the calls it made on a
recording file library. The worker is replaced by a fetcher stub, which keeps
both the byte route and the run engine out of the test path entirely.

Two properties are pinned harder than the rest because a live bridge breaks
them quietly. The first is the downscale: an image that reaches Teams over the
inline budget does not render smaller, it fails to render at all, so the box fit
is asserted by decoding the posted bytes rather than by trusting that Pillow was
called. The second is the note: an image the bridge drops must be *named* to the
user, because the alternative — an answer that mentions a plot nobody can see —
is indistinguishable from a bridge that is broken.

``PIL`` is imported plainly here. Pillow ships in the dev extra, so a machine
without it fails these tests rather than skipping them; the one test that proves
the adapter survives its absence blocks the import itself, through the
``no_pillow`` fixture in ``conftest``.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import os
from collections.abc import Sequence
from typing import Any

import pytest
from PIL import Image

from osprey.bridges.core import FetchedArtifact
from osprey.bridges.teams.client import TokenError
from osprey.bridges.teams.events import MS_SERVICE_URL
from osprey.bridges.teams.graph import GraphError, UploadedFile
from osprey.bridges.teams.ops import (
    ATTACHMENT_CONTENT_TYPE,
    DATA_URL_PREFIX,
    FILE_BUTTON_TEXT,
    FILE_CARD_CONTENT_TYPE,
    IMAGE_BOX_PX,
    MAX_ATTACHMENT_BYTES,
    TeamsOps,
    skipped_files_note,
)
from tests.bridges.teams.test_posting import (
    ACTIVITY_ID,
    APP_ID,
    CHANNEL_CONVERSATION_ID,
    SERVICE_URL,
    TENANT,
    make_config,
)
from tests.bridges.teams.test_posting import (
    RecordingConnector as _PostingConnector,
)
from tests.bridges.teams.test_posting import make_entry as _posting_entry

RUN_ID = "run-7"


# --- fixtures of the world the member runs in --------------------------------


def make_entry(**overrides: Any) -> dict[str, Any]:
    """A persisted channel entry as the claim stamped it, with ``overrides`` applied."""
    return {**_posting_entry(text="Plot the beam current for the last hour."), **overrides}


class RecordingConnector(_PostingConnector):
    """The posting suite's recorder, plus views over the attachments it received."""

    @property
    def activities(self) -> list[dict[str, Any]]:
        return [activity for _, _, _, activity in self.calls]

    @property
    def attachments(self) -> list[dict[str, Any]]:
        """Every attachment posted, flattened across activities."""
        return [
            attachment
            for activity in self.activities
            for attachment in activity.get("attachments", [])
        ]


class RecordingFetcher:
    """A stand-in for ``fetch_artifact`` answering from a canned table."""

    def __init__(self, byte_map: dict[str, FetchedArtifact | None]) -> None:
        self.byte_map = byte_map
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self, _http: Any, _cfg: Any, run_id: str, artifact_id: str
    ) -> FetchedArtifact | None:
        self.calls.append((run_id, artifact_id))
        return self.byte_map.get(artifact_id)


DRIVE_ID = "b!library"
FOLDER = "osprey/answers"
WEB_ROOT = "https://tenant.sharepoint.com/sites/osprey/answers"

MEMBERS = [
    {"id": "29:alice", "name": "Alice", "aadObjectId": "oid-alice", "tenantId": TENANT},
    {"id": "29:bob", "name": "Bob", "aadObjectId": "oid-bob"},
    {"id": f"28:{APP_ID}", "name": "OSPREY", "aadObjectId": "oid-bot"},
]
"""A channel's listing: two people with directory ids, and the bot itself."""

AUDIENCE = ("oid-alice", "oid-bob")


class RecordingFiles:
    """A stand-in for :class:`~osprey.bridges.teams.graph.GraphFiles`.

    Records every upload and share. ``fail_upload`` names files whose upload
    raises (or holds an exception every upload raises); ``fail_share`` is raised
    by every share. Every upload of one folder answers the same folder id.
    """

    def __init__(
        self,
        *,
        fail_upload: set[str] | BaseException = frozenset(),  # type: ignore[assignment]
        fail_share: BaseException | None = None,
    ) -> None:
        self.fail_upload = fail_upload
        self.fail_share = fail_share
        self.uploads: list[tuple[tuple[str, ...], str, bytes, str | None]] = []
        self.shares: list[tuple[str, tuple[str, ...]]] = []

    def upload(
        self, folder: Sequence[str], name: str, data: bytes, content_type: str | None
    ) -> UploadedFile:
        self.uploads.append((tuple(folder), name, data, content_type))
        if isinstance(self.fail_upload, BaseException):
            raise self.fail_upload
        if name in self.fail_upload:
            raise GraphError(f"upload of {name} refused")
        path = "/".join(folder)
        return UploadedFile(
            name=name,
            item_id=f"item-{len(self.uploads)}",
            folder_id=f"folder:{path}",
            web_url=f"{WEB_ROOT}/{path}/{name}",
        )

    def share(self, item_id: str, object_ids: Sequence[str]) -> None:
        self.shares.append((item_id, tuple(object_ids)))
        if self.fail_share is not None:
            raise self.fail_share

    @property
    def names(self) -> list[str]:
        return [name for _, name, _, _ in self.uploads]


def make_ops(
    byte_map: dict[str, FetchedArtifact | None] | None = None,
    *,
    connector: RecordingConnector | None = None,
    fetcher: Any = None,
    files: RecordingFiles | None = None,
    members: list[dict[str, Any]] | BaseException | None = None,
) -> tuple[TeamsOps, RecordingConnector, Any]:
    """A ``TeamsOps`` over a recording connector and a canned artifact fetcher.

    With ``files`` the config names a library and the adapter uploads through that
    double; ``members`` is what the connector's member listing answers (the
    two-person :data:`MEMBERS` by default).
    """
    if connector is None:
        connector = RecordingConnector(members=MEMBERS if members is None else members)
    if fetcher is None:
        fetcher = RecordingFetcher(byte_map if byte_map is not None else {})
    cfg = make_config()
    if files is not None:
        cfg = dataclasses.replace(cfg, files_drive_id=DRIVE_ID, files_folder=FOLDER)
    ops = TeamsOps(cfg, connector, artifact_fetcher=fetcher, files=files)  # type: ignore[arg-type]
    return ops, connector, fetcher


def png_bytes(size: tuple[int, int], *, noise: bool = False) -> bytes:
    """A real PNG of ``size``, flat red or incompressible noise.

    Noise is how an image that is genuinely too big to inline is built without
    pinning a magic byte count: three random bytes per pixel do not deflate, so
    a full-box noise image lands several times over the attachment budget no
    matter what Pillow's encoder does with it.
    """
    if noise:
        image = Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3))
    else:
        image = Image.new("RGB", size, "red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


PNG_STUB = b"\x89PNG\r\n\x1a\nnot really a png"
"""PNG magic over bytes no decoder would accept.

Enough for every path that routes on what the bytes *are* without decoding them
— the document filter, and the whole delivery once Pillow is missing. The tests
that run with Pillow absent must not build a real PNG either: encoding one goes
through ``PIL.PngImagePlugin``, which the block makes unimportable."""


def fetched(data: bytes, content_type: str = "image/png") -> FetchedArtifact:
    return FetchedArtifact(data=data, content_type=content_type)


def descriptor(artifact_id: str, filename: str | None = None) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "filename": filename if filename is not None else f"{artifact_id}.png",
        "delivered_mime": "image/png",
    }


def completed(artifacts: list[Any] | None = None, run_id: str | None = RUN_ID) -> dict[str, Any]:
    return {
        "status": "completed",
        "text_output": "Here is the plot.",
        "run_id": run_id,
        "error": None,
        "artifacts": artifacts if artifacts is not None else [],
    }


def cards(connector: RecordingConnector) -> list[tuple[str, str]]:
    """``(name, url)`` of every file card posted, in order."""
    found = []
    for attachment in connector.attachments:
        if attachment["contentType"] != FILE_CARD_CONTENT_TYPE:
            continue
        content = attachment["content"]
        (block,) = content["body"]
        (action,) = content["actions"]
        assert action["type"] == "Action.OpenUrl"
        assert action["title"] == FILE_BUTTON_TEXT
        found.append((block["text"], action["url"]))
    return found


def run_folder_of(run_id: str = RUN_ID) -> tuple[str, ...]:
    return (*FOLDER.split("/"), run_id)


def posted_image(attachment: dict[str, Any]) -> Image.Image:
    """Decode the image a posted attachment carries."""
    assert attachment["contentUrl"].startswith(DATA_URL_PREFIX)
    raw = base64.b64decode(attachment["contentUrl"][len(DATA_URL_PREFIX) :])
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image


# --- the downscale -----------------------------------------------------------


def test_a_wide_png_is_fitted_into_the_box_keeping_its_aspect() -> None:
    """The canonical case: 1800x900 into a 1024x1024 box is 1024x512."""
    ops, connector, _ = make_ops({"a1": fetched(png_bytes((1800, 900)))})

    ops.deliver_files(make_entry(), completed([descriptor("a1")]))

    assert len(connector.attachments) == 1
    assert posted_image(connector.attachments[0]).size == (1024, 512)


def test_a_tall_png_is_fitted_on_its_long_side_too() -> None:
    # 750x3000 rather than a rounder pair: the height divides the box exactly,
    # so the expected width is the aspect and not a rounding rule.
    ops, connector, _ = make_ops({"a1": fetched(png_bytes((750, 3000)))})

    ops.deliver_files(make_entry(), completed([descriptor("a1")]))

    assert posted_image(connector.attachments[0]).size == (IMAGE_BOX_PX // 4, IMAGE_BOX_PX)


def test_an_image_already_inside_the_box_keeps_its_size() -> None:
    # The fit only ever shrinks: upscaling a small plot to fill the box would
    # post a blurred copy of something that already rendered correctly.
    ops, connector, _ = make_ops({"a1": fetched(png_bytes((320, 200)))})

    ops.deliver_files(make_entry(), completed([descriptor("a1")]))

    assert posted_image(connector.attachments[0]).size == (320, 200)


def test_every_artifact_gets_its_own_activity() -> None:
    # One attachment per activity, because Teams renders a multi-attachment
    # message as a carousel in some clients and as one image in others.
    ops, connector, _ = make_ops(
        {"a1": fetched(png_bytes((100, 100))), "a2": fetched(png_bytes((120, 90)))}
    )

    ops.deliver_files(make_entry(), completed([descriptor("a1"), descriptor("a2")]))

    assert [len(activity["attachments"]) for activity in connector.activities] == [1, 1]


# --- the activity shape ------------------------------------------------------


def test_an_image_rides_an_inline_data_url_on_the_png_content_type() -> None:
    ops, connector, _ = make_ops({"a1": fetched(png_bytes((100, 100)))})

    ops.deliver_files(make_entry(), completed([descriptor("a1", "current.png")]))

    attachment = connector.attachments[0]
    assert attachment["contentType"] == ATTACHMENT_CONTENT_TYPE
    assert attachment["contentUrl"].startswith(DATA_URL_PREFIX)
    assert attachment["name"] == "current.png"


def test_the_attachment_activity_is_addressed_like_every_other_post() -> None:
    ops, connector, _ = make_ops({"a1": fetched(png_bytes((100, 100)))})

    ops.deliver_files(make_entry(), completed([descriptor("a1")]))

    service_url, conversation_id, activity_id, _ = connector.calls[0]
    assert (service_url, conversation_id, activity_id) == (
        SERVICE_URL,
        CHANNEL_CONVERSATION_ID,
        ACTIVITY_ID,
    )


def test_an_artifact_with_no_filename_hint_is_named_after_its_id() -> None:
    ops, connector, _ = make_ops({"a1": fetched(png_bytes((100, 100)))})

    ops.deliver_files(make_entry(), completed([{"artifact_id": "a1"}]))

    assert connector.attachments[0]["name"] == "a1.png"


def test_a_filename_carrying_newlines_is_stripped_before_it_is_posted() -> None:
    # The name is worker-supplied and lands in a serialized activity body.
    ops, connector, _ = make_ops({"a1": fetched(png_bytes((100, 100)))})

    ops.deliver_files(make_entry(), completed([descriptor("a1", "plot\r\nX.png")]))

    assert connector.attachments[0]["name"] == "plotX.png"


# --- what is skipped, and how the user hears about it ------------------------


def test_an_image_still_over_the_budget_is_skipped_and_named() -> None:
    # Noise at full box size cannot be squeezed under the inline budget, so the
    # fit runs and the image is dropped anyway — the case the note exists for.
    big = png_bytes((IMAGE_BOX_PX, IMAGE_BOX_PX), noise=True)
    assert len(big) > MAX_ATTACHMENT_BYTES
    ops, connector, _ = make_ops({"a1": fetched(big)})

    delivered = ops.deliver_files(make_entry(), completed([descriptor("a1", "huge.png")]))

    assert delivered == {}
    assert connector.attachments == []
    assert connector.texts == [skipped_files_note(["huge.png"])]


def test_the_note_names_every_skipped_image_on_one_line() -> None:
    big = png_bytes((IMAGE_BOX_PX, IMAGE_BOX_PX), noise=True)
    ops, connector, _ = make_ops({"a1": fetched(big), "a2": fetched(big)})

    ops.deliver_files(
        make_entry(), completed([descriptor("a1", "one.png"), descriptor("a2", "two.png")])
    )

    assert connector.texts == [skipped_files_note(["one.png", "two.png"])]
    assert "\n" not in connector.texts[0]


def test_a_delivery_that_skipped_nothing_posts_no_note() -> None:
    ops, connector, _ = make_ops({"a1": fetched(png_bytes((100, 100)))})

    ops.deliver_files(make_entry(), completed([descriptor("a1")]))

    assert connector.texts == [""]  # the attachment activity alone


def test_the_surviving_images_are_posted_even_when_a_sibling_is_skipped() -> None:
    ops, connector, _ = make_ops(
        {
            "a1": fetched(png_bytes((IMAGE_BOX_PX, IMAGE_BOX_PX), noise=True)),
            "a2": fetched(png_bytes((200, 100))),
        }
    )

    ops.deliver_files(
        make_entry(), completed([descriptor("a1", "huge.png"), descriptor("a2", "small.png")])
    )

    assert len(connector.attachments) == 1
    assert connector.attachments[0]["name"] == "small.png"
    assert connector.texts[-1] == skipped_files_note(["huge.png"])


def test_a_corrupt_png_is_skipped_and_named_rather_than_raising() -> None:
    ops, connector, _ = make_ops({"a1": fetched(PNG_STUB)})

    delivered = ops.deliver_files(make_entry(), completed([descriptor("a1", "broken.png")]))

    assert delivered == {}
    assert connector.attachments == []
    assert connector.texts == [skipped_files_note(["broken.png"])]


# --- artifacts this member is not for ----------------------------------------


@pytest.mark.parametrize(
    "data,content_type,name",
    [
        (b"%PDF-1.7 ...", "application/pdf", "report.pdf"),
        (b"time,state\n10:00,OPEN\n", "text/csv", "transitions.csv"),
        (b"time\tstate\n10:00\tOPEN\n", "text/tab-separated-values", "transitions.tsv"),
    ],
)
def test_a_document_is_named_when_no_library_is_configured(
    data: bytes, content_type: str, name: str
) -> None:
    # Without a library the bridge cannot deliver a document, and says so: an
    # answer that discusses a table nobody received reads as a broken bridge.
    ops, connector, _ = make_ops({"a1": fetched(data, content_type)})

    assert ops.deliver_files(make_entry(), completed([descriptor("a1", name)])) == {}
    assert connector.texts == [skipped_files_note([name])]


def test_bytes_that_only_claim_to_be_png_take_the_file_path() -> None:
    # delivered_mime is a prediction made before anything was rendered; the
    # magic bytes are what the delivery routes on, so these are not inlined.
    files = RecordingFiles()
    ops, connector, _ = make_ops({"a1": fetched(b"GIF89a not a png", "image/png")}, files=files)

    assert ops.deliver_files(make_entry(), completed([descriptor("a1")])) == {}
    assert files.uploads == [(run_folder_of(), "a1.bin", b"GIF89a not a png", "image/png")]
    assert [att["contentType"] for att in connector.attachments] == [FILE_CARD_CONTENT_TYPE]


def test_an_artifact_that_could_not_be_fetched_costs_only_itself() -> None:
    # A failed fetch costs its own artifact: the sibling is still posted, and the
    # missing one is named under the name its descriptor predicted.
    ops, connector, _ = make_ops(
        {"a1": None, "a2": fetched(png_bytes((100, 100)))},
    )

    ops.deliver_files(make_entry(), completed([descriptor("a1"), descriptor("a2")]))

    assert len(connector.attachments) == 1
    assert connector.texts == ["", skipped_files_note(["a1.png"])]


# --- degenerate inputs -------------------------------------------------------


def test_a_text_only_answer_fetches_nothing_and_posts_nothing() -> None:
    ops, connector, fetcher = make_ops()

    assert ops.deliver_files(make_entry(), completed()) == {}
    assert fetcher.calls == []
    assert connector.calls == []


def test_a_result_with_no_run_id_falls_back_to_the_one_the_entry_carries() -> None:
    # The drain re-attaches a delivery off the persisted entry, where the run id
    # was stamped at claim time; its result need not carry one.
    ops, _, fetcher = make_ops({"a1": fetched(png_bytes((100, 100)))})

    ops.deliver_files(make_entry(run_id=RUN_ID), completed([descriptor("a1")], run_id=None))

    assert fetcher.calls == [(RUN_ID, "a1")]


def test_a_delivery_that_can_name_no_run_at_all_fetches_nothing() -> None:
    ops, connector, fetcher = make_ops({"a1": fetched(png_bytes((100, 100)))})

    assert ops.deliver_files(make_entry(), completed([descriptor("a1")], run_id=None)) == {}
    assert fetcher.calls == []
    assert connector.calls == []


def test_a_malformed_artifacts_field_is_ignored_rather_than_raising() -> None:
    ops, connector, _ = make_ops()
    result = {**completed(), "artifacts": "not a list"}

    assert ops.deliver_files(make_entry(), result) == {}
    assert connector.calls == []


def test_an_entry_that_names_no_conversation_delivers_nothing() -> None:
    # _address raises on a malformed entry; deliver_files owes the engine a
    # return, not an exception.
    ops, connector, _ = make_ops({"a1": fetched(png_bytes((100, 100)))})
    entry = make_entry()
    del entry[MS_SERVICE_URL]

    assert ops.deliver_files(entry, completed([descriptor("a1")])) == {}
    assert connector.calls == []


def test_a_connector_that_refuses_every_post_does_not_raise() -> None:
    # The answer has already landed; no attachment failure may un-deliver it.
    ops, connector, _ = make_ops(
        {"a1": fetched(png_bytes((100, 100)))},
        connector=RecordingConnector(RuntimeError("connector 500")),
    )

    assert ops.deliver_files(make_entry(), completed([descriptor("a1")])) == {}
    assert len(connector.calls) == 1


def test_a_fetcher_that_raises_does_not_raise_out_of_the_member() -> None:
    # fetch_artifact promises never to raise, but the seam takes any callable
    # and the answer has already landed either way.
    def raising_fetcher(*_args: Any, **_kwargs: Any) -> FetchedArtifact | None:
        raise RuntimeError("worker unreachable")

    ops, connector, _ = make_ops(fetcher=raising_fetcher)

    assert ops.deliver_files(make_entry(), completed([descriptor("a1")])) == {}
    assert connector.texts == [skipped_files_note(["a1.png"])]


# --- no public URLs ----------------------------------------------------------


def test_a_fully_delivered_run_still_returns_an_empty_map() -> None:
    # An inline data: URL is not re-fetchable by the engine's unauthenticated
    # GET, so nothing may be stamped onto the descriptors as a public_url: the
    # engine falls back to the worker byte route, which always works.
    ops, connector, _ = make_ops({"a1": fetched(png_bytes((100, 100)))})

    assert ops.deliver_files(make_entry(), completed([descriptor("a1")])) == {}
    assert len(connector.attachments) == 1


# --- with Pillow absent ------------------------------------------------------


@pytest.mark.usefixtures("no_pillow")
def test_every_image_is_skipped_with_the_note_when_pillow_is_missing() -> None:
    # Pillow lives in the optional teams extra. A deployment without it must
    # degrade to "the answer landed, the plots did not, and you were told" —
    # never to a traceback that costs the run its delivery.
    ops, connector, _ = make_ops({"a1": fetched(PNG_STUB)})

    delivered = ops.deliver_files(make_entry(), completed([descriptor("a1", "plot.png")]))

    assert delivered == {}
    assert connector.attachments == []
    assert connector.texts == [skipped_files_note(["plot.png"])]


@pytest.mark.usefixtures("no_pillow")
def test_a_document_is_uploaded_without_pillow() -> None:
    files = RecordingFiles()
    ops, connector, _ = make_ops({"a1": fetched(b"%PDF-1.7 ...", "application/pdf")}, files=files)

    assert ops.deliver_files(make_entry(), completed([descriptor("a1", "report.pdf")])) == {}
    assert files.names == ["report.pdf"]
    assert cards(connector) == [("report.pdf", f"{WEB_ROOT}/osprey/answers/{RUN_ID}/report.pdf")]


@pytest.mark.usefixtures("no_pillow")
def test_every_image_goes_to_the_library_when_pillow_is_missing() -> None:
    files = RecordingFiles()
    ops, connector, _ = make_ops({"a1": fetched(PNG_STUB)}, files=files)

    assert ops.deliver_files(make_entry(), completed([descriptor("a1", "plot.png")])) == {}
    assert files.uploads == [(run_folder_of(), "plot.png", PNG_STUB, "image/png")]
    assert [name for name, _ in cards(connector)] == ["plot.png"]
    assert connector.texts == [""]


# --- files shared from the library -------------------------------------------


@pytest.mark.parametrize(
    "data,content_type,name",
    [
        (b"%PDF-1.7 ...", "application/pdf", "report.pdf"),
        (b"time,state\n10:00,OPEN\n", "text/csv", "transitions.csv"),
        (b"time\tstate\n10:00\tOPEN\n", "text/tab-separated-values", "transitions.tsv"),
    ],
)
def test_a_document_is_uploaded_shared_and_carded(
    data: bytes, content_type: str, name: str
) -> None:
    files = RecordingFiles()
    ops, connector, _ = make_ops({"a1": fetched(data, content_type)}, files=files)

    assert ops.deliver_files(make_entry(), completed([descriptor("a1", name)])) == {}

    assert files.uploads == [(run_folder_of(), name, data, content_type)]
    assert files.shares == [(f"folder:osprey/answers/{RUN_ID}", AUDIENCE)]
    assert cards(connector) == [(name, f"{WEB_ROOT}/osprey/answers/{RUN_ID}/{name}")]
    assert connector.texts == [""]  # the card alone, no note


def test_an_image_over_the_inline_budget_goes_to_the_library_at_full_size() -> None:
    big = png_bytes((IMAGE_BOX_PX * 2, IMAGE_BOX_PX), noise=True)
    files = RecordingFiles()
    ops, connector, _ = make_ops({"a1": fetched(big)}, files=files)

    ops.deliver_files(make_entry(), completed([descriptor("a1", "huge.png")]))

    assert files.uploads == [(run_folder_of(), "huge.png", big, "image/png")]
    assert [name for name, _ in cards(connector)] == ["huge.png"]
    assert connector.texts == [""]


def test_an_image_within_budget_stays_inline_when_a_library_is_configured() -> None:
    files = RecordingFiles()
    ops, connector, _ = make_ops({"a1": fetched(png_bytes((100, 100)))}, files=files)

    ops.deliver_files(make_entry(), completed([descriptor("a1", "small.png")]))

    assert files.uploads == []
    assert files.shares == []
    assert [att["contentType"] for att in connector.attachments] == [ATTACHMENT_CONTENT_TYPE]
    assert connector.member_calls == []


def test_two_files_with_one_name_land_on_two_paths() -> None:
    files = RecordingFiles()
    ops, connector, _ = make_ops(
        {"a1": fetched(b"a\n1\n", "text/csv"), "a2": fetched(b"a\n2\n", "text/csv")},
        files=files,
    )

    ops.deliver_files(
        make_entry(), completed([descriptor("a1", "table.csv"), descriptor("a2", "table.csv")])
    )

    assert files.names == ["table.csv", "table-a2.csv"]
    assert len(cards(connector)) == 2


def test_the_share_names_the_listed_directory_ids_and_no_one_else() -> None:
    members = [
        *MEMBERS,
        {"id": "29:guest", "aadObjectId": "oid-guest", "tenantId": "another-tenant"},
        {"id": "29:nobody"},
        {"id": "28:other-bot", "aadObjectId": "oid-other-bot"},
    ]
    files = RecordingFiles()
    ops, _, _ = make_ops({"a1": fetched(b"%PDF", "application/pdf")}, files=files, members=members)

    ops.deliver_files(make_entry(), completed([descriptor("a1", "r.pdf")]))

    assert [ids for _, ids in files.shares] == [AUDIENCE]


def test_the_listing_is_read_to_its_end_and_not_from_the_roster_cache() -> None:
    files = RecordingFiles()
    ops, connector, _ = make_ops({"a1": fetched(b"%PDF", "application/pdf")}, files=files)
    entry = make_entry()
    ops.room_people(entry)  # fills the roster cache with a capped listing
    capped = list(connector.member_calls)

    ops.deliver_files(entry, completed([descriptor("a1", "r.pdf")]))
    ops.deliver_files(entry, completed([descriptor("a1", "r.pdf")]))

    assert [limit for _, _, limit in capped] == [200]
    assert [limit for _, _, limit in connector.member_calls[len(capped) :]] == [None, None]


def test_a_failed_member_listing_uploads_nothing_and_names_the_files() -> None:
    files = RecordingFiles()
    ops, connector, _ = make_ops(
        {"a1": fetched(b"%PDF", "application/pdf")},
        files=files,
        members=RuntimeError("connector 500"),
    )

    ops.deliver_files(make_entry(), completed([descriptor("a1", "r.pdf")]))

    assert files.uploads == []
    assert connector.texts == [skipped_files_note(["r.pdf"])]


def test_a_conversation_with_no_directory_ids_uploads_nothing_and_names_the_files() -> None:
    files = RecordingFiles()
    ops, connector, _ = make_ops(
        {"a1": fetched(b"%PDF", "application/pdf")},
        files=files,
        members=[{"id": "29:alice", "name": "Alice"}, {"id": f"28:{APP_ID}"}],
    )

    ops.deliver_files(make_entry(), completed([descriptor("a1", "r.pdf")]))

    assert files.uploads == []
    assert files.shares == []
    assert connector.texts == [skipped_files_note(["r.pdf"])]


def test_a_failed_upload_costs_only_that_file() -> None:
    files = RecordingFiles(fail_upload={"one.pdf"})
    ops, connector, _ = make_ops(
        {"a1": fetched(b"%PDF 1", "application/pdf"), "a2": fetched(b"%PDF 2", "application/pdf")},
        files=files,
    )

    ops.deliver_files(
        make_entry(), completed([descriptor("a1", "one.pdf"), descriptor("a2", "two.pdf")])
    )

    assert len(files.shares) == 1
    assert [name for name, _ in cards(connector)] == ["two.pdf"]
    assert connector.texts[-1] == skipped_files_note(["one.pdf"])


def test_a_failed_share_posts_no_card_and_names_every_uploaded_file() -> None:
    files = RecordingFiles(fail_share=GraphError("graph invite refused a recipient"))
    ops, connector, _ = make_ops(
        {"a1": fetched(b"%PDF 1", "application/pdf"), "a2": fetched(b"a\n", "text/csv")},
        files=files,
    )

    ops.deliver_files(
        make_entry(), completed([descriptor("a1", "one.pdf"), descriptor("a2", "two.csv")])
    )

    assert files.names == ["one.pdf", "two.csv"]
    assert cards(connector) == []
    assert connector.texts == [skipped_files_note(["one.pdf", "two.csv"])]


def test_a_run_id_that_is_not_one_segment_uploads_nothing_and_names_the_files() -> None:
    files = RecordingFiles()
    ops, connector, _ = make_ops({"a1": fetched(b"%PDF", "application/pdf")}, files=files)

    ops.deliver_files(make_entry(), completed([descriptor("a1", "r.pdf")], run_id="../elsewhere"))

    assert files.uploads == []
    assert connector.member_calls == []
    assert connector.texts == [skipped_files_note(["r.pdf"])]


def test_a_graph_token_failure_names_the_files_and_keeps_the_inline_images() -> None:
    files = RecordingFiles(fail_upload=TokenError("token endpoint answered HTTP 401"))
    ops, connector, _ = make_ops(
        {"a1": fetched(png_bytes((100, 100))), "a2": fetched(b"%PDF", "application/pdf")},
        files=files,
    )

    ops.deliver_files(
        make_entry(), completed([descriptor("a1", "small.png"), descriptor("a2", "r.pdf")])
    )

    assert [att["contentType"] for att in connector.attachments] == [ATTACHMENT_CONTENT_TYPE]
    assert files.shares == []
    assert connector.texts[-1] == skipped_files_note(["r.pdf"])


def test_the_run_folder_is_shared_once_per_delivery() -> None:
    files = RecordingFiles()
    ops, connector, _ = make_ops(
        {
            "a1": fetched(b"%PDF 1", "application/pdf"),
            "a2": fetched(b"a\n", "text/csv"),
            "a3": fetched(b"{}", "application/json"),
        },
        files=files,
    )

    ops.deliver_files(
        make_entry(),
        completed(
            [descriptor("a1", "a.pdf"), descriptor("a2", "b.csv"), descriptor("a3", "c.json")]
        ),
    )

    assert files.shares == [(f"folder:osprey/answers/{RUN_ID}", AUDIENCE)]
    assert len(cards(connector)) == 3


def test_a_failed_card_is_logged_not_noted() -> None:
    files = RecordingFiles()
    connector = RecordingConnector(RuntimeError("connector 500"), members=MEMBERS)
    ops, _, _ = make_ops(
        {"a1": fetched(b"%PDF", "application/pdf")}, files=files, connector=connector
    )

    assert ops.deliver_files(make_entry(), completed([descriptor("a1", "r.pdf")])) == {}
    assert len(files.shares) == 1
    assert len(connector.calls) == 1  # the card; no note follows it


def test_a_file_delivery_still_returns_an_empty_map() -> None:
    # A SharePoint webUrl needs a signed-in member, so the engine's unauthenticated
    # re-fetch would fail on it exactly as on an inline data: URL.
    files = RecordingFiles()
    ops, connector, _ = make_ops({"a1": fetched(b"%PDF", "application/pdf")}, files=files)

    assert ops.deliver_files(make_entry(), completed([descriptor("a1", "r.pdf")])) == {}
    assert len(cards(connector)) == 1
