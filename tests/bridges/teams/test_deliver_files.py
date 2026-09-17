"""What the Teams adapter does with a run's PNG artifacts.

Teams has no file-upload leg in this bridge — the delivery *is* the post, a
message activity carrying the image inline as a ``data:`` URL — so this suite
asserts the activity bodies the adapter handed the connector and the bytes
inside them. The worker is replaced by a fetcher stub, which keeps both the
byte route and the run engine out of the test path entirely.

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
import io
import os
from typing import Any

from PIL import Image

from osprey.bridges.core import FetchedArtifact
from osprey.bridges.teams.events import MS_SERVICE_URL
from osprey.bridges.teams.ops import (
    ATTACHMENT_CONTENT_TYPE,
    DATA_URL_PREFIX,
    IMAGE_BOX_PX,
    MAX_ATTACHMENT_BYTES,
    TeamsOps,
    skipped_images_note,
)
from tests.bridges.teams.test_posting import (
    ACTIVITY_ID,
    CHANNEL_CONVERSATION_ID,
    SERVICE_URL,
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
        self, http: Any, cfg: Any, run_id: str, artifact_id: str
    ) -> FetchedArtifact | None:
        self.calls.append((run_id, artifact_id))
        return self.byte_map.get(artifact_id)


def make_ops(
    byte_map: dict[str, FetchedArtifact | None] | None = None,
    *,
    connector: RecordingConnector | None = None,
    fetcher: Any = None,
) -> tuple[TeamsOps, RecordingConnector, Any]:
    """A ``TeamsOps`` over a recording connector and a canned artifact fetcher."""
    connector = connector if connector is not None else RecordingConnector()
    if fetcher is None:
        fetcher = RecordingFetcher(byte_map if byte_map is not None else {})
    return TeamsOps(make_config(), connector, artifact_fetcher=fetcher), connector, fetcher


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
    assert connector.texts == [skipped_images_note(["huge.png"])]


def test_the_note_names_every_skipped_image_on_one_line() -> None:
    big = png_bytes((IMAGE_BOX_PX, IMAGE_BOX_PX), noise=True)
    ops, connector, _ = make_ops({"a1": fetched(big), "a2": fetched(big)})

    ops.deliver_files(
        make_entry(), completed([descriptor("a1", "one.png"), descriptor("a2", "two.png")])
    )

    assert connector.texts == [skipped_images_note(["one.png", "two.png"])]
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
    assert connector.texts[-1] == skipped_images_note(["huge.png"])


def test_a_corrupt_png_is_skipped_and_named_rather_than_raising() -> None:
    ops, connector, _ = make_ops({"a1": fetched(PNG_STUB)})

    delivered = ops.deliver_files(make_entry(), completed([descriptor("a1", "broken.png")]))

    assert delivered == {}
    assert connector.attachments == []
    assert connector.texts == [skipped_images_note(["broken.png"])]


# --- artifacts this member is not for ----------------------------------------


def test_a_document_artifact_is_ignored_without_a_note() -> None:
    # Documents are out of scope for v1: naming a PDF in an "I couldn't attach
    # this image" note would report a delivery the bridge never promised.
    ops, connector, _ = make_ops({"a1": fetched(b"%PDF-1.7 ...", "application/pdf")})

    assert ops.deliver_files(make_entry(), completed([descriptor("a1", "report.pdf")])) == {}
    assert connector.calls == []


def test_bytes_that_only_claim_to_be_png_are_ignored() -> None:
    # delivered_mime is a prediction made before anything was rendered; the
    # magic bytes are what the delivery routes on.
    ops, connector, _ = make_ops({"a1": fetched(b"GIF89a not a png", "image/png")})

    assert ops.deliver_files(make_entry(), completed([descriptor("a1")])) == {}
    assert connector.calls == []


def test_an_artifact_that_could_not_be_fetched_costs_only_itself() -> None:
    # A failed fetch says nothing about what the bytes were, so it is logged
    # rather than named as an image: the note must not invent a PNG.
    ops, connector, _ = make_ops(
        {"a1": None, "a2": fetched(png_bytes((100, 100)))},
    )

    ops.deliver_files(make_entry(), completed([descriptor("a1"), descriptor("a2")]))

    assert len(connector.attachments) == 1
    assert connector.texts == [""]


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
    assert connector.calls == []


# --- no public URLs ----------------------------------------------------------


def test_a_fully_delivered_run_still_returns_an_empty_map() -> None:
    # An inline data: URL is not re-fetchable by the engine's unauthenticated
    # GET, so nothing may be stamped onto the descriptors as a public_url: the
    # engine falls back to the worker byte route, which always works.
    ops, connector, _ = make_ops({"a1": fetched(png_bytes((100, 100)))})

    assert ops.deliver_files(make_entry(), completed([descriptor("a1")])) == {}
    assert len(connector.attachments) == 1


# --- with Pillow absent ------------------------------------------------------


def test_every_image_is_skipped_with_the_note_when_pillow_is_missing(no_pillow: None) -> None:
    # Pillow lives in the optional teams extra. A deployment without it must
    # degrade to "the answer landed, the plots did not, and you were told" —
    # never to a traceback that costs the run its delivery.
    ops, connector, _ = make_ops({"a1": fetched(PNG_STUB)})

    delivered = ops.deliver_files(make_entry(), completed([descriptor("a1", "plot.png")]))

    assert delivered == {}
    assert connector.attachments == []
    assert connector.texts == [skipped_images_note(["plot.png"])]


def test_a_document_is_still_ignored_without_pillow(no_pillow: None) -> None:
    ops, connector, _ = make_ops({"a1": fetched(b"%PDF-1.7 ...", "application/pdf")})

    assert ops.deliver_files(make_entry(), completed([descriptor("a1")])) == {}
    assert connector.calls == []
