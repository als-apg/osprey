"""Artifact fetch (dispatch worker) -> publish (GCS) -> public URL.

Both halves are faked: the worker over :class:`httpx.MockTransport`, the storage
client over a hand-written stand-in. No network, no credentials, and no
dependency on ``google-cloud-storage`` being installed — the lazy import in the
default client factory is never reached, which is the point of the seam.

The load-bearing property under test is that the SERVED ``Content-Type`` — what
the worker actually answered the byte route with — decides everything: whether an
artifact may be an image card at all, what extension its object key gets, and
what type the object is served back under. The descriptor's ``delivered_mime``
and the worker's ``filename`` decide nothing.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import uuid as uuid_mod
from typing import Any

import httpx
import pytest

from osprey.bridges.core import (
    MAX_DELIVERED_DOC_BYTES,
    MAX_DELIVERED_IMAGE_BYTES,
    PNG_MAGIC,
    CoreConfig,
)
from osprey.bridges.google_chat.config import GoogleChatBridgeConfig
from osprey.bridges.google_chat.gcs import publish_artifacts, publish_documents

PNG_BYTES = PNG_MAGIC + b"fake-png-body"

WORKER_URL = "http://work:9190"
WORKER_TOKEN = "work-token"
BUCKET = "my-bucket"


def assert_warned(caplog: pytest.LogCaptureFixture, needle: str) -> None:
    """The module degrades quietly, but must always leave a trace: a skipped
    artifact the operator cannot explain is indistinguishable from a bug.

    Matched on level and message rather than logger name, because the two halves
    log from different places: a fetch-side failure (oversize, fetch failed) from
    the engine's artifacts module, a publish-side one (not a PNG, upload failed,
    no client) from this adapter. Root-level capture sees both.
    """
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a WARNING record, got none"
    assert any(needle in r.getMessage() for r in warnings), (
        f"no WARNING containing {needle!r}; got {[r.getMessage() for r in warnings]}"
    )


@pytest.fixture
def cfg() -> GoogleChatBridgeConfig:
    return GoogleChatBridgeConfig(
        sa_key="/nonexistent/sa.json",
        app_id="users/999",
        gcs_bucket=BUCKET,
        gcs_project="my-project",
        core=CoreConfig(worker_url=WORKER_URL, dispatch_worker_token=WORKER_TOKEN),
    )


@pytest.fixture(autouse=True)
def _capture_warnings(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)


# --- the storage-client stand-in ------------------------------------------


class FakeBlob:
    def __init__(self, bucket: FakeBucket, name: str):
        self.bucket = bucket
        self.name = name
        self.content_type: str | None = None
        self.uploaded: bytes | None = None

    def upload_from_string(self, data: bytes, content_type: str) -> None:
        client = self.bucket.client
        index = client.upload_attempts
        client.upload_attempts += 1
        if index in client.failing_uploads:
            raise RuntimeError("upload failed")
        self.content_type = content_type
        self.uploaded = data
        self.bucket.uploads.append(self)


class FakeBucket:
    def __init__(self, client: FakeGcsClient, name: str):
        self.client = client
        self.name = name
        self.uploads: list[FakeBlob] = []

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self, name)


class FakeGcsClient:
    """``raise_on_bucket`` fails the bucket lookup; ``failing_uploads`` names the
    0-based upload attempts that raise (``ALL`` for every one of them).

    Attempts are counted rather than named because the object key is a uuid
    minted inside the publisher, so a specific artifact's upload cannot be
    targeted by name up front.
    """

    ALL = frozenset(range(64))

    def __init__(
        self,
        *,
        raise_on_bucket: bool = False,
        failing_uploads: frozenset[int] | set[int] = frozenset(),
    ):
        self.raise_on_bucket = raise_on_bucket
        self.failing_uploads = failing_uploads
        self.upload_attempts = 0
        self.buckets: dict[str, FakeBucket] = {}

    def bucket(self, name: str) -> FakeBucket:
        if self.raise_on_bucket:
            raise RuntimeError("bucket lookup failed")
        return self.buckets.setdefault(name, FakeBucket(self, name))

    # -- assertion helpers --------------------------------------------------

    @property
    def blobs(self) -> list[FakeBlob]:
        return [blob for bucket in self.buckets.values() for blob in bucket.uploads]

    def only_blob(self) -> FakeBlob:
        blobs = self.blobs
        assert len(blobs) == 1, f"expected exactly one upload, got {[b.name for b in blobs]}"
        return blobs[0]


def factory_for(client: FakeGcsClient) -> Any:
    return lambda _cfg: client


# --- the worker stand-in ---------------------------------------------------


def make_http(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def worker_handler_for(artifacts: dict[str, tuple[bytes, str | None]]) -> Any:
    """``artifact_id -> (bytes, served Content-Type)``; anything else 404s.

    A ``None`` content type sends no header at all, which is a worker that
    reports the type as absent rather than guessing one.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        artifact_id = str(request.url).rsplit("/", 1)[-1]
        entry = artifacts.get(artifact_id)
        if entry is None:
            return httpx.Response(404)
        assert request.headers["Authorization"] == f"Bearer {WORKER_TOKEN}"
        data, content_type = entry
        headers = {} if content_type is None else {"Content-Type": content_type}
        return httpx.Response(200, content=data, headers=headers)

    return handler


def png(data: bytes = PNG_BYTES) -> tuple[bytes, str | None]:
    """A well-formed PNG, served as one."""
    return (data, "image/png")


def descriptor(artifact_id: str, *, mime: str | None = "text/markdown", filename: Any = None):
    """A run-status descriptor. ``mime`` is what the worker *intended* to
    deliver, which the publisher must never read."""
    entry: dict[str, Any] = {"artifact_id": artifact_id}
    if mime is not None:
        entry["delivered_mime"] = mime
    if filename is not None:
        entry["filename"] = filename
    return entry


# --- publish_artifacts: the happy path and the object key ------------------


def test_a_served_png_is_uploaded_under_a_uuid_key_and_returned_as_a_public_url(cfg):
    client = FakeGcsClient()

    urls = publish_artifacts(
        cfg,
        "R1",
        ["a1"],
        http=make_http(worker_handler_for({"a1": png()})),
        client_factory=factory_for(client),
    )

    blob = client.only_blob()
    assert blob.name.startswith("plots/") and blob.name.endswith(".png")
    uuid_mod.UUID(blob.name[len("plots/") : -len(".png")])  # raises unless uuid-shaped
    assert blob.uploaded == PNG_BYTES
    assert blob.content_type == "image/png"
    assert urls == [f"https://storage.googleapis.com/{BUCKET}/{blob.name}"]


def test_the_artifact_id_never_reaches_the_object_key(cfg):
    # The key is a fresh uuid, so nothing worker-supplied addresses an object.
    client = FakeGcsClient()

    publish_artifacts(
        cfg,
        "R1",
        ["orbit-plot.png"],
        http=make_http(worker_handler_for({"orbit-plot.png": png()})),
        client_factory=factory_for(client),
    )

    name = client.only_blob().name
    assert re.fullmatch(r"plots/[0-9a-f-]+\.png", name)
    assert "orbit-plot" not in name


def test_several_artifacts_are_published_in_input_order(cfg):
    client = FakeGcsClient()
    served = {"a1": png(PNG_MAGIC + b"one"), "a2": png(PNG_MAGIC + b"two")}

    urls = publish_artifacts(
        cfg,
        "R1",
        ["a1", "a2"],
        http=make_http(worker_handler_for(served)),
        client_factory=factory_for(client),
    )

    assert len(urls) == 2
    assert [blob.uploaded for blob in client.blobs] == [PNG_MAGIC + b"one", PNG_MAGIC + b"two"]
    assert [f"https://storage.googleapis.com/{BUCKET}/{b.name}" for b in client.blobs] == urls


# --- publish_artifacts: what may NOT become an image card ------------------


def test_png_bytes_served_as_html_are_declined_rather_than_posted_as_an_image(cfg, caplog):
    # The fallback-conversion case: the descriptor said PNG, the worker served
    # something else. Chat fetches the URL itself, so an image card here renders
    # broken — the served type is the only thing that can catch it.
    client = FakeGcsClient()

    urls = publish_artifacts(
        cfg,
        "R1",
        ["a1"],
        http=make_http(worker_handler_for({"a1": (PNG_BYTES, "text/html")})),
        client_factory=factory_for(client),
    )

    assert urls == []
    assert client.buckets == {}  # bucket() is the only route to a blob
    assert_warned(caplog, "not a PNG")
    assert_warned(caplog, "text/html")


def test_an_artifact_served_with_no_content_type_is_never_guessed_into_an_image(cfg, caplog):
    client = FakeGcsClient()

    urls = publish_artifacts(
        cfg,
        "R1",
        ["a1"],
        http=make_http(worker_handler_for({"a1": (PNG_BYTES, None)})),
        client_factory=factory_for(client),
    )

    assert urls == []
    assert client.buckets == {}
    assert_warned(caplog, "not a PNG")


def test_non_png_bytes_announced_as_a_png_are_declined(cfg, caplog):
    # The opposite mislabel: the type is right, the bytes are not.
    client = FakeGcsClient()

    urls = publish_artifacts(
        cfg,
        "R1",
        ["a1"],
        http=make_http(worker_handler_for({"a1": (b"not-a-png-at-all", "image/png")})),
        client_factory=factory_for(client),
    )

    assert urls == []
    assert client.buckets == {}
    assert_warned(caplog, "not a PNG")


def test_bytes_matching_only_the_first_four_magic_bytes_are_declined(cfg):
    """A sloppy ``data[:4] == b"\\x89PNG"`` check would accept this. The full
    eight-byte prefix is what actually distinguishes a PNG."""
    near_miss = PNG_MAGIC[:4] + b"junkjunk"
    assert near_miss.startswith(PNG_MAGIC[:4]) and not near_miss.startswith(PNG_MAGIC)
    client = FakeGcsClient()

    urls = publish_artifacts(
        cfg,
        "R1",
        ["a1"],
        http=make_http(worker_handler_for({"a1": (near_miss, "image/png")})),
        client_factory=factory_for(client),
    )

    assert urls == []
    assert client.buckets == {}


def test_an_image_over_the_render_bound_is_never_fetched_into_a_card(cfg, caplog):
    oversize = PNG_MAGIC + b"x" * (MAX_DELIVERED_IMAGE_BYTES + 1)
    client = FakeGcsClient()

    urls = publish_artifacts(
        cfg,
        "R1",
        ["a1"],
        http=make_http(worker_handler_for({"a1": png(oversize)})),
        client_factory=factory_for(client),
    )

    assert urls == []
    assert client.buckets == {}
    assert_warned(caplog, "oversize")


def test_an_image_of_exactly_the_render_bound_is_uploaded(cfg):
    """The guard is ``>``, not ``>=`` — pin the boundary so a future ``>=`` is caught."""
    exact = PNG_MAGIC + b"x" * (MAX_DELIVERED_IMAGE_BYTES - len(PNG_MAGIC))
    assert len(exact) == MAX_DELIVERED_IMAGE_BYTES
    client = FakeGcsClient()

    urls = publish_artifacts(
        cfg,
        "R1",
        ["a1"],
        http=make_http(worker_handler_for({"a1": png(exact)})),
        client_factory=factory_for(client),
    )

    assert len(urls) == 1


# --- publish_artifacts: every failure costs at most one URL ----------------


def test_an_artifact_the_worker_does_not_have_costs_only_itself(cfg, caplog):
    client = FakeGcsClient()

    urls = publish_artifacts(
        cfg,
        "R1",
        ["a1"],
        http=make_http(worker_handler_for({})),  # a1 absent -> 404
        client_factory=factory_for(client),
    )

    assert urls == []
    assert_warned(caplog, "artifact fetch failed")


def test_an_unreachable_worker_yields_no_urls_and_no_exception(cfg, caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("worker unreachable")

    urls = publish_artifacts(
        cfg,
        "R1",
        ["a1"],
        http=make_http(handler),
        client_factory=factory_for(FakeGcsClient()),
    )

    assert urls == []
    assert_warned(caplog, "artifact fetch failed")


def test_a_control_character_in_an_artifact_id_does_not_escape(cfg, caplog):
    """``httpx.InvalidURL`` does not subclass ``httpx.HTTPError`` and is raised at
    URL-construction time. Artifact ids are worker-supplied, so a control
    character in one must not cost the user the whole text answer."""
    urls = publish_artifacts(
        cfg,
        "R1",
        ["a\nb"],
        http=make_http(worker_handler_for({})),
        client_factory=factory_for(FakeGcsClient()),
    )

    assert urls == []
    assert_warned(caplog, "artifact fetch failed")


def test_a_failing_bucket_lookup_yields_no_urls_and_no_exception(cfg, caplog):
    urls = publish_artifacts(
        cfg,
        "R1",
        ["a1"],
        http=make_http(worker_handler_for({"a1": png()})),
        client_factory=factory_for(FakeGcsClient(raise_on_bucket=True)),
    )

    assert urls == []
    assert_warned(caplog, "GCS upload failed")


def test_a_failing_upload_yields_no_urls_and_no_exception(cfg, caplog):
    urls = publish_artifacts(
        cfg,
        "R1",
        ["a1"],
        http=make_http(worker_handler_for({"a1": png()})),
        client_factory=factory_for(FakeGcsClient(failing_uploads=FakeGcsClient.ALL)),
    )

    assert urls == []
    assert_warned(caplog, "GCS upload failed")


def test_one_good_and_one_undeliverable_artifact_yield_exactly_one_url(cfg):
    client = FakeGcsClient()
    served = {"good": png(), "bad": (b"not-a-png", "image/png")}

    urls = publish_artifacts(
        cfg,
        "R1",
        ["good", "bad"],
        http=make_http(worker_handler_for(served)),
        client_factory=factory_for(client),
    )

    assert len(urls) == 1
    assert client.only_blob().uploaded == PNG_BYTES


def test_a_client_factory_that_raises_skips_every_artifact_without_raising(cfg, caplog):
    def factory(_cfg):
        raise RuntimeError("bad credentials")

    urls = publish_artifacts(
        cfg,
        "R1",
        ["a1"],
        http=make_http(worker_handler_for({"a1": png()})),
        client_factory=factory,
    )

    assert urls == []
    assert_warned(caplog, "failed to build GCS client")


def test_an_abort_midway_keeps_the_urls_already_uploaded(cfg, caplog):
    """Objects already written exist and their URLs are valid, so discarding them
    would orphan them for nothing."""

    class ExplodingIds(list):
        def __iter__(self):
            yield "a1"
            raise RuntimeError("iteration blew up")

    client = FakeGcsClient()

    urls = publish_artifacts(
        cfg,
        "R1",
        ExplodingIds(["a1", "a2"]),
        http=make_http(worker_handler_for({"a1": png()})),
        client_factory=factory_for(client),
    )

    assert len(urls) == 1
    assert urls == [f"https://storage.googleapis.com/{BUCKET}/{client.only_blob().name}"]
    assert_warned(caplog, "aborted")


# --- publish_artifacts: delivery disabled, and the injection seams ---------


def test_an_unset_bucket_publishes_nothing_and_builds_no_client(cfg):
    built = []

    def factory(_cfg):
        built.append(_cfg)
        return FakeGcsClient()

    urls = publish_artifacts(
        dataclasses.replace(cfg, gcs_bucket=""),
        "R1",
        ["a1"],
        http=make_http(worker_handler_for({"a1": png()})),
        client_factory=factory,
    )

    assert urls == []
    assert built == []


def test_no_artifacts_publishes_nothing_and_builds_no_client(cfg):
    built = []

    def factory(_cfg):
        built.append(_cfg)
        return FakeGcsClient()

    assert (
        publish_artifacts(
            cfg, "R1", [], http=make_http(worker_handler_for({})), client_factory=factory
        )
        == []
    )
    assert built == []


def test_an_injected_http_client_is_left_open_for_its_owner(cfg):
    http = make_http(worker_handler_for({"a1": png()}))

    publish_artifacts(cfg, "R1", ["a1"], http=http, client_factory=factory_for(FakeGcsClient()))

    assert http.is_closed is False


def test_the_artifact_is_fetched_from_the_workers_byte_route(cfg):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=PNG_BYTES, headers={"Content-Type": "image/png"})

    publish_artifacts(
        cfg, "R1", ["a1"], http=make_http(handler), client_factory=factory_for(FakeGcsClient())
    )

    assert seen == [f"{WORKER_URL}/dispatch/R1/artifacts/a1"]


# --- publish_documents: the served type settles key and content type -------


def test_a_document_is_keyed_and_served_by_the_type_the_worker_sent(cfg):
    client = FakeGcsClient()

    results = publish_documents(
        cfg,
        "R1",
        [descriptor("a1", filename="report.md")],
        http=make_http(worker_handler_for({"a1": (b"# hello world", "text/markdown")})),
        client_factory=factory_for(client),
    )

    blob = client.only_blob()
    assert blob.name.startswith("docs/") and blob.name.endswith(".md")
    uuid_mod.UUID(blob.name[len("docs/") : -len(".md")])  # raises unless uuid-shaped
    assert blob.uploaded == b"# hello world"
    assert blob.content_type == "text/markdown"
    assert results == [
        {
            "url": f"https://storage.googleapis.com/{BUCKET}/{blob.name}",
            "filename": "report.md",
            "mime": "text/markdown",
        }
    ]


def test_the_served_type_wins_over_the_descriptors_delivered_mime(cfg):
    # The descriptor records only what the worker INTENDED; a conversion that
    # fell back is not corrected there. Keying on it would publish HTML under
    # .pdf and serve it as a PDF.
    client = FakeGcsClient()

    results = publish_documents(
        cfg,
        "R1",
        [descriptor("a1", mime="application/pdf")],
        http=make_http(worker_handler_for({"a1": (b"<html></html>", "text/html")})),
        client_factory=factory_for(client),
    )

    blob = client.only_blob()
    assert blob.name.endswith(".html")
    assert not blob.name.endswith(".pdf")
    assert blob.content_type == "text/html"
    assert results[0]["mime"] == "text/html"


def test_a_content_type_with_parameters_is_normalized_before_it_is_used(cfg):
    client = FakeGcsClient()

    results = publish_documents(
        cfg,
        "R1",
        [descriptor("a1")],
        http=make_http(worker_handler_for({"a1": (b"a,b,c", "TEXT/CSV; charset=utf-8")})),
        client_factory=factory_for(client),
    )

    blob = client.only_blob()
    assert blob.name.endswith(".csv")
    assert blob.content_type == "text/csv"
    assert results[0]["mime"] == "text/csv"


def test_a_document_served_with_no_content_type_falls_back_to_octet_stream(cfg):
    # The descriptor claims markdown; an absent served type is reported as
    # unknown rather than borrowing that claim.
    client = FakeGcsClient()

    results = publish_documents(
        cfg,
        "R1",
        [descriptor("a1", mime="text/markdown")],
        http=make_http(worker_handler_for({"a1": (b"unknown bytes", None)})),
        client_factory=factory_for(client),
    )

    blob = client.only_blob()
    assert blob.name.endswith(".bin")
    assert blob.content_type == "application/octet-stream"
    assert results[0]["mime"] == "application/octet-stream"


def test_an_unrecognized_served_type_keeps_its_own_content_type_under_a_bin_key(cfg):
    # The extension allowlist and the served type answer different questions:
    # the key falls back to .bin, but the object is still served back as what it
    # actually is.
    client = FakeGcsClient()

    results = publish_documents(
        cfg,
        "R1",
        [descriptor("a1")],
        http=make_http(worker_handler_for({"a1": (b"MZ...", "application/x-msdownload")})),
        client_factory=factory_for(client),
    )

    blob = client.only_blob()
    assert blob.name.endswith(".bin")
    assert blob.content_type == "application/x-msdownload"
    assert results[0]["mime"] == "application/x-msdownload"


def test_an_image_declined_by_publish_artifacts_can_be_republished_as_a_document(cfg):
    # publish_documents reads no mime off the descriptor at all, so an image
    # descriptor whose bytes turned out to be something else can be offered here
    # unchanged and lands under the type it really has.
    image_descriptor = descriptor("a1", mime="image/png", filename="orbit.png")
    http = make_http(worker_handler_for({"a1": (b"<html>fallback</html>", "text/html")}))
    client = FakeGcsClient()

    assert publish_artifacts(cfg, "R1", ["a1"], http=http, client_factory=factory_for(client)) == []
    results = publish_documents(
        cfg, "R1", [image_descriptor], http=http, client_factory=factory_for(client)
    )

    blob = client.only_blob()
    assert blob.name.endswith(".html")
    assert blob.content_type == "text/html"
    assert results[0]["filename"] == "orbit.png"


# --- publish_documents: the filename is a label, never an address ----------


def test_a_path_traversal_filename_never_enters_the_object_key(cfg):
    client = FakeGcsClient()

    results = publish_documents(
        cfg,
        "R1",
        [descriptor("a1", filename="../../etc/passwd")],
        http=make_http(worker_handler_for({"a1": (b"1,2,3", "text/csv")})),
        client_factory=factory_for(client),
    )

    blob = client.only_blob()
    assert re.fullmatch(r"docs/[0-9a-f-]+\.csv", blob.name)
    assert ".." not in blob.name and "etc" not in blob.name and "passwd" not in blob.name
    # The malicious name may survive (sanitized) only as the display label.
    assert results[0]["filename"] == "../../etc/passwd"


def test_a_filename_carrying_newlines_is_sanitized_for_the_label(cfg):
    client = FakeGcsClient()

    results = publish_documents(
        cfg,
        "R1",
        [descriptor("a1", filename="orbit\r\nX-Injected: yes.csv")],
        http=make_http(worker_handler_for({"a1": (b"1,2,3", "text/csv")})),
        client_factory=factory_for(client),
    )

    assert "\n" not in results[0]["filename"] and "\r" not in results[0]["filename"]


def test_a_document_with_no_filename_is_labelled_from_its_extension(cfg):
    client = FakeGcsClient()

    results = publish_documents(
        cfg,
        "R1",
        [descriptor("a1")],
        http=make_http(worker_handler_for({"a1": (b"plain", "text/plain")})),
        client_factory=factory_for(client),
    )

    assert results[0]["filename"] == "artifact.txt"


# --- publish_documents: every failure costs at most one entry --------------


def test_a_descriptor_naming_no_artifact_is_skipped_and_the_rest_still_publish(cfg):
    client = FakeGcsClient()

    results = publish_documents(
        cfg,
        "R1",
        [{"filename": "orphan.md"}, descriptor("a1"), {"artifact_id": ""}],
        http=make_http(worker_handler_for({"a1": (b"body", "text/markdown")})),
        client_factory=factory_for(client),
    )

    assert len(results) == 1
    assert client.only_blob().uploaded == b"body"


def test_a_document_over_the_size_bound_is_skipped(cfg, caplog):
    client = FakeGcsClient()

    results = publish_documents(
        cfg,
        "R1",
        [descriptor("a1", mime="text/plain")],
        http=make_http(
            worker_handler_for({"a1": (b"x" * (MAX_DELIVERED_DOC_BYTES + 1), "text/plain")})
        ),
        client_factory=factory_for(client),
    )

    assert results == []
    assert client.buckets == {}
    assert_warned(caplog, "oversize")


def test_a_document_the_worker_does_not_have_costs_only_itself(cfg, caplog):
    results = publish_documents(
        cfg,
        "R1",
        [descriptor("a1")],
        http=make_http(worker_handler_for({})),
        client_factory=factory_for(FakeGcsClient()),
    )

    assert results == []
    assert_warned(caplog, "artifact fetch failed")


def test_one_good_and_one_failing_upload_yield_exactly_one_result(cfg, caplog):
    # Both fetch fine; only the second upload attempt raises.
    client = FakeGcsClient(failing_uploads={1})
    served = {"good": (b"good bytes", "text/plain"), "bad": (b"bad bytes", "text/plain")}

    results = publish_documents(
        cfg,
        "R1",
        [descriptor("good"), descriptor("bad")],
        http=make_http(worker_handler_for(served)),
        client_factory=factory_for(client),
    )

    assert len(results) == 1
    assert client.only_blob().uploaded == b"good bytes"
    assert_warned(caplog, "GCS upload failed")


def test_a_client_factory_that_raises_skips_every_document_without_raising(cfg, caplog):
    def factory(_cfg):
        raise RuntimeError("bad credentials")

    results = publish_documents(
        cfg,
        "R1",
        [descriptor("a1")],
        http=make_http(worker_handler_for({"a1": (b"body", "text/markdown")})),
        client_factory=factory,
    )

    assert results == []
    assert_warned(caplog, "failed to build GCS client")


# --- publish_documents: delivery disabled ----------------------------------


def test_an_unset_bucket_publishes_no_documents_and_builds_no_client(cfg):
    built = []

    def factory(_cfg):
        built.append(_cfg)
        return FakeGcsClient()

    results = publish_documents(
        dataclasses.replace(cfg, gcs_bucket=""),
        "R1",
        [descriptor("a1")],
        http=make_http(worker_handler_for({})),
        client_factory=factory,
    )

    assert results == []
    assert built == []


def test_no_descriptors_publishes_nothing_and_builds_no_client(cfg):
    built = []

    def factory(_cfg):
        built.append(_cfg)
        return FakeGcsClient()

    assert (
        publish_documents(
            cfg, "R1", [], http=make_http(worker_handler_for({})), client_factory=factory
        )
        == []
    )
    assert built == []
