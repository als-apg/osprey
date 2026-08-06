"""NextcloudTalkOps.deliver_files: artifact upload + room share, and the empty return.

Two MockTransports run side by side — one standing in for the dispatch worker's artifact
byte route, one for Nextcloud's DAV and OCS surfaces — so the whole fetch/upload/share
chain is exercised without a socket.

Three properties carry the weight here:

* **The return is always ``{}``**, even on complete success. A Talk room share is
  member-authenticated, and the engine only accepts URLs that survive an unauthenticated
  GET (it re-fetches them for prior-artifact re-injection), so returning one would break
  re-injection far from its cause. Several tests pin the empty return so nobody "fixes"
  it into returning the share URL.
* **Every failure point is swallowed.** The answer text has already landed by the time
  this runs; an exception here would turn a delivered answer into a failed run.
* **No capability or health probe is ever issued** — probing is engine-owned, and the
  recorded request lists are asserted against it.
"""

from urllib.parse import parse_qs

import httpx
import pytest

from osprey.bridges.core import (
    MAX_ARTIFACT_BYTES,
    PNG_MAGIC,
    CoreConfig,
)
from osprey.bridges.nextcloud_talk import NextcloudBridgeConfig, TalkClient
from osprey.bridges.nextcloud_talk.client import UPLOAD_ROOT
from osprey.bridges.nextcloud_talk.ops import NC_ROOM, NextcloudTalkOps, _upload_name

CFG = NextcloudBridgeConfig(
    base_url="https://cloud.example.org",
    bot_account="osprey-bot",
    app_password="app-pw",
    rooms=("roomA",),
    core=CoreConfig(worker_url="http://work:9190", dispatch_worker_token="work-token"),
)

PNG = PNG_MAGIC + b"pretend-image-bytes"
PDF = b"%PDF-1.7 pretend-document"

ENTRY = {NC_ROOM: "roomA", "nc_message_id": 41, "history_key": "nextcloud:roomA"}


def result(*artifacts, run_id="R1"):
    """A terminal result carrying ``artifacts`` for run ``run_id``."""
    return {
        "status": "completed",
        "text_output": "done",
        "run_id": run_id,
        "artifacts": list(artifacts),
    }


def png_descriptor(artifact_id, **extra):
    """A #363 descriptor for a PNG image artifact."""
    return {"artifact_id": artifact_id, "delivered_mime": "image/png", **extra}


class FakeWorker:
    """The worker's artifact byte route: serves per-id payloads, records every request.

    A payload is either raw bytes or ``(bytes, content_type)`` — the byte route's
    Content-Type is what a consumer names a fallback delivery from, so it is part
    of what this fake has to reproduce.
    """

    def __init__(self, payloads=None, *, fail_ids=(), status=500):
        self.payloads = dict(payloads or {})
        self.fail_ids = set(fail_ids)
        self.status = status
        self.requests: list[httpx.Request] = []

    @property
    def fetched(self) -> list[str]:
        """Artifact ids fetched, in order."""
        return [request.url.path.rsplit("/", 1)[-1] for request in self.requests]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        artifact_id = request.url.path.rsplit("/", 1)[-1]
        if artifact_id in self.fail_ids:
            return httpx.Response(self.status, text="worker down")
        payload = self.payloads.get(artifact_id, PNG)
        data, content_type = payload if isinstance(payload, tuple) else (payload, None)
        headers = {"Content-Type": content_type} if content_type else {}
        return httpx.Response(200, content=data, headers=headers)


class FakeNextcloud:
    """Nextcloud's DAV + OCS share surfaces, with a switch per failure point."""

    def __init__(self, *, fail_mkcol=False, fail_put=False, share_status=None, exc=None):
        self.fail_mkcol = fail_mkcol
        self.fail_put = fail_put
        self.share_status = share_status
        self.exc = exc
        self.requests: list[httpx.Request] = []

    def of(self, method: str) -> list[httpx.Request]:
        """Every recorded request with the given method."""
        return [request for request in self.requests if request.method == method]

    @property
    def shares(self) -> list[dict[str, list[str]]]:
        """The form body of every share call, parsed."""
        return [parse_qs(request.content.decode()) for request in self.of("POST")]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.exc is not None:
            raise self.exc
        if request.method == "MKCOL":
            return httpx.Response(500 if self.fail_mkcol else 201)
        if request.method == "PUT":
            return httpx.Response(500 if self.fail_put else 201)
        if request.method == "POST":
            if self.share_status is not None:
                return httpx.Response(
                    self.share_status,
                    json={"ocs": {"meta": {"message": "you are not allowed to share"}}},
                )
            return httpx.Response(200, json={"ocs": {"meta": {"status": "ok"}, "data": {"id": 9}}})
        return httpx.Response(405)


def _ops(worker=None, nextcloud=None):
    """``(ops, worker, nextcloud)`` wired to both fakes over MockTransport."""
    worker = worker if worker is not None else FakeWorker()
    nextcloud = nextcloud if nextcloud is not None else FakeNextcloud()
    ops = NextcloudTalkOps(
        CFG,
        TalkClient(CFG, client=httpx.Client(transport=httpx.MockTransport(nextcloud.handler))),
        httpx.Client(transport=httpx.MockTransport(worker.handler)),
    )
    return ops, worker, nextcloud


# ==========================================================================
# Happy path — and the unconditional empty return
# ==========================================================================


def test_an_image_artifact_is_fetched_uploaded_and_shared():
    ops, worker, nc = _ops()
    ops.deliver_files(ENTRY, result(png_descriptor("plot1")))

    assert worker.fetched == ["plot1"]
    put = nc.of("PUT")[0]
    assert put.content == PNG
    assert put.url.path.endswith(f"{UPLOAD_ROOT}/roomA/R1/plot1.png")
    assert nc.shares == [
        {
            "path": [f"/{UPLOAD_ROOT}/roomA/R1/plot1.png"],
            "shareType": ["10"],
            "shareWith": ["roomA"],
        }
    ]


def test_delivery_returns_an_empty_mapping_on_full_success():
    # THE crux: a room share is member-authenticated, and the engine re-fetches returned
    # URLs unauthenticated for prior-artifact re-injection. Returning one would 401 there.
    ops, _, nc = _ops()
    assert ops.deliver_files(ENTRY, result(png_descriptor("plot1"))) == {}
    assert nc.of("PUT")  # ... and it really did deliver something


def test_the_empty_return_holds_for_several_artifacts():
    ops, worker, nc = _ops()
    artifacts = [png_descriptor("a"), png_descriptor("b"), png_descriptor("c")]
    assert ops.deliver_files(ENTRY, result(*artifacts)) == {}
    assert worker.fetched == ["a", "b", "c"]
    assert len(nc.of("PUT")) == 3
    assert len(nc.shares) == 3


def test_the_upload_directory_chain_is_created_outermost_first():
    # MKCOL is not recursive, so each level is created in turn.
    ops, _, nc = _ops()
    ops.deliver_files(ENTRY, result(png_descriptor("plot1")))
    # url.path is the decoded form; the percent-encoding of the space in UPLOAD_ROOT is
    # the client's concern and is covered by its own tests.
    created = [request.url.path.split("/osprey-bot/", 1)[-1] for request in nc.of("MKCOL")]
    assert created == [UPLOAD_ROOT, f"{UPLOAD_ROOT}/roomA", f"{UPLOAD_ROOT}/roomA/R1"]


def test_artifacts_land_under_the_run_so_two_runs_cannot_collide():
    ops, _, nc = _ops()
    ops.deliver_files(ENTRY, result(png_descriptor("plot"), run_id="R1"))
    ops.deliver_files(ENTRY, result(png_descriptor("plot"), run_id="R2"))
    paths = [request.url.path for request in nc.of("PUT")]
    assert paths[0].endswith("/roomA/R1/plot.png")
    assert paths[1].endswith("/roomA/R2/plot.png")


def test_the_run_id_falls_back_to_the_persisted_entry():
    # The drain's re-attached delivery has the run id on the entry as well as the result.
    ops, worker, nc = _ops()
    entry = {**ENTRY, "run_id": "R9"}
    ops.deliver_files(entry, {"status": "completed", "run_id": None, "artifacts": ["plot"]})
    assert worker.requests[0].url.path == "/dispatch/R9/artifacts/plot"
    assert nc.of("PUT")[0].url.path.endswith("/roomA/R9/plot.png")


# ==========================================================================
# Document artifacts take the non-PNG path and the larger budget
# ==========================================================================


def test_a_document_artifact_keeps_the_workers_filename():
    ops, worker, nc = _ops(
        FakeWorker({"doc1": PDF}),
    )
    ops.deliver_files(
        ENTRY,
        result(
            {
                "artifact_id": "doc1",
                "delivered_mime": "application/pdf",
                "filename": "orbit report.pdf",
            }
        ),
    )
    assert nc.of("PUT")[0].content == PDF
    # Sanitized to one safe segment, extension not doubled.
    assert nc.of("PUT")[0].url.path.endswith("/roomA/R1/orbit_report.pdf")


def test_a_document_without_a_filename_is_named_from_its_id_and_mime():
    ops, _, nc = _ops(FakeWorker({"doc1": PDF}))
    ops.deliver_files(ENTRY, result({"artifact_id": "doc1", "delivered_mime": "application/pdf"}))
    assert nc.of("PUT")[0].url.path.endswith("/roomA/R1/doc1.pdf")


def test_a_document_may_exceed_the_image_budget():
    # Images are capped at MAX_ARTIFACT_BYTES; documents get the larger MAX_DOC_BYTES,
    # so a 3 MiB PDF is delivered where a 3 MiB image would be dropped.
    big = b"%PDF" + b"x" * (MAX_ARTIFACT_BYTES + 1)
    ops, _, nc = _ops(FakeWorker({"doc1": big}))
    ops.deliver_files(ENTRY, result({"artifact_id": "doc1", "delivered_mime": "application/pdf"}))
    assert nc.of("PUT")[0].content == big


def test_an_oversize_image_is_dropped_without_uploading():
    ops, _, nc = _ops(FakeWorker({"plot": PNG_MAGIC + b"x" * MAX_ARTIFACT_BYTES}))
    assert ops.deliver_files(ENTRY, result(png_descriptor("plot"))) == {}
    assert nc.of("PUT") == []


# ==========================================================================
# The bytes decide, not the descriptor
#
# A descriptor is written when a run completes and promises "image/png" for
# everything the worker intends to render. Whether the render actually happens
# is only known at fetch time — a converter dependency can be missing, a
# renderer can crash — and the byte route then serves the ORIGINAL bytes under
# their real Content-Type. Routing on the promise loses those artifacts
# silently (osprey #503), so delivery routes on what arrived.
# ==========================================================================


def test_a_predicted_image_that_arrives_as_text_is_delivered_as_a_document():
    # osprey #503: descriptor says image/png, the conversion failed, the byte
    # route served the original CSV. The file must reach the room, not vanish.
    csv = b"x,sin_x\n0,0\n"
    ops, _, nc = _ops(FakeWorker({"plot": (csv, "text/csv")}))
    assert ops.deliver_files(ENTRY, result(png_descriptor("plot"))) == {}
    assert nc.of("PUT")[0].content == csv
    assert nc.of("PUT")[0].url.path.endswith("/roomA/R1/plot.csv")


def test_a_fallback_replaces_the_predicted_extension_rather_than_stacking_it():
    # The descriptor's filename is predicted too ("data.png" for a render that
    # never happened); the extension still comes from the mime that was served.
    csv = b"x,sin_x\n"
    ops, _, nc = _ops(FakeWorker({"plot": (csv, "text/csv")}))
    ops.deliver_files(ENTRY, result(png_descriptor("plot", filename="data.png")))
    assert nc.of("PUT")[0].url.path.endswith("/roomA/R1/data.csv")


def test_a_fallback_without_a_content_type_falls_back_to_the_predicted_mime():
    # An older worker that serves no Content-Type leaves only the prediction to
    # name the file — still better than dropping it.
    ops, _, nc = _ops(FakeWorker({"doc1": PDF}))
    ops.deliver_files(ENTRY, result({"artifact_id": "doc1", "delivered_mime": "application/pdf"}))
    assert nc.of("PUT")[0].url.path.endswith("/roomA/R1/doc1.pdf")


def test_png_bytes_are_delivered_as_an_image_whatever_the_descriptor_predicted():
    # The mirror case: a descriptor with no mime at all, PNG bytes on the wire.
    ops, _, nc = _ops(FakeWorker({"doc1": (PNG, "image/png")}))
    ops.deliver_files(ENTRY, result({"artifact_id": "doc1"}))
    assert nc.of("PUT")[0].url.path.endswith("/roomA/R1/doc1.png")


# ==========================================================================
# Artifact-entry normalisation (core's artifact_descriptors tolerance)
# ==========================================================================


def test_bare_id_strings_from_an_older_worker_are_delivered_as_images():
    ops, worker, nc = _ops()
    assert ops.deliver_files(ENTRY, result("plot1", "plot2")) == {}
    assert worker.fetched == ["plot1", "plot2"]
    assert len(nc.of("PUT")) == 2


@pytest.mark.parametrize(
    "junk",
    [None, 42, "", {}, {"artifact_id": ""}, {"artifact_id": None}, ["nested"]],
)
def test_malformed_artifact_entries_are_skipped_not_raised_on(junk):
    ops, worker, nc = _ops()
    assert ops.deliver_files(ENTRY, result(junk)) == {}
    assert worker.requests == []
    assert nc.requests == []


def test_a_malformed_entry_does_not_cost_its_valid_siblings():
    ops, worker, nc = _ops()
    ops.deliver_files(ENTRY, result(None, png_descriptor("plot1"), 42, "plot2"))
    assert worker.fetched == ["plot1", "plot2"]
    assert len(nc.of("PUT")) == 2


@pytest.mark.parametrize("artifacts", [None, [], "plot1", {"artifact_id": "plot1"}, 7])
def test_nothing_is_delivered_when_the_artifacts_field_is_absent_or_malformed(artifacts):
    # A non-list would otherwise iterate into characters or dict keys and fabricate ids.
    ops, worker, nc = _ops()
    payload = {"status": "completed", "run_id": "R1", "artifacts": artifacts}
    assert ops.deliver_files(ENTRY, payload) == {}
    assert worker.requests == []
    assert nc.requests == []


# ==========================================================================
# Every failure point is swallowed
# ==========================================================================


def test_a_failed_artifact_fetch_degrades_to_text_only():
    ops, worker, nc = _ops(FakeWorker(fail_ids={"plot"}))
    assert ops.deliver_files(ENTRY, result(png_descriptor("plot"))) == {}
    assert worker.requests  # it tried
    assert nc.requests == []  # nothing to upload


def test_a_worker_transport_error_is_swallowed():
    def broken(request):
        raise httpx.ConnectError("worker unreachable", request=request)

    ops = NextcloudTalkOps(
        CFG,
        TalkClient(
            CFG, client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(201)))
        ),
        httpx.Client(transport=httpx.MockTransport(broken)),
    )
    assert ops.deliver_files(ENTRY, result(png_descriptor("plot"))) == {}


def test_a_failed_mkcol_is_swallowed():
    ops, _, nc = _ops(nextcloud=FakeNextcloud(fail_mkcol=True))
    assert ops.deliver_files(ENTRY, result(png_descriptor("plot"))) == {}
    assert nc.of("PUT") == []  # the chain never got far enough to upload


def test_a_failed_put_is_swallowed():
    ops, _, nc = _ops(nextcloud=FakeNextcloud(fail_put=True))
    assert ops.deliver_files(ENTRY, result(png_descriptor("plot"))) == {}
    assert nc.of("PUT")  # it tried
    assert nc.shares == []  # and did not pretend to share


@pytest.mark.parametrize("share_status", [403, 500])
def test_a_failed_share_is_swallowed(share_status):
    # 403 is how "the bot may not share into that room" arrives — a genuine denial the
    # client refuses to mistake for a tolerated duplicate. It must not escape either.
    ops, _, nc = _ops(nextcloud=FakeNextcloud(share_status=share_status))
    assert ops.deliver_files(ENTRY, result(png_descriptor("plot"))) == {}
    assert nc.shares  # it tried


def test_a_nextcloud_transport_error_is_swallowed():
    ops, _, nc = _ops(nextcloud=FakeNextcloud(exc=httpx.ConnectError("cloud down")))
    assert ops.deliver_files(ENTRY, result(png_descriptor("plot"))) == {}


def test_one_failing_artifact_does_not_stop_the_others():
    ops, worker, nc = _ops(FakeWorker(fail_ids={"broken"}))
    assert ops.deliver_files(ENTRY, result("ok1", "broken", "ok2")) == {}
    assert worker.fetched == ["ok1", "broken", "ok2"]
    uploaded = [request.url.path.rsplit("/", 1)[-1] for request in nc.of("PUT")]
    assert uploaded == ["ok1.png", "ok2.png"]


@pytest.mark.parametrize("room", [None, "", 41, {}])
def test_an_entry_without_a_room_delivers_nothing_quietly(room):
    ops, worker, nc = _ops()
    assert ops.deliver_files({**ENTRY, NC_ROOM: room}, result(png_descriptor("plot"))) == {}
    assert worker.requests == []
    assert nc.requests == []


@pytest.mark.parametrize("run_id", [None, "", 41])
def test_a_result_without_a_usable_run_id_delivers_nothing_quietly(run_id):
    ops, worker, nc = _ops()
    payload = {"status": "completed", "run_id": run_id, "artifacts": ["plot"]}
    assert ops.deliver_files(ENTRY, payload) == {}
    assert worker.requests == []
    assert nc.requests == []


@pytest.mark.parametrize(("room", "run_id"), [("a/b", "R1"), ("roomA", "../etc"), ("..", "R1")])
def test_a_token_that_is_not_one_path_segment_delivers_nothing(room, run_id):
    # Writing outside the artifact tree is not an acceptable degradation; text-only is.
    ops, worker, nc = _ops()
    payload = {"status": "completed", "run_id": run_id, "artifacts": ["plot"]}
    assert ops.deliver_files({**ENTRY, NC_ROOM: room}, payload) == {}
    assert worker.requests == []
    assert nc.requests == []


# ==========================================================================
# The adapter never probes capabilities or health
# ==========================================================================


def test_delivery_touches_only_the_artifact_byte_route_and_the_dav_share_surfaces():
    ops, worker, nc = _ops()
    ops.deliver_files(ENTRY, result(png_descriptor("plot1"), {"artifact_id": "doc1"}))

    worker_paths = [request.url.path for request in worker.requests]
    assert worker_paths == ["/dispatch/R1/artifacts/plot1", "/dispatch/R1/artifacts/doc1"]
    every_path = worker_paths + [request.url.path for request in nc.requests]
    assert not any("health" in path for path in every_path)
    assert not any("capabilities" in path for path in every_path)
    assert {request.method for request in nc.requests} <= {"MKCOL", "PUT", "POST"}


# ==========================================================================
# Upload filenames stay inside the artifact tree
# ==========================================================================


@pytest.mark.parametrize(
    ("label", "extension", "expected"),
    [
        ("plot", ".png", "plot.png"),
        ("plot.png", ".png", "plot.png"),  # not doubled
        ("PLOT.PNG", ".png", "PLOT.PNG"),  # case-insensitive match
        # A predicted name whose extension the delivery contradicted: the mime
        # wins and the stale extension is replaced, never stacked.
        ("data.png", ".csv", "data.csv"),
        ("run_2026.04.01_orbit", ".csv", "run_2026.04.01_orbit.csv"),  # not an extension
        ("orbit report", ".pdf", "orbit_report.pdf"),
        # Separators become "_" and the leading dot/underscore run is stripped, so a
        # traversal attempt collapses into one ordinary filename.
        ("../../etc/passwd", ".bin", "etc_passwd.bin"),
        ("with\nnewline", ".txt", "withnewline.txt"),
        ("", ".png", "fallback.png"),
        (None, ".png", "fallback.png"),
        (42, ".png", "fallback.png"),
        ("..", ".png", "fallback.png"),
        ("/", ".png", "fallback.png"),
        ("x" * 300, ".png", "x" * 120 + ".png"),
    ],
)
def test_upload_names_are_a_single_safe_segment(label, extension, expected):
    name = _upload_name(label, extension, "fallback")
    assert name == expected
    assert "/" not in name
    assert name not in (".", "..")


def test_upload_name_falls_back_twice_when_even_the_fallback_is_unusable():
    assert _upload_name(None, ".png", "..") == "artifact.png"
