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
    MAX_DELIVERED_IMAGE_BYTES,
    PNG_MAGIC,
    CoreConfig,
)
from osprey.bridges.nextcloud_talk import NextcloudBridgeConfig, TalkClient
from osprey.bridges.nextcloud_talk.client import UPLOAD_ROOT
from osprey.bridges.nextcloud_talk.ops import (
    NC_ROOM,
    NextcloudTalkOps,
    _unique_stems,
    _upload_name,
    _upload_stem,
)

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
    # Images are capped at MAX_DELIVERED_IMAGE_BYTES; documents get the larger MAX_DELIVERED_DOC_BYTES,
    # so a 3 MiB PDF is delivered where a 3 MiB image would be dropped.
    big = b"%PDF" + b"x" * (MAX_DELIVERED_IMAGE_BYTES + 1)
    ops, _, nc = _ops(FakeWorker({"doc1": (big, "application/pdf")}))
    ops.deliver_files(ENTRY, result({"artifact_id": "doc1", "delivered_mime": "application/pdf"}))
    assert nc.of("PUT")[0].content == big


def test_an_oversize_image_is_dropped_without_uploading():
    # Fetched under the document budget like everything else, so the payload does
    # arrive; the image bound is applied after, once the bytes prove to be a PNG.
    ops, _, nc = _ops(
        FakeWorker({"plot": (PNG_MAGIC + b"x" * MAX_DELIVERED_IMAGE_BYTES, "image/png")})
    )
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


def test_a_payload_announced_as_a_png_without_the_magic_bytes_is_not_an_image():
    # The magic-byte guard stays on — Talk renders a mislabelled .png as a broken image —
    # but it costs the artifact its image path, not its delivery.
    ops, _, nc = _ops(FakeWorker({"plot": (b"not-a-png", "image/png")}))
    assert ops.deliver_files(ENTRY, result(png_descriptor("plot"))) == {}
    put = nc.of("PUT")[0]
    assert put.content == b"not-a-png"
    assert put.url.path.endswith("/roomA/R1/plot.bin")


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
    name = _upload_name(_upload_stem(label, "fallback"), extension)
    assert name == expected
    assert "/" not in name
    assert name not in (".", "..")


def test_upload_name_falls_back_twice_when_even_the_fallback_is_unusable():
    assert _upload_name(_upload_stem(None, ".."), ".png") == "artifact.png"


# ==========================================================================
# Two artifacts never claim one upload name
# ==========================================================================


def test_same_named_artifacts_are_kept_apart():
    # The worker names an artifact by basename, so a run whose steps each wrote their
    # own plot.png sends two descriptors with one filename. They share a DAV directory,
    # so an undisambiguated second upload would overwrite the first.
    stems = _unique_stems(
        [
            {"artifact_id": "a1", "filename": "plot.png"},
            {"artifact_id": "a2", "filename": "plot.png"},
        ]
    )
    assert stems["a1"] == "plot.png"
    # The id slice goes AHEAD of the trailing dot-suffix, so the name still reads as one.
    assert stems["a2"] == "plot-a2.png"


def test_documents_whose_names_differ_only_by_extension_are_kept_apart():
    # The reachable shape of the stem-vs-name gap, and the reason deduping on the stem
    # alone is not enough: _upload_name does not double an extension the stem already
    # carries, so the DISTINCT stems "report" and "report.bin" both upload as report.bin
    # once the bytes are served as application/octet-stream.
    #
    # octet-stream is what makes the pair reachable. _predicted_filename forces an
    # extension only for image/png and otherwise passes the stored basename through, so
    # a bare "report" survives exactly when its type was never pinned down by an
    # extension — which is the same condition that types it octet-stream. A text/html
    # pair could not arise: a stored file named "report" would not have been typed
    # text/html in the first place.
    opaque = b"\x00\x01opaque-bytes"
    ops, _, nc = _ops(
        FakeWorker(
            {
                "d1": (opaque, "application/octet-stream"),
                "d2": (opaque + b"2", "application/octet-stream"),
            }
        ),
    )
    ops.deliver_files(
        ENTRY,
        result(
            {
                "artifact_id": "d1",
                "delivered_mime": "application/octet-stream",
                "filename": "report",
            },
            {
                "artifact_id": "d2",
                "delivered_mime": "application/octet-stream",
                "filename": "report.bin",
            },
        ),
    )
    puts = nc.of("PUT")
    assert [put.url.path.rsplit("/", 1)[-1] for put in puts] == ["report.bin", "report-d2.bin"]
    assert [put.content for put in puts] == [opaque, opaque + b"2"]


def test_stems_that_differ_but_name_one_file_are_kept_apart():
    # The same gap at the unit level. This particular pair needs a cross-bucket route to
    # occur for real (a doc-bucket artifact whose bytes serve as image/png, next to an
    # image-bucket plot.png), so it pins the MECHANISM rather than a worker output shape.
    stems = _unique_stems(
        [
            {"artifact_id": "a1", "filename": "plot.png"},
            {"artifact_id": "a2", "filename": "plot"},
        ]
    )
    names = [_upload_name(stems[aid], ".png") for aid in ("a1", "a2")]
    assert names == ["plot.png", "plot-a2.png"]


def test_a_disambiguated_truncated_stem_is_still_one_safe_segment():
    # _segment strips its dot runs BEFORE truncating at 120, so a long name cut exactly
    # at a dot ends in one and _disambiguate's partition yields an empty tail. The
    # trailing dot is cosmetic; what must hold is that the name is still a single
    # segment dav_mkcol_put accepts.
    long_name = "a" * 119 + "." + "b" * 30
    stems = _unique_stems(
        [{"artifact_id": "a1", "filename": long_name}, {"artifact_id": "a2", "filename": long_name}]
    )
    for stem in stems.values():
        name = _upload_name(stem, ".png")
        assert "/" not in name
        assert name not in (".", "..")
        assert name.strip(".")
    assert len(set(stems.values())) == 2


@pytest.mark.parametrize(
    ("first", "second", "extension"),
    [
        # Only a KNOWN extension may be stripped when keying: treating ".v2" as one
        # keys these apart, and they then collide on report.v2.pdf.
        ("report.v2", "report.v2.pdf", ".pdf"),
        # One strip is not enough — plot.png.pdf must lose both before the comparison.
        ("plot.png", "plot.png.pdf", ".pdf"),
        # _upload_name's already-suffixed check is case-insensitive, so the key must
        # fold case too.
        ("a.PDF", "a", ".pdf"),
        ("report", "report.html", ".html"),
    ],
)
def test_names_that_would_collide_after_the_extension_are_kept_apart(first, second, extension):
    stems = _unique_stems(
        [{"artifact_id": "d1", "filename": first}, {"artifact_id": "d2", "filename": second}]
    )
    names = [_upload_name(stems[aid], extension) for aid in ("d1", "d2")]
    assert names[0] != names[1], f"{first!r} and {second!r} both upload as {names[0]!r}"


def test_suffixes_that_are_not_extensions_are_not_disambiguated():
    # The flip side of stripping only KNOWN extensions: these two never collide, so
    # neither should be renamed.
    stems = _unique_stems(
        [
            {"artifact_id": "d1", "filename": "report.draft"},
            {"artifact_id": "d2", "filename": "report.final"},
        ]
    )
    assert [_upload_name(stems[aid], ".pdf") for aid in ("d1", "d2")] == [
        "report.draft.pdf",
        "report.final.pdf",
    ]


def test_stems_that_differ_only_by_case_are_kept_apart():
    # A case-insensitive DAV backend would treat these as one path.
    stems = _unique_stems(
        [{"artifact_id": "a1", "filename": "PLOT"}, {"artifact_id": "a2", "filename": "plot"}]
    )
    assert len({stem.lower() for stem in stems.values()}) == 2


def test_stems_stay_unique_when_even_the_id_slice_collides():
    # Distinct ids can sanitize to one slice, so the slice alone is not enough.
    stems = _unique_stems(
        [
            {"artifact_id": "step/one", "filename": "plot"},
            {"artifact_id": "step:one", "filename": "plot"},
            {"artifact_id": "step one", "filename": "plot"},
        ]
    )
    assert sorted(stems.values()) == ["plot", "plot-step_one", "plot-step_one-2"]


def test_stems_are_assigned_across_both_buckets():
    # Images and documents land in the same directory, so a collision between the two
    # buckets is the same overwrite as one within either.
    #
    # Both descriptors are shapes the worker can actually emit: predict_delivery only
    # ever reports a passthrough source mime or image/png, and _predicted_filename
    # forces the .png suffix on the latter — so an image descriptor always carries a
    # .png name, and application/pdf passes through keeping its own.
    stems = _unique_stems(
        [
            png_descriptor("img", filename="report.png"),
            {"artifact_id": "doc", "delivered_mime": "application/pdf", "filename": "report.pdf"},
        ]
    )
    assert len(set(stems.values())) == 2


def test_two_plots_with_one_name_are_both_delivered():
    ops, _, nc = _ops(
        FakeWorker({"a1": (PNG, "image/png"), "a2": (PNG + b"-second", "image/png")}),
    )
    assert (
        ops.deliver_files(
            ENTRY,
            result(
                png_descriptor("a1", filename="plot.png"),
                png_descriptor("a2", filename="plot.png"),
            ),
        )
        == {}
    )
    puts = nc.of("PUT")
    # Two distinct paths, each carrying its own bytes: neither plot is lost.
    assert [put.url.path.rsplit("/", 1)[-1] for put in puts] == ["plot.png", "plot-a2.png"]
    assert [put.content for put in puts] == [PNG, PNG + b"-second"]


def test_names_do_not_shift_when_an_earlier_fetch_fails():
    # Stems are settled before any fetch, so a redelivery in which the first artifact
    # now succeeds cannot rename the second and upload it twice.
    descriptors = [
        png_descriptor("a1", filename="plot.png"),
        png_descriptor("a2", filename="plot.png"),
    ]
    ops, _, nc = _ops(FakeWorker(fail_ids=["a1"]))
    ops.deliver_files(ENTRY, result(*descriptors))
    assert [put.url.path.rsplit("/", 1)[-1] for put in nc.of("PUT")] == ["plot-a2.png"]
