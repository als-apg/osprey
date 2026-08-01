"""NextcloudTalkOps.download_inputs: inbound attachments, from the persisted entry alone.

Every test here builds its ``entry`` as a **plain dict**. Nothing in this module imports
``InboundEvent``, ``parse_event``, or any other wire-event type — deliberately, because
that is the property under test: by the time the retry drain or a post-restart reconcile
calls this member, the wire event no longer exists, so an implementation that reached for
one would pass a live-path test and silently lose every attachment after a restart. The
strongest form of that proof is
:func:`test_only_the_two_file_ref_keys_are_read_from_the_entry`, which hands in an entry
that records each key read and asserts the set is exactly the two file-ref keys.

The other load-bearing property is **provenance separation**: a failure on the user's own
attachment must land in ``skipped`` and one on a quoted message's in ``quoted_skipped``,
since the engine reports them in different places and a user needs to know which file was
dropped.
"""

import base64
from collections.abc import Mapping

import httpx
import pytest

from osprey.bridges.core import RESERVED_ENTRY_KEYS, CoreConfig, InputDownload
from osprey.bridges.nextcloud_talk import NextcloudBridgeConfig, TalkClient
from osprey.bridges.nextcloud_talk.client import MAX_DOWNLOAD_BYTES
from osprey.bridges.nextcloud_talk.events import QUOTED_FILES_KEY
from osprey.bridges.nextcloud_talk.ops import (
    DEFAULT_INPUT_FILENAME,
    NC_FILES,
    NO_PATH_REASON,
    TOO_LARGE_REASON,
    UNAVAILABLE_REASON,
    NextcloudTalkOps,
)

CFG = NextcloudBridgeConfig(
    base_url="https://cloud.example.org",
    bot_account="osprey-bot",
    app_password="app-pw",
    rooms=("roomA",),
    core=CoreConfig(worker_url="http://work:9190", dispatch_worker_token="work-token"),
)

PNG_BYTES = b"\x89PNG\r\n\x1a\nbeam-profile"
CSV_BYTES = b"bpm,x,y\nBPM1,0.1,-0.2\n"


def ref(path, name="photo.png", *, file_id="12345", size=1024, mimetype="image/png"):
    """One inbound file reference, shaped exactly as the parse step persists it."""
    return {"id": file_id, "name": name, "path": path, "size": size, "mimetype": mimetype}


def entry(own=(), quoted=()):
    """A persisted entry carrying the two file-ref buckets — a plain dict, no wire event."""
    return {
        "nc_room": "roomA",
        "nc_message_id": 41,
        "history_key": "nextcloud:roomA",
        "sender_id": "alice",
        "text": "what does this show?",
        NC_FILES: list(own),
        QUOTED_FILES_KEY: list(quoted),
    }


class WatchedEntry(Mapping):
    """A persisted entry that records every key read from it."""

    def __init__(self, data):
        self._data = dict(data)
        self.reads: list[str] = []

    def __getitem__(self, key):
        self.reads.append(key)
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class FakeDav:
    """The bot's DAV tree: serves per-path bytes, records every request."""

    def __init__(self, files=None, *, oversize=(), exc=None, status=404):
        self.files = dict(files or {})
        self.oversize = set(oversize)
        self.exc = exc
        self.status = status
        self.requests: list[httpx.Request] = []

    @property
    def paths(self) -> list[str]:
        """The in-tree path of every request, in order (decoded)."""
        return [request.url.path.split("/osprey-bot/", 1)[-1] for request in self.requests]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.exc is not None:
            raise self.exc
        path = request.url.path.split("/osprey-bot/", 1)[-1]
        if path in self.oversize:
            return httpx.Response(200, content=b"x" * (MAX_DOWNLOAD_BYTES + 1))
        data = self.files.get(path)
        if data is None:
            return httpx.Response(self.status, text="no such file")
        return httpx.Response(200, content=data)


def _ops(dav=None):
    """``(ops, dav)`` wired to a MockTransport-backed DAV tree."""
    dav = dav if dav is not None else FakeDav({"Talk/photo.png": PNG_BYTES})
    ops = NextcloudTalkOps(
        CFG,
        TalkClient(CFG, client=httpx.Client(transport=httpx.MockTransport(dav.handler))),
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404))),
    )
    return ops, dav


def decoded(payload_file) -> bytes:
    """The bytes a worker-payload file dict carries."""
    return base64.b64decode(payload_file["content_b64"])


# ==========================================================================
# From the persisted entry alone
# ==========================================================================


def test_files_are_fetched_from_the_persisted_entry_alone():
    # No InboundEvent exists anywhere in this module: the entry dict is the only input,
    # which is exactly the shape the drain and reconcile paths have after a restart.
    ops, dav = _ops()
    result = ops.download_inputs(entry(own=[ref("Talk/photo.png")]))

    assert dav.paths == ["Talk/photo.png"]
    assert len(result.files) == 1
    assert decoded(result.files[0]) == PNG_BYTES
    assert result.skipped == []


def test_only_the_two_file_ref_keys_are_read_from_the_entry():
    # The strong form of the no-wire-event property: nothing else on the entry is even
    # looked at, so there is no room for a raw event or a live-path shortcut to hide.
    ops, _ = _ops()
    watched = WatchedEntry(entry(own=[ref("Talk/photo.png")]))
    ops.download_inputs(watched)
    assert set(watched.reads) == {NC_FILES, QUOTED_FILES_KEY}


def test_the_dav_path_is_resolved_inside_the_bots_own_tree():
    ops, dav = _ops(FakeDav({"Talk/beam scan.png": PNG_BYTES}))
    result = ops.download_inputs(entry(own=[ref("Talk/beam scan.png")]))
    assert dav.requests[0].url.raw_path == b"/remote.php/dav/files/osprey-bot/Talk/beam%20scan.png"
    assert len(result.files) == 1


# ==========================================================================
# The worker payload shape
# ==========================================================================


def test_a_fetched_file_is_shaped_for_the_worker_payload():
    ops, _ = _ops(FakeDav({"Talk/photo.png": PNG_BYTES}))
    result = ops.download_inputs(entry(own=[ref("Talk/photo.png", "beam.png")]))
    assert result.files == [
        {
            "filename": "beam.png",
            "mime": "image/png",
            "content_b64": base64.b64encode(PNG_BYTES).decode("ascii"),
            # Fresh input rather than a replayed prior artifact.
            "ingest": True,
        }
    ]


def test_the_mime_comes_from_the_reference():
    ops, _ = _ops(FakeDav({"Talk/data.csv": CSV_BYTES}))
    result = ops.download_inputs(entry(own=[ref("Talk/data.csv", "data.csv", mimetype="text/csv")]))
    assert result.files[0]["mime"] == "text/csv"
    assert decoded(result.files[0]) == CSV_BYTES


@pytest.mark.parametrize("mimetype", [None, 42, "", {}])
def test_a_missing_mime_ships_as_an_empty_string(mimetype):
    # Never None: the payload key is always present and always a string.
    ops, _ = _ops(FakeDav({"Talk/f": PNG_BYTES}))
    result = ops.download_inputs(entry(own=[ref("Talk/f", mimetype=mimetype)]))
    assert result.files[0]["mime"] == ""


@pytest.mark.parametrize(
    ("name", "file_id", "expected"),
    [
        ("beam.png", "12345", "beam.png"),
        (None, "12345", "12345"),  # falls back to Talk's file id
        ("", "12345", "12345"),
        (None, None, DEFAULT_INPUT_FILENAME),
        ("", "", DEFAULT_INPUT_FILENAME),
        ("with\nnewline.png", "1", "withnewline.png"),  # CR/LF stripped
        ("x" * 300, "1", "x" * 200),  # length bounded
    ],
)
def test_the_shipped_filename_falls_back_through_name_then_id(name, file_id, expected):
    ops, _ = _ops(FakeDav({"Talk/f": PNG_BYTES}))
    result = ops.download_inputs(entry(own=[ref("Talk/f", name, file_id=file_id)]))
    assert result.files[0]["filename"] == expected


def test_several_files_keep_the_order_talk_listed_them():
    ops, _ = _ops(FakeDav({"Talk/a": b"a", "Talk/b": b"b", "Talk/c": b"c"}))
    result = ops.download_inputs(
        entry(own=[ref("Talk/a", "a"), ref("Talk/b", "b"), ref("Talk/c", "c")])
    )
    assert [item["filename"] for item in result.files] == ["a", "b", "c"]


# ==========================================================================
# Provenance separation
# ==========================================================================


def test_a_quoted_message_file_lands_in_the_quoted_bucket():
    ops, _ = _ops(FakeDav({"Talk/quoted.png": PNG_BYTES}))
    result = ops.download_inputs(entry(quoted=[ref("Talk/quoted.png", "quoted.png")]))
    assert result.files == []
    assert result.skipped == []
    assert [item["filename"] for item in result.quoted_files] == ["quoted.png"]
    assert result.quoted_skipped == []


def test_both_buckets_are_populated_independently():
    ops, _ = _ops(FakeDav({"Talk/own.png": PNG_BYTES, "Talk/quoted.csv": CSV_BYTES}))
    result = ops.download_inputs(
        entry(
            own=[ref("Talk/own.png", "own.png")],
            quoted=[ref("Talk/quoted.csv", "quoted.csv", mimetype="text/csv")],
        )
    )
    assert [item["filename"] for item in result.files] == ["own.png"]
    assert [item["filename"] for item in result.quoted_files] == ["quoted.csv"]
    assert decoded(result.files[0]) == PNG_BYTES
    assert decoded(result.quoted_files[0]) == CSV_BYTES


def test_a_failure_is_reported_in_its_own_bucket_only():
    # The user attached one file and quoted another; only the quoted one is gone. Putting
    # that note in the own bucket would tell them the wrong file was lost.
    ops, _ = _ops(FakeDav({"Talk/own.png": PNG_BYTES}))
    result = ops.download_inputs(
        entry(
            own=[ref("Talk/own.png", "own.png")],
            quoted=[ref("Talk/gone.png", "gone.png")],
        )
    )
    assert [item["filename"] for item in result.files] == ["own.png"]
    assert result.skipped == []
    assert result.quoted_files == []
    assert result.quoted_skipped == [{"filename": "gone.png", "reason": UNAVAILABLE_REASON}]


def test_an_own_file_failure_stays_out_of_the_quoted_bucket():
    ops, _ = _ops(FakeDav({"Talk/quoted.png": PNG_BYTES}))
    result = ops.download_inputs(
        entry(own=[ref("Talk/gone.png", "gone.png")], quoted=[ref("Talk/quoted.png", "q.png")])
    )
    assert result.skipped == [{"filename": "gone.png", "reason": UNAVAILABLE_REASON}]
    assert result.quoted_skipped == []
    assert [item["filename"] for item in result.quoted_files] == ["q.png"]


# ==========================================================================
# Per-file failures become skip notes, never exceptions
# ==========================================================================


def test_a_missing_file_becomes_a_skip_note():
    ops, dav = _ops(FakeDav({}))
    result = ops.download_inputs(entry(own=[ref("Talk/deleted.png", "deleted.png")]))
    assert dav.requests  # it tried
    assert result.files == []
    assert result.skipped == [{"filename": "deleted.png", "reason": UNAVAILABLE_REASON}]


@pytest.mark.parametrize("status", [401, 403, 404, 409, 500, 503])
def test_any_dav_error_status_becomes_a_skip_note(status):
    ops, _ = _ops(FakeDav({}, status=status))
    result = ops.download_inputs(entry(own=[ref("Talk/f", "f.png")]))
    assert result.skipped == [{"filename": "f.png", "reason": UNAVAILABLE_REASON}]


def test_an_oversize_file_becomes_a_skip_note_naming_the_cap():
    ops, _ = _ops(FakeDav({}, oversize={"Talk/huge.tif"}))
    result = ops.download_inputs(entry(own=[ref("Talk/huge.tif", "huge.tif")]))
    assert result.files == []
    assert result.skipped == [{"filename": "huge.tif", "reason": TOO_LARGE_REASON}]
    assert "10 MiB" in TOO_LARGE_REASON


def test_a_transport_failure_becomes_a_skip_note():
    ops, _ = _ops(FakeDav(exc=httpx.ConnectError("cloud unreachable")))
    result = ops.download_inputs(entry(own=[ref("Talk/f", "f.png")]))
    assert result.skipped == [{"filename": "f.png", "reason": UNAVAILABLE_REASON}]


@pytest.mark.parametrize("path", [None, "", 42, {}, "   "])
def test_a_reference_without_a_usable_path_becomes_a_skip_note(path):
    ops, dav = _ops()
    result = ops.download_inputs(entry(own=[{**ref("x", "orphan.png"), "path": path}]))
    assert result.skipped == [{"filename": "orphan.png", "reason": NO_PATH_REASON}]
    # Nothing was even attempted — there was nothing to ask for.
    assert dav.requests == []


def test_one_failing_file_does_not_cost_its_siblings():
    ops, _ = _ops(FakeDav({"Talk/a": b"a", "Talk/c": b"c"}))
    result = ops.download_inputs(
        entry(own=[ref("Talk/a", "a"), ref("Talk/b", "b"), ref("Talk/c", "c")])
    )
    assert [item["filename"] for item in result.files] == ["a", "c"]
    assert result.skipped == [{"filename": "b", "reason": UNAVAILABLE_REASON}]


def test_every_reference_is_accounted_for_exactly_once():
    # A silently absent attachment is the failure mode this rules out: refs in == files
    # plus skips out, always.
    ops, _ = _ops(FakeDav({"Talk/a": b"a"}, oversize={"Talk/big"}))
    refs = [ref("Talk/a", "a"), ref("Talk/big", "big"), ref("Talk/gone", "gone")]
    result = ops.download_inputs(entry(own=refs))
    assert len(result.files) + len(result.skipped) == len(refs)


# ==========================================================================
# Empty and malformed reference lists
# ==========================================================================


def test_no_references_is_a_strict_no_op():
    # Byte-identical to the text-only path: the engine's fold must see the default.
    ops, dav = _ops()
    assert ops.download_inputs(entry()) == InputDownload()
    assert dav.requests == []


def test_an_entry_without_the_ref_keys_at_all_is_a_strict_no_op():
    ops, dav = _ops()
    assert ops.download_inputs({"nc_room": "roomA"}) == InputDownload()
    assert dav.requests == []


@pytest.mark.parametrize("refs", [None, "Talk/photo.png", 7, {"path": "Talk/photo.png"}])
def test_a_malformed_ref_list_is_a_strict_no_op(refs):
    # A non-list would otherwise iterate into characters or dict keys and fabricate refs.
    ops, dav = _ops()
    assert ops.download_inputs({NC_FILES: refs, QUOTED_FILES_KEY: refs}) == InputDownload()
    assert dav.requests == []


@pytest.mark.parametrize("junk", [None, 42, "Talk/photo.png", ["nested"]])
def test_non_mapping_entries_in_a_ref_list_are_dropped(junk):
    ops, _ = _ops(FakeDav({"Talk/photo.png": PNG_BYTES}))
    result = ops.download_inputs(entry(own=[junk, ref("Talk/photo.png", "good.png")]))
    assert [item["filename"] for item in result.files] == ["good.png"]
    assert result.skipped == []


# ==========================================================================
# No capability or health probing
# ==========================================================================


def test_downloading_touches_only_the_bots_dav_tree():
    ops, dav = _ops(FakeDav({"Talk/a": b"a", "Talk/b": b"b"}))
    ops.download_inputs(entry(own=[ref("Talk/a", "a")], quoted=[ref("Talk/b", "b")]))

    paths = [request.url.path for request in dav.requests]
    assert all(path.startswith("/remote.php/dav/files/osprey-bot/") for path in paths)
    assert not any("health" in path for path in paths)
    assert not any("capabilities" in path for path in paths)
    assert {request.method for request in dav.requests} == {"GET"}


def test_the_download_is_unconditional_and_negotiates_no_budget():
    # Whether the bytes ship is the engine's memoized probe to decide; the adapter always
    # downloads, so nothing here consults the pair.
    ops, dav = _ops(FakeDav({"Talk/a": b"a" * 5000}))
    result = ops.download_inputs(entry(own=[ref("Talk/a", "a", size=5000)]))
    assert len(dav.requests) == 1
    assert decoded(result.files[0]) == b"a" * 5000


# ==========================================================================
# Entry-key hygiene
# ==========================================================================


def test_the_file_ref_keys_are_nc_prefixed_and_do_not_collide_with_the_engines():
    assert {NC_FILES, QUOTED_FILES_KEY}.isdisjoint(RESERVED_ENTRY_KEYS)
    assert NC_FILES != QUOTED_FILES_KEY
    assert all(key.startswith("nc_") for key in (NC_FILES, QUOTED_FILES_KEY))
