"""``SlackOps.download_inputs``: the files a question referenced, from the entry.

Every ref is read off the persisted entry — the question's own ``sl_files`` and the
thread parent's ``sl_parent_files`` — because the retry drain and startup reconcile
call this long after the wire event is gone. Pinned here:

* the own and quoted buckets never bleed into each other;
* every ref lands in exactly one of ``files`` / ``skipped`` of its bucket, in order;
* a foreign host, a login page, an oversize file and any other failure each become a
  named skip note with a fixed reason — no URL, token or exception text in it;
* a ref whose declared size is over the cap is skipped before any request;
* an entry with no refs is exactly ``InputDownload()`` and makes no call;
* nothing raises.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from osprey.bridges.core import InputDownload
from osprey.bridges.slack.client import (
    FileTooLargeError,
    ForeignHostError,
    NotAFileError,
    SlackApiError,
    SlackAuthError,
)
from osprey.bridges.slack.events import SL_FILES, SL_PARENT_FILES
from osprey.bridges.slack.ops import (
    DOWNLOAD_FAILED_REASON,
    FOREIGN_HOST_REASON,
    MAX_DOWNLOAD_BYTES,
    NOT_A_FILE_REASON,
    TOO_LARGE_REASON,
)
from tests.bridges.slack._posting import RecordingClient, make_entry, make_ops

FILES_URL = "https://files.slack.com/files-pri/T0HOME-{id}/download/{name}"


def ref(
    file_id: str, name: str = "", *, mimetype: str = "text/plain", size: int = 10, url: str = ""
) -> dict[str, Any]:
    name = name or f"{file_id}.txt"
    return {
        "id": file_id,
        "name": name,
        "mimetype": mimetype,
        "size": size,
        "url": url or FILES_URL.format(id=file_id, name=name),
    }


class DownloadClient(RecordingClient):
    """A recording client whose ``download`` serves bytes or raises per URL."""

    def __init__(self, served: dict[str, bytes | BaseException] | None = None) -> None:
        super().__init__()
        self.served = served or {}
        self.downloads: list[tuple[str, int]] = []

    def download(self, url: str, max_bytes: int) -> bytes:
        self.downloads.append((url, max_bytes))
        outcome = self.served.get(url, b"payload:" + url.encode())
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def entry_with(own: list[Any] | None = None, parent: list[Any] | None = None) -> dict[str, Any]:
    entry = make_entry()
    entry[SL_FILES] = own if own is not None else []
    if parent is not None:
        entry[SL_PARENT_FILES] = parent
    return entry


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_an_entry_without_refs_is_the_empty_download_and_makes_no_call() -> None:
    client = DownloadClient()
    ops, _ = make_ops(client)
    assert ops.download_inputs(entry_with()) == InputDownload()
    assert ops.download_inputs(entry_with([], [])) == InputDownload()
    assert client.downloads == []
    assert client.calls == []


def test_own_files_land_in_the_own_bucket() -> None:
    first, second = ref("F1", "a.txt"), ref("F2", "plot.png", mimetype="image/png")
    client = DownloadClient({first["url"]: b"alpha", second["url"]: b"\x89PNG"})
    ops, _ = make_ops(client)
    download = ops.download_inputs(entry_with([first, second]))
    assert list(download.files) == [
        {"filename": "a.txt", "mime": "text/plain", "content_b64": b64(b"alpha"), "ingest": True},
        {
            "filename": "plot.png",
            "mime": "image/png",
            "content_b64": b64(b"\x89PNG"),
            "ingest": True,
        },
    ]
    assert not download.skipped and not download.quoted_files and not download.quoted_skipped
    assert [url for url, _ in client.downloads] == [first["url"], second["url"]]
    assert all(cap == MAX_DOWNLOAD_BYTES for _, cap in client.downloads)


def test_parent_files_land_in_the_quoted_bucket_only() -> None:
    quoted = ref("F9", "parent.csv", mimetype="text/csv")
    client = DownloadClient({quoted["url"]: b"q"})
    ops, _ = make_ops(client)
    download = ops.download_inputs(entry_with([], [quoted]))
    assert download.files == []
    assert download.skipped == []
    assert download.quoted_files == [
        {"filename": "parent.csv", "mime": "text/csv", "content_b64": b64(b"q"), "ingest": True}
    ]
    assert download.quoted_skipped == []


def test_the_two_buckets_keep_their_own_failures() -> None:
    own_bad = ref("F1", "mine.txt")
    quoted_bad = ref("F2", "theirs.txt")
    client = DownloadClient(
        {
            own_bad["url"]: NotAFileError("html"),
            quoted_bad["url"]: SlackApiError("HTTP 500", method="files.download"),
        }
    )
    ops, _ = make_ops(client)
    download = ops.download_inputs(entry_with([own_bad], [quoted_bad]))
    assert download.skipped == [{"filename": "mine.txt", "reason": NOT_A_FILE_REASON}]
    assert download.quoted_skipped == [{"filename": "theirs.txt", "reason": DOWNLOAD_FAILED_REASON}]
    assert download.files == [] and download.quoted_files == []


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (ForeignHostError("not files.slack.com"), FOREIGN_HOST_REASON),
        (NotAFileError("html login page"), NOT_A_FILE_REASON),
        (FileTooLargeError("over the cap"), TOO_LARGE_REASON),
        (SlackApiError("HTTP 404", method="files.download"), DOWNLOAD_FAILED_REASON),
        (SlackAuthError("revoked", error="token_revoked"), DOWNLOAD_FAILED_REASON),
        (RuntimeError("anything"), DOWNLOAD_FAILED_REASON),
    ],
)
def test_each_failure_maps_to_its_skip_reason(error: BaseException, reason: str) -> None:
    bad = ref("F1", "x.bin")
    ops, _ = make_ops(DownloadClient({bad["url"]: error}))
    download = ops.download_inputs(entry_with([bad]))
    assert download.skipped == [{"filename": "x.bin", "reason": reason}]
    assert download.files == []


def test_the_too_large_check_wins_over_its_not_a_file_parent() -> None:
    assert issubclass(FileTooLargeError, NotAFileError)
    bad = ref("F1")
    ops, _ = make_ops(DownloadClient({bad["url"]: FileTooLargeError("big")}))
    assert ops.download_inputs(entry_with([bad])).skipped[0]["reason"] == TOO_LARGE_REASON


def test_a_declared_size_over_the_cap_is_skipped_before_any_request() -> None:
    big = ref("F1", "huge.bin", size=MAX_DOWNLOAD_BYTES + 1)
    small = ref("F2", "ok.txt")
    client = DownloadClient()
    ops, _ = make_ops(client)
    download = ops.download_inputs(entry_with([big, small]))
    assert download.skipped == [{"filename": "huge.bin", "reason": TOO_LARGE_REASON}]
    assert [f["filename"] for f in download.files] == ["ok.txt"]
    assert [url for url, _ in client.downloads] == [small["url"]]


def test_a_failure_costs_only_its_own_file_and_order_is_kept() -> None:
    refs = [ref("F1", "one.txt"), ref("F2", "two.txt"), ref("F3", "three.txt")]
    client = DownloadClient({refs[1]["url"]: RuntimeError("x")})
    ops, _ = make_ops(client)
    download = ops.download_inputs(entry_with(refs))
    assert [f["filename"] for f in download.files] == ["one.txt", "three.txt"]
    assert [s["filename"] for s in download.skipped] == ["two.txt"]


def test_skip_reasons_carry_no_url_token_or_exception_text() -> None:
    secret = "xoxb-1111-2222-bot-secret"
    bad = ref("F1", "x.txt", url=f"https://evil.example/{secret}")
    error = ForeignHostError(f"https://evil.example/{secret}")
    ops, _ = make_ops(DownloadClient({bad["url"]: error}))
    reason = ops.download_inputs(entry_with([bad])).skipped[0]["reason"]
    assert secret not in reason
    assert "evil.example" not in reason


def test_a_ref_without_a_url_is_skipped_without_a_request() -> None:
    no_url = {"id": "F1", "name": "lost.txt", "mimetype": "text/plain", "size": 3}
    client = DownloadClient()
    ops, _ = make_ops(client)
    download = ops.download_inputs(entry_with([no_url]))
    assert download.skipped == [{"filename": "lost.txt", "reason": DOWNLOAD_FAILED_REASON}]
    assert client.downloads == []


def test_non_mapping_refs_are_ignored() -> None:
    good = ref("F1", "ok.txt")
    client = DownloadClient()
    ops, _ = make_ops(client)
    download = ops.download_inputs(entry_with(["junk", 7, None, good]))
    assert [f["filename"] for f in download.files] == ["ok.txt"]
    assert download.skipped == []


def test_a_malformed_ref_list_is_the_empty_download() -> None:
    ops, _ = make_ops(DownloadClient())
    entry = make_entry()
    entry[SL_FILES] = "not a list"
    entry[SL_PARENT_FILES] = {"id": "F1"}
    assert ops.download_inputs(entry) == InputDownload()


def test_a_nameless_ref_falls_back_to_its_id_then_a_default() -> None:
    by_id = {"id": "F1", "url": FILES_URL.format(id="F1", name="x")}
    anonymous = {"url": FILES_URL.format(id="F2", name="y")}
    ops, _ = make_ops(DownloadClient())
    names = [f["filename"] for f in ops.download_inputs(entry_with([by_id, anonymous])).files]
    assert names[0] == "F1"
    assert names[1]


def test_a_missing_mimetype_ships_as_empty_mime() -> None:
    bare = {"id": "F1", "name": "n.dat", "url": FILES_URL.format(id="F1", name="n.dat")}
    ops, _ = make_ops(DownloadClient())
    assert ops.download_inputs(entry_with([bare])).files[0]["mime"] == ""


def test_a_client_without_download_still_never_raises() -> None:
    ops, _ = make_ops(RecordingClient())
    download = ops.download_inputs(entry_with([ref("F1", "a.txt")]))
    assert download.skipped == [{"filename": "a.txt", "reason": DOWNLOAD_FAILED_REASON}]
