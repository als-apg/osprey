"""TalkClient's file surface: DAV download, MKCOL+PUT upload, and the room share.

Every test drives a :class:`httpx.MockTransport` — nothing here opens a socket.
Four properties get the most attention, because each one is a way to break a live
bridge that no unit test would otherwise catch:

* the download is **capped while streaming**, so an unexpectedly large attachment
  cannot be buffered whole even when the server declares no ``Content-Length``;
* ``MKCOL`` is **not recursive**, so every level of the upload directory is created
  in turn and an "already exists" answer is the normal steady state, not a failure;
* the upload directory contains a **space**, so every path segment is
  percent-encoded individually — a raw space would truncate the request target;
* a duplicate share is tolerated but a *genuine* ``403`` is not, since "the bot may
  not share into that room" must never read as a successful delivery.

The suite also pins that all three verbs propagate transport failures and add no
retry: swallow/degrade decisions belong to the ``ChannelOps`` members above them.
"""

import base64

import httpx
import pytest

from osprey.bridges.core import MAX_DOC_BYTES
from osprey.bridges.nextcloud_talk import NextcloudBridgeConfig, TalkApiError, TalkClient
from osprey.bridges.nextcloud_talk.client import (
    MAX_DOWNLOAD_BYTES,
    SHARE_TYPE_ROOM,
    UPLOAD_ROOT,
    DavTooLargeError,
    upload_dir,
)

CFG = NextcloudBridgeConfig(
    base_url="https://cloud.example.org",
    bot_account="osprey-bot",
    app_password="app-pw",
    rooms=("roomA",),
)

EXPECTED_AUTH = "Basic " + base64.b64encode(b"osprey-bot:app-pw").decode()

DAV_ROOT = "/remote.php/dav/files/osprey-bot"
SHARES_PATH = "/ocs/v2.php/apps/files_sharing/api/v1/shares"

# The upload layout for one run, and the encoded form of its first segment.
RUN_DIR = upload_dir("roomA", "run-1")
ENCODED_ROOT = "OSPREY%20artifacts"


def _ok(request):
    """A Nextcloud that accepts every file-surface call."""
    if request.method == "GET":
        return httpx.Response(200, content=b"png-bytes")
    if request.method == "MKCOL":
        return httpx.Response(201)
    if request.method == "PUT":
        return httpx.Response(201)
    return httpx.Response(
        200,
        json={"ocs": {"meta": {"status": "ok", "statuscode": 200}, "data": {"id": 5}}},
    )


def _client(handler=_ok):
    """A TalkClient over a MockTransport running ``handler``."""
    return TalkClient(CFG, client=httpx.Client(transport=httpx.MockTransport(handler)))


def _recording(handler=_ok):
    """``(client, requests)`` — as :func:`_client`, recording every request."""
    seen: list[httpx.Request] = []

    def recorder(request):
        seen.append(request)
        return handler(request)

    return _client(recorder), seen


def _ocs_failure(status, message):
    """An OCS v2 failure envelope, the shape a rejected share comes back as."""
    return httpx.Response(
        status,
        json={"ocs": {"meta": {"status": "failure", "statuscode": status, "message": message}}},
    )


# --- dav_download ----------------------------------------------------------------


def test_download_returns_bytes_from_the_bots_own_tree():
    client, seen = _recording()

    assert client.dav_download("Talk/shot.png") == b"png-bytes"

    (request,) = seen
    assert request.method == "GET"
    # The account in the URL is the bot's, from config — never anything the
    # caller passed, so a path from a message payload cannot address another
    # user's files.
    assert request.url.path == f"{DAV_ROOT}/Talk/shot.png"
    assert request.headers["Authorization"] == EXPECTED_AUTH


def test_download_does_not_send_the_ocs_headers():
    """DAV is not OCS: asking for ``application/json`` on a binary GET is wrong."""
    client, seen = _recording()

    client.dav_download("Talk/shot.png")

    (request,) = seen
    assert "OCS-APIRequest" not in request.headers
    assert request.headers.get("Accept", "*/*") != "application/json"


def test_download_quotes_each_segment_of_a_spaced_path():
    client, seen = _recording()

    client.dav_download(f"{UPLOAD_ROOT}/roomA/my plot.png")

    (request,) = seen
    target = str(request.url)
    assert f"{ENCODED_ROOT}/roomA/my%20plot.png" in target
    assert " " not in target


def test_download_quotes_a_bot_account_with_a_space():
    cfg = NextcloudBridgeConfig(
        base_url="https://cloud.example.org",
        bot_account="osprey bot",
        app_password="app-pw",
    )
    seen: list[httpx.Request] = []

    def recorder(request):
        seen.append(request)
        return _ok(request)

    client = TalkClient(cfg, client=httpx.Client(transport=httpx.MockTransport(recorder)))
    client.dav_download("Talk/shot.png")

    (request,) = seen
    assert "/remote.php/dav/files/osprey%20bot/Talk/shot.png" in str(request.url)


def test_download_default_cap_comes_from_the_core_budget():
    """Pinned: the cap is the engine's document budget, not a local invention."""
    assert MAX_DOWNLOAD_BYTES == MAX_DOC_BYTES


def test_download_rejects_a_declared_length_over_the_cap():
    def big(request):
        return httpx.Response(200, content=b"x" * 64)

    with pytest.raises(DavTooLargeError, match="over the 32-byte cap"):
        _client(big).dav_download("Talk/big.bin", max_bytes=32)


def test_download_abandons_a_stream_that_passes_the_cap():
    """No declared length, so only the streaming guard can stop this one."""
    sent: list[int] = []

    def chunked(request):
        def body():
            for _ in range(100):
                sent.append(1)
                yield b"x" * 16

        return httpx.Response(200, content=body())

    with pytest.raises(DavTooLargeError, match="download abandoned"):
        _client(chunked).dav_download("Talk/big.bin", max_bytes=32)

    # Abandoned early: a caller must not pay to read a file it will reject.
    assert len(sent) < 100


def test_download_accepts_a_body_exactly_at_the_cap():
    def exact(request):
        return httpx.Response(200, content=b"x" * 32)

    assert _client(exact).dav_download("Talk/edge.bin", max_bytes=32) == b"x" * 32


def test_download_too_large_is_a_talk_api_error():
    """One ``except TalkApiError`` catches both halves of "not usable bytes"."""
    assert issubclass(DavTooLargeError, TalkApiError)


@pytest.mark.parametrize("path", ["", "/", "../other/secrets.txt", "Talk/../../etc/passwd"])
def test_download_rejects_empty_and_traversing_paths(path):
    with pytest.raises(ValueError):
        _client().dav_download(path)


def test_download_rejects_a_non_positive_cap():
    with pytest.raises(ValueError, match="max_bytes"):
        _client().dav_download("Talk/shot.png", max_bytes=0)


def test_download_raises_on_a_missing_file():
    def gone(request):
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        _client(gone).dav_download("Talk/gone.png")


def test_download_propagates_a_transport_failure_without_retrying():
    attempts: list[int] = []

    def broken(request):
        attempts.append(1)
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(httpx.ConnectError):
        _client(broken).dav_download("Talk/shot.png")
    assert len(attempts) == 1


# --- dav_mkcol_put ---------------------------------------------------------------


def test_upload_creates_each_directory_level_then_puts():
    client, seen = _recording()

    path = client.dav_mkcol_put(RUN_DIR, "plot.png", b"bytes")

    assert path == f"{UPLOAD_ROOT}/roomA/run-1/plot.png"
    # ``raw_path``, not ``path``: httpx decodes the latter, which would hide
    # whether the space in UPLOAD_ROOT was ever encoded on the wire.
    # MKCOL is not recursive, so every level is created, outermost first.
    assert [(r.method, r.url.raw_path.decode()) for r in seen] == [
        ("MKCOL", f"{DAV_ROOT}/{ENCODED_ROOT}"),
        ("MKCOL", f"{DAV_ROOT}/{ENCODED_ROOT}/roomA"),
        ("MKCOL", f"{DAV_ROOT}/{ENCODED_ROOT}/roomA/run-1"),
        ("PUT", f"{DAV_ROOT}/{ENCODED_ROOT}/roomA/run-1/plot.png"),
    ]
    assert seen[-1].content == b"bytes"
    assert all(r.headers["Authorization"] == EXPECTED_AUTH for r in seen)


def test_upload_quotes_every_segment_including_the_filename():
    client, seen = _recording()

    client.dav_mkcol_put(RUN_DIR, "orbit response.png", b"bytes")

    targets = [str(r.url) for r in seen]
    assert all(" " not in target for target in targets)
    assert targets[-1].endswith(f"{ENCODED_ROOT}/roomA/run-1/orbit%20response.png")


def test_upload_tolerates_collections_that_already_exist():
    """The steady state: only the very first upload into a tree creates anything."""

    def exists(request):
        if request.method == "MKCOL":
            return httpx.Response(405)
        return httpx.Response(204)

    client, seen = _recording(exists)

    assert client.dav_mkcol_put(RUN_DIR, "plot.png", b"bytes")
    assert [r.method for r in seen] == ["MKCOL", "MKCOL", "MKCOL", "PUT"]


def test_upload_raises_when_a_mkcol_really_fails_and_skips_the_put():
    """A ``409`` is a broken parent chain, not "already there" — it must surface."""

    def conflict(request):
        if request.method == "MKCOL":
            return httpx.Response(409)
        return httpx.Response(201)

    client, seen = _recording(conflict)

    with pytest.raises(httpx.HTTPStatusError):
        client.dav_mkcol_put(RUN_DIR, "plot.png", b"bytes")
    # Stopped at the first level; no bytes were sent to a directory that is not there.
    assert [r.method for r in seen] == ["MKCOL"]


def test_upload_raises_when_the_put_fails():
    def put_fails(request):
        if request.method == "PUT":
            return httpx.Response(507)
        return httpx.Response(201)

    with pytest.raises(httpx.HTTPStatusError):
        _client(put_fails).dav_mkcol_put(RUN_DIR, "plot.png", b"bytes")


def test_upload_propagates_a_transport_failure_without_retrying():
    attempts: list[int] = []

    def broken(request):
        attempts.append(1)
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(httpx.ConnectError):
        _client(broken).dav_mkcol_put(RUN_DIR, "plot.png", b"bytes")
    assert len(attempts) == 1


@pytest.mark.parametrize("name", ["", "nested/plot.png", "..", "."])
def test_upload_rejects_a_filename_that_is_not_one_segment(name):
    with pytest.raises(ValueError):
        _client().dav_mkcol_put(RUN_DIR, name, b"bytes")


@pytest.mark.parametrize("directory", ["", "../elsewhere", f"{UPLOAD_ROOT}/../.."])
def test_upload_rejects_an_empty_or_traversing_directory(directory):
    with pytest.raises(ValueError):
        _client().dav_mkcol_put(directory, "plot.png", b"bytes")


def test_upload_creates_one_level_for_a_single_segment_directory():
    client, seen = _recording()

    client.dav_mkcol_put("Talk", "plot.png", b"bytes")

    assert [r.method for r in seen] == ["MKCOL", "PUT"]


# --- share_to_room ---------------------------------------------------------------


def test_share_posts_a_room_share_for_the_uploaded_path():
    client, seen = _recording()

    client.share_to_room(f"{RUN_DIR}/plot.png", "roomA")

    (request,) = seen
    assert request.method == "POST"
    assert request.url.path == SHARES_PATH
    assert request.headers["OCS-APIRequest"] == "true"
    assert request.headers["Authorization"] == EXPECTED_AUTH
    # Form-encoded, not JSON: the space rides as a form escape, and the path is
    # absolute inside the bot's own tree.
    body = request.content.decode()
    assert "path=%2FOSPREY" in body.replace("+", "%20")
    assert f"shareType={SHARE_TYPE_ROOM}" in body
    assert "shareWith=roomA" in body


def test_share_type_is_the_room_share():
    """Pinned: never ``3`` (public link) — no world-readable URL is ever minted."""
    assert SHARE_TYPE_ROOM == 10


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (403, "Path is already shared with this room"),
        (400, "Path is already shared with this room"),
        (409, "Sharing failed, this item is already shared with roomA"),
        (403, "This share already exists"),
        (403, "PATH IS ALREADY SHARED with this room"),
    ],
)
def test_share_tolerates_a_duplicate(status, message):
    """Redelivering the same artifact must be a no-op, not a failure."""

    def duplicate(request):
        return _ocs_failure(status, message)

    _client(duplicate).share_to_room(f"{RUN_DIR}/plot.png", "roomA")


def test_share_raises_on_a_genuine_permission_failure():
    """A bare 403 is also how "the bot may not share here" arrives."""

    def denied(request):
        return _ocs_failure(403, "You are not allowed to share this file")

    with pytest.raises(httpx.HTTPStatusError):
        _client(denied).share_to_room(f"{RUN_DIR}/plot.png", "roomA")


def test_share_raises_when_a_rejection_is_not_an_ocs_envelope():
    """An HTML error page from a proxy must not be read as a tolerated duplicate."""

    def html(request):
        return httpx.Response(403, text="<html>already shared</html>")

    with pytest.raises(httpx.HTTPStatusError):
        _client(html).share_to_room(f"{RUN_DIR}/plot.png", "roomA")


def test_share_raises_on_a_server_error():
    def broken(request):
        return httpx.Response(500)

    with pytest.raises(httpx.HTTPStatusError):
        _client(broken).share_to_room(f"{RUN_DIR}/plot.png", "roomA")


def test_share_propagates_a_transport_failure_without_retrying():
    attempts: list[int] = []

    def broken(request):
        attempts.append(1)
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(httpx.ConnectError):
        _client(broken).share_to_room(f"{RUN_DIR}/plot.png", "roomA")
    assert len(attempts) == 1


@pytest.mark.parametrize("path", ["", "../other/secrets.txt"])
def test_share_rejects_empty_and_traversing_paths(path):
    with pytest.raises(ValueError):
        _client().share_to_room(path, "roomA")


# --- upload layout ---------------------------------------------------------------


def test_upload_dir_is_room_then_run():
    assert upload_dir("roomA", "run-1") == "OSPREY artifacts/roomA/run-1"


def test_upload_root_is_operator_readable():
    """The space is deliberate; it is why paths are built from encoded segments."""
    assert " " in UPLOAD_ROOT


@pytest.mark.parametrize(
    ("room", "run_id"),
    [("", "run-1"), ("roomA", ""), ("../roomB", "run-1"), ("roomA", "run/1"), ("roomA", "..")],
)
def test_upload_dir_rejects_components_that_are_not_one_segment(room, run_id):
    with pytest.raises(ValueError):
        upload_dir(room, run_id)
