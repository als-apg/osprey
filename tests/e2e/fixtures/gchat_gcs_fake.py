"""A fake Google Cloud Storage server: upload, then public GET.

What it is for
--------------
The Google Chat bridge publishes a run's artifacts to a bucket and posts the
resulting **public URL** into the room; a later turn fetches that URL back when
the engine re-injects a prior artifact. Both halves — the upload and the
anonymous read — are exercised here against a real HTTP server rather than a
mock, so the bytes and the content type actually make the round trip.

Surface implemented
-------------------
``PUT  /{bucket}/{object}``                          direct upload
``POST /upload/storage/v1/b/{bucket}/o?name=...``    JSON-API media upload
``GET  /{bucket}/{object}``                          public read (no auth)
``HEAD /{bucket}/{object}``                          public read, headers only

The public read path is deliberately the same shape as the real one
(``https://storage.googleapis.com/{bucket}/{object}``), so a URL produced against
Google's host maps onto this fake by swapping the origin — see
:meth:`FakeGcsServer.local_url`.

The public-URL trap
-------------------
The bridge builds its public URLs itself as
``https://storage.googleapis.com/{bucket}/{name}`` — a literal, not something the
storage client hands back. So a URL stamped into history by product code points
at **Google**, not at this fake, and a test that fetches it without translating
would leave the hermetic world (and get a 404 from the real internet). Use
:meth:`local_url` on anything the product produced. The blobs handed out by
:meth:`FakeGcsServer.client` report this fake's own URL, so anything that asks
the client for a URL stays local by construction.
"""

from __future__ import annotations

import re
import threading
import urllib.parse
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx

from .gchat_fake_http import FakeHttpService, _FakeHandler

__all__ = ["FakeGcsServer", "StoredObject", "PUBLIC_HOST"]

PUBLIC_HOST = "https://storage.googleapis.com"
"""The real public-read origin the bridge hardcodes. See the module docstring."""

_JSON_UPLOAD = re.compile(r"^/upload/storage/v1/b/(?P<bucket>[^/]+)/o$")
_OBJECT = re.compile(r"^/(?P<bucket>[^/]+)/(?P<obj>.+)$")

HTTP_TIMEOUT = 10.0


@dataclass(frozen=True)
class StoredObject:
    """One uploaded object."""

    bucket: str
    name: str
    data: bytes
    content_type: str

    @property
    def size(self) -> int:
        return len(self.data)


class _GcsHandler(_FakeHandler):
    fake: ClassVar[FakeGcsServer]

    def do_PUT(self) -> None:  # noqa: N802 - http.server API
        path, _ = self.split_path()
        match = _OBJECT.match(path)
        if match is None:
            self.respond_error(404, f"no such route: PUT {path}")
            return
        stored = self.fake._store(
            match.group("bucket"),
            urllib.parse.unquote(match.group("obj")),
            self.read_body(),
            self.headers.get("Content-Type", "application/octet-stream"),
        )
        self.respond_json(200, _metadata(stored))

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        path, params = self.split_path()
        match = _JSON_UPLOAD.match(path)
        if match is None:
            self.respond_error(404, f"no such route: POST {path}")
            return
        name = params.get("name")
        if not name:
            self.respond_error(
                400, "media upload requires a ?name= object name", api_status="INVALID_ARGUMENT"
            )
            return
        stored = self.fake._store(
            match.group("bucket"),
            name,
            self.read_body(),
            self.headers.get("Content-Type", "application/octet-stream"),
        )
        self.respond_json(200, _metadata(stored))

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self._read(write_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - http.server API
        self._read(write_body=False)

    def _read(self, *, write_body: bool) -> None:
        path, _ = self.split_path()
        match = _OBJECT.match(path)
        if match is None:
            self.respond_error(404, f"no such route: {self.command} {path}")
            return
        bucket = match.group("bucket")
        name = urllib.parse.unquote(match.group("obj"))
        stored = self.fake.get_object(bucket, name)
        if stored is None:
            self.respond_error(404, f"no such object: {bucket}/{name}")
            return
        self.respond(200, stored.data if write_body else b"", content_type=stored.content_type)


class FakeGcsServer(FakeHttpService):
    """The fake bucket store. One server can hold objects for several buckets."""

    handler_class: ClassVar[type[_FakeHandler]] = _GcsHandler

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._objects: dict[tuple[str, str], StoredObject] = {}
        self._uploads: list[StoredObject] = []
        super().__init__()

    # -- recording surface --------------------------------------------------

    @property
    def uploads(self) -> list[StoredObject]:
        """Every upload in order, including overwrites of the same object name."""
        with self._lock:
            return list(self._uploads)

    @property
    def object_names(self) -> list[str]:
        with self._lock:
            return [f"{b}/{n}" for (b, n) in self._objects]

    def get_object(self, bucket: str, name: str) -> StoredObject | None:
        with self._lock:
            return self._objects.get((bucket, name))

    def reset(self) -> None:
        with self._lock:
            self._objects.clear()
            self._uploads.clear()

    # -- seeding / URL surface ---------------------------------------------

    def put_object(
        self, bucket: str, name: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> StoredObject:
        """Seed an object directly, without going through the HTTP upload route."""
        return self._store(bucket, name, data, content_type)

    def public_url(self, bucket: str, name: str) -> str:
        """This fake's stand-in for the real public-read URL."""
        return f"{self.base_url}/{bucket}/{urllib.parse.quote(name, safe='/')}"

    def local_url(self, public_url: str) -> str:
        """Map a ``storage.googleapis.com`` URL produced by product code onto this fake.

        Only the origin is replaced; path and query survive. A URL that already
        points at this fake comes back unchanged.
        """
        parts = urllib.parse.urlsplit(public_url)
        return urllib.parse.urlunsplit(
            ("http", f"127.0.0.1:{self.port}", parts.path, parts.query, "")
        )

    # -- route implementation ----------------------------------------------

    def _store(self, bucket: str, name: str, data: bytes, content_type: str) -> StoredObject:
        stored = StoredObject(bucket=bucket, name=name, data=data, content_type=content_type)
        with self._lock:
            self._objects[(bucket, name)] = stored
            self._uploads.append(stored)
        return stored

    # -- the storage-client stand-in ---------------------------------------

    def client(self) -> Any:
        """A ``google.cloud.storage.Client``-shaped stand-in bound to this fake.

        Supports what the publisher uses::

            client.bucket(name).blob(object_name).upload_from_string(data, content_type=...)
            blob.public_url            # this fake's URL, not Google's
            blob.download_as_bytes()
            blob.make_public()         # no-op; the fake serves every object publicly

        Uploads travel over real HTTP to this server.
        """
        return _StorageClient(self)


def _metadata(stored: StoredObject) -> dict[str, Any]:
    """The subset of GCS object metadata a client may read back after upload."""
    return {
        "kind": "storage#object",
        "bucket": stored.bucket,
        "name": stored.name,
        "size": str(stored.size),
        "contentType": stored.content_type,
    }


@dataclass(frozen=True)
class _StorageClient:
    server: FakeGcsServer

    def bucket(self, name: str) -> _StorageBucket:
        return _StorageBucket(self.server, name)

    def get_bucket(self, name: str) -> _StorageBucket:
        return _StorageBucket(self.server, name)


@dataclass(frozen=True)
class _StorageBucket:
    server: FakeGcsServer
    name: str

    def blob(self, object_name: str) -> _StorageBlob:
        return _StorageBlob(self.server, self.name, object_name)


@dataclass(frozen=True)
class _StorageBlob:
    server: FakeGcsServer
    bucket_name: str
    name: str

    @property
    def public_url(self) -> str:
        return self.server.public_url(self.bucket_name, self.name)

    def upload_from_string(
        self, data: bytes | str, content_type: str = "application/octet-stream", **_: Any
    ) -> None:
        payload = data.encode("utf-8") if isinstance(data, str) else data
        url = self.server.public_url(self.bucket_name, self.name)
        response = httpx.put(
            url, content=payload, headers={"Content-Type": content_type}, timeout=HTTP_TIMEOUT
        )
        response.raise_for_status()

    def download_as_bytes(self, **_: Any) -> bytes:
        response = httpx.get(
            self.server.public_url(self.bucket_name, self.name), timeout=HTTP_TIMEOUT
        )
        response.raise_for_status()
        return response.content

    def make_public(self, **_: Any) -> None:
        """No-op: this fake serves every stored object anonymously."""
