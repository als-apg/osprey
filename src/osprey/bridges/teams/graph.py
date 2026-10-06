"""The Teams bridge's Microsoft Graph leg: the one file library files are shared from.

The bridge makes two Graph calls, both against the document library the Microsoft
365 administrator set aside for the bot (``TEAMS_FILES_DRIVE_ID``):

* ``PUT /drives/{drive}/root:/{folder}/{run}/{name}:/content`` uploads one file of a
  run into that run's own folder;
* ``POST /drives/{drive}/items/{folder}/invite`` shares the run's folder with named
  people, by directory id, read-only and sign-in required, mailing no one.

Both carry a bearer for the Graph audience, minted by a
:class:`~osprey.bridges.teams.client.TokenSource` of its own. No organisation-wide
or anonymous link is ever created: ``createLink`` is never called.

Its own HTTP client
-------------------
The Graph leg takes its own injected :class:`httpx.Client` (``graph_http``, wired in
``__main__``), apart from the token and Connector legs. Graph is one well-known host
per cloud, an upload needs a longer timeout than a reply, and a test or deployment
can then give it its own transport without touching the other two.

Failure policy
--------------
Nothing retries, nothing swallows: every failure is :class:`GraphError` or
:class:`~osprey.bridges.teams.client.TokenError`. The ops layer decides what a
failed upload or share costs; this module only makes the outcome routable and
diagnosable from the container log.

Third-party imports
-------------------
``httpx`` only, like :mod:`osprey.bridges.teams.client`.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from .client import TokenSource
from .config import TeamsBridgeConfig

GRAPH_BASE_TEMPLATE = "https://{graph_host}/v1.0"
"""Root of every Graph call. The host follows ``TEAMS_CLOUD``."""

GRAPH_TIMEOUT_SEC = 60.0
"""Timeout for a Graph call when this module builds its own client. Long enough for
an upload at :data:`~osprey.bridges.core.MAX_DELIVERED_DOC_BYTES`, the largest file
the worker hands a bridge."""

MAX_INVITE_RECIPIENTS = 100
"""Recipients per invite request. Graph documents no limit; a conversation larger
than this is shared in several requests."""

_ERROR_BODY_CHARS = 200
"""How much of a failed Graph answer is quoted into :class:`GraphError`. Graph's
``error.code`` leads the body, which is what names the problem."""

_RUN_SEGMENT = re.compile(r"[A-Za-z0-9._-]+")
"""Characters a run id may hold to become one folder segment, as a full match."""


class GraphError(RuntimeError):
    """A Graph call did not do what was asked.

    Covers a transport failure, a non-2xx answer, a body that is not the expected
    JSON, and an invite whose answer carries an ``error`` for any recipient (Graph's
    ``207 Multi-Status`` partial success). The message carries the status and a
    bounded body, never the bearer.
    """


@dataclass(frozen=True)
class UploadedFile:
    """One file that reached the library."""

    name: str
    """The file's name in the library."""

    item_id: str
    """Graph id of the file."""

    folder_id: str
    """Graph id of the folder it landed in (``parentReference.id``), the run folder
    a share is made on."""

    web_url: str
    """The ``https://`` address a signed-in member opens the file at."""


def run_folder(cfg: TeamsBridgeConfig, run_id: str) -> tuple[str, ...]:
    """The folder a run's files are uploaded into, as path segments.

    Args:
        cfg: Bridge config supplying the library folder.
        run_id: The run the files belong to.

    Returns:
        ``cfg.files_folder_segments`` with ``run_id`` appended.

    Raises:
        ValueError: If ``run_id`` is not one segment of letters, digits, ``.``,
            ``_`` and ``-``, or is ``.`` or ``..``.
    """
    if run_id in (".", "..") or not _RUN_SEGMENT.fullmatch(run_id):
        raise ValueError(f"run id {run_id!r} is not one folder segment")
    return (*cfg.files_folder_segments, run_id)


class GraphFiles:
    """Upload a file into the bot's library, and share a folder with named people.

    Thread-safe like :class:`~osprey.bridges.teams.client.ConnectorClient`: the
    bearer is fetched before one HTTP lock is taken, so a token refresh never holds
    up a call that did not need it.
    """

    def __init__(
        self,
        cfg: TeamsBridgeConfig,
        tokens: TokenSource,
        http: httpx.Client | None = None,
    ) -> None:
        """Wire the client to a config, a Graph token source and an HTTP client.

        Args:
            cfg: Bridge config supplying the Graph host and the drive id.
            tokens: A token source for the Graph audience, asked on every call.
            http: Client for the Graph leg — the injected ``graph_http`` seam, or a
                test's :class:`httpx.MockTransport` client. ``None`` builds a
                default one honouring the core config's ``trust_env``.
        """
        self._base = GRAPH_BASE_TEMPLATE.format(graph_host=cfg.graph_host)
        self._drive = cfg.files_drive_id
        self._tokens = tokens
        self._http = (
            http
            if http is not None
            else httpx.Client(timeout=GRAPH_TIMEOUT_SEC, trust_env=cfg.core.trust_env)
        )
        self._lock = threading.Lock()

    def upload(
        self, folder: Sequence[str], name: str, data: bytes, content_type: str | None
    ) -> UploadedFile:
        """Put ``data`` at ``folder/name`` in the library.

        A PUT to an existing path replaces its content, so a redelivery of the same
        run writes the same files again.

        Args:
            folder: Folder segments inside the library, each percent-encoded.
            name: The file's name, percent-encoded.
            data: The file's bytes.
            content_type: The served type, or ``None`` for
                ``application/octet-stream``.

        Returns:
            The uploaded file's name, ids and address.

        Raises:
            GraphError: On any failure, or an answer without an ``id``, a
                ``parentReference.id`` and an ``https://`` ``webUrl``.
            TokenError: If no Graph bearer could be obtained.
        """
        path = "/".join(quote(segment, safe="") for segment in (*folder, name))
        url = f"{self._base}/drives/{self._drive}/root:/{path}:/content"
        body = self._send(
            "PUT",
            url,
            content=data,
            headers={"Content-Type": content_type or "application/octet-stream"},
        )
        item_id = body.get("id")
        parent = body.get("parentReference")
        folder_id = parent.get("id") if isinstance(parent, dict) else None
        web_url = body.get("webUrl")
        if not (
            isinstance(item_id, str)
            and item_id
            and isinstance(folder_id, str)
            and folder_id
            and isinstance(web_url, str)
            and web_url.startswith("https://")
        ):
            raise GraphError(f"graph upload to {url} answered no id, folder id or https webUrl")
        return UploadedFile(name=name, item_id=item_id, folder_id=folder_id, web_url=web_url)

    def share(self, item_id: str, object_ids: Sequence[str]) -> None:
        """Give each directory id read access to ``item_id``, sign-in required.

        No invitation is mailed, and inherited permissions stay as they are.

        Args:
            item_id: The file or folder to share.
            object_ids: Directory ids of the people to share with, sent in requests
                of at most :data:`MAX_INVITE_RECIPIENTS`.

        Raises:
            GraphError: If any request fails, or any recipient's answer carries an
                ``error``.
            TokenError: If no Graph bearer could be obtained.
        """
        url = f"{self._base}/drives/{self._drive}/items/{quote(item_id, safe='')}/invite"
        ids = list(object_ids)
        for start in range(0, len(ids), MAX_INVITE_RECIPIENTS):
            batch = ids[start : start + MAX_INVITE_RECIPIENTS]
            body = self._send(
                "POST",
                url,
                json={
                    "recipients": [{"objectId": object_id} for object_id in batch],
                    "requireSignIn": True,
                    "sendInvitation": False,
                    "roles": ["read"],
                },
            )
            value = body.get("value")
            if not isinstance(value, list):
                raise GraphError(f"graph invite on {url} answered no value list")
            for entry in value:
                if not isinstance(entry, dict) or "error" in entry:
                    error = entry.get("error") if isinstance(entry, dict) else entry
                    raise GraphError(
                        f"graph invite on {url} refused a recipient: "
                        f"{str(error)[:_ERROR_BODY_CHARS]}"
                    )

    def _send(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        """Make one Graph call and return its JSON object answer, or raise."""
        bearer = self._tokens.token()
        headers = {**kwargs.pop("headers", {}), "Authorization": f"Bearer {bearer}"}
        try:
            with self._lock:
                response = self._http.request(method, url, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise GraphError(f"graph request to {url} failed: {exc}") from exc
        if not response.is_success:
            text = response.text.replace(bearer, "<bearer>")[:_ERROR_BODY_CHARS]
            raise GraphError(f"graph answered HTTP {response.status_code} for {url}: {text}")
        try:
            body = response.json()
        except ValueError as exc:
            raise GraphError(f"graph answered a non-JSON body for {url}") from exc
        if not isinstance(body, dict):
            raise GraphError(f"graph answered a JSON value that is not an object for {url}")
        return body
