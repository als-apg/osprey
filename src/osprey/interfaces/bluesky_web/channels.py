"""Serves the build-time channel catalog to the operator panels.

``osprey build`` writes a ``channels.json`` catalog (a sorted array of channel
addresses) next to the config the sidecar container is handed, and the compose
template mounts it read-only as a sibling of ``config.yml``. This router hands
that file to the browser so the plan form's channel fields can offer
autocomplete without a round trip to the control system.

The file is located relative to ``CONFIG_FILE`` -- the environment variable the
compose template already sets to ``/app/project/config.yml``, whose directory is
where the catalog is mounted. This is deliberately NOT
``osprey_connectors.workspace``'s config resolution: that resolver ignores
``CONFIG_FILE`` by design, and what this route needs is precisely the mounted
sibling, not whichever config the process would otherwise load. A direct dev run
(no ``CONFIG_FILE`` set) falls back to ``./config.yml``'s directory, i.e. the
working directory.

The catalog is the one thing this sidecar serves that is worth caching, so this
route opts out of the blanket no-store rule in
``osprey.interfaces.bluesky_web.app`` (which fills its header in with
``setdefault``, yielding to a route that made its own decision). ``no-cache``
means "revalidate every time, but a 304 is fine" -- paired with a strong ETag
over the file bytes, a panel reload costs one empty 304 instead of re-shipping
a catalog that only changes on a rebuild.

An absent catalog answers 404, not an empty list: the file is only mounted when
the build produced one, so its absence is the honest "this deployment has no
catalog" signal. Panels treat that identically to an empty catalog -- no
autocomplete, every other field unaffected.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response

router = APIRouter()


def _catalog_path() -> Path:
    """Locate ``channels.json`` beside the config named by ``CONFIG_FILE``."""
    return Path(os.environ.get("CONFIG_FILE", "config.yml")).parent / "channels.json"


@router.get("/channels")
async def get_channels(request: Request) -> Response:
    """Serve the channel catalog, revalidated against a strong ETag.

    The file is read on every request rather than cached in process: it is a
    small read-only bind mount, and re-reading keeps the ETag honest if the
    mount is swapped underneath a running container.
    """
    path = _catalog_path()

    try:
        payload = path.read_bytes()
    except OSError:
        raise HTTPException(status_code=404, detail="channel catalog not available") from None

    etag = f'"{hashlib.sha256(payload).hexdigest()}"'
    headers = {"Cache-Control": "no-cache", "ETag": etag}

    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)

    return Response(content=payload, media_type="application/json", headers=headers)
