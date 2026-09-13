"""Bot Framework JWT validation for the Microsoft Teams relay.

Teams delivers a message by POSTing an activity to a public HTTPS endpoint. The
relay is that endpoint, and the only thing standing between the open internet
and the queue the bridge consumes is this module: a request is forwarded only
once its bearer token is proven to be signed by a current Bot Framework signing
key, to name this bot as its audience, to come from the cloud's issuer, to be
inside its validity window, and to carry the same ``serviceUrl`` as the activity
body it arrived with.

Nothing outside the standard library and :mod:`jwt` is imported, so the whole
boundary runs as a plain unit test with no Functions host and no network. The
Azure-facing half lives in ``function_app.py``.
"""

from __future__ import annotations

import http.client
import json
import threading
from dataclasses import dataclass
from time import monotonic
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import jwt

KEY_FETCH_TIMEOUT_SEC = 3
"""Seconds allowed for the metadata GET and for each JWKS fetch.

Teams abandons a bot response after 15 seconds. A single cold request can make
three fetches -- the metadata document, the JWKS it names, and PyJWT's one
refresh when a key id misses -- so each has to fit several times over inside
that window with room left for the decode.
"""

CLOCK_SKEW_SEC = 300
"""Leeway granted to ``nbf``, ``iat`` and ``exp`` alike.

Five minutes is the allowance Microsoft's token verification checklist
specifies; PyJWT applies a single ``leeway`` to every time claim.
"""

KEY_SOURCE_LIFESPAN_SEC = 3600
"""How long a built key source is reused before it is discarded and rebuilt.

The metadata document names the JWKS endpoint, and that name is not permanent.
Rebuilding hourly picks up a moved endpoint without asking for the document on
every request.
"""


class ValidationError(Exception):
    """The request is not a valid activity for this bot.

    The token was read and found wanting, or the activity disagrees with it.
    Nothing about retrying would change the answer, so the relay answers 401.
    """


class KeySourceUnavailable(Exception):
    """The signing keys could not be reached, so the token's fate is unknown.

    This is never a statement about the token -- only about the relay's ability
    to judge it. The relay answers 503 and Teams re-delivers.
    """


@dataclass(frozen=True)
class Cloud:
    """Where one Azure cloud publishes its signing keys and what it signs as."""

    metadata_url: str
    issuer: str


CLOUDS: dict[str, Cloud] = {
    "commercial": Cloud(
        metadata_url="https://login.botframework.com/v1/.well-known/openidconfiguration",
        issuer="https://api.botframework.com",
    ),
    "gcchigh": Cloud(
        metadata_url="https://login.botframework.azure.us/v1/.well-known/openidconfiguration",
        issuer="https://api.botframework.us",
    ),
}

_key_sources: dict[str, tuple[float, jwt.PyJWKClient]] = {}
_key_sources_lock = threading.Lock()


def _build_key_source(cloud: Cloud) -> jwt.PyJWKClient:
    """Read the cloud's metadata document and open a client on the JWKS it names.

    The metadata URL is never handed to :class:`jwt.PyJWKClient` itself: that
    document is not a key set, and pointing the client at it would turn every
    verification into a parse failure that reads like a key outage.
    """
    with urlopen(cloud.metadata_url, timeout=KEY_FETCH_TIMEOUT_SEC) as response:
        metadata = json.load(response)
    if not isinstance(metadata, dict):
        raise ValueError("the OpenID metadata document is not a JSON object")
    jwks_uri = metadata.get("jwks_uri")
    if not isinstance(jwks_uri, str):
        raise ValueError("the OpenID metadata document names no string jwks_uri")
    if urlparse(jwks_uri).scheme not in ("http", "https"):
        raise ValueError(f"the metadata document's jwks_uri is not http(s): {jwks_uri!r}")
    return jwt.PyJWKClient(
        uri=jwks_uri,
        lifespan=KEY_SOURCE_LIFESPAN_SEC,
        timeout=KEY_FETCH_TIMEOUT_SEC,
    )


def _key_source(name: str, cloud: Cloud) -> jwt.PyJWKClient:
    """Return the cached key source for ``name``, building it on first use.

    Building happens on a request rather than at import so that a metadata
    outage at cold start is a 503 on one request instead of a worker that
    cannot start at all. The build runs outside the lock: two concurrent cold
    requests may each fetch the document and the last one wins, which costs a
    duplicate GET, where holding the lock across the network would stall every
    request behind whichever one is waiting on it.
    """
    now = monotonic()
    with _key_sources_lock:
        cached = _key_sources.get(name)
        if cached is not None and now - cached[0] < KEY_SOURCE_LIFESPAN_SEC:
            return cached[1]
    source = _build_key_source(cloud)
    with _key_sources_lock:
        _key_sources[name] = (now, source)
    return source


def _activity_from_body(body: bytes | str) -> dict[str, Any]:
    """Parse the POST body into the activity object, or reject the request."""
    try:
        activity = json.loads(body)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"the request body is not JSON: {exc}") from exc
    if not isinstance(activity, dict):
        raise ValidationError("the request body is not a JSON object")
    return activity


def validate_activity(
    body: bytes | str,
    token: str,
    *,
    cloud: str,
    app_id: str,
) -> dict[str, Any]:
    """Return the activity carried by ``body`` once ``token`` proves it genuine.

    The work is split into three stages, each with its own ``try``, because the
    two failures mean opposite things to the caller: a bad token must be
    refused for good (401) and an unreachable key source must be retried (503).
    A single ``try`` around the lot would let a malformed header be reported as
    an outage, re-delivering a request that can never succeed, and let a key
    outage be reported as a forgery, dropping a message that was fine.

    Stage A reads the header and touches no network, so a garbage bearer value
    costs nothing. Stage B reaches the signing keys, and the only token-derived
    value it is given is the key id stage A already proved to be a string. Stage
    C verifies the signature and the claims.

    Raises:
        ValidationError: the token or the activity is not acceptable.
        KeySourceUnavailable: the signing keys could not be reached.
    """
    try:
        cloud_config = CLOUDS[cloud]
    except KeyError as exc:  # a mis-configured relay, not a bad request
        raise ValueError(f"unknown Teams cloud {cloud!r}") from exc

    activity = _activity_from_body(body)

    # Stage A -- header parse, no I/O.
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if not isinstance(kid, str):
            raise ValidationError("the token header carries no string kid")
    except jwt.PyJWTError as exc:
        raise ValidationError(f"the bearer value is not a readable JWT: {exc}") from exc

    # Stage B -- the key source, the only stage that reaches the network.
    try:
        signing_key = _key_source(cloud, cloud_config).get_signing_key(kid)
    except (jwt.PyJWTError, OSError, http.client.HTTPException, ValueError) as exc:
        # Everything PyJWT and urllib raise on this path: a refused or reset
        # connection, a truncated or non-JSON body, a key set that is missing,
        # empty or not an object, a key id still absent after PyJWT's one
        # refresh (a rotation race a retry heals), and a relay built without
        # the cryptography extra. Anything else is a bug in this module and
        # propagates as a 500 rather than being masked as a retry.
        raise KeySourceUnavailable(
            f"the Bot Framework signing keys are unreachable: {exc}"
        ) from exc

    # Stage C -- signature and claims.
    try:
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=app_id,
            issuer=cloud_config.issuer,
            leeway=CLOCK_SKEW_SEC,
        )
    except jwt.PyJWTError as exc:
        raise ValidationError(f"the token was rejected: {exc}") from exc

    if claims.get("serviceurl") != activity.get("serviceUrl"):
        raise ValidationError("the token's serviceurl claim is not the activity's serviceUrl")
    return activity
