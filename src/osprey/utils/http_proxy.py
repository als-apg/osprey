"""Shared constants for OSPREY's HTTP reverse-proxy / relay hops.

OSPREY has more than one component that relays an upstream HTTP response back
to a browser: the web-terminal panel proxy
(``osprey.interfaces.web_terminal.routes.proxy``) and the Bluesky panels
sidecar's SSE draft relay (``osprey.interfaces.bluesky_panels.draft_relay``).
Each must strip the same hop-by-hop headers before re-emitting the upstream's
headers downstream -- a header list that had been copied verbatim between them.
Keeping it here, imported by both, prevents the two lists from drifting.
"""

from __future__ import annotations

# Hop-by-hop headers (RFC 7230 sec. 6.1, plus the two content-framing headers a
# reverse proxy must recompute) that must never be relayed from an upstream
# response onto the downstream client connection.
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-encoding",
        "content-length",
    }
)
