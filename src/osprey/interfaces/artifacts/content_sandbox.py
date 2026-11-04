"""Default-deny opaque origin for the artifact gallery's content routes.

Artifacts are agent-authored: an HTML artifact, a rendered notebook or a
rendered markdown page may run script. Served on the gallery's own origin,
that script could call every same-origin endpoint with the operator's
session. :class:`ContentSandboxMiddleware` therefore answers every response
with :data:`~osprey.interfaces.opaque_origin.OPAQUE_CSP`, which puts the
document in an opaque origin (``self.origin === 'null'``): it runs, it can
offer a download or print itself, and it holds no cookie and no same-origin
reach. A response a route served under the opaque origin's own rule for an
inert type (:func:`~osprey.interfaces.opaque_origin.document_headers`: a type
on the closed inert list -- PDF, plain text, JSON, CSV, raster images -- with
``X-Content-Type-Options: nosniff``) is left as its route served it, so a PDF
renders; an error page or any other answer is sandboxed whatever its type.

The policy is the default. A route is exempt only when its endpoint is marked
with :func:`not_content` -- the gallery shell, which must stay same-origin to
drive the API. A route added later without the mark is sandboxed. The header
is appended as its own policy, so it intersects with (and can never be
loosened by) any other ``Content-Security-Policy`` on the response; a response
already carrying this very policy is left with its one copy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from osprey.interfaces.opaque_origin import OPAQUE_CSP, is_inert

_NOT_CONTENT_ATTR = "__osprey_not_content__"

F = TypeVar("F", bound=Callable[..., Any])


def not_content(endpoint: F) -> F:
    """Mark a route endpoint as same-origin application chrome, not artifact content."""
    setattr(endpoint, _NOT_CONTENT_ATTR, True)
    return endpoint


def is_not_content(endpoint: object) -> bool:
    return bool(getattr(endpoint, _NOT_CONTENT_ATTR, False))


class ContentSandboxMiddleware:
    """Append the content sandbox policy to every response not marked :func:`not_content`.

    Pure ASGI: the router writes the matched endpoint into the shared request
    scope before the response starts, so the decision is made per route on
    the ``http.response.start`` message -- every status, error pages and
    unmatched paths included.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_sandbox(message: Message) -> None:
            if message["type"] == "http.response.start" and not is_not_content(
                scope.get("endpoint")
            ):
                headers = MutableHeaders(scope=message)
                # A route that already set this very policy (opaque_headers())
                # needs no second, identical copy; any other policy is joined.
                already = OPAQUE_CSP in [
                    value.strip() for value in headers.getlist("content-security-policy")
                ]
                # An inert type a route served under the opaque origin's rule
                # (document_headers(): the type plus nosniff) keeps its headers,
                # so a PDF still renders; every other answer is sandboxed.
                served_inert = (
                    is_inert(headers.get("content-type"))
                    and headers.get("x-content-type-options", "").strip().lower() == "nosniff"
                )
                if not already and not served_inert:
                    headers.append("Content-Security-Policy", OPAQUE_CSP)
            await send(message)

        await self.app(scope, receive, send_with_sandbox)
