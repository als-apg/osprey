"""The injected prior-artifact fetcher: worker byte route first, published URL second.

A follow-up question re-injects the bytes of an earlier turn's images. The engine's own
fetcher knows only the worker byte route, and for this channel that is the *ephemeral*
copy — swept once the run is old, gone after a restart — while the GCS object the space is
already linking to is permanent. Hence two routes, and hence this file: the fallback is
the branch a live conversation reaches days after the answer was posted, long after any
test would notice it missing.

Both routes are faked over :class:`httpx.MockTransport`, so a run's whole re-fetch is
deterministic and no network, credential, or Google library is involved. What the fetcher
returns is the engine's contract — ``(bytes, served content type)``, ``(None, None)`` for
every failure — and the served type is read off the response rather than off the
descriptor, because a mislabelled artifact must be skipped, never injected as an image.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from osprey.bridges.core import PNG_MAGIC, CoreConfig
from osprey.bridges.google_chat import ops as ops_module
from osprey.bridges.google_chat.ops import build_prior_artifact_fetcher

WORKER_URL = "http://work:9190"
WORKER_TOKEN = "work-token"
RUN_ID = "R1"
ARTIFACT_ID = "a1"
PUBLIC_URL = "https://storage.googleapis.com/test-bucket/plots/1.png"

PNG_BYTES = PNG_MAGIC + b"fake-png-body"
GCS_BYTES = PNG_MAGIC + b"the-copy-that-survived"

BUDGET = 1_000_000

CFG = CoreConfig(worker_url=WORKER_URL, dispatch_worker_token=WORKER_TOKEN)

Handler = Callable[[httpx.Request], httpx.Response]


def descriptor(**overrides: Any) -> dict[str, Any]:
    """One prior-turn OUTPUT descriptor, as the history store round-trips it."""
    return {
        "entry_id": ARTIFACT_ID,
        "filename": "orbit.png",
        "delivered_mime": "image/png",
        "public_url": PUBLIC_URL,
        "origin": "output",
        **overrides,
    }


# --- the two faked routes ----------------------------------------------------


def route(handler: Handler) -> tuple[httpx.Client, list[httpx.Request]]:
    """A client over ``handler``, plus the list of requests it was asked to make — which
    is how a test asserts that a route was never touched at all."""
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(recording)), seen


def worker_serving(artifacts: dict[str, tuple[bytes, str | None]]) -> Handler:
    """``artifact_id -> (bytes, served Content-Type)``; anything else 404s, which is what
    a swept run looks like."""

    def handler(request: httpx.Request) -> httpx.Response:
        entry = artifacts.get(str(request.url).rsplit("/", 1)[-1])
        if entry is None:
            return httpx.Response(404)
        assert request.headers["Authorization"] == f"Bearer {WORKER_TOKEN}"
        return respond(*entry)

    return handler


def gcs_serving(objects: dict[str, tuple[bytes, str | None]]) -> Handler:
    """``url -> (bytes, served Content-Type)``; anything else 404s."""

    def handler(request: httpx.Request) -> httpx.Response:
        entry = objects.get(str(request.url))
        if entry is None:
            return httpx.Response(404)
        return respond(*entry)

    return handler


def respond(data: bytes, content_type: str | None) -> httpx.Response:
    """A 200 carrying ``data``. A ``None`` type sends no header at all — a source that
    reports the type as absent rather than guessing one."""
    headers = {} if content_type is None else {"Content-Type": content_type}
    return httpx.Response(200, content=data, headers=headers)


def fetch(
    worker: Handler,
    public: Handler,
    *,
    desc: dict[str, Any] | None = None,
    run_id: str | None = RUN_ID,
    max_bytes: int = BUDGET,
    cfg: CoreConfig = CFG,
) -> tuple[tuple[bytes | None, str | None], list[httpx.Request], list[httpx.Request]]:
    """Run one re-fetch over both faked routes.

    Returns the fetcher's own result plus what each route was asked for, so a test can
    assert both the bytes and which route produced them.
    """
    worker_client, worker_seen = route(worker)
    public_client, public_seen = route(public)
    with worker_client, public_client:
        fetcher = build_prior_artifact_fetcher(worker_http=worker_client, public_http=public_client)
        result = fetcher(cfg, run_id, descriptor() if desc is None else desc, max_bytes)
    return result, worker_seen, public_seen


# --- the worker route --------------------------------------------------------


def test_the_worker_route_answers_and_the_published_url_is_never_touched():
    # The worker is tried first because it is the authoritative copy while it exists, and
    # reaching for GCS anyway would put a public fetch on the hot path of every follow-up.
    result, worker_seen, public_seen = fetch(
        worker_serving({ARTIFACT_ID: (PNG_BYTES, "image/png")}),
        gcs_serving({PUBLIC_URL: (GCS_BYTES, "image/png")}),
    )

    assert result == (PNG_BYTES, "image/png")
    assert len(worker_seen) == 1
    assert public_seen == []


def test_the_served_type_is_read_off_the_response_not_off_the_descriptor():
    # The descriptor claims PNG; the worker serves the HTML a fallback conversion really
    # produced. Reporting the served type is what lets the engine skip the artifact
    # instead of shipping a web page to the agent labelled as an image.
    result, _, _ = fetch(
        worker_serving({ARTIFACT_ID: (b"<html>not a plot</html>", "text/html")}),
        gcs_serving({}),
    )

    assert result == (b"<html>not a plot</html>", "text/html")


# --- the fallback ------------------------------------------------------------


def test_a_swept_artifact_falls_back_to_the_url_the_bridge_published():
    # The branch this whole seam exists for: the worker has 404'd the artifact away, and
    # the GCS object the space still links to answers instead.
    result, worker_seen, public_seen = fetch(
        worker_serving({}),
        gcs_serving({PUBLIC_URL: (GCS_BYTES, "image/png")}),
    )

    assert result == (GCS_BYTES, "image/png")
    assert len(worker_seen) == 1
    assert [str(request.url) for request in public_seen] == [PUBLIC_URL]


def test_the_fallback_is_fetched_without_the_worker_token():
    # The object is anonymously readable — it has to be, since Chat itself fetches it to
    # render the card — so the internal token has no business travelling off-site.
    _, _, public_seen = fetch(
        worker_serving({}),
        gcs_serving({PUBLIC_URL: (GCS_BYTES, "image/png")}),
    )

    assert "Authorization" not in public_seen[0].headers


def test_an_object_served_with_no_content_type_reports_no_served_type():
    # Absent is reported as absent, never guessed from the bytes: the engine admits only a
    # provably-PNG artifact, so an unknown type must skip rather than slip through.
    result, _, _ = fetch(
        worker_serving({}),
        gcs_serving({PUBLIC_URL: (GCS_BYTES, None)}),
    )

    assert result == (GCS_BYTES, None)


def test_the_served_type_is_normalized_to_a_bare_mime():
    result, _, _ = fetch(
        worker_serving({}),
        gcs_serving({PUBLIC_URL: (GCS_BYTES, "Image/PNG; charset=binary")}),
    )

    assert result == (GCS_BYTES, "image/png")


def test_a_worker_body_over_budget_still_falls_back_to_the_published_url():
    # Every worker-side failure falls through, not just the 404: the fallback is one GET
    # against a URL this bridge published itself, and distinguishing the causes would buy
    # nothing but a way to get the routing wrong.
    result, _, public_seen = fetch(
        worker_serving({ARTIFACT_ID: (b"x" * 500, "image/png")}),
        gcs_serving({PUBLIC_URL: (GCS_BYTES, "image/png")}),
        max_bytes=100,
    )

    assert result == (GCS_BYTES, "image/png")
    assert len(public_seen) == 1


def test_a_turn_with_no_run_id_goes_straight_to_the_published_url():
    result, worker_seen, _ = fetch(
        worker_serving({ARTIFACT_ID: (PNG_BYTES, "image/png")}),
        gcs_serving({PUBLIC_URL: (GCS_BYTES, "image/png")}),
        run_id=None,
    )

    assert result == (GCS_BYTES, "image/png")
    assert worker_seen == []


def test_a_descriptor_naming_no_artifact_goes_straight_to_the_published_url():
    result, worker_seen, _ = fetch(
        worker_serving({ARTIFACT_ID: (PNG_BYTES, "image/png")}),
        gcs_serving({PUBLIC_URL: (GCS_BYTES, "image/png")}),
        desc=descriptor(entry_id=None),
    )

    assert result == (GCS_BYTES, "image/png")
    assert worker_seen == []


# --- failure is always (None, None) -----------------------------------------


def test_a_descriptor_with_no_published_url_reports_nothing_when_the_worker_has_swept_it():
    # A run whose publish failed, or a deployment with no bucket at all: there is no
    # second copy, so re-injection simply degrades and the engine flags the descriptor.
    result, _, public_seen = fetch(
        worker_serving({}),
        gcs_serving({PUBLIC_URL: (GCS_BYTES, "image/png")}),
        desc=descriptor(public_url=None),
    )

    assert result == (None, None)
    assert public_seen == []


def test_both_routes_failing_reports_nothing_rather_than_raising():
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    result, _, _ = fetch(worker_serving({}), failing)

    assert result == (None, None)


def test_a_transport_error_on_the_fallback_reports_nothing_rather_than_raising():
    def exploding(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    result, _, _ = fetch(worker_serving({}), exploding)

    assert result == (None, None)


def test_a_malformed_published_url_reports_nothing_rather_than_raising():
    # The URL comes off a stored descriptor, and httpx raises on a scheme it cannot use
    # before any request is made — a raise here would cost the dispatch, not one image.
    result, _, _ = fetch(
        worker_serving({}),
        gcs_serving({}),
        desc=descriptor(public_url="not-a-url"),
    )

    assert result == (None, None)


def test_an_over_budget_object_is_dropped_whole_rather_than_truncated():
    result, _, _ = fetch(
        worker_serving({}),
        gcs_serving({PUBLIC_URL: (b"x" * 500, "image/png")}),
        max_bytes=100,
    )

    assert result == (None, None)


def test_an_exhausted_budget_fetches_nothing_at_all():
    # The engine walks prior images newest-first and the budget can run out mid-walk; the
    # remaining ones must cost no request on either route.
    result, worker_seen, public_seen = fetch(
        worker_serving({ARTIFACT_ID: (PNG_BYTES, "image/png")}),
        gcs_serving({PUBLIC_URL: (GCS_BYTES, "image/png")}),
        max_bytes=0,
    )

    assert result == (None, None)
    assert worker_seen == []
    assert public_seen == []


# --- the clients the fetcher builds for itself -------------------------------


def recording_clients(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> list[httpx.Client]:
    """Replace the ops module's own ``httpx`` binding, so every client the fetcher builds
    for itself is recorded and answered from ``handler``.

    The module's binding rather than ``httpx.Client`` itself: patching the library would
    reach every client in the process, including ones this test says nothing about.
    """
    built: list[httpx.Client] = []

    class Recording(httpx.Client):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)
            built.append(self)

    monkeypatch.setattr(ops_module, "httpx", SimpleNamespace(Client=Recording))
    return built


def test_the_worker_route_is_reached_directly_whatever_the_environment_proxies(
    monkeypatch: pytest.MonkeyPatch,
):
    # The worker is an internal service reached directly. A proxy inherited from the
    # deployment's environment must never mount itself in front of it — not even at a site
    # that legitimately sets trust_env for its egress.
    built = recording_clients(monkeypatch, worker_serving({ARTIFACT_ID: (PNG_BYTES, "image/png")}))
    proxied = CoreConfig(worker_url=WORKER_URL, dispatch_worker_token=WORKER_TOKEN, trust_env=True)

    build_prior_artifact_fetcher()(proxied, RUN_ID, descriptor(), BUDGET)

    assert [client.trust_env for client in built] == [False]


@pytest.mark.parametrize("trust_env", [True, False])
def test_the_fallback_route_honors_the_configured_proxy_setting(
    monkeypatch: pytest.MonkeyPatch, trust_env: bool
):
    # This leg leaves the deployment's network for storage.googleapis.com, so a site
    # behind an egress proxy can reach it only through that proxy.
    built = recording_clients(monkeypatch, gcs_serving({PUBLIC_URL: (GCS_BYTES, "image/png")}))
    cfg = CoreConfig(worker_url=WORKER_URL, dispatch_worker_token=WORKER_TOKEN, trust_env=trust_env)

    result = build_prior_artifact_fetcher()(cfg, RUN_ID, descriptor(), BUDGET)

    assert result == (GCS_BYTES, "image/png")
    worker_client, public_client = built
    assert worker_client.trust_env is False
    assert public_client.trust_env is trust_env


def test_every_client_the_fetcher_builds_is_closed(monkeypatch: pytest.MonkeyPatch):
    # One re-fetch per prior image, on a long-lived process: a client left open leaks a
    # connection pool per follow-up question.
    built = recording_clients(monkeypatch, gcs_serving({PUBLIC_URL: (GCS_BYTES, "image/png")}))

    build_prior_artifact_fetcher()(CFG, RUN_ID, descriptor(), BUDGET)

    assert built and all(client.is_closed for client in built)


def test_an_injected_client_is_left_open_for_its_owner():
    # The entrypoint's wiring hands in shared clients it closes at shutdown; closing one
    # here would break every later re-fetch.
    worker_client, _ = route(worker_serving({ARTIFACT_ID: (PNG_BYTES, "image/png")}))
    public_client, _ = route(gcs_serving({PUBLIC_URL: (GCS_BYTES, "image/png")}))
    fetcher = build_prior_artifact_fetcher(worker_http=worker_client, public_http=public_client)

    fetcher(CFG, RUN_ID, descriptor(), BUDGET)
    fetcher(CFG, RUN_ID, descriptor(entry_id="gone"), BUDGET)

    assert not worker_client.is_closed
    assert not public_client.is_closed
    worker_client.close()
    public_client.close()
