"""Tests for api/main.py: lifespan, health, OpenAPI contract, event-loop
worker-pool dispatch, and the body-size-limit / CORS-ordering middleware.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from osprey.services.optimizer.api import main
from osprey.services.optimizer.api.dependencies import get_environment_registry
from osprey.services.optimizer.api.main import MAX_JSON_BODY_BYTES, BodySizeLimitMiddleware, app
from osprey.services.optimizer.models.protocols import PVReading

from ._fakes import MockEnvironment


def _drive_lifespan(fake_app: SimpleNamespace) -> None:
    """Run the async lifespan to completion synchronously."""

    async def _run() -> None:
        cm = main.lifespan(fake_app)  # type: ignore[arg-type]
        await cm.__aenter__()
        await cm.__aexit__(None, None, None)

    asyncio.run(_run())


class TestLifespanFailFast:
    def test_lifespan_aborts_when_redis_unreachable(self) -> None:
        """Redis ping failure must propagate out of the lifespan, not fall back to a stub."""
        with (
            patch(
                "osprey.services.optimizer.api.main.ping_redis", side_effect=ConnectionError("boom")
            ),
            pytest.raises(RuntimeError, match="Redis unreachable"),
            TestClient(app),
        ):
            pass

    def test_lifespan_env_registry_failure_raises(self) -> None:
        """Env registry discovery failure becomes a fatal RuntimeError."""
        fake_app = SimpleNamespace(state=SimpleNamespace())
        bad_registry = MagicMock()
        bad_registry.discover_environments.side_effect = RuntimeError("disc boom")
        with (
            patch.object(main, "ping_redis"),
            patch.object(main, "EnvironmentRegistry", return_value=bad_registry),
            pytest.raises(RuntimeError, match="Failed to initialize environment registry"),
        ):
            _drive_lifespan(fake_app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.fixture
def fresh_openapi_cache() -> Generator[None]:
    """Clear ``app.openapi_schema`` around a test that regenerates the schema, so
    neither the test nor its neighbours see a stale cached document."""
    app.openapi_schema = None
    yield
    app.openapi_schema = None


class TestOpenAPIContract:
    """The published schema must match runtime, so typed clients and schema
    fuzzers don't flag every string-detail error as a contract violation.
    """

    def test_validation_detail_allows_string_and_array(self, client: TestClient) -> None:
        """Manual 422s return a string ``detail``; FastAPI's own 422 returns an array.
        The shared ``HTTPValidationError`` schema must allow both.
        """
        schema = client.get("/openapi.json").json()
        detail = schema["components"]["schemas"]["HTTPValidationError"]["properties"]["detail"]
        kinds = {branch.get("type") for branch in detail["anyOf"]}
        assert kinds == {"array", "string"}

    def test_operations_document_their_error_responses(self, client: TestClient) -> None:
        """Path-param lookups document 404, control-system reads document 503, and
        body-validating operations document 400."""
        schema = client.get("/openapi.json").json()
        for path, method, status in [
            ("/environments/{name}", "get", "404"),
            ("/jobs/{job_id}", "get", "404"),
            ("/environments/{name}/pv", "get", "503"),
            ("/environments/{name}/variables", "get", "503"),
            ("/environments/{name}/observables", "get", "503"),
            ("/environments/{name}/initial-bounds", "get", "503"),
            ("/environments/{name}/evaluate", "post", "503"),
            ("/environments/{name}/resolve-bounds", "post", "400"),
        ]:
            assert status in schema["paths"][path][method]["responses"], f"{method} {path}"

    def test_custom_openapi_caches_and_returns_same_schema(self, fresh_openapi_cache: None) -> None:
        """Calling again returns the cached schema (early-return path)."""
        schema = main.custom_openapi()
        assert main.custom_openapi() is schema

    def test_custom_openapi_documents_503_for_503_ops(self, fresh_openapi_cache: None) -> None:
        """Collection listings and env-reads must document 503 in the published contract."""
        paths = main.custom_openapi()["paths"]
        for path in ("/jobs", "/results", "/models"):
            assert "503" in paths[path]["get"]["responses"], f"{path} GET missing documented 503"
        assert "503" in paths["/environments/{name}/evaluate"]["post"]["responses"]

    def test_custom_openapi_skips_non_dict_and_responseless_ops(
        self, fresh_openapi_cache: None
    ) -> None:
        """An operation lacking 'responses' is skipped without error."""

        def fake_get_openapi(**_: object) -> dict[str, object]:
            return {
                "paths": {
                    "/weird": {
                        # str value -> not a dict, hits the `continue`
                        "summary": "not an operation dict",
                        # dict but no "responses" key -> also hits the `continue`
                        "get": {"operationId": "noResponses"},
                    }
                },
            }

        with patch.object(main, "get_openapi", side_effect=fake_get_openapi):
            out = main.custom_openapi()
        assert out["paths"]["/weird"]["get"] == {"operationId": "noResponses"}

    def test_custom_openapi_detail_already_anyof_branch(self, fresh_openapi_cache: None) -> None:
        """HTTPValidationError.detail already carrying anyOf is left untouched."""

        def fake_get_openapi(**_: object) -> dict[str, object]:
            return {
                "components": {
                    "schemas": {
                        "HTTPValidationError": {
                            "properties": {
                                "detail": {"anyOf": [{"type": "string"}], "title": "Detail"}
                            }
                        }
                    }
                },
                "paths": {},
            }

        with patch.object(main, "get_openapi", side_effect=fake_get_openapi):
            out = main.custom_openapi()
        detail = out["components"]["schemas"]["HTTPValidationError"]["properties"]["detail"]
        assert detail == {"anyOf": [{"type": "string"}], "title": "Detail"}


# Worker-pool starvation: a blocking control-system read must not freeze the
# event loop and starve unrelated endpoints.


class TestEventLoopNotStarved:
    """Read handlers do blocking EPICS I/O, so they run as sync ``def`` and are
    dispatched to the threadpool. A wedged read must occupy only a threadpool
    thread, leaving the event loop free to serve other requests.
    """

    def test_blocking_read_does_not_starve_event_loop(
        self, mock_registry: MagicMock, mock_env: MockEnvironment
    ) -> None:
        started = threading.Event()
        release = threading.Event()

        def _blocking_read(variable_names: list[str]) -> dict[str, PVReading]:
            started.set()
            if not release.wait(timeout=10):
                raise AssertionError("release event was never set")
            return {n: PVReading(value=5.0) for n in variable_names}

        mock_env.get_variables = _blocking_read  # type: ignore[method-assign]
        app.dependency_overrides[get_environment_registry] = lambda: mock_registry

        async def _probe() -> None:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                slow = asyncio.create_task(
                    ac.get(
                        "/environments/sim/variables", params={"names": ["var1"], "target": "va"}
                    )
                )
                # Wait until the blocking read is actually in-flight (in the threadpool).
                await asyncio.to_thread(started.wait, 5)
                assert started.is_set()
                # The event loop must still serve a fast request while the read blocks.
                # On the old async-def code this times out (loop frozen inside the read).
                health = await asyncio.wait_for(ac.get("/health"), timeout=5)
                assert health.status_code == 200
                # Release the blocked read and confirm it completed normally.
                release.set()
                slow_resp = await asyncio.wait_for(slow, timeout=5)
                assert slow_resp.status_code == 200

        try:
            asyncio.run(_probe())
        finally:
            release.set()
            app.dependency_overrides.clear()


@pytest.fixture
def panel_origin_allowed() -> Generator[None, None, None]:
    """Allow one cross-origin client for the test; production allows none by default.

    The CORS middleware keeps a reference to this list, so extending it in place
    reaches the middleware however early its stack was built.
    """
    main._cors_origins.append("http://panel.example.org")
    try:
        yield
    finally:
        main._cors_origins.remove("http://panel.example.org")


def _client() -> TestClient:
    """Client for the middleware tests. The Redis seam is already stubbed by the
    autouse ``mock_api_redis_client`` fixture, so the lifespan ping succeeds."""
    return TestClient(app)


class TestBodySizeLimit:
    def test_oversize_json_body_rejected_with_413(self) -> None:
        # A >1 MB JSON payload to a JSON endpoint is rejected before handling.
        big = "x" * (MAX_JSON_BODY_BYTES + 1)
        with _client() as client:
            resp = client.post(
                "/optimizations",
                content=f'{{"environment_name": "{big}"}}',
                headers={"content-type": "application/json"},
            )
        assert resp.status_code == 413
        assert resp.json() == {"detail": "Request body too large"}

    def test_small_json_body_passes_middleware(self) -> None:
        # A small body is NOT blocked by the cap; it reaches normal handling
        # (it may 422/400 on validation, but never 413).
        with _client() as client:
            resp = client.post(
                "/optimizations",
                json={"environment_name": "test_env"},
            )
        assert resp.status_code != 413

    def test_models_multipart_exempt_from_json_cap(self) -> None:
        # A >1 MB multipart upload to POST /models is NOT rejected by this
        # middleware; the route keeps its own 50 MB cap. We assert the
        # middleware lets it through (a non-.joblib name 422s in the handler,
        # proving the request reached the route rather than being 413'd here).
        big = b"x" * (MAX_JSON_BODY_BYTES + 1)
        with _client() as client:
            resp = client.post(
                "/models",
                files={"file": ("model.bin", big, "application/octet-stream")},
            )
        assert resp.status_code != 413

    def test_multipart_to_non_models_path_still_capped(self) -> None:
        # The exemption is POST /models only: a >1 MB multipart body to any
        # other route remains subject to the 1 MB cap and is 413'd.
        big = b"x" * (MAX_JSON_BODY_BYTES + 1)
        with _client() as client:
            resp = client.post(
                "/optimizations",
                files={"file": ("blob.bin", big, "application/octet-stream")},
            )
        assert resp.status_code == 413

    @pytest.mark.usefixtures("panel_origin_allowed")
    def test_oversize_413_carries_cors_header(self) -> None:
        # CORS must be the outermost middleware so a cross-origin oversize POST
        # still gets the access-control-allow-origin header on its 413.
        big = "x" * (MAX_JSON_BODY_BYTES + 1)
        with _client() as client:
            resp = client.post(
                "/optimizations",
                content=f'{{"environment_name": "{big}"}}',
                headers={
                    "content-type": "application/json",
                    "origin": "http://panel.example.org",
                },
            )
        assert resp.status_code == 413
        assert resp.headers["access-control-allow-origin"] == "http://panel.example.org"

    @pytest.mark.usefixtures("panel_origin_allowed")
    def test_preflight_options_unaffected(self) -> None:
        # OPTIONS preflight (no body) is handled by CORS, not 413'd by the cap.
        with _client() as client:
            resp = client.options(
                "/optimizations",
                headers={
                    "origin": "http://panel.example.org",
                    "access-control-request-method": "POST",
                },
            )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://panel.example.org"

    def test_non_numeric_content_length_passes_through(self) -> None:
        # A non-integer Content-Length can't be range-checked, so the middleware lets
        # the request reach the app (the route's own validation still applies). Driven
        # at the ASGI layer because httpx rejects malformed Content-Length before it ships.
        forwarded: dict[str, bool] = {"called": False}

        async def dummy_app(scope: object, receive: object, send: object) -> None:
            forwarded["called"] = True

        middleware = BodySizeLimitMiddleware(dummy_app)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/optimizations",
            "headers": [(b"content-length", b"not-a-number")],
        }

        async def receive() -> dict[str, object]:
            return {"type": "http.request"}

        async def send(_message: object) -> None:
            return None

        asyncio.run(middleware(scope, receive, send))
        assert forwarded["called"] is True
