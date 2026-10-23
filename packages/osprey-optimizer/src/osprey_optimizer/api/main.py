"""FastAPI application factory.

Creates the app, wires up the lifespan handler (shared singletons on
``app.state``), and includes all domain routers.
"""

import logging
import math
import os
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from osprey_optimizer.api.job_tracker import JobTracker
from osprey_optimizer.api.routers.environments import router as environments_router
from osprey_optimizer.api.routers.jobs import router as jobs_router
from osprey_optimizer.api.routers.models import router as models_router
from osprey_optimizer.api.routers.optimization import router as optimization_router
from osprey_optimizer.api.routers.results import router as results_router
from osprey_optimizer.api.routers.run_config import router as run_config_router
from osprey_optimizer.config.logging import setup_logging
from osprey_optimizer.models.process_scope import allow_multiple_targets
from osprey_optimizer.services.plugin_registry import EnvironmentRegistry
from osprey_optimizer.utils.json_safe import map_floats
from osprey_optimizer.utils.redis_client import get_redis_url, ping_redis

logger = logging.getLogger(__name__)


# Cap JSON request bodies so a buggy/runaway client can't buffer an unbounded
# body into a worker before validation (C1). The multipart upload route
# POST /models is exempt — it enforces its own 50 MB cap in routers/models.py.
MAX_JSON_BODY_BYTES = 1_048_576  # 1 MB


class BodySizeLimitMiddleware:
    """Reject request bodies whose declared Content-Length exceeds the cap with 413.

    Checked before the body is read, so an oversize body is rejected without
    buffering it. Every client sends Content-Length; a request without one is
    passed through and the route's own validation still applies.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or (scope["method"] == "POST" and scope["path"] == "/models"):
            await self.app(scope, receive, send)
            return

        try:
            declared = int(Headers(scope=scope).get("content-length", "0"))
        except ValueError:
            # Non-integer Content-Length can't be range-checked; pass it through
            # and let the route's own validation apply.
            declared = 0
        if declared > MAX_JSON_BODY_BYTES:
            response = Response(
                b'{"detail":"Request body too large"}',
                status_code=413,
                media_type="application/json",
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


@contextmanager
def _init_or_fatal(label: str) -> Iterator[None]:
    """Abort startup if a required singleton can't be initialized.

    ``label`` is both the critical-log prefix and the ``RuntimeError`` message:
    a half-wired app must fail fast rather than serve with a missing singleton.
    """
    try:
        yield
    except Exception as e:
        logger.critical("FATAL: %s: %s", label, e)
        raise RuntimeError(label) from e


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize shared singletons and store them on ``app.state``.

    Logging is configured here rather than at module scope: ``setup_logging``
    calls ``basicConfig(force=True)``, so doing it on import made merely
    importing this module rewrite the root handlers of whatever process did so
    — including a pytest session during collection. Startup runs in every
    uvicorn worker, so ``--workers N`` still gets configured logging, and it
    must come first so nothing below logs into an unconfigured root.
    """
    setup_logging(level=logging.INFO)
    logger.info("CORS enabled for origins: %s", _cors_origins)
    # The API serves every environment, so the plugin hands each control-system
    # target to a child process of its own; job workers it spawns stay bound to one.
    allow_multiple_targets()

    redis_url = get_redis_url()
    with _init_or_fatal(f"Redis unreachable at {redis_url}"):
        ping_redis()
    logger.info("Connected to Redis at %s", redis_url)
    app.state.job_tracker = JobTracker()

    with _init_or_fatal("Failed to initialize environment registry"):
        env_registry = EnvironmentRegistry()
        env_registry.discover_environments()
    app.state.environment_registry = env_registry

    yield

    logger.info("Shutting down API")


app = FastAPI(title="OSPREY Optimizer API", lifespan=lifespan)

# Order matters: add_middleware prepends, so the last-added middleware is the
# OUTERMOST wrapper. Add the body cap first and CORS last so CORS wraps the
# body guard — a cross-origin oversize POST then gets its 413 WITH CORS
# headers, and preflight OPTIONS (no body) is handled by CORS before the cap.
app.add_middleware(BodySizeLimitMiddleware)

# No cross-origin client by default: OSPREY's web panels reach this service
# through the web terminal's same-origin proxy. A deployment that serves its
# own UI from elsewhere lists that origin here.
_cors_raw = os.environ.get("OPTIMIZER_CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sanitize_non_finite(obj: Any) -> Any:
    """Replace non-finite floats (nan/inf) in a nested structure with their string repr, in-place.

    FastAPI's 422 body echoes the offending input under each error's ``input``
    key; a ``nan``/``inf`` there (reachable via the JSON literals ``NaN`` /
    ``1e999``) makes Starlette's ``allow_nan=False`` renderer raise, degrading
    the 422 into a generic 500. Stringifying the value keeps the response
    serializable so validation failures stay clean 422s.
    """
    return map_floats(obj, lambda f: f if math.isfinite(f) else repr(f))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Mirror FastAPI's default 422 handler, but sanitize non-finite floats.

    Keeps the error body serializable (see ``_sanitize_non_finite``).
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": _sanitize_non_finite(jsonable_encoder(exc.errors()))},
    )


# Inclusion order sets the tag-group order in the Swagger UI / OpenAPI schema.
# The job-creating section (optimizations) precedes the unified jobs plane.
app.include_router(optimization_router)
app.include_router(jobs_router)
app.include_router(environments_router)
app.include_router(run_config_router)
app.include_router(results_router)
app.include_router(models_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


# Operations that surface a 503 instead of a (possibly misleading) 2xx, mapped to
# the reason documented in the published contract. Two causes:
#  - env reads when the CA worker is unresponsive (see routers/environments.py)
#  - collection listings when their backing store (Redis/disk) is unreachable; an
#    outage there must not masquerade as an empty list on a control system
#    (see list_jobs / list_results / list_models).
_OPS_503: dict[tuple[str, str], str] = {
    ("/environments/{name}/pv", "get"): "Control system worker unresponsive",
    ("/environments/{name}/variables", "get"): "Control system worker unresponsive",
    ("/environments/{name}/observables", "get"): "Control system worker unresponsive",
    ("/environments/{name}/initial-bounds", "get"): "Control system worker unresponsive",
    ("/environments/{name}/evaluate", "post"): "Control system worker unresponsive",
    ("/jobs", "get"): "Job-state backing store unavailable",
    ("/results", "get"): "Results backing store unavailable",
    ("/models", "get"): "Model backing store unavailable",
}


def _json_error(description: str) -> dict[str, Any]:
    """An error response documented with the shared ``ErrorResponse`` schema."""
    return {
        "description": description,
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}},
    }


def custom_openapi() -> dict[str, Any]:
    """Generate the OpenAPI schema, then reconcile it with what handlers really return.

    FastAPI infers only the happy-path 2xx and its own request-validation 422.
    Our handlers also raise string-detail HTTPExceptions (404 / 400 / 503) and
    return a string ``detail`` for semantic 422s. Document those so the published
    contract matches runtime — otherwise typed clients and schema fuzzers see
    every such response as a violation.
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    components.setdefault(
        "ErrorResponse",
        {"type": "object", "properties": {"detail": {"type": "string"}}, "required": ["detail"]},
    )

    # Manual 422s return {"detail": "<message>"} while FastAPI's request-validation
    # 422 returns {"detail": [ValidationError, ...]}. Allow both in the shared model.
    http_val = components.get("HTTPValidationError", {})
    detail = http_val.get("properties", {}).get("detail")
    if detail is not None and "anyOf" not in detail:
        http_val["properties"]["detail"] = {
            "anyOf": [detail, {"type": "string"}],
            "title": detail.get("title", "Detail"),
        }

    for path, methods in schema["paths"].items():
        for method, op in methods.items():
            if not isinstance(op, dict) or "responses" not in op:
                continue
            responses = op["responses"]
            if any(p.get("in") == "path" for p in op.get("parameters", [])):
                responses.setdefault("404", _json_error("Resource not found"))
            if "requestBody" in op:
                responses.setdefault("400", _json_error("Malformed request body"))
            description_503 = _OPS_503.get((path, method))
            if description_503 is not None:
                responses.setdefault("503", _json_error(description_503))

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi  # type: ignore[method-assign]
