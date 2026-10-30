"""Every mutating operator route of the web terminal takes a JSON body only.

A form is the one way a page can make a credentialed write without running a
line of script, so the script-blocking CSP on the console's JupyterLab does not
stop it, and a form posted from a page on the console's own origin carries that
origin and passes the Origin check. What a form cannot choose is its body type:
``application/x-www-form-urlencoded``, ``multipart/form-data`` or
``text/plain``. Each of those is refused on every mutating route this app
answers itself, including the routes that read no body at all (a form posted to
``/api/terminal/restart`` needs no body to do its damage).

The routes are read off the booted app rather than listed, so a mutating route
added later is covered without anyone remembering this file exists. Paths under
``/panel/`` are relayed to the panel's own app and are left out: a first-party
panel app runs the same gate on its own routes.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest
from fastapi.routing import APIRoute

from osprey.interfaces.common_middleware import SAFE_METHODS, session_cookie_name

FORM_TYPES = (
    "application/x-www-form-urlencoded",
    "multipart/form-data; boundary=----osprey",
    "text/plain;charset=UTF-8",
)

REFUSAL = {"detail": "this route accepts a JSON body only"}

_PARAM = re.compile(r"\{[^}]+\}")


def _api_routes(routes) -> Iterator[APIRoute]:
    """Every :class:`APIRoute` under *routes*, descending into included routers.

    FastAPI keeps an included router as one lazy entry that holds the original
    router, so a flat read of ``app.routes`` sees almost nothing. No router in
    this app is included with a prefix, so each route's own path is the path it
    answers; :func:`test_the_sweep_agrees_with_the_schema` checks that.
    """
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
        inner = getattr(route, "original_router", None)
        if inner is not None:
            yield from _api_routes(inner.routes)


def _mutating_routes(app) -> list[tuple[str, str]]:
    """``(method, concrete path)`` for every mutating route the app answers itself."""
    found: list[tuple[str, str]] = []
    for route in _api_routes(app.routes):
        if route.path.startswith("/panel/"):
            continue
        for method in sorted(route.methods - SAFE_METHODS):
            found.append((method, _PARAM.sub("probe", route.path)))
    return found


def test_the_sweep_sees_the_routes_it_exists_for(isolated_app):
    """A sweep that found nothing would pass vacuously; pin the ones that matter."""
    with isolated_app() as client:
        routes = set(_mutating_routes(client.app))
    for expected in [
        ("POST", "/api/terminal/posture"),
        ("POST", "/api/terminal/restart"),
        ("POST", "/api/terminal/logout"),
        ("PATCH", "/api/config"),
        ("POST", "/api/chat"),
    ]:
        assert expected in routes, f"{expected} is not a mutating route of the app"


def test_the_sweep_agrees_with_the_schema(isolated_app):
    """Every mutating operation the OpenAPI schema lists is one the sweep visits."""
    with isolated_app() as client:
        swept = {
            (method, _PARAM.sub("probe", path)) for method, path in _mutating_routes(client.app)
        }
        schema = client.app.openapi()["paths"]
    listed = {
        (method.upper(), _PARAM.sub("probe", path))
        for path, operations in schema.items()
        if not path.startswith("/panel/")
        for method in operations
        if method.upper() not in SAFE_METHODS
    }
    assert listed, "the schema lists no mutating operation"
    assert listed <= swept, f"in the schema but not swept: {sorted(listed - swept)}"


@pytest.mark.parametrize("content_type", FORM_TYPES)
def test_every_mutating_route_refuses_a_form_body(isolated_app, content_type):
    with isolated_app() as client:
        routes = _mutating_routes(client.app)
        answered = {
            (method, path): client.request(
                method, path, content=b"a=1", headers={"content-type": content_type}
            )
            for method, path in routes
        }
    wrong = {
        key: (response.status_code, response.text)
        for key, response in answered.items()
        if response.status_code != 415 or response.json() != REFUSAL
    }
    assert not wrong, f"routes that did not refuse a {content_type} body: {wrong}"


@pytest.mark.no_auth_seam
def test_a_same_origin_cookie_form_post_is_refused(isolated_app):
    """The vector itself: a form on the console's own origin, the cookie attached.

    Out of the auth seam, so the cookie is the only credential the request carries.
    """
    with isolated_app() as client:
        session = client.app.state.web_credentials.create_session()
        client.cookies.set(session_cookie_name(), session)
        response = client.post(
            "/api/terminal/restart",
            content=b"",
            headers={
                "content-type": "application/x-www-form-urlencoded",
                "origin": "http://testserver",
            },
        )
    assert response.status_code == 415
    assert response.json() == REFUSAL
