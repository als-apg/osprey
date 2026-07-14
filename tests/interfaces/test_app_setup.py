"""Tests for the shared interface app-setup helper."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces.common_middleware import (
    ExceptionLoggingMiddleware,
    NoCacheStaticMiddleware,
)


def _configure(static_dir) -> FastAPI:
    app = FastAPI()
    configure_interface_app(app, static_dir=static_dir)
    return app


class TestStaticMounts:
    def test_all_three_mounts_present(self, tmp_path):
        app = _configure(tmp_path)
        mount_paths = {route.path for route in app.routes if hasattr(route, "app")}
        assert "/static/fonts" in mount_paths
        assert "/design-system" in mount_paths
        assert "/static" in mount_paths

    def test_static_mount_registered_last(self, tmp_path):
        # The /static catch-all must come after the more specific mounts so
        # Starlette (declaration-order matching) resolves /static/fonts first.
        app = _configure(tmp_path)
        mount_paths = [route.path for route in app.routes if hasattr(route, "app")]
        assert mount_paths.index("/static/fonts") < mount_paths.index("/static")
        assert mount_paths.index("/design-system") < mount_paths.index("/static")

    def test_accepts_str_static_dir(self, tmp_path):
        app = _configure(str(tmp_path))
        mount_paths = {route.path for route in app.routes if hasattr(route, "app")}
        assert "/static" in mount_paths


class TestMiddleware:
    def test_nocache_and_exception_middleware_present(self, tmp_path):
        app = _configure(tmp_path)
        classes = {mw.cls for mw in app.user_middleware}
        assert NoCacheStaticMiddleware in classes
        assert ExceptionLoggingMiddleware in classes

    def test_cors_middleware_present(self, tmp_path):
        app = _configure(tmp_path)
        classes = {mw.cls for mw in app.user_middleware}
        assert CORSMiddleware in classes

    def test_cors_allow_credentials_not_enabled(self, tmp_path):
        app = _configure(tmp_path)
        cors = next(mw for mw in app.user_middleware if mw.cls is CORSMiddleware)
        # allow_credentials must be left at Starlette's default (False): it is
        # never passed, so it must not appear as a truthy kwarg.
        assert not cors.kwargs.get("allow_credentials", False)
        assert cors.kwargs.get("allow_origins") == ["*"]
