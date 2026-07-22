"""Unit tests for the translation-proxy lifecycle (start/stop/port/detection).

These cover the module-level singleton in ``proxy.lifecycle`` without starting a
real uvicorn server or binding a long-lived port: ``create_proxy_app`` and the
``uvicorn`` Config/Server pair are faked so ``start_proxy`` exercises its state
machine (idempotent start, readiness wait, clean stop) deterministically. A
fixture snapshots and restores the shared ``_state`` dict so the serial unit
lane never leaks a "running" proxy between tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import osprey.infrastructure.proxy.lifecycle as lifecycle


@pytest.fixture
def clean_proxy_state():
    """Isolate the module-global proxy state for a single test."""
    saved = dict(lifecycle._state)
    lifecycle._state.update({"server": None, "thread": None, "port": None})
    try:
        yield
    finally:
        lifecycle._state.update({"server": None, "thread": None, "port": None})
        lifecycle._state.update(saved)


class _FakeServer:
    """Stand-in for ``uvicorn.Server`` that never touches a socket."""

    def __init__(self, config):
        self.config = config
        self.started = True
        self.should_exit = False
        self.run_count = 0

    def run(self):  # invoked in the daemon thread; must return promptly
        self.run_count += 1


def _install_fake_uvicorn(monkeypatch, server_cls=_FakeServer):
    """Patch ``create_proxy_app`` and the uvicorn Config/Server constructors.

    Returns the MagicMock standing in for ``create_proxy_app`` so tests can
    assert how many times (and with what upstream) the app was built.
    """
    import uvicorn

    app_factory = MagicMock(return_value=object())
    monkeypatch.setattr(
        "osprey.infrastructure.proxy.app.create_proxy_app", app_factory, raising=True
    )
    monkeypatch.setattr(uvicorn, "Config", lambda app=None, **kw: {"app": app, **kw})
    monkeypatch.setattr(uvicorn, "Server", server_cls)
    return app_factory


# ---------------------------------------------------------------------------
# is_proxy_needed — provider protocol detection
# ---------------------------------------------------------------------------


class TestIsProxyNeeded:
    @pytest.mark.parametrize("provider", ["anthropic", "cborg", "als-apg"])
    def test_native_providers_never_need_proxy(self, provider):
        assert lifecycle.is_proxy_needed(provider) is False

    def test_unknown_provider_needs_proxy(self):
        assert lifecycle.is_proxy_needed("some-openai-clone") is True

    def test_explicit_anthropic_protocol_skips_proxy(self):
        api = {"custom": {"api_protocol": "anthropic"}}
        assert lifecycle.is_proxy_needed("custom", api_providers=api) is False

    def test_non_anthropic_protocol_needs_proxy(self):
        api = {"custom": {"api_protocol": "openai"}}
        assert lifecycle.is_proxy_needed("custom", api_providers=api) is True

    def test_provider_absent_from_config_needs_proxy(self):
        api = {"other": {"api_protocol": "anthropic"}}
        assert lifecycle.is_proxy_needed("custom", api_providers=api) is True


# ---------------------------------------------------------------------------
# find_free_port
# ---------------------------------------------------------------------------


class TestFindFreePort:
    def test_returns_a_bindable_port(self):
        import socket

        port = lifecycle.find_free_port()
        assert isinstance(port, int)
        assert 1 <= port <= 65535

        # The OS just handed it back as free, so we must be able to bind it.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))


# ---------------------------------------------------------------------------
# start_proxy / stop_proxy / get_proxy_url
# ---------------------------------------------------------------------------


class TestStartStop:
    def test_start_returns_port_and_populates_state(self, monkeypatch, clean_proxy_state):
        app_factory = _install_fake_uvicorn(monkeypatch)

        port = lifecycle.start_proxy("https://up.example/v1", upstream_api_key="k")

        assert isinstance(port, int)
        assert lifecycle._state["port"] == port
        assert isinstance(lifecycle._state["server"], _FakeServer)
        assert lifecycle.get_proxy_url() == f"http://127.0.0.1:{port}"
        app_factory.assert_called_once_with("https://up.example/v1", "k")

    def test_start_is_idempotent(self, monkeypatch, clean_proxy_state):
        app_factory = _install_fake_uvicorn(monkeypatch)

        first = lifecycle.start_proxy("https://up.example/v1")
        server = lifecycle._state["server"]
        second = lifecycle.start_proxy("https://up.example/v1")

        assert first == second
        # Repeated calls must not rebuild the app or swap the server out.
        app_factory.assert_called_once()
        assert lifecycle._state["server"] is server

    def test_start_waits_for_server_readiness(self, monkeypatch, clean_proxy_state):
        class _SlowStartServer(_FakeServer):
            def __init__(self, config):
                super().__init__(config)
                self._checks = 0

            @property
            def started(self):  # type: ignore[override]
                self._checks += 1
                return self._checks > 2

            @started.setter
            def started(self, _value):
                # ``_FakeServer.__init__`` assigns ``started``; ignore it since
                # this subclass computes readiness from the probe count.
                pass

        _install_fake_uvicorn(monkeypatch, server_cls=_SlowStartServer)
        sleep = MagicMock()
        monkeypatch.setattr(lifecycle.time, "sleep", sleep)

        port = lifecycle.start_proxy("https://up.example/v1")

        assert lifecycle._state["port"] == port
        # It polled ``started`` and slept while the server was still coming up.
        assert sleep.called

    def test_stop_shuts_down_and_clears_state(self, monkeypatch, clean_proxy_state):
        _install_fake_uvicorn(monkeypatch)
        lifecycle.start_proxy("https://up.example/v1")
        server = lifecycle._state["server"]

        lifecycle.stop_proxy()

        assert server.should_exit is True
        assert lifecycle._state["server"] is None
        assert lifecycle._state["thread"] is None
        assert lifecycle._state["port"] is None
        assert lifecycle.get_proxy_url() is None

    def test_stop_when_not_running_is_a_noop(self, clean_proxy_state):
        # Nothing started — stop must not raise and state stays empty.
        lifecycle.stop_proxy()
        assert lifecycle._state["server"] is None
        assert lifecycle.get_proxy_url() is None

    def test_get_proxy_url_none_before_start(self, clean_proxy_state):
        assert lifecycle.get_proxy_url() is None
