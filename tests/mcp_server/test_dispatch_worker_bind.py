"""Tests for the dispatch worker's bind-host resolution.

``DISPATCH_WORKER_BIND`` controls the host uvicorn binds to. It defaults to
``0.0.0.0`` (today's bridge behavior); host-mode renders set it to
``127.0.0.1``, and genuinely remote callers can override it explicitly via
service config.
"""

from unittest.mock import patch

from osprey.mcp_server.dispatch_worker.__main__ import main


def _run_main_and_capture_host(monkeypatch, env_value: str | None) -> str:
    if env_value is None:
        monkeypatch.delenv("DISPATCH_WORKER_BIND", raising=False)
    else:
        monkeypatch.setenv("DISPATCH_WORKER_BIND", env_value)

    with patch("osprey.mcp_server.dispatch_worker.__main__.uvicorn.run") as mock_run:
        main()

    assert mock_run.call_count == 1
    _, kwargs = mock_run.call_args
    return kwargs["host"]


def test_bind_defaults_to_all_interfaces(monkeypatch):
    """No override preserves today's bridge behavior of binding 0.0.0.0."""
    host = _run_main_and_capture_host(monkeypatch, None)
    assert host == "0.0.0.0"


def test_bind_honors_loopback_override(monkeypatch):
    """Host-mode renders set DISPATCH_WORKER_BIND=127.0.0.1."""
    host = _run_main_and_capture_host(monkeypatch, "127.0.0.1")
    assert host == "127.0.0.1"


def test_bind_honors_explicit_override(monkeypatch):
    """Remote callers can point the worker at any explicit bind address."""
    host = _run_main_and_capture_host(monkeypatch, "10.0.0.5")
    assert host == "10.0.0.5"
