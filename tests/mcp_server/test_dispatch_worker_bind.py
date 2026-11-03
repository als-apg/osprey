"""Tests for the dispatch worker's bind-host resolution.

``DISPATCH_WORKER_BIND`` controls the host uvicorn binds to. It defaults to
``0.0.0.0`` (today's bridge behavior); host-mode renders set it to
``127.0.0.1``, and genuinely remote callers can override it explicitly via
service config.
"""

from unittest.mock import patch

import pytest

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


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        # No override preserves today's bridge behavior of binding 0.0.0.0.
        pytest.param(None, "0.0.0.0", id="defaults-to-all-interfaces"),
        # Host-mode renders set DISPATCH_WORKER_BIND=127.0.0.1.
        pytest.param("127.0.0.1", "127.0.0.1", id="loopback-override"),
        # Remote callers can point the worker at any explicit bind address.
        pytest.param("10.0.0.5", "10.0.0.5", id="explicit-override"),
    ],
)
def test_bind_follows_the_override_else_all_interfaces(monkeypatch, env_value, expected):
    host = _run_main_and_capture_host(monkeypatch, env_value)
    assert host == expected
