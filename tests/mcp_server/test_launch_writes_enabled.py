"""Safety proof: launch_scan refuses when writes are disabled, even with a
VALID promote token — zero HTTP calls issued.

This is the authoritative in-tool guard (see ``scan/tools/launch.py``): the
``control_system.writes_enabled`` re-check runs BEFORE any HTTP call, so a
caller that bypasses the PreToolUse approval hook entirely (e.g. a direct MCP
call) and supplies a correct ``BLUESKY_PROMOTE_TOKEN`` is still refused. General
unit coverage of ``launch_scan``'s error mapping lives in
``test_launch_scan.py``; this file exists as the standalone, easy-to-locate
proof of the one property that matters most: writes-disabled beats a valid
token, unconditionally, with no network call ever issued.

Do NOT relax this test. It is the load-bearing safety contract for the scan
write path (analogous to ``tests/e2e/test_query_write_refused_e2e.py`` for
``channel_write``).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml

from osprey.mcp_server.scan.server_context import initialize_server_context, reset_server_context
from osprey.mcp_server.scan.tools.launch import launch_scan
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.scan.tools.launch"


@pytest.fixture(autouse=True)
def _reset_scan_context():
    yield
    reset_server_context()


async def test_writes_disabled_refuses_launch_with_valid_token_and_zero_http_calls(
    tmp_path, monkeypatch
):
    """The load-bearing case: writes_enabled=false + a VALID token -> refused, no network call.

    A malicious or hook-bypassed caller holding a correct BLUESKY_PROMOTE_TOKEN
    must not be able to promote a run when this deployment has writes
    disabled. The in-tool re-check must reject before ``_http_post_json`` is
    ever invoked — asserted directly here, not inferred from the error type.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(yaml.dump({"control_system": {"writes_enabled": False}}))
    # A genuinely VALID token — the point is that even a correct credential
    # does not help once writes are disabled.
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "genuinely-valid-token")
    initialize_server_context()

    fn = get_tool_fn(launch_scan)
    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await fn(run_id="some-run-id")

    mock_post.assert_not_called()
    assert "genuinely-valid-token" not in ctx["envelope"]["error_message"]


async def test_writes_disabled_refuses_launch_even_without_a_token(tmp_path, monkeypatch):
    """Non-regression: the writes-disabled refusal doesn't depend on token presence either way."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(yaml.dump({"control_system": {"writes_enabled": False}}))
    monkeypatch.delenv("BLUESKY_PROMOTE_TOKEN", raising=False)
    initialize_server_context()

    fn = get_tool_fn(launch_scan)
    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="writes_disabled"):
            await fn(run_id="some-run-id")

    mock_post.assert_not_called()


async def test_writes_enabled_with_valid_token_does_reach_the_bridge(tmp_path, monkeypatch):
    """Contrast case: once writes are enabled, a valid token does reach the HTTP layer.

    Confirms the refusal above is genuinely gated on writes_enabled and not
    some unrelated failure that would make the "zero HTTP calls" assertion
    vacuously true.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(yaml.dump({"control_system": {"writes_enabled": True}}))
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "genuinely-valid-token")
    initialize_server_context()

    fn = get_tool_fn(launch_scan)
    with patch(
        f"{_MOD}._http_post_json", return_value=(200, {"id": "x", "status": "running"})
    ) as m:
        await fn(run_id="some-run-id")

    m.assert_called_once()


async def test_missing_config_fails_closed_even_with_valid_token(tmp_path, monkeypatch):
    """No config.yml at all -> writes_enabled defaults False (fail-closed), token notwithstanding.

    A project directory with no config.yml (e.g. a broken deployment, or a
    cwd mismatch) must never be interpreted as "writes enabled" by omission —
    ``_writes_enabled``'s except clause defaults to False on
    FileNotFoundError/RuntimeError, mirroring ``ControlSystemConnector._writes_enabled``.
    """
    monkeypatch.chdir(tmp_path)  # no config.yml written here
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "genuinely-valid-token")
    initialize_server_context()

    fn = get_tool_fn(launch_scan)
    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="writes_disabled"):
            await fn(run_id="some-run-id")

    mock_post.assert_not_called()


async def test_writes_enabled_true_but_token_unset_refused_client_side(tmp_path, monkeypatch):
    """The second gate: writes_enabled=true is not sufficient on its own.

    With no BLUESKY_PROMOTE_TOKEN configured for this MCP server, launch_scan
    must still refuse — client-side, before any HTTP call — rather than
    sending a promote request with no credential at all.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(yaml.dump({"control_system": {"writes_enabled": True}}))
    monkeypatch.delenv("BLUESKY_PROMOTE_TOKEN", raising=False)
    initialize_server_context()

    fn = get_tool_fn(launch_scan)
    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="scan_promote_unarmed"):
            await fn(run_id="some-run-id")

    mock_post.assert_not_called()
