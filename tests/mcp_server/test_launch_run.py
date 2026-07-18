"""Unit tests for the launch_run MCP tool — the sole bluesky write path.

Covers both in-tool safety layers (writes_enabled re-read, client-side token
presence) and the bridge response mapping for ``POST /draft/run``. The HTTP
boundary (``_http_post_json``) is patched here so these run with no Bluesky
bridge process and no network.
"""

from unittest.mock import patch

import pytest
import yaml

from osprey.mcp_server.bluesky.server_context import initialize_server_context, reset_server_context
from osprey.mcp_server.bluesky.tools import launch
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.bluesky.tools.launch"


def _fn():
    return get_tool_fn(launch.launch_run)


def _write_config(tmp_path, writes_enabled: bool, launch_token: str | None = None) -> None:
    config: dict = {"control_system": {"writes_enabled": writes_enabled}}
    if launch_token is not None:
        config["bluesky"] = {"launch_token": launch_token}
    (tmp_path / "config.yml").write_text(yaml.dump(config))


@pytest.fixture(autouse=True)
def _reset_server_context():
    yield
    reset_server_context()


# ── Layer 1: in-tool writes_enabled re-check (authoritative) ────────────────


async def test_refuses_when_writes_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=False)
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "valid-token")
    initialize_server_context()

    with patch(f"{_MOD}._http_post_json") as m:
        with assert_raises_error(error_type="writes_disabled"):
            await _fn()(draft_revision=7)
    m.assert_not_called()


async def test_writes_disabled_refusal_does_not_leak_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=False)
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "s3cr3t-token")
    initialize_server_context()

    with patch(f"{_MOD}._http_post_json"):
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await _fn()(draft_revision=7)
    assert "s3cr3t-token" not in ctx["envelope"]["error_message"]


async def test_writes_disabled_wins_even_with_no_token(tmp_path, monkeypatch):
    """writes_enabled is checked FIRST — order matters, per the module contract."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=False)
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)
    initialize_server_context()

    with assert_raises_error(error_type="writes_disabled"):
        await _fn()(draft_revision=7)


async def test_writes_enabled_missing_config_fails_closed(tmp_path, monkeypatch):
    """No config.yml at all -> writes_enabled defaults False (fail-closed)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "valid-token")
    initialize_server_context()

    with assert_raises_error(error_type="writes_disabled"):
        await _fn()(draft_revision=7)


# ── Layer 2: client-side token presence (no network call) ──────────────────


async def test_refuses_client_side_when_no_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)
    initialize_server_context()

    with patch(f"{_MOD}._http_post_json") as m:
        with assert_raises_error(error_type="run_launch_unarmed"):
            await _fn()(draft_revision=7)
    m.assert_not_called()


# ── Success path + bridge response mapping ──────────────────────────────────


async def test_launch_run_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "valid-token")
    initialize_server_context()

    body = {"id": "abc123", "status": "running", "launched_by": "draft"}
    with patch(f"{_MOD}._http_post_json", return_value=(200, body)) as m:
        result = await _fn()(draft_revision=7)

    assert m.call_args.args[0] == "/draft/run"
    assert m.call_args.args[1] == {"draft_revision": 7}
    assert m.call_args.kwargs["headers"] == {"X-Launch-Token": "valid-token"}
    data = extract_response_dict(result)
    assert data["status"] == "running"
    assert data["launched_by"] == "draft"


async def test_launch_run_forbidden_token_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "valid-token")
    initialize_server_context()

    with patch(
        f"{_MOD}._http_post_json",
        return_value=(403, {"detail": "invalid or missing launch token"}),
    ):
        with assert_raises_error(error_type="run_launch_forbidden"):
            await _fn()(draft_revision=7)


async def test_launch_run_conflict_stale_revision(tmp_path, monkeypatch):
    """409 stale_draft_revision: the bridge's code + fresh revision surface verbatim."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "valid-token")
    initialize_server_context()

    with patch(
        f"{_MOD}._http_post_json",
        return_value=(
            409,
            {
                "detail": {
                    "code": "stale_draft_revision",
                    "detail": "draft revision 7 is stale",
                    "revision": 8,
                }
            },
        ),
    ):
        with assert_raises_error(error_type="run_launch_conflict") as ctx:
            await _fn()(draft_revision=7)
    envelope = ctx["envelope"]
    assert envelope["details"]["code"] == "stale_draft_revision"
    assert envelope["details"]["revision"] == 8
    assert "stale" in envelope["error_message"]


async def test_launch_run_conflict_already_launched(tmp_path, monkeypatch):
    """409 draft_revision_already_launched: code + revision surface verbatim, distinct hint."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "valid-token")
    initialize_server_context()

    with patch(
        f"{_MOD}._http_post_json",
        return_value=(
            409,
            {
                "detail": {
                    "code": "draft_revision_already_launched",
                    "detail": "revision 7 already launched",
                    "revision": 7,
                }
            },
        ),
    ):
        with assert_raises_error(error_type="run_launch_conflict") as ctx:
            await _fn()(draft_revision=7)
    envelope = ctx["envelope"]
    assert envelope["details"]["code"] == "draft_revision_already_launched"
    assert envelope["details"]["revision"] == 7
    assert any("set_draft" in s for s in envelope["suggestions"])


async def test_launch_run_conflict_string_detail(tmp_path, monkeypatch):
    """A 409 with a plain-string detail (e.g. validation gate) still maps to run_launch_conflict."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "valid-token")
    initialize_server_context()

    with patch(
        f"{_MOD}._http_post_json",
        return_value=(409, {"detail": "no passing validation record for this plan"}),
    ):
        with assert_raises_error(error_type="run_launch_conflict") as ctx:
            await _fn()(draft_revision=7)
    assert "no passing validation record" in ctx["envelope"]["error_message"]
    assert "details" not in ctx["envelope"]


async def test_launch_run_bridge_unarmed(tmp_path, monkeypatch):
    """The bridge process itself has no BLUESKY_LAUNCH_TOKEN configured (503)."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "valid-token")
    initialize_server_context()

    with patch(
        f"{_MOD}._http_post_json",
        return_value=(
            503,
            {"detail": "launch disabled: BLUESKY_LAUNCH_TOKEN is not configured"},
        ),
    ):
        with assert_raises_error(error_type="run_launch_unarmed") as ctx:
            await _fn()(draft_revision=7)
    assert "not configured" in ctx["envelope"]["error_message"]


async def test_launch_bluesky_bridge_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "valid-token")
    initialize_server_context()

    with patch(f"{_MOD}._http_post_json", return_value=(500, {"detail": "launch failed: boom"})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _fn()(draft_revision=7)
    assert "boom" in ctx["envelope"]["error_message"]


async def test_old_promote_error_codes_are_gone(tmp_path, monkeypatch):
    """The renamed refusal codes replace the old run_promote_* codes entirely."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, writes_enabled=True)
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)
    initialize_server_context()

    with patch(f"{_MOD}._http_post_json"):
        with assert_raises_error() as ctx:
            await _fn()(draft_revision=7)
    assert not ctx["envelope"]["error_type"].startswith("run_promote_")
