"""Unit tests for the Bluesky bridge's launch-token gate."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from osprey.services.bluesky_bridge.security import require_armed, verify_launch_token

_ENV_VAR = "BLUESKY_LAUNCH_TOKEN"


def test_require_armed_raises_503_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV_VAR, raising=False)
    with pytest.raises(HTTPException) as exc_info:
        require_armed()
    assert exc_info.value.status_code == 503


def test_require_armed_error_does_not_leak_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV_VAR, raising=False)
    with pytest.raises(HTTPException) as exc_info:
        require_armed()
    detail = str(exc_info.value.detail)
    assert _ENV_VAR in detail
    # No token value exists to leak when unset, but guard against ever
    # embedding one (e.g. an accidental default) in the error text.
    assert "token=" not in detail.lower()


def test_require_armed_raises_503_when_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV_VAR, "")
    with pytest.raises(HTTPException) as exc_info:
        require_armed()
    assert exc_info.value.status_code == 503


def test_require_armed_returns_configured_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV_VAR, "s3cr3t")
    assert require_armed() == "s3cr3t"


def test_verify_launch_token_passes_on_matching_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV_VAR, "s3cr3t")
    verify_launch_token("s3cr3t")  # must not raise


def test_verify_launch_token_raises_403_on_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV_VAR, "s3cr3t")
    with pytest.raises(HTTPException) as exc_info:
        verify_launch_token("wrong-token")
    assert exc_info.value.status_code == 403


def test_verify_launch_token_raises_403_on_missing_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV_VAR, "s3cr3t")
    with pytest.raises(HTTPException) as exc_info:
        verify_launch_token(None)
    assert exc_info.value.status_code == 403


def test_verify_launch_token_error_does_not_leak_expected_or_received(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_ENV_VAR, "s3cr3t")
    with pytest.raises(HTTPException) as exc_info:
        verify_launch_token("guessed-wrong")
    detail = str(exc_info.value.detail).lower()
    assert "s3cr3t" not in detail
    assert "guessed-wrong" not in detail


def test_verify_launch_token_raises_503_when_unarmed_even_with_a_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_ENV_VAR, raising=False)
    with pytest.raises(HTTPException) as exc_info:
        verify_launch_token("anything")
    assert exc_info.value.status_code == 503
