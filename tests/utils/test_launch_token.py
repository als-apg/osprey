"""The shared launch-token check: per-request env read, coded refusals, byte compare."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from osprey.utils.launch_token import (
    LAUNCH_TOKEN_INVALID,
    LAUNCH_TOKEN_REQUIRED,
    require_launch_token,
    verify_launch_token,
)

_ENV = "EXAMPLE_LAUNCH_TOKEN"
_OTHER_ENV = "OTHER_LAUNCH_TOKEN"


def _refusal(excinfo: pytest.ExceptionInfo[HTTPException]) -> tuple[int, dict]:
    return excinfo.value.status_code, excinfo.value.detail


def test_codes_are_the_published_names() -> None:
    assert LAUNCH_TOKEN_REQUIRED == "launch_token_required"
    assert LAUNCH_TOKEN_INVALID == "launch_token_invalid"


@pytest.mark.parametrize("value", [None, "", "   ", "\t\n"])
def test_unset_or_blank_is_503_launch_token_required(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, value)
    with pytest.raises(HTTPException) as excinfo:
        require_launch_token(_ENV)
    status, detail = _refusal(excinfo)
    assert status == 503
    assert detail["code"] == LAUNCH_TOKEN_REQUIRED
    assert _ENV in detail["detail"]


@pytest.mark.parametrize("presented", [None, "", "tok", "anything"])
def test_unconfigured_refuses_503_whatever_is_presented(
    monkeypatch: pytest.MonkeyPatch, presented: str | None
) -> None:
    monkeypatch.setenv(_ENV, "  ")
    with pytest.raises(HTTPException) as excinfo:
        verify_launch_token(presented, _ENV)
    assert _refusal(excinfo)[0] == 503
    assert _refusal(excinfo)[1]["code"] == LAUNCH_TOKEN_REQUIRED


def test_require_returns_the_configured_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "s3cr3t")
    assert require_launch_token(_ENV) == "s3cr3t"


def test_matching_token_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "s3cr3t")
    verify_launch_token("s3cr3t", _ENV)


@pytest.mark.parametrize("presented", [None, "", "wrong", "s3cr3t ", "S3CR3T"])
def test_missing_or_wrong_is_403_launch_token_invalid(
    monkeypatch: pytest.MonkeyPatch, presented: str | None
) -> None:
    monkeypatch.setenv(_ENV, "s3cr3t")
    with pytest.raises(HTTPException) as excinfo:
        verify_launch_token(presented, _ENV)
    status, detail = _refusal(excinfo)
    assert status == 403
    assert detail["code"] == LAUNCH_TOKEN_INVALID


def test_refusals_never_echo_either_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "expected-value-xyz")
    with pytest.raises(HTTPException) as excinfo:
        verify_launch_token("guessed-value-abc", _ENV)
    text = repr(excinfo.value.detail)
    assert "expected-value-xyz" not in text
    assert "guessed-value-abc" not in text


def test_env_is_read_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "first")
    verify_launch_token("first", _ENV)
    monkeypatch.setenv(_ENV, "second")
    with pytest.raises(HTTPException) as excinfo:
        verify_launch_token("first", _ENV)
    assert _refusal(excinfo)[0] == 403
    verify_launch_token("second", _ENV)
    monkeypatch.delenv(_ENV)
    with pytest.raises(HTTPException) as excinfo:
        verify_launch_token("second", _ENV)
    assert _refusal(excinfo)[0] == 503


def test_env_name_selects_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "one")
    monkeypatch.setenv(_OTHER_ENV, "two")
    verify_launch_token("one", _ENV)
    verify_launch_token("two", _OTHER_ENV)
    with pytest.raises(HTTPException) as excinfo:
        verify_launch_token("one", _OTHER_ENV)
    assert _refusal(excinfo)[0] == 403


def test_non_ascii_header_is_refused_not_crashed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A header arrives as arbitrary latin-1 text; the str compare_digest would
    # raise TypeError (a 500) on it. The byte compare refuses it instead.
    monkeypatch.setenv(_ENV, "s3cr3t")
    with pytest.raises(HTTPException) as excinfo:
        verify_launch_token("s3cr3té", _ENV)
    assert _refusal(excinfo)[0] == 403


def test_non_ascii_configured_token_matches_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "tokén")
    verify_launch_token("tokén", _ENV)
