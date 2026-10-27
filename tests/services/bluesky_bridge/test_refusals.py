"""The one constructor every bridge refusal body is made by."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from osprey.services.bluesky_bridge import queue_backend as qb
from osprey.services.bluesky_bridge.refusals import refusal
from osprey.utils import launch_token


def test_an_undeclared_code_fails_where_it_is_made() -> None:
    with pytest.raises(ValueError, match="not_a_declared_code"):
        refusal("not_a_declared_code", "why")


@pytest.mark.parametrize("key", ["code", "detail", "retryable"])
def test_an_extra_cannot_override_the_refusal_keys(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        refusal(qb.REASON_NO_DRAFT, "why", **{key: "other"})


@pytest.mark.parametrize("code", sorted(qb.REFUSAL_REASONS))
def test_every_declared_code_answers_with_its_row(code: str) -> None:
    """Every queue refusal also carries ``retryable``, and its status is the row's."""
    row = qb.REFUSAL_REASONS[code]

    exc = refusal(code, "why", plan="my_scan")

    assert isinstance(exc, HTTPException)
    assert exc.status_code == row.status
    assert exc.detail == {
        "code": code,
        "detail": "why",
        "retryable": row.retryable,
        "plan": "my_scan",
    }
    assert exc.detail["retryable"] is row.retryable


@pytest.mark.parametrize(
    ("code", "env_value", "presented"),
    [
        pytest.param(launch_token.LAUNCH_TOKEN_REQUIRED, "", "anything", id="nothing-configured"),
        pytest.param(launch_token.LAUNCH_TOKEN_INVALID, "secret", "", id="missing"),
        pytest.param(launch_token.LAUNCH_TOKEN_INVALID, "secret", "wrong", id="wrong"),
    ],
)
def test_each_launch_token_code_is_declared_with_the_status_the_check_raises(
    monkeypatch: pytest.MonkeyPatch, code: str, env_value: str, presented: str
) -> None:
    monkeypatch.setenv("TEST_REFUSALS_LAUNCH_TOKEN", env_value)

    with pytest.raises(HTTPException) as raised:
        launch_token.verify_launch_token(presented, "TEST_REFUSALS_LAUNCH_TOKEN")

    assert raised.value.detail["code"] == code
    assert raised.value.status_code == qb.REFUSAL_REASONS[code].status
