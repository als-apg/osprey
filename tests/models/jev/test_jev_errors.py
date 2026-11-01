"""The Jev error tree and the pure response-to-error mapper."""

from __future__ import annotations

import json
import logging
import traceback

import pytest

from osprey.models.jev import errors as jev_errors
from osprey.models.jev.errors import (
    DecisionAuthError,
    DecisionError,
    DecisionRequestError,
    DecisionUnavailableError,
    IncompleteAnswersError,
    error_from_response,
)

STATE_TEXT = "SECRET-STATE-alpha-bravo-knob-list"


def _json(obj: object) -> bytes:
    return json.dumps(obj).encode()


# ---------------------------------------------------------------------------
# Class tree
# ---------------------------------------------------------------------------


def test_every_error_is_a_decision_error() -> None:
    for cls in (
        DecisionAuthError,
        DecisionRequestError,
        DecisionUnavailableError,
        IncompleteAnswersError,
    ):
        assert issubclass(cls, DecisionError)
    assert issubclass(IncompleteAnswersError, DecisionRequestError)
    assert issubclass(DecisionError, Exception)


def test_error_carries_fields_and_empty_completed_calls_by_default() -> None:
    err = DecisionUnavailableError("gone", status=None, error_type="timeout")
    assert err.status is None
    assert err.error_type == "timeout"
    assert err.request_id is None
    assert err.completed_calls == []
    assert str(err) == "gone"


def test_completed_calls_default_is_not_shared() -> None:
    a = DecisionRequestError("a", error_type="x")
    b = DecisionRequestError("b", error_type="x")
    a.completed_calls.append("call")
    assert b.completed_calls == []


def test_completed_calls_are_kept() -> None:
    calls = ["c1", "c2"]
    err = DecisionAuthError("no", error_type="no_key", completed_calls=calls)
    assert err.completed_calls == ["c1", "c2"]


def test_incomplete_answers_error_has_a_default_error_type() -> None:
    err = IncompleteAnswersError("missing ids")
    assert isinstance(err, DecisionRequestError)
    assert err.error_type == "incomplete_answers"


@pytest.mark.parametrize(
    ("cls", "error_type"),
    [
        (DecisionUnavailableError, "timeout"),
        (DecisionUnavailableError, "transport"),
        (DecisionUnavailableError, "deadline_exceeded"),
        (DecisionRequestError, "state_too_large"),
        (DecisionAuthError, "no_key"),
    ],
)
def test_responseless_failures_are_constructible(cls: type[DecisionError], error_type: str) -> None:
    err = cls("local failure", status=None, error_type=error_type)
    assert err.status is None
    assert err.error_type == error_type


# ---------------------------------------------------------------------------
# Status table
# ---------------------------------------------------------------------------

_OBJ = {"detail": {"error_type": "max_tokens_exceeded", "message": "too many tokens"}}


@pytest.mark.parametrize(
    ("status", "cls"),
    [
        (401, DecisionAuthError),
        (403, DecisionAuthError),
        (400, DecisionRequestError),
        (404, DecisionRequestError),
        (413, DecisionRequestError),
        (422, DecisionRequestError),
        (402, DecisionUnavailableError),
        (408, DecisionUnavailableError),
        (429, DecisionUnavailableError),
        (500, DecisionUnavailableError),
        (502, DecisionUnavailableError),
        (503, DecisionUnavailableError),
        (504, DecisionUnavailableError),
        (524, DecisionUnavailableError),
        (529, DecisionUnavailableError),
    ],
)
def test_status_table(status: int, cls: type[DecisionError]) -> None:
    err = error_from_response(status, {}, _json(_OBJ))
    assert type(err) is cls
    assert err.status == status
    assert err.error_type == "max_tokens_exceeded"
    assert "too many tokens" in str(err)


def test_other_4xx_is_a_request_error() -> None:
    err = error_from_response(409, {}, b"")
    assert type(err) is DecisionRequestError
    assert err.status == 409


def test_html_524_is_unavailable() -> None:
    body = b"<html><body>A timeout occurred</body></html>"
    err = error_from_response(524, {"content-type": "text/html"}, body)
    assert type(err) is DecisionUnavailableError
    assert err.status == 524
    assert err.error_type == "http_524"
    assert "<html>" not in str(err)


@pytest.mark.parametrize("body", [b"not json", b"", None, "plain text"])
def test_non_json_5xx_is_unavailable(body: bytes | str | None) -> None:
    err = error_from_response(503, {}, body)
    assert type(err) is DecisionUnavailableError
    assert err.error_type == "http_503"


# ---------------------------------------------------------------------------
# Detail shapes
# ---------------------------------------------------------------------------


def test_object_detail_uses_its_error_type_and_message() -> None:
    body = {"detail": {"error_type": "max_tokens_exceeded", "message": "limit 8000"}}
    err = error_from_response(400, {}, _json(body))
    assert err.error_type == "max_tokens_exceeded"
    assert "limit 8000" in str(err)


def test_list_detail_is_validation_error_without_input() -> None:
    body = {
        "detail": [
            {
                "type": "string_too_short",
                "loc": ["body", "questions", 0, "instructions"],
                "msg": "String should have at least 1 character",
                "input": STATE_TEXT,
            }
        ]
    }
    err = error_from_response(422, {}, _json(body))
    assert type(err) is DecisionRequestError
    assert err.error_type == "validation_error"
    assert "at least 1 character" in str(err)
    assert STATE_TEXT not in str(err)
    assert STATE_TEXT not in repr(err)


def test_string_detail_is_api_usage_error() -> None:
    body = {"detail": "choice questions accept at most 255 options"}
    err = error_from_response(400, {}, _json(body))
    assert type(err) is DecisionRequestError
    assert err.error_type == "api_usage_error"
    assert "255 options" in str(err)


def test_object_detail_without_error_type_falls_back() -> None:
    err = error_from_response(400, {}, _json({"detail": {"message": "odd"}}))
    assert err.error_type == "http_400"
    assert "odd" in str(err)


# ---------------------------------------------------------------------------
# OpenRouter error objects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "cls"),
    [
        (401, DecisionAuthError),
        (402, DecisionUnavailableError),
        (429, DecisionUnavailableError),
        (400, DecisionRequestError),
        (502, DecisionUnavailableError),
    ],
)
def test_200_with_openrouter_error_maps_by_code(code: int, cls: type[DecisionError]) -> None:
    body = {"id": "gen-dec-abc", "error": {"code": code, "message": "upstream said no"}}
    err = error_from_response(200, {}, _json(body))
    assert type(err) is cls
    assert err.status == code
    assert err.error_type == f"openrouter_{code}"
    assert "upstream said no" in str(err)
    assert err.request_id == "gen-dec-abc"


def test_non_200_with_openrouter_error_keeps_http_status() -> None:
    body = {"error": {"code": 429, "message": "rate limited"}}
    err = error_from_response(429, {}, _json(body))
    assert type(err) is DecisionUnavailableError
    assert err.status == 429
    assert err.error_type == "openrouter_429"


def test_openrouter_string_code_is_accepted() -> None:
    body = {"error": {"code": "401", "message": "bad key"}}
    err = error_from_response(200, {}, _json(body))
    assert type(err) is DecisionAuthError
    assert err.status == 401


def test_openrouter_error_metadata_is_not_in_the_message() -> None:
    body = {
        "error": {
            "code": 400,
            "message": "provider rejected",
            "metadata": {"raw": STATE_TEXT},
        }
    }
    err = error_from_response(200, {}, _json(body))
    assert STATE_TEXT not in str(err)


# ---------------------------------------------------------------------------
# Malformed 200
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        b"{not json",
        b"",
        None,
        _json([1, 2, 3]),
        _json({"answers": [{"id": "q1"}]}),  # answers missing ``type``
        b"\xff\xfe",
    ],
)
def test_malformed_200_is_a_request_error(body: bytes | None) -> None:
    err = error_from_response(200, {}, body)
    assert type(err) is DecisionRequestError
    assert err.status == 200
    assert err.error_type == "malformed_response"


# ---------------------------------------------------------------------------
# Request id
# ---------------------------------------------------------------------------


def test_request_id_prefers_the_header() -> None:
    headers = {"x-typesafe-request-id": "req-123"}
    err = error_from_response(500, headers, _json({"id": "gen-dec-x", "detail": "boom"}))
    assert err.request_id == "req-123"


def test_request_id_header_lookup_is_case_insensitive() -> None:
    headers = {"X-Typesafe-Request-Id": "req-456"}
    err = error_from_response(500, headers, b"oops")
    assert err.request_id == "req-456"


def test_request_id_falls_back_to_body_id() -> None:
    err = error_from_response(502, {}, _json({"id": "gen-dec-9", "detail": "bad gateway"}))
    assert err.request_id == "gen-dec-9"


def test_request_id_absent() -> None:
    err = error_from_response(502, {}, b"")
    assert err.request_id is None


def test_httpx_headers_are_accepted() -> None:
    httpx = pytest.importorskip("httpx")
    headers = httpx.Headers({"X-TYPESAFE-REQUEST-ID": "req-h"})
    err = error_from_response(401, headers, b"")
    assert err.request_id == "req-h"
    assert type(err) is DecisionAuthError


# ---------------------------------------------------------------------------
# No state text anywhere
# ---------------------------------------------------------------------------


def test_422_text_log_and_traceback_hold_no_state(caplog: pytest.LogCaptureFixture) -> None:
    body = {
        "detail": [
            {
                "type": "missing",
                "loc": ["body", "state"],
                "msg": "Field required",
                "input": {"state": STATE_TEXT},
            }
        ]
    }
    caplog.set_level(logging.DEBUG)
    err = error_from_response(422, {}, _json(body))
    try:
        raise err
    except DecisionError:
        formatted = traceback.format_exc()
    assert STATE_TEXT not in str(err)
    assert STATE_TEXT not in formatted
    assert err.__cause__ is None
    assert err.__context__ is None
    assert all(STATE_TEXT not in r.getMessage() for r in caplog.records)
    assert not caplog.records


def test_malformed_200_carries_no_chained_decode_error() -> None:
    err = error_from_response(200, {}, b"{" + STATE_TEXT.encode())
    try:
        raise err
    except DecisionError:
        formatted = traceback.format_exc()
    assert err.__cause__ is None
    assert err.__context__ is None
    assert STATE_TEXT not in formatted


def test_errors_are_all_defined_in_the_errors_module() -> None:
    names = {
        "DecisionError",
        "DecisionAuthError",
        "DecisionRequestError",
        "DecisionUnavailableError",
        "IncompleteAnswersError",
    }
    for name in names:
        assert getattr(jev_errors, name).__module__ == "osprey.models.jev.errors"
