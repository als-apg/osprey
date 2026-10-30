"""Tests for the opaque-origin header every agent-writable response carries."""

from __future__ import annotations

import pytest

from osprey.interfaces.opaque_origin import (
    INERT_TYPES,
    OPAQUE_CSP,
    is_inert,
    opaque_headers,
)

#: The exact sandbox token set. ``allow-modals`` is in it because printing
#: from a sandboxed frame needs it.
EXPECTED_TOKENS = {
    "sandbox",
    "allow-scripts",
    "allow-modals",
    "allow-popups",
    "allow-downloads",
}


def test_csp_is_the_exact_token_set():
    tokens = OPAQUE_CSP.split(" ")
    assert set(tokens) == EXPECTED_TOKENS
    assert len(tokens) == len(EXPECTED_TOKENS)
    assert tokens[0] == "sandbox"


def test_csp_spelling_is_pinned():
    assert OPAQUE_CSP == "sandbox allow-scripts allow-modals allow-popups allow-downloads"


@pytest.mark.parametrize(
    "token",
    ["allow-same-origin", "allow-forms", "allow-top-navigation"],
)
def test_csp_never_grants(token):
    assert token not in OPAQUE_CSP


def test_csp_is_a_single_sandbox_directive():
    assert ";" not in OPAQUE_CSP
    assert "," not in OPAQUE_CSP


def test_opaque_headers_carry_csp_and_nosniff():
    assert opaque_headers() == {
        "Content-Security-Policy": OPAQUE_CSP,
        "X-Content-Type-Options": "nosniff",
    }


def test_opaque_headers_returns_a_fresh_dict():
    first = opaque_headers()
    first["Content-Security-Policy"] = "tampered"
    assert opaque_headers()["Content-Security-Policy"] == OPAQUE_CSP


def test_inert_list_is_closed():
    assert INERT_TYPES == frozenset(
        {
            "application/pdf",
            "text/plain",
            "application/json",
            "text/csv",
            "image/png",
            "image/jpeg",
            "image/gif",
            "image/webp",
        }
    )


@pytest.mark.parametrize("mime", sorted(INERT_TYPES))
def test_inert_types_are_not_opaque(mime):
    assert is_inert(mime) is True


@pytest.mark.parametrize(
    "mime",
    [
        "text/plain; charset=utf-8",
        "TEXT/PLAIN",
        "  application/json  ",
        "Image/PNG; foo=bar",
    ],
)
def test_inert_normalises_case_and_parameters(mime):
    assert is_inert(mime) is True


@pytest.mark.parametrize(
    "mime",
    [
        "text/html",
        "text/html; charset=utf-8",
        "application/xhtml+xml",
        "image/svg+xml",
        "text/xml",
        "application/xml",
        "application/rss+xml",
        "application/xslt+xml",
        "text/javascript",
        "application/octet-stream",
        "application/x-unknown-thing",
        "image/bmp",
        "image/jpg",
        "",
        "garbage",
        None,
    ],
)
def test_everything_else_is_opaque(mime):
    assert is_inert(mime) is False
