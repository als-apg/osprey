"""Tests for the opaque-origin header every agent-writable response carries."""

from __future__ import annotations

import re
from pathlib import Path

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


#: The browser-side mirrors of the sandbox. Each frames an opaque document
#: with an iframe ``sandbox`` attribute, which takes the CSP's tokens without
#: the ``sandbox`` keyword; a token the server adds and a mirror lacks is a
#: capability the document has when opened directly but not in its frame.
_STATIC = Path(__file__).resolve().parents[2] / "src" / "osprey" / "interfaces"
JS_MIRRORS = {
    "web-terminal panel frame": _STATIC / "web_terminal/static/js/panel-iframe-factory.js",
    "artifacts gallery": _STATIC / "artifacts/static/js/types.js",
}


def _js_string_const(path: Path, name: str) -> str:
    match = re.search(rf"^export const {name} = '([^']*)';$", path.read_text(), re.M)
    assert match is not None, f"no `export const {name} = '...';` in {path}"
    return match.group(1)


@pytest.mark.parametrize("path", JS_MIRRORS.values(), ids=JS_MIRRORS.keys())
def test_js_sandbox_is_the_csp_minus_the_sandbox_keyword(path):
    tokens = _js_string_const(path, "OPAQUE_SANDBOX").split(" ")

    assert len(tokens) == len(set(tokens)), tokens
    assert set(tokens) == set(OPAQUE_CSP.split(" ")) - {"sandbox"}


def test_js_inert_list_is_the_server_list():
    text = JS_MIRRORS["artifacts gallery"].read_text()
    body = re.search(r"^const INERT_TYPES = new Set\(\[(.*?)\]\);$", text, re.M | re.S)
    assert body is not None, "no `const INERT_TYPES = new Set([...]);` in types.js"

    assert set(re.findall(r"'([^']+)'", body.group(1))) == INERT_TYPES


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
