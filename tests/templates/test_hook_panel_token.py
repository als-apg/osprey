"""Shipped hooks authenticate their terminal-API calls with the panel token.

Three template hooks talk to an app the web terminal serves — the SessionStart
panels-context hook and the UserPromptSubmit workspace-delta hook both
``GET /api/panels``, and the approval hook ``POST``s ``/api/focus`` at the
artifact gallery. Once those endpoints require a bearer token, an unauthenticated
hook goes quietly blind: it keeps exiting 0 and simply stops reporting anything.

So these tests pin the contract from the hook side:

* the token is read from ``OSPREY_PANEL_TOKEN`` in the inherited environment at
  call time — no file, no import, nothing OSPREY-specific;
* a variable that is unset, empty *or whitespace-only* sends **no**
  ``Authorization`` header, rather than a blank bearer, which is what these
  calls have always done on loopback — the same rule
  ``mcp_server.http._panel_auth_headers`` applies on the server side, and a real
  value arrives stripped for the same reason;
* every one of the three stays fail-open. A refused call — 401 included — must
  leave the session, the prompt, and the approval prompt unblocked.

The hooks ship into built projects and run under a bare ``python3``, so the
tests exercise them with a stubbed ``urlopen`` and never touch the network.
"""

from __future__ import annotations

import importlib
import io
import json
import sys
import urllib.error
import urllib.request

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_panels_context as panels
import osprey.templates.claude_code.claude.hooks.osprey_workspace_delta as delta

TOKEN_ENV = "OSPREY_PANEL_TOKEN"


def _import_approval():
    """Import the approval hook in-process, undoing its ``sys.path`` shim.

    ``osprey_approval.py`` prepends its own directory to ``sys.path`` at import
    so it can pick up its ``osprey_hook_log`` sibling. Left in place, that entry
    would let any later ``import osprey_hook_log`` anywhere in the worker
    resolve to the template copy, so the pre-import path is put back on the way
    out. The import happens here rather than at module scope both for that
    reason and because ``tests/infrastructure/test_import_time_audit.py``
    forbids import-time mutation from a test module.
    """
    before = list(sys.path)
    try:
        return importlib.import_module("osprey.templates.claude_code.claude.hooks.osprey_approval")
    finally:
        sys.path[:] = before


class _FakeResponse:
    """A minimal ``urlopen`` answer: readable once, and a context manager."""

    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _record_urlopen(monkeypatch, request_module, *, raises=None):
    """Stub ``urlopen`` on *request_module* and hand back the targets it saw.

    The two panel hooks import ``urllib.request`` at module scope, so their own
    binding is the one to patch. ``osprey_approval`` imports it inside the
    function instead — a fresh lookup on every call — so its stub goes onto the
    stdlib module itself, which ``monkeypatch`` restores at teardown.
    """
    seen = []

    def _fake(target, timeout=None):
        seen.append(target)
        if raises is not None:
            raise raises
        return _FakeResponse({"enabled": [], "custom": []})

    monkeypatch.setattr(request_module, "urlopen", _fake)
    return seen


def _unauthorized():
    """The 401 a token-requiring terminal API answers an anonymous call with."""
    return urllib.error.HTTPError("http://127.0.0.1:8087/api/panels", 401, "Unauthorized", {}, None)


def _authorization_of(target):
    """The ``Authorization`` header on a urlopen target, or None for a bare URL."""
    if isinstance(target, str):
        return None
    return target.get_header("Authorization")


def _url_of(target):
    """The URL a urlopen target addresses, whether it is a string or a Request."""
    return target if isinstance(target, str) else target.full_url


@pytest.fixture(autouse=True)
def _clean_token_env(monkeypatch):
    """No ambient panel token: every test states the environment it wants."""
    monkeypatch.delenv(TOKEN_ENV, raising=False)


# ---------------------------------------------------------------------------
# GET /api/panels — panels-context (SessionStart) and workspace-delta
# ---------------------------------------------------------------------------


def _run_panels(monkeypatch):
    """Drive the panels-context hook's ``main`` with an empty stdin payload."""
    monkeypatch.setattr(panels.sys, "stdin", io.StringIO(json.dumps({})))
    return panels.main()


def _run_delta(monkeypatch, tmp_path):
    """Drive the workspace-delta hook's ``main`` with a known session id."""
    monkeypatch.setattr(delta.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(delta.sys, "stdin", io.StringIO(json.dumps({"session_id": "abc123"})))
    return delta.main()


#: Carrier values that must all read as "no token": unset is covered separately,
#: the empty string is what an uninterpolated compose variable interpolates to,
#: and a whitespace-only value is what a hand-edited ``.env`` or a copy-paste
#: leaves behind. ``mcp_server.http._panel_auth_headers`` — the server-side
#: reader of the same carrier — has always ``.strip()``ed before deciding, so a
#: hook that only tested truthiness would send ``Authorization: Bearer    `` for
#: a value the rest of the system calls absent.
BLANK_TOKENS = ("", "   ", "\t\n")


def test_panels_context_sends_bearer_when_token_set(monkeypatch):
    """A set token becomes ``Authorization: Bearer <token>`` on the GET."""
    monkeypatch.setenv(TOKEN_ENV, "s3cret-panel-token")
    seen = _record_urlopen(monkeypatch, panels.urllib.request)

    assert _run_panels(monkeypatch) == 0
    assert len(seen) == 1
    assert _authorization_of(seen[0]) == "Bearer s3cret-panel-token"
    assert seen[0].get_method() == "GET"
    assert _url_of(seen[0]).endswith("/api/panels")


def test_panels_context_sends_no_header_when_token_unset(monkeypatch):
    """No token in the environment -> no ``Authorization`` header at all."""
    seen = _record_urlopen(monkeypatch, panels.urllib.request)

    assert _run_panels(monkeypatch) == 0
    assert len(seen) == 1
    assert _authorization_of(seen[0]) is None
    assert _url_of(seen[0]).endswith("/api/panels")


@pytest.mark.parametrize("value", BLANK_TOKENS)
def test_panels_context_sends_no_header_for_blank_token(monkeypatch, value):
    """A blank token is absent, not a credential: still no header."""
    monkeypatch.setenv(TOKEN_ENV, value)
    seen = _record_urlopen(monkeypatch, panels.urllib.request)

    assert _run_panels(monkeypatch) == 0
    assert _authorization_of(seen[0]) is None


def test_panels_context_strips_a_padded_token(monkeypatch):
    """Surrounding whitespace is trimmed, not sent — the server trims too."""
    monkeypatch.setenv(TOKEN_ENV, "  s3cret-panel-token\n")
    seen = _record_urlopen(monkeypatch, panels.urllib.request)

    assert _run_panels(monkeypatch) == 0
    assert _authorization_of(seen[0]) == "Bearer s3cret-panel-token"


def test_panels_context_reads_token_at_call_time(monkeypatch):
    """The token is read per call, so a value exported after import is used."""
    seen = _record_urlopen(monkeypatch, panels.urllib.request)
    assert _run_panels(monkeypatch) == 0

    monkeypatch.setenv(TOKEN_ENV, "arrived-later")
    assert _run_panels(monkeypatch) == 0

    assert _authorization_of(seen[0]) is None
    assert _authorization_of(seen[1]) == "Bearer arrived-later"


def test_panels_context_fails_open_on_401(monkeypatch, capsys):
    """A refused fetch drops the live parts and never blocks the session."""
    monkeypatch.setenv(TOKEN_ENV, "wrong-token")
    _record_urlopen(monkeypatch, panels.urllib.request, raises=_unauthorized())

    assert _run_panels(monkeypatch) == 0
    assert capsys.readouterr().out == ""


def test_workspace_delta_sends_bearer_when_token_set(monkeypatch, tmp_path):
    """The delta hook authenticates its ``GET /api/panels`` the same way."""
    monkeypatch.setenv(TOKEN_ENV, "s3cret-panel-token")
    seen = _record_urlopen(monkeypatch, delta.urllib.request)

    assert _run_delta(monkeypatch, tmp_path) == 0
    assert len(seen) == 1
    assert _authorization_of(seen[0]) == "Bearer s3cret-panel-token"
    assert seen[0].get_method() == "GET"
    assert _url_of(seen[0]).endswith("/api/panels")


def test_workspace_delta_sends_no_header_when_token_unset(monkeypatch, tmp_path):
    """No token -> the delta hook's GET carries no ``Authorization``."""
    seen = _record_urlopen(monkeypatch, delta.urllib.request)

    assert _run_delta(monkeypatch, tmp_path) == 0
    assert _authorization_of(seen[0]) is None


@pytest.mark.parametrize("value", BLANK_TOKENS)
def test_workspace_delta_sends_no_header_for_blank_token(monkeypatch, tmp_path, value):
    """A blank token sends no header rather than a blank bearer."""
    monkeypatch.setenv(TOKEN_ENV, value)
    seen = _record_urlopen(monkeypatch, delta.urllib.request)

    assert _run_delta(monkeypatch, tmp_path) == 0
    assert _authorization_of(seen[0]) is None


def test_workspace_delta_strips_a_padded_token(monkeypatch, tmp_path):
    """Surrounding whitespace is trimmed, not sent."""
    monkeypatch.setenv(TOKEN_ENV, " s3cret-panel-token ")
    seen = _record_urlopen(monkeypatch, delta.urllib.request)

    assert _run_delta(monkeypatch, tmp_path) == 0
    assert _authorization_of(seen[0]) == "Bearer s3cret-panel-token"


def test_workspace_delta_fails_open_on_401(monkeypatch, tmp_path, capsys):
    """A 401 reads as an unknown workspace: silent, exit 0, prompt unblocked."""
    monkeypatch.setenv(TOKEN_ENV, "wrong-token")
    _record_urlopen(monkeypatch, delta.urllib.request, raises=_unauthorized())

    assert _run_delta(monkeypatch, tmp_path) == 0
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# POST /api/focus — the approval hook's gallery focus call
# ---------------------------------------------------------------------------


def test_focus_artifact_sends_bearer_when_token_set(monkeypatch):
    """The focus POST carries the bearer alongside its JSON content type."""
    approval = _import_approval()
    monkeypatch.setenv(TOKEN_ENV, "s3cret-panel-token")
    seen = _record_urlopen(monkeypatch, urllib.request)

    approval._focus_artifact("http://127.0.0.1:8087", "artifact-7")

    assert len(seen) == 1
    request = seen[0]
    assert _authorization_of(request) == "Bearer s3cret-panel-token"
    assert request.get_header("Content-type") == "application/json"
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {"artifact_id": "artifact-7"}


def test_focus_artifact_sends_no_header_when_token_unset(monkeypatch):
    """No token -> no ``Authorization``; the content type is untouched."""
    approval = _import_approval()
    seen = _record_urlopen(monkeypatch, urllib.request)

    approval._focus_artifact("http://127.0.0.1:8087", "artifact-7")

    assert _authorization_of(seen[0]) is None
    assert seen[0].get_header("Content-type") == "application/json"


@pytest.mark.parametrize("value", BLANK_TOKENS)
def test_focus_artifact_sends_no_header_for_blank_token(monkeypatch, value):
    """A blank token is not a credential: the header stays off."""
    approval = _import_approval()
    monkeypatch.setenv(TOKEN_ENV, value)
    seen = _record_urlopen(monkeypatch, urllib.request)

    approval._focus_artifact("http://127.0.0.1:8087", "artifact-7")

    assert _authorization_of(seen[0]) is None


def test_focus_artifact_strips_a_padded_token(monkeypatch):
    """Surrounding whitespace is trimmed, not sent."""
    approval = _import_approval()
    monkeypatch.setenv(TOKEN_ENV, "\t s3cret-panel-token ")
    seen = _record_urlopen(monkeypatch, urllib.request)

    approval._focus_artifact("http://127.0.0.1:8087", "artifact-7")

    assert _authorization_of(seen[0]) == "Bearer s3cret-panel-token"


def test_focus_artifact_swallows_401(monkeypatch):
    """A gallery that refuses the focus call never breaks the approval prompt."""
    approval = _import_approval()
    monkeypatch.setenv(TOKEN_ENV, "wrong-token")
    _record_urlopen(monkeypatch, urllib.request, raises=_unauthorized())

    assert approval._focus_artifact("http://127.0.0.1:8087", "artifact-7") is None
