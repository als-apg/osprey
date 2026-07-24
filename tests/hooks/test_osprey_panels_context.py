"""Tests for the SessionStart panels-context hook.

``osprey_panels_context.py`` fetches the web-terminal panel inventory at session
start and injects it as ``additionalContext``. It is a stdlib-only, fail-open
hook: any unreachable web terminal or parse error must leave the session
unblocked (silent, exit 0). These tests cover the inventory-string builder as a
pure function and the ``main`` envelope/fail-open behavior with a stubbed
``urlopen`` (no real HTTP).
"""

from __future__ import annotations

import json

import osprey.templates.claude_code.claude.hooks.osprey_panels_context as panels


class _FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data


def _stub_urlopen(monkeypatch, *, payload=None, raises=None, bad_body=False):
    """Point the hook's urlopen at a fake response, an error, or garbage body."""

    def _fake(url, timeout=None):
        if raises is not None:
            raise raises
        if bad_body:

            class _Bad:
                def read(self):
                    return b"not json{{{"

            return _Bad()
        return _FakeResponse(payload or {})

    monkeypatch.setattr(panels.urllib.request, "urlopen", _fake)


# ---------------------------------------------------------------------------
# _build_inventory (pure)
# ---------------------------------------------------------------------------


def test_inventory_lists_enabled_panels_with_visibility():
    data = {
        "enabled": ["archiver", "scan"],
        "labels": {"archiver": "Archiver", "scan": "Scan"},
        "visible": ["archiver"],
        "active": "archiver",
    }
    result = panels._build_inventory(data)
    assert "Archiver (id=archiver, shown)" in result
    assert "Scan (id=scan, hidden)" in result
    assert "Active tab: archiver." in result


def test_inventory_label_falls_back_to_uppercase_id():
    data = {"enabled": ["tuning"], "labels": {}, "visible": []}
    result = panels._build_inventory(data)
    assert "TUNING (id=tuning, hidden)" in result


def test_inventory_includes_custom_panels():
    data = {
        "enabled": [],
        "labels": {},
        "visible": ["mypanel"],
        "custom": [{"id": "mypanel", "label": "My Panel"}],
    }
    result = panels._build_inventory(data)
    assert "My Panel (id=mypanel, shown)" in result


def test_inventory_skips_custom_panel_without_id():
    data = {"enabled": [], "custom": [{"label": "No Id"}, {"id": "ok", "label": "OK"}]}
    result = panels._build_inventory(data)
    assert "No Id" not in result
    assert "OK (id=ok, hidden)" in result


def test_inventory_no_active_tab_phrase():
    data = {"enabled": ["scan"], "labels": {"scan": "Scan"}, "visible": ["scan"]}
    result = panels._build_inventory(data)
    assert "No active tab." in result


def test_inventory_empty_returns_none():
    assert panels._build_inventory({"enabled": [], "custom": []}) is None
    assert panels._build_inventory({}) is None


def test_inventory_mentions_control_tools():
    data = {"enabled": ["scan"], "labels": {"scan": "Scan"}, "visible": ["scan"]}
    result = panels._build_inventory(data)
    assert "show_panel" in result and "hide_panel" in result


# ---------------------------------------------------------------------------
# main() — envelope + fail-open
# ---------------------------------------------------------------------------


def test_main_emits_additional_context_envelope(monkeypatch, capsys):
    _stub_urlopen(
        monkeypatch,
        payload={
            "enabled": ["scan"],
            "labels": {"scan": "Scan"},
            "visible": ["scan"],
            "active": "scan",
        },
    )
    rc = panels.main()
    assert rc == 0

    out = capsys.readouterr().out
    envelope = json.loads(out)
    hook_out = envelope["hookSpecificOutput"]
    assert hook_out["hookEventName"] == "SessionStart"
    assert "Scan (id=scan, shown)" in hook_out["additionalContext"]


def test_main_empty_inventory_writes_nothing(monkeypatch, capsys):
    _stub_urlopen(monkeypatch, payload={"enabled": [], "custom": []})
    rc = panels.main()
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_fails_open_when_web_terminal_down(monkeypatch, capsys):
    """urlopen raising (terminal unreachable) -> silent, exit 0."""
    _stub_urlopen(monkeypatch, raises=ConnectionRefusedError("down"))
    rc = panels.main()
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_fails_open_on_unparseable_body(monkeypatch, capsys):
    """A 200 response with a non-JSON body must not crash the session start."""
    _stub_urlopen(monkeypatch, bad_body=True)
    rc = panels.main()
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_uses_configured_web_port(monkeypatch):
    """OSPREY_WEB_PORT selects the port the inventory is fetched from."""
    monkeypatch.setenv("OSPREY_WEB_PORT", "9123")
    seen = {}

    def _fake(url, timeout=None):
        seen["url"] = url
        return _FakeResponse({"enabled": [], "custom": []})

    monkeypatch.setattr(panels.urllib.request, "urlopen", _fake)
    panels.main()
    assert "127.0.0.1:9123/api/panels" in seen["url"]
