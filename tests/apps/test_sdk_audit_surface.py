"""``osprey.apps.sdk.audit_surface``: the word an application's audit records file under.

It is the application's name, the word its tool server's records already file
under (the server is launched as ``mcp__<app>__``); there is no second spelling
for a backend's records.
"""

from __future__ import annotations

import pytest

from osprey.apps.errors import AppManifestError
from osprey.apps.sdk import audit_surface
from osprey.apps.sdk._approval import APP_NAME_ENV_VAR


def test_the_surface_is_the_application_name(monkeypatch):
    monkeypatch.setenv(APP_NAME_ENV_VAR, "knob-example")
    assert audit_surface() == "knob-example"


def test_surrounding_whitespace_is_not_part_of_the_name(monkeypatch):
    monkeypatch.setenv(APP_NAME_ENV_VAR, " knob-example\n")
    assert audit_surface() == "knob-example"


@pytest.mark.parametrize("value", [None, "", "  "])
def test_an_unnamed_application_has_no_surface(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(APP_NAME_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(APP_NAME_ENV_VAR, value)
    with pytest.raises(AppManifestError) as info:
        audit_surface()
    assert info.value.rule == "unnamed_app"
