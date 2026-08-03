"""The config wizard must not offer an archiver the environment cannot build.

``mongodb_archiver`` only works when the ``archiver-mongodb`` extra (pymongo) is
installed, and the connector imports pymongo lazily inside ``connect()`` — so an
ungated wizard choice writes an ``archiver.type`` that fails much later, at
runtime. These tests pin the gate and the install hint that replaces it.
"""

from pathlib import Path

import pytest

from osprey.cli import project_actions
from osprey.connectors import types

pytestmark = pytest.mark.unit


def _mongodb_choice(choices):
    return next(c for c in choices if c.value == types.MONGODB_ARCHIVER)


def test_mongodb_offered_when_extra_importable(monkeypatch):
    monkeypatch.setattr(project_actions, "find_spec", lambda name: object())

    choices = project_actions.build_archiver_choices()

    assert _mongodb_choice(choices).disabled is None


def test_mongodb_disabled_when_extra_missing(monkeypatch):
    monkeypatch.setattr(project_actions, "find_spec", lambda name: None)

    choices = project_actions.build_archiver_choices()

    assert _mongodb_choice(choices).disabled is not None


def test_gate_probes_pymongo(monkeypatch):
    """The gate must key off pymongo, not some other import."""
    probed = []

    def fake_find_spec(name):
        probed.append(name)
        return None

    monkeypatch.setattr(project_actions, "find_spec", fake_find_spec)

    assert project_actions.mongodb_archiver_available() is False
    assert probed == ["pymongo"]


def test_other_archivers_never_gated(monkeypatch):
    """Only MongoDB is gated — mock and EPICS stay selectable either way."""
    for spec in (object(), None):
        monkeypatch.setattr(project_actions, "find_spec", lambda name, s=spec: s)
        choices = project_actions.build_archiver_choices()
        ungated = {c.value for c in choices if c.disabled is None}
        assert {types.MOCK_ARCHIVER, types.EPICS_ARCHIVER} <= ungated


def test_disabled_message_is_actionable(monkeypatch):
    """The reason names the extra and the command that installs it."""
    monkeypatch.setattr(project_actions, "find_spec", lambda name: None)

    reason = _mongodb_choice(project_actions.build_archiver_choices()).disabled

    assert "archiver-mongodb" in reason
    assert "pip install 'osprey-framework[archiver-mongodb]'" in reason


CONTROL_ASSISTANT_CONFIG = (
    Path(project_actions.__file__).parents[1]
    / "templates"
    / "apps"
    / "control_assistant"
    / "config.yml.j2"
)


def test_template_documents_mongodb_option():
    """The options comment must list mongodb_archiver and name the extra."""
    template = CONTROL_ASSISTANT_CONFIG.read_text(encoding="utf-8")

    assert "#   - mongodb_archiver:" in template
    assert "archiver-mongodb" in template


def test_template_shows_required_mongodb_keys():
    """A commented block covers every key the connector refuses to default."""
    template = CONTROL_ASSISTANT_CONFIG.read_text(encoding="utf-8")

    assert "# mongodb_archiver:" in template
    for key in ("host", "name", "collection", "auth", "username", "password_env"):
        assert f"#   {key}:" in template, f"required key {key!r} missing from example block"
