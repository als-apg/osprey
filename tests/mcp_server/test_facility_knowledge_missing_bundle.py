"""Starting the facility-knowledge server with no bundle configured.

``facility_knowledge.bundle_path`` is optional: a deployment can run the server
before it has a bundle to point it at, and the tools are supposed to refuse
with ``server_not_initialised`` until it does. The lookup used to subscript its
way to that key, so how the absence was *spelled* decided what happened — a
missing block raised ``KeyError`` and was caught, while a block present but
empty raised ``TypeError`` and took the server down at startup. Same missing
configuration, two different outcomes, one of them a crash.

These tests pin the six spellings of "no bundle path" to one behaviour:
:func:`~osprey.mcp_server.facility_knowledge.server._resolve_bundle_path`
answers ``None`` and ``create_server`` starts with a warning that names the
config file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from osprey.mcp_server.facility_knowledge import server as fk_server

#: Every way a config can fail to name a bundle, with what it looks like in
#: YAML. Each was a distinct failure mode before the lookup used ``.get``.
UNCONFIGURED: dict[str, Any] = {
    "no facility_knowledge block": {},
    "block present but empty": {"facility_knowledge": None},
    "block present with no bundle_path": {"facility_knowledge": {}},
    "bundle_path present but empty": {"facility_knowledge": {"bundle_path": "   "}},
    "bundle_path present but null": {"facility_knowledge": {"bundle_path": None}},
    "facility_knowledge is not a mapping": {"facility_knowledge": "okf"},
}


@pytest.mark.parametrize("spelling", sorted(UNCONFIGURED))
def test_unconfigured_bundle_path_resolves_to_none(spelling: str, tmp_path: Path) -> None:
    """No spelling of "no bundle path" raises out of the lookup."""
    assert fk_server._resolve_bundle_path(UNCONFIGURED[spelling], tmp_path) is None


def test_a_configured_relative_path_still_resolves(tmp_path: Path) -> None:
    """The ``.get`` rewrite did not change what a configured value resolves to."""
    config = {"facility_knowledge": {"bundle_path": "data/okf"}}
    resolved = fk_server._resolve_bundle_path(config, tmp_path)
    assert resolved is not None
    assert resolved.is_absolute()
    assert resolved.name == "okf"


@pytest.mark.parametrize("spelling", sorted(UNCONFIGURED))
def test_create_server_starts_and_says_why_there_is_no_bundle(
    spelling: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server starts, leaves the bundle unset, and names the config file.

    Startup is what regressed: an empty ``facility_knowledge:`` block used to
    raise out of ``create_server`` before any tool was registered, so the
    deployment got a dead server instead of one whose tools explain themselves.
    """
    config_path = tmp_path / "config.yml"
    config_path.write_text("# rendered config\n", encoding="utf-8")
    monkeypatch.setattr(fk_server, "_bundle", None, raising=False)
    # ``create_server`` imports both names from ``osprey.utils.workspace``,
    # which is a ``sys.modules`` alias of the connectors module — so the
    # connectors module is the one place to patch them.
    monkeypatch.setattr("osprey_connectors.workspace.resolve_config_path", lambda: config_path)
    monkeypatch.setattr(
        "osprey_connectors.workspace.load_osprey_config", lambda: UNCONFIGURED[spelling]
    )

    with caplog.at_level(logging.WARNING, logger=fk_server.logger.name):
        assert fk_server.create_server() is fk_server.mcp

    assert fk_server._bundle is None
    warnings = [record.getMessage() for record in caplog.records]
    assert any("facility_knowledge.bundle_path" in message for message in warnings)
    assert any(str(config_path) in message for message in warnings)
