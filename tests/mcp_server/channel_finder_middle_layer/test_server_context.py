"""Tests for Middle Layer channel finder MCP registry."""

import json

import pytest

from osprey.mcp_server.channel_finder_middle_layer.server_context import (
    get_cf_ml_context,
    initialize_cf_ml_context,
)


@pytest.mark.unit
def test_registry_database_not_configured(tmp_path, monkeypatch):
    """Registry raises when no database path configured."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_ml_context()
    reg = get_cf_ml_context()
    with pytest.raises(RuntimeError, match="not configured"):
        _ = reg.database


@pytest.mark.unit
def test_registry_facility_name_default(tmp_path, monkeypatch):
    """Facility name defaults to 'control system' when not in config."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_ml_context()
    assert get_cf_ml_context().facility_name == "control system"


@pytest.mark.unit
def test_registry_facility_name_from_config(tmp_path, monkeypatch):
    """Facility name loaded from config."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text('facility:\n  name: "ALS"')
    initialize_cf_ml_context()
    assert get_cf_ml_context().facility_name == "ALS"


@pytest.mark.unit
def test_registry_loads_database(tmp_path, monkeypatch):
    """Registry initializes database when path is valid."""
    monkeypatch.chdir(tmp_path)
    # Create a minimal middle layer DB JSON
    db_data = {"SR": {"BPM": {"Monitor": {"ChannelNames": ["SR:BPM1"]}}}}
    db_file = tmp_path / "test_db.json"
    db_file.write_text(json.dumps(db_data))
    config = (
        "channel_finder:\n"
        "  pipelines:\n"
        "    middle_layer:\n"
        "      database:\n"
        f'        path: "{db_file}"'
    )
    (tmp_path / "config.yml").write_text(config)
    initialize_cf_ml_context()
    reg = get_cf_ml_context()
    assert reg.database is not None
