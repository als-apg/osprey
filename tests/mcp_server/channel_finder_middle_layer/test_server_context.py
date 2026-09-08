"""Tests for Middle Layer channel finder MCP registry."""

import json

import pytest
import yaml

from osprey.mcp_server.channel_finder_middle_layer.server_context import (
    DEFAULT_QUERY_MAX_ROWS,
    QUERY_MAX_ROWS_CONFIG_KEY,
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


@pytest.mark.unit
def test_query_max_rows_defaults_when_unset(tmp_path, monkeypatch):
    """A config naming no cap gets the shipped one."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_ml_context()

    assert get_cf_ml_context().query_max_rows == DEFAULT_QUERY_MAX_ROWS


@pytest.mark.unit
def test_query_max_rows_is_read_from_the_top_level_block(tmp_path, monkeypatch):
    """The key sits on `channel_finder`, outside the derived `pipelines` prefix."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("channel_finder:\n  query_max_rows: 50")
    initialize_cf_ml_context()

    assert get_cf_ml_context().query_max_rows == 50


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["50", 0, -1, True, 1.5])
def test_an_unusable_cap_warns_and_keeps_the_default(tmp_path, monkeypatch, caplog, bad):
    """A typo must not leave the agent with no channel tools, so it warns."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        yaml.safe_dump({"channel_finder": {"query_max_rows": bad}})
    )

    with caplog.at_level("WARNING"):
        initialize_cf_ml_context()

    assert get_cf_ml_context().query_max_rows == DEFAULT_QUERY_MAX_ROWS
    assert any(QUERY_MAX_ROWS_CONFIG_KEY in record.getMessage() for record in caplog.records)
