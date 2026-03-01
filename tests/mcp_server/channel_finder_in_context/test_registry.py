"""Tests for ChannelFinderICRegistry."""

import json

import pytest

from osprey.mcp_server.channel_finder_in_context.registry import (
    get_cf_ic_registry,
    initialize_cf_ic_registry,
)


@pytest.mark.unit
def test_registry_not_initialized():
    with pytest.raises(RuntimeError, match="not initialized"):
        get_cf_ic_registry()


@pytest.mark.unit
def test_registry_database_not_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_ic_registry()
    reg = get_cf_ic_registry()
    with pytest.raises(RuntimeError, match="not available"):
        _ = reg.database


@pytest.mark.unit
def test_registry_loads_flat_database(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_data = [
        {"channel": "CH1", "address": "PV:CH1", "description": "Channel 1"},
        {"channel": "CH2", "address": "PV:CH2", "description": "Channel 2"},
    ]
    db_file = tmp_path / "test_db.json"
    db_file.write_text(json.dumps(db_data))
    config = (
        f"channel_finder:\n"
        f"  pipelines:\n"
        f"    in_context:\n"
        f"      database:\n"
        f'        path: "{db_file}"\n'
        f'        type: "flat"\n'
    )
    (tmp_path / "config.yml").write_text(config)
    initialize_cf_ic_registry()
    reg = get_cf_ic_registry()
    assert reg.database is not None
    assert len(reg.database.get_all_channels()) == 2


@pytest.mark.unit
def test_registry_facility_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text('facility:\n  name: "ALS"')
    initialize_cf_ic_registry()
    assert get_cf_ic_registry().facility_name == "ALS"
