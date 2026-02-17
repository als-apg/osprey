"""Tests for ChannelFinderHierRegistry."""

import pytest

from osprey.services.channel_finder.mcp.hierarchical.registry import (
    get_cf_hier_registry,
    initialize_cf_hier_registry,
)


@pytest.mark.unit
def test_registry_not_initialized():
    with pytest.raises(RuntimeError, match="not initialized"):
        get_cf_hier_registry()


@pytest.mark.unit
def test_registry_database_not_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_hier_registry()
    reg = get_cf_hier_registry()
    with pytest.raises(RuntimeError, match="not configured"):
        _ = reg.database


@pytest.mark.unit
def test_registry_facility_name_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_hier_registry()
    assert get_cf_hier_registry().facility_name == "control system"


@pytest.mark.unit
def test_registry_facility_name_from_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text('facility:\n  name: "NSLS-II"')
    initialize_cf_hier_registry()
    assert get_cf_hier_registry().facility_name == "NSLS-II"
