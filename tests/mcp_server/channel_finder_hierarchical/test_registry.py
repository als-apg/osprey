"""Tests for ChannelFinderHierRegistry."""

import textwrap

import pytest

from osprey.mcp_server.channel_finder_hierarchical.registry import (
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


# ------------------------------------------------------------------
# Feedback store initialization
# ------------------------------------------------------------------


@pytest.mark.unit
def test_registry_feedback_store_initialized(tmp_path, monkeypatch):
    """Config with feedback enabled creates a FeedbackStore."""
    monkeypatch.chdir(tmp_path)
    store_path = tmp_path / "feedback.json"
    config = textwrap.dedent(f"""\
        channel_finder:
          pipelines:
            hierarchical:
              feedback:
                enabled: true
                store_path: "{store_path}"
    """)
    (tmp_path / "config.yml").write_text(config)
    initialize_cf_hier_registry()
    reg = get_cf_hier_registry()
    assert reg.feedback_store is not None


@pytest.mark.unit
def test_registry_feedback_store_disabled(tmp_path, monkeypatch):
    """Config with feedback disabled results in None feedback_store."""
    monkeypatch.chdir(tmp_path)
    config = textwrap.dedent("""\
        channel_finder:
          pipelines:
            hierarchical:
              feedback:
                enabled: false
                store_path: "feedback.json"
    """)
    (tmp_path / "config.yml").write_text(config)
    initialize_cf_hier_registry()
    reg = get_cf_hier_registry()
    assert reg.feedback_store is None


@pytest.mark.unit
def test_registry_feedback_store_no_config(tmp_path, monkeypatch):
    """No feedback section in config results in None feedback_store."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_hier_registry()
    reg = get_cf_hier_registry()
    assert reg.feedback_store is None
