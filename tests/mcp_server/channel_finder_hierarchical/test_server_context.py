"""Tests for ChannelFinderHierContext."""

import json
import textwrap

import pytest

from osprey.mcp_server.channel_finder_hierarchical.server_context import (
    get_cf_hier_context,
    initialize_cf_hier_context,
)
from osprey.utils.facility import facility_identity


def test_registry_not_initialized():
    with pytest.raises(RuntimeError, match="not initialized"):
        get_cf_hier_context()


def test_registry_database_not_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_hier_context()
    reg = get_cf_hier_context()
    with pytest.raises(RuntimeError, match="not configured"):
        _ = reg.database


def test_registry_facility_name_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_hier_context()
    assert get_cf_hier_context().facility_name == "control system"


def test_registry_facility_name_is_the_project_name_without_a_facility_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("project_name: 1st-lab\n")
    initialize_cf_hier_context()
    assert get_cf_hier_context().facility_name == "1st-lab"
    assert facility_identity(tmp_path, "1st-lab")["code"] == "x1st_lab"


def test_registry_facility_name_from_the_facility_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("project_name: 1st-lab\n")
    (tmp_path / "facility.json").write_text(
        json.dumps({"identity": {"code": "demo", "name": "Demo Lab"}})
    )
    initialize_cf_hier_context()
    assert get_cf_hier_context().facility_name == "Demo Lab"


def test_registry_loads_the_index_the_build_writes(tmp_path, monkeypatch):
    """The configured database is the build's index; each channel loads as its address."""
    from osprey.facility.views import ViewInputs
    from osprey.facility.views.channel_finder import write_hierarchical

    monkeypatch.chdir(tmp_path)
    addresses = ["SR01C___QF1____AM00", "SR:Q1:CURRENT:SP"]
    (index,) = write_hierarchical(
        tmp_path / "data" / "channel_finder",
        ViewInputs(
            doc={
                "places": [{"id": "SR", "level": "machine"}],
                "devices": [{"id": "SR/Q1", "class": "Quadrupole", "place": "SR"}],
                "channels": [
                    {"id": addresses[0], "on": {"place": "SR"}},
                    {"id": addresses[1], "on": {"device": "SR/Q1"}, "signal": "current_setpoint"},
                ],
            },
            rendered_config={},
            facility_dir=tmp_path,
            served=[],
        ),
    )
    config = textwrap.dedent(f"""\
        channel_finder:
          pipelines:
            hierarchical:
              database:
                path: "{index}"
    """)
    (tmp_path / "config.yml").write_text(config)

    initialize_cf_hier_context()
    database = get_cf_hier_context().database

    assert json.loads(index.read_text())["schema"] == "osprey.facility.channel_finder/1"
    assert sorted(database.channel_map) == sorted(addresses)
    assert all(name == entry["channel"] for name, entry in database.channel_map.items())


# ------------------------------------------------------------------
# Feedback store initialization
# ------------------------------------------------------------------


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
    initialize_cf_hier_context()
    reg = get_cf_hier_context()
    assert reg.feedback_store is not None


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
    initialize_cf_hier_context()
    reg = get_cf_hier_context()
    assert reg.feedback_store is None


def test_registry_feedback_store_no_config(tmp_path, monkeypatch):
    """No feedback section in config results in None feedback_store."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_hier_context()
    reg = get_cf_hier_context()
    assert reg.feedback_store is None
