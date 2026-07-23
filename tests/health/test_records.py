"""Unit tests for :mod:`osprey.health.records`.

These pin the extracted config-loading and record-assembly core directly, at
the function boundary — the surface-agnostic mechanism the ``osprey health`` CLI
(and, later, non-CLI surfaces) share. The CLI-level contract tests
(``tests/cli/test_health_cmd.py``) remain the end-to-end regression net.
"""

from __future__ import annotations

import logging
from pathlib import Path

from osprey.health.config import (
    DEFAULT_ON_DEMAND_CALLABLE_TIMEOUT_S,
    CategoryOverride,
    CategoryRecord,
    Cost,
    HealthSettings,
)
from osprey.health.core import CORE_CATEGORY_NAMES
from osprey.health.models import Status
from osprey.health.records import (
    CONFIG_DEPENDENT,
    ON_DEMAND_CORE,
    build_records,
    core_record,
    load_config,
    skip_record,
)

_VALID_CONFIG = """\
project_name: test_project
models:
  python_code_generator:
    provider: mock
    model_id: mock-model
"""


def _write_config(project: Path, body: str) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    path = project / "config.yml"
    path.write_text(body)
    return path


# --------------------------------------------------------------------------- #
# Module-level policy sets
# --------------------------------------------------------------------------- #


class TestPolicySets:
    def test_config_dependent_names(self):
        assert CONFIG_DEPENDENT == frozenset(
            {
                "openobserve",
                "providers",
                "claude_cli_pinned",
                "model_chat",
                "ariel",
                "channel_finder",
            }
        )

    def test_on_demand_core_names(self):
        assert ON_DEMAND_CORE == frozenset({"claude_cli_pinned", "model_chat"})

    def test_on_demand_core_is_subset_of_config_dependent(self):
        # Both on_demand core categories also read config, so they must degrade.
        assert ON_DEMAND_CORE <= CONFIG_DEPENDENT


# --------------------------------------------------------------------------- #
# load_config
# --------------------------------------------------------------------------- #


class TestLoadConfig:
    def test_missing_file_degrades(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        state, expanded, settings, config_ok = load_config(project / "config.yml", project)
        assert config_ok is False
        assert expanded is None
        assert settings is None
        assert state.exists is False

    def test_valid_config_parses(self, tmp_path):
        project = tmp_path / "proj"
        config_path = _write_config(project, _VALID_CONFIG)
        state, expanded, settings, config_ok = load_config(config_path, project)
        assert config_ok is True
        assert expanded is not None
        assert expanded["project_name"] == "test_project"
        assert isinstance(settings, HealthSettings)
        assert state.exists is True

    def test_empty_file_degrades(self, tmp_path):
        project = tmp_path / "proj"
        config_path = _write_config(project, "")
        state, expanded, settings, config_ok = load_config(config_path, project)
        assert config_ok is False
        assert expanded is None
        assert settings is None
        assert state.exists is True

    def test_malformed_yaml_degrades_without_raising(self, tmp_path):
        project = tmp_path / "proj"
        config_path = _write_config(project, "invalid: yaml: content:\n")
        state, expanded, settings, config_ok = load_config(config_path, project)
        assert config_ok is False
        assert settings is None
        # The load failure is captured on the state, never raised.
        assert state.yaml_error is not None

    def test_invalid_health_section_degrades_with_expanded(self, tmp_path):
        project = tmp_path / "proj"
        config_path = _write_config(
            project,
            "project_name: bad_health\nhealth:\n  suite_timeout_s: not-a-number\n",
        )
        state, expanded, settings, config_ok = load_config(config_path, project)
        assert config_ok is False
        assert settings is None
        # A bad health: section still surfaces the expanded mapping and records
        # the health error on the state (yaml itself parsed fine).
        assert expanded is not None
        assert state.yaml_error is None
        assert state.health_error is not None


# --------------------------------------------------------------------------- #
# core_record
# --------------------------------------------------------------------------- #


def _noop_callable():
    return []


class TestCoreRecord:
    def test_poll_cost_uses_suite_timeout(self):
        rec = core_record("file_system", _noop_callable, Cost.POLL, None, None, 30.0)
        assert isinstance(rec, CategoryRecord)
        assert rec.cost is Cost.POLL
        assert rec.timeout_s == 30.0
        assert rec.func is _noop_callable

    def test_on_demand_cost_uses_callable_default(self):
        rec = core_record("claude_cli_pinned", _noop_callable, Cost.ON_DEMAND, None, None, 30.0)
        assert rec.cost is Cost.ON_DEMAND
        assert rec.timeout_s == DEFAULT_ON_DEMAND_CALLABLE_TIMEOUT_S

    def test_override_cost_and_timeout_win(self):
        settings = HealthSettings(
            suite_timeout_s=30.0,
            interval_s=0.0,
            on_demand_timeout_s=None,
            overrides={"file_system": CategoryOverride(cost=Cost.ON_DEMAND, timeout_s=7.0)},
        )
        rec = core_record("file_system", _noop_callable, Cost.POLL, settings, None, 30.0)
        assert rec.cost is Cost.ON_DEMAND
        assert rec.timeout_s == 7.0

    def test_model_chat_on_demand_sizes_budget_by_item_count(self):
        # Two distinct model pairings → budget = 2 * per-item default.
        expanded = {
            "models": {
                "a": {"provider": "mock", "model_id": "m1"},
                "b": {"provider": "mock", "model_id": "m2"},
            }
        }
        rec = core_record("model_chat", _noop_callable, Cost.ON_DEMAND, None, expanded, 30.0)
        assert rec.cost is Cost.ON_DEMAND
        assert rec.timeout_s == 2 * DEFAULT_ON_DEMAND_CALLABLE_TIMEOUT_S


# --------------------------------------------------------------------------- #
# skip_record
# --------------------------------------------------------------------------- #


class TestSkipRecord:
    def test_skip_record_is_poll_with_single_skip_row(self):
        rec = skip_record("providers", "config unavailable", 30.0)
        assert rec.cost is Cost.POLL
        assert rec.timeout_s == 30.0
        rows = rec.func()
        assert len(rows) == 1
        assert rows[0].status is Status.SKIP
        assert rows[0].message == "config unavailable"
        assert rows[0].category == "providers"


# --------------------------------------------------------------------------- #
# build_records
# --------------------------------------------------------------------------- #


class TestBuildRecords:
    def test_config_failure_degrades_config_dependent_to_skips(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        # A missing-config state: config_ok False.
        state, expanded, settings, config_ok = load_config(project / "config.yml", project)
        assert config_ok is False

        records, extra_rows = build_records(state, expanded, settings, config_ok, project, 30.0)
        by_name = {r.name: r for r in records}

        # configuration is always present.
        assert "configuration" in by_name
        # Every config-dependent core category collapsed to a poll-cost skip.
        for name in CONFIG_DEPENDENT:
            rec = by_name[name]
            assert rec.cost is Cost.POLL
            rows = rec.func()
            assert rows[0].status is Status.SKIP
            assert rows[0].message == "config unavailable"
        # No YAML/plugin categories loaded under a config failure.
        assert extra_rows == []

    def test_all_core_categories_present_on_valid_config(self, tmp_path):
        project = tmp_path / "proj"
        config_path = _write_config(project, _VALID_CONFIG)
        state, expanded, settings, config_ok = load_config(config_path, project)
        assert config_ok is True

        records, extra_rows = build_records(state, expanded, settings, config_ok, project, 30.0)
        names = {r.name for r in records}
        # Every core category is present when the config is healthy.
        assert set(CORE_CATEGORY_NAMES) <= names
        # A valid config with no plugins yields no plugin-load error rows.
        assert extra_rows == []

    def test_on_demand_core_categories_default_to_on_demand_cost(self, tmp_path):
        project = tmp_path / "proj"
        config_path = _write_config(project, _VALID_CONFIG)
        state, expanded, settings, config_ok = load_config(config_path, project)

        records, _ = build_records(state, expanded, settings, config_ok, project, 30.0)
        by_name = {r.name: r for r in records}
        for name in ON_DEMAND_CORE:
            assert by_name[name].cost is Cost.ON_DEMAND


# --------------------------------------------------------------------------- #
# build_records: auto-derived mcp_servers category
# --------------------------------------------------------------------------- #


# A config carrying a build-emitted ``claude_code.servers`` entry: a top-level
# ``url`` plus the ``network`` block (with host/docker URLs) and a permissions
# block that ``osprey build`` writes. This is the cross-surface seam every
# surface (CLI, web, agent) inherits through ``build_records``.
_CONFIG_WITH_MCP_SERVER = """\
project_name: test_project
models:
  python_code_generator:
    provider: mock
    model_id: mock-model
claude_code:
  servers:
    myserver:
      url: "http://example/mcp"
      network:
        port: 9000
        host_url: "http://localhost:9000/mcp"
        docker_url: "http://myserver:9000/mcp"
      permissions:
        allow:
          - "mcp__myserver__tool_a"
        ask:
          - "mcp__myserver__tool_b"
"""


class TestBuildRecordsMcpServers:
    def test_derived_category_from_cross_surface_config(self, tmp_path):
        # A config with a build-emitted server block yields the derived
        # ``mcp_servers`` category from the single seam every surface consumes.
        project = tmp_path / "proj"
        config_path = _write_config(project, _CONFIG_WITH_MCP_SERVER)
        state, expanded, settings, config_ok = load_config(config_path, project)
        assert config_ok is True

        records, _ = build_records(state, expanded, settings, config_ok, project, 30.0)
        by_name = {r.name: r for r in records}
        assert "mcp_servers" in by_name

        rec = by_name["mcp_servers"]
        assert rec.cost is Cost.POLL
        assert rec.func is None
        assert rec.checks is not None
        assert len(rec.checks) == 1

        check = rec.checks[0]
        assert check.name == "myserver"
        assert check.type == "mcp"
        # On a host (not containerized), the derived probe targets host_url.
        assert check.params["url"] == "http://localhost:9000/mcp"
        # Ordered, de-duplicated union of permissions.allow + permissions.ask.
        assert check.params["expect_tools"] == [
            "mcp__myserver__tool_a",
            "mcp__myserver__tool_b",
        ]

    def test_auto_disabled_omits_category(self, tmp_path):
        config = _CONFIG_WITH_MCP_SERVER + "health:\n  auto:\n    mcp:\n      enabled: false\n"
        project = tmp_path / "proj"
        config_path = _write_config(project, config)
        state, expanded, settings, config_ok = load_config(config_path, project)
        assert config_ok is True

        records, _ = build_records(state, expanded, settings, config_ok, project, 30.0)
        assert "mcp_servers" not in {r.name for r in records}

    def test_vanilla_config_yields_empty_category(self, tmp_path):
        # No claude_code servers → the derived category is present but empty
        # (an empty category is valid and must be kept, not dropped).
        project = tmp_path / "proj"
        config_path = _write_config(project, _VALID_CONFIG)
        state, expanded, settings, config_ok = load_config(config_path, project)

        records, _ = build_records(state, expanded, settings, config_ok, project, 30.0)
        by_name = {r.name: r for r in records}
        assert "mcp_servers" in by_name
        assert by_name["mcp_servers"].checks == []

    def test_yaml_collision_skips_derived_and_warns(self, tmp_path, caplog):
        # A YAML-declared mcp_servers category wins; the derived one is skipped
        # and a warning naming the collision source is logged.
        config = """\
project_name: test_project
health:
  categories:
    mcp_servers:
      checks:
        - name: manual_probe
          type: mcp
          url: "http://manual/mcp"
"""
        project = tmp_path / "proj"
        config_path = _write_config(project, config)
        state, expanded, settings, config_ok = load_config(config_path, project)
        assert config_ok is True

        with caplog.at_level(logging.WARNING, logger="osprey.health.records"):
            records, _ = build_records(state, expanded, settings, config_ok, project, 30.0)

        mcp_recs = [r for r in records if r.name == "mcp_servers"]
        assert len(mcp_recs) == 1
        # The kept record is the YAML-declared one.
        assert mcp_recs[0].checks is not None
        assert mcp_recs[0].checks[0].name == "manual_probe"
        # A warning was logged naming the collision source clearly.
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("mcp_servers" in r.getMessage() for r in warnings)
        assert any("YAML config" in r.getMessage() for r in warnings)

    def test_override_applied_to_derived_record(self, tmp_path):
        # A metadata-only override for mcp_servers is applied to the derived
        # record exactly as core overrides apply.
        config = _CONFIG_WITH_MCP_SERVER + (
            "health:\n"
            "  categories:\n"
            "    mcp_servers:\n"
            "      cost: on_demand\n"
            "      timeout_s: 5.0\n"
        )
        project = tmp_path / "proj"
        config_path = _write_config(project, config)
        state, expanded, settings, config_ok = load_config(config_path, project)
        assert config_ok is True

        records, _ = build_records(state, expanded, settings, config_ok, project, 30.0)
        by_name = {r.name: r for r in records}
        assert "mcp_servers" in by_name
        rec = by_name["mcp_servers"]
        assert rec.cost is Cost.ON_DEMAND
        assert rec.timeout_s == 5.0
        # The override does not disturb the derived checks.
        assert rec.checks is not None
        assert len(rec.checks) == 1
