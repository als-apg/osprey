"""The ``agent_record:`` profile block: parsing, refusals, wiring and render.

The block is a root-profile key with defaults for every field, so the tests
pin both halves: a profile that spells it gets what it spelled, and a profile
that does not still gets a whole record configuration — the embedded backend.
Each Requirement-10 refusal is checked for naming its fix, because a refusal
that says only "invalid" sends the operator to the docs for the answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.agent_record.config import (
    BACKEND_CLICKHOUSE,
    BACKEND_EMBEDDED,
    DEFAULT_CLICKHOUSE_IMAGE,
    DEFAULT_RETENTION_DAYS,
    HISTORY_SERVICE,
    STORE_HTTP_CONTAINER_PORT,
    STORE_MEMORY_RATIO,
    AgentRecordConfig,
    agent_record_config_overrides,
    agent_record_errors,
    history_app_port,
    history_origin_port,
    parse_agent_record_block,
    parse_memory,
    reader_errors,
    reader_warnings,
)
from osprey.agent_record.contract import KINDS
from osprey.cli.build_profile_deploy import (
    deploy_aware_config_errors,
    deploy_aware_config_warnings,
)
from osprey.cli.build_profile_load import _parse_profile, load_profile_document
from osprey.port_layout import default_port
from osprey_connectors.errors import BuildProfileError


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _parse(block: Any) -> AgentRecordConfig:
    return parse_agent_record_block({"agent_record": block})


def _roster(*names: str, method: str = "token") -> dict[str, Any]:
    return {
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": [{"name": name} for name in names],
                "auth": {"method": method},
            }
        }
    }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_a_profile_with_no_block_gets_the_defaults() -> None:
    cfg = parse_agent_record_block({"name": "site"})

    assert cfg == AgentRecordConfig()
    assert cfg.backend == BACKEND_EMBEDDED
    assert cfg.retention_days == DEFAULT_RETENTION_DAYS == 14
    assert cfg.readers == ()
    assert cfg.port == default_port("agent_record")
    assert cfg.clickhouse.image == DEFAULT_CLICKHOUSE_IMAGE
    assert DEFAULT_CLICKHOUSE_IMAGE == "clickhouse/clickhouse-server:26.8.18.2"
    assert agent_record_errors(cfg) == []


def test_an_unspelled_port_lands_at_the_profiles_base() -> None:
    assert parse_agent_record_block({}, base=30000).port == default_port("agent_record", base=30000)
    stated = parse_agent_record_block({"agent_record": {"port": 4321}}, base=30000)
    assert stated.port == 4321


def test_a_root_profile_with_the_block_parses() -> None:
    block = {
        "backend": "clickhouse",
        "retention_days": 30,
        "readers": [{"name": "ops", "kinds": ["prompt", "answer"], "days_back": 7}],
        "port": 12345,
        "memory": "2g",
        "clickhouse": {"image": "clickhouse/clickhouse-server:custom", "memory": "4g"},
    }

    profile = _parse_profile({"name": "site", "agent_record": block})

    cfg = profile.agent_record
    assert cfg.backend == BACKEND_CLICKHOUSE
    assert cfg.retention_days == 30
    assert cfg.readers == ({"name": "ops", "kinds": ["prompt", "answer"], "days_back": 7},)
    assert cfg.port == 12345
    assert cfg.memory == "2g"
    assert cfg.clickhouse.image == "clickhouse/clickhouse-server:custom"
    assert cfg.clickhouse.memory == "4g"
    assert agent_record_errors(cfg) == []


def test_a_root_profile_file_with_the_block_loads(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    profile = _write_yaml(
        tmp_path / "profile.yml",
        {"name": "site", "data": "data", "agent_record": {"retention_days": 7}},
    )

    load_profile_document(profile)


def test_a_profile_without_the_block_parses_to_the_defaults() -> None:
    profile = _parse_profile({"name": "site"})

    assert profile.agent_record.backend == BACKEND_EMBEDDED
    assert profile.agent_record.retention_days == DEFAULT_RETENTION_DAYS


def test_structural_problems_are_reported_together() -> None:
    with pytest.raises(BuildProfileError) as excinfo:
        _parse(
            {
                "backnd": "embedded",
                "retention_days": True,
                "readers": "ops",
                "clickhouse": {"imag": "x"},
            }
        )

    message = str(excinfo.value)
    assert "Profile 'agent_record' block is invalid" in message
    assert "unknown key 'backnd' (did you mean 'backend'?)" in message
    assert "'retention_days' must be an integer (got True)" in message
    assert "'readers' must be a list of reader entries" in message
    assert "unknown key 'imag' (did you mean 'image'?)" in message


def test_a_block_that_is_not_a_mapping_is_refused() -> None:
    with pytest.raises(BuildProfileError, match="'agent_record' must be a mapping"):
        _parse(["embedded"])


# ---------------------------------------------------------------------------
# Requirement 10: each refusal names its fix
# ---------------------------------------------------------------------------


def test_a_bad_backend_is_refused_naming_the_backends() -> None:
    (error,) = agent_record_errors(_parse({"backend": "duckdb"}))

    assert "agent_record.backend must be one of 'embedded', 'clickhouse'" in error
    assert "(got 'duckdb')" in error
    assert "Remove the key" in error
    assert "backend: clickhouse" in error


def test_the_profile_validation_reports_the_bad_backend(tmp_path: Path) -> None:
    profile = _parse_profile({"name": "site", "agent_record": {"backend": "duckdb"}})

    with pytest.raises(BuildProfileError, match="agent_record.backend must be one of"):
        profile.validate(tmp_path)


def test_a_reader_with_an_unknown_kind_is_refused_naming_the_kinds() -> None:
    cfg = _parse({"readers": [{"name": "ops", "kinds": ["prompt", "tables", "rows"]}]})

    (error,) = reader_errors(cfg, {})

    assert error == (
        "agent_record.readers entry 'ops' names rows, tables, which the record does not "
        f"have; the kinds are {', '.join(KINDS)}"
    )


def test_a_reader_whose_kinds_are_not_a_list_is_refused() -> None:
    (error,) = reader_errors(_parse({"readers": [{"name": "ops", "kinds": "prompt"}]}), {})

    assert "agent_record.readers entry 'ops' kinds must be a list of kind names" in error
    assert "Remove the key to let this reader see every kind" in error


@pytest.mark.parametrize("days", [0, -3, "7", True, 1.5])
def test_a_bad_days_back_is_refused_naming_the_fix(days: Any) -> None:
    (error,) = reader_errors(_parse({"readers": [{"name": "ops", "days_back": days}]}), {})

    assert "agent_record.readers entry 'ops' days_back must be a whole number of days" in error
    assert "Remove the key to let this reader see everything the record keeps" in error


def test_a_days_back_past_retention_is_refused_naming_both_fixes() -> None:
    cfg = _parse({"retention_days": 10, "readers": [{"name": "ops", "days_back": 11}]})

    (error,) = reader_errors(cfg, {})

    assert "days_back is 11, more than agent_record.retention_days (10)" in error
    assert "Lower it or raise retention_days" in error


def test_a_block_in_a_tier_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    _write_yaml(tmp_path / "profile.yml", {"name": "site", "data": "data"})
    tier = _write_yaml(
        tmp_path / "tiers" / "ro.yml", {"name": "ro", "agent_record": {"backend": "embedded"}}
    )

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile_document(tier)

    message = str(excinfo.value)
    assert "'agent_record' (legal only in the deployment's profile.yml)" in message
    assert "valid keys in a tier are:" in message


def test_the_build_lint_refuses_a_bad_reader() -> None:
    cfg = _parse({"readers": [{"name": "ops", "kinds": ["nope"]}]})

    errors = deploy_aware_config_errors(None, {}, agent_record=cfg)

    assert any("which the record does not have" in error for error in errors)
    assert deploy_aware_config_errors(None, {}, agent_record=AgentRecordConfig()) == []


# ---------------------------------------------------------------------------
# The rest of the reader rules, in today's message shapes
# ---------------------------------------------------------------------------


def test_a_reader_without_a_usable_name_is_refused() -> None:
    (error,) = reader_errors(_parse({"readers": [{"name": "Ops"}]}), {})

    assert error.startswith("agent_record.readers[0] must be a mapping whose 'name' is")


def test_a_reader_stated_twice_is_refused() -> None:
    (error,) = reader_errors(_parse({"readers": [{"name": "ops"}, {"name": "ops"}]}), {})

    assert "agent_record.readers names 'ops' more than once" in error


def test_an_unknown_reader_key_is_refused_naming_the_keys() -> None:
    (error,) = reader_errors(_parse({"readers": [{"name": "ops", "tables": ["x"]}]}), {})

    assert "agent_record.readers entry 'ops' names tables" in error
    assert "the keys are name, kinds, days_back, oidc_subject" in error


def test_a_roster_reader_with_a_subject_is_refused() -> None:
    cfg = _parse({"readers": [{"name": "alice", "oidc_subject": "sub-1"}]})

    (error,) = reader_errors(cfg, _roster("alice", method="oidc"))

    assert "oidc_subject is set, and 'alice' is also a modules.web_terminals.users" in error
    assert "Remove oidc_subject from the reader entry" in error


def test_an_empty_subject_is_refused() -> None:
    cfg = _parse({"readers": [{"name": "ops", "oidc_subject": " "}]})

    assert reader_errors(cfg, {}) == [
        "agent_record.readers entry 'ops' oidc_subject must be a non-empty string"
    ]


def test_a_subject_without_oidc_is_a_warning_not_an_error() -> None:
    cfg = _parse({"readers": [{"name": "ops", "oidc_subject": "sub-1"}]})

    assert reader_errors(cfg, _roster("alice")) == []
    (warning,) = reader_warnings(cfg, _roster("alice"))
    assert "auth.method is 'token'" in warning
    assert deploy_aware_config_warnings(None, _roster("alice"), agent_record=cfg)[-1] == warning


def test_a_sound_reader_passes() -> None:
    cfg = _parse(
        {
            "retention_days": 14,
            "readers": [{"name": "ops", "kinds": ["prompt"], "days_back": 14, "oidc_subject": "s"}],
        }
    )

    assert reader_errors(cfg, _roster("alice", method="oidc")) == []
    assert reader_warnings(cfg, _roster("alice", method="oidc")) == []


# ---------------------------------------------------------------------------
# Value rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("block", "fragment"),
    [
        ({"retention_days": 0}, "agent_record.retention_days must be a whole number of days"),
        ({"port": 70000}, "agent_record.port must be in 1..65535"),
        ({"memory": "lots"}, "agent_record.memory must be a memory size"),
        ({"clickhouse": {"memory": "0"}}, "agent_record.clickhouse.memory must be a memory size"),
        ({"clickhouse": {"image": " "}}, "agent_record.clickhouse.image must be a non-empty"),
    ],
)
def test_value_rules(block: dict[str, Any], fragment: str) -> None:
    errors = agent_record_errors(_parse(block))

    assert any(fragment in error for error in errors), errors


@pytest.mark.parametrize(
    "config", [{"agent_record.backend": "clickhouse"}, {"agent_record": {"backend": "x"}}]
)
def test_the_config_block_may_not_state_the_record_settings(config: dict[str, Any]) -> None:
    (error,) = agent_record_errors(AgentRecordConfig(), config)

    assert "top-level `agent_record:` block" in error


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def test_the_resolved_block_renders_every_key() -> None:
    cfg = _parse({"readers": [{"name": "ops", "kinds": ["prompt"]}]})

    assert agent_record_config_overrides(cfg) == {
        "agent_record.backend": "embedded",
        "agent_record.retention_days": 14,
        "agent_record.readers": [{"name": "ops", "kinds": ["prompt"]}],
        "agent_record.port": default_port("agent_record"),
        "agent_record.memory": "1g",
        "agent_record.clickhouse.image": DEFAULT_CLICKHOUSE_IMAGE,
        "agent_record.clickhouse.memory": "2g",
    }


def test_the_rendered_block_is_what_the_credential_table_reads(tmp_path: Path) -> None:
    """The render shape is the one ``identities.reader_entries`` consumes."""
    from osprey.agent_record.identities import credential_table
    from osprey.utils.config_writer import config_update_fields

    config_path = tmp_path / "config.yml"
    config_path.write_text("{}\n", encoding="utf-8")
    cfg = _parse({"readers": [{"name": "ops", "kinds": ["prompt"], "days_back": 3}]})
    config_update_fields(config_path, agent_record_config_overrides(cfg))
    rendered = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert rendered["agent_record"]["backend"] == "embedded"
    assert rendered["agent_record"]["clickhouse"]["image"] == DEFAULT_CLICKHOUSE_IMAGE
    grant = credential_table(rendered)["reader.ops"]
    assert grant.scope.kinds == frozenset({"prompt"})
    assert grant.scope.days_back == 3


def test_a_deploying_profile_claims_the_record_port() -> None:
    profile = _parse_profile({"name": "site", "agent_record": {"port": 23456}})

    assert profile._claimed_ports()["agent_record.port"] == 23456


# ---------------------------------------------------------------------------
# host_identity on a store-less build (P2, B1-11)
# ---------------------------------------------------------------------------


def test_a_store_less_build_stamps_host_identity_and_files_audit_under_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded_entries
) -> None:
    from osprey.agent_record import writer as record_writer
    from osprey.agent_record.contract import SPOOL_DIR_ENV
    from osprey.agent_record.credentials import export_built_host_identity
    from osprey.audit import writer
    from osprey.cli import build_cmd
    from osprey_connectors.identity import IDENTITY_ENV_LADDER

    monkeypatch.setattr("getpass.getuser", lambda: "builder")
    render = tmp_path / "build"
    render.mkdir()
    (render / "config.yml").write_text(yaml.safe_dump({"deployed_services": []}), encoding="utf-8")
    rendered = yaml.safe_load((render / "config.yml").read_text(encoding="utf-8"))
    assert build_cmd._record_host_identity(render, rendered) is True

    stamped = yaml.safe_load((render / "config.yml").read_text(encoding="utf-8"))
    assert stamped["deployment"]["host_identity"] == "builder"

    # A host launch exports that name: an audit entry names it as the actor and
    # the default spool directory is its own, even though the account running
    # the launch is someone else.
    monkeypatch.setattr("getpass.getuser", lambda: "someone-else")
    for name in IDENTITY_ENV_LADDER:
        monkeypatch.delenv(name, raising=False)
    environ: dict[str, str] = {}
    exported = export_built_host_identity(environ, tmp_path)
    assert exported
    for name, value in environ.items():
        monkeypatch.setenv(name, value)

    entry_id = writer.record(
        surface="terminal",
        posture="sandbox",
        posture_source="process",
        session=None,
        subject="host-launch",
        decision="allowed",
        reason="allowed",
    )
    assert entry_id is not None
    (entry,) = recorded_entries("audit")
    assert entry.body["actor"] == "builder"

    monkeypatch.delenv(SPOOL_DIR_ENV)
    assert record_writer.spool_dir().name == "builder"


# ── Names the record shares with the history page and the render ─────────────

MIB = 1024 * 1024


def test_the_shared_names_keep_their_values() -> None:
    assert HISTORY_SERVICE == "history"
    assert STORE_HTTP_CONTAINER_PORT == 8123
    assert STORE_MEMORY_RATIO == 0.9


def test_the_history_ports_default_to_their_slots_at_the_deployments_base() -> None:
    config = {"deployment": {"port_base": 20000}}
    assert history_origin_port(config) == 20002
    assert history_app_port(config) == 20003
    assert history_origin_port({}) == default_port("history_origin")
    assert history_app_port({}) == default_port("history")


def test_a_stated_history_origin_port_wins_over_the_layout() -> None:
    config = {"modules": {"web_terminals": {"history_port": 8443}}}
    assert history_origin_port(config) == 8443


@pytest.mark.parametrize("value", ["8443", 0, 70000, True, 1.5])
def test_an_unusable_history_origin_port_is_refused_naming_the_key(value: Any) -> None:
    config = {"modules": {"web_terminals": {"history_port": value}}}
    with pytest.raises(
        ValueError, match=r"modules\.web_terminals\.history_port must be an integer port"
    ):
        history_origin_port(config)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2g", 2 * 1024 * MIB),
        ("2GiB", 2 * 1024 * MIB),
        ("512m", 512 * MIB),
        ("1.5gb", 1536 * MIB),
        ("64k", 65536),
        (1048576, MIB),
        ("1048576", MIB),
    ],
)
def test_memory_is_read_as_compose_reads_it(value: Any, expected: int) -> None:
    assert parse_memory(value, "k") == expected


@pytest.mark.parametrize("value", ["", "2 gigs", "g", 0, -1, True, None, "0m"])
def test_an_unusable_memory_value_is_refused_naming_the_key(value: Any) -> None:
    with pytest.raises(ValueError, match=r"agent_record\.memory must be a memory size"):
        parse_memory(value, "agent_record.memory")


def test_an_unusable_record_memory_is_reported_naming_the_key() -> None:
    errors = agent_record_errors(AgentRecordConfig(memory="2 gigs"))
    assert any(e.startswith("agent_record.memory must be a memory size") for e in errors)


def test_the_history_pages_max_days_is_the_records_retention() -> None:
    """The page offers what the record keeps."""
    from osprey.deployment.compose_generator import _resolve_history_render_context

    stated = {"agent_record": {"retention_days": 45}}
    assert _resolve_history_render_context(stated)["max_days"] == 45
    assert _resolve_history_render_context({})["max_days"] == DEFAULT_RETENTION_DAYS
