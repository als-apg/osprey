"""Tests for the channel-finder config path and loader in ``channel_finder_common``."""

from __future__ import annotations

import logging

from osprey.mcp_server.channel_finder_common import (
    load_cf_config,
    resolve_cf_path,
    resolve_cf_state_path,
)

LOGGER_NAME = "test.channel_finder_common"


def _logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def _warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno >= logging.WARNING]


def test_osprey_config_is_read_with_its_variables_expanded(tmp_path, monkeypatch):
    cf_dir = tmp_path / "deploy"
    cf_dir.mkdir()
    (cf_dir / "config.yml").write_text("facility:\n  name: ERF\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CF_DIR", str(cf_dir))
    monkeypatch.setenv("OSPREY_CONFIG", "$CF_DIR/config.yml")

    assert load_cf_config(_logger()) == {"facility": {"name": "ERF"}}
    assert resolve_cf_path("data/x.json") == str((cf_dir / "data" / "x.json").resolve())


def test_a_render_under_the_cwd_is_found_without_osprey_config(tmp_path, monkeypatch, caplog):
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "config.yml").write_text("facility:\n  name: ERF\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    assert load_cf_config(_logger()) == {"facility": {"name": "ERF"}}
    assert _warnings(caplog) == []
    root = tmp_path.resolve()
    assert resolve_cf_path("data/x.json") == str(root / "build" / "data" / "x.json")
    assert resolve_cf_state_path("var/x.json") == str(root / "var" / "x.json")


def test_a_flat_project_config_under_the_cwd_is_found(tmp_path, monkeypatch):
    (tmp_path / "config.yml").write_text("facility:\n  name: ERF\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)

    assert load_cf_config(_logger()) == {"facility": {"name": "ERF"}}
    root = tmp_path.resolve()
    assert resolve_cf_path("data/x.json") == str(root / "data" / "x.json")
    assert resolve_cf_state_path("var/x.json") == str(root / "var" / "x.json")


def _write_osprey_config(tmp_path, monkeypatch, text: str):
    config = tmp_path / "config.yml"
    config.write_text(text)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    return config


def _ml_database(raw: dict) -> dict:
    return raw["channel_finder"]["pipelines"]["middle_layer"]["database"]


_PLACEHOLDER_CONFIG = (
    "channel_finder:\n  pipelines:\n    middle_layer:\n      database:\n        path: {value}\n"
)


def test_env_placeholders_under_channel_finder_resolve(tmp_path, monkeypatch, caplog):
    _write_osprey_config(tmp_path, monkeypatch, _PLACEHOLDER_CONFIG.format(value="${CF_DB}"))
    monkeypatch.setenv("CF_DB", "/abs/db.json")
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    raw = load_cf_config(_logger())

    assert _ml_database(raw) == {"path": "/abs/db.json"}
    assert _warnings(caplog) == []


def test_a_placeholder_default_applies_when_the_variable_is_unset(tmp_path, monkeypatch, caplog):
    _write_osprey_config(
        tmp_path, monkeypatch, _PLACEHOLDER_CONFIG.format(value="${CF_DB:-data/demo.json}")
    )
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    monkeypatch.delenv("CF_DB", raising=False)
    assert _ml_database(load_cf_config(_logger())) == {"path": "data/demo.json"}

    import osprey.utils.config as _cfg

    _cfg._config_cache.clear()
    monkeypatch.setenv("CF_DB", "")
    assert _ml_database(load_cf_config(_logger())) == {"path": "data/demo.json"}
    assert _warnings(caplog) == []


def test_a_missing_config_warns_and_returns_empty(tmp_path, monkeypatch, caplog):
    missing = tmp_path / "absent.yml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OSPREY_CONFIG", str(missing))
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    assert load_cf_config(_logger()) == {}
    records = _warnings(caplog)
    assert len(records) == 1
    assert "Config file not found" in records[0].getMessage()
    assert str(missing) in records[0].getMessage()


def test_malformed_yaml_warns_and_returns_empty(tmp_path, monkeypatch, caplog):
    config = _write_osprey_config(tmp_path, monkeypatch, "channel_finder: [unclosed\n")
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    assert load_cf_config(_logger()) == {}
    messages = [r.getMessage() for r in _warnings(caplog)]
    assert any(str(config) in m and "could not be loaded" in m for m in messages)


def test_a_non_mapping_config_warns_and_returns_empty(tmp_path, monkeypatch, caplog):
    config = _write_osprey_config(tmp_path, monkeypatch, "- a\n- b\n")
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    assert load_cf_config(_logger()) == {}
    messages = [r.getMessage() for r in _warnings(caplog)]
    assert any(str(config) in m and "could not be loaded" in m for m in messages)


def test_the_primed_builder_is_reused(tmp_path, monkeypatch):
    import osprey.utils.config as _cfg
    from osprey.mcp_server.startup import prime_config_builder

    _write_osprey_config(tmp_path, monkeypatch, "facility:\n  name: ERF\n")
    prime_config_builder()
    primed = _cfg._default_config
    assert primed is not None

    raw = load_cf_config(_logger())

    assert _cfg._default_config is primed
    assert raw is primed.raw_config
