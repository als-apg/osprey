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
