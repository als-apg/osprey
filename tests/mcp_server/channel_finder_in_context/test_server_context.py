"""Tests for ChannelFinderICContext."""

import json

import pytest

from osprey.mcp_server.channel_finder_in_context.server_context import (
    PROVIDER_RPM,
    get_cf_ic_context,
    initialize_cf_ic_context,
)
from osprey.services.channel_finder.rate_limiter import configure_rate_limiter, get_rate_limiter

_MINIMAL_MODEL_CONFIG = "claude_code:\n  model: test-model\n  provider: anthropic\n"


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    yield
    configure_rate_limiter(None)


@pytest.mark.unit
def test_context_not_initialized():
    with pytest.raises(RuntimeError, match="not initialized"):
        get_cf_ic_context()


@pytest.mark.unit
def test_context_database_not_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(_MINIMAL_MODEL_CONFIG)
    initialize_cf_ic_context()
    reg = get_cf_ic_context()
    with pytest.raises(RuntimeError, match="not available"):
        _ = reg.database


@pytest.mark.unit
def test_context_loads_flat_database(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_data = [
        {"channel": "CH1", "address": "PV:CH1", "description": "Channel 1"},
        {"channel": "CH2", "address": "PV:CH2", "description": "Channel 2"},
    ]
    db_file = tmp_path / "test_db.json"
    db_file.write_text(json.dumps(db_data))
    config = (
        _MINIMAL_MODEL_CONFIG
        + "channel_finder:\n"
        + "  pipelines:\n"
        + "    in_context:\n"
        + "      database:\n"
        + f'        path: "{db_file}"\n'
        + '        type: "flat"\n'
    )
    (tmp_path / "config.yml").write_text(config)
    initialize_cf_ic_context()
    reg = get_cf_ic_context()
    assert reg.database is not None
    assert len(reg.database.get_all_channels()) == 2


@pytest.mark.unit
def test_context_facility_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(_MINIMAL_MODEL_CONFIG + 'facility:\n  name: "ALS"\n')
    initialize_cf_ic_context()
    assert get_cf_ic_context().facility_name == "ALS"


@pytest.mark.unit
def test_context_raises_when_no_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    with pytest.raises(RuntimeError, match="channel_finder.pipelines.in_context.subagent_model"):
        initialize_cf_ic_context()


@pytest.mark.unit
def test_context_subagent_model_from_ic_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = (
        "claude_code:\n  model: fallback-model\n  provider: anthropic\n"
        "channel_finder:\n  pipelines:\n    in_context:\n      subagent_model: ic-model\n"
    )
    (tmp_path / "config.yml").write_text(config)
    initialize_cf_ic_context()
    reg = get_cf_ic_context()
    assert reg.subagent_model_id == "ic-model"
    assert reg.subagent_provider == "anthropic"


@pytest.mark.unit
def test_context_subagent_model_fallback_to_claude_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(_MINIMAL_MODEL_CONFIG)
    initialize_cf_ic_context()
    reg = get_cf_ic_context()
    assert reg.subagent_model_id == "test-model"


@pytest.mark.unit
def test_context_rate_limiter_armed_for_cborg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("claude_code:\n  model: m\n  provider: cborg\n")
    initialize_cf_ic_context()
    limiter = get_rate_limiter()
    assert limiter is not None
    assert limiter.max_calls == PROVIDER_RPM["cborg"]


@pytest.mark.unit
def test_context_rate_limiter_none_for_anthropic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(_MINIMAL_MODEL_CONFIG)
    initialize_cf_ic_context()
    assert get_rate_limiter() is None


@pytest.mark.unit
def test_context_system_prompt_contains_final_tags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_data = [{"channel": "CH1", "address": "PV:CH1", "description": "Channel 1"}]
    db_file = tmp_path / "test_db.json"
    db_file.write_text(json.dumps(db_data))
    config = (
        _MINIMAL_MODEL_CONFIG
        + "channel_finder:\n  pipelines:\n    in_context:\n"
        + f'      database:\n        path: "{db_file}"\n        type: "flat"\n'
    )
    (tmp_path / "config.yml").write_text(config)
    initialize_cf_ic_context()
    prompt = get_cf_ic_context().system_prompt_with_db
    assert "<final>" in prompt
    assert "</final>" in prompt
    assert "CH1" in prompt
