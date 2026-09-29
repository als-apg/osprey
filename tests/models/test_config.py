"""Provider and model lookups on a loaded ``config.yml``."""

from __future__ import annotations

import pytest

from osprey.models.config import (
    get_model_config,
    get_provider_config,
    main_model_id,
    provider_requests_per_minute,
)


def _config(*, provider="als-apg", default_model=None, entry_default="claude-haiku-4-5-20251001"):
    claude_code = {"provider": provider}
    if default_model:
        claude_code["default_model"] = default_model
    entry = {"base_url": "https://gateway.example", "models": ["a", "b"]}
    if entry_default:
        entry["default_model"] = entry_default
    return {"claude_code": claude_code, "api": {"providers": {"als-apg": entry}}}


def test_the_deployment_default_model_wins():
    assert main_model_id(_config(default_model="claude-sonnet-5"), "als-apg") == "claude-sonnet-5"


def test_the_provider_entry_default_answers_when_the_deployment_names_none():
    assert main_model_id(_config(), "als-apg") == "claude-haiku-4-5-20251001"


def test_a_job_on_another_provider_takes_that_provider_default():
    config = _config(provider="cborg", default_model="claude-sonnet-5")
    assert main_model_id(config, "als-apg") == "claude-haiku-4-5-20251001"


def test_the_deployment_default_applies_when_no_deployment_provider_is_named():
    config = {"claude_code": {"default_model": "claude-opus-5-5"}, "api": {"providers": {}}}
    assert main_model_id(config, "als-apg") == "claude-opus-5-5"


def test_neither_key_is_an_error_naming_both():
    with pytest.raises(ValueError) as excinfo:
        main_model_id(_config(entry_default=None), "als-apg")
    message = str(excinfo.value)
    assert "claude_code.default_model" in message
    assert "api.providers.als-apg.default_model" in message


def _gateway(**extra):
    entry = {"base_url": "https://gateway.example", "default_model": "m-1", "models": ["m-1"]}
    return {"api": {"providers": {"gw": {**entry, **extra}}}}


def test_a_provider_cap_is_read_from_its_catalog_entry():
    assert provider_requests_per_minute(_gateway(requests_per_minute=18), "gw") == 18


def test_a_provider_without_a_cap_is_not_paced():
    assert provider_requests_per_minute(_gateway(), "gw") is None
    assert provider_requests_per_minute(_gateway(requests_per_minute=18), "other") is None
    assert provider_requests_per_minute({"claude_code": {"provider": "gw"}}, "gw") is None


@pytest.mark.parametrize("value", [0, -1, 2.5, "18", True, None])
def test_a_malformed_cap_is_refused_naming_the_key(value):
    with pytest.raises(ValueError) as excinfo:
        provider_requests_per_minute(_gateway(requests_per_minute=value), "gw")
    assert "api.providers.gw.requests_per_minute" in str(excinfo.value)


def test_a_non_mapping_config_entry_reads_as_absent(monkeypatch):
    import osprey_connectors.config as connectors_config

    monkeypatch.setattr(
        connectors_config,
        "_get_configurable",
        lambda config_path=None: {"model_configs": "oops", "provider_configs": {"p": "oops"}},
    )
    assert get_model_config("m") == {}
    assert get_provider_config("p") == {}
