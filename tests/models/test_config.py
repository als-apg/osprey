"""``main_model_id``: the model a job runs when it names none of its own."""

from __future__ import annotations

import pytest

from osprey.models.config import main_model_id


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
