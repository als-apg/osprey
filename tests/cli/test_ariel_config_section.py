"""Tests for how the ARIEL CLI reads its config section."""

import pytest

import osprey.cli.ariel as ariel_cli


def test_a_non_mapping_ariel_section_is_refused_as_unconfigured(monkeypatch):
    """An ``ariel:`` section that is not a mapping takes the not-configured path."""
    monkeypatch.setattr(ariel_cli, "get_config_value", lambda *_args, **_kwargs: True)

    with pytest.raises(SystemExit) as excinfo:
        ariel_cli._load_ariel_config()

    assert excinfo.value.code == 1
