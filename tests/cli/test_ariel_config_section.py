"""Tests for how the ARIEL CLI reads its config section."""

import pytest

import osprey.cli.ariel as ariel_cli
from osprey_connectors.testing import config_of
from osprey_connectors_tests._config_trees import serve_config


def test_a_non_mapping_ariel_section_is_refused_as_unconfigured():
    """An ``ariel:`` section that is not a mapping takes the not-configured path."""
    serve_config(config_of({"ariel": True}))

    with pytest.raises(SystemExit) as excinfo:
        ariel_cli._load_ariel_config()

    assert excinfo.value.code == 1
