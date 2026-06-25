"""Unit tests for tests/e2e/sdk_helpers.py pure-logic helpers.

No LLM, no build, no DB — fast logic checks. Excluded from the model-capability
matrix (scripts/benchmark/matrix_e2e_config.json) since it measures nothing
about the model under test.
"""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import _DEFAULT_ARIEL_DB_URI, _override_ariel_db_uri

_PER_CELL_URI = "postgresql://ariel:ariel@localhost:5432/ariel_gpt-oss-20b_seed1"


def _write_config(tmp_path, uri: str = _DEFAULT_ARIEL_DB_URI):
    """Write a minimal rendered config.yml with an ARIEL DB uri line."""
    (tmp_path / "config.yml").write_text(
        "ariel:\n"
        "  database:\n"
        f"    uri: {uri}\n"
        "  default_max_results: 10\n",
        encoding="utf-8",
    )
    return tmp_path / "config.yml"


@pytest.mark.unit
def test_override_rewrites_uri_when_env_set(tmp_path, monkeypatch):
    """With OSPREY_ARIEL_DB_URI set, the rendered config points at the per-cell DB."""
    config_path = _write_config(tmp_path)
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", _PER_CELL_URI)

    _override_ariel_db_uri(tmp_path)

    text = config_path.read_text(encoding="utf-8")
    assert _PER_CELL_URI in text
    # The bare default must no longer stand alone as the configured uri.
    assert f"uri: {_DEFAULT_ARIEL_DB_URI}\n" not in text


@pytest.mark.unit
def test_override_noop_when_env_unset(tmp_path, monkeypatch):
    """No override env → config is left untouched (shared default DB)."""
    config_path = _write_config(tmp_path)
    monkeypatch.delenv("OSPREY_ARIEL_DB_URI", raising=False)

    _override_ariel_db_uri(tmp_path)

    assert f"uri: {_DEFAULT_ARIEL_DB_URI}\n" in config_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_override_noop_when_env_equals_default(tmp_path, monkeypatch):
    """Override that equals the default is a no-op (no needless rewrite)."""
    config_path = _write_config(tmp_path)
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", _DEFAULT_ARIEL_DB_URI)

    _override_ariel_db_uri(tmp_path)

    assert f"uri: {_DEFAULT_ARIEL_DB_URI}\n" in config_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_override_noop_for_non_ariel_project(tmp_path, monkeypatch):
    """A project without the default ARIEL URI (e.g. the hello_world preset,
    ``ariel: {enabled: false}``) is left untouched — there is no real ARIEL DB
    to redirect, so the override must not fail the build.
    """
    (tmp_path / "config.yml").write_text(
        "ariel: {enabled: false}\nmodel: haiku\n", encoding="utf-8"
    )
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", _PER_CELL_URI)

    # Must not raise.
    _override_ariel_db_uri(tmp_path)

    text = (tmp_path / "config.yml").read_text(encoding="utf-8")
    assert _PER_CELL_URI not in text
    assert "ariel: {enabled: false}" in text
