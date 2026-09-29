"""Verify the build-artifact package's public API, and that no copy of it remains at a former path."""

from __future__ import annotations

import importlib

import pytest


def test_public_api_surface() -> None:
    from osprey.agent_runner.build_artifacts import BuildArtifact, BuildArtifactCatalog

    catalog = BuildArtifactCatalog.default()
    names = catalog.all_names()
    assert len(names) > 0

    sample = catalog.get(names[0])
    assert isinstance(sample, BuildArtifact)
    assert sample.canonical_name == names[0]


def test_legacy_package_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("osprey.services.prompts")


def test_the_catalog_has_no_home_outside_the_harness_adapter() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("osprey.services.build_artifacts")
