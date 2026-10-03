"""Tests for the packaged ontology branch set used by the MML mapping."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from importlib import resources

import pytest

from osprey.services.mml.mapping.branches import (
    ROOT_CLASS,
    PackagedClass,
    is_pn_local,
    packaged_classes,
)


def _vocabulary_classes() -> dict[str, dict]:
    table = resources.files("osprey.facility.schema._generated") / "vocabulary.json"
    rows = json.loads(table.read_text(encoding="utf-8"))["classes"]
    return {row["name"]: row for row in rows}


def test_expected_branches_present() -> None:
    classes = packaged_classes()
    for name in ("Instrumentation", "Magnet", "RadioFrequency", "Vacuum", "HCorrector"):
        assert name in classes


def test_hcorrector_parent_is_corrector() -> None:
    assert packaged_classes()["HCorrector"].parent == "Corrector"


def test_every_entry_is_a_frozen_record_equal_to_the_vocabulary() -> None:
    table = _vocabulary_classes()
    for name, klass in packaged_classes().items():
        assert isinstance(klass, PackagedClass)
        row = table[name]
        assert klass.name == name == row["name"]
        assert klass.parent == row["parent"]
        assert klass.iri == row["iri"]
        assert klass.alt_labels == tuple(row["aliases"])
        assert klass.parent is not None


def test_record_is_frozen() -> None:
    klass = packaged_classes()["HCorrector"]
    with pytest.raises(FrozenInstanceError):
        klass.parent = "Magnet"  # type: ignore[misc]


def test_root_excluded_and_matches_packaged_table() -> None:
    table = _vocabulary_classes()
    roots = [name for name, row in table.items() if row["parent"] is None]
    assert roots == [ROOT_CLASS]
    assert ROOT_CLASS not in packaged_classes()
    assert set(packaged_classes()) == set(table) - {ROOT_CLASS}


def test_packaged_classes_returns_independent_copy() -> None:
    first = packaged_classes()
    first.pop("HCorrector")
    assert "HCorrector" in packaged_classes()


@pytest.mark.parametrize("token", ["Magnet", "_x", "HCM1", "a_b_2"])
def test_is_pn_local_accepts(token: str) -> None:
    assert is_pn_local(token)


@pytest.mark.parametrize("token", ["", "1abc", "has space", "a-b", "Magnet\n", "é"])
def test_is_pn_local_rejects(token: str) -> None:
    assert not is_pn_local(token)
