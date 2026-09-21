"""Tests for the packaged ontology branch set used by the MML mapping."""

from __future__ import annotations

import pytest

from osprey.services.facility_knowledge.ttl_generator.ontology_map import (
    ClassDef,
    load_demo_ontology,
)
from osprey.services.mml.mapping.branches import ROOT_CLASS, is_pn_local, packaged_classes


def test_expected_branches_present() -> None:
    classes = packaged_classes()
    for name in ("Instrumentation", "Magnet", "RadioFrequency", "Vacuum", "HCorrector"):
        assert name in classes


def test_hcorrector_parent_is_corrector() -> None:
    assert packaged_classes()["HCorrector"].parent == "Corrector"


def test_every_entry_is_a_classdef_with_an_iri() -> None:
    for name, klass in packaged_classes().items():
        assert isinstance(klass, ClassDef)
        assert klass.name == name
        assert klass.iri
        assert klass.parent is not None


def test_root_excluded_and_matches_packaged_table() -> None:
    table = load_demo_ontology().classes
    roots = [name for name, klass in table.items() if klass.parent is None]
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
