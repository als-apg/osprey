"""The relation stage's record warnings: shared aliases, ambiguous identity, part cycles, stubs,
and the places relations layers create."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.build import LATER_STAGES
from osprey.facility.validate import StageReport, run_stages


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = data if isinstance(data, str) else yaml.safe_dump(data, sort_keys=False)
        path.write_text(text, encoding="utf-8")
    return root


def _run(root: Path, files: dict[str, Any]) -> StageReport:
    report = run_stages(_write(root, files), project_name="p", later=LATER_STAGES)
    assert report.failed is None, [e.format_message() for e in report.errors]
    return report


def _lines(report: StageReport, kind: str) -> list[str]:
    prefix = f"facility: {kind}:"
    return [w.line for w in report.validated.warnings if w.line.startswith(prefix)]


def _claim(rid: str, rtype: str, subject: str, obj: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": rid,
        "type": rtype,
        "subject": {"device": subject},
        "object": {"device": obj},
        **extra,
    }


def _device(rid: str, cls: str = "Quadrupole", **extra: Any) -> dict[str, Any]:
    return {"id": rid, "class": cls, **extra}


def _alias(name: str, scheme: str = "legacy") -> dict[str, Any]:
    return {"name": name, "scheme": scheme}


_DOCS = [
    {"id": "spec-1", "title": "Magnet list", "kind": "spec"},
    {"id": "dwg-1", "title": "Cabling drawing", "kind": "drawing"},
]

_POLICY = {
    "schema": "osprey.facility.policy/1",
    "relations": {"precedence": {"default": ["spec", "drawing"]}},
}


def _tree(devices: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    files: dict[str, Any] = {
        "records/places.yaml": [{"id": "SR"}, {"id": "SR/S01"}],
        "records/devices.yaml": devices,
        "documents.yaml": _DOCS,
    }
    files.update(extra)
    return files


class TestAliasShared:
    def test_two_records_with_one_name_in_a_scheme_warn(self, tmp_path: Path) -> None:
        files = _tree(
            [
                _device("QF1", scheme_names=[_alias("Q1")]),
                _device("QF2", scheme_names=[_alias("Q1"), _alias("Q2")]),
                _device("QF3", scheme_names=[_alias("Q1", scheme="other")]),
            ]
        )
        assert _lines(_run(tmp_path, files), "alias-shared") == [
            "facility: alias-shared: scheme legacy Q1 — devices QF1, QF2 carry the name; "
            "fix: give each record its own name in the scheme if the scheme is one-to-one"
        ]

    def test_a_place_and_a_device_sharing_a_name_warn(self, tmp_path: Path) -> None:
        files = _tree(
            [_device("QF1", scheme_names=[_alias("S1")])],
            **{
                "records/places.yaml": [
                    {"id": "SR"},
                    {"id": "SR/S01", "scheme_names": [_alias("S1")]},
                ]
            },
        )
        assert _lines(_run(tmp_path, files), "alias-shared") == [
            "facility: alias-shared: scheme legacy S1 — device QF1, place SR/S01 carry the name; "
            "fix: give each record its own name in the scheme if the scheme is one-to-one"
        ]

    def test_distinct_names_do_not_warn(self, tmp_path: Path) -> None:
        files = _tree(
            [
                _device("QF1", scheme_names=[_alias("Q1")]),
                _device("QF2", scheme_names=[_alias("Q2")]),
            ]
        )
        assert _lines(_run(tmp_path, files), "alias-shared") == []


def _layer_devices(*ids: str, cls: str = "Quadrupole") -> list[dict[str, Any]]:
    return [_device(rid, cls) for rid in ids]


class TestIdentityAmbiguous:
    def test_one_import_describing_one_thing_twice_warns(self, tmp_path: Path) -> None:
        files = _tree(
            [_device("C1")],
            **{
                "imported/lat/devices.yaml": _layer_devices("A1", "A2"),
                "records/relations.yaml": [
                    _claim("s1", "same_as", "A1", "C1"),
                    _claim("s2", "same_as", "C1", "A2"),
                ],
            },
        )
        assert _lines(_run(tmp_path, files), "identity-ambiguous") == [
            "facility: identity-ambiguous: layer lat — same_as joins devices A1, A2, C1 into one, "
            "and this layer states A1, A2 apart; "
            "fix: correct the same_as claims or the import that describes one device twice"
        ]

    def test_records_from_different_layers_do_not_warn(self, tmp_path: Path) -> None:
        files = _tree(
            [],
            **{
                "imported/a/devices.yaml": _layer_devices("A1"),
                "imported/b/devices.yaml": _layer_devices("B1"),
                "records/relations.yaml": [_claim("s1", "same_as", "A1", "B1")],
            },
        )
        assert _lines(_run(tmp_path, files), "identity-ambiguous") == []

    def test_authored_records_are_exempt(self, tmp_path: Path) -> None:
        files = _tree(
            [_device("C1"), _device("C2")],
            **{"records/relations.yaml": [_claim("s1", "same_as", "C1", "C2")]},
        )
        assert _lines(_run(tmp_path, files), "identity-ambiguous") == []


class TestPartCycle:
    def test_part_of_and_runs_on_together_form_one_cycle(self, tmp_path: Path) -> None:
        files = _tree(
            [_device("C", "PowerSupply"), _device("A", "PowerSupply"), _device("B", "PowerSupply")],
            **{
                "records/relations.yaml": [
                    _claim("p1", "part_of", "B", "C"),
                    _claim("p2", "runs_on", "C", "A"),
                    _claim("p3", "part_of", "A", "B"),
                ]
            },
        )
        assert _lines(_run(tmp_path, files), "part-cycle") == [
            "facility: part-cycle: device A — part_of and runs_on claims loop A -> B -> C -> A; "
            "fix: correct the claim that closes the loop"
        ]

    def test_two_cycles_give_two_lines(self, tmp_path: Path) -> None:
        files = _tree(
            [_device(rid, "PowerSupply") for rid in ("A", "B", "X", "Y")],
            **{
                "records/relations.yaml": [
                    _claim("p1", "part_of", "A", "B"),
                    _claim("p2", "part_of", "B", "A"),
                    _claim("p3", "runs_on", "Y", "X"),
                    _claim("p4", "runs_on", "X", "Y"),
                ]
            },
        )
        assert _lines(_run(tmp_path, files), "part-cycle") == [
            "facility: part-cycle: device A — part_of and runs_on claims loop A -> B -> A; "
            "fix: correct the claim that closes the loop",
            "facility: part-cycle: device X — part_of and runs_on claims loop X -> Y -> X; "
            "fix: correct the claim that closes the loop",
        ]

    def test_a_chain_does_not_warn(self, tmp_path: Path) -> None:
        files = _tree(
            [_device(rid, "PowerSupply") for rid in ("A", "B", "C")],
            **{
                "records/relations.yaml": [
                    _claim("p1", "part_of", "A", "B"),
                    _claim("p2", "part_of", "B", "C"),
                ]
            },
        )
        assert _lines(_run(tmp_path, files), "part-cycle") == []

    def test_a_superseded_edge_breaks_the_cycle(self, tmp_path: Path) -> None:
        devices = [_device(rid, "PowerSupply") for rid in ("A", "B", "C")]
        claims = [
            _claim("p1", "part_of", "A", "B", source={"document": "spec-1"}),
            _claim("p2", "part_of", "B", "A", source={"document": "dwg-1"}),
            _claim("p3", "part_of", "B", "C", source={"document": "spec-1"}),
        ]
        contested = _run(
            tmp_path / "contested", _tree(devices, **{"records/relations.yaml": claims})
        )
        assert _lines(contested, "part-cycle") == [
            "facility: part-cycle: device A — part_of and runs_on claims loop A -> B -> A; "
            "fix: correct the claim that closes the loop"
        ]
        ranked = _run(
            tmp_path / "ranked",
            _tree(devices, **{"records/relations.yaml": claims, "policy.yaml": _POLICY}),
        )
        assert _lines(ranked, "part-cycle") == []


class TestRelationStub:
    def test_unidentified_devices_named_by_claims_count_per_layer(self, tmp_path: Path) -> None:
        files = _tree(
            [_device("PS1", "PowerSupply"), _device("PS2", "PowerSupply")],
            **{
                "imported/cab/devices.yaml": _layer_devices("U1", "U2", "U3", cls="Unidentified"),
                "imported/net/devices.yaml": _layer_devices("N1", cls="Unidentified"),
                "records/relations.yaml": [
                    _claim("r1", "part_of", "U1", "PS1"),
                    _claim("r2", "part_of", "U2", "PS1"),
                    _claim("r3", "runs_on", "N1", "PS2"),
                ],
            },
        )
        assert _lines(_run(tmp_path, files), "relation-stub") == [
            "facility: relation-stub: layer cab — 2 devices of class Unidentified stand only "
            "for claim ends; fix: import the devices the claims name, or name known devices",
            "facility: relation-stub: layer net — 1 device of class Unidentified stands only "
            "for claim ends; fix: import the devices the claims name, or name known devices",
        ]

    def test_authored_unidentified_devices_never_count(self, tmp_path: Path) -> None:
        files = _tree(
            [_device("PS1", "PowerSupply"), _device("U1", "Unidentified")],
            **{"records/relations.yaml": [_claim("r1", "part_of", "U1", "PS1")]},
        )
        assert _lines(_run(tmp_path, files), "relation-stub") == []


class TestLayerPermutation:
    def test_lines_do_not_depend_on_layer_order(self, tmp_path: Path) -> None:
        layers = [
            _layer_devices("A1", "A2"),
            _layer_devices("U1", "U2", cls="Unidentified"),
            _layer_devices("B1"),
        ]
        claims = [
            _claim("s1", "same_as", "A1", "B1"),
            _claim("s2", "same_as", "B1", "A2"),
            _claim("r1", "part_of", "U1", "A1"),
            _claim("r2", "part_of", "A1", "U1"),
            _claim("r3", "runs_on", "U2", "B1"),
        ]
        seen: list[list[str]] = []
        for index, order in enumerate(itertools.permutations(range(3))):
            files = _tree([], **{"records/relations.yaml": claims})
            for layer, which in zip(("a", "b", "c"), order, strict=True):
                files[f"imported/{layer}/devices.yaml"] = layers[which]
            report = _run(tmp_path / str(index), files)
            seen.append(
                sorted(
                    line.split(" — ", 1)[1]
                    for kind in ("identity-ambiguous", "part-cycle", "relation-stub")
                    for line in _lines(report, kind)
                )
            )
        assert seen[0]
        assert all(other == seen[0] for other in seen)


#: The file the relations importer always copies into its layer, marking it a relations layer.
_MARK = "relations.csv"

_DISAGREE_FIX = "place it under one parent, or give each place its own name"
_UNPLACED_FIX = "give it a parent or a citation in the layer's places table"


def _place_tree(**extra: Any) -> dict[str, Any]:
    """Authored SR with two sectors; *extra* adds layer files."""
    files: dict[str, Any] = {
        "records/places.yaml": [{"id": "SR"}, {"id": "SR/S01"}, {"id": "SR/S02"}],
        "documents.yaml": _DOCS,
    }
    files.update(extra)
    return files


def _relations_layer(layer: str, places: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        f"imported/{layer}/places.yaml": places,
        f"imported/{layer}/{_MARK}": "subject,type,object\n",
    }


def _place_lines(report: StageReport) -> list[str]:
    return [*_lines(report, "place-parents-disagree"), *_lines(report, "place-unplaced")]


class TestPlaceParentsDisagree:
    def test_one_layer_placing_one_name_under_two_parents(self, tmp_path: Path) -> None:
        files = _place_tree(
            **_relations_layer("cab", [{"id": "SR/S01/RACK"}, {"id": "SR/S02/RACK"}])
        )
        assert _lines(_run(tmp_path, files), "place-parents-disagree") == [
            "facility: place-parents-disagree: place RACK — relations layers create it under "
            f"2 parents: SR/S01/RACK, SR/S02/RACK; fix: {_DISAGREE_FIX}",
        ]

    def test_two_layers_placing_one_name_under_two_parents(self, tmp_path: Path) -> None:
        files = _place_tree(
            **_relations_layer("cab", [{"id": "SR/S01/RACK"}]),
            **_relations_layer("net", [{"id": "SR/S02/RACK"}]),
        )
        assert _lines(_run(tmp_path, files), "place-parents-disagree") == [
            "facility: place-parents-disagree: place RACK — relations layers create it under "
            f"2 parents: SR/S01/RACK, SR/S02/RACK; fix: {_DISAGREE_FIX}",
        ]

    def test_two_layers_on_one_parent_agree(self, tmp_path: Path) -> None:
        files = _place_tree(
            **_relations_layer("cab", [{"id": "SR/S01/RACK"}]),
            **_relations_layer("net", [{"id": "SR/S01/RACK"}]),
        )
        assert _place_lines(_run(tmp_path, files)) == []

    def test_a_top_level_place_disagrees_with_nothing(self, tmp_path: Path) -> None:
        files = _place_tree(
            **_relations_layer("cab", [{"id": "SR/S01/RACK"}]),
            **_relations_layer("net", [{"id": "RACK", "citations": [{"document": "dwg-1"}]}]),
        )
        assert _place_lines(_run(tmp_path, files)) == []

    def test_repeated_authored_and_mml_leaf_names_raise_nothing(self, tmp_path: Path) -> None:
        files = _place_tree(
            **{
                "records/places.yaml": [
                    {"id": "SR"},
                    {"id": "SR/S01"},
                    {"id": "SR/S02"},
                    {"id": "SR/S01/RACK"},
                    {"id": "SR/S02/RACK"},
                ],
                "imported/mml/places.yaml": [{"id": "SR/S01/BPM"}, {"id": "SR/S02/BPM"}],
            }
        )
        assert _place_lines(_run(tmp_path, files)) == []

    def test_a_place_another_format_also_states_is_not_the_relations_layers(
        self, tmp_path: Path
    ) -> None:
        files = _place_tree(
            **_relations_layer("cab", [{"id": "SR/S01/RACK"}, {"id": "SR/S02/RACK"}]),
            **{"imported/mml/places.yaml": [{"id": "SR/S02/RACK"}]},
        )
        assert _place_lines(_run(tmp_path, files)) == []


class TestPlaceUnplaced:
    def test_a_top_level_place_without_a_citation(self, tmp_path: Path) -> None:
        files = _place_tree(**_relations_layer("cab", [{"id": "HUT"}]))
        assert _lines(_run(tmp_path, files), "place-unplaced") == [
            "facility: place-unplaced: place HUT — relations layer cab creates it top-level "
            f"with no parent and no citation; fix: {_UNPLACED_FIX}",
        ]

    def test_layers_are_named_in_order(self, tmp_path: Path) -> None:
        files = _place_tree(
            **_relations_layer("net", [{"id": "HUT"}]),
            **_relations_layer("cab", [{"id": "HUT"}]),
        )
        assert _lines(_run(tmp_path, files), "place-unplaced") == [
            "facility: place-unplaced: place HUT — relations layers cab, net create it top-level "
            f"with no parent and no citation; fix: {_UNPLACED_FIX}",
        ]

    def test_a_places_only_relations_layer_still_warns(self, tmp_path: Path) -> None:
        files = _place_tree(**_relations_layer("cab", [{"id": "HUT"}]))
        report = _run(tmp_path, files)
        assert not report.validated.document.get("relations")
        assert len(_lines(report, "place-unplaced")) == 1

    def test_a_layer_with_claims_still_warns(self, tmp_path: Path) -> None:
        files = _place_tree(
            **_relations_layer("cab", [{"id": "HUT"}]),
            **{
                "records/devices.yaml": [_device("PS1", "PowerSupply"), _device("Q1")],
                "imported/cab/relations.yaml": [_claim("cab/r1", "part_of", "Q1", "PS1")],
            },
        )
        assert len(_lines(_run(tmp_path, files), "place-unplaced")) == 1

    def test_a_cited_top_level_place_is_placed_by_its_source(self, tmp_path: Path) -> None:
        files = _place_tree(
            **_relations_layer("cab", [{"id": "HUT", "citations": [{"document": "dwg-1"}]}])
        )
        assert _place_lines(_run(tmp_path, files)) == []

    def test_a_parented_place_is_placed(self, tmp_path: Path) -> None:
        files = _place_tree(**_relations_layer("cab", [{"id": "SR/S01/RACK"}]))
        assert _place_lines(_run(tmp_path, files)) == []

    def test_an_authored_root_raises_nothing(self, tmp_path: Path) -> None:
        files = _place_tree(**{"records/places.yaml": [{"id": "SR"}, {"id": "HUT"}]})
        assert _place_lines(_run(tmp_path, files)) == []

    def test_an_authored_place_a_relations_layer_restates_is_not_created_by_it(
        self, tmp_path: Path
    ) -> None:
        files = _place_tree(**_relations_layer("cab", [{"id": "SR"}]))
        assert _place_lines(_run(tmp_path, files)) == []

    def test_another_formats_top_level_place_raises_nothing(self, tmp_path: Path) -> None:
        files = _place_tree(**{"imported/mml/places.yaml": [{"id": "HUT"}]})
        assert _place_lines(_run(tmp_path, files)) == []


def test_the_bundled_example_raises_no_place_warning() -> None:
    example = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example"
    report = run_stages(example, project_name="demo", later=LATER_STAGES)
    assert report.failed is None, [e.format_message() for e in report.errors]
    assert _place_lines(report) == []


def test_the_relations_layers_are_the_marked_ones(tmp_path: Path) -> None:
    from osprey.facility.sources import load_sources

    files = _place_tree(
        **_relations_layer("cab", [{"id": "HUT"}]),
        **{"imported/mml/places.yaml": [{"id": "BOX"}]},
    )
    result = load_sources(_write(tmp_path, files))
    assert result.errors == []
    assert result.sources.relations_layers == frozenset({"cab"})


@pytest.mark.parametrize(
    "kind",
    [
        "alias-shared",
        "identity-ambiguous",
        "part-cycle",
        "relation-stub",
        "place-parents-disagree",
        "place-unplaced",
    ],
)
def test_kinds_are_registered(kind: str) -> None:
    from osprey.facility.errors import WARNING_KINDS

    assert kind in WARNING_KINDS
