"""The compute stage projects a device's place from its ``located_in`` claims.

A device no source and no span places takes the place its ``located_in`` key's
accepted, non-coarse claims all name, recorded ``place_from: relation``, when no
claim on the key is contested. A place stated by any source or set by a span
always wins, and a projected place is numbered like any other.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.build import LATER_STAGES
from osprey.facility.validate import StageReport, run_stages
from tests.facility._appendix_a import VARIANTS, build_appendix_a
from tests.facility._synthetic_trees import deck_tree, write_tree

_POWER_SUPPLY = "PowerSupply"


def _located(rid: str, device: str, place: str, **object_extra: Any) -> dict[str, Any]:
    return {
        "id": rid,
        "type": "located_in",
        "subject": {"device": device},
        "object": {"place": place, **object_extra},
    }


def _tree(*claims: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """The deck tree with a hall of two racks and three unplaced supplies."""
    tree = deck_tree()
    tree["records/places.yaml"] += [{"id": "HALL"}, {"id": "HALL/RK1"}, {"id": "HALL/RK2"}]
    tree["records/devices.yaml"] += [
        {"id": "PS1", "class": _POWER_SUPPLY},
        {"id": "PS2", "class": _POWER_SUPPLY},
        {"id": "PS3", "class": _POWER_SUPPLY},
    ]
    if claims:
        tree["records/relations.yaml"] = list(claims)
    tree.update(extra)
    return tree


def _run(root: Path, tree: dict[str, Any]) -> StageReport:
    report = run_stages(write_tree(root / "facility", tree), project_name="p", later=LATER_STAGES)
    assert report.failed is None, [e.format_message() for e in report.errors]
    return report


def _devices(report: StageReport) -> dict[str, dict[str, Any]]:
    document = report.validated.document
    assert document is not None
    return {device["id"]: device for device in document["devices"]}


def _place(device: dict[str, Any]) -> tuple[str | None, str | None]:
    return device.get("place"), device.get("provenance", {}).get("place_from")


def _lines(report: StageReport) -> list[str]:
    return [warning.line for warning in report.validated.warnings]


class TestProjects:
    def test_one_claim_projects(self, tmp_path: Path) -> None:
        devices = _devices(_run(tmp_path, _tree(_located("ps1-rk1", "PS1", "HALL/RK1"))))
        assert _place(devices["PS1"]) == ("HALL/RK1", "relation")
        assert _place(devices["PS2"]) == (None, None)

    def test_the_unit_is_not_copied(self, tmp_path: Path) -> None:
        devices = _devices(_run(tmp_path, _tree(_located("ps1-rk1", "PS1", "HALL/RK1", unit=22))))
        assert _place(devices["PS1"]) == ("HALL/RK1", "relation")
        assert "unit" not in devices["PS1"]

    def test_the_same_fact_in_two_layers_projects(self, tmp_path: Path) -> None:
        tree = _tree(
            **{
                "imported/mml/relations.yaml": [_located("ps1-mml", "PS1", "HALL/RK1", unit=4)],
                "imported/csv/relations.yaml": [_located("ps1-csv", "PS1", "HALL/RK1", unit=4)],
            }
        )
        report = _run(tmp_path, tree)
        assert _place(_devices(report)["PS1"]) == ("HALL/RK1", "relation")
        assert not [line for line in _lines(report) if "relation-contested" in line]

    def test_a_coarse_claim_beside_a_finer_one_on_one_rack_projects(self, tmp_path: Path) -> None:
        report = _run(
            tmp_path,
            _tree(
                _located("ps1-fine", "PS1", "HALL/RK1", unit=22),
                _located("ps1-coarse", "PS1", "HALL/RK1"),
            ),
        )
        assert _place(_devices(report)["PS1"]) == ("HALL/RK1", "relation")
        assert not [line for line in _lines(report) if "relation-contested" in line]

    def test_an_inferred_claim_projects(self, tmp_path: Path) -> None:
        claim = {**_located("ps1-rk1", "PS1", "HALL/RK1"), "basis": "inferred", "rule": "same-rack"}
        devices = _devices(_run(tmp_path, _tree(claim)))
        assert _place(devices["PS1"]) == ("HALL/RK1", "relation")

    def test_a_superseded_claim_does_not_block_the_accepted_one(self, tmp_path: Path) -> None:
        tree = _tree(
            {**_located("ps1-spec", "PS1", "HALL/RK1"), "source": {"document": "spec-1"}},
            {**_located("ps1-dwg", "PS1", "HALL/RK2"), "source": {"document": "dwg-1"}},
            **{
                "documents.yaml": [
                    {"id": "spec-1", "title": "Rack layout", "kind": "spec"},
                    {"id": "dwg-1", "title": "Cable schedule", "kind": "drawing"},
                ],
                "policy.yaml": {
                    "schema": "osprey.facility.policy/1",
                    "relations": {"precedence": {"located_in": ["spec", "drawing"]}},
                },
            },
        )
        assert _place(_devices(_run(tmp_path, tree))["PS1"]) == ("HALL/RK1", "relation")


class TestDisputes:
    def test_contested_racks_project_nothing_and_warn(self, tmp_path: Path) -> None:
        report = _run(
            tmp_path,
            _tree(_located("ps1-a", "PS1", "HALL/RK1"), _located("ps1-b", "PS1", "HALL/RK2")),
        )
        assert _place(_devices(report)["PS1"]) == (None, None)
        contested = [line for line in _lines(report) if "relation-contested" in line]
        assert len(contested) == 1 and "device PS1" in contested[0]

    def test_a_unit_only_disagreement_projects_nothing(self, tmp_path: Path) -> None:
        report = _run(
            tmp_path,
            _tree(
                _located("ps1-a", "PS1", "HALL/RK1", unit=22),
                _located("ps1-b", "PS1", "HALL/RK1", unit=23),
            ),
        )
        assert _place(_devices(report)["PS1"]) == (None, None)
        assert [line for line in _lines(report) if "relation-contested" in line]

    def test_a_coarse_claim_beside_a_contested_pair_projects_nothing(self, tmp_path: Path) -> None:
        report = _run(
            tmp_path,
            _tree(
                _located("ps1-a", "PS1", "HALL/RK1", unit=22),
                _located("ps1-b", "PS1", "HALL/RK1", unit=23),
                _located("ps1-coarse", "PS1", "HALL/RK1"),
            ),
        )
        assert _place(_devices(report)["PS1"]) == (None, None)

    def test_a_dispute_on_one_device_leaves_another_projected(self, tmp_path: Path) -> None:
        devices = _devices(
            _run(
                tmp_path,
                _tree(
                    _located("ps1-a", "PS1", "HALL/RK1"),
                    _located("ps1-b", "PS1", "HALL/RK2"),
                    _located("ps2-a", "PS2", "HALL/RK2"),
                ),
            )
        )
        assert _place(devices["PS1"]) == (None, None)
        assert _place(devices["PS2"]) == ("HALL/RK2", "relation")


class TestStatedPlacesWin:
    def test_an_authored_place_wins(self, tmp_path: Path) -> None:
        tree = _tree(_located("ps1-rk1", "PS1", "HALL/RK1"))
        [ps1] = [d for d in tree["records/devices.yaml"] if d["id"] == "PS1"]
        ps1["place"] = "HALL/RK2"
        assert _place(_devices(_run(tmp_path, tree))["PS1"]) == ("HALL/RK2", "authored")

    def test_an_imported_place_wins(self, tmp_path: Path) -> None:
        tree = _tree(
            _located("ps1-rk1", "PS1", "HALL/RK1"),
            **{"imported/mml/devices.yaml": [{"id": "PS1", "place": "HALL/RK2"}]},
        )
        assert _place(_devices(_run(tmp_path, tree))["PS1"]) == ("HALL/RK2", "imported")

    def test_a_fix_place_wins(self, tmp_path: Path) -> None:
        tree = _tree(
            _located("ps1-rk1", "PS1", "HALL/RK1"),
            **{
                "imported/mml/devices.yaml": [{"id": "PS1", "place": "HALL/RK2"}],
                "fixes.yaml": {
                    "schema": "osprey.facility.fixes/1",
                    "fixes": [
                        {
                            "op": "set",
                            "kind": "device",
                            "id": "PS1",
                            "fields": {"place": "HALL/RK2"},
                            "was": {"place": {"mml": "HALL/RK2"}},
                            "why": "The supply sits in the second rack.",
                        }
                    ],
                },
            },
        )
        assert _place(_devices(_run(tmp_path, tree))["PS1"]) == ("HALL/RK2", "fix")

    def test_a_span_place_wins(self, tmp_path: Path) -> None:
        devices = _devices(_run(tmp_path, _tree(_located("qd-rk1", "SR/QD", "HALL/RK1"))))
        assert _place(devices["SR/QD"]) == ("SR", "span")


class TestNumbering:
    def test_projected_devices_are_numbered_in_their_place(self, tmp_path: Path) -> None:
        tree = _tree(
            _located("ps1-rk1", "PS1", "HALL/RK1"),
            _located("ps2-rk1", "PS2", "HALL/RK1"),
            _located("ps3-rk2", "PS3", "HALL/RK2"),
        )
        for device in tree["records/devices.yaml"]:
            if device["id"] in ("PS1", "PS2", "PS3"):
                device["s"] = {"PS1": 2.0, "PS2": 1.0, "PS3": 1.0}[device["id"]]
        devices = _devices(_run(tmp_path, tree))
        assert [devices[rid].get("ordinalInPlace") for rid in ("PS1", "PS2", "PS3")] == [2, 1, 1]

    def test_projected_devices_join_the_place_count_without_s(self, tmp_path: Path) -> None:
        devices = _devices(_run(tmp_path, _tree(_located("ps1-rk1", "PS1", "HALL/RK1"))))
        assert devices["PS1"]["place"] == "HALL/RK1"
        assert "ordinalInPlace" not in devices["PS1"]


class TestOrderIndependence:
    _LAYERS = (
        [_located("ps1-a", "PS1", "HALL/RK1", unit=3)],
        [_located("ps1-b", "PS1", "HALL/RK1"), _located("ps2-a", "PS2", "HALL/RK1")],
        [_located("ps2-b", "PS2", "HALL/RK2"), _located("ps3-a", "PS3", "HALL/RK2", unit=7)],
    )

    def test_the_result_does_not_depend_on_layer_order(self, tmp_path: Path) -> None:
        results = set()
        for n, order in enumerate(itertools.permutations(range(3))):
            files = {
                f"imported/{layer}/relations.yaml": self._LAYERS[index]
                for layer, index in zip(("a", "b", "c"), order, strict=True)
            }
            devices = _devices(_run(tmp_path / str(n), _tree(**files)))
            results.add(tuple(_place(devices[rid]) for rid in ("PS1", "PS2", "PS3")))
        assert results == {
            (("HALL/RK1", "relation"), (None, None), ("HALL/RK2", "relation")),
        }


@pytest.fixture(scope="module")
def appendix_a(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    return {
        variant: {
            device["id"]: device
            for device in build_appendix_a(tmp_path_factory.mktemp(variant) / "facility", variant)[
                "devices"
            ]
        }
        for variant in VARIANTS
    }


class TestAppendixA:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_the_readout_electronics_go_to_rk08(
        self, appendix_a: dict[str, dict[str, Any]], variant: str
    ) -> None:
        assert _place(appendix_a[variant]["SR/BPM13-EL"]) == ("HALL/ROW3/RK08", "relation")

    def test_the_supply_goes_to_rk07_under_the_precedence_policy(
        self, appendix_a: dict[str, dict[str, Any]]
    ) -> None:
        devices = appendix_a["located_in_precedence"]
        assert _place(devices["SR/QF05-PS"]) == ("HALL/ROW3/RK07", "relation")

    def test_the_supply_has_no_place_without_a_policy(
        self, appendix_a: dict[str, dict[str, Any]]
    ) -> None:
        assert _place(appendix_a["no_policy"]["SR/QF05-PS"]) == (None, None)
