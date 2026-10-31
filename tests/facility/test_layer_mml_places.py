"""The mml layer placing each imported device from its export's ``DeviceList``.

The rule cases call :func:`place_devices` on hand-built slots, one fixture
per shape the rule distinguishes; the mapping cases parse and check the
``models.<M>.places`` block; the import cases run the importer over the
nsls2 tree with place answers added to its mapping, and over spear3 without.
"""

from __future__ import annotations

import contextlib
import io
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.layers.mml.importer import LAYER_DIR, MappingProblems, import_mml
from osprey.facility.layers.mml.mapping import (
    MAPPING_FILE,
    MappingError,
    Model,
    PlaceRules,
    SectorList,
    check_mapping,
    parse_mapping,
)
from osprey.facility.layers.mml.places import Slot, place_devices
from osprey.facility.validate import run_stages

pytestmark = pytest.mark.xdist_group("layer_mml_import")

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mml"

SR_SECTORS = SectorList(count=12, token="SR{sector:02d}")
BR_SECTORS = SectorList(count=4, token="BR{sector}")


def _model(raw: str, places: PlaceRules | None) -> Model:
    return Model(raw=raw, name=raw, description=None, provenance="stated", places=places)


def _slots(system: str, family: str, rows: list[tuple[str, tuple[int, ...] | None]]) -> list[Slot]:
    return [Slot(system, family, device, row) for device, row in rows]


# --- the rule ------------------------------------------------------------------


class TestPlaceTree:
    def test_a_sectored_model_is_a_place_with_every_sector_under_it(self):
        placed = place_devices({"SR": _model("SR", PlaceRules(sectors=SR_SECTORS))}, [], {})
        assert [p["id"] for p in placed.places] == ["SR", *(f"SR/SR{k:02d}" for k in range(1, 13))]
        assert placed.devices == {}

    def test_a_model_without_sectors_is_one_place(self):
        placed = place_devices(
            {"LTB": _model("LTB", PlaceRules())}, _slots("LTB", "Q", [("LTB/Q1", (3, 1))]), {}
        )
        assert placed.places == [{"id": "LTB"}]
        assert placed.devices == {"LTB/Q1": "LTB"}

    def test_a_model_without_place_rules_places_nothing(self):
        placed = place_devices(
            {"SR": _model("SR", None)}, _slots("SR", "QF", [("SR/QF_1", (1, 1))]), {}
        )
        assert placed.places == []
        assert placed.devices == {}
        assert placed.problems == []


class TestRule:
    def test_series_supply_magnets_are_each_placed_by_their_own_row(self):
        slots = _slots("SR", "SF", [(f"SR/SF_{k}", (k, 1)) for k in range(1, 13)])
        # One sector-1 supply address feeds every magnet: it is no device's alone.
        placed = place_devices({"SR": _model("SR", PlaceRules(sectors=SR_SECTORS))}, slots, {})
        assert placed.problems == []
        assert placed.devices == {f"SR/SF_{k}": f"SR/SR{k:02d}" for k in range(1, 13)}

    def test_a_model_wide_singleton_on_a_lone_first_row_sits_on_the_model(self):
        rules = PlaceRules(sectors=SR_SECTORS, model_wide=("RF",))
        placed = place_devices(
            {"SR": _model("SR", rules)},
            _slots("SR", "RF", [("SR/RF", (1, 1))]),
            {"SR/RF": ["SR01C___RF_____AC00"]},
        )
        assert placed.problems == []
        assert placed.devices == {"SR/RF": "SR"}

    def test_a_singleton_without_an_answer_takes_its_rows_sector(self):
        placed = place_devices(
            {"SR": _model("SR", PlaceRules(sectors=SR_SECTORS))},
            _slots("SR", "TUNE", [("SR/TUNE", (1, 1))]),
            {},
        )
        assert placed.devices == {"SR/TUNE": "SR/SR01"}

    def test_a_sector_override_places_a_singleton_whose_row_is_an_index(self):
        rules = PlaceRules(sectors=BR_SECTORS, sector_override={"ExtractionKicker": 2})
        placed = place_devices(
            {"BR": _model("BR", rules)},
            _slots("BR", "ExtractionKicker", [("BR/EK", (1, 2))]),
            {"BR/EK": ["BR2_____KE_____AC00", "BR2_____KE_____AM00"]},
        )
        assert placed.problems == []
        assert placed.devices == {"BR/EK": "BR/BR2"}

    def test_model_wide_outranks_a_sector_override(self):
        rules = PlaceRules(sectors=BR_SECTORS, model_wide=("A",), sector_override={"B": 2})
        slots = [Slot("BR", "A", "BR/X", (1, 1)), Slot("BR", "B", "BR/X", (1, 1))]
        assert place_devices({"BR": _model("BR", rules)}, slots, {}).devices == {"BR/X": "BR"}

    def test_a_device_whose_rows_span_sectors_is_model_wide(self):
        slots = [Slot("SR", "BEND", "SR/B", (1, 1)), Slot("SR", "BEND", "SR/B", (2, 1))]
        placed = place_devices({"SR": _model("SR", PlaceRules(sectors=SR_SECTORS))}, slots, {})
        assert placed.devices == {"SR/B": "SR"}

    def test_a_device_with_no_row_sits_on_its_model(self):
        placed = place_devices(
            {"SR": _model("SR", PlaceRules(sectors=SR_SECTORS))},
            _slots("SR", "DCCT", [("SR/DCCT", None)]),
            {"SR/DCCT": ["SR05W___DCCT2__AM01"]},
        )
        assert placed.problems == []
        assert placed.devices == {"SR/DCCT": "SR"}

    def test_a_family_sectored_on_one_model_is_model_wide_on_another(self):
        models = {
            "SR": _model("SR", PlaceRules(sectors=SR_SECTORS)),
            "BR": _model("BR", PlaceRules(sectors=BR_SECTORS, model_wide=("BEND",))),
        }
        slots = [Slot("SR", "BEND", "SR/BEND_3", (3, 1)), Slot("BR", "BEND", "BR/BEND_3", (3, 1))]
        placed = place_devices(models, slots, {})
        assert placed.devices == {"SR/BEND_3": "SR/SR03", "BR/BEND_3": "BR"}


class TestRefusals:
    def test_a_row_sector_beyond_the_list_is_refused(self):
        placed = place_devices(
            {"BR": _model("BR", PlaceRules(sectors=BR_SECTORS))},
            _slots("BR", "GammaRad", [("BR/GammaRad_5", (5, 1))]),
            {},
        )
        assert [str(p) for p in placed.problems] == [
            "models.BR.places.sectors: BR/GammaRad_5 has DeviceList sector 5 beyond BR's 4; "
            "answer it with sector_override or model_wide"
        ]

    def test_a_model_wide_family_beyond_the_list_is_answered(self):
        rules = PlaceRules(sectors=BR_SECTORS, model_wide=("GammaRad",))
        placed = place_devices(
            {"BR": _model("BR", rules)},
            _slots("BR", "GammaRad", [("BR/GammaRad_5", (5, 1))]),
            {},
        )
        assert placed.problems == []
        assert placed.devices == {"BR/GammaRad_5": "BR"}

    def test_an_address_of_another_sector_is_refused(self):
        placed = place_devices(
            {"BR": _model("BR", PlaceRules(sectors=BR_SECTORS))},
            _slots("BR", "ExtractionKicker", [("BR/EK", (1, 2))]),
            {"BR/EK": ["BR2_____KE_____AC00"]},
        )
        assert [str(p) for p in placed.problems] == [
            "models.BR.places: BR/EK is in DeviceList sector 1 (BR1) but is bound alone to "
            "BR2_____KE_____AC00, an address of BR2; answer it with sector_override or model_wide"
        ]

    def test_a_straight_section_address_of_its_own_sector_agrees(self):
        placed = place_devices(
            {"SR": _model("SR", PlaceRules(sectors=SR_SECTORS))},
            _slots("SR", "SCRAPER", [("SR/SCRAPER", (3, 1))]),
            {"SR/SCRAPER": ["SR03S___SCRAPB_AC02"]},
        )
        assert placed.problems == []
        assert placed.devices == {"SR/SCRAPER": "SR/SR03"}

    def test_a_straight_section_address_of_another_sector_is_refused(self):
        placed = place_devices(
            {"SR": _model("SR", PlaceRules(sectors=SR_SECTORS))},
            _slots("SR", "SCRAPER", [("SR/SCRAPER", (4, 1))]),
            {"SR/SCRAPER": ["SR03S___SCRAPB_AC02"]},
        )
        assert [p.key for p in placed.problems] == ["models.SR.places"]

    def test_the_longest_token_names_the_address_sector(self):
        sectors = SectorList(count=12, token="S{sector}")
        placed = place_devices(
            {"SR": _model("SR", PlaceRules(sectors=sectors))},
            _slots("SR", "Q", [("SR/Q", (12, 1))]),
            {"SR/Q": ["S12_Q"]},
        )
        assert placed.problems == []
        assert placed.devices == {"SR/Q": "SR/S12"}

    def test_two_overrides_disagreeing_are_refused(self):
        rules = PlaceRules(sectors=BR_SECTORS, sector_override={"A": 1, "B": 2})
        slots = [Slot("BR", "A", "BR/X", (1, 1)), Slot("BR", "B", "BR/X", (1, 1))]
        placed = place_devices({"BR": _model("BR", rules)}, slots, {})
        assert [str(p) for p in placed.problems] == [
            "models.BR.places.sector_override: BR/X's families answer sectors 1, 2"
        ]


# --- the mapping -----------------------------------------------------------------


def _family() -> dict[str, Any]:
    return {"aliases": [], "description": "x", "provenance": "stated", "channels": 1, "fields": {}}


def _document(places: Any) -> dict[str, Any]:
    return {
        "models": {
            "BR": {"name": "BR", "description": None, "provenance": "stated", "places": places}
        },
        "section_order": ["BR"],
        "families": {"RF": _family(), "EK": _family()},
        "directions": {},
    }


class TestMapping:
    def test_the_block_parses(self):
        mapping = parse_mapping(
            _document(
                {
                    "sectors": {"count": 4, "token": "BR{sector}"},
                    "model_wide": ["RF"],
                    "sector_override": {"EK": 2},
                }
            )
        )
        assert mapping.models["BR"].places == PlaceRules(
            sectors=BR_SECTORS, model_wide=("RF",), sector_override={"EK": 2}
        )
        assert check_mapping(mapping) == []

    def test_a_model_without_the_block_has_no_rules(self):
        document = _document({})
        del document["models"]["BR"]["places"]
        assert parse_mapping(document).models["BR"].places is None

    @pytest.mark.parametrize(
        ("places", "key", "message"),
        [
            ({"sector": {}}, "models.BR.places.sector", "unknown key"),
            (
                {"sectors": {"count": 0, "token": "BR{sector}"}},
                "models.BR.places.sectors.count",
                "must be a positive integer, got 0",
            ),
            (
                {"sectors": {"count": 4, "token": "BR"}},
                "models.BR.places.sectors.token",
                "must name the sector as {sector}, got 'BR'",
            ),
            (
                {"sectors": {"count": 4, "token": "BR{sector:z}"}},
                "models.BR.places.sectors.token",
                None,
            ),
            (
                {"sectors": {"count": 4, "token": "BR{sector:.0}"}},
                "models.BR.places.sectors.token",
                None,
            ),
            ({"model_wide": ["RF"]}, "models.BR.places.model_wide", "needs the model's sectors"),
            (
                {"sectors": {"count": 4, "token": "BR{sector}"}, "sector_override": {"EK": "2"}},
                "models.BR.places.sector_override.EK",
                "must be a sector number, got str",
            ),
        ],
    )
    def test_a_wrong_shape_is_refused(self, places, key, message):
        with pytest.raises(MappingError) as caught:
            parse_mapping(_document(places))
        assert caught.value.key == key
        if message is not None:
            assert caught.value.message == message

    def test_answers_must_name_families_and_sectors_the_model_has(self):
        mapping = parse_mapping(
            _document(
                {
                    "sectors": {"count": 4, "token": "BR{sector}"},
                    "model_wide": ["NOPE", "RF"],
                    "sector_override": {"EK": 5, "RF": 1},
                }
            )
        )
        assert [str(p) for p in check_mapping(mapping)] == [
            "models.BR.places.model_wide[0]: NOPE is no family",
            "models.BR.places.sector_override.EK: sector 5 is not one of BR's 4",
            "models.BR.places.sector_override.RF: RF is also model_wide; answer it once",
        ]

    def test_a_sector_name_must_be_pn_local(self):
        mapping = parse_mapping(_document({"sectors": {"count": 4, "token": "BR {sector}"}}))
        assert [p.key for p in check_mapping(mapping)] == ["models.BR.places.sectors.token"]


# --- the import ------------------------------------------------------------------

NSLS2_STEMS = ("nsls2.storagering", "nsls2.ltb")


def _nsls2(tmp_path: Path, places: dict[str, Any]) -> Path:
    facility = tmp_path / "data" / "facility"
    target = facility / MAPPING_FILE
    target.parent.mkdir(parents=True)
    document = yaml.safe_load((FIXTURES / "nsls2" / MAPPING_FILE).read_text(encoding="utf-8"))
    for raw, rules in places.items():
        document["models"][raw]["places"] = rules
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return facility


def _import(facility: Path, tree: str = "nsls2", stems: tuple[str, ...] = NSLS2_STEMS) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        import_mml([FIXTURES / tree / f"{stem}.ao.json" for stem in stems], facility)


def _read(facility: Path, name: str) -> Any:
    return yaml.safe_load((facility / LAYER_DIR / name).read_text(encoding="utf-8"))


NSLS2_PLACES = {
    "StorageRing": {
        "sectors": {"count": 30, "token": "C{sector:02d}"},
        "model_wide": ["DCCT", "RF", "TUNE"],
    },
    "LTB": {"sectors": {"count": 1, "token": "LTB{sector}"}, "model_wide": ["BPMx", "BPMy"]},
}


@pytest.fixture(scope="module")
def nsls2_placed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    facility = _nsls2(tmp_path_factory.mktemp("nsls2-places"), NSLS2_PLACES)
    _import(facility)
    return facility


def test_the_import_writes_the_place_tree_from_the_mapping(nsls2_placed: Path) -> None:
    ids = [place["id"] for place in _read(nsls2_placed, "places.yaml")]
    assert ids == sorted(
        ["LTB", "LTB/LTB1", "StorageRing", *(f"StorageRing/C{k:02d}" for k in range(1, 31))]
    )


def test_every_device_of_a_placed_model_gets_a_place(nsls2_placed: Path) -> None:
    devices = _read(nsls2_placed, "devices.yaml")
    assert devices
    assert all("place" in device for device in devices)


def test_the_import_places_by_row_and_by_answer(nsls2_placed: Path) -> None:
    rows = _read_rows(nsls2_placed)
    devices = {device["id"]: device for device in _read(nsls2_placed, "devices.yaml")}
    for row in rows:
        device = devices[row["device"]]
        if row["model"] == "StorageRing" and row["family"] == "BPMx":
            assert device["place"] == f"StorageRing/C{row['device_list'][0]:02d}"
        if row["model"] == "LTB" and row["family"] == "BPMx":
            assert device["place"] == "LTB"
        if row["family"] in {"DCCT", "RF", "TUNE"}:
            assert device["place"] == "StorageRing"


def _read_rows(facility: Path) -> list[dict[str, Any]]:
    import json

    return json.loads((facility / LAYER_DIR / "rows.json").read_text(encoding="utf-8"))["rows"]


def test_placed_records_pass_the_stages(nsls2_placed: Path) -> None:
    report = run_stages(nsls2_placed, project_name="demo")
    assert report.ok, [str(error) for error in report.errors]


def test_a_placement_the_mapping_leaves_open_stops_before_writing(tmp_path: Path) -> None:
    places = {"StorageRing": {"sectors": {"count": 29, "token": "C{sector:02d}"}}}
    facility = _nsls2(tmp_path, places)
    with pytest.raises(MappingProblems) as caught:
        _import(facility)
    assert any("beyond StorageRing's 29" in str(p) for p in caught.value.problems)
    assert not (facility / LAYER_DIR / "devices.yaml").exists()


def test_a_mapping_without_place_rules_writes_no_places(tmp_path: Path) -> None:
    facility = tmp_path / "data" / "facility"
    (facility / LAYER_DIR).mkdir(parents=True)
    shutil.copyfile(FIXTURES / "spear3" / MAPPING_FILE, facility / MAPPING_FILE)
    (facility / LAYER_DIR / "places.yaml").write_text("[]\n", encoding="utf-8")
    _import(facility, "spear3", ("spear3.storagering",))
    assert not (facility / LAYER_DIR / "places.yaml").exists()
    assert not [d for d in _read(facility, "devices.yaml") if "place" in d]
