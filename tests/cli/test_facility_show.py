"""``osprey facility show``: the facility the repo builds, or one of its records.

The verb builds in memory as ``osprey facility validate`` does and lists what
the build holds: the identity, the records per kind, the models and every view
of the main render. With an ID it prints that record with its provenance and
the fixes applied to it. Under ``--json`` stdout holds one document whose keys
are pinned by the goldens in ``tests/cli/data/json_keysets/``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.main import cli
from tests._builds import init_project
from tests.cli.test_json_keyset_capture import _assert_matches_golden
from tests.facility._appendix_a import CLAIM_IDS

pytestmark = pytest.mark.slow

#: A demo device the authored records place in a sector by span.
DEVICE = "SR/BPM01"

#: A demo channel with a ``unit`` its authored layer states as ``mm``.
CHANNEL = "BR:DIAG:BPM:01:POSITION:X"

#: The channel-finder views, one per ``channel_finder_mode``.
CHANNEL_FINDER_VIEWS = ("in_context", "hierarchical", "middle_layer")


@pytest.fixture(scope="module")
def initialised(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A demo repo, initialised once for this module and never edited."""
    return init_project(tmp_path_factory.mktemp("ca"), "demo", "demo")


@pytest.fixture
def repo(initialised: Path, tmp_path: Path) -> Iterator[Path]:
    """A copy of the initialised repo this test may edit."""
    copy = tmp_path / initialised.name
    shutil.copytree(initialised, copy, symlinks=True)
    yield copy


def _show(repo: Path, *args: str) -> Result:
    return CliRunner().invoke(cli, ["facility", "show", "--repo", str(repo), *args])


def _set_mode(path: Path, mode: str) -> None:
    """Set ``channel_finder_mode`` in a profile or layer."""
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.startswith("channel_finder_mode:")]
    path.write_text("\n".join([*lines, f"channel_finder_mode: {mode}", ""]), encoding="utf-8")


def _append(path: Path, *records: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(yaml.safe_dump(list(records), sort_keys=False))


# --- the facility ------------------------------------------------------------------


def test_the_document_matches_its_golden(repo: Path) -> None:
    result = _show(repo, "--json")

    assert result.exit_code == 0, result.output
    _assert_matches_golden(json.loads(result.stdout), "facility_show")


def test_a_view_carries_a_reason_exactly_when_it_is_not_written(repo: Path) -> None:
    views = json.loads(_show(repo, "--json").stdout)["views"]

    omitted = [view for view in views if not view["written"]]
    assert omitted, "the demo render omits no view"
    for view in views:
        assert ("reason" in view) is (not view["written"]), view
        assert set(view) == {"name", "path", "written"} | (
            {"reason"} if "reason" in view else set()
        )
    assert all(view["reason"] for view in omitted)


def test_the_demo_lists_its_records_models_and_views(repo: Path) -> None:
    document = json.loads(_show(repo, "--json").stdout)

    assert document["identity"]["name"] == "Example Research Facility"
    assert document["counts"]["devices"] > 0
    assert document["counts"]["wiring"]["SR"] > 0
    assert {"name": "SR", "engine": "pyat", "served": True, "solve": "periodic"} in document[
        "models"
    ]
    paths = {view["name"]: view["path"] for view in document["views"]}
    assert paths["simulator"] == "data/simulator"
    assert paths["graph"] == "data/graph"
    assert paths["limits"] == "data"


def test_the_human_output_names_the_config_it_judged_the_views_against(repo: Path) -> None:
    result = _show(repo)

    assert result.exit_code == 0, result.output
    assert "views of build/config.yml" in result.stdout
    assert "in_context     data/channel_finder, not written: channel_finder.pipeline_mode" in (
        result.stdout
    )
    assert "graph          data/graph, written" in result.stdout


def test_the_views_are_the_primary_configs_whatever_the_renders_select(repo: Path) -> None:
    _set_mode(repo / "profile.yml", "hierarchical")
    _set_mode(repo / "tiers" / "readonly.yml", "in_context")
    _set_mode(repo / "tiers" / "readwrite.yml", "middle_layer")

    result = _show(repo, "--json")

    assert result.exit_code == 0, result.output
    written = {view["name"]: view["written"] for view in json.loads(result.stdout)["views"]}
    assert {name: written[name] for name in CHANNEL_FINDER_VIEWS} == {
        "in_context": False,
        "hierarchical": True,
        "middle_layer": False,
    }


def test_the_repo_is_byte_unchanged(repo: Path) -> None:
    before = {
        path.relative_to(repo).as_posix(): path.read_bytes()
        for path in sorted(repo.rglob("*"))
        if path.is_file()
    }

    assert _show(repo, "--json").exit_code == 0
    assert _show(repo, DEVICE).exit_code == 0

    after = {
        path.relative_to(repo).as_posix(): path.read_bytes()
        for path in sorted(repo.rglob("*"))
        if path.is_file()
    }
    assert after == before


def test_a_build_error_exits_1_with_an_empty_stdout(repo: Path) -> None:
    fixes = {
        "schema": "osprey.facility.fixes/1",
        "fixes": [{"op": "drop", "kind": "device", "id": "SR/Q9", "why": "gone"}],
    }
    (repo / "data" / "facility" / "fixes.yaml").write_text(
        yaml.safe_dump(fixes, sort_keys=False), encoding="utf-8"
    )

    result = _show(repo, "--json")

    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert "SR/Q9" in result.stderr


# --- one record --------------------------------------------------------------------


def test_a_record_matches_its_golden(repo: Path) -> None:
    result = _show(repo, "--json", DEVICE)

    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    _assert_matches_golden(document, "facility_show_record")
    (entry,) = document["records"]
    assert entry["kind"] == "device"
    assert entry["record"]["id"] == DEVICE
    assert "provenance" not in entry["record"]
    assert entry["provenance"]["sources"][0]["file"] == "records/devices.yaml"
    assert entry["fixes_applied"] == entry["provenance"]["fixes"]


def test_a_fixed_record_lists_the_fix_applied(repo: Path) -> None:
    facility = repo / "data" / "facility"
    imported = facility / "imported" / "mml" / "channels.yaml"
    imported.parent.mkdir(parents=True)
    imported.write_text(yaml.safe_dump([{"id": CHANNEL, "unit": "mm"}]), encoding="utf-8")
    fix = {
        "op": "set",
        "kind": "channel",
        "id": CHANNEL,
        "fields": {"unit": "um"},
        "was": {"unit": {"authored": "mm", "mml": "mm"}},
        "why": "The unit is micrometres.",
    }
    (facility / "fixes.yaml").write_text(
        yaml.safe_dump({"schema": "osprey.facility.fixes/1", "fixes": [fix]}, sort_keys=False),
        encoding="utf-8",
    )

    result = _show(repo, "--json", CHANNEL)

    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    (entry,) = document["records"]
    assert entry["kind"] == "channel"
    assert entry["fixes_applied"] == [{"op": "set", "why": "The unit is micrometres."}]
    assert entry["record"]["unit"] == "um"


def test_the_human_record_prints_its_provenance(repo: Path) -> None:
    result = _show(repo, DEVICE)

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[0] == f"device {DEVICE}"
    assert "provenance" in lines
    assert "    file: records/devices.yaml" in lines
    assert lines[-2:] == ["fixes applied", "  none"]


@pytest.mark.parametrize("as_json", [False, True])
def test_an_unknown_id_exits_1(repo: Path, as_json: bool) -> None:
    result = _show(repo, *(["--json"] if as_json else []), "NO/SUCH")

    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert result.stderr.splitlines()[-1] == "facility show: no record NO/SUCH"


@pytest.mark.parametrize("as_json", [False, True])
def test_an_id_naming_a_place_and_a_device_shows_both(repo: Path, as_json: bool) -> None:
    _append(
        repo / "data" / "facility" / "records" / "devices.yaml",
        {"id": "BR", "class": "BeamPositionMonitor", "place": "BR"},
    )

    result = _show(repo, *(["--json"] if as_json else []), "BR")

    assert result.exit_code == 0, result.output
    if as_json:
        records = json.loads(result.stdout)["records"]
        assert [entry["kind"] for entry in records] == ["place", "device"]
        for entry in records:
            assert entry["record"]["id"] == "BR"
            assert "provenance" not in entry["record"]
            assert entry["fixes_applied"] == list(entry["provenance"].get("fixes") or [])
        files = [{source["file"] for source in entry["provenance"]["sources"]} for entry in records]
        assert "records/devices.yaml" not in files[0]
        assert "records/devices.yaml" in files[1]
    else:
        lines = result.stdout.splitlines()
        assert lines[0] == "place BR"
        device_at = lines.index("device BR")
        assert lines[device_at - 1] == ""
        assert lines.count("") == 1


# --- claims and name lookup --------------------------------------------------------


def _with_shared_name(repo: Path) -> None:
    """Give a place and a device of the demo's Appendix A records one shared name."""
    facility = repo / "data" / "facility"
    for relpath, id_ in (
        ("records/places.yaml", "HALL/ROW3/RK08"),
        ("records/devices.yaml", "SR/BPM13-EL"),
    ):
        path = facility / relpath
        entries = yaml.safe_load(path.read_text(encoding="utf-8"))
        for entry in entries:
            if entry["id"] == id_:
                entry["names"] = [SHARED_NAME]
        path.write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")


#: A name the rack ``HALL/ROW3/RK08`` and the device ``SR/BPM13-EL`` both carry.
SHARED_NAME = "row 3 readout rack"


def test_the_overview_counts_the_claims(repo: Path) -> None:
    """The demo ships the Appendix A claims, so its overview counts every one."""
    result = _show(repo, "--json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["counts"]["relations"] == len(CLAIM_IDS)


def test_an_id_hit_is_marked_as_an_id_match(repo: Path) -> None:
    (entry,) = json.loads(_show(repo, "--json", DEVICE).stdout)["records"]

    assert entry["match"] == "id"
    assert entry["claims"] == []


@pytest.mark.parametrize("as_json", [False, True])
def test_a_claim_id_shows_the_claim(repo: Path, as_json: bool) -> None:
    _with_shared_name(repo)

    result = _show(repo, *(["--json"] if as_json else []), "bpm13-el-rack")

    assert result.exit_code == 0, result.output
    if as_json:
        (entry,) = json.loads(result.stdout)["records"]
        assert entry["kind"] == "relation"
        assert entry["match"] == "claim"
        assert entry["record"]["id"] == "bpm13-el-rack"
        assert entry["record"]["type"] == "located_in"
        assert entry["record"]["object"] == {"place": "HALL/ROW3/RK08", "unit": 22}
        assert entry["record"]["standing"] == "accepted"
        assert "provenance" not in entry["record"]
        assert "claims" not in entry
    else:
        lines = result.stdout.splitlines()
        assert lines[0] == "relation bpm13-el-rack (match: claim)"
        assert "  type: located_in" in lines
        assert lines[-2:] == ["fixes applied", "  none"]


@pytest.mark.parametrize("as_json", [False, True])
def test_a_scheme_name_finds_the_device(repo: Path, as_json: bool) -> None:
    _with_shared_name(repo)

    result = _show(repo, *(["--json"] if as_json else []), "PS-Q-0301")

    assert result.exit_code == 0, result.output
    if as_json:
        (entry,) = json.loads(result.stdout)["records"]
        assert entry["kind"] == "device"
        assert entry["match"] == "name"
        assert entry["record"]["id"] == "SR/QF05-PS"
        claims = {claim["id"]: claim for claim in entry["claims"]}
        assert claims["qf05-ps-rack-a"]["other"] == "HALL/ROW3/RK07"
        assert claims["qf05-ps-rack-a"]["role"] == "subject"
        assert claims["qf05-ps-rack-a"]["unit"] == 12
        assert claims["qf05-ps-rack-b"]["standing"] == "contested"
        assert "unit" not in claims["qf05-ps-rack-b"]
    else:
        lines = result.stdout.splitlines()
        assert lines[0] == "device SR/QF05-PS (match: name)"
        assert "claims" in lines
        assert any(line.startswith("  qf05-ps-rack-a ") for line in lines)


@pytest.mark.parametrize("as_json", [False, True])
def test_a_shared_name_prints_both_records(repo: Path, as_json: bool) -> None:
    _with_shared_name(repo)

    result = _show(repo, *(["--json"] if as_json else []), SHARED_NAME)

    assert result.exit_code == 0, result.output
    if as_json:
        records = json.loads(result.stdout)["records"]
        assert [(entry["kind"], entry["record"]["id"], entry["match"]) for entry in records] == [
            ("place", "HALL/ROW3/RK08", "name"),
            ("device", "SR/BPM13-EL", "name"),
        ]
    else:
        lines = result.stdout.splitlines()
        assert lines[0] == "place HALL/ROW3/RK08 (match: name)"
        assert "device SR/BPM13-EL (match: name)" in lines


@pytest.mark.parametrize("as_json", [False, True])
def test_a_device_lists_its_located_in_claim_with_its_unit(repo: Path, as_json: bool) -> None:
    _with_shared_name(repo)

    result = _show(repo, *(["--json"] if as_json else []), "SR/BPM13-EL")

    assert result.exit_code == 0, result.output
    if as_json:
        (entry,) = json.loads(result.stdout)["records"]
        _assert_matches_golden({"records": [entry]}, "facility_show_record")
        located = [claim for claim in entry["claims"] if claim["type"] == "located_in"]
        assert located == [
            {
                "id": "bpm13-el-rack",
                "type": "located_in",
                "other": "HALL/ROW3/RK08",
                "standing": "accepted",
                "role": "subject",
                "unit": 22,
            }
        ]
    else:
        lines = result.stdout.splitlines()
        assert lines[0] == "device SR/BPM13-EL"
        assert "  bpm13-el-rack  located_in HALL/ROW3/RK08, accepted, unit 22" in lines
        assert lines.index("claims") < lines.index("provenance")


def test_a_place_lists_the_claims_it_is_an_end_of(repo: Path) -> None:
    _with_shared_name(repo)

    (entry,) = json.loads(_show(repo, "--json", "HALL/ROW3/RK07").stdout)["records"]

    assert entry["match"] == "id"
    assert {
        "id": "qf05-ps-rack-a",
        "type": "located_in",
        "other": "SR/QF05-PS",
        "standing": "contested",
        "role": "object",
        "unit": 12,
    } in entry["claims"]


def test_a_place_prints_its_object_end_claim_in_the_claims_direction(repo: Path) -> None:
    """The rack is the object of ``qf05-ps-rack-a``; the row still reads subject first."""
    lines = _show(repo, "HALL/ROW3/RK07").stdout.splitlines()

    assert "  qf05-ps-rack-a  SR/QF05-PS located_in (this), contested, unit 12" in lines


@pytest.mark.parametrize("as_json", [False, True])
def test_a_device_lists_object_end_claims_in_the_claims_direction(
    repo: Path, as_json: bool
) -> None:
    """``qf05-power`` reads as SR/QF05 powered by SR/QF05-PS, not the other way round."""
    result = _show(repo, *(["--json"] if as_json else []), "SR/QF05-PS")

    assert result.exit_code == 0, result.output
    if as_json:
        (entry,) = json.loads(result.stdout)["records"]
        claims = {claim["id"]: claim for claim in entry["claims"]}
        assert claims["qf05-power"] == {
            "id": "qf05-power",
            "type": "powered_by",
            "other": "SR/QF05",
            "standing": "accepted",
            "role": "object",
        }
        assert claims["qf05-ps-mains"]["role"] == "subject"
        assert claims["qf05-ps-mains"]["other"] == "HALL/PANEL-3A"
    else:
        lines = result.stdout.splitlines()
        assert "  qf05-power  SR/QF05 powered_by (this), accepted" in lines
        assert "  qf05-ps-mains  powered_by HALL/PANEL-3A, accepted" in lines


@pytest.mark.parametrize("as_json", [False, True])
def test_a_claim_naming_the_record_at_both_ends_is_listed_once(repo: Path, as_json: bool) -> None:
    """A jumper between two ports of one device names it at both ends [CF-9]."""
    _append(
        repo / "data" / "facility" / "records" / "relations.yaml",
        {
            "id": "qf05-ps-jumper",
            "type": "cabled_to",
            "subject": {"device": "SR/QF05-PS", "port": "OUT2"},
            "object": {"device": "SR/QF05-PS", "port": "OUT3"},
            "source": {
                "document": "PS-LIST",
                "version": "C",
                "state": "released",
                "locator": "row 42",
            },
        },
    )

    result = _show(repo, *(["--json"] if as_json else []), "SR/QF05-PS")

    assert result.exit_code == 0, result.output
    if as_json:
        (entry,) = json.loads(result.stdout)["records"]
        jumpers = [claim for claim in entry["claims"] if claim["id"] == "qf05-ps-jumper"]
        assert jumpers == [
            {
                "id": "qf05-ps-jumper",
                "type": "cabled_to",
                "other": "SR/QF05-PS",
                "standing": "accepted",
                "role": "both",
            }
        ]
    else:
        lines = result.stdout.splitlines()
        assert [line for line in lines if "qf05-ps-jumper" in line] == [
            "  qf05-ps-jumper  (this) cabled_to (this), accepted"
        ]


@pytest.mark.parametrize("as_json", [False, True])
def test_unknown_text_exits_1_with_claims_present(repo: Path, as_json: bool) -> None:
    _with_shared_name(repo)

    result = _show(repo, *(["--json"] if as_json else []), "no such thing")

    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert result.stderr.splitlines()[-1] == "facility show: no record no such thing"
