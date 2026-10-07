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

pytestmark = pytest.mark.slow

#: A demo device the authored records place in a sector by span.
DEVICE = "SR/BPM01"

#: A demo channel with a ``unit`` its authored layer states as ``mm``.
CHANNEL = "BR:DIAG:BPM:01:POSITION:X"

#: The channel-finder views, one per ``channel_finder_mode``.
CHANNEL_FINDER_VIEWS = ("in_context", "hierarchical", "middle_layer")


@pytest.fixture(scope="module")
def initialised(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A control-assistant repo, initialised once for this module and never edited."""
    return init_project(tmp_path_factory.mktemp("ca"), "control-assistant", "demo")


@pytest.fixture
def repo(initialised: Path, tmp_path: Path) -> Iterator[Path]:
    """A copy of the initialised repo this test may edit."""
    copy = tmp_path / initialised.name
    shutil.copytree(initialised, copy, symlinks=True)
    yield copy


def _show(repo: Path, *args: str) -> Result:
    return CliRunner().invoke(cli, ["facility", "show", "--repo", str(repo), *args])


def _set_mode(path: Path, mode: str) -> None:
    """Set ``channel_finder_mode`` in a profile or persona delta."""
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


def test_the_views_are_the_primary_configs_whatever_the_personas_select(repo: Path) -> None:
    _set_mode(repo / "profile.yml", "hierarchical")
    _set_mode(repo / "personas" / "readonly.yml", "in_context")
    _set_mode(repo / "personas" / "readwrite.yml", "middle_layer")

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
    assert document["kind"] == "device"
    assert document["record"]["id"] == DEVICE
    assert "provenance" not in document["record"]
    assert document["provenance"]["sources"][0]["file"] == "records/devices.yaml"
    assert document["fixes_applied"] == document["provenance"]["fixes"]


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
    assert document["kind"] == "channel"
    assert document["fixes_applied"] == [{"op": "set", "why": "The unit is micrometres."}]
    assert document["record"]["unit"] == "um"


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
def test_an_id_naming_a_place_and_a_device_exits_1(repo: Path, as_json: bool) -> None:
    _append(
        repo / "data" / "facility" / "records" / "devices.yaml",
        {"id": "BR", "class": "BeamPositionMonitor", "place": "BR"},
    )

    result = _show(repo, *(["--json"] if as_json else []), "BR")

    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert result.stderr.splitlines()[-1] == "facility show: BR names a place and a device"
