"""Tests for the All PVs origin join: one address normaliser and the curated marks.

The marks are read from the facility file a build writes, through
``channel_roster.registered_channels``. An absent roster says nothing about
any name, so it marks ``unknown``, and the lifecycle test at the end pins the
join's direction: this package reads the roster and never writes it.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import osprey.channel_roster as channel_roster
from osprey.services import pv_directory
from osprey.services.pv_directory import origin
from osprey.services.pv_directory.origin import NormalisedAddress, mark, normalise_address


@pytest.fixture(autouse=True)
def cold_cache() -> Iterator[None]:
    """Each test reads its own facility file, not a memoized one."""
    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


def _render(render: Path, *addresses: str) -> dict[str, Any]:
    """Write a facility file declaring *addresses* and return its project's config."""
    channels = [{"id": address, "role": "none"} for address in addresses]
    (render / "facility.json").write_text(json.dumps({"channels": channels}), encoding="utf-8")
    return {"config_dir": str(render)}


class TestNormaliseAddress:
    @pytest.mark.parametrize("spelling", ["SR:QF:01", "ca://SR:QF:01", "pva://SR:QF:01"])
    def test_protocol_prefix_is_stripped(self, spelling: str) -> None:
        assert normalise_address(spelling) == NormalisedAddress("SR:QF:01", None)

    @pytest.mark.parametrize(
        ("spelling", "field"),
        [("SR:QF:01.VAL", "VAL"), ("SR:QF:01.HIHI", "HIHI"), ("SR:QF:01.DESC$", "DESC$")],
    )
    def test_field_suffix_is_split_and_kept_for_display(self, spelling: str, field: str) -> None:
        address = normalise_address(spelling)

        assert address == NormalisedAddress("SR:QF:01", field)
        assert address.display == spelling

    def test_prefix_and_field_are_handled_together(self) -> None:
        address = normalise_address("pva://SR:QF:01.RBV")

        assert address == NormalisedAddress("SR:QF:01", "RBV")
        assert address.display == "SR:QF:01.RBV"

    def test_a_bare_name_displays_as_itself(self) -> None:
        assert normalise_address("SR:QF:01").display == "SR:QF:01"

    def test_padding_is_not_part_of_the_name(self) -> None:
        assert normalise_address("  ca://SR:QF:01.VAL\n") == NormalisedAddress("SR:QF:01", "VAL")

    @pytest.mark.parametrize("name", ["sim.bpm.x", "SR:QF.01", "trailing.", ".VAL", "SR.Val"])
    def test_a_dot_that_introduces_no_field_stays_in_the_name(self, name: str) -> None:
        assert normalise_address(name) == NormalisedAddress(name, None)

    @pytest.mark.parametrize("other", ["http://SR:QF:01", "CA://SR:QF:01", "xca://SR:QF:01"])
    def test_only_the_two_protocols_are_prefixes(self, other: str) -> None:
        assert normalise_address(other).name == other


class TestMark:
    def test_a_rostered_name_is_curated_and_any_other_is_directory(self, tmp_path: Path) -> None:
        config = _render(tmp_path, "SR:QF:01", "SR:QD:01")

        assert mark(["SR:QF:01", "SR:HEALTH:HEARTBEAT"], config) == {
            "SR:QF:01": "curated",
            "SR:HEALTH:HEARTBEAT": "directory",
        }

    @pytest.mark.parametrize(
        "spelling", ["ca://SR:QF:01", "pva://SR:QF:01", "SR:QF:01.HIHI", "pva://SR:QF:01.VAL"]
    )
    def test_a_name_marks_through_the_normaliser_and_keeps_its_spelling(
        self, tmp_path: Path, spelling: str
    ) -> None:
        assert mark([spelling], _render(tmp_path, "SR:QF:01")) == {spelling: "curated"}

    def test_a_rostered_spelling_goes_through_the_same_normaliser(self, tmp_path: Path) -> None:
        config = _render(tmp_path, "ca://SR:QF:01.VAL")

        assert mark(["SR:QF:01"], config) == {"SR:QF:01": "curated"}

    def test_the_join_is_exact_with_no_alias_resolution(self, tmp_path: Path) -> None:
        config = _render(tmp_path, "SR:QF:01")

        assert mark(["sr:qf:01", "SR:QF:01:ALIAS", "SR:QF"], config) == {
            "sr:qf:01": "directory",
            "SR:QF:01:ALIAS": "directory",
            "SR:QF": "directory",
        }

    def test_a_roster_never_built_marks_unknown_never_directory(self, tmp_path: Path) -> None:
        config = {"config_dir": str(tmp_path)}

        assert mark(["SR:QF:01", "SR:QD:01"], config) == {
            "SR:QF:01": "unknown",
            "SR:QD:01": "unknown",
        }

    @pytest.mark.parametrize("payload", ["{not json", '{"channels": []}'], ids=["corrupt", "empty"])
    def test_any_other_absence_marks_unknown_too(self, tmp_path: Path, payload: str) -> None:
        (tmp_path / "facility.json").write_text(payload, encoding="utf-8")

        assert mark(["SR:QF:01"], {"config_dir": str(tmp_path)}) == {"SR:QF:01": "unknown"}

    def test_no_names_is_no_marks(self, tmp_path: Path) -> None:
        assert mark([], _render(tmp_path, "SR:QF:01")) == {}

    def test_the_marks_come_only_from_registered_channels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: list[dict[str, Any]] = []
        config = _render(tmp_path, "SR:QF:01")
        roster = channel_roster.registered_channels(config)

        def spy(given: dict[str, Any]) -> channel_roster.RosterResult:
            asked.append(given)
            return roster

        monkeypatch.setattr(origin, "registered_channels", spy)

        assert mark(["SR:QF:01"], config) == {"SR:QF:01": "curated"}
        assert asked == [config]

    def test_the_canopy_channel_list_is_not_a_source(self) -> None:
        assert "channels.json" not in Path(origin.__file__).read_text(encoding="utf-8")


class TestRosterLifecycle:
    """The roster is the facility file; ``services/pv_directory`` only ever reads it."""

    def test_marking_leaves_the_facility_file_untouched(self, tmp_path: Path) -> None:
        config = _render(tmp_path, "SR:QF:01")
        facility_file = tmp_path / "facility.json"
        before = (facility_file.read_bytes(), facility_file.stat().st_mtime_ns)

        mark(["SR:QF:01", "SR:QD:01"], config)

        assert (facility_file.read_bytes(), facility_file.stat().st_mtime_ns) == before
        assert [path.name for path in tmp_path.iterdir()] == ["facility.json"]

    def test_nothing_in_the_package_can_reach_the_roster_to_write_it(self) -> None:
        """Only ``registered_channels`` crosses into the roster, and it hands back records.

        To write the roster a module would have to name the file, locate it through
        the package's resolvers, or reach into the package's own stages or memo.
        """
        package_dir = Path(pv_directory.__path__[0])
        modules = sorted(package_dir.rglob("*.py"))
        assert package_dir / "origin.py" in modules

        reaches = [
            f"{module.relative_to(package_dir)}: {reach}"
            for module in modules
            for reach in _roster_reaches(ast.parse(module.read_text(encoding="utf-8")))
        ]

        assert reaches == []


def _roster_reaches(tree: ast.Module) -> Iterator[str]:
    """Yield each way *tree* goes past ``registered_channels`` towards the roster's file."""
    documented = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, documented) and ast.get_docstring(node, clean=False) is not None
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "osprey.channel_roster"
        ):
            for alias in node.names:
                if node.module != "osprey.channel_roster" or alias.name != "registered_channels":
                    yield f"imports {alias.name} from {node.module}"
        elif isinstance(node, ast.ImportFrom) and node.module == "osprey":
            for alias in node.names:
                if alias.name == "channel_roster":
                    yield "imports the channel_roster module whole"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("osprey.channel_roster"):
                    yield f"imports {alias.name} whole"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "facility.json" in node.value and id(node) not in docstrings:
                yield "names facility.json"
