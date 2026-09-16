"""``osprey mml import`` on every committed export form, and the ALS parity lanes.

Each synthetic fixture under ``tests/fixtures/mml/`` is imported through the real
command under ``CliRunner``, in the way its form requires: the paired exporter
output with no flags (its system comes from the sibling ``<stem>.ad.json``), the
``.mat`` with no flags (its system comes from the ``AD`` variable), the
system-keyed dialect with no flags (and refusing ``--system``), and every flat
export with ``--system``. The refusals the command owes a facility (a v7.3
MAT-file, a system token imported twice) are pinned here too, as is the promise
that ``PROFILE.md`` lists every ``MemberOf`` tag of the source verbatim.

Two lanes run against a facility's real data, which never enters the repo:

* ``OSPREY_ALS_MML_EXPORT`` names the ALS JSON export; the import-walk census
  totals are pinned to the numbers of success criterion 5.
* ``OSPREY_ALS_MML_MAT`` names the ALS ``.mat``; it imports as ``SR`` and the
  loader assertions of success criterion 1 hold on the written ``ao.json``.

Each lane skips with the reason when its variable is unset.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"

ALS_EXPORT_ENV = "OSPREY_ALS_MML_EXPORT"
ALS_MAT_ENV = "OSPREY_ALS_MML_MAT"

#: The save call the v7.3 refusal must name.
SAVE_V7 = "save('-v7', ...)"

#: Each committed fixture: its input files (one ``mml import`` call), the
#: ``--system`` arguments it needs, and the systems ``_import_order`` must record.
FIXTURE_IMPORTS: dict[str, tuple[tuple[str, ...], tuple[str, ...], list[str]]] = {
    "paired": (("paired/quokka.ring.ao.json",), (), ["RING"]),
    "mat": (("mat/quokka_booster.mat",), (), ["BOOSTER"]),
    "dialect": (("dialect/export.json",), (), ["RING", "BOOST"]),
    "tango": (("tango/export.json",), ("--system", "RING"), ["RING"]),
    "dualkey": (("dualkey/export.json",), ("--system", "STOR"), ["STOR"]),
    "casedup": (("casedup/export.json",), ("--system", "MAIN"), ["MAIN"]),
    "wrapped": (("wrapped/export.json",), ("--system", "INJ"), ["INJ"]),
    "nsls2": (
        ("nsls2/nsls2.storagering.ao.json", "nsls2/nsls2.ltb.ao.json"),
        (),
        ["StorageRing", "LTB"],
    ),
    "spear3": (("spear3/spear3.storagering.ao.json",), (), ["StorageRing"]),
}


def _inputs(sources: tuple[str, ...]) -> list[str]:
    """The fixture files as ``mml import`` arguments."""
    return [str(FIXTURES / source) for source in sources]


_MEMBER_OF_HEADING = "### MemberOf census"
_TABLE_CELL = re.compile(r"^\| ((?:[^|\\]|\\.)+?) \|")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal deployment repo (a ``profile.yml`` marker) as the cwd."""
    root = tmp_path / "deploy"
    root.mkdir()
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


def _invoke(*args: str):
    return CliRunner().invoke(cli, ["mml", *args], catch_exceptions=False)


def _out(repo: Path) -> Path:
    return repo / "data" / "mml"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield ``(key path, value)`` for every value nested in dicts and lists."""
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, (*path, str(index)))


def _source_member_of_tags(fixture: str) -> set[str]:
    """Every ``MemberOf`` tag in a fixture's source exports, read without the importer."""
    sources, _, _ = FIXTURE_IMPORTS[fixture]
    return set().union(*(_member_of_tags_in(FIXTURES / source) for source in sources))


def _member_of_tags_in(path: Path) -> set[str]:
    """Every ``MemberOf`` tag in one export file."""
    if path.suffix == ".mat":
        from scipy.io import loadmat

        data = _mat_plain(loadmat(str(path), struct_as_record=False, squeeze_me=True)["AO"])
    else:
        data = _read_json(path)

    tags: set[str] = set()
    for key_path, value in _walk(data):
        if not key_path or key_path[-1] != "MemberOf":
            continue
        items = [value] if isinstance(value, str) else value
        tags.update(item.strip() for item in items if isinstance(item, str) and item.strip())
    return tags


def _mat_plain(value: Any) -> Any:
    """Just enough of a MAT-file decode to reach every ``MemberOf`` string."""
    import numpy as np
    from scipy.io.matlab import mat_struct

    if isinstance(value, mat_struct):
        return {name: _mat_plain(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "U":
            return [str(row) for row in value.ravel()]
        if value.dtype.kind == "O":
            return [_mat_plain(item) for item in value.ravel()]
        return value.tolist()
    if isinstance(value, np.str_):
        return str(value)
    return value


def _profile_member_of_tags(profile: str) -> set[str]:
    """The first-column cells of every system's ``MemberOf census`` table."""
    tags: set[str] = set()
    in_table = False
    for line in profile.splitlines():
        if line.startswith("#"):
            in_table = line.strip() == _MEMBER_OF_HEADING
            continue
        if not in_table:
            continue
        match = _TABLE_CELL.match(line)
        if match and match.group(1) not in {"Tag", "---"} and not line.startswith("|---"):
            tags.add(match.group(1).replace("\\|", "|"))
    return tags


class TestEveryFixtureImports:
    @pytest.mark.parametrize("fixture", sorted(FIXTURE_IMPORTS))
    def test_imports_in_the_way_its_form_requires(self, repo: Path, fixture: str) -> None:
        sources, flags, systems = FIXTURE_IMPORTS[fixture]

        result = _invoke("import", *_inputs(sources), *flags)

        assert result.exit_code == 0, result.output
        assert "Traceback" not in result.output
        out = _out(repo)
        for name in ("ao.json", "ad.json", "PROFILE.md"):
            assert (out / name).is_file(), name
        ao = _read_json(out / "ao.json")
        assert ao["_import_order"] == systems
        assert [key for key in ao if not key.startswith("_")] == sorted(systems)

    @pytest.mark.parametrize("fixture", sorted(FIXTURE_IMPORTS))
    def test_profile_lists_every_member_of_tag_verbatim(self, repo: Path, fixture: str) -> None:
        sources, flags, _ = FIXTURE_IMPORTS[fixture]
        expected = _source_member_of_tags(fixture)
        assert expected, f"{fixture} carries no MemberOf tag to check"

        result = _invoke("import", *_inputs(sources), *flags)

        assert result.exit_code == 0, result.output
        profile = (_out(repo) / "PROFILE.md").read_text(encoding="utf-8")
        listed = _profile_member_of_tags(profile)
        assert expected <= listed, f"missing from PROFILE.md: {sorted(expected - listed)}"

    def test_paired_export_takes_its_system_and_ad_from_the_sibling_ad_file(
        self, repo: Path
    ) -> None:
        shutil.copy(FIXTURES / "paired" / "quokka.ring.ao.json", repo / "quokka.ring.ao.json")
        shutil.copy(FIXTURES / "paired" / "quokka.ring.ad.json", repo / "quokka.ring.ad.json")

        result = _invoke("import", "quokka.ring.ao.json")

        assert result.exit_code == 0, result.output
        ao = _read_json(_out(repo) / "ao.json")
        ad = _read_json(_out(repo) / "ad.json")
        assert ao["_import_order"] == ["RING"]
        assert ad["RING"]["SubMachine"] == "RING"
        assert ad["RING"]["Machine"] == "Quokka"

    def test_paired_export_without_its_ad_file_needs_a_system(self, repo: Path) -> None:
        shutil.copy(FIXTURES / "paired" / "quokka.ring.ao.json", repo / "quokka.ring.ao.json")

        result = _invoke("import", "quokka.ring.ao.json")

        assert result.exit_code == 2, result.output
        assert "--system" in result.output
        assert not _out(repo).exists()

    def test_system_keyed_dialect_refuses_an_explicit_system(self, repo: Path) -> None:
        result = _invoke("import", str(FIXTURES / "dialect" / "export.json"), "--system", "RING")

        assert result.exit_code == 2, result.output
        assert "drop --system" in result.output
        assert "Traceback" not in result.output
        assert not _out(repo).exists()

    @pytest.mark.parametrize("fixture", ["tango", "dualkey", "casedup", "wrapped"])
    def test_flat_export_without_a_system_is_refused(self, repo: Path, fixture: str) -> None:
        sources, _, _ = FIXTURE_IMPORTS[fixture]

        result = _invoke("import", *_inputs(sources))

        assert result.exit_code == 2, result.output
        assert "--system" in result.output
        assert not _out(repo).exists()


class TestRefusals:
    def test_v73_mat_file_is_refused_with_the_save_v7_sentence(self, repo: Path) -> None:
        header = b"MATLAB 7.3 MAT-file, Platform: GLNXA64".ljust(116, b" ")
        header += b"\x00" * 8  # subsystem data offset
        header += b"\x00\x02" + b"IM"  # version 0x0200 written little-endian
        (repo / "hdf5.mat").write_bytes(header + b"\x00" * 512)

        result = _invoke("import", "hdf5.mat", "--system", "SR")

        assert result.exit_code == 1, result.output
        assert SAVE_V7 in result.output
        assert "7.3" in result.output
        assert "Traceback" not in result.output
        assert not _out(repo).exists()

    def test_system_token_resolved_twice_across_inputs_is_refused(self, repo: Path) -> None:
        # The paired export resolves RING from its AD; the dialect names RING itself.
        result = _invoke(
            "import",
            str(FIXTURES / "paired" / "quokka.ring.ao.json"),
            str(FIXTURES / "dialect" / "export.json"),
        )

        assert result.exit_code == 2, result.output
        assert "'RING' is imported twice" in result.output
        assert "Traceback" not in result.output
        assert not _out(repo).exists()

    def test_same_explicit_token_on_two_flat_inputs_is_refused(self, repo: Path) -> None:
        shutil.copy(FIXTURES / "tango" / "export.json", repo / "a.json")
        shutil.copy(FIXTURES / "wrapped" / "export.json", repo / "b.json")

        result = _invoke(
            "import", "a.json", "b.json", "--system", "a.json=X", "--system", "b.json=X"
        )

        assert result.exit_code == 2, result.output
        assert "'X' is imported twice" in result.output
        assert not _out(repo).exists()


@pytest.mark.skipif(
    not os.environ.get(ALS_EXPORT_ENV),
    reason=f"{ALS_EXPORT_ENV} is not set; the ALS export census lane needs the real JSON export",
)
def test_als_export_import_walk_census(repo: Path) -> None:
    from osprey.services.mml.canonical import read_canonical
    from osprey.services.mml.census import take_census

    export = Path(os.environ[ALS_EXPORT_ENV]).expanduser().resolve()

    result = _invoke("import", str(export))

    assert result.exit_code == 0, result.output
    ao, ad = read_canonical(_out(repo))
    assert ao["_import_order"] == ["SR", "BR", "GTL", "LN", "LTB", "BTS"]
    totals = take_census(ao, ad).totals
    assert {
        "fields": totals.fields,
        "raw_slots": totals.raw_slots,
        "blank": totals.blank,
        "raw_non_blank": totals.raw_non_blank,
        "bindings": totals.bindings,
        "broadcast_fields": totals.broadcast_fields,
        "distinct_pvs": totals.distinct_pvs,
        "system_families": totals.system_families,
        "families": totals.families,
        "devices": totals.devices,
        "setup_families": totals.setup_families,
        "fallback_families": totals.fallback_families,
    } == {
        "fields": 1079,
        "raw_slots": 14470,
        "blank": 827,
        "raw_non_blank": 13643,
        "bindings": 13674,
        "broadcast_fields": 1,
        "distinct_pvs": 11209,
        "system_families": 110,
        "families": 72,
        "devices": 1404,
        "setup_families": 108,
        "fallback_families": 2,
    }
    assert totals.raw_slots == totals.raw_non_blank + totals.blank
    assert "11209 distinct PVs" in result.output


@pytest.mark.skipif(
    not os.environ.get(ALS_MAT_ENV),
    reason=f"{ALS_MAT_ENV} is not set; the ALS .mat import lane needs the real MAT-file",
)
def test_als_mat_imports_as_one_system_with_loader_assertions(repo: Path) -> None:
    mat = Path(os.environ[ALS_MAT_ENV]).expanduser().resolve()

    header = mat.read_bytes()[:128]
    version = header[124:126]
    major = version[1] if header[126:128] == b"IM" else version[0]
    assert major == 1, f"header bytes 124-125 of {mat} do not read v7: {version!r}"

    result = _invoke("import", str(mat), "--system", "SR")

    assert result.exit_code == 0, result.output
    ao_path = _out(repo) / "ao.json"
    ao = json.loads(ao_path.read_text(encoding="utf-8"), parse_constant=_refuse_constant)
    assert [key for key in ao if not key.startswith("_")] == ["SR"]
    assert ao["_import_order"] == ["SR"]

    for key_path, value in _walk(ao["SR"]):
        where = ".".join(key_path)
        if isinstance(value, str):
            assert value == value.rstrip(), f"{where} keeps trailing blanks: {value!r}"
        elif isinstance(value, float):
            assert math.isfinite(value), f"{where} holds a non-finite float"
        elif isinstance(value, dict):
            assert "function_handle" not in value, f"{where} holds an unfolded handle"
            if "$fn" in value:
                assert set(value) == {"$fn", "file"}, f"{where} handle is {sorted(value)}"
        if key_path and key_path[-1].endswith("Fcn"):
            # A bare name or the integer 1 under a *Fcn key is a handle left unfolded.
            assert not isinstance(value, str) and value != 1, (
                f"{where} is not a {{$fn, file}} handle: {value!r}"
            )

    typo_handle = ao["SR"].get("SD", {}).get("SetpointGolden", {}).get("HW2PhysicSDcn")
    assert isinstance(typo_handle, dict) and set(typo_handle) == {"$fn", "file"}

    init = _invoke("map", "--init")
    assert init.exit_code == 0, init.output
    assert (_out(repo) / "mapping.yaml").is_file()


def _refuse_constant(token: str) -> float:
    raise AssertionError(f"ao.json holds the bare non-finite token {token}")
