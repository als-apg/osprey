"""The canonical ``va.json``/``response.json`` lane of ``osprey mml import``.

The unit lanes hold the writer to the byte rules of the canonical pair -- the
same serialisation, the same untouched file on an unchanged re-import -- and
the merge to one block per system. The command lane runs the real ``mml
import`` on the committed synthetic 2.0 export: naming its ``ao.json`` alone
must pull in the siblings that sit beside it, and importing a 1.0 export must
leave ``data/mml/`` with the three files it always had.

Every key set asserted here comes from ``tests/templates/mml_export_contract``,
so a block that quietly stops carrying a key fails in the contract's terms.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.services.mml import canonical as ao_canonical
from osprey.services.mml.va import canonical as va_canonical
from osprey.services.mml.va.canonical import (
    RESPONSE_FILENAME,
    VA_FILENAME,
    merge_response_inputs,
    merge_va_inputs,
    sibling_system,
    write_va_canonical,
)
from tests.templates.mml_export_contract import (
    EXPORT_BLOCK_KEYS,
    EXPORTER_VERSION,
    VA_LATTICE_KEYS,
    VA_RESPONSE_BLOCK_KEYS,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mml"
SYNTHETIC = FIXTURES / "synthetic"

#: The system the synthetic export carries, from its own ``_export``.
SYSTEM = "SR"

#: The files a 1.0 import writes, and all it writes.
ONE_ZERO_OUTPUTS = ["PROFILE.md", "ad.json", "ao.json"]


def _document(name: str) -> dict:
    """One committed sibling document of the synthetic export."""
    return json.loads((SYNTHETIC / name).read_text(encoding="utf-8"))


def _copy(repo: Path, *names: str) -> None:
    """Put the named synthetic export files into the deployment repo."""
    for name in names:
        shutil.copy(SYNTHETIC / name, repo / name)


def _import(*args: str):
    """Run ``osprey mml import`` with *args* in the current directory."""
    return CliRunner().invoke(cli, ["mml", "import", *args], catch_exceptions=False)


def _written(repo: Path, name: str) -> dict:
    """Read one document the import wrote."""
    return json.loads((repo / "data" / "mml" / name).read_text(encoding="utf-8"))


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal deployment repo (a ``profile.yml`` marker) as the cwd."""
    root = tmp_path / "deploy"
    root.mkdir()
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


class TestCanonicalBytes:
    """The siblings are written by the rules the canonical pair is written by."""

    def test_the_two_writers_spell_the_same_document_the_same_way(self):
        """One byte rule, not two: the sibling writer never drifts from the pair."""
        document = {
            "b": {"z": 1, "a": [1.5, "Inf", None]},
            "a": {"unicode": "µm", "empty": {}},
        }

        assert va_canonical._dumps(document) == ao_canonical._dumps(document)

    def test_the_document_is_sorted_indented_and_newline_terminated(self, tmp_path: Path):
        """The bytes are a pure function of the content, so the digest is too."""
        write_va_canonical({"B": {"k": 1}, "A": {"k": 2}}, {}, tmp_path)

        text = (tmp_path / VA_FILENAME).read_text(encoding="utf-8")

        assert text == '{\n  "A": {\n    "k": 2\n  },\n  "B": {\n    "k": 1\n  }\n}\n'

    def test_an_unchanged_reimport_leaves_the_file_alone(self, tmp_path: Path):
        """An identical import keeps the file, so a digest stamped from it holds."""
        write_va_canonical({SYSTEM: {"k": 1}}, {SYSTEM: {"k": 2}}, tmp_path)
        before = (tmp_path / VA_FILENAME).stat().st_mtime_ns

        write_va_canonical({SYSTEM: {"k": 1}}, {SYSTEM: {"k": 2}}, tmp_path)

        assert (tmp_path / VA_FILENAME).stat().st_mtime_ns == before

    def test_a_non_finite_float_is_refused_before_anything_is_written(self, tmp_path: Path):
        """A stray ``inf`` raises instead of writing a bare ``Infinity`` token."""
        with pytest.raises(ValueError):
            write_va_canonical({SYSTEM: {"k": 1}}, {SYSTEM: {"k": float("inf")}}, tmp_path)

        assert not (tmp_path / VA_FILENAME).exists()
        assert not (tmp_path / RESPONSE_FILENAME).exists()

    def test_an_export_without_siblings_writes_neither_file(self, tmp_path: Path):
        """A 1.0 import leaves the directory with the files it had."""
        assert write_va_canonical({}, {}, tmp_path) == (None, None)
        assert list(tmp_path.iterdir()) == []

    def test_each_file_is_written_only_for_the_document_that_exists(self, tmp_path: Path):
        """An export whose response measurement failed still files its facts."""
        va_path, response_path = write_va_canonical({SYSTEM: {"k": 1}}, {}, tmp_path)

        assert va_path == tmp_path / VA_FILENAME
        assert response_path is None
        assert not (tmp_path / RESPONSE_FILENAME).exists()


class TestMerge:
    """One block per system, keyed as the AO and AD are keyed."""

    def test_each_input_lands_under_its_own_system(self):
        """Two sub-machines of one facility import side by side."""
        merged = merge_va_inputs(
            [
                (Path("a.va.json"), {"lattice": {"elements": 3}}, "StorageRing"),
                (Path("b.va.json"), {"lattice": {"elements": 4}}, "LTB"),
            ]
        )

        assert merged == {
            "StorageRing": {"lattice": {"elements": 3}},
            "LTB": {"lattice": {"elements": 4}},
        }

    def test_the_block_is_stored_as_the_exporter_wrote_it(self):
        """Nothing is normalised and nothing is dropped, provenance included."""
        document = _document("quokka.sr.va.json")

        merged = merge_va_inputs([(SYNTHETIC / "quokka.sr.va.json", document, SYSTEM)])

        assert merged[SYSTEM] == document

    def test_two_virtual_accelerator_inputs_for_one_system_are_refused(self):
        """One system takes one export, as one system takes one AO."""
        inputs = [
            (Path("first.va.json"), {}, SYSTEM),
            (Path("second.va.json"), {}, SYSTEM),
        ]

        with pytest.raises(click.UsageError) as excinfo:
            merge_va_inputs(inputs)

        message = str(excinfo.value)
        assert "first.va.json" in message
        assert "second.va.json" in message
        assert SYSTEM in message

    def test_two_response_inputs_for_one_system_are_refused(self):
        """The response document is keyed by the same rule and refuses the same."""
        inputs = [
            (Path("first.response.json"), {}, SYSTEM),
            (Path("second.response.json"), {}, SYSTEM),
        ]

        with pytest.raises(click.UsageError, match="response matrices"):
            merge_response_inputs(inputs)

    def test_an_empty_input_list_is_an_empty_document(self):
        """Nothing imported is nothing written."""
        assert merge_va_inputs([]) == {}
        assert merge_response_inputs([]) == {}


class TestSiblingSystem:
    """Which system a sibling document belongs to."""

    def test_a_known_token_wins_over_the_document(self):
        """The export it was paired with, or ``--system``, settles it first."""
        document = {"_export": {"submachine": "SR"}}

        assert sibling_system(Path("x.va.json"), document, "LTB") == "LTB"

    def test_the_export_block_names_the_system_otherwise(self):
        """A sibling carries the sub-machine the exporter sampled."""
        document = _document("quokka.sr.va.json")

        assert sibling_system(Path("quokka.sr.va.json"), document, None) == SYSTEM

    def test_a_token_is_stripped(self):
        """A padded token names the same system as the bare one."""
        assert sibling_system(Path("x.va.json"), {}, "  SR  ") == SYSTEM

    def test_a_sibling_that_names_no_system_is_refused_by_name(self):
        """The refusal names the file and the option that settles it."""
        with pytest.raises(click.UsageError) as excinfo:
            sibling_system(Path("orphan.va.json"), {}, None)

        message = str(excinfo.value)
        assert "orphan.va.json" in message
        assert "--system" in message

    def test_a_token_starting_with_an_underscore_is_refused(self):
        """A bookkeeping key of the canonical document is not a system."""
        with pytest.raises(click.UsageError, match="_exports"):
            sibling_system(Path("x.va.json"), {}, "_exports")


class TestImportSiblings:
    """``osprey mml import`` files the siblings beside the canonical pair."""

    def test_naming_the_ao_alone_imports_the_whole_export(self, repo: Path):
        """The siblings are named after the export, so the AO finds them."""
        _copy(
            repo,
            "quokka.sr.ao.json",
            "quokka.sr.ad.json",
            "quokka.sr.va.json",
            "quokka.sr.response.json",
        )

        result = _import("quokka.sr.ao.json")

        assert result.exit_code == 0, result.output
        assert _written(repo, VA_FILENAME) == {SYSTEM: _document("quokka.sr.va.json")}
        assert _written(repo, RESPONSE_FILENAME) == {SYSTEM: _document("quokka.sr.response.json")}

    def test_the_imported_block_spends_the_frozen_key_sets(self, repo: Path):
        """A key that stops reaching ``data/mml/`` fails in the contract's terms."""
        _copy(
            repo,
            "quokka.sr.ao.json",
            "quokka.sr.ad.json",
            "quokka.sr.va.json",
            "quokka.sr.response.json",
        )

        _import("quokka.sr.ao.json")

        block = _written(repo, VA_FILENAME)[SYSTEM]
        response = _written(repo, RESPONSE_FILENAME)[SYSTEM]
        assert set(block["lattice"]) == set(VA_LATTICE_KEYS)
        assert set(block["_export"]) == set(EXPORT_BLOCK_KEYS)
        assert block["_export"]["exporter"] == EXPORTER_VERSION
        assert set(response["blocks"][0]) == set(VA_RESPONSE_BLOCK_KEYS)

    def test_the_lattice_beside_the_ao_is_filed_under_its_system(self, repo: Path):
        """The deck pairs by name like the other siblings, with no flag."""
        _copy(repo, "quokka.sr.ao.json", "quokka.sr.ad.json", "quokka.sr.lattice.mat")

        result = _import("quokka.sr.ao.json")

        assert result.exit_code == 0, result.output
        copied = repo / "data" / "mml" / "lattice" / f"{SYSTEM}.mat"
        assert copied.read_bytes() == (SYNTHETIC / "quokka.sr.lattice.mat").read_bytes()

    def test_a_sibling_named_on_the_command_line_is_imported_once(self, repo: Path):
        """Naming a file the import would have found anyway is not two inputs."""
        _copy(repo, "quokka.sr.ao.json", "quokka.sr.ad.json", "quokka.sr.va.json")

        result = _import("quokka.sr.ao.json", "quokka.sr.va.json")

        assert result.exit_code == 0, result.output
        assert _written(repo, VA_FILENAME) == {SYSTEM: _document("quokka.sr.va.json")}

    def test_a_sibling_of_a_system_nothing_else_carries_is_refused(self, repo: Path):
        """A block with no AO to read it against is refused before anything is written."""
        _copy(repo, "quokka.sr.va.json")

        result = _import("quokka.sr.va.json")

        assert result.exit_code != 0
        assert SYSTEM in result.output
        assert not (repo / "data" / "mml").exists()

    def test_two_virtual_accelerator_exports_for_one_system_are_refused(self, repo: Path):
        """Both name the same sub-machine, and the refusal names both files."""
        _copy(repo, "quokka.sr.ao.json", "quokka.sr.ad.json", "quokka.sr.va.json")
        shutil.copy(SYNTHETIC / "quokka.sr.va.json", repo / "second.va.json")

        result = _import("quokka.sr.ao.json", "second.va.json")

        assert result.exit_code != 0
        assert "second.va.json" in result.output

    def test_the_system_option_names_the_system_of_a_sibling(self, repo: Path):
        """A sibling whose provenance says nothing takes the token it is given."""
        _copy(repo, "quokka.sr.ao.json", "quokka.sr.ad.json")
        document = _document("quokka.sr.va.json")
        del document["_export"]
        (repo / "loose.va.json").write_text(json.dumps(document), encoding="utf-8")

        result = _import(
            "quokka.sr.ao.json", "loose.va.json", "--system", f"loose.va.json={SYSTEM}"
        )

        assert result.exit_code == 0, result.output
        assert _written(repo, VA_FILENAME)[SYSTEM]["lattice"] == document["lattice"]

    def test_a_reimport_of_the_same_export_leaves_the_files_alone(self, repo: Path):
        """The digests stamped into the emitted artifacts survive a re-import."""
        _copy(
            repo,
            "quokka.sr.ao.json",
            "quokka.sr.ad.json",
            "quokka.sr.va.json",
            "quokka.sr.response.json",
        )
        _import("quokka.sr.ao.json")
        written = repo / "data" / "mml"
        before = {
            name: (written / name).stat().st_mtime_ns for name in (VA_FILENAME, RESPONSE_FILENAME)
        }

        _import("quokka.sr.ao.json")

        assert {
            name: (written / name).stat().st_mtime_ns for name in (VA_FILENAME, RESPONSE_FILENAME)
        } == before

    def test_a_one_zero_import_writes_what_it_always_wrote(self, repo: Path):
        """An export with no siblings gains no files and no directory."""
        result = _import(str(FIXTURES / "paired" / "quokka.ring.ao.json"))

        assert result.exit_code == 0, result.output
        assert sorted(path.name for path in (repo / "data" / "mml").iterdir()) == ONE_ZERO_OUTPUTS
