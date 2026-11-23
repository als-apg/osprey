"""The shared checked read of ``facility.json``: header, shape, refusal wording."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from osprey.facility import reading, validate
from osprey.facility.errors import FacilityFileError
from osprey.facility.reading import READS, check_header, load_facility
from tests._facility_file import FACILITY_V4, write_facility_document, write_facility_v4
from tests.facility._reader_roster import CHANNELS, ROSTER

HEADER = "osprey.facility.facility/4"
REBUILD = "-- rebuild with osprey build"


def _refusal(source: object, found: str) -> str:
    return f"{source} is {found}; this reader reads {HEADER} {REBUILD}"


# --- the contract ------------------------------------------------------------


def test_reads_holds_the_header_the_build_writes():
    assert validate.FACILITY_HEADER in READS


def test_reads_is_the_one_current_header():
    assert READS == frozenset({HEADER})


def test_the_test_writer_header_is_read():
    assert FACILITY_V4 in READS


def test_reads_is_declared_as_a_literal():
    tree = ast.parse(Path(reading.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "READS":
            assert ast.unparse(node.value) == "frozenset({'osprey.facility.facility/4'})"
            return
    pytest.fail("READS is not assigned at module level")


def test_imports_only_the_error_module_and_the_stdlib():
    tree = ast.parse(Path(reading.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    foreign = {
        name
        for name in imported
        if name.split(".")[0] not in sys.stdlib_module_names and name != "osprey.facility.errors"
    }
    assert foreign == set()
    assert "osprey.facility.errors" in imported


def test_the_refusal_is_a_value_error():
    assert issubclass(FacilityFileError, ValueError)


# --- load_facility -----------------------------------------------------------


def test_a_v4_document_is_read(tmp_path):
    path = write_facility_v4(tmp_path, {"channels": [{"id": "SR:BPM1:X"}]})
    assert load_facility(path) == {"channels": [{"id": "SR:BPM1:X"}], "schema": HEADER}


def test_a_str_path_is_read(tmp_path):
    path = write_facility_v4(tmp_path)
    assert load_facility(str(path)) == {"schema": HEADER}


@pytest.mark.foreign_header
def test_a_v3_document_is_refused(tmp_path):
    path = write_facility_document(tmp_path, {"channels": []}, schema="osprey.facility.facility/3")
    with pytest.raises(FacilityFileError) as caught:
        load_facility(path)
    assert str(caught.value) == _refusal(path, "osprey.facility.facility/3")


@pytest.mark.foreign_header
@pytest.mark.parametrize(
    ("header", "found"),
    [
        ("osprey.facility.facility/5", "osprey.facility.facility/5"),
        ("osprey.facility.facility/4 ", "'osprey.facility.facility/4 '"),
        ("", "''"),
        (4, "4"),
        (None, "None"),
        (["osprey.facility.facility/4"], "['osprey.facility.facility/4']"),
    ],
)
def test_other_headers_are_refused(tmp_path, header, found):
    path = write_facility_document(tmp_path, {"channels": [], "schema": header}, schema=None)
    with pytest.raises(FacilityFileError) as caught:
        load_facility(path)
    assert str(caught.value) == _refusal(path, found)


@pytest.mark.foreign_header
def test_a_header_less_document_is_refused(tmp_path):
    path = write_facility_document(tmp_path, {"channels": []}, schema=None)
    with pytest.raises(FacilityFileError) as caught:
        load_facility(path)
    assert str(caught.value) == _refusal(path, "a document without a schema header")


@pytest.mark.foreign_header
@pytest.mark.parametrize(
    ("payload", "word"),
    [
        ([], "a list"),
        (4, "a number"),
        (1.5, "a number"),
        ("x", "a string"),
        (True, "a boolean"),
        (None, "null"),
    ],
)
def test_a_non_object_document_is_refused(tmp_path, payload, word):
    path = write_facility_document(tmp_path, payload)
    with pytest.raises(FacilityFileError) as caught:
        load_facility(path)
    assert str(caught.value) == f"{path} is {word}, not a JSON object {REBUILD}"


def test_a_file_that_is_not_json_is_refused(tmp_path):
    path = tmp_path / "facility.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(FacilityFileError) as caught:
        load_facility(path)
    message = str(caught.value)
    assert message.startswith(f"{path} is not JSON: ")
    assert message.endswith(REBUILD)
    assert isinstance(caught.value.__cause__, ValueError)


def test_a_file_that_is_not_utf8_is_refused(tmp_path):
    path = tmp_path / "facility.json"
    path.write_bytes(b'{"schema": "\xff"}')
    with pytest.raises(FacilityFileError) as caught:
        load_facility(path)
    message = str(caught.value)
    assert message.startswith(f"{path} cannot be read: ")
    assert message.endswith(REBUILD)


def test_a_file_that_cannot_be_read_is_refused(tmp_path):
    path = tmp_path / "facility.json"
    path.mkdir()
    with pytest.raises(FacilityFileError) as caught:
        load_facility(path)
    message = str(caught.value)
    assert message.startswith(f"{path} cannot be read: ")
    assert message.endswith(REBUILD)


def test_an_absent_file_is_an_absence(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_facility(tmp_path / "facility.json")


def test_an_absent_file_is_not_a_refusal(tmp_path):
    with pytest.raises(FileNotFoundError) as caught:
        load_facility(tmp_path / "facility.json")
    assert not isinstance(caught.value, FacilityFileError)


# --- check_header ------------------------------------------------------------


def test_check_header_passes_a_v4_document():
    assert check_header({"schema": HEADER}, "doc") is None


def test_check_header_refuses_a_v3_document():
    with pytest.raises(FacilityFileError) as caught:
        check_header({"schema": "osprey.facility.facility/3"}, "build/facility.json")
    assert str(caught.value) == _refusal("build/facility.json", "osprey.facility.facility/3")


def test_check_header_refuses_a_header_less_document():
    with pytest.raises(FacilityFileError) as caught:
        check_header({}, "doc")
    assert str(caught.value) == _refusal("doc", "a document without a schema header")


@pytest.mark.parametrize(
    ("document", "word"),
    [(7, "a number"), ("osprey.facility.facility/4", "a string"), ([], "a list")],
)
def test_check_header_refuses_a_non_object_document(document, word):
    with pytest.raises(FacilityFileError) as caught:
        check_header(document, "doc")
    assert str(caught.value) == f"doc is {word}, not a JSON object {REBUILD}"


# --- the test writers --------------------------------------------------------


def test_the_v4_writer_refuses_another_header(tmp_path):
    with pytest.raises(ValueError, match="write_facility_document"):
        write_facility_v4(tmp_path, {"schema": "osprey.facility.facility/3"})
    assert not (tmp_path / "facility.json").exists()


@pytest.mark.foreign_header
def test_the_raw_writer_writes_a_non_object_untouched(tmp_path):
    path = write_facility_document(tmp_path, [1, 2])
    assert path == tmp_path / "facility.json"
    assert path.read_text(encoding="utf-8").split() == ["[", "1,", "2", "]"]


def test_the_raw_writer_does_not_change_the_callers_document(tmp_path):
    document = {"schema": "osprey.facility.facility/3"}
    write_facility_document(tmp_path, document)
    assert document == {"schema": "osprey.facility.facility/3"}
    assert load_facility(tmp_path / "facility.json") == {"schema": HEADER}


# --- the roster --------------------------------------------------------------


def test_every_roster_entry_names_a_channel_and_a_test():
    assert ROSTER
    for entry in ROSTER:
        assert entry.channel in CHANNELS
        assert "::" in entry.test


def test_this_module_s_roster_tests_exist():
    names = {name for name in globals() if name.startswith("test_")}
    own = [entry for entry in ROSTER if entry.test.startswith("tests/facility/test_reading.py::")]
    assert own
    for entry in own:
        assert entry.test.split("::", 1)[1] in names
