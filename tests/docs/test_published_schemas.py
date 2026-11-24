"""The docs build serves the facility.json schemas at the URL their ``$id`` names.

Drives the build hook's copy-and-map function from ``docs/source/conf.py`` into a
temporary directory, so no docs build is needed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.docs._conf import run_conf

PACKAGE_SCHEMAS = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "osprey-facility"
    / "src"
    / "osprey_facility"
    / "schema"
    / "_generated"
)


@pytest.fixture
def conf(monkeypatch: pytest.MonkeyPatch) -> dict:
    return run_conf(monkeypatch)


def _served_base(conf: dict) -> str:
    return conf["html_baseurl"] + "latest/"


def test_the_hook_reads_the_package_schemas(conf: dict) -> None:
    assert Path(conf["FACILITY_SCHEMA_DIR"]).resolve() == PACKAGE_SCHEMAS


def test_copied_schemas_equal_the_package_files(conf: dict, tmp_path: Path) -> None:
    served = conf["_copy_published_schemas"](str(tmp_path))
    assert served, "no schema was served"
    for path, source in served.items():
        assert Path(source).parent.resolve() == PACKAGE_SCHEMAS
        assert (tmp_path / path).read_bytes() == Path(source).read_bytes(), path


def test_every_served_id_equals_its_served_url(conf: dict, tmp_path: Path) -> None:
    served = conf["_copy_published_schemas"](str(tmp_path))
    with_id = 0
    for path in served:
        schema = json.loads((tmp_path / path).read_text(encoding="utf-8"))
        if "$id" in schema:
            with_id += 1
            assert schema["$id"] == _served_base(conf) + path, path
    assert with_id >= 1, "the current schema is served without an $id"


def test_current_and_floor_are_served_at_their_names(conf: dict, tmp_path: Path) -> None:
    served = conf["_copy_published_schemas"](str(tmp_path))
    current = json.loads((PACKAGE_SCHEMAS / "facility.schema.json").read_text(encoding="utf-8"))
    n = current["properties"]["schema"]["const"].rsplit("/", 1)[-1]
    assert Path(served[f"schemas/facility/{n}.schema.json"]).name == "facility.schema.json"
    assert Path(served[f"schemas/facility/{n}.floor.schema.json"]).name == (
        f"facility-{n}.schema.json"
    )


def _fake_generated(directory: Path, n: int) -> Path:
    base = "https://als-apg.github.io/osprey/latest/schemas/facility"
    directory.mkdir()

    def write(name: str, schema: dict) -> None:
        (directory / name).write_text(json.dumps(schema), encoding="utf-8")

    header = {"properties": {"schema": {"const": f"osprey.facility.facility/{n}"}}}
    write("facility.schema.json", {"$id": f"{base}/{n}.schema.json", **header})
    write(f"facility-{n - 1}.final.schema.json", {"$id": f"{base}/{n - 1}.schema.json"})
    write(f"facility-{n - 1}.schema.json", {})
    write(f"facility-{n}.schema.json", {})
    write("vocabulary.json", {})
    return directory


def test_finals_and_floors_map_after_a_bump(conf: dict, tmp_path: Path) -> None:
    generated = _fake_generated(tmp_path / "generated", 5)
    served = conf["_published_schemas"](str(generated))
    assert {path: Path(source).name for path, source in served.items()} == {
        "schemas/facility/5.schema.json": "facility.schema.json",
        "schemas/facility/4.schema.json": "facility-4.final.schema.json",
        "schemas/facility/4.floor.schema.json": "facility-4.schema.json",
        "schemas/facility/5.floor.schema.json": "facility-5.schema.json",
    }
    out = tmp_path / "html"
    conf["_copy_published_schemas"](str(out), str(generated))
    for path in served:
        schema = json.loads((out / path).read_text(encoding="utf-8"))
        if "$id" in schema:
            assert schema["$id"] == _served_base(conf) + path


def test_two_files_claiming_one_url_is_an_error(conf: dict, tmp_path: Path) -> None:
    generated = _fake_generated(tmp_path / "generated", 5)
    shutil.copyfile(
        generated / "facility-4.final.schema.json", generated / "facility-5.final.schema.json"
    )
    with pytest.raises(ValueError, match="5.schema.json"):
        conf["_published_schemas"](str(generated))
