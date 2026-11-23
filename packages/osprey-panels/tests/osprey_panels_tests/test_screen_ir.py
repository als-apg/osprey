"""The Screen-IR version, the vendored schema, and what the panels write against them."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from osprey_panels import emit, screen_ir
from osprey_panels_tests._fixture import PANELS_DIR
from osprey_panels_tests._harness._paths import PACKAGE_SRC, REPO_ROOT
from osprey_panels_tests._views import fixture_views, invented_views, stub_render

#: The engine's sources.
ENGINE = PACKAGE_SRC / "osprey_panels"

#: The one line of the engine allowed to spell the version.
CONSTANT_LINE = 'SCREEN_IR_VERSION = "1.1"'

#: sha256 of the parity baseline's ``digests.sha256``: one ``<sha256>  ./<path>`` line
#: per file the fixture's ``write`` put in ``out`` (60 screens and ``index.json``),
#: sorted by path. Taken at the phase base with the framework command, before the
#: engine moved; the output must not change while the version stays the same.
PARITY_DIGEST = "34d1aedcbb516c9952457627ca86f0f2e800f8c917767a8a3f380d03fc6b512d"
PARITY_FILES = 61


@pytest.fixture(scope="module")
def document(panel_graph) -> dict:
    """One drawn fixture screen."""
    return stub_render(fixture_views(panel_graph)[0], panel_graph)


def test_the_schema_loads_and_declares_its_draft() -> None:
    schema = screen_ir.schema()
    assert schema["$schema"].startswith("https://json-schema.org/")
    assert "schemaVersion" in schema["required"]
    assert screen_ir.SCREEN_IR_VERSION in schema["properties"]["schemaVersion"]["enum"]


def test_every_call_hands_out_its_own_copy() -> None:
    first = screen_ir.schema()
    first["required"].clear()
    assert screen_ir.schema()["required"]


def test_the_schema_is_package_data_not_a_versioned_file() -> None:
    assert (ENGINE / "screen-ir.schema.json").is_file()
    assert not list(ENGINE.glob("screen-ir*[0-9]*.json"))


def test_every_fixture_view_draws_a_valid_document(panel_graph) -> None:
    views = fixture_views(panel_graph)
    assert views
    for view in views:
        doc = stub_render(view, panel_graph)
        assert doc["schemaVersion"] == screen_ir.SCREEN_IR_VERSION
        assert screen_ir.problems(doc) == [], view.id


def test_every_invented_view_draws_a_valid_document(panel_graph) -> None:
    for view in invented_views():
        assert screen_ir.problems(stub_render(view, panel_graph)) == [], view.id


@pytest.mark.parametrize("key", ["schemaVersion", "kind", "id", "metadata", "macros", "root"])
def test_a_document_missing_a_required_key_has_a_problem(document, key) -> None:
    broken = copy.deepcopy(document)
    del broken[key]
    found = screen_ir.problems(broken)
    assert found
    assert any(repr(key) in problem for problem in found)


@pytest.mark.parametrize("version", ["9.9", "1", 1.1, None])
def test_a_wrong_schema_version_has_a_problem(document, version) -> None:
    broken = {**document, "schemaVersion": version}
    found = screen_ir.problems(broken)
    assert found
    assert all(problem.startswith("$.schemaVersion") for problem in found)


def test_problems_are_sorted_lines(document) -> None:
    broken = {k: v for k, v in document.items() if k not in ("kind", "id")}
    broken["schemaVersion"] = "0"
    found = screen_ir.problems(broken)
    assert len(found) >= 3
    assert found == sorted(found)
    assert all("\n" not in problem for problem in found)


def test_the_version_is_spelled_only_at_the_constant() -> None:
    quoted = re.compile(r"""(["'])1\.1\1""")
    named = re.compile(r"Screen-IR\s+v?\d+\.\d+")
    found = []
    for path in sorted(ENGINE.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            where = f"{path.relative_to(ENGINE)}:{number}"
            if named.search(line):
                found.append(where)
            if quoted.search(line) and not (
                path.name == "screen_ir.py" and line.strip() == CONSTANT_LINE
            ):
                found.append(where)
    assert found == []
    assert CONSTANT_LINE in (ENGINE / "screen_ir.py").read_text(encoding="utf-8")


def test_the_layout_reads_the_constant() -> None:
    source = (ENGINE / "layout.py").read_text(encoding="utf-8")
    assert '"schemaVersion": SCREEN_IR_VERSION' in source


def test_the_fixture_output_equals_the_parity_baseline(panel_graph, tmp_path) -> None:
    out = tmp_path / "out"
    emit.write(panel_graph, out, draw=emit.load_draw(PANELS_DIR / emit.DRAW_FILE))
    files = sorted(
        (path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()),
        key=str.encode,
    )
    lines = "".join(
        f"{hashlib.sha256((out / name).read_bytes()).hexdigest()}  ./{name}\n" for name in files
    )
    assert len(files) == PARITY_FILES
    assert hashlib.sha256(lines.encode()).hexdigest() == PARITY_DIGEST
    for name in files:
        if name.endswith(emit.SCREEN_SUFFIX):
            doc = json.loads((out / name).read_bytes())
            assert screen_ir.problems(doc) == [], name


#: The variable naming an already built osprey-panels wheel.
WHEEL_VARIABLE = "OSPREY_PANELS_WHEEL"


@pytest.mark.wheel_build
def test_the_wheel_ships_the_schema(tmp_path) -> None:
    named = os.environ.get(WHEEL_VARIABLE)
    if named:
        wheel = Path(named)
        assert wheel.is_file(), f"{WHEEL_VARIABLE}={named} names no file"
    else:
        uv = shutil.which("uv")
        assert uv, f"{WHEEL_VARIABLE} is unset and uv is not on PATH to build the wheel"
        subprocess.run(
            [uv, "build", "--package", "osprey-panels", "--wheel", "--out-dir", str(tmp_path)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        (wheel,) = tmp_path.glob("osprey_panels-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        shipped = archive.read("osprey_panels/screen-ir.schema.json")
    assert shipped == (ENGINE / "screen-ir.schema.json").read_bytes()
