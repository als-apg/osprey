"""A facility that declares no model inputs renders the view it rendered before inputs existed.

``fixtures/pre_inputs_views/digests.json`` holds, for the example facility, the
injection fixture and each MML tree the build case holds to a fingerprint, the
SHA-256 of every file the simulator view wrote under ``data/simulator/`` before
model inputs were added (``osprey.facility.simulator/3``). Each facility is
built and rendered here exactly as it was then; every file must match its digest
once ``variables.json``'s schema is written back as ``/3``, the one byte change
a facility without inputs is allowed. The digests stand in for the files, which
come to some twelve megabytes; a mismatch names the facility and the file.

The digests were taken from a render of the commit before model inputs, under
the configuration and facility trees below. A change that alters the view of a
facility without inputs on purpose replaces them, and says why.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.build import build_facility
from osprey.facility.render import render_facility_outputs
from osprey.facility.views.simulator import VARIABLES_SCHEMA
from tests.facility._mml_built import BUILT, build_tree
from tests.facility.served_tree import IN_PROCESS_RENDER_CONFIG

pytestmark = [pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[2]
DIGESTS = Path(__file__).resolve().parent / "fixtures" / "pre_inputs_views" / "digests.json"

#: The schema every pre-inputs view carried.
PRE_INPUTS_SCHEMA = "osprey.facility.simulator/3"

#: The facility trees copied and built as they are, by the name the digests use.
TREES = {
    "example": REPO_ROOT / "src" / "osprey" / "templates" / "facilities" / "example",
    "injection": Path(__file__).resolve().parent / "fixtures" / "injection",
}


def _built(name: str, root: Path) -> tuple[dict[str, Any], Path]:
    if name in TREES:
        facility_dir = root / "data" / "facility"
        shutil.copytree(TREES[name], facility_dir)
        return build_facility(facility_dir, project_name=name), facility_dir
    return build_tree(name.removeprefix("mml-"), root)


def _view_files(root: Path, document: dict[str, Any], facility_dir: Path) -> list[Path]:
    render_dir = root / "render"
    render_dir.mkdir()
    render_facility_outputs(
        render_dir, document, dict(IN_PROCESS_RENDER_CONFIG), facility_dir, omitted_reported=set()
    )
    simulator = render_dir / "data" / "simulator"
    return [path for path in sorted(simulator.rglob("*")) if path.is_file()]


def _as_before_inputs(relative: str, data: bytes) -> bytes:
    if relative != "variables.json":
        return data
    old = json.dumps({"schema": PRE_INPUTS_SCHEMA})[1:-1].encode()
    new = json.dumps({"schema": VARIABLES_SCHEMA})[1:-1].encode()
    assert data.count(new) == 1, "variables.json names its schema once"
    return data.replace(new, old)


EXPECTED: dict[str, dict[str, str]] = json.loads(DIGESTS.read_text(encoding="utf-8"))


def test_the_digests_cover_every_facility_without_inputs() -> None:
    assert sorted(EXPECTED) == sorted([*TREES, *(f"mml-{tree}" for tree in BUILT)])


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_a_facility_without_inputs_renders_the_pre_inputs_view(tmp_path: Path, name: str) -> None:
    document, facility_dir = _built(name, tmp_path)
    assert not any(model.get("inputs") for model in document["models"])

    files = _view_files(tmp_path, document, facility_dir)
    simulator = tmp_path / "render" / "data" / "simulator"
    rendered = {}
    for path in files:
        relative = path.relative_to(simulator).as_posix()
        data = _as_before_inputs(relative, path.read_bytes())
        rendered[relative] = hashlib.sha256(data).hexdigest()

    assert sorted(rendered) == sorted(EXPECTED[name])
    assert [file for file in sorted(rendered) if rendered[file] != EXPECTED[name][file]] == []
