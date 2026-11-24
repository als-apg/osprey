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

Replaced since: the four ``addresses.json`` digests, retaken when the view began
listing the IOC health addresses, which are part of the view of every served
model, and retaken again when the key that lists them, ``ioc_health``, took its
server word, ``server_health``; with the old spelling put back each file has
the previous digest; and the injection fixture's ``variables.json``, whose request record now
carries the ``command`` mark the injection engine describes it with; with that
one mark removed the file still has its pre-inputs digest; and the example
facility's ``scenarios.json``, retaken when the facility gained its
stopped-server scenario; with that one scenario removed the file still
has its pre-inputs digest, and retaken again when that scenario's name and its
``stopped_servers`` key took server words for their EPICS ones; with the old
spellings put back the file has the previous digest. ``follows.json``,
written since, is no file of the pre-inputs view and is left out; it reports
what ``variables.json`` already holds. The example facility's
``addresses.json``, ``served_models.json``, ``variables.json``,
``scenarios.json`` and ``decks/BTS.json`` were retaken, and its fifth deck
dropped, when the facility lost its synthetic transfer line, the BTS
quadrupoles gained their aperture and the stopped-server scenario moved to
``sim-BTS``; every other file of the example kept its digest. Its
``addresses.json``, ``served_models.json``, ``variables.json``,
``scenarios.json`` and ``seeds.json`` were retaken, and ``decks/BR.json``
added, when the booster gained its model and its beam readings their gates.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.views.outputs import render_facility_outputs
from osprey.facility.views.simulator import FOLLOWS_FILE, VARIABLES_SCHEMA
from osprey_facility.build import build_facility
from tests._builds import FRAMEWORK_CHECKS
from tests.facility._mml_built import BUILT, BuiltModel
from tests.facility._mml_built import TREES as MML_TREES
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


def _built(
    name: str, root: Path, mml_built: Callable[[str, str], BuiltModel]
) -> tuple[dict[str, Any], Path]:
    if name in TREES:
        facility_dir = root / "data" / "facility"
        shutil.copytree(TREES[name], facility_dir)
        return build_facility(
            facility_dir, project_name=name, extra_checks=FRAMEWORK_CHECKS
        ), facility_dir
    # An MML tree is the run's one ``build_tree`` of it; the render only reads it.
    tree = name.removeprefix("mml-")
    model = mml_built(tree, MML_TREES[tree][0])
    return model.document, model.facility


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
def test_a_facility_without_inputs_renders_the_pre_inputs_view(
    tmp_path: Path, name: str, mml_built: Callable[[str, str], BuiltModel]
) -> None:
    document, facility_dir = _built(name, tmp_path, mml_built)
    assert not any(model.get("inputs") for model in document["models"])

    files = _view_files(tmp_path, document, facility_dir)
    simulator = tmp_path / "render" / "data" / "simulator"
    rendered = {}
    for path in files:
        relative = path.relative_to(simulator).as_posix()
        if relative == FOLLOWS_FILE:
            continue
        data = _as_before_inputs(relative, path.read_bytes())
        rendered[relative] = hashlib.sha256(data).hexdigest()

    assert sorted(rendered) == sorted(EXPECTED[name])
    assert [file for file in sorted(rendered) if rendered[file] != EXPECTED[name][file]] == []
