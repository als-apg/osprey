"""The facility pipeline reads a deployment's content from its content root.

A deployment whose profile mounts a machine root reads the machine facts from
that root and its own ``knowledge/``, ``ariel/`` and ``benchmarks/`` from the
content root, ``<data>/facility``. The pipeline takes the content root as the
keyword-only ``content_dir``; left out, the one tree plays both roles, as
before. Version-control metadata is never a fact of either root.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.bundled import classify
from osprey.facility.settings import (
    BENCHMARK_DATASET_PATH,
    VOCABULARY_ENABLED_KEY,
    fill_facility_settings,
)
from osprey.facility.views.benchmarks import BENCHMARK_DATASET_KEY
from osprey.facility.views.outputs import render_facility_outputs
from osprey.simulation import DEFAULT_SCENARIOS_KEY
from osprey_facility.build import build_facility_with_warnings
from osprey_facility.errors import FacilityBuildError
from osprey_facility.records import FacilityRecords
from osprey_facility.render import facility_digest
from osprey_facility.roots import DEPLOYMENT_CONTENT_DIRS
from tests._builds import FRAMEWORK_CHECKS

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "src/osprey/templates/facilities/example"
TEMPLATE_ROOT = REPO_ROOT / "src/osprey/templates"

#: A render config that asks for the benchmark view, so the content root's
#: query sources reach the render.
CONFIG: dict[str, Any] = {
    "channel_finder": {
        "pipeline_mode": "hierarchical",
        "benchmark": {"dataset_path": "data/benchmarks/queries.json"},
    }
}


def _today_digest(facility_dir: Path) -> str:
    """The digest as it was before the content root: every file of one tree."""
    digest = hashlib.sha256()
    for path in sorted(facility_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(facility_dir).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big") + relative)
        digest.update(len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


def _single(root: Path) -> Path:
    tree = root / "single" / "facility"
    shutil.copytree(EXAMPLE, tree)
    return tree


def _split(root: Path) -> tuple[Path, Path]:
    """The example as a machine root without content and a content root with only it."""
    machine = root / "machine"
    shutil.copytree(EXAMPLE, machine, ignore=shutil.ignore_patterns(*DEPLOYMENT_CONTENT_DIRS))
    content = root / "deployment" / "facility"
    content.mkdir(parents=True)
    for name in DEPLOYMENT_CONTENT_DIRS:
        if (EXAMPLE / name).is_dir():
            shutil.copytree(EXAMPLE / name, content / name)
    return machine, content


def _git(tree: Path, nested: str = "") -> Path:
    """A fake ``.git`` directory at ``tree/nested``, as a clone would leave."""
    git = tree / nested / ".git"
    (git / "objects").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    return git


def _rendered(render_dir: Path) -> dict[str, bytes]:
    return {
        path.relative_to(render_dir).as_posix(): path.read_bytes()
        for path in sorted(render_dir.rglob("*"))
        if path.is_file()
    }


# --- digest ------------------------------------------------------------------------------


def test_the_single_tree_digest_is_unchanged(tmp_path: Path) -> None:
    tree = _single(tmp_path)

    assert facility_digest(tree) == _today_digest(tree)
    assert facility_digest(tree, content_dir=tree) == _today_digest(tree)


def test_a_missing_tree_hashes_as_empty(tmp_path: Path) -> None:
    assert facility_digest(tmp_path / "gone") == hashlib.sha256().hexdigest()


def test_a_nested_git_edit_leaves_the_digest_unchanged(tmp_path: Path) -> None:
    tree = _single(tmp_path)
    before = facility_digest(tree)
    top = _git(tree)
    nested = _git(tree, "records")
    (top / "HEAD").write_text("ref: refs/heads/other\n")
    (nested / "objects" / "ab").write_bytes(b"\x00blob")
    (tree / "simulation" / "decks" / ".git").write_text("gitdir: ../../.git/modules/decks\n")

    assert facility_digest(tree) == before


def test_a_gitignore_still_moves_the_digest(tmp_path: Path) -> None:
    tree = _single(tmp_path)
    before = facility_digest(tree)
    (tree / ".gitignore").write_text("build/\n")

    assert facility_digest(tree) != before


def test_a_split_hashes_as_the_one_tree_it_splits(tmp_path: Path) -> None:
    tree = _single(tmp_path)
    machine, content = _split(tmp_path)
    _git(machine)

    assert facility_digest(machine, content_dir=content) == facility_digest(tree)


def test_mounted_content_in_the_machine_root_is_not_hashed(tmp_path: Path) -> None:
    machine, content = _split(tmp_path)
    before = facility_digest(machine, content_dir=content)
    (machine / "knowledge").mkdir()
    (machine / "knowledge" / "stale.md").write_text("# stale\n")

    assert facility_digest(machine, content_dir=content) == before


def test_mounted_content_in_the_content_root_is_hashed(tmp_path: Path) -> None:
    machine, content = _split(tmp_path)
    before = facility_digest(machine, content_dir=content)
    (content / "knowledge" / "new.md").write_text("# new\n")

    assert facility_digest(machine, content_dir=content) != before


# --- build and render --------------------------------------------------------------------


#: A knowledge page naming a device no tree holds, so a build that reads it warns.
_GONE = "---\ntype: device\ndevice_id: NO/SUCH/DEVICE\n---\n\n# Gone\n"

#: The same, naming another missing device, left in a mounting deployment's machine root.
_STALE = "---\ntype: device\ndevice_id: NO/SUCH/STALE\n---\n\n# Stale\n"


@pytest.fixture(scope="module")
def built_both_ways(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The example built and rendered as one tree and as a split, once for the module.

    Both carry the dangling page ``knowledge/gone.md`` in the root a build reads knowledge
    from: the one tree, or the split's content root. The split also leaves
    ``knowledge/stale.md``, naming another missing device, in its machine root, where a
    mounted build must not read it.
    """
    root = tmp_path_factory.mktemp("roots")
    tree = _single(root)
    machine, content = _split(root)
    (tree / "knowledge" / "gone.md").write_text(_GONE)
    (content / "knowledge" / "gone.md").write_text(_GONE)
    (machine / "knowledge").mkdir()
    (machine / "knowledge" / "stale.md").write_text(_STALE)

    one_doc, one_warnings = build_facility_with_warnings(
        tree, project_name="site", extra_checks=FRAMEWORK_CHECKS
    )
    split_doc, split_warnings = build_facility_with_warnings(
        machine, project_name="site", content_dir=content, extra_checks=FRAMEWORK_CHECKS
    )
    one_render = root / "render-one"
    split_render = root / "render-split"
    one_render.mkdir()
    split_render.mkdir()
    render_facility_outputs(one_render, one_doc, CONFIG, tree)
    render_facility_outputs(split_render, split_doc, CONFIG, machine, content_dir=content)
    return {
        "one": (one_doc, list(one_warnings), _rendered(one_render)),
        "split": (split_doc, list(split_warnings), _rendered(split_render)),
    }


def test_a_split_renders_the_same_views_as_one_tree(built_both_ways: dict[str, Any]) -> None:
    one_doc, one_warnings, one = built_both_ways["one"]
    split_doc, split_warnings, split = built_both_ways["split"]

    assert split_doc == one_doc
    assert [w.line for w in split_warnings] == [w.line for w in one_warnings]
    assert "data/benchmarks/queries.json" in one
    assert split == one


def test_a_dangling_knowledge_link_warns_when_mounted_as_in_one_tree(
    built_both_ways: dict[str, Any],
) -> None:
    _, one, _ = built_both_ways["one"]
    _, split, _ = built_both_ways["split"]

    dangling = [w.line for w in split if w.kind == "link-dangling"]
    assert any("knowledge/gone.md" in line and "NO/SUCH/DEVICE" in line for line in dangling)
    assert dangling == [w.line for w in one if w.kind == "link-dangling"]


def test_a_knowledge_page_left_in_the_machine_root_is_not_read_when_mounted(
    built_both_ways: dict[str, Any],
) -> None:
    _, warnings, _ = built_both_ways["split"]

    assert any("NO/SUCH/DEVICE" in w.line for w in warnings), "the content root was not read"
    assert not any("NO/SUCH/STALE" in w.line for w in warnings)


# --- facility settings -------------------------------------------------------------------


def _fill(
    facility_dir: Path,
    profile: dict[str, Any],
    document: dict[str, Any] | None = None,
    *,
    content_dir: Path | None = None,
) -> dict[str, Any]:
    return fill_facility_settings(
        FacilityRecords(document or {}),
        facility_dir,
        facility_dir.parent,
        {},
        profile,
        None,
        content_dir=content_dir,
    )


def test_vocabulary_and_benchmarks_are_read_from_the_content_root(tmp_path: Path) -> None:
    machine, content = _split(tmp_path)

    fill = _fill(machine, {}, content_dir=content)

    assert fill[VOCABULARY_ENABLED_KEY] is True
    assert fill[BENCHMARK_DATASET_KEY] == BENCHMARK_DATASET_PATH


def test_vocabulary_and_benchmarks_left_in_the_machine_root_are_not_read(
    tmp_path: Path,
) -> None:
    machine = tmp_path / "machine"
    shutil.copytree(EXAMPLE, machine)
    content = tmp_path / "deployment" / "facility"
    content.mkdir(parents=True)

    assert _fill(machine, {}, content_dir=content) == {}


SCENARIOS = {"scenarios": [{"name": "warm"}, {"name": "cold"}]}
MARKED = {"scenarios": [{"name": "warm", "start": True}, {"name": "cold"}]}


def test_mounted_without_marks_refuses_a_stated_start_set(tmp_path: Path) -> None:
    machine, content = _split(tmp_path)

    with pytest.raises(FacilityBuildError) as caught:
        _fill(machine, {DEFAULT_SCENARIOS_KEY: ["warm"]}, SCENARIOS, content_dir=content)

    assert caught.value.kind == "profile-invalid"
    assert caught.value.record_id == DEFAULT_SCENARIOS_KEY
    assert "mounts a machine root" in caught.value.detail
    assert f"`{DEFAULT_SCENARIOS_KEY}: null`" in caught.value.remedy


def test_mounted_without_marks_a_dotted_null_passes_and_starts_nothing(tmp_path: Path) -> None:
    machine, content = _split(tmp_path)

    fill = _fill(machine, {DEFAULT_SCENARIOS_KEY: None}, SCENARIOS, content_dir=content)

    assert DEFAULT_SCENARIOS_KEY not in fill


def test_mounted_with_marks_derives_the_start_set(tmp_path: Path) -> None:
    machine, content = _split(tmp_path)

    fill = _fill(machine, {}, MARKED, content_dir=content)

    assert fill[DEFAULT_SCENARIOS_KEY] == ["warm"]


def test_a_single_tree_without_marks_keeps_a_stated_start_set(tmp_path: Path) -> None:
    tree = tmp_path / "facility"
    tree.mkdir()

    assert DEFAULT_SCENARIOS_KEY not in _fill(tree, {DEFAULT_SCENARIOS_KEY: ["warm"]}, SCENARIOS)
    assert DEFAULT_SCENARIOS_KEY not in _fill(
        tree, {DEFAULT_SCENARIOS_KEY: ["warm"]}, SCENARIOS, content_dir=tree
    )


# --- bundled classification --------------------------------------------------------------


def test_classify_skips_version_control_metadata(tmp_path: Path) -> None:
    tree = _single(tmp_path)
    before = classify(tree, TEMPLATE_ROOT)
    _git(tree)
    _git(tree, "records")
    (tree / "simulation" / "decks" / ".git").write_text("gitdir: ../../.git/modules/decks\n")

    assert classify(tree, TEMPLATE_ROOT) == before
