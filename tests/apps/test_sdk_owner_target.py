"""Which machine an owner's session is on, as an application backend holding the tree reads it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.apps.sdk import owner_target
from osprey_connectors import control_context, posture_store

VA_DEPLOYMENT = {"type": "virtual_accelerator"}


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tree = tmp_path / posture_store.STATE_DIR_NAME
    tree.mkdir()
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    return tree


def _record(tree: Path, owner: str, target: str) -> Path:
    record = control_context.ControlContext(target=target, generation=3, switch={}, switch_epoch=1)
    path = tree / owner / control_context.RECORD_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record.to_payload()), encoding="utf-8")
    return path


def test_the_owners_record_names_the_machine(tree: Path) -> None:
    _record(tree, "alice", "standin")
    assert owner_target(VA_DEPLOYMENT, "alice") == "standin"


def test_each_owner_is_on_their_own_machine(tree: Path) -> None:
    _record(tree, "alice", "standin")
    _record(tree, "bob", "live")
    assert owner_target(VA_DEPLOYMENT, "bob") == "live"


@pytest.mark.usefixtures("tree")
def test_an_owner_with_no_record_has_not_switched() -> None:
    assert owner_target(VA_DEPLOYMENT, "alice") == "va"


def test_a_call_naming_nobody_is_answered_from_the_baseline(tree: Path) -> None:
    _record(tree, "alice", "standin")
    assert owner_target(VA_DEPLOYMENT, None) == "va"


def test_a_render_with_no_tree_is_answered_from_the_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    assert owner_target(VA_DEPLOYMENT, "alice") == "va"


def test_a_record_that_cannot_be_read_names_no_machine(tree: Path) -> None:
    _record(tree, "alice", "standin").write_text("{not json", encoding="utf-8")
    assert owner_target(VA_DEPLOYMENT, "alice") is None


def test_a_tree_that_is_not_provisioned_names_no_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tmp_path / "absent"))
    assert owner_target(VA_DEPLOYMENT, "alice") is None


def test_a_link_planted_under_an_owners_name_names_no_machine(tree: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    _record(elsewhere, "mallory", "live")
    (tree / "alice").symlink_to(elsewhere / "mallory")
    assert owner_target(VA_DEPLOYMENT, "alice") is None


def test_a_linked_record_names_no_machine(tree: Path, tmp_path: Path) -> None:
    planted = _record(tmp_path, "mallory", "live")
    (tree / "alice").mkdir()
    (tree / "alice" / control_context.RECORD_FILENAME).symlink_to(planted)
    assert owner_target(VA_DEPLOYMENT, "alice") is None


@pytest.mark.usefixtures("tree")
@pytest.mark.parametrize("owner", ["..", "a/b"])
def test_a_name_that_is_not_one_path_segment_names_no_machine(owner: str) -> None:
    assert owner_target(VA_DEPLOYMENT, owner) is None
