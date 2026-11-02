"""The connectors compatibility matrix reads the declared state and refuses an inconsistent one."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = REPO_ROOT / "scripts" / "ci" / "connectors_compat_matrix.py"
PYAML_PYPROJECT = REPO_ROOT / "packages" / "pyaml-cs-osprey" / "pyproject.toml"
_spec = importlib.util.spec_from_file_location("connectors_compat_matrix", _MODULE_PATH)
compat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compat)


def _connectors(frameworks: Any, **extra: Any) -> dict[str, Any]:
    return {"tool": {"osprey-connectors": {"compat": {"frameworks": frameworks, **extra}}}}


def _root(*dependencies: str) -> dict[str, Any]:
    return {"project": {"dependencies": list(dependencies)}}


@pytest.mark.parametrize(
    ("lower", "higher"),
    [
        ("2026.6.2", "2026.10.0"),
        ("2026.9.0b3", "2026.9.0"),
        ("2026.9.0b3", "2026.9.0b3.post3499"),
        ("2026.9.0b3.post3499", "2026.9.0rc1"),
        ("2026.10.0.dev1", "2026.10.0a1"),
        ("2026.10.0", "2026.10.0.post1"),
        ("2026.9.0", "2026.9.1"),
    ],
)
def test_versions_order_as_pep_440_orders_them(lower: str, higher: str) -> None:
    assert compat.version_key(lower) < compat.version_key(higher)


@pytest.mark.parametrize("version", ["latest", "2026.9.0-beta", "v2026.9.0", ""])
def test_a_non_release_version_is_refused(version: str) -> None:
    with pytest.raises(ValueError, match="not a release version"):
        compat.version_key(version)


def test_the_listed_frameworks_are_read_as_declared() -> None:
    assert compat.supported_frameworks(_connectors(["2026.10.0", "2026.11.0b1"])) == [
        "2026.10.0",
        "2026.11.0b1",
    ]


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"tool": {}}, "not declared"),
        (_connectors("2026.10.0"), "list of version strings"),
        (_connectors(["2026.10.0", "2026.10.0"]), "twice"),
        (_connectors(["newest"]), "not a release version"),
    ],
)
def test_a_malformed_list_is_refused(document: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compat.supported_frameworks(document)


def test_the_floor_is_the_one_bare_connectors_floor() -> None:
    root = _root("pyyaml>=6", "osprey-connectors>=2026.10.0", "pyaml-cs-osprey>=2026.6.2")
    assert compat.connectors_floor(root) == "2026.10.0"


@pytest.mark.parametrize(
    "dependencies",
    [("pyyaml>=6",), ("osprey-connectors>=2026.10.0,<2027",), ("osprey-connectors==2026.10.0",)],
)
def test_a_floor_that_is_not_one_bare_floor_is_refused(dependencies: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="expected one"):
        compat.connectors_floor(_root(*dependencies))


@pytest.mark.parametrize(
    ("floor", "applicable"),
    [("2026.6.2", False), ("2026.9.0b3", False), ("2026.10.0", True), ("2026.11.1", True)],
)
def test_the_floor_lane_applies_from_the_first_independent_release(
    floor: str, applicable: bool
) -> None:
    assert compat.floor_applicable(floor) is applicable


def test_an_empty_list_is_one_empty_cell_that_says_why() -> None:
    """GitHub refuses a matrix with no cell; the empty one is the stand-down."""
    assert compat.matrix([]) == {
        "include": [{"framework": "", "reason": compat.NO_FRAMEWORK_LISTED}]
    }


def test_a_listed_release_is_one_cell_each() -> None:
    assert compat.matrix(["2026.10.0", "2026.11.0"]) == {
        "include": [{"framework": "2026.10.0"}, {"framework": "2026.11.0"}]
    }


def test_retired_is_optional_and_read_as_declared() -> None:
    assert compat.retired_frameworks(_connectors([])) == []
    assert compat.retired_frameworks(_connectors([], retired=["2026.10.0"])) == ["2026.10.0"]
    with pytest.raises(ValueError, match="twice"):
        compat.retired_frameworks(_connectors([], retired=["2026.10.0", "2026.10.0"]))


# -- the rules: R1 the floor, R2 every superseded release accounted for, R3 the
# lists name tagged releases only and are disjoint ------------------------------

FIRST = "2026.10.0"
CONNECTORS_TAG = f"osprey-connectors-v{FIRST}"
#: Tags this repository carries today: none after the last paired framework.
TODAY = ["v2026.6.2", "v2026.9.0b1", "v2026.9.0b2", "v2026.9.0b3", "osprey-connectors-v0.1.0"]


def _problems(
    frameworks: list[str] | None = None,
    retired: list[str] | None = None,
    floor: str = "2026.6.2",
    tags: list[str] | None = TODAY,
    **kwargs: Any,
) -> list[str]:
    return compat.problems(frameworks or [], retired or [], floor, tags, **kwargs)


def test_the_constants_name_the_cut_over() -> None:
    assert compat.FIRST_INDEPENDENT_CONNECTORS == FIRST
    assert compat.LAST_PAIRED_FRAMEWORK == "2026.9.0b3"


def test_todays_state_is_consistent() -> None:
    assert _problems() == []


def test_a_floor_at_the_first_independent_release_needs_its_tag() -> None:
    assert any("cut and push its tag first" in p for p in _problems(floor=FIRST))
    assert _problems(floor=FIRST, tags=[*TODAY, CONNECTORS_TAG]) == []


@pytest.mark.parametrize("floor", ["2026.10.0b1", "2026.10.0rc1", "2026.10.0.dev3"])
def test_a_pre_release_floor_is_refused(floor: str) -> None:
    assert any("pre-release" in p for p in _problems(floor=floor, tags=None))


def test_the_newest_tagged_release_may_be_unlisted() -> None:
    """Its publish and the PR that lists it follow its tag."""
    assert _problems(tags=[*TODAY, "v2026.10.0b1"]) == []


def test_a_superseded_release_must_be_listed_or_retired() -> None:
    tags = [*TODAY, "v2026.10.0b1", "v2026.10.0"]
    assert any("2026.10.0b1 is tagged and a newer" in p for p in _problems(tags=tags))
    assert _problems(["2026.10.0b1"], tags=tags) == []
    assert _problems(retired=["2026.10.0b1"], tags=tags) == []


def test_a_pre_split_release_cannot_be_listed() -> None:
    found = _problems(["2026.9.0b3"])
    assert any("2026.9.0b3 is in [tool.osprey-connectors.compat] frameworks" in p for p in found)


def test_an_untagged_release_cannot_be_listed_or_retired() -> None:
    assert _problems(["2026.11.0"])
    assert _problems(retired=["2026.11.0"])


def test_a_release_both_listed_and_retired_is_refused() -> None:
    tags = [*TODAY, "v2026.10.0"]
    assert any(
        "both listed and retired" in p for p in _problems(["2026.10.0"], ["2026.10.0"], tags=tags)
    )
    assert any(
        "both listed and retired" in p for p in _problems(["2026.10.0"], ["2026.10.0"], tags=None)
    )


def test_before_tagging_treats_the_named_tag_as_the_newest() -> None:
    tags = [*TODAY, "v2026.10.0b1"]
    assert _problems(tags=tags) == []
    assert any(
        "2026.10.0b1 is tagged" in p for p in _problems(tags=tags, before_tagging="v2026.10.0")
    )
    assert _problems(["2026.10.0b1"], tags=tags, before_tagging="v2026.10.0") == []
    assert _problems(tags=TODAY, before_tagging="v2026.10.0b1") == []


@pytest.mark.parametrize("tag", ["2026.10.0", "v-latest", "osprey-connectors-v2026.10.0"])
def test_before_tagging_needs_a_framework_tag(tag: str) -> None:
    with pytest.raises(ValueError, match="not a framework release tag"):
        _problems(before_tagging=tag)


def test_unknown_tags_skip_only_the_rules_that_read_them() -> None:
    assert _problems(["2026.9.0b3"], floor=FIRST, tags=None) == []


def test_tags_that_are_not_framework_releases_are_ignored() -> None:
    assert compat.framework_releases(
        ["v2026.9.0b3", "v2026.10.0", "vnext", "v2026.10.0-rc", "osprey-connectors-v2026.11.0"]
    ) == {"2026.10.0"}


def test_a_shallow_checkout_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "source"
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "init", "-q", str(source)], check=True, timeout=60)
    for message in ("one", "two"):
        subprocess.run(
            [*git, "-C", str(source), "commit", "-q", "--allow-empty", "-m", message],
            check=True,
            timeout=60,
        )
    subprocess.run([*git, "-C", str(source), "tag", "v2026.10.0"], check=True, timeout=60)
    assert compat.reachable_tags(source) == ["v2026.10.0"]
    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{source}", str(shallow)],
        check=True,
        timeout=60,
    )
    with pytest.raises(ValueError, match="shallow"):
        compat.reachable_tags(shallow)


# -- what the lanes read beside the matrix ---------------------------------------


@pytest.mark.parametrize(
    ("framework", "required"),
    [("2026.9.0b2", False), ("2026.9.0b3", False), ("2026.10.0b1", True), ("2026.10.0", True)],
)
def test_every_release_that_can_be_listed_must_write_the_view(
    framework: str, required: bool
) -> None:
    assert compat.view_required(framework) is required


def test_every_source_text_test_names_a_test_at_head() -> None:
    """A deselect entry that matches nothing would deselect nothing, silently."""
    for node_id in compat.source_text_tests():
        path, _, name = node_id.partition("::")
        assert (REPO_ROOT / path).is_file(), node_id
        if name:
            tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
            defined = {
                node.name
                for node in tree.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            }
            assert name in defined, node_id


@pytest.mark.parametrize("entry", ["src/osprey/x.py", "tests/a.py::", "tests/a.py::b::c", 3])
def test_a_malformed_source_text_test_is_refused(entry: Any) -> None:
    with pytest.raises(ValueError, match="node ids"):
        compat.source_text_tests(_connectors([], **{"source-text-tests": [entry]}))


def test_the_pyaml_backend_declares_the_framework_floor() -> None:
    """pyaml-cs-osprey imports the connectors too, so it states the same floor."""
    with PYAML_PYPROJECT.open("rb") as handle:
        member = tomllib.load(handle)
    assert compat.connectors_floor(member) == compat.connectors_floor()


def _today_with_the_floor_tag() -> list[str]:
    """``TODAY`` plus the tag of the repository's real floor, which ``main`` reads."""
    return [*TODAY, f"osprey-connectors-v{compat.connectors_floor()}"]


def test_the_plan_job_output_names_every_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(compat, "reachable_tags", _today_with_the_floor_tag)
    assert compat.main(["github-output"]) == 0
    pairs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert set(pairs) == {"matrix", "floor", "floor-applicable"}
    assert json.loads(pairs["matrix"]) == compat.matrix(compat.supported_frameworks())
    assert pairs["floor"] == compat.connectors_floor()
    if not compat.supported_frameworks():
        assert compat.NO_FRAMEWORK_LISTED in capsys.readouterr().err


def test_a_dispatched_framework_is_the_one_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(compat, "reachable_tags", _today_with_the_floor_tag)
    assert compat.main(["github-output", "--framework", "2026.9.0b3"]) == 0
    pairs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert json.loads(pairs["matrix"]) == {"include": [{"framework": "2026.9.0b3"}]}


def test_an_inconsistent_state_fails_every_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compat, "reachable_tags", lambda: TODAY)
    monkeypatch.setattr(compat, "connectors_floor", lambda: FIRST)
    assert compat.main(["matrix"]) == 1
