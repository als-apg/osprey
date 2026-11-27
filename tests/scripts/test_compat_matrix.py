"""The per-package compatibility matrix reads the declared state and refuses an inconsistent one.

Each case runs for ``--package connectors`` and ``--package facility`` where the
rule is shared; the connectors-only parts (the simulator view, the pyaml floor)
and the facility-only parts (the optional floor, the refused subcommands, the
cutoff held to the tags) have their own tests.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = REPO_ROOT / "scripts" / "ci"
PYAML_PYPROJECT = REPO_ROOT / "packages" / "pyaml-cs-osprey" / "pyproject.toml"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # import-time required because scripts/ is not a package: the script is loaded by
    # path and registered before exec so its dataclasses resolve cls.__module__, and
    # its tables feed module-level parametrize cases.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compat = _load("compat_matrix", _SCRIPTS / "compat_matrix.py")


class Case(NamedTuple):
    """One package's constants and the tag history its rule tests run against."""

    package: Any
    #: Tags this repository carries today, as far as the package's rules read them.
    today: list[str]
    #: Today's floor: one that stands Lane B down.
    floor: str | None
    #: A floor the lane applies to, so R1 needs its tag.
    first: str
    #: The release at the cutoff: tagged, but never listable.
    pre_split: str
    #: Two releases after the cutoff, oldest first.
    older: str
    newer: str
    #: A release after the cutoff with no tag.
    untagged: str

    @property
    def dist(self) -> str:
        return self.package.dist


#: Tags this repository carries today: none after the last paired framework.
TODAY = ["v2026.6.2", "v2026.9.0b1", "v2026.9.0b2", "v2026.9.0b3", "osprey-connectors-v0.1.0"]
FIRST = "2026.10.0"
CONNECTORS_TAG = f"osprey-connectors-v{FIRST}"

CONNECTORS = Case(
    package=compat.CONNECTORS,
    today=TODAY,
    floor="2026.6.2",
    first=FIRST,
    pre_split="2026.9.0b3",
    older="2026.10.0b1",
    newer="2026.10.0",
    untagged="2026.11.0",
)
FACILITY = Case(
    package=compat.FACILITY,
    today=["v2026.9.0b3", "v2026.11.0", "v2026.11.1", "osprey-connectors-v2026.11.15"],
    floor=None,
    first="2026.12.0",
    pre_split="2026.11.1",
    older="2026.12.0b1",
    newer="2026.12.0",
    untagged="2026.12.5",
)
CASES = [pytest.param(CONNECTORS, id="connectors"), pytest.param(FACILITY, id="facility")]
BY_KEY = {"connectors": CONNECTORS, "facility": FACILITY}


def _table(case: Case, frameworks: Any, **extra: Any) -> dict[str, Any]:
    return {"tool": {case.dist: {"compat": {"frameworks": frameworks, **extra}}}}


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


@pytest.mark.parametrize("case", CASES)
def test_the_listed_frameworks_are_read_as_declared(case: Case) -> None:
    assert compat.supported_frameworks(
        case.package, _table(case, ["2026.10.0", "2026.11.0b1"])
    ) == ["2026.10.0", "2026.11.0b1"]


_MALFORMED = [
    ("undeclared", "not declared"),
    ("not-a-list", "list of version strings"),
    ("repeated", "twice"),
    ("not-a-version", "not a release version"),
]


def _malformed(case: Case, kind: str) -> dict[str, Any]:
    return {
        "undeclared": {"tool": {}},
        "not-a-list": _table(case, "2026.10.0"),
        "repeated": _table(case, ["2026.10.0", "2026.10.0"]),
        "not-a-version": _table(case, ["newest"]),
    }[kind]


@pytest.mark.parametrize(
    ("key", "kind", "message"),
    [(key, kind, message) for key in BY_KEY for kind, message in _MALFORMED],
    ids=[f"{key}-{kind}" for key in BY_KEY for kind, _ in _MALFORMED],
)
def test_a_malformed_list_is_refused(key: str, kind: str, message: str) -> None:
    case = BY_KEY[key]
    with pytest.raises(ValueError, match=message):
        compat.supported_frameworks(case.package, _malformed(case, kind))


def test_an_undeclared_table_names_the_package_table() -> None:
    with pytest.raises(ValueError, match=r"\[tool\.osprey-connectors\.compat\] frameworks"):
        compat.supported_frameworks(compat.CONNECTORS, {"tool": {}})
    with pytest.raises(ValueError, match=r"\[tool\.osprey-facility\.compat\] frameworks"):
        compat.supported_frameworks(compat.FACILITY, {"tool": {}})


@pytest.mark.parametrize("case", CASES)
def test_the_floor_is_the_one_bare_connectors_floor(case: Case) -> None:
    """For each package: the one bare ``<dist>>=`` floor among the dependencies."""
    root = _root("pyyaml>=6", f"{case.dist}>=2026.10.0", "pyaml-cs-osprey>=2026.6.2")
    assert compat.floor(case.package, root) == "2026.10.0"


def _not_one_bare_floor(dist: str) -> dict[str, tuple[str, ...]]:
    return {
        "absent": ("pyyaml>=6",),
        "ceiling": (f"{dist}>=2026.10.0,<2027",),
        "pin": (f"{dist}==2026.10.0",),
        "duplicate": (f"{dist}>=2026.10.0", f"{dist}>=2026.10.0"),
        "pin-beside-a-floor": (f"{dist}>=2026.10.0", f"{dist}==2026.10.0"),
        "marker": (f"{dist}>=2026.10.0; python_version >= '3.11'",),
    }


_FORMS = list(_not_one_bare_floor("x"))


@pytest.mark.parametrize(
    ("key", "form"),
    [(key, form) for key in BY_KEY for form in _FORMS],
    ids=[f"{key}-{form}" for key in BY_KEY for form in _FORMS],
)
def test_a_floor_that_is_not_one_bare_floor_is_refused(key: str, form: str) -> None:
    case = BY_KEY[key]
    dependencies = _not_one_bare_floor(case.dist)[form]
    with pytest.raises(ValueError, match="expected one"):
        compat.floor(case.package, _root(*dependencies))


def test_only_an_optional_floor_stands_down_on_a_bare_dependency() -> None:
    assert compat.CONNECTORS.floor_optional is False
    assert compat.FACILITY.floor_optional is True
    assert compat.floor(compat.FACILITY, _root("pyyaml>=6", "osprey-facility")) is None
    with pytest.raises(ValueError, match="expected one 'osprey-connectors>='"):
        compat.floor(compat.CONNECTORS, _root("pyyaml>=6", "osprey-connectors"))


@pytest.mark.parametrize(
    ("key", "floor", "applicable"),
    [
        ("connectors", "2026.6.2", False),
        ("connectors", "2026.9.0b3", False),
        ("connectors", "2026.10.0", True),
        ("connectors", "2026.11.1", True),
        ("connectors", None, False),
        ("facility", "2026.6.2", True),
        ("facility", "2026.12.0", True),
        ("facility", None, False),
    ],
    ids=lambda value: str(value),
)
def test_the_floor_lane_applies_from_the_first_independent_release(
    key: str, floor: str | None, applicable: bool
) -> None:
    """Connectors from its first independent release; facility whenever a floor exists."""
    assert compat.floor_applicable(BY_KEY[key].package, floor) is applicable


@pytest.mark.parametrize("case", CASES)
def test_an_empty_list_is_one_empty_cell_that_says_why(case: Case) -> None:
    """GitHub refuses a matrix with no cell; the empty one is the stand-down."""
    reason = compat.no_framework_listed(case.package)
    assert compat.matrix(case.package, []) == {"include": [{"framework": "", "reason": reason}]}
    assert f"[tool.{case.dist}.compat] frameworks" in reason
    assert f"packages/{case.dist}/pyproject.toml" in reason


@pytest.mark.parametrize("case", CASES)
def test_a_listed_release_is_one_cell_each(case: Case) -> None:
    assert compat.matrix(case.package, ["2026.10.0", "2026.11.0"]) == {
        "include": [{"framework": "2026.10.0"}, {"framework": "2026.11.0"}]
    }


@pytest.mark.parametrize("case", CASES)
def test_retired_is_optional_and_read_as_declared(case: Case) -> None:
    assert compat.retired_frameworks(case.package, _table(case, [])) == []
    assert compat.retired_frameworks(case.package, _table(case, [], retired=["2026.10.0"])) == [
        "2026.10.0"
    ]
    with pytest.raises(ValueError, match="twice"):
        compat.retired_frameworks(
            case.package, _table(case, [], retired=["2026.10.0", "2026.10.0"])
        )


# -- the rules: R1 the floor, R2 every superseded release accounted for, R3 the
# lists name tagged releases only and are disjoint ------------------------------

_DEFAULT = object()


def _problems(
    case: Case,
    frameworks: list[str] | None = None,
    retired: list[str] | None = None,
    floor: Any = _DEFAULT,
    tags: Any = _DEFAULT,
    **kwargs: Any,
) -> list[str]:
    return compat.problems(
        case.package,
        frameworks or [],
        retired or [],
        case.floor if floor is _DEFAULT else floor,
        case.today if tags is _DEFAULT else tags,
        **kwargs,
    )


def _tag(case: Case, version: str) -> str:
    return f"{case.package.tag_prefix}{version}"


@pytest.mark.parametrize(
    ("key", "first_independent", "cutoff"),
    [("connectors", FIRST, "2026.9.0b3"), ("facility", None, "2026.11.1")],
    ids=["connectors", "facility"],
)
def test_the_constants_name_the_cut_over(
    key: str, first_independent: str | None, cutoff: str
) -> None:
    package = BY_KEY[key].package
    assert package.first_independent == first_independent
    assert package.cutoff == cutoff
    assert compat.FIRST_INDEPENDENT_CONNECTORS == FIRST
    assert compat.LAST_PAIRED_FRAMEWORK == "2026.9.0b3"
    assert compat.LAST_FRAMEWORK_WITHOUT == "2026.11.1"
    assert package.tag_prefix == f"{package.dist}-v"


@pytest.mark.parametrize("case", CASES)
def test_todays_state_is_consistent(case: Case) -> None:
    assert _problems(case) == []


@pytest.mark.parametrize("case", CASES)
def test_a_floor_at_the_first_independent_release_needs_its_tag(case: Case) -> None:
    assert any("cut and push its tag first" in p for p in _problems(case, floor=case.first))
    assert any(f"no tag {_tag(case, case.first)}" in p for p in _problems(case, floor=case.first))
    assert _problems(case, floor=case.first, tags=[*case.today, _tag(case, case.first)]) == []


@pytest.mark.parametrize("floor", ["2026.10.0b1", "2026.10.0rc1", "2026.10.0.dev3"])
@pytest.mark.parametrize("case", CASES)
def test_a_pre_release_floor_is_refused(case: Case, floor: str) -> None:
    assert any("pre-release" in p for p in _problems(case, floor=floor, tags=None))


@pytest.mark.parametrize("case", CASES)
def test_a_stood_down_floor_leaves_r1_nothing_to_hold(case: Case) -> None:
    assert _problems(case, floor=None) == []


@pytest.mark.parametrize("case", CASES)
def test_the_newest_tagged_release_may_be_unlisted(case: Case) -> None:
    """Its publish and the PR that lists it follow its tag."""
    assert _problems(case, tags=[*case.today, f"v{case.older}"]) == []


@pytest.mark.parametrize("case", CASES)
def test_a_superseded_release_must_be_listed_or_retired(case: Case) -> None:
    """R2, unchanged for facility: a framework tagged after the cutoff and followed by a
    newer one fails while it is neither listed nor retired."""
    tags = [*case.today, f"v{case.older}", f"v{case.newer}"]
    found = _problems(case, tags=tags)
    assert any(f"{case.older} is tagged and a newer" in p for p in found)
    assert any(f"[tool.{case.dist}.compat] frameworks nor retired" in p for p in found)
    assert _problems(case, [case.older], tags=tags) == []
    assert _problems(case, retired=[case.older], tags=tags) == []


@pytest.mark.parametrize("case", CASES)
def test_a_pre_split_release_cannot_be_listed(case: Case) -> None:
    found = _problems(case, [case.pre_split])
    assert any(f"{case.pre_split} is in [tool.{case.dist}.compat] frameworks" in p for p in found)


@pytest.mark.parametrize("case", CASES)
def test_an_untagged_release_cannot_be_listed_or_retired(case: Case) -> None:
    assert _problems(case, [case.untagged])
    assert _problems(case, retired=[case.untagged])


@pytest.mark.parametrize("case", CASES)
def test_a_release_both_listed_and_retired_is_refused(case: Case) -> None:
    tags = [*case.today, f"v{case.newer}"]
    assert any(
        "both listed and retired" in p
        for p in _problems(case, [case.newer], [case.newer], tags=tags)
    )
    assert any(
        "both listed and retired" in p
        for p in _problems(case, [case.newer], [case.newer], tags=None)
    )


@pytest.mark.parametrize("case", CASES)
def test_before_tagging_treats_the_named_tag_as_the_newest(case: Case) -> None:
    tags = [*case.today, f"v{case.older}"]
    assert _problems(case, tags=tags) == []
    assert any(
        f"{case.older} is tagged" in p
        for p in _problems(case, tags=tags, before_tagging=f"v{case.newer}")
    )
    assert _problems(case, [case.older], tags=tags, before_tagging=f"v{case.newer}") == []
    assert _problems(case, tags=case.today, before_tagging=f"v{case.older}") == []


@pytest.mark.parametrize("tag", ["2026.10.0", "v-latest", "osprey-connectors-v2026.10.0"])
@pytest.mark.parametrize("case", CASES)
def test_before_tagging_needs_a_framework_tag(case: Case, tag: str) -> None:
    with pytest.raises(ValueError, match="not a framework release tag"):
        _problems(case, before_tagging=tag)


@pytest.mark.parametrize("case", CASES)
def test_unknown_tags_skip_only_the_rules_that_read_them(case: Case) -> None:
    assert _problems(case, [case.pre_split], floor=case.first, tags=None) == []


@pytest.mark.parametrize(
    ("key", "tags", "expected"),
    [
        (
            "connectors",
            ["v2026.9.0b3", "v2026.10.0", "vnext", "v2026.10.0-rc", "osprey-connectors-v2026.11.0"],
            {"2026.10.0"},
        ),
        (
            "facility",
            ["v2026.11.1", "v2026.12.0", "vnext", "v2026.12.0-rc", "osprey-facility-v2026.12.1"],
            {"2026.12.0"},
        ),
    ],
    ids=["connectors", "facility"],
)
def test_tags_that_are_not_framework_releases_are_ignored(
    key: str, tags: list[str], expected: set[str]
) -> None:
    """Only ``v``-tags of release versions after the package's cutoff count."""
    assert compat.framework_releases(BY_KEY[key].package, tags) == expected


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


def _framework_tags_newest_first() -> list[str]:
    listed = subprocess.run(
        ["git", "tag", "--list", "v*"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.split()
    releases = [tag for tag in listed if compat._VERSION_RE.match(tag[1:])]
    return sorted(releases, key=lambda tag: compat.version_key(tag[1:]), reverse=True)


def _depends_on_facility(tag: str) -> bool:
    text = subprocess.run(
        ["git", "show", f"{tag}:pyproject.toml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    dependencies = tomllib.loads(text).get("project", {}).get("dependencies", [])
    return any(
        compat._normalized(match["name"]) == "osprey-facility"
        for requirement in dependencies
        if (match := compat._REQUIREMENT_RE.match(requirement.strip()))
    )


def test_the_facility_cutoff_is_the_newest_framework_tag_without_the_package() -> None:
    """``LAST_FRAMEWORK_WITHOUT`` is read off the tags, not chosen: the newest ``v``-tag
    whose root ``[project].dependencies`` has no osprey-facility."""
    tags = _framework_tags_newest_first()
    assert f"v{compat.LAST_FRAMEWORK_WITHOUT}" in tags, (
        "the framework tags are not visible here: check out with full history"
    )
    without = next(tag for tag in tags if not _depends_on_facility(tag))
    assert without == f"v{compat.LAST_FRAMEWORK_WITHOUT}"
    assert compat.FACILITY.cutoff == compat.LAST_FRAMEWORK_WITHOUT


# -- what the lanes read beside the matrix ---------------------------------------


@pytest.mark.parametrize(
    ("framework", "required"),
    [("2026.9.0b2", False), ("2026.9.0b3", False), ("2026.10.0b1", True), ("2026.10.0", True)],
)
def test_every_release_that_can_be_listed_must_write_the_view(
    framework: str, required: bool
) -> None:
    assert compat.view_required(framework) is required


@pytest.mark.parametrize("case", CASES)
def test_every_source_text_test_names_a_test_at_head(case: Case) -> None:
    """A deselect entry that matches nothing would deselect nothing, silently."""
    for node_id in compat.source_text_tests(case.package):
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
@pytest.mark.parametrize("case", CASES)
def test_a_malformed_source_text_test_is_refused(case: Case, entry: Any) -> None:
    with pytest.raises(ValueError, match="node ids"):
        compat.source_text_tests(case.package, _table(case, [], **{"source-text-tests": [entry]}))


def test_the_pyaml_backend_declares_the_framework_floor() -> None:
    """pyaml-cs-osprey imports the connectors too, so it states the same floor."""
    with PYAML_PYPROJECT.open("rb") as handle:
        member = tomllib.load(handle)
    assert compat.floor(compat.CONNECTORS, member) == compat.floor(compat.CONNECTORS)


def _facility_connectors_floor_problems(root: Path) -> list[str]:
    """``framework-at-facility-floor`` installs the root's connectors floor beside
    the facility floor, so the facility must state that same connectors floor."""
    with compat.FACILITY.pyproject(root).open("rb") as handle:
        facility = tomllib.load(handle)
    stated = compat.floor(compat.CONNECTORS, facility)
    root_floor = compat.floor(compat.CONNECTORS, root=root)
    if stated != root_floor:
        return [f"osprey-facility states connectors floor {stated}, the root {root_floor}"]
    return []


def test_the_facility_declares_the_root_connectors_floor() -> None:
    assert _facility_connectors_floor_problems(compat.REPO_ROOT) == []


def test_the_facility_declares_the_root_connectors_floor__mutation_a_scratch_root_differs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    (root / compat.FACILITY.package_dir).mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "scratch"\n'
        'dependencies = ["osprey-connectors>=2026.11.15", "osprey-facility"]\n'
    )
    (compat.FACILITY.pyproject(root)).write_text(
        '[project]\nname = "osprey-facility"\ndependencies = ["osprey-connectors>=2026.10.0"]\n'
    )
    assert _facility_connectors_floor_problems(root) == [
        "osprey-facility states connectors floor 2026.10.0, the root 2026.11.15"
    ]


def test_the_facility_window_is_empty_at_head() -> None:
    """No framework release depends on a released osprey-facility yet; HEAD
    declares the floor the next one will."""
    assert compat.supported_frameworks(compat.FACILITY) == []
    assert compat.retired_frameworks(compat.FACILITY) == []
    assert compat.source_text_tests(compat.FACILITY) == []
    assert compat.floor(compat.FACILITY) is not None


# -- the command line -------------------------------------------------------------


def _head_tags(case: Case) -> Any:
    """``today`` plus the tags ``main`` reads: the real floor's and each listed framework's."""

    def tags(_repo: Path = REPO_ROOT) -> list[str]:
        listed = [
            *compat.supported_frameworks(case.package),
            *compat.retired_frameworks(case.package),
        ]
        declared = compat.floor(case.package)
        return [
            *case.today,
            *([_tag(case, declared)] if declared else []),
            *(f"v{framework}" for framework in listed),
        ]

    return tags


@pytest.mark.parametrize("case", CASES)
def test_the_plan_job_output_names_every_key(
    case: Case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(compat, "reachable_tags", _head_tags(case))
    assert compat.main(["--package", case.package.key, "github-output"]) == 0
    pairs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert set(pairs) == {"matrix", "floor", "floor-applicable"}
    frameworks = compat.supported_frameworks(case.package)
    assert json.loads(pairs["matrix"]) == compat.matrix(case.package, frameworks)
    assert pairs["floor"] == (compat.floor(case.package) or "")
    err = capsys.readouterr().err
    if not frameworks:
        assert compat.no_framework_listed(case.package) in err


@pytest.mark.parametrize("case", CASES)
def test_a_dispatched_framework_is_the_one_cell(
    case: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(compat, "reachable_tags", _head_tags(case))
    assert (
        compat.main(["--package", case.package.key, "github-output", "--framework", "2026.9.0b3"])
        == 0
    )
    pairs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert json.loads(pairs["matrix"]) == {"include": [{"framework": "2026.9.0b3"}]}


@pytest.mark.parametrize("case", CASES)
def test_an_inconsistent_state_fails_every_command(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compat, "reachable_tags", lambda repo=REPO_ROOT: case.today)
    monkeypatch.setattr(compat, "floor", lambda package, document=None, root=REPO_ROOT: case.first)
    assert compat.main(["--package", case.package.key, "matrix"]) == 1
    commands = [["floor"], ["check"], ["github-output"]]
    if not case.package.refused:
        commands += [["view-required", "2026.10.0"], ["source-text-tests"]]
    for command in commands:
        assert compat.main(["--package", case.package.key, *command]) == 1, command


def test_the_package_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        compat.main(["floor"])
    assert raised.value.code == 2
    assert "--package" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["view-required", "2026.12.0"], "view-required is connectors-only"),
        (["source-text-tests"], "source-text-tests is connectors-only"),
    ],
    ids=["view-required", "source-text-tests"],
)
def test_facility_refuses_the_connectors_only_subcommands(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert compat.main(["--package", "facility", *argv]) != 0
    assert message in capsys.readouterr().err


# -- a scratch tree through --root ------------------------------------------------

_GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]


def _toml_list(values: list[str]) -> str:
    return json.dumps(values)


def _scratch_root(
    tmp_path: Path,
    dependencies: list[str],
    *,
    facility: list[str] | None = None,
    facility_retired: list[str] | None = None,
    connectors: list[str] | None = None,
    tags: tuple[str, ...] | list[str] = (),
) -> Path:
    """A repository holding the root and both packages' pyprojects, tagged ``tags``."""
    root = tmp_path / "root"
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "scratch"\ndependencies = {_toml_list(dependencies)}\n'
    )
    for dist, listed, retired in (
        ("osprey-facility", facility or [], facility_retired or []),
        ("osprey-connectors", connectors or [], []),
    ):
        package = root / "packages" / dist
        package.mkdir(parents=True)
        (package / "pyproject.toml").write_text(
            f"[tool.{dist}.compat]\nframeworks = {_toml_list(listed)}\n"
            f"retired = {_toml_list(retired)}\n"
        )
    subprocess.run([*_GIT, "init", "-q", str(root)], check=True, timeout=60)
    subprocess.run([*_GIT, "-C", str(root), "add", "."], check=True, timeout=60)
    subprocess.run(
        [*_GIT, "-C", str(root), "commit", "-q", "-m", "scratch"], check=True, timeout=60
    )
    for tag in tags:
        subprocess.run([*_GIT, "-C", str(root), "tag", tag], check=True, timeout=60)
    return root


_HEAD_DEPENDENCIES = ["osprey-connectors>=2026.11.15", "osprey-facility", "pyyaml>=6"]
_HEAD_TAGS = ["v2026.11.0", "v2026.11.1", "osprey-connectors-v2026.11.15"]


def test_an_empty_facility_window_stands_down_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    root = _scratch_root(tmp_path, _HEAD_DEPENDENCIES, tags=_HEAD_TAGS)
    assert compat.main(["--package", "facility", "--root", str(root), "github-output"]) == 0
    captured = capsys.readouterr()
    pairs = dict(line.split("=", 1) for line in captured.out.splitlines())
    reason = compat.no_framework_listed(compat.FACILITY)
    assert json.loads(pairs["matrix"]) == {"include": [{"framework": "", "reason": reason}]}
    assert pairs["floor-applicable"] == "false"
    assert reason in captured.err
    assert compat.floor_stand_down(compat.FACILITY) in captured.err


def test_with_no_facility_floor_floor_prints_nothing_and_says_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _scratch_root(tmp_path, _HEAD_DEPENDENCIES, tags=_HEAD_TAGS)
    assert compat.main(["--package", "facility", "--root", str(root), "floor"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "the floor lane stands down" in captured.err


def test_a_declared_facility_floor_is_printed_once_its_tag_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dependencies = ["osprey-connectors>=2026.11.15", "osprey-facility>=2026.12.0"]
    untagged = _scratch_root(tmp_path / "a", dependencies, tags=_HEAD_TAGS)
    assert compat.main(["--package", "facility", "--root", str(untagged), "floor"]) == 1
    assert "no tag osprey-facility-v2026.12.0" in capsys.readouterr().err
    tagged = _scratch_root(
        tmp_path / "b", dependencies, tags=[*_HEAD_TAGS, "osprey-facility-v2026.12.0"]
    )
    assert compat.main(["--package", "facility", "--root", str(tagged), "floor"]) == 0
    assert capsys.readouterr().out == "2026.12.0\n"


_CLI_FORMS = ["pin", "ceiling", "duplicate"]


@pytest.mark.parametrize(
    ("key", "form"),
    [(key, form) for key in BY_KEY for form in _CLI_FORMS],
    ids=[f"{key}-{form}" for key in BY_KEY for form in _CLI_FORMS],
)
def test_a_floor_that_is_not_one_bare_floor_fails_the_command(
    key: str, form: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = BY_KEY[key]
    other = [d for d in _HEAD_DEPENDENCIES if not d.startswith(case.dist)]
    dependencies = [*other, *_not_one_bare_floor(case.dist)[form]]
    root = _scratch_root(tmp_path, dependencies, tags=_HEAD_TAGS)
    assert compat.main(["--package", key, "--root", str(root), "floor"]) != 0
    assert "expected one" in capsys.readouterr().err


def test_a_connectors_floor_removed_from_the_root_fails_the_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The connectors floor is required: its absence is an error, never a stand-down."""
    root = _scratch_root(tmp_path, ["osprey-facility", "pyyaml>=6"], tags=_HEAD_TAGS)
    assert compat.main(["--package", "connectors", "--root", str(root), "floor"]) != 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "expected one 'osprey-connectors>=' dependency, found 0" in captured.err


def test_a_facility_release_after_the_cutoff_must_be_listed_or_retired(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tags = [*_HEAD_TAGS, "v2026.12.0", "v2026.12.1"]
    unlisted = _scratch_root(tmp_path / "a", _HEAD_DEPENDENCIES, tags=tags)
    assert compat.main(["--package", "facility", "--root", str(unlisted), "check"]) == 1
    assert "framework 2026.12.0 is tagged and a newer release follows it" in (
        capsys.readouterr().err
    )
    retired = _scratch_root(
        tmp_path / "b", _HEAD_DEPENDENCIES, facility_retired=["2026.12.0"], tags=tags
    )
    assert compat.main(["--package", "facility", "--root", str(retired), "check"]) == 0
