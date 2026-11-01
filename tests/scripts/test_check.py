"""Tests for the one check command and the declaration it runs from.

Three things have to hold. The declaration is well formed: every check has a
tier, every path it names exists, and every file the parallel sweep leaves out
is run by a check or a CI lane. Each check resolves to one known command, so a
CI step that names a check runs what the declaration says and nothing else.
And the command judges honestly: a range selects at least the checks it touches, a missing
toolchain is a refusal, and what did not run here is named in the verdict.

No test here runs a real check. The commands are recorded by a stand-in for the
one function that starts a process, or answered by a stub executable on PATH.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import importlib.util
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load(relative: str, name: str) -> ModuleType:
    # import-time required because scripts/ is not a package: each module is
    # loaded by path, under a name no import uses, and registered in
    # sys.modules before exec so @dataclass can resolve annotations through
    # cls.__module__.
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check = _load("scripts/check.py", "_check_by_path")
fw = check.FRAMEWORK
kit = check.kit

BY_ID = {c.id: c for c in check.CHECKS}


def _replace(monkeypatch: pytest.MonkeyPatch, check_id: str, **changes: Any) -> None:
    """The declaration with *check_id*'s record changed by *changes*, for one test."""
    changed = tuple(
        dataclasses.replace(c, **changes) if c.id == check_id else c for c in check.CHECKS
    )
    monkeypatch.setattr(check, "CHECKS", changed)


def _on_venue(monkeypatch: pytest.MonkeyPatch, here: bool) -> None:
    """This machine is, or is not, the live Channel Access venue, for one test.

    Every other declared venue answers yes, so the verdict names the live
    Channel Access venue alone, whatever platform runs the test.
    """
    for name, venue in list(check.VENUES.items()):
        answer = here if name == fw.LIVE_CA_VENUE else True
        monkeypatch.setitem(
            check.VENUES, name, dataclasses.replace(venue, probe=lambda answer=answer: answer)
        )


WT = "src/osprey/interfaces/web_terminal/static/js/"
TW = "tests/interfaces/web_terminal/"


# ---------------------------------------------------------------------------
# The declaration is well formed


def malformed(checks: tuple[Any, ...]) -> list[str]:
    """What is wrong with *checks* as a table, one line per defect."""
    problems = []
    ids = [c.id for c in checks]
    problems += [f"{name}: declared twice" for name in sorted(set(ids)) if ids.count(name) > 1]
    for c in checks:
        if c.kind not in check.KINDS:
            problems.append(f"{c.id}: unknown kind {c.kind}")
        if not c.tiers:
            problems.append(f"{c.id}: in no tier")
        if not c.tiers <= set(check.TIERS):
            problems.append(f"{c.id}: unknown tier in {sorted(c.tiers)}")
        problems += [
            f"{c.id}: unknown need {need}"
            for need in c.needs
            if need != "base" and need not in check.NEED_PROBES
        ]
        if c.venue and c.venue not in check.VENUES:
            problems.append(f"{c.id}: unknown venue {c.venue}")
        if c.report is not None and c.report not in check.REPORT_FLAGS:
            problems.append(f"{c.id}: unknown report {c.report}")
        if not c.argv and not c.builtin:
            problems.append(f"{c.id}: neither a command nor a builtin")
    return problems


def unhosted(unit: Any, checks: tuple[Any, ...], lanes: tuple[Any, ...]) -> list[str]:
    """The paths the sweep ignores that no check or lane has in its scope."""
    scopes = {c.id: c.scope for c in checks} | {lane.job: lane.scope for lane in lanes}
    return [
        path
        for path, host in unit.ignores
        if host not in scopes
        or not any(
            check.matches(path, pattern) or pattern.startswith(path + "/")
            for pattern in scopes[host]
        )
    ]


def test_the_checks_are_well_formed() -> None:
    assert malformed(check.CHECKS) == []


def test_a_check_in_no_tier_is_reported() -> None:
    tierless = dataclasses.replace(BY_ID["mypy"], id="orphan", tiers=frozenset())
    assert malformed((*check.CHECKS, tierless)) == ["orphan: in no tier"]


def test_a_duplicate_id_and_an_unknown_tier_are_reported() -> None:
    twin = dataclasses.replace(BY_ID["ruff"], tiers=frozenset({"nightly"}))
    assert malformed((*check.CHECKS, twin)) == [
        "ruff: declared twice",
        "ruff: unknown tier in ['nightly']",
    ]


def test_the_check_ids_are_the_public_names() -> None:
    assert [c.id for c in check.CHECKS] == [
        "lockfile",
        "ruff",
        "ruff-format",
        "config-keys",
        "changelog",
        "plugin-version",
        "tracked-ignored",
        "gitleaks",
        "invariant-tests",
        "mypy",
        "import-contracts",
        "private-imports",
        "tsc",
        "eslint",
        "prettier",
        "actionlint",
        "pytest",
        "pytest-pty",
        "pytest-wallclock",
        "readonly-warmups-py314",
        "live-ca",
        "vitest",
        "app-check-js",
        "dock-engine-tests",
        "dock-engine-bundle",
        "docs",
        "linkcheck",
        "package",
    ]
    assert [c.id for c in check.CHECKS if "quick" not in c.tiers] == [
        "pytest-pty",
        "pytest-wallclock",
        "readonly-warmups-py314",
        "live-ca",
        "app-check-js",
        "docs",
        "linkcheck",
        "package",
    ]


def test_the_js_check_lane_opens_the_test_it_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OSPREY_JS_CHECK", "0")  # Puts the variable back after the test.
    monkeypatch.setattr(kit, "ROOT", tmp_path)

    assert BY_ID["app-check-js"].prepare(None) == 0

    assert os.environ["OSPREY_JS_CHECK"] == "1"


#: Modules whose import is a read of the whole tests tree.
_TREE_READER_MODULES = frozenset({"tests._tree_scan"})
_GLOB_METHODS = frozenset({"glob", "rglob"})
#: Collectors that walk each root they are handed (``tests._text_sweep``).
_ROOT_WALKERS = frozenset({"text_files"})

#: A resolved expression: ("path", parts relative to the repository root) or ("str", text).
_Value = tuple[str, Any]


def _imports_a_tree_reader(tree: ast.Module, here: tuple[str, ...]) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name in _TREE_READER_MODULES for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            package = list(here[:-1])[: len(here) - node.level] if node.level else []
            base = ".".join([*package, *([node.module] if node.module else [])])
            if base in _TREE_READER_MODULES:
                return True
            if any(f"{base}.{alias.name}" in _TREE_READER_MODULES for alias in node.names):
                return True
    return False


def _resolve(node: ast.expr, env: dict[str, _Value | None], here: tuple[str, ...]) -> _Value | None:
    """The static value of *node*: a path under the repository root, a string, or None."""
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return ("str", node.value)
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Name)
            and func.id == "Path"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "__file__"
        ):
            return ("path", here)
        if isinstance(func, ast.Attribute) and func.attr == "resolve" and not node.args:
            return _resolve(func.value, env, here)
        return None
    drop = None
    inner: ast.expr | None = None
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        drop, inner = 1, node.value
    elif (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "parents"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
    ):
        drop, inner = node.slice.value + 1, node.value.value
    if drop is not None and inner is not None:
        value = _resolve(inner, env, here)
        if value is None or value[0] != "path" or len(value[1]) < drop:
            return None
        return ("path", value[1][: len(value[1]) - drop])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _resolve(node.left, env, here)
        right = _resolve(node.right, env, here)
        if left is None or right is None or left[0] != "path" or right[0] != "str":
            return None
        return ("path", (*left[1], *right[1].split("/")))
    return None


def _globs_the_tests_tree(
    node: ast.AST, env: dict[str, _Value | None], here: tuple[str, ...], roots: frozenset[str]
) -> bool:
    """Whether an expression under *node* globs the tests tree.

    A comprehension over one of *roots* binds its variable to ``"tests"`` for
    the comprehension only.
    """
    tests = ("path", ("tests",))
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        env = dict(env)
        for generator in node.generators:
            name = _root_loop_variable(generator.target, generator.iter, roots)
            if name is not None:
                env[name] = ("str", "tests")
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _GLOB_METHODS:
            if _resolve(func.value, env, here) == tests:
                return True
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in _ROOT_WALKERS and any(
            _names_the_tests_root(arg, roots)
            for arg in (*node.args, *(keyword.value for keyword in node.keywords))
        ):
            return True
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "walk"
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
            and node.args
            and _resolve(node.args[0], env, here) == tests
        ):
            return True
    return any(
        _globs_the_tests_tree(child, env, here, roots) for child in ast.iter_child_nodes(node)
    )


def _names_the_tests_root(node: ast.expr, roots: frozenset[str]) -> bool:
    """Whether *node* is one of *roots* or a literal tuple or list naming ``"tests"``."""
    if isinstance(node, ast.Name):
        return node.id in roots
    return isinstance(node, (ast.Tuple, ast.List)) and any(
        isinstance(e, ast.Constant) and e.value == "tests" for e in node.elts
    )


def _root_loop_variable(target: ast.expr, iterable: ast.expr, roots: frozenset[str]) -> str | None:
    """The name a loop over one of *roots* binds, or None for any other loop."""
    if isinstance(target, ast.Name) and isinstance(iterable, ast.Name) and iterable.id in roots:
        return target.id
    return None


def _bind(stmt: ast.stmt, env: dict[str, _Value | None], here: tuple[str, ...]) -> None:
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        env[stmt.targets[0].id] = _resolve(stmt.value, env, here)
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value:
        env[stmt.target.id] = _resolve(stmt.value, env, here)


def _scan_block(
    body: list[ast.stmt],
    env: dict[str, _Value | None],
    here: tuple[str, ...],
    roots: frozenset[str],
) -> bool:
    """Whether a statement in *body* globs the tests tree, binding names in source order.

    *roots* names the module-level string tuples that contain ``"tests"``: a
    loop over one binds its variable to ``"tests"`` for the loop only, so a
    glob of ``<repo root> / <loop variable>`` inside it reads the tree. A
    definition's decorators, defaults and bases are read in the enclosing
    scope; its body is read in a scope of its own.
    """
    for stmt in body:
        _bind(stmt, env, here)
        scoped = isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        name = (
            _root_loop_variable(stmt.target, stmt.iter, roots)
            if isinstance(stmt, (ast.For, ast.AsyncFor))
            else None
        )
        inner = dict(env) if scoped or name is not None else env
        if name is not None:
            inner[name] = ("str", "tests")
        for _field, value in ast.iter_fields(stmt):
            items = value if isinstance(value, list) else [value]
            block = [item for item in items if isinstance(item, ast.stmt)]
            if block and _scan_block(block, inner, here, roots):
                return True
            for item in items:
                if isinstance(item, ast.AST) and not isinstance(item, ast.stmt):
                    if _globs_the_tests_tree(item, env if scoped else inner, here, roots):
                        return True
    return False


#: Text without which no import names a tree-reader module: every spelling of
#: such an import, absolute or relative, writes the module's last name.
_TREE_READER_WORDS = tuple(module.rpartition(".")[2] for module in _TREE_READER_MODULES)
#: Text without which no call globs or walks a tree: each call the rule follows
#: names its method or function in the source as written.
_GLOB_WORDS = (*_GLOB_METHODS, "walk", *_ROOT_WALKERS)


def _source_reads_the_tests_tree(source: str, here: tuple[str, ...]) -> bool:
    """Whether the module at repo-relative parts *here* with text *source* reads the tests tree.

    A module whose text holds no word the rule keys on cannot match it, so it
    is passed over before the parse; parsing and walking every test module is
    what makes the whole-tree scan slow, and most modules hold none of them.
    """
    imports = any(word in source for word in _TREE_READER_WORDS)
    globs = any(word in source for word in _GLOB_WORDS)
    if not imports and not globs:
        return False
    tree = ast.parse(source)
    if imports and _imports_a_tree_reader(tree, here):
        return True
    if not globs:
        return False
    roots = frozenset(
        stmt.targets[0].id
        for stmt in tree.body
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, (ast.Tuple, ast.List))
        and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in stmt.value.elts)
        and any(e.value == "tests" for e in stmt.value.elts if isinstance(e, ast.Constant))
    )
    module_env: dict[str, _Value | None] = {}
    for stmt in tree.body:
        _bind(stmt, module_env, here)
    return _scan_block(tree.body, module_env, here, roots)


def _reads_the_tests_tree(path: Path) -> bool:
    return _source_reads_the_tests_tree(
        path.read_text(encoding="utf-8"), path.relative_to(REPO_ROOT).parts
    )


def test_every_invariant_test_exists() -> None:
    assert [p for p in fw.INVARIANT_TESTS if not (REPO_ROOT / p).is_file()] == []
    assert "tests/scripts/test_check.py" in fw.INVARIANT_TESTS


@functools.cache
def _tree_readers() -> tuple[str, ...]:
    """The test modules that read the whole tests tree, in path order.

    The scan is read once per session: the tests that ask which readers a
    listing leaves out differ only in the listing, not in what the tree holds.
    """
    return tuple(
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted((REPO_ROOT / "tests").rglob("test_*.py"))
        if "__pycache__" not in path.parts and _reads_the_tests_tree(path)
    )


def _tree_readers_outside(invariant: Sequence[str]) -> list[str]:
    """The test modules that read the whole tests tree and are not in *invariant*."""
    listed = set(invariant)
    return [relative for relative in _tree_readers() if relative not in listed]


def test_every_module_that_reads_the_tests_tree_is_an_invariant_test() -> None:
    """A test that reads the whole tests tree can break on a change anywhere in
    it, so the invariant-tests check, which no change selection leaves out,
    runs it."""
    assert _tree_readers_outside(fw.INVARIANT_TESTS) == []


def test_a_tree_reader_left_out_of_the_invariant_tests_is_named() -> None:
    reader = "tests/test_no_legacy_symbols.py"
    assert _tree_readers_outside(tuple(p for p in fw.INVARIANT_TESTS if p != reader)) == [reader]


def test_the_need_probes_of_the_new_toolchains_name_their_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The secret scan needs gitleaks on PATH and the workflow lint needs
    shellcheck in the venv: actionlint passes over every ``run:`` block without
    a word when shellcheck is absent, so its absence is a refusal."""
    monkeypatch.setattr(fw.shutil, "which", lambda name: None)
    assert "brew install gitleaks" in (fw._need_gitleaks() or "")
    monkeypatch.setattr(fw.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert fw._need_gitleaks() is None
    monkeypatch.setattr(kit, "capture", lambda argv: (127, ""))
    assert "uv sync --extra dev" in (fw._need_workflow_linters() or "")
    monkeypatch.setattr(kit, "capture", lambda argv: (0, "ShellCheck"))
    assert fw._need_workflow_linters() is None
    assert "--frozen" in fw.WORKFLOW_LINT_PROBE


_PREAMBLE = (
    "from pathlib import Path\nROOT = Path(__file__).resolve().parents[1]\n"
    'ROOTS = ("src", "tests")\n'
)


@pytest.mark.parametrize(
    ("body", "reads"),
    [
        ('def scan(files=tuple((ROOT / "tests").rglob("*.py"))):\n    return files\n', True),
        (
            "def mark(found):\n    return lambda f: f\n\n\n"
            '@mark(sorted((ROOT / "tests").glob("*.py")))\ndef scan():\n    return 1\n',
            True,
        ),
        ('FILES = [p for root in ROOTS for p in (ROOT / root).rglob("*")]\n', True),
        (
            'for root in ROOTS:\n    pass\nroot = "docs"\nFILES = list((ROOT / root).rglob("*"))\n',
            False,
        ),
        (
            "def scan(root):\n    for root in ROOTS:\n        pass\n"
            '    return list((ROOT / "docs" / root).rglob("*"))\n',
            False,
        ),
        (
            "def names():\n    for root in ROOTS:\n        print(root)\n"
            '    for root in ("docs",):\n        list((ROOT / root).rglob("*"))\n',
            False,
        ),
        ("for root in ROOTS:\n    FILES = list((ROOT / root).rglob('*'))\n", True),
        ("FILES = text_files(ROOT, ROOTS)\n", True),
        ('FILES = text_files(ROOT, roots=("docs", "tests"))\n', True),
        ('FILES = text_files(ROOT, ("src", "docs"))\n', False),
        ("from tests._tree_scan import test_modules\n", True),
        ("from . import _tree_scan\n", True),
    ],
    ids=[
        "argument-default",
        "decorator",
        "comprehension-over-roots",
        "name-rebound-after-loop",
        "other-path-after-loop",
        "other-loop-after-loop",
        "loop-over-roots",
        "walker-over-roots",
        "walker-over-literal-roots",
        "walker-elsewhere",
        "tree-reader-import",
        "relative-tree-reader-import",
    ],
)
def test_the_tree_reader_rule_follows_where_a_name_is_bound(body: str, reads: bool) -> None:
    assert _source_reads_the_tests_tree(_PREAMBLE + body, ("tests", "test_sample.py")) is reads


def test_every_literal_scope_path_exists() -> None:
    scopes = (
        *(c.scope for c in check.CHECKS),
        *(lane.scope for lane in check.LANES),
    )
    literal = [
        pattern for scope in scopes for pattern in scope if "*" not in pattern and "/" in pattern
    ]
    assert [p for p in literal if not (REPO_ROOT / p).exists()] == []


def test_every_ignored_path_exists_and_has_a_host() -> None:
    assert [p for p, _ in fw.UNIT_SELECTION.ignores if not (REPO_ROOT / p).exists()] == []
    assert unhosted(fw.UNIT_SELECTION, check.CHECKS, check.LANES) == []


def test_an_ignore_whose_host_is_gone_is_reported() -> None:
    without = tuple(lane for lane in check.LANES if lane.job != "channel-finder-benchmarks")
    assert unhosted(fw.UNIT_SELECTION, check.CHECKS, without) == [
        "tests/services/channel_finder/graph_index/test_scale.py"
    ]


def test_an_ignore_outside_its_hosts_scope_is_reported() -> None:
    moved = dataclasses.replace(fw.UNIT_SELECTION, ignores=(("tests/cli/test_main.py", "live-ca"),))
    assert unhosted(moved, check.CHECKS, check.LANES) == ["tests/cli/test_main.py"]


def workflow() -> dict[str, Any]:
    return yaml.safe_load(CI_YML.read_text(encoding="utf-8"))


def test_every_required_ci_job_is_a_caller_or_a_lane() -> None:
    jobs = workflow()["jobs"]
    lanes = [lane.job for lane in check.LANES]
    assert len(lanes) == len(set(lanes))
    assert [job for job in (*lanes, *check.CALLER_JOBS) if job not in jobs] == []
    required = jobs["all-checks-passed"]["needs"]
    assert sorted(set(required) - set(lanes) - set(check.CALLER_JOBS)) == []
    assert set(lanes) - set(required) == {"channel-finder-benchmarks"}


# ---------------------------------------------------------------------------
# Each check resolves to one known command


def run_of(job: str, containing: str) -> str:
    """The ``run:`` text of the one step of *job* that contains *containing*."""
    found = [
        step["run"]
        for step in workflow()["jobs"][job]["steps"]
        if containing in step.get("name", "") or containing in step.get("run", "")
    ]
    assert len(found) == 1, f"{job}: {len(found)} steps match {containing!r}"
    return found[0].strip()


def after_bound(argv: list[str]) -> list[str]:
    """A bounded command without its ``run_bounded.py SECONDS --`` prefix."""
    assert argv[:5] == [*kit.UV_RUN, "python", "scripts/run_bounded.py"]
    return argv[argv.index("--") + 1 :]


#: The execution flags the CI unit step passes after ``--``.
CI_UNIT_EXEC_FLAGS = [
    "--durations=50",
    "--timeout=600",
    "--timeout-method=signal",
    "-n",
    "4",
]

#: What the parallel sweep leaves out in every shard, in the order it says so.
SWEEP_IGNORES = [
    "--ignore=tests/e2e",
    "--ignore=tests/va/test_apply_fault.py",
    "--ignore=tests/va/test_lume_pva_seam.py",
    "--ignore=tests/services/channel_finder/graph_index/test_scale.py",
    "--ignore=tests/integration/test_build_boot.py",
    "--ignore=tests/integration/test_preset_static.py",
    "--ignore=tests/integration/test_preset_build.py",
]

#: The sweep each shard selects: its paths, then everything it ignores.
SHARD_SELECTIONS = {
    "cli": ["tests/cli", *SWEEP_IGNORES],
    "rest-a": ["tests/facility", "tests/mcp_server", "tests/deployment", *SWEEP_IGNORES],
    "rest-b": [
        "tests/",
        "packages/osprey-connectors/tests",
        "--ignore=tests/cli",
        "--ignore=tests/facility",
        "--ignore=tests/mcp_server",
        "--ignore=tests/deployment",
        *SWEEP_IGNORES,
    ],
}

#: The sweep's marker expression in the full tier.
SWEEP_MARKER = ["-m", "not pty and not wallclock"]


def pytest_report(check_id: str, reports: str = "{reports}") -> list[str]:
    """The flags that make a pytest check write its report into *reports*."""
    return [f"--junitxml={reports}/{check_id}/junit.xml", "-o", "junit_family=xunit1"]


#: The flags that make Vitest write its report beside its console output.
VITEST_REPORT = [
    "--reporter=default",
    "--reporter=junit",
    "--outputFile.junit={reports}/vitest/junit.xml",
]


def test_run_resolves_each_check_to_its_one_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    def run(*words: str) -> list[list[str]]:
        return check.resolve(["run", *words])

    assert run("lockfile") == [["uv", "lock", "--check"]]
    assert run("ruff-format") == [[*fw.UV_RUN, "ruff", "format", "--check", "."]]
    assert run("config-keys") == [
        [*fw.UV_RUN, "python", "scripts/check_config_keys.py", "--back-test"]
    ]
    assert run("mypy") == [[*fw.UV_RUN, "python", "scripts/mypy_gate.py"]]
    assert run("changelog", "--base", "origin/main") == [
        ["python3", "scripts/changelog_fragments.py", "check", "--base", "origin/main"]
    ]
    assert run("plugin-version", "--base", "origin/main") == [
        ["python3", "scripts/plugin_version.py", "check", "--base", "origin/main"]
    ]
    assert run("import-contracts") == [[*fw.UV_RUN, "lint-imports"]]
    private = [*fw.UV_RUN, "ruff", "check", "--preview", "--select=PLC2701", "."]
    assert run("private-imports") == [private]
    assert run("tracked-ignored") == []
    assert run("gitleaks", "--base", "origin/main") == [
        ["gitleaks", "git", "--log-opts=origin/main..HEAD", "--no-banner", "--redact"]
    ]
    assert run("invariant-tests") == [
        [*fw.UV_RUN, "pytest", *fw.INVARIANT_TESTS, *pytest_report("invariant-tests")]
    ]
    assert run("actionlint") == [[*fw.UV_RUN, "actionlint"]]

    assert run("ruff") == [[*fw.UV_RUN, "ruff", "check", "."]]
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert run("ruff") == [[*fw.UV_RUN, "ruff", "check", ".", "--output-format=github"]]
    assert run("private-imports") == [[*private, "--output-format=github"]]
    monkeypatch.delenv("GITHUB_ACTIONS")

    scripts = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))["scripts"]
    assert run("tsc") == [["npm", "run", "typecheck"]] and "typecheck" in scripts
    assert run("eslint") == [["npm", "run", "lint"]] and "lint" in scripts
    assert run("vitest") == [["npx", *shlex.split(scripts["test:js"]), *VITEST_REPORT]]
    engine_report = [flag.replace("/vitest/", "/dock-engine-tests/") for flag in VITEST_REPORT]
    assert run("dock-engine-tests") == [
        ["npx", *shlex.split(scripts["test:dock-engine"]), *engine_report]
    ]
    assert run("dock-engine-bundle") == [shlex.split(scripts["check:dock-engine"])]

    for shard, selection in SHARD_SELECTIONS.items():
        (sweep,) = run("pytest", "--shard", shard, "--", *CI_UNIT_EXEC_FLAGS)
        assert after_bound(sweep) == [
            *fw.UV_RUN,
            "pytest",
            *selection,
            *SWEEP_MARKER,
            *CI_UNIT_EXEC_FLAGS,
            *("--dist", "loadgroup"),
            *pytest_report("pytest"),
        ]
    (pty,) = run("pytest-pty", "--", "-v", "--tb=short")
    assert after_bound(pty) == [
        *fw.UV_RUN,
        "pytest",
        "tests/pty/",
        "-m",
        "pty",
        "-v",
        "--tb=short",
        *pytest_report("pytest-pty"),
    ]
    (wallclock,) = run("pytest-wallclock")
    assert after_bound(wallclock) == [
        *fw.UV_RUN,
        *("pytest", "tests/integration", "-m", "wallclock"),
        *pytest_report("pytest-wallclock"),
    ]
    assert run("live-ca") == [[*fw.UV_RUN, "python", "scripts/va/live_ca/gate.py", "--pva"]]

    assert run("docs") == [[*fw.UV_RUN, "make", "html", "SPHINXOPTS=-W --keep-going"]]
    assert run("linkcheck") == [[*fw.UV_RUN, "make", "linkcheck"]]
    assert [c.id for c in check.CHECKS if c.workdir] == ["docs", "linkcheck"]
    assert BY_ID["linkcheck"].workdir == BY_ID["docs"].workdir
    assert (REPO_ROOT / BY_ID["docs"].workdir / "Makefile").is_file()
    assert (REPO_ROOT / fw.DOCS_HTML).parent.parent == REPO_ROOT / BY_ID["docs"].workdir

    lean, pyaml = "{scratch}/lean/bin/python", "{scratch}/pyaml/bin/python"
    assert run("package") == [
        ["uv", "build"],
        ["uvx", "twine", "check", "dist/*"],
        ["uv", "build", "--package", "osprey-connectors"],
        ["uv", "venv", "{scratch}/lean", "--python", "3.11"],
        ["uv", "pip", "install", "--python", lean, "dist/osprey_connectors-*.whl"],
        [lean, "-I", fw.LEAN_IMPORT_WALK],
        ["python3", "-c", fw.DOCK_ENGINE_WHEEL_PROBE, "dist/osprey_framework-*.whl"],
        ["uv", "build", "--package", "pyaml-cs-osprey"],
        ["uv", "venv", "{scratch}/pyaml", "--python", "3.11"],
        [
            *("uv", "pip", "install", "--python", pyaml, "dist/osprey_framework-*.whl"),
            *("dist/osprey_connectors-*.whl", "dist/pyaml_cs_osprey-*.whl"),
        ],
        [pyaml, "-c", fw.PYAML_WHEEL_PROBE],
    ]


def test_a_dropped_ignore_changes_the_sweep() -> None:
    kept = fw.UNIT_SELECTION.ignores[:-1]
    narrowed = dataclasses.replace(fw.UNIT_SELECTION, ignores=kept)
    rest = SHARD_SELECTIONS["rest-b"]
    assert fw.unit_argv("full", shard="rest-b") == [*rest, *SWEEP_MARKER]
    assert fw.unit_argv("full", shard="rest-b", unit=narrowed) == [*rest[:-1], *SWEEP_MARKER]


def test_the_shards_cover_the_sweep_roots_exactly_once() -> None:
    unit = fw.UNIT_SELECTION
    tops = sorted(
        {path.as_posix() for path in (REPO_ROOT / "tests").iterdir() if path.is_dir()}
        | {(REPO_ROOT / "packages/osprey-connectors/tests").as_posix()}
    )

    def owners(top: str) -> list[str]:
        relative = Path(top).relative_to(REPO_ROOT).as_posix()
        return [
            name
            for name, (paths, ignores) in unit.shards.items()
            if any(relative == p.rstrip("/") or relative.startswith(p) for p in paths)
            and not any(relative == i or relative.startswith(i + "/") for i in ignores)
        ]

    assert {top: owners(top) for top in tops if len(owners(top)) != 1} == {}
    assert fw.unit_argv("full") == [*unit.roots, *SWEEP_IGNORES, *SWEEP_MARKER]


def test_the_shards_cover_the_sweep_roots__mutation_drops_an_ignore() -> None:
    unit = fw.UNIT_SELECTION
    paths, ignores = unit.shards["rest-b"]
    assert "tests/cli" in ignores
    kept = tuple(i for i in ignores if i != "tests/cli")
    shards = {**unit.shards, "rest-b": (paths, kept)}
    narrowed = dataclasses.replace(unit, shards=shards)
    selection = fw.unit_argv("full", shard="rest-b", unit=narrowed)
    assert "--ignore=tests/cli" not in selection


#: Files the macOS trim must keep, with what each exercises that ubuntu
#: cannot answer for. Not the whole kept set: the slow ones a later trim is
#: most likely to reach for.
MACOS_KEEPS = {
    "tests/test_offline_build_env.py": "the one real venv install (arm64 wheels)",
    "tests/connectors/ipc/test_pool.py": "child spawn, SIGKILL and replace",
    "tests/connectors/ipc/test_host_child.py": "the getppid watchdog written for macOS",
    "tests/mcp_server/test_switch_lifecycle.py": "real connector-host children",
    "tests/mcp_server/test_sandbox_guard_characterization.py": "TMPDIR and /private/var",
    "tests/services/python_executor/test_secret_zone_guard.py": "case-insensitive APFS",
    "tests/hooks/test_approval_hook.py": "symlink, hard link and FIFO matrix",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py": "symlink containment",
    "tests/interfaces/web_terminal/test_supervised_child.py": "process groups, held socket",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py": "killpg and parent death",
    "tests/deployment/test_compose_generator.py": "setgid directories, O_NOFOLLOW",
    "tests/facility/test_check_a.py": "pyAT against fixed MATLAB references",
    "tests/connectors/test_epics_ca_softioc.py": "native EPICS libraries",
    "tests/agent_runner/test_unrendered_mcp_servers.py": "the native claude binary",
    "tests/scripts/test_run_bounded.py": "killpg and signal forwarding",
    "tests/cli/test_web_cmd.py": "real socket binds",
    "tests/cli/test_build_swap.py": "os.replace directory swap, SIGKILL mid-render",
}


def test_the_macos_trim_is_the_whole_sweep_minus_its_files() -> None:
    unit = fw.UNIT_SELECTION
    trimmed = [f"--ignore={path}" for path in unit.trims["macos"]]
    assert fw.unit_argv("full", shard="macos") == [
        *unit.roots,
        *trimmed,
        *SWEEP_IGNORES,
        *SWEEP_MARKER,
    ]


def test_the_macos_trim_names_existing_test_files_the_sweep_runs() -> None:
    # pytest silently accepts an --ignore of a missing path, so a rename would
    # void the entry unnoticed; a directory would trim files nobody read.
    unit = fw.UNIT_SELECTION
    trimmed = unit.trims["macos"]
    swept = {path for path, _lane in unit.ignores}
    assert len(set(trimmed)) == len(trimmed)
    assert [p for p in trimmed if not (REPO_ROOT / p).is_file()] == []
    assert [p for p in trimmed if not Path(p).name.startswith("test_")] == []
    assert [p for p in trimmed if p in swept or p.startswith("tests/e2e/")] == []


def _trimmed_keeps(unit: Any) -> dict[str, str]:
    """The keeps the macOS selection of *unit* leaves out, with why each matters."""
    selection = fw.unit_argv("full", shard="macos", unit=unit)
    return {p: why for p, why in MACOS_KEEPS.items() if f"--ignore={p}" in selection}


def test_the_macos_trim_keeps_what_macos_can_answer() -> None:
    assert [p for p in MACOS_KEEPS if not (REPO_ROOT / p).is_file()] == []
    assert _trimmed_keeps(fw.UNIT_SELECTION) == {}


def test_the_macos_trim_keeps_what_macos_can_answer__mutation_trims_a_keep() -> None:
    unit = fw.UNIT_SELECTION
    pool = "tests/connectors/ipc/test_pool.py"
    narrowed = dataclasses.replace(unit, trims={"macos": (*unit.trims["macos"], pool)})
    assert _trimmed_keeps(narrowed) == {pool: MACOS_KEEPS[pool]}


#: Uses whose outcome can depend on the OS. Matched in the syntax tree, so a
#: comment or a string that names one is not a use.
OS_SENSITIVE_USES = frozenset(
    {
        "os.kill",
        "os.killpg",
        "os.getppid",
        "os.setsid",
        "os.setpgrp",
        "os.mkfifo",
        "os.symlink",
        "os.link",
        "sys.platform",
        "platform.system",
        "subprocess.Popen",
        "asyncio.create_subprocess_exec",
        "asyncio.create_subprocess_shell",
        "socket.socket",
        "signal.*",
        "start_new_session",
        "symlink_to",
        "hardlink_to",
    }
)

#: Trimmed files read and judged OS-independent despite these uses.
MACOS_TRIM_VETTED = {
    # A symlinked checkout resolves to one repo identity on any POSIX host.
    "tests/deployment/test_up_as_built.py": ("symlink_to",),
    # A child is killed with SIGKILL, in teardown and to make a dead child the
    # supervisor replaces; the assertions are on the envelope's frames. A dead
    # connector-host child is detected by tests/connectors/ipc/test_pool.py,
    # kept on macOS.
    "tests/mcp_server/test_error_envelope_ipc.py": ("asyncio.create_subprocess_exec", "os.kill"),
}


def _os_sensitive_uses(source: str) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            found.add(node.attr)
            if isinstance(node.value, ast.Name):
                found |= {f"{node.value.id}.{node.attr}", f"{node.value.id}.*"}
        elif isinstance(node, ast.keyword) and node.arg:
            found.add(node.arg)
    return found & OS_SENSITIVE_USES


def _unvetted_trims(trims: Sequence[str]) -> dict[str, list[str]]:
    """The trimmed files that gained an OS-sensitive use nobody has judged."""
    return {
        path: sorted(uses)
        for path in trims
        if (
            uses := _os_sensitive_uses((REPO_ROOT / path).read_text(encoding="utf-8"))
            - set(MACOS_TRIM_VETTED.get(path, ()))
        )
    }


def test_the_macos_trim_holds_no_unread_os_sensitive_use() -> None:
    # A trimmed file that gains a signal, link or platform branch would stay
    # off macOS silently: read it, then keep it on macOS or vet the use here.
    trimmed = fw.UNIT_SELECTION.trims["macos"]
    assert [p for p in MACOS_TRIM_VETTED if p not in trimmed] == []
    assert _unvetted_trims(trimmed) == {}


def test_the_macos_trim_holds_no_unread_os_sensitive_use__mutation_trims_a_keep() -> None:
    pool = "tests/connectors/ipc/test_pool.py"
    assert _unvetted_trims((pool,)) == {pool: ["os.kill", "signal.*", "subprocess.Popen"]}
    assert _os_sensitive_uses("# os.kill is not the signal here\nx = 'socket.socket'\n") == set()


def test_the_tiers_bound_their_pytest_runs() -> None:
    def bounds(tier: str) -> list[str]:
        script = "scripts/run_bounded.py"
        return [argv[argv.index(script) + 1] for argv in check.resolve([tier]) if script in argv]

    assert bounds("quick") == ["1800"]
    assert bounds("full") == ["5400", "600", "900", "900", "600"]


# ---------------------------------------------------------------------------
# Selection

FILES = [
    WT + "settings.js",
    TW + "settings.test.mjs",
    WT + "toast.js",
    "src/osprey/interfaces/health/static/js/dashboard.js",
    "src/osprey/interfaces/design_system/static/js/tokens.js",
    "src/osprey/cli/__init__.py",
    "src/osprey/cli/main.py",
    "src/osprey/cli/templates/help.txt",
    "tests/cli/test_main.py",
    "src/osprey/lonely/__init__.py",
    "src/osprey/lonely/thing.py",
    "tests/scripts/test_scenario_plots.py",
]


def selected(tier: str, *paths: str, named: tuple[str, ...] = ()) -> dict[str, Any]:
    return check.select(tier, list(paths), named, FILES).selections


#: The JavaScript module-to-test mapping arrives with the JS module splits;
#: until then Vitest cannot be narrowed and these cases have nothing to read.
needs_mapping = pytest.mark.skipif(
    not (REPO_ROOT / fw.GUARD_MODULE).is_file(),
    reason=f"{fw.GUARD_MODULE} is not in this tree",
)


def test_without_the_mapping_any_javascript_change_widens_vitest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(kit, "ROOT", tmp_path)
    assert fw._guard() is None
    got = selected("quick", WT + "settings.js")
    assert got["vitest"].state == check.WHOLE
    assert selected("quick", "src/osprey/cli/main.py")["vitest"].state == check.NOT_TOUCHED
    assert selected("quick", "package.json")["vitest"].state == check.WHOLE


def test_without_the_mapping__mutation_a_python_change_reaches_vitest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(kit, "ROOT", tmp_path)
    monkeypatch.setattr(fw, "JS_SUFFIXES", (".js", ".mjs", ".cjs", ".py"))
    assert selected("quick", "src/osprey/cli/main.py")["vitest"].state == check.WHOLE


def test_without_a_range_the_tier_runs_whole() -> None:
    plan = check.select("quick", None)
    assert set(plan.selections) == {c.id for c in check.CHECKS if "quick" in c.tiers}
    assert {s.state for s in plan.selections.values()} == {check.WHOLE}
    assert plan.lanes == ()
    assert check.select("full", None).lanes == tuple(lane.job for lane in check.LANES)


@needs_mapping
def test_a_module_selects_its_own_test() -> None:
    got = selected("quick", WT + "settings.js")
    assert got["vitest"] == check.Selection(check.PATHS, (TW + "settings.test.mjs",))
    assert got["tsc"].state == got["eslint"].state == check.WHOLE
    assert got["pytest"].state == got["mypy"].state == check.NOT_TOUCHED


@needs_mapping
def test_a_module_without_a_test_widens_to_its_interface() -> None:
    assert selected("quick", WT + "toast.js")["vitest"] == check.Selection(check.PATHS, (TW,))


def test_a_module_in_an_interface_without_tests_widens_to_every_test() -> None:
    dashboard = "src/osprey/interfaces/health/static/js/dashboard.js"
    assert selected("quick", dashboard)["vitest"].state == check.WHOLE


@pytest.mark.parametrize(
    "path",
    [
        "vitest.config.js",
        "tests/vitest.setup.mjs",
        "package.json",
        "package-lock.json",
        TW + "panel-labels-fixture.mjs",
        "src/osprey/interfaces/design_system/static/js/tokens.js",
        WT + "vendor/xterm.js",
    ],
)
def test_a_shared_javascript_input_widens_vitest_to_every_test(path: str) -> None:
    assert selected("quick", path)["vitest"].state == check.WHOLE


@needs_mapping
def test_a_changed_vitest_file_runs_itself() -> None:
    got = selected("quick", TW + "new.test.mjs", WT + "settings.js")
    assert got["vitest"].paths == (TW + "new.test.mjs", TW + "settings.test.mjs")


@needs_mapping
def test_a_same_named_test_in_another_directory_is_not_selected() -> None:
    files = [WT + "utils.js", TW + "scaffold/utils.test.mjs"]
    selections = check.select("quick", [WT + "utils.js"], (), files).selections
    assert selections["vitest"] == check.Selection(check.PATHS, (TW,))


def test_a_changed_pytest_file_runs_itself() -> None:
    got = selected("quick", "tests/cli/test_main.py")
    assert got["pytest"] == check.Selection(check.PATHS, ("tests/cli/test_main.py",))
    assert got["mypy"].state == check.WHOLE
    assert got["vitest"].state == got["tsc"].state == check.NOT_TOUCHED


def test_a_source_file_selects_its_mirrored_tests() -> None:
    for path in ("src/osprey/cli/main.py", "src/osprey/cli/templates/help.txt"):
        assert selected("quick", path)["pytest"] == check.Selection(check.PATHS, ("tests/cli",))


@pytest.mark.parametrize(
    "path",
    [
        "tests/conftest.py",
        "tests/cli/conftest.py",
        "tests/_shared_contract.py",
        "src/osprey/lonely/thing.py",
        "src/osprey/__init__.py",
        "packages/osprey-connectors/src/osprey_connectors/errors.py",
        "pyproject.toml",
        "uv.lock",
    ],
)
def test_what_the_mapping_cannot_place_widens_pytest_to_the_sweep(path: str) -> None:
    assert selected("quick", path)["pytest"].state == check.WHOLE


def test_a_helper_script_selects_only_its_mirror() -> None:
    assert selected("quick", "scripts/scenario_plots.py")["pytest"] == check.Selection(
        check.PATHS, ("tests/scripts/test_scenario_plots.py",)
    )


@pytest.mark.parametrize(
    "path", ["scripts/plugin_version.py", "scripts/run_bounded.py", "scripts/check.py"]
)
def test_a_script_a_check_runs_through_selects_the_scripts_tests(path: str) -> None:
    assert path in check.SHAPING_SCRIPTS
    assert selected("quick", path)["pytest"] == check.Selection(check.PATHS, ("tests/scripts",))


def test_a_file_under_a_tests_directory_selects_that_directory() -> None:
    fixture = "tests/cli/data/json_keysets/ariel_status.json"
    assert selected("quick", fixture)["pytest"] == check.Selection(check.PATHS, ("tests/cli",))
    assert selected("quick", "tests/README.md")["pytest"].state == check.WHOLE


def test_a_file_under_a_tests_directory_without_test_modules_widens() -> None:
    assert not any(name.startswith("tests/fixtures/") for name in FILES)
    assert selected("quick", "tests/fixtures/config.yml")["pytest"].state == check.WHOLE


def test_a_script_without_a_mirror_widens() -> None:
    assert selected("quick", "scripts/_tier_db_common.py")["pytest"].state == check.WHOLE


def test_a_new_check_naming_a_script_makes_it_shaping(monkeypatch: pytest.MonkeyPatch) -> None:
    extra = check.Check("plots", "invariant", check.BOTH, ("python3", "scripts/scenario_plots.py"))
    checks = (*check.CHECKS, extra)
    monkeypatch.setattr(check, "CHECKS", checks)
    monkeypatch.setattr(check, "SHAPING_SCRIPTS", check._shaping_scripts(checks))
    assert "scripts/scenario_plots.py" in check.SHAPING_SCRIPTS
    assert selected("quick", "scripts/scenario_plots.py")["pytest"] == check.Selection(
        check.PATHS, ("tests/scripts",)
    )


def test_an_image_selects_the_invariants_only() -> None:
    invariants = {c.id for c in check.CHECKS if c.kind == "invariant"}
    got = selected("full", "docs/source/_static/logo.png")
    ran = {name for name, selection in got.items() if selection.state != check.NOT_TOUCHED}
    assert ran == invariants | {"docs", "linkcheck"}
    got = selected("full", WT + "img/icon.png")
    ran = {name for name, selection in got.items() if selection.state != check.NOT_TOUCHED}
    assert ran == invariants | {"package"}
    got = selected("quick", "tests/interfaces/design_system/baselines/dark.png")
    assert {n for n, s in got.items() if s.state != check.NOT_TOUCHED} == {
        c.id for c in check.CHECKS if c.kind == "invariant"
    }


def test_a_file_the_sweep_ignores_goes_to_its_host() -> None:
    got = selected("full", "tests/va/test_apply_fault.py", "tests/pty/test_spinner.py")
    assert got["pytest"].state == check.NOT_TOUCHED
    assert got["live-ca"].state == got["pytest-pty"].state == check.WHOLE
    plan = check.select("full", ["tests/integration/test_build_boot.py"], (), FILES)
    assert plan.selections["pytest"].state == check.NOT_TOUCHED
    assert plan.lanes == ("build-boot-smoke",)


def test_a_range_names_the_ci_lanes_it_touches() -> None:
    plan = check.select("quick", ["src/osprey/mcp_server/bluesky/server.py"], (), FILES)
    assert {
        "bluesky-deploy-e2e",
        "bluesky-web-deploy-e2e",
        "bluesky-queue-e2e",
        "tiled-roundtrip-e2e",
        "bluesky-catalog-e2e",
        "bluesky-sandbox-escape-e2e",
        "orm-roundtrip-e2e",
    } <= set(plan.lanes)
    assert "plugin-smoke" not in plan.lanes
    assert check.select("quick", ["README.md"], (), FILES).lanes == ()


def test_a_named_vitest_file_alone_is_accepted_and_routed_to_vitest() -> None:
    got = selected("quick", named=(TW + "toast.test.mjs",))
    assert got["vitest"] == check.Selection(check.PATHS, (TW + "toast.test.mjs",))
    assert got["pytest"].state == check.NOT_TOUCHED


def test_named_files_are_added_to_what_the_range_selects() -> None:
    got = selected(
        "quick",
        "tests/cli/test_main.py",
        named=(
            "tests/docs/test_shipped_identity_literals.py",
            "tests/cli/test_main.py::test_x",
            TW + "toast.test.js",
        ),
    )
    assert got["pytest"].paths == (
        "tests/cli/test_main.py",
        "tests/docs/test_shipped_identity_literals.py",
        "tests/cli/test_main.py::test_x",
    )
    assert got["vitest"].paths == (TW + "toast.test.js",)
    assert selected("quick", "uv.lock", named=("tests/cli/test_main.py",))["pytest"] == (
        check.Selection(check.WHOLE)
    )


def test_a_named_path_that_is_not_a_test_file_is_refused() -> None:
    with pytest.raises(check.Refused, match="notes.txt"):
        selected("quick", named=("notes.txt",))


def test_a_named_node_id_goes_to_pytest_as_given() -> None:
    got = selected("quick", named=("tests/cli/test_main.py::test_x",))
    assert got["pytest"] == check.Selection(check.PATHS, ("tests/cli/test_main.py::test_x",))
    assert got["vitest"].state == check.NOT_TOUCHED


def test_a_named_directory_goes_to_pytest_as_the_directory() -> None:
    got = selected("quick", named=("tests/cli/",))
    assert got["pytest"].paths == ("tests/cli",)
    assert got["vitest"].state == check.NOT_TOUCHED


def test_a_named_node_id_whose_file_is_absent_is_refused() -> None:
    with pytest.raises(check.Refused, match="tests/cli/test_nope.py is not a file"):
        selected("quick", named=("tests/cli/test_nope.py::test_x",))


def test_a_node_id_of_a_file_whose_runner_takes_none_is_refused() -> None:
    with pytest.raises(check.Refused, match="toast.test.js::t: only a .py file takes a node id"):
        selected("quick", named=(TW + "toast.test.js::t",))


def test_a_named_directory_without_a_pytest_file_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "notes.txt").write_text("")
    monkeypatch.setattr(check, "ROOT", tmp_path)
    with pytest.raises(check.Refused, match="notes: holds no pytest file"):
        selected("quick", named=("notes",))


def test_a_node_id_names_the_hosts_of_its_file() -> None:
    hosts = check._hosts_over(["tests/integration/test_build_boot.py"])
    assert "build-boot-smoke" in hosts
    assert check._hosts_over(["tests/integration/test_build_boot.py::test_x"]) == hosts


def test_the_pytest_line_carries_a_named_node_id() -> None:
    pytest_check = BY_ID["pytest"]
    selection = check.Selection(check.PATHS, ("tests/cli/test_main.py::test_x",))
    (line,) = check.commands(pytest_check, selection, check.Context(tier="quick"))
    argv = line[line.index("--") + 1 :]
    assert argv[: len(pytest_check.argv)] == list(pytest_check.argv)
    rest = argv[len(pytest_check.argv) :]
    first_ignore = next(i for i, arg in enumerate(rest) if arg.startswith("--ignore="))
    assert rest[:first_ignore] == ["tests/cli/test_main.py::test_x"]


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("src/osprey/cli/main.py", "*.py", True),
        ("src/osprey/cli/main.pyi", "*.py", False),
        ("package-lock.json", "package*.json", True),
        ("docs/source/index.rst", "docs/**", True),
        ("src/docs/index.rst", "docs/**", False),
        (WT + "settings.js", "src/osprey/interfaces/**/static/js/**/*.js", True),
        (WT + "scaffold/cards.js", "src/osprey/interfaces/**/static/js/**/*.js", True),
        ("src/osprey/interfaces/a.js", "src/osprey/interfaces/**/static/js/**/*.js", False),
        ("tests/e2e/test_bluesky_deploy.py", "tests/e2e/test_bluesky**", True),
        ("tests/e2e/test_tiled.py", "tests/e2e/test_bluesky**", False),
    ],
)
def test_scope_globs(path: str, pattern: str, expected: bool) -> None:
    assert check.matches(path, pattern) is expected


def git(repo: Path, *args: str) -> None:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(repo),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, check=True, timeout=60)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository on branch ``work``, one commit past ``base``, as the command's root."""
    for name in ("kept.py", "gone.py"):
        (tmp_path / name).write_text("x = 1\n", encoding="utf-8")
    git(tmp_path, "init", "--initial-branch", "base")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "--no-gpg-sign", "-m", "base")
    git(tmp_path, "checkout", "-b", "work")
    (tmp_path / "kept.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "gone.py").unlink()
    (tmp_path / "added.py").write_text("y = 1\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "--no-gpg-sign", "-m", "work")
    monkeypatch.setattr(kit, "ROOT", tmp_path)
    return tmp_path


def test_the_range_is_committed_state_without_deleted_files(repo: Path) -> None:
    (repo / "kept.py").write_text("x = 3\n", encoding="utf-8")
    (repo / "loose.py").write_text("z = 1\n", encoding="utf-8")
    assert check.touched("base") == ["added.py", "kept.py"]


def test_the_working_tree_range_adds_uncommitted_and_untracked_files(repo: Path) -> None:
    (repo / "added.py").write_text("y = 2\n", encoding="utf-8")
    (repo / "loose.py").write_text("z = 1\n", encoding="utf-8")
    assert check.touched("base", worktree=True) == ["added.py", "kept.py", "loose.py"]
    assert "loose.py" in check.repository_files(worktree=True)
    assert "loose.py" not in check.repository_files()


def test_a_path_with_a_space_is_one_path_in_the_working_tree_range(repo: Path) -> None:
    (repo / "src" / "osprey").mkdir(parents=True)
    (repo / "src" / "osprey" / "a b.py").write_text("z = 1\n", encoding="utf-8")
    assert check.touched("base", worktree=True).count("src/osprey/a b.py") == 1
    assert check.repository_files(worktree=True).count("src/osprey/a b.py") == 1
    assert not {"src/osprey/a", "b.py"} & set(check.touched("base", worktree=True))


def test_a_renamed_tracked_path_with_a_space_is_one_path(repo: Path) -> None:
    (repo / "x y.py").write_text("v = 1\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "--no-gpg-sign", "-m", "add")
    git(repo, "mv", "x y.py", "z w.py")
    git(repo, "commit", "--no-gpg-sign", "-m", "rename")
    paths = check.touched("base")
    assert paths.count("z w.py") == 1
    assert not {"z", "w.py", "x", "y.py"} & set(paths)
    assert check.repository_files().count("z w.py") == 1


def test_a_scope_reads_the_paths_a_range_deleted_and_moved_away(repo: Path) -> None:
    (repo / "dir").mkdir()
    git(repo, "mv", "kept.py", "dir/kept.py")
    git(repo, "commit", "--no-gpg-sign", "-m", "move")
    assert check.touched("base") == ["added.py", "dir/kept.py"]
    assert check.touched("base", scope=True) == ["added.py", "dir/kept.py", "gone.py", "kept.py"]
    (repo / "dir" / "kept.py").unlink()
    assert check.touched("base", worktree=True) == ["added.py"]
    assert check.touched("base", worktree=True, scope=True) == ["added.py", "gone.py", "kept.py"]


def test_a_scope_holding_only_a_deleted_path_is_touched_and_no_runner_gets_it() -> None:
    gone = "src/osprey/cli/gone.py"
    plan = check.select("full", [], (), FILES, reach=[gone])
    assert plan.selections["pytest-pty"].state == check.WHOLE
    assert plan.selections["pytest"].state == check.NOT_TOUCHED
    assert check.select("full", [], (), FILES).selections["pytest-pty"].state == check.NOT_TOUCHED
    left = "tests/integration/test_build_boot.py"
    assert check.select("quick", [], (), FILES, reach=[left]).lanes == ("build-boot-smoke",)


@pytest.mark.usefixtures("repo")
def test_a_base_that_does_not_resolve_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert check.main(["quick", "--changed", "no-such-ref"]) == 2
    assert "no-such-ref does not resolve" in capsys.readouterr().err
    assert check.main(["run", "changelog", "--base", "no-such-ref"]) == 2
    assert check.main(["full", "--base", "no-such-ref"]) == 2


# ---------------------------------------------------------------------------
# Running: every process is recorded, none is started


class Recorder:
    """Stands in for the one function that starts a check's process."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.failing: dict[str, int] = {}
        #: Line text to the report body the runner writes when that line runs.
        self.reports: dict[str, str] = {}
        #: What every install probe answers: its exit status and its line.
        self.install: tuple[int, str] = (0, "the install the check proves against")

    def __call__(self, argv: Any, cwd: Path | None = None) -> int:  # noqa: ARG002
        argv = list(argv)
        self.calls.append(argv)
        line = shlex.join(argv)
        for text, body in self.reports.items():
            if text in line:
                (target,) = (
                    part.split("=", 1)[1]
                    for part in argv
                    if part.startswith(("--junitxml=", "--outputFile.junit="))
                )
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                Path(target).write_text(body, encoding="utf-8")
        return next((code for text, code in self.failing.items() if text in line), 0)

    def ran(self, text: str) -> bool:
        return any(text in shlex.join(argv) for argv in self.calls)


@pytest.fixture
def recorder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """An empty root, every need met, the venue met, every install proven, and
    no process started."""
    recorder = Recorder()
    monkeypatch.setattr(kit, "ROOT", tmp_path)
    monkeypatch.setattr(kit, "run", recorder)
    monkeypatch.setattr(check, "probe_install", lambda install: recorder.install)
    monkeypatch.setattr(kit, "capture", lambda argv: (0, ""))
    for need in check.NEED_PROBES:
        monkeypatch.setitem(check.NEED_PROBES, need, lambda: None)
    _on_venue(monkeypatch, True)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    return recorder


def last_line(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().out.strip().splitlines()[-1]


def test_a_clean_full_run_names_what_it_did_not_run(
    recorder: Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert check.main(["full"]) == 0
    for text in (
        "uv lock --check",
        "npm run typecheck",
        "npx vitest run",
        "live_ca/gate.py",
        "uv build",
    ):
        assert recorder.ran(text), text
    verdict = last_line(capsys)
    assert verdict.startswith(
        "check: clean; "
        "not run, no base given: changelog, plugin-version, gitleaks; "
        "run on CI, never run here: dependency-floor, "
    )
    assert all(lane.job in verdict for lane in check.LANES)


def test_every_check_runs_after_a_failure_and_the_verdict_names_them(
    recorder: Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder.failing = {"ruff check .": 1, "vitest run --reporter": 1}
    assert check.main(["quick"]) == 1
    assert recorder.ran("mypy_gate.py") and recorder.ran("run_bounded.py 1800")
    out = capsys.readouterr().out.strip().splitlines()
    assert f"check: ruff: failed (exit 1; fix: {check.FIX_HINT})" in out
    assert "check: mypy: ran" in out
    assert out[-1] == (
        f"check: not clean: failed: ruff (fix: {check.FIX_HINT}), vitest; "
        "not run, no base given: changelog, plugin-version, gitleaks"
    )


@pytest.fixture
def any_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every base resolves and its range is empty."""
    monkeypatch.setattr(check, "require_base", lambda base: None)


def _pytest_argvs(argvs: list[list[str]]) -> list[list[str]]:
    return [argv for argv in argvs if "pytest" in argv]


@pytest.mark.usefixtures("any_base")
def test_full_with_a_base_arms_every_range_check_and_runs_whole(
    recorder: Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert check.main(["full", "--base", "B"]) == 0
    assert recorder.ran("changelog_fragments.py check --base B")
    assert recorder.ran("plugin_version.py check --base B")
    assert recorder.ran("gitleaks git --log-opts=B..HEAD")
    verdict = last_line(capsys)
    assert "no base given" not in verdict
    assert "run on CI, never run here" in verdict
    assert _pytest_argvs(check.resolve(["full", "--base", "B"])) == _pytest_argvs(
        check.resolve(["full"])
    )


@pytest.mark.usefixtures("any_base")
def test_changed_and_a_different_base_are_refused(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert check.main(["quick", "--changed", "B", "--base", "C"]) == 2
    assert recorder.calls == []
    monkeypatch.setattr(check, "touched", lambda base, worktree=False, scope=False: [])
    monkeypatch.setattr(check, "repository_files", lambda worktree=False: [])
    monkeypatch.setattr(kit, "ROOT", REPO_ROOT)
    assert check.main(["quick", "--changed", "B", "--base", "B"]) == 0


@pytest.mark.usefixtures("any_base")
def test_a_new_range_check_is_armed_by_the_base_alone(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = check.Check("range-probe", "invariant", check.BOTH, ("echo", "{base}"), needs=("base",))
    monkeypatch.setattr(check, "CHECKS", (*check.CHECKS, probe))
    assert check.main(["quick", "--base", "B"]) == 0
    assert ["echo", "B"] in recorder.calls


GITLEAKS_WORKTREE = [
    ["gitleaks", "git", "--log-opts=B..HEAD", "--no-banner", "--redact"],
    ["gitleaks", "git", "--staged", "--no-banner", "--redact"],
    ["gitleaks", "git", "--pre-commit", "--no-banner", "--redact"],
]


@pytest.mark.usefixtures("any_base")
@pytest.mark.parametrize("untracked", [[], ["new.py", "a b.txt"]])
def test_the_working_tree_reaches_every_range_check(
    monkeypatch: pytest.MonkeyPatch, untracked: list[str]
) -> None:
    monkeypatch.setattr(check, "touched", lambda base, worktree=False, scope=False: [])
    monkeypatch.setattr(check, "repository_files", lambda worktree=False: [])
    monkeypatch.setattr(kit, "git_paths", lambda *args: untracked if "--others" in args else [])
    argvs = check.resolve(["quick", "--changed", "B", "--worktree"])
    assert ["python3", "scripts/changelog_fragments.py", "check", "--base", "B", "--worktree"] in (
        argvs
    )
    assert ["python3", "scripts/plugin_version.py", "check", "--base", "B", "--worktree"] in argvs
    scans = [argv for argv in argvs if argv[:1] == ["gitleaks"]]
    assert scans == [
        *GITLEAKS_WORKTREE,
        *(["gitleaks", "dir", path, "--no-banner", "--redact"] for path in untracked),
    ]
    # A worktree line holding {untracked} runs once per untracked path, none when there is none.
    lines = (("scan", "--since={base}"), ("scan", "{untracked}"))
    assert check._worktree_lines(lines, "B") == [
        ["scan", "--since=B"],
        *(["scan", path] for path in untracked),
    ]


def test_every_range_check_declares_its_working_tree_form() -> None:
    """A range check under ``--worktree`` still judges the range: one of its
    working-tree lines carries the base, beside the lines that read the index,
    the unstaged diff and the untracked files."""
    ranged = [c for c in check.CHECKS if "base" in c.needs]
    assert {c.id for c in ranged} == {"changelog", "plugin-version", "gitleaks"}
    for declared in ranged:
        assert declared.worktree, f"{declared.id} judges a range but not the working tree"
        assert any("{base}" in part for line in declared.worktree for part in line), (
            f"{declared.id}: its working-tree form no longer judges the range"
        )


def test_a_pytest_run_stopped_at_its_bound_is_a_failure(
    recorder: Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder.failing = {"tests/pty/": 124}
    assert check.main(["run", "pytest-pty"]) == 1
    assert "check: pytest-pty: failed (still running at its bound)" in capsys.readouterr().out


def test_named_paths_whose_tests_are_all_deselected_are_named_as_not_run(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check, "repository_files", lambda worktree=False: [])
    # A check whose scope is derived hosts the tests under tests/hosted/.
    _replace(monkeypatch, "live-ca", derive_scope=lambda files: ("tests/hosted/**",))
    recorder.failing = {"scripts/run_bounded.py": 5}
    context = check.Context(tier="quick")
    pytest_check = BY_ID["pytest"]

    hosted = check.Selection(check.PATHS, ("tests/hosted/test_contract.py",))
    (outcome,) = check.execute([(pytest_check, hosted)], context)
    assert outcome == check.Outcome("pytest", check.NOT_RUN, check.DESELECTED, hosts=("live-ca",))
    capsys.readouterr()
    assert check.report([outcome], (), scoped=True) == 0
    out = capsys.readouterr().out.strip().splitlines()
    named = "the marker expression deselects every test in these paths"
    assert out[0] == f"check: pytest: not run, {named} (run by live-ca)"
    assert out[-1] == f"check: clean; not run, {named}: pytest (run by live-ca)"

    mirrored = check.Selection(check.PATHS, ("tests/hosted",))
    assert check.execute([(pytest_check, mirrored)], context)[0].hosts == ("live-ca",)
    lane_hosted = check.Selection(check.PATHS, ("tests/integration/test_build_boot.py",))
    (by_lane,) = check.execute([(pytest_check, lane_hosted)], context)
    assert check._fact(by_lane) == "run on CI: build-boot-smoke"
    unhosted_paths = check.Selection(check.PATHS, ("tests/cli/test_main.py",))
    (alone,) = check.execute([(pytest_check, unhosted_paths)], context)
    assert (alone.state, alone.reason, alone.hosts) == (check.NOT_RUN, check.DESELECTED, ())
    assert check.verdict([alone], (), scoped=True) == f"clean; not run, {named}: pytest"

    whole = check.Selection(check.WHOLE)
    assert check.execute([(pytest_check, whole)], context)[0].state == check.FAILED


def test_a_fixture_change_under_a_hosted_directory_names_its_host(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = [
        "tests/hosted/suite/test_contract.py",
        "tests/hosted/suite/fixtures/ir-map.json",
    ]
    monkeypatch.setattr(check, "repository_files", lambda worktree=False: files)
    _replace(monkeypatch, "live-ca", derive_scope=lambda files: ("tests/hosted/suite/**",))
    recorder.failing = {"scripts/run_bounded.py": 5}
    selection = fw._select_pytest(["tests/hosted/suite/fixtures/ir-map.json"], files)
    assert selection == check.Selection(check.PATHS, ("tests/hosted",))
    (outcome,) = check.execute([(BY_ID["pytest"], selection)], check.Context(tier="quick"))
    assert (outcome.reason, outcome.hosts) == (check.DESELECTED, ("live-ca",))


@pytest.mark.parametrize(
    "fields",
    [
        {"state": "skipped"},
        {"state": check.NOT_RUN},
        {"state": check.NOT_RUN, "reason": "busy"},
        {"state": check.RAN, "reason": check.NOT_TOUCHED},
        {"state": check.NOT_RUN, "reason": check.NO_BASE, "hosts": ("e2e-tests",)},
        {"state": check.NOT_RUN, "reason": check.NOT_INSTALLED},
        {"state": check.NOT_RUN, "reason": check.NO_BASE, "attestation": "not installed"},
        {"state": check.REFUSED, "need": "run npm ci", "attestation": "proven"},
        {"state": check.FAILED},
        {"state": check.RAN, "status": 1},
        {"state": check.RAN, "failures": ("tests/test_a.py::test_b",)},
        {"state": check.REFUSED},
        {"state": check.FAILED, "status": 1, "need": "run npm ci"},
    ],
)
def test_an_outcome_outside_the_model_is_rejected(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="^pytest: "):
        check.Outcome("pytest", **fields)


def test_a_refusal_names_every_unmet_need(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(check.NEED_PROBES, "node", fw._need_node)
    monkeypatch.setitem(check.NEED_PROBES, "docs-extra", fw._need_docs_extra)
    monkeypatch.setattr(kit, "capture", lambda argv: (1, ""))
    monkeypatch.setattr(fw.shutil, "which", lambda name: None)
    assert check.main(["full"]) == 2
    assert recorder.calls == []
    captured = capsys.readouterr()
    assert captured.out == ""
    err = captured.err.strip().splitlines()
    node = "the Node toolchain is not installed: install Node, then run npm ci"
    docs = "the docs extra is not installed: run uv sync --extra docs"
    assert err == [
        f"check: tsc: refused ({node})",
        f"check: eslint: refused ({node})",
        f"check: prettier: refused ({node})",
        f"check: vitest: refused ({node})",
        f"check: app-check-js: refused ({node})",
        f"check: dock-engine-tests: refused ({node})",
        f"check: dock-engine-bundle: refused ({node})",
        f"check: docs: refused ({docs})",
        f"check: linkcheck: refused ({docs})",
        f"check: refused: tsc, eslint, prettier, vitest, app-check-js, dock-engine-tests, "
        f"dock-engine-bundle ({node}); "
        f"refused: docs, linkcheck ({docs})",
    ]


def test_fix_rewrites_nothing_when_a_need_is_refused(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(check.NEED_PROBES, "node", lambda: "Node is not installed")
    assert check.main(["quick", "--fix"]) == 2
    assert recorder.calls == []


def last_error_line(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().err.strip().splitlines()[-1]


def _faulting(probe: str, fact: str) -> Any:
    """A need probe that says this machine cannot judge."""

    def need() -> str | None:
        raise kit.EnvironmentFault(probe, fact)

    return need


def test_a_raising_need_gives_the_environment_verdict_and_runs_nothing(
    recorder: Recorder, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fault = _faulting("docker storage", "volume create input/output error")
    monkeypatch.setitem(check.NEED_PROBES, "uv", fault)
    assert check.main(["quick", "--fix"]) == kit.ENVIRONMENT_EXIT == 3
    assert last_error_line(capsys) == (
        "check: environment: docker storage: volume create input/output error"
    )
    assert recorder.calls == []


def test_a_fault_wins_over_a_refusal(
    recorder: Recorder, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(check.NEED_PROBES, "uv", lambda: "uv is not on PATH")
    monkeypatch.setitem(check.NEED_PROBES, "node", _faulting("free space", "1 MiB free"))
    assert check.main(["full"]) == 3
    assert last_error_line(capsys) == "check: environment: free space: 1 MiB free"
    assert recorder.calls == []


def test_a_reports_directory_that_cannot_open_is_the_environment_verdict(
    recorder: Recorder, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def full(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(check.tempfile, "TemporaryDirectory", full)
    assert check.main(["quick"]) == 3
    assert last_error_line(capsys) == (
        "check: environment: free space: [Errno 28] No space left on device"
    )
    assert recorder.calls == []


def test_a_selected_need_is_probed_from_the_selection_and_the_run(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Any, Any]] = []
    probed: list[str] = []

    def hook(selection: Any, context: Any) -> tuple[str, ...]:
        seen.append((selection, context))
        return ("gitleaks",)

    monkeypatch.setitem(check.NEED_PROBES, "gitleaks", lambda: probed.append("gitleaks"))
    _replace(monkeypatch, "pytest", selected_needs=hook)
    assert check.main(["run", "pytest", "--shard", "cli"]) == 0
    ((selection, context),) = seen
    assert selection.state == check.WHOLE
    assert context.shard == "cli"
    assert probed == ["gitleaks"]
    assert recorder.ran("pytest")


def test_an_unmet_selected_need_is_a_refusal(
    recorder: Recorder, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(check.NEED_PROBES, "gitleaks", lambda: "gitleaks is not on PATH")
    _replace(monkeypatch, "pytest", selected_needs=lambda _s, _c: ("gitleaks",))
    assert check.main(["run", "pytest"]) == 2
    assert "gitleaks is not on PATH" in last_error_line(capsys)
    assert recorder.calls == []


@pytest.mark.parametrize("words", [["list"], ["list", "--json"]])
def test_list_probes_nothing(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, words: list[str]
) -> None:
    for need in check.NEED_PROBES:
        monkeypatch.setitem(check.NEED_PROBES, need, _faulting("free space", "listed"))
    assert check.main(words) == 0
    assert recorder.calls == []


def test_the_selection_hook_stays_out_of_the_declaration() -> None:
    assert all("selected_needs" not in entry for entry in check.declaration()["checks"])


def _probing(monkeypatch: pytest.MonkeyPatch, *answers: tuple[int, str]) -> list[list[str]]:
    """``_probe`` answers each call with the next (status, stdout); the argv are returned."""
    calls: list[list[str]] = []
    queue = list(answers)

    def probe(argv: Sequence[str], timeout: float | None = None) -> tuple[int, str, str]:  # noqa: ARG001
        calls.append(list(argv))
        status, out = queue.pop(0)
        return status, out, ""

    monkeypatch.setattr(fw, "_probe", probe)
    return calls


STALE_ROW = "osprey-connectors\t2026.9.0b3\t2026.10.0\n"


def test_a_current_install_syncs_nothing(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    read = _probing(monkeypatch, (0, ""))
    assert fw._need_install_current() is None
    assert read == [list(fw.INSTALLED_VERSIONS_PROBE)]
    assert "--no-sync" in fw.INSTALLED_VERSIONS_PROBE
    assert recorder.calls == []


def test_a_stale_install_is_resynced_and_rechecked(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    read = _probing(monkeypatch, (0, STALE_ROW + "pyaml-cs-osprey\t2026.9.0\t2026.10.0\n"), (0, ""))
    assert fw._need_install_current() is None
    assert len(read) == 2
    assert recorder.calls == [
        [
            "uv",
            "sync",
            "--frozen",
            "--inexact",
            "--reinstall-package",
            "osprey-connectors",
            "--reinstall-package",
            "pyaml-cs-osprey",
        ]
    ]
    assert "osprey-connectors installed 2026.9.0b3" in capsys.readouterr().out


def test_an_install_still_stale_after_the_sync_is_a_fault(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    _probing(monkeypatch, (0, STALE_ROW), (0, STALE_ROW))
    with pytest.raises(kit.EnvironmentFault) as fault:
        fw._need_install_current()
    assert fault.value.probe == "install currency"
    assert fault.value.fact == (
        "osprey-connectors installed 2026.9.0b3, tree describes 2026.10.0 after uv sync"
    )
    assert len(recorder.calls) == 1


def test_a_failed_sync_is_a_fault_whatever_the_reread_prints(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder.failing = {"uv sync": 1}
    _probing(monkeypatch, (0, STALE_ROW), (1, ""))
    with pytest.raises(kit.EnvironmentFault, match="after uv sync"):
        fw._need_install_current()


@pytest.mark.parametrize("status", [127, 1])
def test_a_reader_that_cannot_run_answers_met(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """127: the ``uv`` need refuses. Otherwise a checkout with no environment
    yet: the check's own ``uv run`` creates it current."""
    _probing(monkeypatch, (status, ""))
    assert fw._need_install_current() is None
    assert recorder.calls == []


def _free(monkeypatch: pytest.MonkeyPatch, **by_place: int) -> list[str]:
    """``shutil.disk_usage`` answers *temp* and *root* bytes free; the paths asked are returned."""
    asked: list[str] = []

    def usage(path: Any) -> Any:
        asked.append(str(path))
        free = by_place["temp"] if str(path) == fw.tempfile.gettempdir() else by_place["root"]
        return shutil._ntuple_diskusage(10**15, 10**15 - free, free)

    monkeypatch.setattr(fw.shutil, "disk_usage", usage)
    return asked


GIB = 2**30


def test_free_space_above_the_floor_is_met(monkeypatch: pytest.MonkeyPatch) -> None:
    _free(monkeypatch, temp=fw.FREE_SPACE_FLOOR_GIB * GIB, root=fw.FREE_SPACE_FLOOR_GIB * GIB)
    assert fw._need_free_space() is None


def test_free_space_under_the_floor_names_the_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fw.tempfile, "gettempdir", lambda: str(tmp_path))
    _free(monkeypatch, temp=313 * 2**20, root=100 * GIB)
    with pytest.raises(kit.EnvironmentFault) as fault:
        fw._need_free_space()
    assert fault.value.probe == "free space"
    assert fault.value.fact == f"313 MiB free on {tmp_path}, floor 5 GiB"


def test_free_space_is_read_once_per_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fw.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(kit, "ROOT", tmp_path / "root")
    (tmp_path / "root").mkdir()
    asked = _free(monkeypatch, temp=100 * GIB, root=100 * GIB)
    assert fw._need_free_space() is None
    assert asked == [str(tmp_path)]


def test_a_second_device_is_read_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    monkeypatch.setattr(fw.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(kit, "ROOT", root)
    monkeypatch.setattr(fw, "_device", lambda path: str(path))
    asked = _free(monkeypatch, temp=100 * GIB, root=1 * GIB)
    with pytest.raises(kit.EnvironmentFault, match="1024 MiB free on .*root"):
        fw._need_free_space()
    assert asked == [str(tmp_path), str(root)]


def test_no_usable_temporary_directory_is_a_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    def nowhere() -> str:
        raise FileNotFoundError(2, "No usable temporary directory found")

    monkeypatch.setattr(fw.tempfile, "gettempdir", nowhere)
    with pytest.raises(kit.EnvironmentFault, match="No usable temporary directory found"):
        fw._need_free_space()


class FakeDocker:
    """Answers ``_probe`` for the docker commands; no test here starts a real one."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, **answers: Any) -> None:
        self.calls: list[list[str]] = []
        #: The verb (``info``, ``create``, ``rm``) to (status, stderr) or an exception.
        self.answers = answers
        self.timeouts: list[float | None] = []
        monkeypatch.setattr(fw.shutil, "which", lambda name: f"/stub/{name}")
        monkeypatch.setattr(fw, "_probe", self)

    def __call__(self, argv: Sequence[str], timeout: float | None = None) -> tuple[int, str, str]:
        self.calls.append(list(argv))
        self.timeouts.append(timeout)
        answer = self.answers.get(argv[1] if argv[1] == "info" else argv[2], (0, ""))
        if isinstance(answer, Exception):
            raise answer
        return answer[0], "", answer[1]


def _volume(verb: str) -> list[str]:
    return ["docker", "volume", verb, f"osprey-check-probe-{os.getpid()}"]


def test_docker_storage_creates_and_removes_a_probe_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = FakeDocker(monkeypatch)
    assert fw._need_docker_storage() is None
    assert docker.calls == [["docker", "info"], _volume("create"), _volume("rm")]
    assert set(docker.timeouts) == {60}


def test_no_docker_on_the_path_is_met(monkeypatch: pytest.MonkeyPatch) -> None:
    docker = FakeDocker(monkeypatch)
    monkeypatch.setattr(fw.shutil, "which", lambda name: None)
    assert fw._need_docker_storage() is None
    assert docker.calls == []


@pytest.mark.parametrize(
    "info", [(1, "Cannot connect to the Docker daemon"), subprocess.TimeoutExpired("docker", 60)]
)
def test_a_daemon_that_does_not_answer_is_met(monkeypatch: pytest.MonkeyPatch, info: Any) -> None:
    docker = FakeDocker(monkeypatch, info=info)
    assert fw._need_docker_storage() is None
    assert docker.calls == [["docker", "info"]]


def test_a_failing_volume_create_is_a_fault_in_the_daemons_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = FakeDocker(
        monkeypatch,
        create=(
            1,
            "Error response from daemon: mkdir /var/lib/docker/volumes/x: input/output error\n",
        ),
    )
    with pytest.raises(kit.EnvironmentFault) as fault:
        fw._need_docker_storage()
    assert fault.value.probe == "docker storage"
    assert fault.value.fact == "volume create input/output error"
    assert _volume("rm") not in docker.calls


def test_a_volume_create_that_does_not_answer_is_a_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeDocker(monkeypatch, create=subprocess.TimeoutExpired("docker", 60))
    with pytest.raises(kit.EnvironmentFault) as fault:
        fw._need_docker_storage()
    assert fault.value.fact == "volume create did not answer in 60 s"


def test_a_failing_volume_remove_is_a_fault_after_the_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = FakeDocker(monkeypatch, rm=(1, "remove x: volume is in use"))
    with pytest.raises(kit.EnvironmentFault) as fault:
        fw._need_docker_storage()
    assert fault.value.fact == "volume rm volume is in use"
    assert docker.calls[-1] == _volume("rm")


@pytest.fixture
def group_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A git tree with a file carrying the docker group, one without, and a
    package whose conftest carries it."""
    carrying = 'import pytest\npytestmark = pytest.mark.xdist_group("docker")\n'
    for path, text in {
        "tests/plain/test_plain.py": "def test_x(): pass\n",
        "tests/containers/test_db.py": carrying,
        "tests/suite/conftest.py": carrying,
        "tests/suite/deep/test_deep.py": "def test_y(): pass\n",
        "tests/README.md": 'pytest.mark.xdist_group("docker")\n',
    }.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.setattr(kit, "ROOT", tmp_path)
    return tmp_path


def _hook(*roots: str) -> Any:
    return fw._docker_group_needs(lambda _context: roots)


@pytest.mark.parametrize(
    ("paths", "needs"),
    [
        (("tests/plain/test_plain.py",), ()),
        (("tests/plain/test_plain.py::test_x",), ()),
        (("tests/containers/test_db.py::test_q",), ("docker-storage",)),
        (("tests/plain/test_plain.py", "tests/containers/test_db.py"), ("docker-storage",)),
        (("tests/containers/",), ("docker-storage",)),
        (("tests/suite/deep/test_deep.py",), ("docker-storage",)),
        (("tests/suite/deep",), ("docker-storage",)),
        (("tests/gone/test_deleted.py",), ()),
        (("tests/gone/test_deleted.py", "tests/containers/test_db.py"), ("docker-storage",)),
    ],
)
def test_the_docker_group_is_read_from_the_selected_paths(
    group_tree: Path,  # noqa: ARG001
    paths: tuple[str, ...],
    needs: tuple[str, ...],
) -> None:
    selection = check.Selection(check.PATHS, paths)
    assert _hook("tests/")(selection, check.Context()) == needs


def test_the_docker_group_under_a_whole_selection_is_read_from_the_roots(
    group_tree: Path,  # noqa: ARG001
) -> None:
    whole = check.Selection(check.WHOLE)
    assert _hook("tests/plain")(whole, check.Context()) == ()
    assert _hook("tests/containers")(whole, check.Context()) == ("docker-storage",)


def test_a_hook_that_cannot_read_the_tree_answers_no_need(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kit, "ROOT", tmp_path)
    selection = check.Selection(check.WHOLE)
    assert _hook("tests/")(selection, check.Context()) == ()
    monkeypatch.setattr(fw, "_probe", lambda argv: (127, "", ""))
    assert _hook("tests/")(selection, check.Context()) == ()


def test_a_check_off_its_venue_has_its_needs_left_unprobed(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    probed: list[str] = []

    def uv_probe() -> str | None:
        probed.append("uv")
        return "uv is not on PATH"

    _on_venue(monkeypatch, False)
    monkeypatch.setitem(check.NEED_PROBES, "uv", uv_probe)
    assert check.main(["run", "live-ca"]) == 0
    assert probed == [] and recorder.calls == []
    needs = check.VENUES[fw.LIVE_CA_VENUE].words
    assert f"check: not run here: live-ca ({needs})" in capsys.readouterr().out.splitlines()


# ---------------------------------------------------------------------------
# Failing tests named from the runners' reports


#: A pytest xunit1 report: a class test, a parametrised id holding a dot, a
#: module whose import fails, a pass, and a failure reported twice.
PYTEST_REPORT_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="3" tests="5">
<testcase file="tests/sub/test_a.py" classname="tests.sub.test_a.TestK" name="test_x">
<failure message="assert 1 == 2">assert 1 == 2</failure></testcase>
<testcase file="tests/sub/test_a.py" classname="tests.sub.test_a" name="test_p[a.b]">
<failure message="boom">boom</failure></testcase>
<testcase file="tests/sub/test_broken.py" classname="" name="tests.sub.test_broken">
<error message="collection failure">ImportError</error></testcase>
<testcase file="tests/sub/test_a.py" classname="tests.sub.test_a" name="test_ok" />
<testcase file="tests/sub/test_a.py" classname="tests.sub.test_a.TestK" name="test_x">
<failure message="assert 1 == 2">assert 1 == 2</failure></testcase>
</testsuite></testsuites>
"""

#: A Vitest junit report: a failing test in a group, a file that cannot import, a pass.
VITEST_REPORT_XML = """<?xml version="1.0" encoding="UTF-8" ?>
<testsuites name="vitest tests" tests="3" failures="2" errors="0">
<testsuite name="js/a.test.mjs" tests="2" failures="1">
<testcase classname="js/a.test.mjs" name="grp &gt; fails one" time="0.01">
<failure message="expected 1 to be 2" type="AssertionError">AssertionError</failure>
</testcase>
<testcase classname="js/a.test.mjs" name="grp &gt; passes" time="0.01" />
</testsuite>
<testsuite name="js/b.test.mjs" tests="1" failures="1">
<testcase classname="js/b.test.mjs" name="js/b.test.mjs" time="0">
<failure message="Failed to load url ./missing.mjs">Error</failure>
</testcase>
</testsuite>
</testsuites>
"""


def _write_report(reports: Path, check_id: str, body: str, name: str = "junit.xml") -> None:
    (reports / check_id).mkdir(parents=True, exist_ok=True)
    (reports / check_id / name).write_text(body, encoding="utf-8")


def test_failing_ids_are_read_from_a_pytest_report(tmp_path: Path) -> None:
    _write_report(tmp_path, "pytest", PYTEST_REPORT_XML)
    assert check.failing_tests(BY_ID["pytest"], str(tmp_path)) == (
        "tests/sub/test_a.py::TestK::test_x",
        "tests/sub/test_a.py::test_p[a.b]",
        "tests/sub/test_broken.py",
    )


def test_failing_ids_are_read_from_a_vitest_report(tmp_path: Path) -> None:
    _write_report(tmp_path, "vitest", VITEST_REPORT_XML)
    assert check.failing_tests(BY_ID["vitest"], str(tmp_path)) == (
        "js/a.test.mjs > grp > fails one",
        "js/b.test.mjs",
    )


#: A test module with a class-method failure, a parametrized failure whose id
#: holds a dot, and a pass.
ROUND_TRIP_MODULE = """\
import pytest


class TestK:
    def test_x(self):
        assert 1 == 2


@pytest.mark.parametrize("value", ["a.b", 2])
def test_p(value):
    assert value == 2


def test_ok():
    pass
"""


def test_the_failing_ids_are_the_node_ids_the_installed_pytest_prints(tmp_path: Path) -> None:
    module = tmp_path / "tests" / "sub" / "test_round_trip.py"
    module.parent.mkdir(parents=True)
    module.write_text(ROUND_TRIP_MODULE, encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    reports = tmp_path / "reports"
    flags = [
        part.format(reports=reports, id="pytest") for part in check.REPORT_FLAGS["junit-pytest"]
    ]
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:randomly",
            "-p",
            "no:xdist",
            "-p",
            "no:cacheprovider",
            "-rf",
            "-q",
            *flags,
            "tests/sub/test_round_trip.py",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    printed = tuple(re.findall(r"^FAILED (\S+)", done.stdout, re.MULTILINE))
    assert printed == (
        "tests/sub/test_round_trip.py::TestK::test_x",
        "tests/sub/test_round_trip.py::test_p[a.b]",
    ), done.stdout
    assert check.failing_tests(BY_ID["pytest"], str(reports)) == printed


def test_a_missing_or_broken_report_gives_no_names(tmp_path: Path) -> None:
    pytest_check = BY_ID["pytest"]
    assert check.failing_tests(pytest_check, str(tmp_path)) == ()
    _write_report(tmp_path, "pytest", "")
    assert check.failing_tests(pytest_check, str(tmp_path)) == ()
    _write_report(tmp_path, "pytest", PYTEST_REPORT_XML[:200])
    assert check.failing_tests(pytest_check, str(tmp_path)) == ()
    _write_report(tmp_path, "pytest", PYTEST_REPORT_XML, name="later.xml")
    assert len(check.failing_tests(pytest_check, str(tmp_path))) == 3


def test_a_failed_pytest_names_its_failing_tests(
    recorder: Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = "".join(
        f'<testcase file="tests/test_m.py" classname="tests.test_m" name="test_{index}">'
        "<failure>no</failure></testcase>"
        for index in range(7)
    )
    recorder.reports = {
        "run_bounded.py": f"<testsuites><testsuite>{cases}</testsuite></testsuites>"
    }
    recorder.failing = {"run_bounded.py": 1}
    assert check.main(["run", "pytest"]) == 1
    out = capsys.readouterr().out.strip().splitlines()
    shown = ", ".join(f"tests/test_m.py::test_{index}" for index in range(5))
    assert f"check: pytest: failed ({shown}, +2)" in out
    assert out[-1] == f"check: not clean: failed: pytest ({shown}, +2)"


def test_a_failed_check_without_a_report_keeps_its_exit_status(
    recorder: Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder.reports = {"tests/pty/": PYTEST_REPORT_XML}
    recorder.failing = {"tests/pty/": 124}
    assert check.main(["run", "pytest-pty"]) == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert "check: pytest-pty: failed (still running at its bound)" in out
    assert out[-1] == "check: not clean: failed: pytest-pty"


def test_the_verdict_caps_the_named_failures_at_five() -> None:
    names = tuple(f"tests/test_m.py::test_{index}" for index in range(8))
    outcomes = [
        check.Outcome("pytest", check.FAILED, status=1, failures=names),
        check.Outcome("vitest", check.FAILED, status=1, failures=names[:2]),
        check.Outcome("ruff", check.FAILED, status=1),
    ]
    assert check.verdict(outcomes, (), scoped=False) == (
        f"not clean: failed: pytest ({', '.join(names[:5])}, +3), "
        f"vitest ({', '.join(names[:2])}), ruff (fix: {check.FIX_HINT})"
    )


def test_a_failed_check_that_declares_a_fix_names_it() -> None:
    outcome = check.Outcome("ruff-format", check.FAILED, status=1)
    assert check._line(outcome) == (
        "ruff-format: failed (exit 1; fix: uv run python scripts/check.py quick --fix)"
    )
    assert check.verdict([outcome], (), scoped=False) == (
        "not clean: failed: ruff-format (fix: uv run python scripts/check.py quick --fix)"
    )


def test_a_failed_check_without_a_fix_names_none() -> None:
    outcome = check.Outcome("mypy", check.FAILED, status=1)
    assert check._line(outcome) == "mypy: failed (exit 1)"
    assert check.verdict([outcome], (), scoped=False) == "not clean: failed: mypy"


def test_dropping_the_fix_from_the_row_drops_the_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _replace(monkeypatch, "ruff-format", fix=())
    outcome = check.Outcome("ruff-format", check.FAILED, status=1)
    assert check._line(outcome) == "ruff-format: failed (exit 1)"
    assert check.verdict([outcome], (), scoped=False) == "not clean: failed: ruff-format"


def test_a_check_off_its_venue_is_named_and_not_run(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _on_venue(monkeypatch, False)
    assert check.main(["full"]) == 0
    assert not recorder.ran("live_ca/gate.py")
    named = "not run here: live-ca (needs linux x86_64 or aarch64)"
    out = capsys.readouterr().out.strip().splitlines()
    assert f"check: {named}" in out
    assert named in out[-1]


def test_the_live_channel_access_venue_is_linux_x86_64_or_aarch64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_probe(*_args: Any) -> tuple[int, str]:
        raise AssertionError("the venue is the platform alone and probes nothing")

    monkeypatch.setattr(kit, "capture", no_probe)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(fw.platform, "machine", lambda: "x86_64")
    assert fw._on_live_ca_venue() is False
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(fw.platform, "machine", lambda: "armv7l")
    assert fw._on_live_ca_venue() is False
    monkeypatch.setattr(fw.platform, "machine", lambda: "aarch64")
    assert fw._on_live_ca_venue() is True
    monkeypatch.setattr(fw.platform, "machine", lambda: "x86_64")
    assert fw._on_live_ca_venue() is True


def test_the_venue_is_the_platform_the_virtual_accelerator_extra_installs_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra = pyproject["project"]["optional-dependencies"]["virtual-accelerator"]
    markers = {entry.split(";", 1)[1].strip() for entry in extra if ";" in entry}
    assert len(markers) == 1, markers
    found: dict[str, set[str]] = {}
    for key, value in re.findall(r"(sys_platform|platform_machine) == '([^']+)'", markers.pop()):
        found.setdefault(key, set()).add(value)
    assert set(found) == {"sys_platform", "platform_machine"}, found
    (system,) = found["sys_platform"]
    monkeypatch.setattr(kit, "capture", lambda argv: (1, ""))
    monkeypatch.setattr(sys, "platform", system)
    for machine in found["platform_machine"]:
        monkeypatch.setattr(fw.platform, "machine", lambda machine=machine: machine)
        assert fw._on_live_ca_venue() is True, machine
    monkeypatch.setattr(fw.platform, "machine", lambda: "amd64")
    assert fw._on_live_ca_venue() is False


def test_live_ca_on_its_platform_without_the_extra_is_refused(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _on_venue(monkeypatch, True)
    monkeypatch.setitem(check.NEED_PROBES, "va-extra", fw._need_va_extra)
    monkeypatch.setattr(kit, "capture", lambda argv: (1, ""))
    assert check.main(["run", "live-ca"]) == 2
    assert "uv sync --extra virtual-accelerator" in capsys.readouterr().err
    assert check.main(["full"]) == 2
    assert "uv sync --extra virtual-accelerator" in capsys.readouterr().err
    assert recorder.calls == []


def test_live_ca_off_its_platform_probes_nothing(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def capture(argv: Sequence[str]) -> tuple[int, str]:
        if tuple(argv) == fw.LIVE_CA_PROBE:
            raise AssertionError("the extra is probed only where live-ca runs")
        return 0, ""

    _on_venue(monkeypatch, False)
    monkeypatch.setitem(check.NEED_PROBES, "va-extra", fw._need_va_extra)
    monkeypatch.setattr(kit, "capture", capture)
    assert check.main(["full"]) == 0
    assert not recorder.ran("live_ca/gate.py")


def test_the_need_probe_imports_what_the_live_gate_requires() -> None:
    tree = ast.parse((REPO_ROOT / "scripts/va/live_ca/gate.py").read_text(encoding="utf-8"))
    required = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "PVA_REQUIRED_MODULES" for t in node.targets)
    )
    statement = ast.parse(fw.LIVE_CA_PROBE[-1]).body
    assert len(statement) == 1 and isinstance(statement[0], ast.Import)
    assert tuple(alias.name for alias in statement[0].names) == tuple(required)


def test_full_without_the_docs_extra_is_refused(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    probed: list[tuple[str, ...]] = []

    def failing(argv: Any) -> tuple[int, str]:
        probed.append(tuple(argv))
        return 1, ""

    monkeypatch.setattr(kit, "capture", failing)
    monkeypatch.setitem(check.NEED_PROBES, "docs-extra", fw._need_docs_extra)
    assert check.main(["full"]) == 2
    assert "uv sync --extra docs" in capsys.readouterr().err
    assert probed == [fw.DOCS_EXTRA_PROBE]
    assert recorder.calls == []


def test_full_with_the_docs_extra_plans_the_docs_build(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(check.NEED_PROBES, "docs-extra", fw._need_docs_extra)
    assert check.main(["full"]) == 0
    assert [*fw.UV_RUN, "make", "html", "SPHINXOPTS=-W --keep-going"] in recorder.calls


@pytest.mark.usefixtures("recorder")
def test_quick_does_not_ask_for_the_docs_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(check.NEED_PROBES, "docs-extra", lambda: "refused")
    assert check.main(["quick"]) == 0


def stub(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def stub_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A PATH holding only a stub ``uv`` and ``git``, under an empty root with JS deps."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub(bin_dir, "uv", f'echo "$@" >> "{tmp_path}/uv.log"\necho stub uv says hello')
    stub(bin_dir, "git", "echo false")
    (tmp_path / "node_modules").mkdir()
    monkeypatch.setattr(kit, "ROOT", tmp_path)
    monkeypatch.setenv("PATH", str(bin_dir))
    return bin_dir


def test_a_tier_without_node_is_refused_and_names_the_install(
    stub_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder = Recorder()
    monkeypatch.setattr(kit, "run", recorder)
    assert check.main(["quick"]) == 2
    install = "the Node toolchain is not installed: install Node, then run npm ci"
    refused = "tsc, eslint, prettier, vitest, dock-engine-tests, dock-engine-bundle"
    assert f"check: refused: {refused} ({install})" in capsys.readouterr().err
    probes = {shlex.join(argv[1:]) for argv in _declared_argvs() if argv[:1] == ("uv",)}
    log = tmp_path / "uv.log"
    logged = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    assert set(logged) <= probes, logged
    assert recorder.calls == []
    stub(stub_path, "node", "exit 0")
    stub(stub_path, "npm", "exit 0")
    _lockfiles(tmp_path, installed="1.0.0")
    assert fw._need_node() is None
    (tmp_path / "node_modules/.package-lock.json").unlink()
    (tmp_path / "node_modules").rmdir()
    answer = fw._need_node() or ""
    assert recorder.calls == [["npm", "ci"]]
    assert "node_modules/.package-lock.json is absent" in answer


# ---------------------------------------------------------------------------
# Provisioning: the needs put the tree's pins in place before any check runs


def _declared_argvs() -> list[tuple[str, ...]]:
    """Every argv the checks modules declare: each module-level tuple of words
    (and each step of a tuple of them), every check's argv and every fix."""
    found: list[tuple[str, ...]] = []
    for module in check.DECLARED.modules.values():
        for value in vars(module).values():
            if not isinstance(value, tuple):
                continue
            steps = value if value and all(isinstance(v, tuple) for v in value) else (value,)
            found += [s for s in steps if s and all(isinstance(w, str) for w in s)]
    for declared in check.CHECKS:
        found += [declared.argv, *declared.fix]
    return found


def _assert_never_relocks(argv: Sequence[str]) -> None:
    if tuple(argv[:2]) == ("uv", "run"):
        assert argv[2] == "--frozen", argv


def test_every_declared_uv_run_never_relocks() -> None:
    """A plain ``uv run`` rewrites a stale uv.lock, and the lockfile check must
    judge it as the tree left it."""
    argvs = _declared_argvs()
    assert fw.CLAUDE_PIN_PROBE in argvs
    assert BY_ID["pytest"].argv in argvs
    for argv in argvs:
        _assert_never_relocks(argv)


def test_every_resolved_uv_run_never_relocks() -> None:
    """The commands a tier starts, wrappers such as the bound included, keep
    the declared argvs' ``--frozen``."""
    for tier in check.TIERS:
        resolved = check.resolve([tier])
        assert any("scripts/run_bounded.py" in argv for argv in resolved), tier
        for argv in resolved:
            _assert_never_relocks(argv)


def test_every_declared_uv_run_never_relocks__mutation_drops_frozen() -> None:
    relocking = tuple(part for part in fw.CLAUDE_PIN_PROBE if part != "--frozen")
    with pytest.raises(AssertionError):
        _assert_never_relocks(relocking)


def test_no_check_sets_uv_frozen_for_its_children() -> None:
    """Under ``UV_FROZEN`` the lockfile check validates the lock and no longer
    judges whether it is stale."""
    for script in sorted((REPO_ROOT / "scripts").glob("*check*.py")):
        assert "UV_FROZEN" not in script.read_text(encoding="utf-8"), script


def test_the_pin_probe_runs_the_resolver() -> None:
    assert fw.CLAUDE_PIN_PROBE[3:] == ("python", "scripts/pinned_claude.py")
    assert (REPO_ROOT / fw.CLAUDE_PIN_PROBE[-1]).is_file()


def test_the_test_sweep_needs_the_pin() -> None:
    assert BY_ID["pytest"].needs == ("uv", "claude-pin")


@pytest.mark.usefixtures("recorder")
def test_the_pin_goes_first_on_path_before_any_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = tmp_path / "pin"
    pin.mkdir()
    probed: list[tuple[str, ...]] = []

    def probe(argv: Sequence[str]) -> tuple[int, str, str]:
        probed.append(tuple(argv))
        return 0, f"{pin}\n", ""

    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.setattr(fw, "_probe", probe)
    monkeypatch.setitem(check.NEED_PROBES, "claude-pin", fw._need_claude_pin)
    assert check.main(["quick"]) == 0
    assert os.environ["PATH"].startswith(str(pin) + os.pathsep)
    assert probed == [fw.CLAUDE_PIN_PROBE]


@pytest.mark.parametrize(
    ("answer", "refusal"),
    [
        (
            (
                1,
                "",
                "This project pins Claude Code X and none was found.\n"
                "  npm install -g @anthropic-ai/claude-code@X\n"
                "then run again.\n",
            ),
            "This project pins Claude Code X and none was found. "
            "npm install -g @anthropic-ai/claude-code@X then run again.",
        ),
        ((0, "not a directory\n", ""), "printed no directory"),
        ((3, "", ""), "exited 3"),
    ],
    ids=["refused", "no-directory", "silent"],
)
def test_no_pin_is_a_refusal_and_nothing_runs(
    recorder: Recorder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: tuple[int, str, str],
    refusal: str,
) -> None:
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.setattr(fw, "_probe", lambda argv: answer)
    monkeypatch.setitem(check.NEED_PROBES, "claude-pin", fw._need_claude_pin)
    assert check.main(["quick"]) == 2
    assert recorder.calls == []
    err = capsys.readouterr().err
    (line,) = (line for line in err.splitlines() if line.startswith("check: pytest: refused ("))
    assert refusal in line


_LOCK = {
    "name": "fixture",
    "lockfileVersion": 3,
    "packages": {
        "": {"name": "fixture", "devDependencies": {"a": "1.0.0"}},
        "node_modules/a": {"version": "1.0.0", "integrity": "sha512-a", "dev": True},
    },
}


def _lockfiles(root: Path, *, installed: str | None) -> None:
    """``package-lock.json`` under *root*, and npm's hidden lockfile recording
    package ``a`` at *installed*, or no ``node_modules`` when ``None``."""
    (root / "package-lock.json").write_text(json.dumps(_LOCK), encoding="utf-8")
    if installed is None:
        return
    hidden = json.loads(json.dumps(_LOCK))
    del hidden["packages"][""]
    hidden["packages"]["node_modules/a"]["version"] = installed
    (root / "node_modules").mkdir(exist_ok=True)
    (root / "node_modules/.package-lock.json").write_text(json.dumps(hidden), encoding="utf-8")


@pytest.fixture
def node_need(recorder: Recorder, monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """The real ``node`` need, with ``node`` and ``npm`` found and the recorder
    standing in for npm."""
    monkeypatch.setitem(check.NEED_PROBES, "node", fw._need_node)
    monkeypatch.setattr(fw.shutil, "which", lambda name: f"/usr/bin/{name}")
    return recorder


def test_npm_ci_does_not_run_when_node_modules_is_current(
    node_need: Recorder, tmp_path: Path
) -> None:
    _lockfiles(tmp_path, installed="1.0.0")
    assert check.main(["quick"]) == 0
    assert ["npm", "ci"] not in node_need.calls


def test_npm_ci_runs_before_any_check_when_node_modules_is_stale(
    node_need: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _lockfiles(tmp_path, installed="0.9.0")
    assert check.main(["quick"]) == 2
    assert node_need.calls[0] == ["npm", "ci"]
    assert ["uv", "lock", "--check"] not in node_need.calls
    captured = capsys.readouterr()
    assert "check: node: node_modules/a differs from package-lock.json; npm ci" in captured.out
    assert "node_modules still differs from package-lock.json after npm ci" in captured.err


def test_a_failed_npm_ci_is_a_refusal(
    node_need: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _lockfiles(tmp_path, installed="0.9.0")
    node_need.failing = {"npm ci": 1}
    assert check.main(["quick"]) == 2
    assert node_need.calls == [["npm", "ci"]]
    assert "npm ci failed (exit 1): node_modules/a differs" in capsys.readouterr().err


def test_a_run_without_uv_is_refused(stub_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (stub_path / "uv").unlink()
    assert check.main(["run", "lockfile"]) == 2
    assert "uv is not on PATH" in capsys.readouterr().err


@pytest.mark.usefixtures("stub_path")
def test_a_pytest_check_leaves_its_output_on_the_callers_stdout(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    assert check.main(["run", "pytest-pty", "--", "-q"]) == 0
    assert "stub uv says hello" in capfd.readouterr().out
    logged = (tmp_path / "uv.log").read_text(encoding="utf-8").strip()
    pty = shlex.join(BY_ID["pytest-pty"].argv)
    command = f"{shlex.join(kit.UV_RUN[1:])} python scripts/run_bounded.py 600 -- {pty} -q"
    written = re.fullmatch(re.escape(command) + r" --junitxml=(\S+) -o junit_family=xunit1", logged)
    assert written, logged
    assert written[1].endswith("/pytest-pty/junit.xml")


def test_run_refuses_an_unknown_check(
    recorder: Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert check.main(["run", "ruff", "pylint"]) == 2
    assert "unknown check: pylint" in capsys.readouterr().err
    assert recorder.calls == []


@pytest.mark.parametrize(
    "flags",
    [
        ["-m", "slow"],
        ["-k", "spinner"],
        ["--ignore=tests/cli"],
        ["--deselect=tests/cli/test_main.py::test_help"],
        ["tests/cli"],
        ["-n"],
        ["-n", "tests/cli"],
        ["-x"],
        ["--junitxml=report.xml"],
        ["-o", "junit_family=xunit2"],
        ["--reporter=junit"],
    ],
)
def test_run_refuses_a_flag_that_selects(
    recorder: Recorder, capsys: pytest.CaptureFixture[str], flags: list[str]
) -> None:
    assert check.main(["run", "pytest", "--", *flags]) == 2
    assert "not an execution flag" in capsys.readouterr().err
    assert recorder.calls == []


def test_run_refuses_a_report_flag_for_vitest(
    recorder: Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert check.main(["run", "vitest", "--", "--reporter=junit"]) == 2
    assert "not an execution flag" in capsys.readouterr().err
    assert recorder.calls == []


#: A flag pytest counts, so passing it again raises or lowers the level once more.
_COUNTED_VERBOSITY = re.compile(r"-v+|-q+|--verbose|--quiet")


def test_the_full_tier_does_not_double_the_configured_verbosity() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    addopts = shlex.split(pyproject["tool"]["pytest"]["ini_options"]["addopts"])
    configured = {word for word in addopts if _COUNTED_VERBOSITY.fullmatch(word)}
    assert configured, "addopts states no verbosity; this test pins nothing"
    repeated = {
        check_id: sorted(configured.intersection(flags))
        for check_id, flags in check._tier_flags("full").items()
        if configured.intersection(flags)
    }
    assert repeated == {}


def test_run_passes_execution_flags_to_the_pytest_checks_only(recorder: Recorder) -> None:
    flags = ["-q", "-n", "4", "--cov=src/osprey", "--cov-report=xml", "--maxfail=1", "-nauto"]
    assert check.main(["run", "ruff", "pytest", "--", *flags]) == 0
    ruff, sweep = recorder.calls
    assert ruff == [*fw.UV_RUN, "ruff", "check", "."]
    assert sweep[:-3][-len(flags) - 2 :] == [*flags, "--dist", "loadgroup"]
    assert sweep[-3].endswith("/pytest/junit.xml") and sweep[-2:] == ["-o", "junit_family=xunit1"]
    assert check.main(["run", "ruff", "--", "-v"]) == 2


@pytest.mark.usefixtures("recorder")
def test_run_needs_a_base_for_the_checks_that_judge_a_range(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert check.main(["run", "changelog", "plugin-version"]) == 2
    assert "changelog, plugin-version: needs --base BASE" in capsys.readouterr().err
    assert check.main(["run", "plugin-version"]) == 2
    assert "plugin-version: needs --base BASE" in capsys.readouterr().err


def test_fix_lets_ruff_write_before_the_checks_and_only_in_quick(
    recorder: Recorder,
) -> None:
    assert check.main(["quick", "--fix"]) == 0
    assert recorder.calls[:4] == [
        [*fw.UV_RUN, "ruff", "check", ".", "--fix"],
        [*fw.UV_RUN, "ruff", "format", "."],
        ["npm", "run", "format"],
        ["uv", "lock", "--check"],
    ]
    assert [list(argv) for argv in check.FIX_COMMANDS] == recorder.calls[:3]
    with pytest.raises(SystemExit) as refused:
        check.main(["full", "--fix"])
    assert refused.value.code == 2


@pytest.mark.usefixtures("any_base")
def test_fix_runs_only_the_fixes_of_the_checks_the_range_touches(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kit, "ROOT", REPO_ROOT)
    monkeypatch.setattr(
        check, "touched", lambda base, worktree=False, scope=False: ["scripts/check.py"]
    )
    monkeypatch.setattr(check, "repository_files", lambda worktree=False: FILES)
    assert check.main(["quick", "--changed", "B", "--fix"]) == 0
    assert not recorder.ran("npm run format")
    assert recorder.calls[:2] == [
        [*fw.UV_RUN, "ruff", "check", ".", "--fix"],
        [*fw.UV_RUN, "ruff", "format", "."],
    ]

    # A fix declared on a scoped check runs only when the range touches its scope.
    fixing = dataclasses.replace(BY_ID["eslint"], fix=(("npm", "run", "lint:fix"),))
    checks = tuple(fixing if c.id == "eslint" else c for c in check.CHECKS)
    monkeypatch.setattr(check, "CHECKS", checks)
    recorder.calls.clear()
    assert check.main(["quick", "--changed", "B", "--fix"]) == 0
    assert not recorder.ran("npm run lint:fix")
    monkeypatch.setattr(
        check, "touched", lambda base, worktree=False, scope=False: [WT + "settings.js"]
    )
    recorder.calls.clear()
    assert check.main(["quick", "--changed", "B", "--fix"]) == 0
    assert recorder.calls[2] == ["npm", "run", "lint:fix"]


@pytest.mark.usefixtures("any_base")
@pytest.mark.parametrize("flags", [[], ["--base", "B"]])
def test_the_working_tree_flag_needs_a_changed_range(
    flags: list[str], recorder: Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert check.main(["quick", "--worktree", *flags]) == 2
    assert recorder.calls == []
    assert "--worktree needs --changed BASE" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The checks implemented in the command


def test_a_redirect_without_its_stub_page_is_reported(tmp_path: Path) -> None:
    (tmp_path / "guides").mkdir()
    (tmp_path / "guides" / "old.html").write_text("", encoding="utf-8")
    assert fw.missing_redirect_stubs(["guides/old", "guides/lost"], tmp_path) == ["guides/lost"]


@pytest.mark.usefixtures("recorder")
def test_the_docs_check_fails_when_a_redirect_stub_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kit, "capture", lambda argv: (0, "old/page\n"))
    assert check.main(["run", "docs"]) == 1
    stub_page = tmp_path / fw.DOCS_HTML / "old" / "page.html"
    stub_page.parent.mkdir(parents=True)
    stub_page.write_text("", encoding="utf-8")
    assert check.main(["run", "docs"]) == 0


def test_the_docs_check_cleans_the_build_tree_before_it_builds(
    recorder: Recorder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    where: list[Path | None] = []

    def placed(argv: Any, cwd: Path | None = None) -> int:
        where.append(cwd)
        return recorder(argv, cwd)

    monkeypatch.setattr(kit, "run", placed)
    assert check.main(["run", "docs"]) == 0
    assert recorder.calls == [
        [*fw.UV_RUN, "make", "clean"],
        [*fw.UV_RUN, "make", "html", "SPHINXOPTS=-W --keep-going"],
    ]
    assert where == [tmp_path / "docs", tmp_path / "docs"]

    recorder.calls.clear()
    recorder.failing = {"make clean": 2}
    assert check.main(["run", "docs"]) == 1
    assert recorder.calls == [[*fw.UV_RUN, "make", "clean"]]


@pytest.mark.parametrize("check_id", ["pytest", "pytest-pty"])
def test_a_pytest_check_prunes_stale_bytecode_before_it_runs(
    check_id: str, recorder: Recorder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "src" / "osprey"
    removed = package / "removed" / "inner"
    kept = package / "kept"
    outside = tmp_path / "tests" / "__pycache__"
    for cache in (removed / "__pycache__", kept / "__pycache__", outside):
        cache.mkdir(parents=True)
        (cache / "module.cpython-311.pyc").write_bytes(b"")
    (kept / "module.py").write_text("", encoding="utf-8")
    seen: list[bool] = []

    def running(argv: Any, cwd: Path | None = None) -> int:
        seen.append((package / "removed").exists())
        return recorder(argv, cwd)

    monkeypatch.setattr(kit, "run", running)
    assert check.main(["run", check_id]) == 0
    assert seen == [False]
    assert (kept / "module.py").is_file() and not (kept / "__pycache__").exists()
    assert outside.is_dir()


def test_a_check_that_imports_nothing_leaves_the_bytecode(
    recorder: Recorder, tmp_path: Path
) -> None:
    cache = tmp_path / "src" / "osprey" / "kept" / "__pycache__"
    cache.mkdir(parents=True)
    assert check.main(["run", "lockfile"]) == 0
    assert recorder.ran("uv lock --check") and cache.is_dir()


def test_the_package_check_builds_into_an_empty_dist(
    recorder: Recorder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = tmp_path / "dist" / "osprey-0.0.1.tar.gz"
    stale.parent.mkdir()
    stale.write_text("", encoding="utf-8")
    seen: list[bool] = []

    def building(argv: Any, cwd: Path | None = None) -> int:
        if list(argv) == ["uv", "build"]:
            seen.append(stale.exists())
            (tmp_path / "dist").mkdir()
            (tmp_path / "dist" / "osprey-1.0.tar.gz").write_text("", encoding="utf-8")
            (tmp_path / "dist" / "osprey_connectors-1.0-py3-none-any.whl").write_text("")
        return recorder(argv, cwd)

    monkeypatch.setattr(kit, "run", building)
    assert check.main(["run", "package"]) == 0
    assert seen == [False]
    build, twine, connectors, venv, install, probe, *rest = recorder.calls
    wheel_probe, pyaml_build, pyaml_venv, pyaml_install, pyaml_probe = rest
    assert twine == [
        "uvx",
        "twine",
        "check",
        "dist/osprey-1.0.tar.gz",
        "dist/osprey_connectors-1.0-py3-none-any.whl",
    ]
    assert connectors == ["uv", "build", "--package", "osprey-connectors"]
    python = f"{venv[2]}/bin/python"
    assert venv[:2] == ["uv", "venv"] and not Path(venv[2]).exists()
    assert Path(venv[2]).parent == Path(pyaml_venv[2]).parent
    assert install == [
        "uv",
        "pip",
        "install",
        "--python",
        python,
        "dist/osprey_connectors-1.0-py3-none-any.whl",
    ]
    assert probe == [python, "-I", fw.LEAN_IMPORT_WALK]
    # A glob no built file matches is passed on as written, for the step to fail on.
    assert wheel_probe == [
        "python3",
        "-c",
        fw.DOCK_ENGINE_WHEEL_PROBE,
        "dist/osprey_framework-*.whl",
    ]
    assert pyaml_build == ["uv", "build", "--package", "pyaml-cs-osprey"]
    pyaml_python = f"{pyaml_venv[2]}/bin/python"
    assert pyaml_install[:5] == ["uv", "pip", "install", "--python", pyaml_python]
    assert pyaml_probe == [pyaml_python, "-c", fw.PYAML_WHEEL_PROBE]


def test_the_package_check_stops_at_the_first_failing_step(recorder: Recorder) -> None:
    recorder.failing = {"pyaml-cs-osprey": 3}
    assert check.main(["run", "package"]) == 1
    assert recorder.calls[-1] == ["uv", "build", "--package", "pyaml-cs-osprey"]


def test_the_package_check_reads_twines_exit_and_stops(recorder: Recorder) -> None:
    recorder.failing = {"twine check": 1}
    assert check.main(["run", "package"]) == 1
    assert [argv[:2] for argv in recorder.calls] == [["uv", "build"], ["uvx", "twine"]]


# ---------------------------------------------------------------------------
# The declaration as data


def test_list_json_round_trips(capsys: pytest.CaptureFixture[str]) -> None:
    assert check.main(["list", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)

    def rebuilt(item: dict[str, Any]) -> Any:
        fields = {
            key: tuple(value) if isinstance(value, list) else value for key, value in item.items()
        }
        fix = tuple(tuple(argv) for argv in item["fix"])
        worktree = tuple(tuple(line) for line in item["worktree"])
        return check.Check(
            **{**fields, "tiers": frozenset(item["tiers"]), "fix": fix, "worktree": worktree}
        )

    derived = {
        item["id"]: item["scope"] for item in data["checks"] if BY_ID[item["id"]].derive_scope
    }
    assert derived == {}
    declared = [{**item, "scope": []} if item["id"] in derived else item for item in data["checks"]]
    assert tuple(rebuilt(item) for item in declared) == check.CHECKS
    assert "touched" not in data
    assert tuple(check.Lane(i["job"], i["why"], tuple(i["scope"])) for i in data["lanes"]) == (
        check.LANES
    )
    assert data["tiers"] == ["quick", "full"]
    unit = data["unit_selection"]
    assert [tuple(pair) for pair in unit["ignores"]] == list(fw.UNIT_SELECTION.ignores)
    assert unit["marker"] == fw.UNIT_SELECTION.marker


@pytest.mark.usefixtures("any_base")
@pytest.mark.parametrize(
    ("changed", "hit"),
    [("src/osprey/runtime/budget.py", True), ("docs/source/index.rst", False)],
)
def test_list_changed_names_the_checks_and_lanes_a_range_touches(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], changed: str, hit: bool
) -> None:
    monkeypatch.setattr(check, "touched", lambda base, worktree=False, scope=False: [changed])
    assert check.main(["list", "--changed", "B", "--json"]) == 0
    touched = json.loads(capsys.readouterr().out)["touched"]
    assert ("pytest-wallclock" in touched) is hit
    assert ("deploy-e2e" in touched) is hit
    assert "lockfile" in touched and "pytest-pty" not in touched
    assert check.main(["list", "--changed", "B"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == f"touched: {', '.join(touched)}"
    assert check.resolve(["list", "--changed", "B"]) == []


@pytest.mark.usefixtures("repo")
def test_list_changed_refuses_a_base_that_does_not_resolve(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert check.main(["list", "--changed", "no-such-ref"]) == 2
    assert "no-such-ref does not resolve" in capsys.readouterr().err


def test_list_filters_by_tier(capsys: pytest.CaptureFixture[str]) -> None:
    assert check.main(["list", "--tier", "quick", "--json"]) == 0
    listed = [item["id"] for item in json.loads(capsys.readouterr().out)["checks"]]
    assert listed == [c.id for c in check.CHECKS if "quick" in c.tiers]
    assert check.main(["list", "--tier", "quick"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines] == listed


def test_the_command_needs_only_the_standard_library() -> None:
    # -I ignores the environment and the user site; -S leaves site-packages off the path.
    done = subprocess.run(
        [sys.executable, "-I", "-S", str(REPO_ROOT / "scripts" / "check.py"), "list"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[0].split()[:3] == ["lockfile", "invariant", "quick,full"]


# ---------------------------------------------------------------------------
# Checks the retired shell gates ran that CI's lint job also runs


def test_a_tracked_file_that_is_gitignored_fails_and_is_named(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    listings: list[tuple[str, ...]] = []

    def listing(argv: Any) -> tuple[int, str]:
        listings.append(tuple(argv))
        return 0, "build/kept.txt\n"

    monkeypatch.setattr(kit, "capture", listing)
    assert check.main(["run", "tracked-ignored"]) == 1
    assert listings == [fw.TRACKED_IGNORED]
    assert fw.TRACKED_IGNORED == ("git", "ls-files", "-ci", "--exclude-standard")
    assert "build/kept.txt" in capsys.readouterr().out
    assert recorder.calls == []


@pytest.mark.usefixtures("recorder")
def test_a_tracked_file_that_is_gitignored__mutation_nothing_listed_is_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kit, "capture", lambda argv: (0, ""))
    assert check.main(["run", "tracked-ignored"]) == 0
    monkeypatch.setattr(kit, "capture", lambda argv: (128, ""))
    assert check.main(["run", "tracked-ignored"]) == 1


def test_the_docs_build_refuses_without_graphviz(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(check.NEED_PROBES, "graphviz", fw._need_graphviz)
    monkeypatch.setattr(fw.shutil, "which", lambda name: None)
    assert check.main(["run", "docs"]) == 2
    assert "graphviz is not on PATH" in capsys.readouterr().err
    assert recorder.calls == []
    monkeypatch.setattr(fw.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert check.main(["run", "docs"]) == 0


def test_the_dock_engine_steps_run_in_both_tiers() -> None:
    for check_id in ("dock-engine-bundle", "dock-engine-tests"):
        declared = BY_ID[check_id]
        assert declared.tiers == check.BOTH and declared.needs == ("node",)
        assert fw.DOCK_ENGINE in declared.scope
    assert fw.DOCK_ENGINE in fw.TSC_SCOPE
    got = selected("quick", "packages/dock-engine/src/rules.ts")
    assert got["dock-engine-bundle"].state == got["dock-engine-tests"].state == check.WHOLE
    got = selected("quick", "src/osprey/cli/main.py")
    assert got["dock-engine-bundle"].state == got["dock-engine-tests"].state == check.NOT_TOUCHED


# ---------------------------------------------------------------------------
# Each owner declares its checks; the command collects them

#: The framework's module in a scratch tree: one check, so the tree has the
#: module every collection starts from.
FIXTURE_FRAMEWORK = """\
from check_kit import BOTH, Check

CHECKS = (Check("lockfile", "invariant", BOTH, ("uv", "lock", "--check")),)
CALLER_JOBS = ("lint",)
"""

#: A module an integration branch would bring: a check of its own, with a lane,
#: a caller job and a need, and nothing in scripts/check.py that names it.
FIXTURE_OWNER = """\
from check_kit import FULL, Check, Lane

def _need_widget():
    return None

CHECKS = (
    Check(
        "widget-contract",
        "tests",
        FULL,
        ("uv", "run", "pytest", "tests/widget"),
        scope=("src/widget/**",),
        needs=("widget",),
        bound_s=300,
        exec_flags=True,
        tier_flags={"full": ("--tb=short",)},
    ),
)
LANES = (Lane("widget-e2e", "containers", ("src/widget/**",)),)
CALLER_JOBS = ("widget-contract",)
NEEDS = {"widget": _need_widget}
"""


def _owner_tree(root: Path, modules: dict[str, str], *, tracked: bool = True) -> None:
    """A git tree at *root* holding *modules*, committed when *tracked*."""
    git(root, "init", "--initial-branch", "main")
    for path, source in modules.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(source, encoding="utf-8")
    if tracked:
        git(root, "add", "-A")


def _collected(monkeypatch: pytest.MonkeyPatch, root: Path) -> Any:
    """The declaration of the tree at *root*, installed as the command's own."""
    declared = check.collect(check.owner_modules(root), root)
    for name, value in (
        ("DECLARED", declared),
        ("CHECKS", declared.checks),
        ("LANES", declared.lanes),
        ("CALLER_JOBS", declared.caller_jobs),
        ("NEED_PROBES", declared.needs),
    ):
        monkeypatch.setattr(check, name, value)
    return declared


def test_a_module_the_branch_adds_contributes_its_check_without_editing_check_py(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A branch adds a check by adding a ``repo_checks.py`` beside its code:
    the command collects it, after the framework's, with its lane, its caller
    job and its need, and runs it from its record."""
    _owner_tree(
        tmp_path,
        {
            check.FRAMEWORK_MODULE: FIXTURE_FRAMEWORK,
            "src/widget/repo_checks.py": FIXTURE_OWNER,
        },
    )
    declared = _collected(monkeypatch, tmp_path)
    assert [c.id for c in declared.checks] == ["lockfile", "widget-contract"]
    assert declared.owners == {
        "lockfile": check.FRAMEWORK_MODULE,
        "widget-contract": "src/widget/repo_checks.py",
    }
    assert [lane.job for lane in declared.lanes] == ["widget-e2e"]
    assert declared.caller_jobs == ("lint", "widget-contract")
    assert check.resolve(["run", "widget-contract", "--", "-v"]) == [
        [*fw_bound(300), "uv", "run", "pytest", "tests/widget", "-v"]
    ]
    (line,) = [argv for argv in check.resolve(["full"]) if "tests/widget" in argv]
    assert line[-1] == "--tb=short"
    assert check.select("full", ["src/widget/a.py"]).lanes == ("widget-e2e",)
    assert check.main(["list", "--tier", "full"]) == 0
    assert "widget-contract" in capsys.readouterr().out


def fw_bound(seconds: int) -> list[str]:
    """The prefix of a command stopped at *seconds*."""
    return kit.bounded(seconds, [])


def test_an_untracked_module_is_collected(tmp_path: Path) -> None:
    """A module not yet added to git declares its checks already, so a check is
    judged in the change that adds it."""
    _owner_tree(tmp_path, {check.FRAMEWORK_MODULE: FIXTURE_FRAMEWORK})
    (tmp_path / "src/widget").mkdir(parents=True)
    (tmp_path / "src/widget/repo_checks.py").write_text(FIXTURE_OWNER, encoding="utf-8")
    assert check.owner_modules(tmp_path) == [check.FRAMEWORK_MODULE, "src/widget/repo_checks.py"]


def test_the_framework_module_is_collected_first(tmp_path: Path) -> None:
    """The framework's checks hold for the whole repository and check the
    lockfile before any ``uv run``, so they come first, whatever sorts before."""
    _owner_tree(
        tmp_path,
        {"aaa/repo_checks.py": FIXTURE_OWNER, check.FRAMEWORK_MODULE: FIXTURE_FRAMEWORK},
    )
    assert check.owner_modules(tmp_path) == [check.FRAMEWORK_MODULE, "aaa/repo_checks.py"]


def test_a_tree_without_the_framework_module_is_refused(tmp_path: Path) -> None:
    _owner_tree(tmp_path, {"src/widget/repo_checks.py": FIXTURE_OWNER})
    with pytest.raises(check.Refused, match="scripts/repo_checks.py is missing"):
        check.owner_modules(tmp_path)


@pytest.mark.parametrize(
    ("second", "refusal"),
    [
        (FIXTURE_FRAMEWORK, "check lockfile is declared by both"),
        (FIXTURE_OWNER.replace('"widget-contract",\n', '"other",\n', 1), "lane widget-e2e"),
        (
            FIXTURE_OWNER.replace("widget-contract", "other").replace("widget-e2e", "other-e2e"),
            "need widget is declared by both",
        ),
        ("X = 1\n", "declares no CHECKS"),
    ],
)
def test_what_two_modules_declare_has_one_owner(tmp_path: Path, second: str, refusal: str) -> None:
    """A check, a lane or a need belongs to one module; a second declaration of
    it is refused, not merged."""
    _owner_tree(
        tmp_path,
        {
            check.FRAMEWORK_MODULE: FIXTURE_FRAMEWORK,
            "src/widget/repo_checks.py": FIXTURE_OWNER,
            "src/zz/repo_checks.py": second,
        },
    )
    with pytest.raises(check.Refused, match=refusal):
        check.collect(check.owner_modules(tmp_path), tmp_path)


def test_every_checks_module_shapes_every_verdict() -> None:
    """A change to a checks module changes what the command runs, so it selects
    the command's own tests, as a change to check.py does."""
    for path in (*check.DECLARED.modules, "scripts/check.py", "scripts/check_kit.py"):
        assert path in check.SHAPING_SCRIPTS
        assert fw._select_pytest([path], [], check.SHAPING_SCRIPTS) == check.Selection(
            check.PATHS, ("tests/scripts",)
        )


def test_no_check_is_keyed_by_its_id_in_the_command() -> None:
    """Everything the command does with a check is read from its record: no
    table keyed by a check id and no branch on one is left in the command."""
    source = (REPO_ROOT / "scripts/check.py").read_text(encoding="utf-8")
    keyed = re.findall(r"check\.id (?:==|in) [\"'(]", source)
    named = sorted(
        {c.id for c in check.CHECKS if re.search(rf"[\"']{re.escape(c.id)}[\"']", source)}
    )
    assert keyed == [] and named == []


def test_the_connectors_package_declares_its_own_lanes() -> None:
    """The lanes that install the connectors on their own are the package's,
    declared beside it and collected after the framework's."""
    connectors = "packages/osprey-connectors/repo_checks.py"
    assert list(check.DECLARED.modules) == [check.FRAMEWORK_MODULE, connectors]
    own = {lane.job for lane in check.DECLARED.modules[connectors].LANES}
    assert own == {"connectors-standalone", "connectors-compat-plan", "connectors-compat"}
    assert own <= {lane.job for lane in check.LANES}
    assert not own & {lane.job for lane in fw.LANES}
