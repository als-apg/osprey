"""The ``canopy`` marker reaches every CANOPY test, and none of them can skip.

These run without CANOPY. They read every test module and exercise the
helper that marks them, so the default sweep keeps the promise the contract
relies on: a test that needs CANOPY runs or fails, in every environment.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests import _canopy_contract as contract
from tests._tree_scan import python_sources

TESTS = Path(__file__).resolve().parents[2]

_SKIP_CALLS = ("skip", "skipif", "importorskip")
_CANOPY_WORDS = ("canopy", "pydmconverter")


@cache
def _every_module() -> dict[str, str]:
    """Every Python module under tests/ with its text, keyed by its path relative to tests/."""
    return dict(python_sources(Path(__file__)))


def _parsed(relative: str) -> ast.Module:
    """The module at *relative*, parsed afresh.

    Not cached: every tree of the suite kept for the worker's lifetime is a heap
    that each later full ``gc.collect()`` on that worker has to walk.
    """
    return ast.parse(_every_module()[relative], filename=relative)


def _defines(tree: ast.Module) -> set[str]:
    return {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _calls(tree: ast.AST, names: tuple[str, ...]) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and contract.dotted(node.func).rsplit(".", 1)[-1] in names
    ]


def _names_canopy(call: ast.Call) -> bool:
    """Whether any string inside *call*'s arguments names a CANOPY package."""
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and any(word in node.value.lower() for word in _CANOPY_WORDS)
        for argument in [*call.args, *(keyword.value for keyword in call.keywords)]
        for node in ast.walk(argument)
    )


def importorskip_sites(tree: ast.AST) -> list[int]:
    return [call.lineno for call in _calls(tree, ("importorskip",))]


def canopy_skip_sites(tree: ast.AST) -> list[int]:
    """Lines where a skip is decided by whether CANOPY is there."""
    return [call.lineno for call in _calls(tree, _SKIP_CALLS) if _names_canopy(call)]


def could_skip_on_canopy(text: str) -> bool:
    """Whether *text* can hold a site ``canopy_skip_sites`` reports, read off the raw text.

    Such a site is a call whose name ends in ``skip`` and a string argument naming
    a CANOPY package, so a module without both words has none and is not parsed.
    """
    lowered = text.lower()
    return "skip" in lowered and any(word in lowered for word in _CANOPY_WORDS)


def canopy_skip_modules(modules: dict[str, str]) -> dict[str, list[int]]:
    """The modules among *modules* (path to text) with a CANOPY-decided skip, and its lines."""
    sites = {
        relative: canopy_skip_sites(ast.parse(text, filename=relative))
        for relative, text in modules.items()
        if could_skip_on_canopy(text)
    }
    return {relative: lines for relative, lines in sites.items() if lines}


hand_marked = contract.hand_marks


# --- every test module -----------------------------------------------------------------


def _members(kind: str | None = None) -> list[str]:
    members = contract.contract_members(TESTS)
    return [relative for relative in members if kind is None or members[relative] == kind]


def test_no_canopy_module_calls_importorskip() -> None:
    """A module-level ``importorskip`` skips at collection, before the refusal can act."""
    sites = {relative: importorskip_sites(_parsed(relative)) for relative in _members()}
    assert {name: lines for name, lines in sites.items() if lines} == {}


def test_no_skip_here_is_decided_by_whether_canopy_is_installed() -> None:
    assert canopy_skip_modules(_every_module()) == {}


def test_a_hand_marked_module_is_not_also_marked_from_its_imports() -> None:
    """A hand mark exists only where the module itself imports without CANOPY."""
    assert [relative for relative in _members("derived") if hand_marked(_parsed(relative))] == []


def test_the_guards_see_what_they_guard_against() -> None:
    tree = ast.parse(
        "import subprocess, sys\n"
        "import pytest\n"
        "pytest.importorskip('uvicorn')\n"
        "gate = pytest.mark.skipif(\n"
        "    subprocess.run([sys.executable, '-c', 'import canopy_mcp.report']).returncode != 0,\n"
        "    reason='not there',\n"
        ")\n"
        "def test_a():\n"
        "    pytest.skip('PyDMConverter is missing')\n"
        "def test_b():\n"
        "    pytest.skip('no display')\n"
        "@pytest.mark.canopy\n"
        "def test_c(): ...\n"
        "class TestD:\n"
        "    @pytest.mark.canopy\n"
        "    def test_e(self): ...\n"
        "    @pytest.mark.slow\n"
        "    def test_f(self): ...\n"
    )
    assert importorskip_sites(tree) == [3]
    assert canopy_skip_sites(tree) == [4, 9]
    assert hand_marked(tree) == {"test_c", "TestD.test_e"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("import pytest\npytest.importorskip('canopy')\n", {"m.py": [2]}),
        ("from pytest import skip\ndef test_a():\n    skip('CANOPY absent')\n", {"m.py": [3]}),
        (
            "import pytest\ngate = pytest.mark.skipif(True, reason='no PyDMConverter')\n",
            {"m.py": [2]},
        ),
        ("import pytest\ndef test_a():\n    pytest.skip('no display')\n", {}),
        ("import canopy\ndef test_a():\n    assert canopy\n", {}),
    ],
    ids=["importorskip", "bare-skip", "skipif-pydmconverter", "skip-not-canopy", "no-skip"],
)
def test_the_text_filter_keeps_every_module_the_skip_guard_reports(
    text: str, expected: dict[str, list[int]]
) -> None:
    """Reading the raw text first drops no module the parsed guard would report."""
    assert canopy_skip_modules({"m.py": text}) == expected
    every = {"m.py": canopy_skip_sites(ast.parse(text))}
    assert {name: lines for name, lines in every.items() if lines} == expected


# --- which modules need CANOPY ----------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import canopy\n", True),
        ("import canopy_mcp.report\n", True),
        ("from canopy.x import y\n", True),
        ("from pydmconverter import convert\n", True),
        ("class TestA:\n    import canopy_mcp\n", True),
        ("try:\n    import canopy\nexcept ImportError:\n    canopy = None\n", True),
        ("def test_a():\n    import canopy\n", False),
        ("async def test_a():\n    from canopy.x import y\n", False),
        ("class TestA:\n    def test_a(self):\n        import canopy_mcp\n", False),
        ("import canopyx\n", False),
        ("from osprey.integrations.canopy import push\n", False),
        ("from . import canopy\n", False),
        ("COMMAND = 'python -m canopy_mcp.report'\n", False),
    ],
)
def test_needs_canopy_reads_import_time_imports_only(
    tmp_path: Path, source: str, expected: bool
) -> None:
    module = tmp_path / "test_module.py"
    module.write_text(source, encoding="utf-8")
    assert contract.needs_canopy(module) is expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"def (:\n", False),
        (b"import canopy (\n", False),
        (b"\xff\xfe not utf-8 canopy\n", False),
        (b"import canopy\n", True),
    ],
    ids=["unparseable", "unparseable-naming-canopy", "undecodable", "canopy-import"],
)
def test_needs_canopy_leaves_an_unreadable_module_to_pytest(
    tmp_path: Path, data: bytes, expected: bool
) -> None:
    """A module that does not parse is pytest's to report, not the marker's."""
    module = tmp_path / "test_module.py"
    module.write_bytes(data)
    assert contract.needs_canopy(module) is expected


# --- what a marker expression leaves out ------------------------------------------------


@pytest.mark.parametrize(
    ("markexpr", "expected"),
    [
        ("not canopy", True),
        ("not pty and not canopy", True),
        ("not slow and not pty and not canopy", True),
        ("not pty", False),
        ("", False),
        ("canopy", False),
    ],
)
def test_deselects_canopy(markexpr: str, expected: bool) -> None:
    assert contract.deselects_canopy(markexpr) is expected


@pytest.mark.parametrize(
    ("markexpr", "lines"),
    [
        ("not pty and not canopy", [contract.NOT_EXERCISED_TEXT]),
        ("not pty", []),
        ("", []),
    ],
)
def test_summary_names_a_run_that_left_the_contract_out(markexpr: str, lines: list[str]) -> None:
    config = SimpleNamespace(option=SimpleNamespace(markexpr=markexpr))
    assert contract.summary_lines(config, {}) == lines


def test_the_not_exercised_line_starts_as_the_ci_summary_reads_it() -> None:
    assert contract.NOT_EXERCISED_TEXT.startswith("CANOPY contract not exercised here")


def test_the_not_exercised_line_names_the_command_that_runs_the_contract() -> None:
    assert contract.NOT_EXERCISED_TEXT == (
        "CANOPY contract not exercised here: run uv run python scripts/check.py run canopy-contract"
    )


# --- the stand-in ---------------------------------------------------------------------


@pytest.fixture
def canopy_module(tmp_path: Path) -> Path:
    module = tmp_path / "test_needs_it.py"
    module.write_text("import canopy\n\n\ndef test_a():\n    pass\n", encoding="utf-8")
    return module


def test_without_canopy_a_module_is_one_failing_marked_item(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, canopy_module: Path
) -> None:
    monkeypatch.setattr(contract, "canopy_installed", lambda: False)
    node = contract.module_node(canopy_module, request.session)
    assert isinstance(node, contract.CanopyAbsent)
    (item,) = node.collect()
    assert item.name == "canopy_not_installed"
    assert item.nodeid.endswith("test_needs_it.py::canopy_not_installed")
    assert item.get_closest_marker("canopy") is not None
    with pytest.raises(pytest.fail.Exception) as failure:
        item.runtest()
    assert str(failure.value) == contract.ABSENT_TEXT
    assert failure.value.pytrace is False


def test_with_canopy_a_module_is_collected_as_itself_and_marked(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, canopy_module: Path
) -> None:
    monkeypatch.setattr(contract, "canopy_installed", lambda: True)
    node = contract.module_node(canopy_module, request.session)
    assert type(node) is pytest.Module
    assert node.get_closest_marker("canopy") is not None


def test_a_module_without_a_canopy_import_is_left_to_pytest(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    module = tmp_path / "test_plain.py"
    module.write_text("def test_a():\n    pass\n", encoding="utf-8")
    assert contract.module_node(module, request.session) is None


def test_the_absent_text_names_both_ways_out() -> None:
    assert "install it at the commit" in contract.ABSENT_TEXT
    assert '-m "not canopy"' in contract.ABSENT_TEXT


# --- refusing to skip -------------------------------------------------------------------


def _item(*markers: str) -> SimpleNamespace:
    return SimpleNamespace(
        get_closest_marker=lambda name: object() if name in markers else None,
        stash={},
    )


def _skipped(**extra: object) -> SimpleNamespace:
    return SimpleNamespace(
        skipped=True,
        outcome="skipped",
        longrepr=("test_x.py", 3, "Skipped: no display"),
        **extra,
    )


def test_a_skipped_canopy_test_is_reported_failed_with_its_reason() -> None:
    report = _skipped()
    contract.refuse_skip(_item("canopy"), report)
    assert report.outcome == "failed"
    assert report.longrepr == "a CANOPY contract test may not skip: Skipped: no display"


def test_a_skip_of_any_other_test_stands() -> None:
    report = _skipped()
    contract.refuse_skip(_item("slow"), report)
    assert report.outcome == "skipped"


def test_an_expected_failure_of_a_canopy_test_stands() -> None:
    report = _skipped(wasxfail="known")
    contract.refuse_skip(_item("canopy"), report)
    assert report.outcome == "skipped"


def test_a_passed_canopy_test_is_left_alone() -> None:
    report = SimpleNamespace(skipped=False, outcome="passed", longrepr=None)
    contract.refuse_skip(_item("canopy"), report)
    assert report.outcome == "passed"


# --- the hooks live in the root conftest -------------------------------------------------

_HOOKS = {"pytest_pycollect_makemodule", "pytest_runtest_makereport"}


def test_the_root_conftest_carries_both_hooks_and_no_other_conftest_does() -> None:
    conftests = [relative for relative in _every_module() if Path(relative).name == "conftest.py"]
    carriers = {
        relative: _HOOKS & _defines(_parsed(relative))
        for relative in conftests
        if _HOOKS & _defines(_parsed(relative))
    }
    assert carriers == {"conftest.py": _HOOKS}


def _root_conftest(request: pytest.FixtureRequest) -> object:
    root = str(TESTS / "conftest.py")
    (plugin,) = [
        p for p in request.config.pluginmanager.get_plugins() if getattr(p, "__file__", "") == root
    ]
    return plugin


def test_the_root_hooks_mark_and_refuse_outside_the_canopy_directory(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    canopy_module: Path,
) -> None:
    conftest = _root_conftest(request)
    monkeypatch.setattr(contract, "canopy_installed", lambda: False)
    node = conftest.pytest_pycollect_makemodule(module_path=canopy_module, parent=request.session)
    assert isinstance(node, contract.CanopyAbsent)
    (item,) = node.collect()
    assert item.get_closest_marker("canopy") is not None

    report = _skipped(when="call")
    wrapper = conftest.pytest_runtest_makereport(_item("canopy"), None)
    next(wrapper)
    with pytest.raises(StopIteration) as finished:
        wrapper.send(report)
    assert finished.value.value is report
    assert report.outcome == "failed"
