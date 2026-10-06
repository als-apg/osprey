"""Tests for the type-check gate that fails on any error over the declared trees.

None of these runs a type checker. What the gate has to get right: the verdict is
mypy's exit status, taken over the trees ``[tool.mypy] files`` names; a located error
is never hidden from the report; and a run taken without the stub distributions is
refused rather than judged.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

# scripts/ is not a package, so the gate is loaded by path. It is deliberately not
# registered in sys.modules: nothing in it resolves its own module name, and an
# import-time write to sys.modules is process state no fixture can undo.
_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "mypy_gate.py"
_spec = importlib.util.spec_from_file_location("mypy_gate", _MODULE_PATH)
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def test_notes_are_not_errors() -> None:
    """mypy's explanatory notes carry no verdict and must not enter the measurement."""
    output = (
        'src/osprey/a.py:10: error: Returning Any from function declared to return "int"'
        "  [no-any-return]\n"
        'src/osprey/a.py:10: note: Superclass declares "int"\n'
        "src/osprey/a.py:11: note: See https://mypy.readthedocs.io/\n"
    )
    assert gate.parse_errors(output) == [
        (
            "src/osprey/a.py",
            "no-any-return",
            'Returning Any from function declared to return "int"',
        )
    ]


def test_a_located_error_without_a_code_is_kept() -> None:
    """An error mypy emits without a code still counts as an error."""
    assert gate.parse_errors("src/osprey/a.py:3: error: Unsupported thing\n") == [
        ("src/osprey/a.py", "", "Unsupported thing")
    ]


class _FakeCompleted:
    def __init__(self, stdout: str, returncode: int = 1, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


class _FakeSubprocess:
    def __init__(self, completed: _FakeCompleted) -> None:
        self._completed = completed
        self.calls: list[list[str]] = []

    def run(self, *args: Any, **_kwargs: Any) -> _FakeCompleted:
        self.calls.append(list(args[0]))
        return self._completed


def _fake(monkeypatch: Any, completed: _FakeCompleted) -> _FakeSubprocess:
    fake = _FakeSubprocess(completed)
    monkeypatch.setattr(gate, "subprocess", fake)
    return fake


def test_a_clean_run_passes(monkeypatch: Any) -> None:
    """A tree mypy reports nothing on is green."""
    _fake(monkeypatch, _FakeCompleted("", returncode=0))
    assert gate.main([]) == 0


def test_any_error_fails_the_gate(monkeypatch: Any, capsys: Any) -> None:
    """One error is enough, and the report names it with its line number."""
    line = 'src/osprey/c.py:7: error: Name "x" is not defined  [name-defined]'
    _fake(monkeypatch, _FakeCompleted(line + "\n", returncode=1))

    assert gate.main([]) == 1
    out = capsys.readouterr().out
    assert line in out
    assert "1 error(s)" in out


def test_a_failing_exit_without_a_located_error_still_fails(monkeypatch: Any) -> None:
    """The exit status decides; an output line the parse misses can never read as clean."""
    _fake(monkeypatch, _FakeCompleted("pyproject.toml: error: bad option\n", returncode=1))
    assert gate.main([]) == 1


def test_a_checker_crash_is_refused(monkeypatch: Any, capsys: Any) -> None:
    """A checker exiting with anything but 0 or 1 checked nothing."""
    _fake(monkeypatch, _FakeCompleted("", returncode=2, stderr="Traceback ..."))
    assert gate.main([]) == 2
    assert "nothing was checked" in capsys.readouterr().out


def test_a_run_without_stub_packages_is_refused_rather_than_scored(
    monkeypatch: Any, capsys: Any
) -> None:
    """A checker without the declared stubs reports a weaker, larger error set.

    Judging it would blame the code for an environment fault, so the gate refuses and
    names the command that restores the environment.
    """
    output = (
        'src/osprey/a.py:3: error: Library stubs not installed for "yaml"  [import-untyped]\n'
        'src/osprey/a.py:10: error: Returning Any from function declared to return "int"'
        "  [no-any-return]\n"
    )
    _fake(monkeypatch, _FakeCompleted(output))

    assert gate.main([]) == 2
    assert "uv sync --extra dev" in capsys.readouterr().out


def test_the_refresh_flag_is_gone() -> None:
    """No escape hatch survives: the gate takes no option to record errors."""
    with pytest.raises(SystemExit) as excinfo:
        gate.main(["--update"])
    assert excinfo.value.code == 2


def test_no_tolerated_errors_file_remains() -> None:
    """An error is fixed where it is reported, never listed in a side file."""
    assert not (gate.REPO_ROOT / "scripts" / "mypy_baseline.json").exists()


def test_the_checker_runs_over_the_declared_trees(monkeypatch: Any) -> None:
    """The subprocess gets exactly the trees ``[tool.mypy] files`` names."""
    fake = _fake(monkeypatch, _FakeCompleted("", returncode=0))
    gate.main([])
    assert fake.calls == [
        [sys.executable, "-m", "mypy", *gate.declared_targets(gate.PYPROJECT), "--no-error-summary"]
    ]


def test_the_gate_checks_the_declared_trees() -> None:
    """The build names no trees of its own, so it cannot drift from a local run."""
    assert gate.declared_targets(gate.PYPROJECT) == ["src", "packages/osprey-connectors/src"]
