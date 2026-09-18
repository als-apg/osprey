"""Tests for the type-check gate that scores a run against a written-down baseline.

None of these runs a type checker. Each feeds ``parse_errors`` or ``compare`` literal
mypy output, because what the gate has to get right is the *key* it scores on: the line
number is deliberately absent from it, the message deliberately part of it, and a run
taken without the stub distributions is deliberately not scored at all.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

# scripts/ is not a package, so the gate is loaded by path. It is deliberately not
# registered in sys.modules: nothing in it resolves its own module name, and an
# import-time write to sys.modules is process state no fixture can undo.
_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "mypy_gate.py"
_spec = importlib.util.spec_from_file_location("mypy_gate", _MODULE_PATH)
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


BASELINE_OUTPUT = (
    'src/osprey/a.py:10: error: Returning Any from function declared to return "int"'
    "  [no-any-return]\n"
    'src/osprey/b.py:44: error: Argument 1 has incompatible type "str"; expected "int"'
    "  [arg-type]\n"
)


def _tally(output: str) -> Any:
    return gate.tally(gate.parse_errors(output))


def test_an_error_the_baseline_carries_passes() -> None:
    """A tree that reports exactly what was written down is green."""
    added, stale = gate.compare(_tally(BASELINE_OUTPUT), _tally(BASELINE_OUTPUT))
    assert added == []
    assert stale == []


def test_a_new_error_fails_the_gate() -> None:
    """An error nobody wrote down is the failure mode the gate exists for."""
    run = BASELINE_OUTPUT + 'src/osprey/c.py:7: error: Name "x" is not defined  [name-defined]\n'
    added, stale = gate.compare(_tally(run), _tally(BASELINE_OUTPUT))
    assert len(added) == 1
    assert "src/osprey/c.py" in added[0]
    assert "[name-defined]" in added[0]
    assert stale == []


def test_an_error_swapped_for_another_in_the_same_file_fails() -> None:
    """The case a per-file count would wave through, and why the message is in the key."""
    run = (
        'src/osprey/a.py:10: error: Returning Any from function declared to return "int"'
        "  [no-any-return]\n"
        'src/osprey/b.py:44: error: Argument 1 has incompatible type "float"; expected "bytes"'
        "  [arg-type]\n"
    )
    added, stale = gate.compare(_tally(run), _tally(BASELINE_OUTPUT))
    assert len(added) == 1
    assert '"float"' in added[0]
    assert len(stale) == 1
    assert '"str"' in stale[0]


def test_an_error_that_moved_line_still_matches() -> None:
    """An edit above an error moves its line; the line never entered the key."""
    run = BASELINE_OUTPUT.replace(":10:", ":912:").replace(":44:", ":3:")
    added, stale = gate.compare(_tally(run), _tally(BASELINE_OUTPUT))
    assert added == []
    assert stale == []


def test_a_fixed_error_is_reported_stale_and_does_not_fail() -> None:
    """A run reporting fewer errors is an improvement, not a regression."""
    run = (
        'src/osprey/a.py:10: error: Returning Any from function declared to return "int"'
        "  [no-any-return]\n"
    )
    added, stale = gate.compare(_tally(run), _tally(BASELINE_OUTPUT))
    assert added == []
    assert len(stale) == 1
    assert "src/osprey/b.py" in stale[0]


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


class _FakeCompleted:
    def __init__(self, stdout: str, returncode: int = 1, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


class _FakeSubprocess:
    def __init__(self, completed: _FakeCompleted) -> None:
        self._completed = completed

    def run(self, *_args: Any, **_kwargs: Any) -> _FakeCompleted:
        return self._completed


def test_a_run_without_stub_packages_is_refused_rather_than_scored(
    monkeypatch: Any, capsys: Any
) -> None:
    """A checker without the declared stubs reports a different, larger error set.

    Scoring it against the baseline would compare two different measurements, so the
    gate refuses and names the command that restores the environment.
    """
    output = (
        'src/osprey/a.py:3: error: Library stubs not installed for "yaml"  [import-untyped]\n'
        'src/osprey/a.py:10: error: Returning Any from function declared to return "int"'
        "  [no-any-return]\n"
    )
    monkeypatch.setattr(gate, "subprocess", _FakeSubprocess(_FakeCompleted(output)))

    assert gate.main([]) == 2
    assert "uv sync --extra dev" in capsys.readouterr().out


def test_the_gate_checks_the_declared_trees() -> None:
    """The build names no trees of its own, so it cannot drift from a local run."""
    assert gate.declared_targets(gate.PYPROJECT) == ["src", "packages/osprey-connectors/src"]
