"""Guards the bluesky-free boundary of the plan-metadata extraction chain.

``plan_metadata.py`` parses a plan module's ``PLAN_METADATA`` block out of
source with `ast`, never by importing it, and ``plan_loader.py`` layers plan
files without pulling the RunEngine stack into the reading process. Both
modules document themselves as free of `bluesky`, `ophyd` and `tiled`; these
tests turn that docstring promise into an enforced contract, so any consumer
(the bridge, the CLI's plan-index builder) can import them without dragging in
the run-time stack.

This MUST run in a fresh subprocess. All three packages are installed in the
dev venv and other tests in this suite legitimately import them, so by the time
an in-process test runs `sys.modules` is already contaminated -- an in-process
assertion would pass or fail depending on test order, not on what the module
under test actually imports. Only a fresh interpreter, spawned before anything
else has touched `sys.modules`, can answer the question.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

# The modules whose import must stay free of the run-time stack. Extending the
# guard to another module -- e.g. the CLI's plan-index builder, which imports
# from both of these and carries the same requirement -- is one entry here.
GUARDED_MODULES = (
    "osprey.services.bluesky_bridge.plan_metadata",
    "osprey.services.bluesky_bridge.plan_loader",
)

# The run-time stack: importing any of these means the module under test grew a
# dependency on the RunEngine/device/data-access machinery it is defined to
# stay clear of.
FORBIDDEN_MODULES = ("bluesky", "ophyd", "tiled")


def _run_import_check(module: str, forbidden: str) -> subprocess.CompletedProcess[str]:
    """Import ``module`` in a clean interpreter and assert ``forbidden`` stayed out.

    Args:
        module: Dotted path of the module under test.
        forbidden: Top-level package name that must be absent from the child's
            ``sys.modules`` afterwards.

    Returns:
        The completed child process; a non-zero return code means the import
        leaked (or, distinguished by the caller, failed outright).
    """
    # `SystemExit`, not `assert`: the child inherits this process's environment,
    # so under `PYTHONOPTIMIZE` a bare assert is stripped out and the child
    # exits 0 no matter what `sys.modules` holds -- the guard would pass while
    # proving nothing.
    code = f"import {module}, sys; raise SystemExit(1 if {forbidden!r} in sys.modules else 0)"
    # `sys.executable` -- not a bare `python` -- so the child is this tree's
    # venv interpreter; a bare `python` in a worktree resolves to the main
    # checkout's osprey and would silently test the wrong source.
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


@pytest.mark.parametrize("forbidden", FORBIDDEN_MODULES)
def test_forbidden_modules_are_installed(forbidden: str) -> None:
    """Fails loudly if the environment cannot contaminate itself.

    An absent package makes every assertion below pass without proving
    anything: the import could not have happened either way.
    """
    assert importlib.util.find_spec(forbidden) is not None, (
        f"{forbidden!r} is not installed, so the import-cleanliness guards below "
        "cannot fail and prove nothing -- install the bluesky stack to run them"
    )


@pytest.mark.parametrize("module", GUARDED_MODULES)
@pytest.mark.parametrize("forbidden", FORBIDDEN_MODULES)
def test_guarded_module_does_not_import_run_time_stack(module: str, forbidden: str) -> None:
    result = _run_import_check(module, forbidden)
    # A module that fails to import also exits non-zero, and reporting that as
    # a leak would send whoever reads it hunting the wrong bug.
    assert not any(
        failure in result.stderr for failure in ("ImportError", "ModuleNotFoundError")
    ), f"{module} could not be imported at all, so the guard proved nothing:\n{result.stderr}"
    assert result.returncode == 0, (
        f"importing {module} leaked a top-level import of {forbidden!r} "
        f"(child exit {result.returncode}):\n{result.stderr}"
    )
