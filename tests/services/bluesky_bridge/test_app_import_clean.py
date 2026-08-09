"""Guards `app.py`'s `_BRIDGE_ONLY_MODULES` invariant (FR8): importing
`osprey.services.bluesky_bridge.app` must never pull in `bluesky`, `ophyd`,
`ophyd_async`, or `tiled`. The bluesky stack is a core dependency, so this is
an import-hygiene boundary, not an install-size one. The bridge runs no plans
— the queueserver worker does — so nothing in this process has any business
importing the RunEngine stack, and the Channel Access client libraries have no
business here either: every device lives in the worker.

This MUST run in a fresh subprocess. The dev venv has all four packages
installed, and other tests in this suite legitimately import them, so by
the time an in-process test runs, `sys.modules` is already contaminated —
an in-process assertion would pass or fail depending on test order, not on
what `app.py` actually imports. Only a fresh interpreter, spawned before
anything else has touched `sys.modules`, can answer the question.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from osprey.services.bluesky_bridge.app import _BRIDGE_ONLY_MODULES


def _run_import_check(module: str) -> subprocess.CompletedProcess[str]:
    code = f"import osprey.services.bluesky_bridge.app, sys; assert {module!r} not in sys.modules"
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )


def test_bridge_only_modules_is_nonempty() -> None:
    # Guards against a vacuously-passing suite if the constant is ever
    # emptied out from under this test.
    assert _BRIDGE_ONLY_MODULES == {"bluesky", "ophyd", "ophyd_async", "tiled"}


def test_importing_app_does_not_import_tiled() -> None:
    result = _run_import_check("tiled")
    assert result.returncode == 0, (
        "importing osprey.services.bluesky_bridge.app leaked a top-level "
        f"import of 'tiled' (child exit {result.returncode}):\n{result.stderr}"
    )


def test_importing_app_does_not_import_bridge_only_modules() -> None:
    failures: dict[str, subprocess.CompletedProcess[str]] = {}
    for module in sorted(_BRIDGE_ONLY_MODULES):
        result = _run_import_check(module)
        if result.returncode != 0:
            failures[module] = result

    assert not failures, "\n".join(
        f"importing osprey.services.bluesky_bridge.app leaked a top-level "
        f"import of {module!r} (child exit {result.returncode}):\n{result.stderr}"
        for module, result in failures.items()
    )


@pytest.mark.parametrize("module", ["pyepics", "epics"])
def test_importing_app_does_not_import_channel_access_clients(module: str) -> None:
    """The Channel Access client libraries are not in `_BRIDGE_ONLY_MODULES` but
    must stay out of the bridge's import path all the same: devices — and every
    CA connection — belong to the queueserver worker, so a top-level `epics`
    import here would mean this process had grown a way to talk to hardware.
    """
    result = _run_import_check(module)
    assert result.returncode == 0, (
        "importing osprey.services.bluesky_bridge.app leaked a top-level "
        f"import of {module!r} (child exit {result.returncode}):\n{result.stderr}"
    )
