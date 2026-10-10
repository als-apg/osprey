"""Guards `app.py`'s `_BRIDGE_ONLY_MODULES` invariant (FR8): importing
`osprey.services.bluesky_bridge.app` must never pull in `bluesky`,
`bluesky_tiled_plugins`, `ophyd`, `ophyd_async`, or `tiled`. The bluesky stack
is a core dependency, so this is
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

import json
import subprocess
import sys

import pytest

from osprey.services.bluesky_bridge.app import _BRIDGE_ONLY_MODULES

#: Every module the tests below keep out of the bridge's import path.
_GUARDED = sorted(_BRIDGE_ONLY_MODULES | {"pyepics", "epics"})


@pytest.fixture(scope="module")
def leaked() -> set[str]:
    """The guarded modules one fresh interpreter has loaded after importing the app.

    One child answers every test here: each used to spawn its own to run the
    same import and then read ``sys.modules``, and the import is the slow part.
    """
    code = (
        "import json, sys\n"
        "import osprey.services.bluesky_bridge.app\n"
        f"print(json.dumps([m for m in {_GUARDED!r} if m in sys.modules]))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"importing osprey.services.bluesky_bridge.app failed "
        f"(child exit {result.returncode}):\n{result.stderr}"
    )
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def _leak_message(module: str) -> str:
    return f"importing osprey.services.bluesky_bridge.app leaked a top-level import of {module!r}"


def test_bridge_only_modules_is_nonempty() -> None:
    # Guards against a vacuously-passing suite if the constant is ever
    # emptied out from under this test.
    assert _BRIDGE_ONLY_MODULES == {
        "bluesky",
        "bluesky_tiled_plugins",
        "ophyd",
        "ophyd_async",
        "tiled",
    }


def test_importing_app_does_not_import_tiled(leaked: set[str]) -> None:
    assert "tiled" not in leaked, _leak_message("tiled")


def test_importing_app_does_not_import_bridge_only_modules(leaked: set[str]) -> None:
    offenders = sorted(leaked & _BRIDGE_ONLY_MODULES)
    assert not offenders, "\n".join(_leak_message(module) for module in offenders)


@pytest.mark.parametrize("module", ["pyepics", "epics"])
def test_importing_app_does_not_import_channel_access_clients(
    module: str, leaked: set[str]
) -> None:
    """The Channel Access client libraries are not in `_BRIDGE_ONLY_MODULES` but
    must stay out of the bridge's import path all the same: devices — and every
    CA connection — belong to the queueserver worker, so a top-level `epics`
    import here would mean this process had grown a way to talk to hardware.
    """
    assert module not in leaked, _leak_message(module)
