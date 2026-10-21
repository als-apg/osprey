"""The framework at HEAD and the connectors at HEAD agree, as the compat lanes check.

The checks the ``connectors-compat`` lane runs against a released framework
live in the connectors package (``osprey_connectors_tests.compat``), so its
own suite stays framework-free. These run the same checks against this
checkout's framework, which is the pairing every PR changes: a framework
import of a name the connectors dropped, or a view the framework writes that
the connectors no longer open, is caught here before any release carries it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors_tests.compat import contracts, scan_framework_imports

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = REPO_ROOT / "src" / "osprey"
CONNECTORS_SRC = REPO_ROOT / "packages" / "osprey-connectors" / "src"
TEMPLATE_FACILITY = FRAMEWORK / "templates" / "facilities" / "example"

#: The config the example's view is rendered with: every model served.
VIEW_CONFIG: dict[str, Any] = {"control_system": {"type": "virtual_accelerator"}}


def test_every_connectors_name_the_framework_imports_resolves() -> None:
    """Run as the lanes run it: ``python -I``, the checkout's connectors named."""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            scan_framework_imports.__file__,
            str(FRAMEWORK),
            "--connectors-path",
            str(CONNECTORS_SRC),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"unresolved connectors imports:\n{result.stdout}{result.stderr}"
    assert " 0 unresolved" in result.stderr


def test_the_framework_hook_reads_the_record_the_connectors_write(tmp_path: Path) -> None:
    hook = FRAMEWORK / contracts.HOOK_RELPATH
    assert contracts.check_control_context_hook(hook, tmp_path).startswith(
        "control-context hook OK"
    )


def test_the_connectors_open_the_view_the_framework_writes(tmp_path: Path) -> None:
    from osprey.facility.build import build_facility
    from osprey.facility.served import resolve_served
    from osprey.facility.views import ViewInputs
    from osprey.facility.views.simulator import write_simulator_view

    doc = build_facility(TEMPLATE_FACILITY, project_name="example")
    write_simulator_view(
        tmp_path / "data" / "simulator",
        ViewInputs(
            doc=doc,
            rendered_config=VIEW_CONFIG,
            facility_dir=TEMPLATE_FACILITY,
            served=resolve_served(VIEW_CONFIG, doc),
            reported=None,
        ),
    )
    summary = contracts.check_simulator_view(tmp_path)
    assert summary.startswith("simulator view OK"), summary


def test_a_render_with_no_view_breaks_the_contract(tmp_path: Path) -> None:
    with pytest.raises(contracts.ContractError, match="NoSimulatorView"):
        contracts.check_simulator_view(tmp_path)
