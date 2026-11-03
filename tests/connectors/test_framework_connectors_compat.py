"""The framework at HEAD and the connectors at HEAD agree, as the compat lanes check.

The checks the ``connectors-compat`` lane runs against a released framework
live in the connectors package (``osprey_connectors_tests.compat``), so its
own suite stays framework-free. These run the same checks against this
checkout's framework, which is the pairing every PR changes: a framework
import of a name the connectors dropped, or a view the framework writes that
the connectors no longer open, is caught here before any release carries it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import osprey
import osprey_connectors
from osprey_connectors_tests.compat import contracts, scan_framework_imports

# The framework and the connectors this suite imports, wherever they are
# installed: both from this checkout in most lanes, the floor connectors from
# site-packages in the framework-at-connectors-floor lane.
FRAMEWORK = Path(osprey.__file__).resolve().parent
CONNECTORS_ROOT = Path(osprey_connectors.__file__).resolve().parents[1]
TEMPLATE_FACILITY = FRAMEWORK / "templates" / "facilities" / "example"


def _package_dir(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin is not None, f"{name} is not importable"
    return Path(spec.origin).resolve().parent


#: The config the example's view is rendered with: every model served.
VIEW_CONFIG: dict[str, Any] = {"control_system": {"type": "virtual_accelerator"}}


def test_every_connectors_name_the_framework_imports_resolves() -> None:
    """Run as the lanes run it: ``python -I``, over the framework and the pyAML
    backend it depends on, against the connectors this suite imports."""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            scan_framework_imports.__file__,
            str(FRAMEWORK),
            str(_package_dir("pyaml_cs_osprey")),
            "--connectors-path",
            str(CONNECTORS_ROOT),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"unresolved connectors imports:\n{result.stdout}{result.stderr}"
    assert result.stderr.count(" 0 unresolved") == 2, result.stderr


def test_the_framework_hook_reads_the_record_the_connectors_write(tmp_path: Path) -> None:
    hook = FRAMEWORK / contracts.HOOK_RELPATH
    assert contracts.check_control_context_hook(hook, tmp_path).startswith(
        "control-context hook OK"
    )


def test_the_framework_ships_the_view_writer_where_the_contract_looks() -> None:
    """The lane renders the view only where the writer is; a writer that moved
    would leave every later release failing the contract as one with none."""
    assert (FRAMEWORK / contracts.VIEW_WRITER_RELPATH).is_file()


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
