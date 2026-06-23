"""Import-isolation guard: rdflib must stay OUT of sys.modules when the
facility-knowledge runtime read path is imported.

rdflib belongs exclusively to the offline seeder (``knowledge`` extra).
Importing the MCP server or service package must never pull it in — even
transitively — so that the common runtime path has no rdflib dependency.

Each test spawns a fresh subprocess so that rdflib already loaded by pytest
(or a previously-run test) cannot cause a false pass.
"""

from __future__ import annotations

import subprocess
import sys


def _run_isolation_check(import_stmt: str) -> tuple[bool, str]:
    """Run *import_stmt* in a subprocess and check that rdflib is not in sys.modules.

    Args:
        import_stmt: A Python ``import`` statement to execute first.

    Returns:
        Tuple of (rdflib_absent, stderr_output).  *rdflib_absent* is True when
        rdflib was not imported as a side-effect.
    """
    code = (
        f"{import_stmt}\n"
        "import sys\n"
        "assert 'rdflib' not in sys.modules, "
        "f'rdflib leaked into sys.modules: {[k for k in sys.modules if k.startswith(\"rdflib\")]}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stderr


class TestRdflibImportIsolation:
    """rdflib must not appear in sys.modules after importing the runtime read path."""

    def test_mcp_server_facility_knowledge_does_not_import_rdflib(self):
        """Importing osprey.mcp_server.facility_knowledge must not pull in rdflib."""
        ok, stderr = _run_isolation_check("import osprey.mcp_server.facility_knowledge")
        assert ok, f"rdflib leaked after importing mcp_server.facility_knowledge:\n{stderr}"

    def test_services_facility_knowledge_does_not_import_rdflib(self):
        """Importing osprey.services.facility_knowledge must not pull in rdflib."""
        ok, stderr = _run_isolation_check("import osprey.services.facility_knowledge")
        assert ok, f"rdflib leaked after importing services.facility_knowledge:\n{stderr}"

    def test_services_facility_knowledge_does_not_export_seeder(self):
        """The services.facility_knowledge __init__ must not re-export the seeder.

        Presence of ``seed_from_ttl`` or ``DeviceStub`` in the package namespace
        would indicate the seeder (and its lazy rdflib import path) was exposed on
        the main read path.
        """
        code = (
            "import osprey.services.facility_knowledge as fk\n"
            "import sys\n"
            "assert not hasattr(fk, 'seed_from_ttl'), "
            "'seed_from_ttl must not appear in facility_knowledge namespace'\n"
            "assert not hasattr(fk, 'DeviceStub'), "
            "'DeviceStub must not appear in facility_knowledge namespace'\n"
            "assert 'osprey.services.facility_knowledge.seeder' not in sys.modules, "
            "'seeder subpackage must not be imported via facility_knowledge'\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Seeder leaked into facility_knowledge namespace:\n{result.stderr}"
        )
