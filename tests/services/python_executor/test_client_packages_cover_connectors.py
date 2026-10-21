"""The one client list covers every client the connectors use and ship.

``CLIENT_PACKAGES`` is what the executor's import check denies and what the
``no_client`` block refuses at import; ``_CLIENT_LIBRARY_NAMES`` is what the
block refuses at load. A connector that starts importing a new client, or a
client package that starts shipping a new shared library, must land in these
tables or in a reviewed exemption below, never fall between them.
"""

from __future__ import annotations

import ast
import importlib.metadata
import sys
from pathlib import Path

import osprey_connectors
from osprey.runtime.raw_put_block import _matches_stem
from osprey.services.python_executor.write_surface import (
    _CLIENT_LIBRARY_NAMES,
    CLIENT_PACKAGES,
)

#: Third-party packages the connectors import that are not control-system
#: clients, each with the reason.
_NOT_A_CLIENT: dict[str, str] = {
    "numpy": "array maths on values already read",
    "pandas": "archiver results as data frames",
    "lume": "the simulator's model interface, in process",
    "pymongo": "a database client for an archive, not the control system",
    "jlab_archiver_client": "an HTTP client for an archive, not the control system",
}

#: Shared libraries the client packages ship that are no client on their own,
#: by basename stem, each with the reason.
_NOT_A_CLIENT_LIBRARY: dict[str, str] = {
    "_base": "epicscorelibs' Python extension that locates its libraries",
    "libCom": "EPICS base's OS layer, loaded by every EPICS library",
    "libComPYEPICS": "pyepics' build of EPICS base's OS layer",
    "Com.dll": "pyepics' Windows build of EPICS base's OS layer",
    "libdbCore": "the IOC database, server side",
    "libdbRecStd": "the IOC's standard record types, server side",
    "libpvAccessIOC": "the PVAccess server inside an IOC",
    "libpvData": "PVAccess data containers, no network code",
    "libqsrv": "the IOC's PVAccess group server, server side",
    "libevent_core": "a generic event loop",
    "libevent_pthread": "a generic event loop's thread support",
    "libpvxsIoc": "the PVXS server inside an IOC",
    "_gw": "p4p's gateway extension, server side",
    "_p4p": "p4p's Python extension, reached only through the p4p package",
}

_CONNECTORS = Path(osprey_connectors.__file__).parent
_SHARED_LIBRARY_SUFFIXES = (".so", ".dylib", ".dll")


def _connector_imports() -> set[str]:
    found: set[str] = set()
    for subpackage in ("control_system", "archiver"):
        for path in sorted((_CONNECTORS / subpackage).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    found.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])
    return {
        name
        for name in found
        if name not in sys.stdlib_module_names and name not in ("osprey_connectors", "__future__")
    }


def test_every_connector_import_is_a_client_or_reviewed():
    unaccounted = sorted(_connector_imports() - CLIENT_PACKAGES - set(_NOT_A_CLIENT))
    assert unaccounted == [], (
        f"connector imports {unaccounted} are in neither CLIENT_PACKAGES nor _NOT_A_CLIENT"
    )


def test_no_exemption_shadows_a_client():
    assert not set(_NOT_A_CLIENT) & CLIENT_PACKAGES


def _shipped_libraries() -> dict[str, str]:
    """Every shared library an installed client package ships, basename -> distribution."""
    distributions = importlib.metadata.packages_distributions()
    shipped: dict[str, str] = {}
    for package in sorted(CLIENT_PACKAGES | {"pvxslibs"}):
        for distribution in distributions.get(package, ()):
            for file in importlib.metadata.distribution(distribution).files or ():
                name = Path(str(file)).name
                if any(
                    name.endswith(suffix) or f"{suffix}." in name
                    for suffix in _SHARED_LIBRARY_SUFFIXES
                ):
                    shipped[name] = distribution
    return shipped


def test_every_shipped_library_is_a_client_stem_or_reviewed():
    unaccounted = sorted(
        f"{name} ({distribution})"
        for name, distribution in _shipped_libraries().items()
        if not any(_matches_stem(name, stem) for stem in _CLIENT_LIBRARY_NAMES)
        and not any(_matches_stem(name, stem) for stem in _NOT_A_CLIENT_LIBRARY)
    )
    assert unaccounted == [], (
        f"shared libraries {unaccounted} match neither _CLIENT_LIBRARY_NAMES "
        "nor _NOT_A_CLIENT_LIBRARY"
    )


def test_no_library_exemption_shadows_a_client_stem():
    shadowing = sorted(
        exempt
        for exempt in _NOT_A_CLIENT_LIBRARY
        for stem in _CLIENT_LIBRARY_NAMES
        if _matches_stem(exempt, stem) or _matches_stem(stem, exempt)
    )
    assert shadowing == []


def test_the_shipped_client_libraries_hit_their_stems():
    for basename in (
        "libca.7.0.10.99.0.dylib",
        "libpvAccessCA.dylib",
        "libpvxs.1.5.dylib",
        "ca.dll",
    ):
        assert any(_matches_stem(basename, stem) for stem in _CLIENT_LIBRARY_NAMES), basename
