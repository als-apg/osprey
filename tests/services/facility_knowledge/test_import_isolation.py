"""Import-isolation guard: rdflib and neo4j must stay OUT of sys.modules when
the facility-knowledge runtime read path is imported.

rdflib belongs exclusively to the offline TTL seeder (``knowledge`` extra) and
the TTL generator (:mod:`osprey.services.facility_knowledge.ttl_generator`);
neo4j belongs exclusively to the graph seeder (:mod:`.seeder.graph_seeder`) and
the graph MCP server (:mod:`osprey.mcp_server.graph`), both of which import it
lazily inside the functions that dial the store. Importing any of those modules
— or constructing the graph MCP server — must never pull in either driver, even
transitively, so that the common runtime path has no rdflib or neo4j dependency.

Each test spawns a fresh subprocess so that rdflib/neo4j already loaded by
pytest (or a previously-run test) cannot cause a false pass. Both detectors are
negative-controlled (:class:`TestIsolationDetectorsFire`) so a guard that has
gone blind — an uninstalled package, a broken template — fails loudly instead of
passing vacuously.

This module also holds the **driver-API floor** check
(:class:`TestDriverApiFloor`): an AST walk over every graph module asserting it
uses only neo4j-driver APIs that exist on both the 5.x and 6.x driver lines.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

#: Repo root, from ``tests/services/facility_knowledge/`` up three levels.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Ceiling on every isolation subprocess. Server construction reads config and
#: must never dial the store; a hang here is itself a failure, not a wait.
SUBPROCESS_TIMEOUT_S = 120

_ABSENT_CHECK_TEMPLATE = """\
{setup}
import sys
_prefix = {package!r} + "."
_leaked = [m for m in sys.modules if m == {package!r} or m.startswith(_prefix)]
assert not _leaked, {package!r} + " leaked into sys.modules: " + repr(_leaked)
print("OK")
"""


def _run_absent_check(
    package: str,
    setup: str,
    *,
    env: dict[str, str | None] | None = None,
) -> tuple[bool, str]:
    """Run *setup* in a fresh interpreter and assert *package* stayed unimported.

    Args:
        package: Top-level distribution package that must stay out of
            ``sys.modules`` (``"rdflib"`` or ``"neo4j"``). Submodules count as
            leaks too.
        setup: Python source executed before the check — an import statement, or
            a few lines that also construct something.
        env: Environment overlay on top of ``os.environ``. A ``None`` value
            removes that variable from the child's environment.

    Returns:
        Tuple of ``(package_absent, stderr_output)``. *package_absent* is True
        when *setup* did not pull the package in as a side-effect.
    """
    code = _ABSENT_CHECK_TEMPLATE.format(setup=setup, package=package)
    child_env: dict[str, str] | None = None
    if env is not None:
        child_env = dict(os.environ)
        for key, value in env.items():
            if value is None:
                child_env.pop(key, None)
            else:
                child_env[key] = value
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
        env=child_env,
    )
    return result.returncode == 0, result.stderr


def _run_isolation_check(import_stmt: str) -> tuple[bool, str]:
    """Run *import_stmt* in a subprocess and check that rdflib is not in sys.modules.

    Args:
        import_stmt: A Python ``import`` statement to execute first.

    Returns:
        Tuple of (rdflib_absent, stderr_output).  *rdflib_absent* is True when
        rdflib was not imported as a side-effect.
    """
    return _run_absent_check("rdflib", import_stmt)


def _run_neo4j_isolation_check(import_stmt: str) -> tuple[bool, str]:
    """Run *import_stmt* in a subprocess and check that neo4j is not in sys.modules.

    Mirrors :func:`_run_isolation_check` for the neo4j driver, which
    :mod:`osprey.services.facility_knowledge.seeder.graph_seeder` imports lazily
    inside ``open_session`` rather than at module top.

    Args:
        import_stmt: A Python ``import`` statement to execute first.

    Returns:
        Tuple of (neo4j_absent, stderr_output).  *neo4j_absent* is True when
        neo4j was not imported as a side-effect.
    """
    return _run_absent_check("neo4j", import_stmt)


# ---------------------------------------------------------------------------
# Module discovery — the guard grows with the packages it protects
# ---------------------------------------------------------------------------


def _discover_modules(package: str, *, exclude: frozenset[str] = frozenset()) -> tuple[str, ...]:
    """Return *package* plus every importable ``.py`` module beneath it.

    Discovery is by source file rather than a hand-maintained list so a module
    added later (a new graph tool, the TTL emitter) is guarded the day it lands
    instead of the day someone remembers to extend this file.

    Args:
        package: Dotted package name rooted under ``src/osprey``.
        exclude: Unqualified module names to skip (e.g. ``__main__``, whose body
            is an entry point rather than a library surface).

    Returns:
        Sorted tuple of dotted module names, the package itself included.
    """
    root = REPO_ROOT / "src" / Path(*package.split("."))
    assert root.is_dir(), f"package directory not found: {root}"
    found = {package}
    for path in root.rglob("*.py"):
        parts = path.relative_to(root).with_suffix("").parts
        if parts[-1] in exclude:
            continue
        parts = parts[:-1] if parts[-1] == "__init__" else parts
        found.add(".".join((package, *parts)) if parts else package)
    return tuple(sorted(found))


#: Every module of the read-only graph MCP server. ``__main__`` is excluded: it
#: is the ``python -m`` entry point, not an importable surface.
GRAPH_MCP_MODULES = _discover_modules("osprey.mcp_server.graph", exclude=frozenset({"__main__"}))

#: Every module of the offline TTL generator.
TTL_GENERATOR_MODULES = _discover_modules("osprey.services.facility_knowledge.ttl_generator")

#: Modules the guard must cover no matter what discovery turns up — a floor that
#: turns a silently-empty ``rglob`` into a red test rather than a vacuous pass.
REQUIRED_GRAPH_MCP_MODULES = frozenset(
    {
        "osprey.mcp_server.graph",
        "osprey.mcp_server.graph.server",
        "osprey.mcp_server.graph.server_context",
        "osprey.mcp_server.graph.tools",
        "osprey.mcp_server.graph.tools.capabilities",
        "osprey.mcp_server.graph.tools.example_queries",
        "osprey.mcp_server.graph.tools.get_schema",
        "osprey.mcp_server.graph.tools.read_cypher",
    }
)

#: Likewise for the TTL generator. ``emitter`` is intentionally absent: it is
#: discovered automatically once it lands, so no follow-up edit is needed here.
REQUIRED_TTL_GENERATOR_MODULES = frozenset(
    {
        "osprey.services.facility_knowledge.ttl_generator",
        "osprey.services.facility_knowledge.ttl_generator.direction",
        "osprey.services.facility_knowledge.ttl_generator.model",
        "osprey.services.facility_knowledge.ttl_generator.ontology_map",
    }
)

#: Constructing the server: ``create_server()`` primes config and calls
#: ``initialize_server_context()``, which resolves the connection but does not
#: dial it — the driver is created on the first query.
_CREATE_SERVER_SETUP = "from osprey.mcp_server.graph.server import create_server\ncreate_server()\n"

#: Same, but tolerating the exception raised when no project config exists at
#: all: the point of that variant is the driver, not the construction outcome.
_CREATE_SERVER_NO_CONFIG_SETUP = (
    "try:\n"
    "    from osprey.mcp_server.graph.server import create_server\n"
    "    create_server()\n"
    "except Exception:\n"
    "    pass\n"
)


@pytest.fixture
def minimal_config_env(tmp_path: Path) -> dict[str, str | None]:
    """A project config with no ``services.graphdb`` block, as env overrides.

    The graph context must resolve this to "not configured" by reading alone —
    without importing the driver and without touching the network.
    """
    config = tmp_path / "config.yml"
    config.write_text(f"project_root: {tmp_path}\n", encoding="utf-8")
    return {"CONFIG_FILE": str(config), "OSPREY_CONFIG": str(config)}


@pytest.fixture
def no_config_env() -> dict[str, str | None]:
    """An environment with every config pointer stripped."""
    return {"CONFIG_FILE": None, "OSPREY_CONFIG": None}


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

    def test_services_facility_knowledge_does_not_export_graph_seeder(self):
        """The services.facility_knowledge __init__ must not re-export graph_seeder.

        ``graph_seeder`` (added in task 2.1) is only importable from
        ``osprey.services.facility_knowledge.seeder`` (or ``.seeder.graph_seeder``
        directly) — never from the parent ``facility_knowledge`` package. Presence
        of any of its public symbols on ``fk``, or of the ``graph_seeder`` module
        itself in ``sys.modules``, would indicate the neo4j-driven seeder (and its
        lazy neo4j import path) was exposed on the main read path.
        """
        code = (
            "import osprey.services.facility_knowledge as fk\n"
            "import sys\n"
            "assert not hasattr(fk, 'bootstrap'), "
            "'bootstrap must not appear in facility_knowledge namespace'\n"
            "assert not hasattr(fk, 'open_session'), "
            "'open_session must not appear in facility_knowledge namespace'\n"
            "assert not hasattr(fk, 'import_ttl'), "
            "'import_ttl must not appear in facility_knowledge namespace'\n"
            "assert 'osprey.services.facility_knowledge.seeder.graph_seeder' not in sys.modules, "
            "'graph_seeder module must not be imported via facility_knowledge'\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"graph_seeder leaked into facility_knowledge namespace:\n{result.stderr}"
        )


class TestNeo4jImportIsolation:
    """neo4j must not appear in sys.modules after importing the runtime read path.

    ``graph_seeder.open_session`` imports the neo4j driver lazily (see task 2.1),
    mirroring how ``ttl_seeder`` treats rdflib. This mirrors
    ``TestRdflibImportIsolation`` above to pin that contract.
    """

    def test_mcp_server_facility_knowledge_does_not_import_neo4j(self):
        """Importing osprey.mcp_server.facility_knowledge must not pull in neo4j."""
        ok, stderr = _run_neo4j_isolation_check("import osprey.mcp_server.facility_knowledge")
        assert ok, f"neo4j leaked after importing mcp_server.facility_knowledge:\n{stderr}"

    def test_services_facility_knowledge_does_not_import_neo4j(self):
        """Importing osprey.services.facility_knowledge must not pull in neo4j."""
        ok, stderr = _run_neo4j_isolation_check("import osprey.services.facility_knowledge")
        assert ok, f"neo4j leaked after importing services.facility_knowledge:\n{stderr}"


class TestIsolationDetectorsFire:
    """Negative controls: both detectors must fail when the driver IS imported.

    Without these, an uninstalled package or a broken code template would let
    every isolation test above pass while checking nothing. rdflib ships in the
    ``dev`` extra and neo4j in the base dependencies, so both imports below are
    expected to succeed — and the assertion in the child is expected to trip.
    """

    def test_rdflib_detector_fires_when_rdflib_is_imported(self):
        """A subprocess that imports rdflib on purpose must be reported as a leak."""
        ok, stderr = _run_isolation_check("import rdflib")
        assert not ok, "the rdflib detector passed a subprocess that imported rdflib"
        assert "rdflib leaked into sys.modules" in stderr, stderr

    def test_neo4j_detector_fires_when_neo4j_is_imported(self):
        """A subprocess that imports neo4j on purpose must be reported as a leak."""
        ok, stderr = _run_neo4j_isolation_check("import neo4j")
        assert not ok, "the neo4j detector passed a subprocess that imported neo4j"
        assert "neo4j leaked into sys.modules" in stderr, stderr

    def test_submodule_only_import_is_still_a_leak(self):
        """Importing a submodule without the parent must still be detected."""
        ok, stderr = _run_neo4j_isolation_check("import neo4j.exceptions")
        assert not ok, "the neo4j detector missed a submodule-only import"
        assert "neo4j leaked into sys.modules" in stderr, stderr


class TestGraphMcpServerDriverIsolation:
    """The graph MCP server must import neither driver until a query is issued.

    Every module of ``osprey.mcp_server.graph`` is discovered from disk, so a
    tool module added later is covered without editing this file.
    """

    def test_discovery_covers_the_known_graph_modules(self):
        """Discovery must find at least the modules the server is built from."""
        missing = REQUIRED_GRAPH_MCP_MODULES - set(GRAPH_MCP_MODULES)
        assert not missing, f"module discovery missed graph modules: {sorted(missing)}"

    @pytest.mark.parametrize("module", GRAPH_MCP_MODULES)
    def test_graph_module_does_not_import_neo4j(self, module: str):
        """Importing any graph MCP module must leave neo4j out of sys.modules."""
        ok, stderr = _run_neo4j_isolation_check(f"import {module}")
        assert ok, f"neo4j leaked after importing {module}:\n{stderr}"

    @pytest.mark.parametrize("module", GRAPH_MCP_MODULES)
    def test_graph_module_does_not_import_rdflib(self, module: str):
        """Importing any graph MCP module must leave rdflib out of sys.modules."""
        ok, stderr = _run_isolation_check(f"import {module}")
        assert ok, f"rdflib leaked after importing {module}:\n{stderr}"

    def test_create_server_does_not_import_neo4j(self, minimal_config_env):
        """create_server() reads config only — the driver is built on first query."""
        ok, stderr = _run_absent_check("neo4j", _CREATE_SERVER_SETUP, env=minimal_config_env)
        assert ok, f"neo4j leaked while constructing the graph server:\n{stderr}"

    def test_create_server_does_not_import_rdflib(self, minimal_config_env):
        """create_server() must not pull in rdflib either."""
        ok, stderr = _run_absent_check("rdflib", _CREATE_SERVER_SETUP, env=minimal_config_env)
        assert ok, f"rdflib leaked while constructing the graph server:\n{stderr}"

    def test_create_server_resolves_not_configured_without_dialing(self, minimal_config_env):
        """A config with no services.graphdb block resolves by reading, not dialing.

        ``configured`` must come out False, the driver must still be unimported,
        and the whole thing must finish well inside the subprocess timeout — a
        hang here would mean the context tried to reach the store at startup.
        """
        ok, stderr = _run_absent_check(
            "neo4j",
            _CREATE_SERVER_SETUP
            + "from osprey.mcp_server.graph.server_context import get_server_context\n"
            "assert get_server_context().configured is False, "
            "'a config without services.graphdb must resolve to not-configured'\n",
            env=minimal_config_env,
        )
        assert ok, f"graph server startup misbehaved without a graphdb block:\n{stderr}"

    def test_create_server_without_any_config_does_not_import_neo4j(self, no_config_env):
        """With no project config at all, startup fails — but never via the driver.

        ``create_server()`` currently raises ``FileNotFoundError`` when neither
        ``CONFIG_FILE`` nor ``OSPREY_CONFIG`` is set (``get_config_value`` builds
        a ConfigBuilder and finds no ``config.yml``). Whatever the outcome, the
        neo4j driver must stay out of ``sys.modules``.
        """
        ok, stderr = _run_absent_check("neo4j", _CREATE_SERVER_NO_CONFIG_SETUP, env=no_config_env)
        assert ok, f"neo4j leaked while the graph server failed to start:\n{stderr}"


class TestTtlGeneratorRdflibIsolation:
    """The TTL generator holds rdflib behind function-local imports.

    Its modules are pure data/model code; only the emitter's serialization step
    touches rdflib, and it does so inside the function that writes Turtle.
    """

    def test_discovery_covers_the_known_ttl_generator_modules(self):
        """Discovery must find at least the modules the generator is built from."""
        missing = REQUIRED_TTL_GENERATOR_MODULES - set(TTL_GENERATOR_MODULES)
        assert not missing, f"module discovery missed ttl_generator modules: {sorted(missing)}"

    @pytest.mark.parametrize("module", TTL_GENERATOR_MODULES)
    def test_ttl_generator_module_does_not_import_rdflib(self, module: str):
        """Importing any ttl_generator module must leave rdflib out of sys.modules."""
        ok, stderr = _run_isolation_check(f"import {module}")
        assert ok, f"rdflib leaked after importing {module}:\n{stderr}"

    @pytest.mark.parametrize("module", TTL_GENERATOR_MODULES)
    def test_ttl_generator_module_does_not_import_neo4j(self, module: str):
        """The generator writes a file; it must never reach for the graph driver."""
        ok, stderr = _run_neo4j_isolation_check(f"import {module}")
        assert ok, f"neo4j leaked after importing {module}:\n{stderr}"


# ---------------------------------------------------------------------------
# Driver-API floor
# ---------------------------------------------------------------------------

#: neo4j-driver APIs the graph code is allowed to reach for. The floor is the
#: intersection of the 5.x and 6.x driver lines, restricted to the read path:
#: ``GraphDatabase.driver``, ``Driver.execute_query``, ``Session.execute_read``,
#: ``Session.run`` and the ``neo4j.unit_of_work`` decorator. Anything outside it
#: is either write-capable or gone in 6.x.
ALLOWED_DRIVER_APIS = frozenset({"driver", "execute_query", "execute_read", "run", "unit_of_work"})

#: Attribute/name deny-list. Each entry is either removed in neo4j 6.0 or opens
#: a write path, and neither is acceptable in a read-only server pinned to work
#: across both driver lines.
FORBIDDEN_DRIVER_APIS = frozenset(
    {
        "read_transaction",  # removed in 6.0 — superseded by execute_read
        "write_transaction",  # removed in 6.0 — and write-capable
        "execute_write",  # write-capable: this server is read-only
        "begin_transaction",  # unmanaged tx: write-capable, no retry/timeout floor
        "last_bookmark",  # removed in 6.0 — superseded by last_bookmarks
    }
)

#: Files held to the floor: the whole graph MCP server plus the graph seeder.
DRIVER_API_FLOOR_FILES = tuple(
    sorted(
        [*(REPO_ROOT / "src" / "osprey" / "mcp_server" / "graph").rglob("*.py")]
        + [
            REPO_ROOT
            / "src"
            / "osprey"
            / "services"
            / "facility_knowledge"
            / "seeder"
            / "graph_seeder.py"
        ]
    )
)


def _scan_driver_api(source: str, filename: str) -> list[str]:
    """Return ``file:line: name`` for every forbidden driver API used in *source*.

    The scan is AST-based, not textual: an attribute access (``tx.read_transaction``),
    a bare name (``read_transaction(...)`` after ``from neo4j import ...``) and the
    import itself all count, while the same word inside a docstring, a comment or a
    string literal does not.

    Args:
        source: Python source text.
        filename: Label used in the returned locations.

    Returns:
        One ``"<filename>:<lineno>: <name>"`` string per violation, in source order.
    """
    tree = ast.parse(source, filename=filename)
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_DRIVER_APIS:
            violations.append((node.lineno, node.attr))
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_DRIVER_APIS:
            violations.append((node.lineno, node.id))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN_DRIVER_APIS:
                    violations.append((node.lineno, alias.name))
    return [f"{filename}:{lineno}: {name}" for lineno, name in sorted(set(violations))]


class TestDriverApiFloor:
    """Every graph module must stay on the 5.x/6.x-common neo4j driver surface."""

    def test_floor_file_list_is_not_empty(self):
        """Anti-vacuity: the scan must actually have files to walk."""
        assert DRIVER_API_FLOOR_FILES, "driver-API floor found no files to scan"
        names = {path.name for path in DRIVER_API_FLOOR_FILES}
        assert {"server_context.py", "graph_seeder.py"} <= names, sorted(names)

    @pytest.mark.parametrize(
        "path", DRIVER_API_FLOOR_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT))
    )
    def test_module_uses_only_floor_driver_apis(self, path: Path):
        """No forbidden driver API anywhere in the graph read path."""
        rel = str(path.relative_to(REPO_ROOT))
        violations = _scan_driver_api(path.read_text(encoding="utf-8"), rel)
        assert not violations, (
            "neo4j APIs outside the 5.x/6.x floor "
            f"({sorted(ALLOWED_DRIVER_APIS)} are the allowed calls):\n" + "\n".join(violations)
        )

    def test_checker_flags_an_attribute_call(self):
        """Negative control: an attribute access on the deny-list is caught."""
        source = "def read(tx):\n    return tx.read_transaction(lambda t: None)\n"
        assert _scan_driver_api(source, "inline.py") == ["inline.py:2: read_transaction"]

    def test_checker_flags_a_bare_imported_name(self):
        """Negative control: ``from neo4j import write_transaction`` is caught."""
        source = "from neo4j import write_transaction\n\nuse = write_transaction\n"
        assert _scan_driver_api(source, "inline.py") == [
            "inline.py:1: write_transaction",
            "inline.py:3: write_transaction",
        ]

    def test_checker_flags_every_forbidden_api(self):
        """Negative control: no deny-list entry is dead weight."""
        for name in sorted(FORBIDDEN_DRIVER_APIS):
            source = f"session.{name}()\n"
            assert _scan_driver_api(source, "inline.py") == [f"inline.py:1: {name}"], name

    def test_checker_ignores_prose_mentions(self):
        """A docstring or comment naming a forbidden API is not a violation.

        ``graph_seeder.py`` documents the floor in its module docstring; a
        textual grep would flag it, an AST walk must not.
        """
        source = (
            '"""Uses execute_read, never Session.read_transaction (removed in 6.0)."""\n'
            "\n"
            "# write_transaction is forbidden here\n"
            'BANNED = "execute_write"\n'
        )
        assert _scan_driver_api(source, "inline.py") == []

    def test_allowed_and_forbidden_lists_are_disjoint(self):
        """The floor cannot both permit and deny the same call."""
        assert not (ALLOWED_DRIVER_APIS & FORBIDDEN_DRIVER_APIS)
