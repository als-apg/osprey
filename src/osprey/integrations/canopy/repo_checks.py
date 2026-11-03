"""The CANOPY integration's checks: its contract, and the rule its marker keeps.

``scripts/check.py`` collects this module after the framework's. The contract
proves the CANOPY code against CANOPY and PyDMConverter installed at the
commits ``proven.py`` pins, so it is declared here, beside that code, with
everything the command does with it: its scope, derived from what the CANOPY
code imports; the install it proves against; the line its verdict ends with.

This module is a declaration for the check command, not part of the
integration: it is left out of the wheel, and nothing imports it.

Standard library only, for the reason ``scripts/check.py`` gives.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from types import ModuleType

import check_kit as kit
from check_kit import BOTH, Check, Install, Refused, Selection, touches

#: The stdlib-only module beside this one that holds the commits the CANOPY
#: contract is proven against and words the line that says so; read by path,
#: as CI reads it.
PROVEN_MODULE = Path(__file__).resolve().with_name("proven.py")

#: Judges the installed CANOPY inside the project's environment and prints the
#: line that says what it found: exit 0 at the proven commits, 2 when neither
#: CANOPY nor PyDMConverter is installed, 1 for anything else.
CANOPY_PROBE = (*kit.UV_RUN, "--no-sync", "python", "-m", "osprey.integrations.canopy")

#: The contract's tests: every directory holding a ``canopy``-marked module (the
#: integration and the shipped screen library's), whole, with no marker expression.
CONTRACT_TESTS = ("tests/integrations/canopy", "tests/templates/canopy")

#: The tests that hold the ``canopy`` marker's rules over the whole tests tree:
#: a change anywhere in it can break them, so no change selection leaves them out.
INVARIANT_TESTS = ("tests/integrations/canopy/test_contract_marker.py",)

#: The CANOPY contract's declared roots: the CANOPY code, the contract, the
#: marker's rules, the data the contract reads, the job that runs it on CI and
#: the dependency declarations that decide what it runs against.
#: The Python under the first four is where the scope's import hop starts.
CANOPY_CONTRACT_ROOTS = (
    "src/osprey/integrations/canopy/**",
    "tests/integrations/canopy/**",
    "tests/_canopy_contract.py",
    "tests/templates/canopy/**",
    "src/osprey/templates/apps/control_assistant/data/canopy/**",
    ".github/workflows/ci.yml",
    "pyproject.toml",
    "uv.lock",
    "packages/osprey-connectors/pyproject.toml",
)
CANOPY_IMPORTING_ROOTS = CANOPY_CONTRACT_ROOTS[:4]

#: The repository's own top-level packages; an import of anything else is not followed.
OWN_PACKAGES = frozenset({"osprey", "osprey_connectors", "tests"})
#: Where an own package is imported from, in the order a module is looked up.
PACKAGE_BASES = (
    "src/",
    "packages/osprey-connectors/src/",
    "",
)


# ---------------------------------------------------------------------------
# The scope: the declared roots and what the CANOPY code imports, one hop


def _import_nodes(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """Every import statement in *tree*: module, class and function level alike."""
    return [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]


def _module_file(module: str, known: set[str]) -> str | None:
    """The file of the own module *module* (dotted), or ``None`` when it is not one."""
    if module.split(".", 1)[0] not in OWN_PACKAGES:
        return None
    relative = module.replace(".", "/")
    for base in PACKAGE_BASES:
        for candidate in (f"{base}{relative}.py", f"{base}{relative}/__init__.py"):
            if candidate in known:
                return candidate
    return None


def _relative_file(directory: PurePosixPath, dotted: str, known: set[str]) -> str | None:
    """The file *dotted* names relative to the package *directory*, or ``None``."""
    stem = directory.joinpath(*dotted.split(".")).as_posix() if dotted else directory.as_posix()
    return next((f for f in (f"{stem}.py", f"{stem}/__init__.py") if f in known), None)


def _imported_files(path: str, tree: ast.Module, known: set[str]) -> set[str]:
    """The files of the own modules *tree*, the module at *path*, imports directly.

    ``from a.b import c`` is the module ``a.b.c`` when there is one, else ``a.b``.
    """
    found: set[str] = set()
    for node in _import_nodes(tree):
        if isinstance(node, ast.Import):
            hits = [_module_file(alias.name, known) for alias in node.names]
        elif node.level:
            parents = PurePosixPath(path).parents
            if node.level > len(parents):
                continue
            package = parents[node.level - 1]
            module = node.module or ""
            hits = [
                _relative_file(package, ".".join(filter(None, (module, alias.name))), known)
                or _relative_file(package, module, known)
                for alias in node.names
            ]
        else:
            module = node.module or ""
            hits = [
                _module_file(f"{module}.{alias.name}", known) or _module_file(module, known)
                for alias in node.names
            ]
        found.update(hit for hit in hits if hit)
    return found


def contract_scope(files: Sequence[str]) -> tuple[str, ...]:
    """The CANOPY contract's scope in the tree whose files are *files*.

    It is the declared roots, plus the file of every own module that a Python
    file under the importing roots imports, one hop. Parent packages are not
    added: importing ``a.b.c`` loads ``a/__init__.py``, and a change there is
    not a change to what the CANOPY code calls.
    """
    known = set(files)
    imported: set[str] = set()
    for path in sorted(known):
        if not path.endswith(".py") or not touches([path], CANOPY_IMPORTING_ROOTS):
            continue
        try:
            tree = ast.parse((kit.ROOT / path).read_text(encoding="utf-8"), filename=path)
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        imported |= _imported_files(path, tree, known)
    extra = sorted(path for path in imported if not touches([path], CANOPY_CONTRACT_ROOTS))
    return (*CANOPY_CONTRACT_ROOTS, *extra)


# ---------------------------------------------------------------------------
# The install: CANOPY and PyDMConverter at the pinned commits


def _proven() -> ModuleType:
    """The pin module, loaded by path from this tree under a name no import uses."""
    name = "_check_canopy_proven"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PROVEN_MODULE)
    if spec is None or spec.loader is None:
        raise Refused(f"cannot load {PROVEN_MODULE}")
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs, so its dataclasses resolve their module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _canopy_install() -> str:
    pins = _proven().PROVEN_COMMITS
    return (
        f"install canopy {pins['canopy']} and PyDMConverter {pins['PyDMConverter']} "
        "from git, PyDMConverter also as a uv pip --override"
    )


def _canopy_attestation(passed: bool) -> str:
    return str(_proven().attestation(passed))


CANOPY_INSTALL = Install(CANOPY_PROBE, _canopy_install, _canopy_attestation)


def _contract_command(
    check: Check,
    _selection: Selection,
    _context: kit.Context,
    flags: Sequence[str],
    written: Sequence[str],
) -> list[list[str]]:
    """The contract stopped at its bound. Both ``uv run`` calls pass
    ``--no-sync`` after the kit's ``--frozen``: a sync would move back the
    locked distributions the CANOPY install changed, and the contract would
    prove the lockfile instead."""
    argv = [*check.argv, *flags, *written]
    bounded = kit.bounded(check.bound_s, argv)
    frozen = len(kit.UV_RUN)
    return [[*bounded[:frozen], "--no-sync", *bounded[frozen:]]]


# ---------------------------------------------------------------------------
# The checks

CHECKS: tuple[Check, ...] = (
    Check(
        "canopy-marker",
        "invariant",
        BOTH,
        (*kit.UV_RUN, "pytest", *INVARIANT_TESTS),
        needs=("uv",),
        report="junit-pytest",
    ),
    Check(
        "canopy-contract",
        "tests",
        BOTH,
        (
            *(*kit.UV_RUN, "--no-sync", "pytest", *CONTRACT_TESTS),
            *("-p", "no:randomly", "--timeout=600"),
        ),
        needs=("uv",),
        bound_s=1800,
        exec_flags=True,
        derive_scope=contract_scope,
        command=_contract_command,
        install=CANOPY_INSTALL,
    ),
)

#: The CI job that calls the check command for the contract.
CALLER_JOBS = ("canopy-contract",)
