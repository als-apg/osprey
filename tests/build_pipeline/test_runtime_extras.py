"""Every package a runtime extra installs is imported by the code that ships.

An optional extra exists to make some import under ``src/`` or ``packages/``
work. A package no shipped file imports is dead weight on every install that
asks for the extra, so each runtime extra's requirements are mapped from
distribution to module and looked up in the imports of the shipped sources.
``docs`` and ``dev`` serve the repository, not an install, and are exempt;
``all`` only aggregates the others.
"""

from __future__ import annotations

import ast
import importlib.metadata
import re
import tomllib
from functools import cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

RUNTIME_EXTRAS: tuple[str, ...] = (
    "ariel",
    "ariel-proxy",
    "gchat",
    "knowledge",
    "screen-capture-linux",
    "teams",
    "virtual-accelerator",
)

NON_RUNTIME_EXTRAS: frozenset[str] = frozenset({"all", "dev", "docs"})

# Packages an extra installs to change another package's behaviour at run time
# rather than to be imported by the shipped code.
RUNTIME_ENABLERS: dict[str, str] = {
    "pysocks": "httplib2 honours HTTP(S)_PROXY only with PySocks installed",
}

# The modules each runtime-extra distribution provides. Read when the
# distribution is not installed in the test environment, or when its top-level
# name is a namespace several distributions share.
STATIC_MODULES: dict[str, tuple[str, ...]] = {
    "aioca": ("aioca",),
    "aiohttp-socks": ("aiohttp_socks",),
    "azure-servicebus": ("azure.servicebus",),
    "google-api-python-client": ("googleapiclient",),
    "google-auth": ("google.auth", "google.oauth2"),
    "google-cloud-pubsub": ("google.cloud.pubsub", "google.cloud.pubsub_v1"),
    "google-cloud-storage": ("google.cloud.storage",),
    "linkml-runtime": ("linkml_runtime",),
    "lume-pva-apg": ("lume_pva_apg",),
    "mss": ("mss",),
    "pcaspy": ("pcaspy",),
    "pillow": ("PIL",),
    "psycopg": ("psycopg",),
    "psycopg-pool": ("psycopg_pool",),
    "python-xlib": ("Xlib",),
}

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _optional_dependencies() -> dict[str, list[str]]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["optional-dependencies"]


def _distribution(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(requirement)
    assert match, f"unparseable requirement {requirement!r}"
    return _canonical(match.group(1))


@cache
def _installed_modules() -> dict[str, tuple[str, ...]]:
    """Map each installed distribution to the top-level modules only it provides."""
    owners: dict[str, set[str]] = {}
    for module, distributions in importlib.metadata.packages_distributions().items():
        canonical = {_canonical(d) for d in distributions}
        if len(canonical) == 1:
            owners.setdefault(canonical.pop(), set()).add(module)
    return {dist: tuple(sorted(modules)) for dist, modules in owners.items()}


def _modules_for(distribution: str) -> tuple[str, ...]:
    installed = _installed_modules().get(distribution)
    if installed:
        return installed
    return STATIC_MODULES.get(distribution, ())


@cache
def _shipped_imports() -> frozenset[str]:
    """Every dotted module name an import statement under src/ or packages/ names."""
    names: set[str] = set()
    for root in (REPO_ROOT / "src", REPO_ROOT / "packages"):
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names.add(node.module)
                    names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return frozenset(names)


def _imported(module: str) -> bool:
    prefix = f"{module}."
    return any(name == module or name.startswith(prefix) for name in _shipped_imports())


def test_every_extra_is_classified() -> None:
    declared = set(_optional_dependencies())
    assert declared == set(RUNTIME_EXTRAS) | NON_RUNTIME_EXTRAS


def test_runtime_enablers_are_declared() -> None:
    extras = _optional_dependencies()
    declared = {_distribution(r) for name in RUNTIME_EXTRAS for r in extras[name]}
    assert set(RUNTIME_ENABLERS) <= declared


@pytest.mark.parametrize("extra", RUNTIME_EXTRAS)
def test_every_runtime_extra_package_is_imported(extra: str) -> None:
    unimported: list[str] = []
    for requirement in _optional_dependencies()[extra]:
        distribution = _distribution(requirement)
        if distribution in RUNTIME_ENABLERS:
            continue
        modules = _modules_for(distribution)
        assert modules, f"{distribution}: add its modules to STATIC_MODULES"
        if not any(_imported(module) for module in modules):
            unimported.append(f"{distribution} ({', '.join(modules)})")
    assert not unimported, f"[{extra}] installs packages no shipped file imports: {unimported}"
