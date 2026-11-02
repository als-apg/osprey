"""The sidecar panel registry is a leaf, readable without the web stack.

``osprey.profiles.web_panels`` reads the companion-server registry and the
shared middleware, so importing it loads starlette and the web credential
module. The MCP server registry and the credential module both need to know
what a sidecar panel declares, and neither may pay that price: the first is
imported by every build and by the agent runner, and the second is imported by
the middleware ``web_panels`` itself reaches, so going through ``web_panels``
would be a cycle.

``osprey.profiles.sidecar_panels`` is therefore a module that imports nothing
but ``dataclasses``. Each check runs in a fresh interpreter: a module an
earlier test in this session imported would make an in-process ``sys.modules``
assertion pass or fail for the wrong reason.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

LEAF = "osprey.profiles.sidecar_panels"
LEAF_SOURCE = (
    Path(__file__).resolve().parents[2] / "src" / "osprey" / "profiles" / "sidecar_panels.py"
)


def _modules_after_importing(module: str) -> set[str]:
    """The names in ``sys.modules`` of a fresh interpreter that imported *module*."""
    probe = f"import json, sys; import {module}; print(json.dumps(sorted(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return set(json.loads(result.stdout))


def _top_level(modules: set[str]) -> set[str]:
    return {name.partition(".")[0] for name in modules}


@pytest.mark.parametrize(
    "module",
    [
        "starlette",
        "fastapi",
        "osprey.interfaces.web_auth",
        "osprey.interfaces.common_middleware",
        "osprey.registry.web",
        "osprey.registry.mcp",
        "osprey.profiles.web_panels",
    ],
)
def test_importing_the_leaf_loads_none_of_its_readers(module):
    loaded = _modules_after_importing(LEAF)

    assert module not in loaded


def test_the_leaf_imports_only_dataclasses():
    """Read from the source, so a deferred import inside a function counts too."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(LEAF_SOURCE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))

    assert imported == {"__future__", "dataclasses"}
