"""``osprey.deployment.write_gates``: the one list of write-gated tools.

The list is the floor every write gate degrades to, so it is held equal to the
copies that cannot import it (the hooks' shared literal) and to the readers
that do (the headless runner, the audit middleware), and the module stays a
leaf a renderer can import without the web stack.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from osprey.agent_runner import write_tools as runner
from osprey.deployment.write_gates import WRITE_GATED_TOOLS
from osprey.mcp_server import audit_middleware

REPO = Path(__file__).resolve().parents[2]
HOOK_LOG = (
    REPO
    / "src"
    / "osprey"
    / "templates"
    / "claude_code"
    / "claude"
    / "hooks"
    / "osprey_hook_log.py"
)
LEAF = "osprey.deployment.write_gates"
LEAF_SOURCE = REPO / "src" / "osprey" / "deployment" / "write_gates.py"


def _hook_literal(name: str) -> list[str]:
    """The list literal ``name`` is assigned in the hooks' shared helper file."""
    tree = ast.parse(HOOK_LOG.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned in {HOOK_LOG}")


class TestOneList:
    def test_every_name_is_fully_qualified(self):
        assert all(name.startswith("mcp__") and name.count("__") == 2 for name in WRITE_GATED_TOOLS)
        assert len(set(WRITE_GATED_TOOLS)) == len(WRITE_GATED_TOOLS)

    def test_it_is_the_hooks_fallback_write_tools(self):
        assert list(WRITE_GATED_TOOLS) == _hook_literal("FALLBACK_WRITE_TOOLS")

    def test_the_runner_and_the_middleware_read_it(self):
        assert runner._FALLBACK_WRITE_TOOLS == list(WRITE_GATED_TOOLS)
        assert audit_middleware._FALLBACK_WRITE_TOOLS == list(WRITE_GATED_TOOLS)


# ---------------------------------------------------------------------------
# A leaf
# ---------------------------------------------------------------------------


def _modules_after_importing(module: str) -> set[str]:
    probe = f"import json, sys; import {module}; print(json.dumps(sorted(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return set(json.loads(result.stdout))


class TestLeaf:
    def test_importing_it_loads_no_web_stack(self):
        loaded = {name.partition(".")[0] for name in _modules_after_importing(LEAF)}

        assert "starlette" not in loaded
        assert "fastapi" not in loaded
        assert "yaml" not in loaded

    def test_it_imports_no_osprey_module(self):
        imported: set[str] = set()
        for node in ast.walk(ast.parse(LEAF_SOURCE.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.partition(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.partition(".")[0])

        assert "osprey" not in imported
        assert "osprey_connectors" not in imported
