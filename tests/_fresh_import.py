"""What a module costs to import, measured in a fresh interpreter.

The test process has imported half the tree already, so ``sys.modules`` there
says nothing about what one import pulls in. A child interpreter with only
``src/`` on its path does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Collection, Mapping
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")


def modules_added_by_import(
    module: str, *, preload: Collection[str] = (), env: Mapping[str, str] | None = None
) -> set[str]:
    """Return the module names a fresh import of ``module`` adds to ``sys.modules``.

    Args:
        module: Dotted name to import in a child interpreter.
        preload: Modules the child imports before it takes the baseline, so
            their own import graph is not counted against ``module``.
        env: Variables set in the child on top of this process's environment,
            e.g. an empty ``PATH`` to measure the import with no ``git``.

    Returns:
        Every module name the import added, stdlib included.

    Raises:
        AssertionError: If the child interpreter fails to import the module.
    """
    code = (
        "import json, sys;"
        + "".join(f"import {name};" for name in preload)
        + "before = set(sys.modules);"
        + f"import {module};"
        + "print(json.dumps(sorted(set(sys.modules) - before)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {}), "PYTHONPATH": _SRC},
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, f"fresh import of {module} failed:\n{result.stderr}"
    return set(json.loads(result.stdout))


def non_stdlib_added_by_import(
    module: str,
    allowed: Collection[str] = (),
    *,
    preload: Collection[str] = (),
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """The non-stdlib modules a fresh import of ``module`` adds beyond ``allowed``.

    Returns:
        The offending module names, sorted — empty when the import stays
        inside ``allowed`` and the stdlib.
    """
    added = modules_added_by_import(module, preload=preload, env=env) - set(allowed)
    return sorted(name for name in added if name.split(".")[0] not in sys.stdlib_module_names)
