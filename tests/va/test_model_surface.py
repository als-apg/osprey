"""The model surface stays importable and testable without a server library.

The verbs of the surface over a composite are pinned in
``test_model_surface_for_view.py``; this module holds the import boundary.
"""

from __future__ import annotations

import ast
from pathlib import Path

from osprey.services.virtual_accelerator.serving import model_surface


def test_the_partition_module_imports_no_server_library() -> None:
    """The verbs run on the composite alone, so the module must stay
    importable and testable in process, with neither Channel Access server
    nor serving runner behind it."""
    tree = ast.parse(Path(model_surface.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    roots = {name.split(".")[0] for name in imported}
    assert roots.isdisjoint({"pcaspy", "p4p", "lume_pva_apg"})
    assert not any(name.endswith("serving.runner") for name in imported)
