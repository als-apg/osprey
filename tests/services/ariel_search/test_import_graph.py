"""Import-graph guard for the ARIEL search package.

The query tokenizer lives in ``vocabulary/model.py`` and the expander under
``vocabulary/``, so ``config.py`` importing ``vocabulary.entry`` at module level
and ``search_fts.py`` importing ``vocabulary.model`` close no import cycle.
Each module is imported in a fresh, isolated interpreter so that a module
already cached in this test process cannot mask a cycle.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

ENTRY_MODULES = (
    "osprey.services.ariel_search.database.search_fts",
    "osprey.services.ariel_search",
    "osprey.services.ariel_search.search.qmd",
    "osprey.services.ariel_search.enhancement.text_embedding.embedder",
)


@pytest.mark.parametrize("module", ENTRY_MODULES)
def test_module_imports_cleanly_in_fresh_interpreter(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"importing {module} failed (exit {result.returncode}):\n{result.stderr}"
    )
    assert "ImportError" not in result.stderr, result.stderr
