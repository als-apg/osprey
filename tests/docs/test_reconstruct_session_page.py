"""The reconstruct-a-session page shows the recipe the store runs, not a copy of it.

The page literalincludes the recipe file the history package loads, so the SQL
a reader copies from the docs is the SQL the tests run as a reader. A moved or
renamed recipe must fail here, not render as a Sphinx warning on an empty block.
"""

from __future__ import annotations

import re
from pathlib import Path

from osprey.history import queries

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "docs/source/how-to/health-and-monitoring/reconstruct-session.rst"
INDEX = PAGE.parent / "index.rst"
RECIPE = REPO_ROOT / "src/osprey/history/queries" / f"{queries.RECIPE}.sql"

_INCLUDE = re.compile(r"^\.\. literalinclude:: (\S+)\s*$", re.MULTILINE)


def test_the_page_includes_the_recipe_file() -> None:
    includes = _INCLUDE.findall(PAGE.read_text(encoding="utf-8"))
    resolved = [(PAGE.parent / target).resolve() for target in includes]
    assert RECIPE.resolve() in resolved, f"the page includes {includes}, not {RECIPE}"
    assert RECIPE.read_text(encoding="utf-8") == queries.load(queries.RECIPE)


def test_the_page_is_in_the_section_toctree() -> None:
    index = INDEX.read_text(encoding="utf-8")
    toctree = index[index.index(".. toctree::") :]
    assert re.search(r"^\s+reconstruct-session\s*$", toctree, re.MULTILINE)
