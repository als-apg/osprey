"""The hooks' ``osprey_safety_file.py`` is a byte-identical copy of the package module.

The hooks run under the system ``python3`` and import nothing from OSPREY, so
the one way OSPREY replaces a file whole ships beside them as a copy. The copy
is never edited on its own: a change is made in the package module and copied.
"""

from __future__ import annotations

from pathlib import Path

import osprey_connectors.safety_file as package_module
from osprey.cli.templates.artifact_library import parse_hook_frontmatter
from tests.hooks.conftest import import_hook

HOOK_COPY = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "claude_code"
    / "claude"
    / "hooks"
    / "osprey_safety_file.py"
)


def test_the_hook_copy_is_byte_identical_to_the_package_module():
    source = Path(package_module.__file__)

    assert source.read_bytes() == HOOK_COPY.read_bytes(), (
        f"edit {source.relative_to(HOOK_COPY.parents[6])} and copy it to "
        f"{HOOK_COPY.relative_to(HOOK_COPY.parents[6])}"
    )


def test_the_hook_copy_imports_as_a_helper_wired_to_no_event():
    module = import_hook("osprey_safety_file")

    assert callable(module.atomic_replace)
    assert parse_hook_frontmatter(HOOK_COPY) is None
