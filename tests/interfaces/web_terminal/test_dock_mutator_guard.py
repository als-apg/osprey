"""Only the layout holder mutates dockview.

The dock's layout lives in one place, the holder in ``dock-apply.js``: every
change to the tiles is computed as a new layout by the engine and applied by
the holder through ``api.fromJSON``. A module that calls a dockview mutator
itself (``api.addPanel``, ``dockApi.removePanel``, ``api.moveTo`` ...) changes
the dock behind the holder's back, so the holder's layout and the screen
disagree and the next apply undoes the change. This guard keeps the old
model from creeping back: no static JS file other than ``dock-apply.js``
calls a mutator on an ``api`` or ``dockApi`` receiver.

Comments are not exempt on purpose: a call quoted in a comment is either a
leftover or documentation that belongs in ``dock-apply.js``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The dockview mutators. Reading the dock (``api.panels``, ``getPanel``,
#: ``toJSON``) is fine anywhere; changing it is the holder's job.
MUTATORS = ("addPanel", "removePanel", "addGroup", "removeGroup", "fromJSON", "clear", "moveTo")

#: A mutator called on an ``api`` or ``dockApi`` receiver, optional chaining
#: and whitespace around the dot allowed. ``\b`` keeps ``holder.clear()`` and
#: ``myapi.clear()`` out.
PATTERN = re.compile(r"\b(?:api|dockApi)\s*\??\.\s*(?:" + "|".join(MUTATORS) + r")\s*\(")

#: Where static JS lives.
SEARCHED = ("src/osprey/interfaces",)

#: The one module allowed to mutate dockview, with the reason.
ALLOWED: dict[str, str] = {
    "src/osprey/interfaces/web_terminal/static/js/dock-apply.js": (
        "the layout holder: the single place a computed layout reaches dockview"
    ),
}


def _is_static_js(name: str) -> bool:
    parts = name.split("/")
    return (
        "static" in parts
        and "vendor" not in parts
        and name.endswith((".js", ".mjs"))
        and not name.endswith(".min.js")
    )


def _static_js_files() -> list[str]:
    """Every static JS file git knows under :data:`SEARCHED`, tracked or new and
    not ignored (a module not yet added is still the tree)."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *SEARCHED],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    names = [name for name in listed.stdout.decode("utf-8").split("\0") if name]
    return sorted(name for name in names if _is_static_js(name))


def files_calling_a_dock_mutator() -> dict[str, list[str]]:
    """Each searched file that calls a mutator, with the offending lines."""
    found: dict[str, list[str]] = {}
    for name in _static_js_files():
        path = REPO_ROOT / name
        if not path.is_file():
            continue  # deleted in the working tree
        lines = [
            f"{number}: {line.strip()}"
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if PATTERN.search(line)
        ]
        if lines:
            found[name] = lines
    return found


def test_only_the_layout_holder_calls_a_dockview_mutator() -> None:
    offenders = {
        name: lines for name, lines in files_calling_a_dock_mutator().items() if name not in ALLOWED
    }
    assert offenders == {}, (
        "static JS that mutates dockview outside the layout holder (compute the layout "
        f"and hand it to dock-apply.js instead): {offenders}"
    )


def test_the_scan_sees_the_static_js_tree() -> None:
    """A scan that lists nothing passes vacuously; it must at least see the
    holder and its neighbours."""
    files = _static_js_files()
    assert "src/osprey/interfaces/web_terminal/static/js/dock-apply.js" in files
    assert "src/osprey/interfaces/web_terminal/static/js/dock-sync.js" in files
    assert not any("/vendor/" in name for name in files)


def test_every_allowed_module_exists() -> None:
    """A stale exemption would excuse whatever later takes its name; and the
    holder really is found by the scan, so the pattern matches the real call
    shape (``api.fromJSON(...)``), not only the samples below."""
    found = files_calling_a_dock_mutator()
    for name in ALLOWED:
        assert (REPO_ROOT / name).is_file(), name
        assert name in found, name


def test_the_pattern_flags_mutator_calls() -> None:
    flagged = [
        "api.addPanel({ id: 'x' });",
        "dockApi.removePanel(panel);",
        "api?.fromJSON(json);",
        "dockApi .moveTo ( group );",
        "this.api.addGroup();",
        "api.removeGroup(g)",
        "if (x) api.clear();",
    ]
    for line in flagged:
        assert PATTERN.search(line), line


def test_the_pattern_ignores_reads_and_other_receivers() -> None:
    ignored = [
        "holder.clear();",
        "myapi.clear();",
        "dockApiFor(x).addPanel",
        "api.panels.length",
        "api.getPanel(id)",
        "api.toJSON()",
        "group.api.setActive()",
        "const addPanel = api.addPanel;",
    ]
    for line in ignored:
        assert not PATTERN.search(line), line
