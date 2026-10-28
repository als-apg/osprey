"""Release gate: no shipped file spells a retired write key.

``control_system.writes`` (the tier) and the per-login switch replaced the
per-type posture keys ``control_system.writes_enabled`` and
``control_system.connector.<type>.writes_enabled``. The build still reads the
retired spellings for one release (PR-3) and refuses them from 2026.11.0; past
that, a file under ``src/``, ``packages/``, ``plugins/`` or ``docs/source``
that spells one is either migration code or stale advice.

What counts as a spelling: the dotted key (``control_system.writes_enabled``,
``control_system.connector.<type>.writes_enabled``, ``connector.epics.writes_enabled``
and any placeholder in the type segment), the segment tuple
``("control_system", "writes_enabled")``, and an indented YAML key
``writes_enabled:`` in a non-Python file. Bare ``writes_enabled`` is NOT a
retired key: it is a live keyword, report field and compose render variable.

``control_system.write_tools`` is exempt until its sunset (PR-5a): it is not a
pattern here at all.

The allow-list is closed and names paths. Two entries are narrowed further:
the manifest may spell a retired key only inside a retired key's own row, and
the connector contracts page only inside its upgrade section. Entries outside
the scan roots (the migration tests, ``previous_release_repo``, the changelog)
are listed so the list is the whole record; every entry is checked live, so a
path that is gone or no longer spells a retired key fails as stale.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

ROOTS = ("src", "packages", "plugins", "docs/source")
SKIP_DIRS = frozenset(
    {
        "__pycache__",
        "node_modules",
        ".venv",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)
SCAN_SUFFIXES = (
    ".py",
    ".md",
    ".rst",
    ".txt",
    ".js",
    ".ts",
    ".html",
    ".j2",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".sh",
)

_BOUND = r"(?<![A-Za-z0-9_])"
# Dotted spelling: control_system.writes_enabled, control_system.connector.<t>.writes_enabled,
# connector.<t>.writes_enabled. Each segment is one run of non-space, non-dot characters,
# so a placeholder (<type>, {type}, *) counts and ``connector._writes_enabled`` does not.
_DOTTED = re.compile(
    _BOUND + r"(?:control_system|connector)(?:\.[^\s.'\"`]+)*\.writes_enabled(?![A-Za-z0-9_])"
)
_TUPLE = re.compile(r"""['"]control_system['"]\s*,\s*['"]writes_enabled['"]""")
# A YAML key in a document or a snippet. Python is excluded: there ``writes_enabled:``
# is a live annotation or dict key.
_YAML_KEY = re.compile(r"^\s*(?:-\s*)?writes_enabled\s*:")


def _hit_lines(path: Path, content: str) -> list[int]:
    hits: list[int] = []
    yaml_like = path.suffix != ".py"
    for i, line in enumerate(content.splitlines(), start=1):
        if _DOTTED.search(line) or _TUPLE.search(line) or (yaml_like and _YAML_KEY.search(line)):
            hits.append(i)
    return hits


# --- narrowing scopes: the lines of an allowed file a retired key may occupy ---

_MANIFEST_ROW = re.compile(r"^  (\S[^:]*):\s*$")


def _manifest_retired_rows(content: str) -> set[int]:
    """Lines inside a manifest row whose own key is a retired spelling."""
    allowed: set[int] = set()
    in_retired = False
    for i, line in enumerate(content.splitlines(), start=1):
        row = _MANIFEST_ROW.match(line)
        if row or (line and not line.startswith(" ")):
            in_retired = bool(row and _DOTTED.fullmatch(row.group(1)))
        if in_retired:
            allowed.add(i)
    return allowed


def _docs_upgrade_section(content: str) -> set[int]:
    """Lines from ``.. _write-safety-upgrade:`` up to the next label."""
    allowed: set[int] = set()
    inside = False
    for i, line in enumerate(content.splitlines(), start=1):
        if line.startswith(".. _"):
            inside = line.strip() == ".. _write-safety-upgrade:"
        if inside:
            allowed.add(i)
    return allowed


Scope = Callable[[str], set[int]] | None

# Closed allow-list: repo-relative path (file, or directory covering its tree)
# -> (why, scope). A scope of None allows the whole file.
ALLOWED: dict[str, tuple[str, Scope]] = {
    # Migration code: the PR-3 readers, the normalizer, revalidate, and the
    # lint that names the retired leaf it warns on.
    "src/osprey/cli/build_profile_document.py": ("migration normalizer (PR-3)", None),
    "src/osprey/cli/init_revalidate.py": ("osprey init --revalidate rewrite", None),
    "src/osprey/deployment/web_terminals/lint.py": ("retired-leaf lint (PR-3)", None),
    # Migration tests.
    "tests/cli/test_retired_write_keys.py": ("migration test", None),
    "tests/cli/test_retired_write_keys_cli_set.py": ("migration test", None),
    "tests/cli/test_init_revalidate.py": ("migration test", None),
    "tests/cli/test_init_revalidate_write_keys.py": ("migration test", None),
    "tests/deployment/web_terminals/test_lint.py": ("migration test", None),
    # A repo as the previous release left it.
    "tests/fixtures/previous_release_repo": ("previous release fixture", None),
    # The manifest's retired rows, and nothing else in it.
    "src/osprey/profiles/config_key_manifest.yml": ("retired rows", _manifest_retired_rows),
    # The changelog names what it retired.
    "CHANGELOG.md": ("changelog", None),
    # The docs upgrade section, and nothing else on the page.
    "docs/source/reference/contracts/connectors.rst": (
        "upgrade section",
        _docs_upgrade_section,
    ),
}


def _walk(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix in SCAN_SUFFIXES and ".bak" not in name:
                files.append(path)
    return files


def _entry_for(rel: str, allowed: dict[str, tuple[str, Scope]]) -> str | None:
    for key in allowed:
        if rel == key or rel.startswith(key.rstrip("/") + "/"):
            return key
    return None


def _offenders(
    repo_root: Path = _REPO_ROOT,
    roots: tuple[str, ...] = ROOTS,
    allowed: dict[str, tuple[str, Scope]] = ALLOWED,
) -> list[str]:
    offenders: list[str] = []
    for root in roots:
        base = repo_root / root
        if not base.exists():
            continue
        for path in _walk(base):
            if path.resolve() == Path(__file__).resolve():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            hits = _hit_lines(path, content)
            if not hits:
                continue
            rel = path.relative_to(repo_root).as_posix()
            entry = _entry_for(rel, allowed)
            if entry is not None:
                scope = allowed[entry][1]
                if scope is None:
                    continue
                permitted = scope(content)
                hits = [n for n in hits if n not in permitted]
                if not hits:
                    continue
            lines = content.splitlines()
            excerpt = "; ".join(f"{n}: {lines[n - 1].strip()}" for n in hits[:3])
            offenders.append(f"{rel} -> {excerpt}")
    return offenders


def _stale_entries(
    repo_root: Path = _REPO_ROOT,
    allowed: dict[str, tuple[str, Scope]] = ALLOWED,
) -> list[str]:
    stale: list[str] = []
    for rel, (_why, scope) in allowed.items():
        target = repo_root / rel
        if not target.exists():
            stale.append(f"{rel}: path is gone")
            continue
        live = False
        for path in _walk(target):
            content = path.read_text(encoding="utf-8", errors="ignore")
            hits = _hit_lines(path, content)
            if scope is not None:
                permitted = scope(content)
                hits = [n for n in hits if n in permitted]
            if hits:
                live = True
                break
        if not live:
            stale.append(f"{rel}: no longer spells a retired write key")
    return stale


def test_no_retired_write_keys() -> None:
    offenders = _offenders()
    assert not offenders, (
        "Retired write keys spelled outside the closed allow-list. Name "
        "control_system.writes (the tier) and the switch instead:\n  " + "\n  ".join(offenders)
    )


def test_allow_list_has_no_stale_entry() -> None:
    stale = _stale_entries()
    assert not stale, "Stale allow-list entries; remove them:\n  " + "\n  ".join(stale)


@pytest.mark.parametrize(
    ("relpath", "text", "line"),
    [
        ("packages/pkg/src/mod.py", 'KEY = "control_system.writes_enabled"\n', 1),
        ("src/osprey/x.md", "set `control_system.connector.<type>.writes_enabled`\n", 1),
        ("plugins/p/skill.md", "use connector.epics.writes_enabled here\n", 1),
        ("docs/source/page.rst", "control_system:\n  writes_enabled: true\n", 2),
        ("src/osprey/y.py", 'path = ("control_system", "writes_enabled")\n', 1),
    ],
)
def test_negative_control_planted_key_fails(
    tmp_path: Path, relpath: str, text: str, line: int
) -> None:
    planted = tmp_path / relpath
    planted.parent.mkdir(parents=True)
    planted.write_text(text, encoding="utf-8")
    expected = text.splitlines()[line - 1].strip()
    assert _offenders(tmp_path, ROOTS, {}) == [f"{relpath} -> {line}: {expected}"]


@pytest.mark.parametrize(
    "text",
    [
        "control_system.writes: allowed\n",
        "control_system.write_tools: [x]\n",
        "def f(*, writes_enabled: bool) -> None: ...\n",
        'report = {"writes_enabled": True}\n',
        "allowed = connector._writes_enabled\n",
        "{% if svc.writes_enabled %}\n",
    ],
)
def test_live_spellings_are_not_retired_keys(tmp_path: Path, text: str) -> None:
    planted = tmp_path / "src" / "live.py"
    planted.parent.mkdir(parents=True)
    planted.write_text(text, encoding="utf-8")
    assert _offenders(tmp_path, ROOTS, {}) == []


def test_scoped_entry_rejects_a_hit_outside_its_scope(tmp_path: Path) -> None:
    page = tmp_path / "docs/source/reference/contracts/connectors.rst"
    page.parent.mkdir(parents=True)
    page.write_text(
        "Intro names ``control_system.writes_enabled``.\n\n"
        ".. _write-safety-upgrade:\n\n"
        "Upgrade: ``control_system.writes_enabled`` is retired.\n\n"
        ".. _next:\n",
        encoding="utf-8",
    )
    rel = "docs/source/reference/contracts/connectors.rst"
    assert _offenders(tmp_path, ROOTS, {rel: ALLOWED[rel]}) == [
        f"{rel} -> 1: Intro names ``control_system.writes_enabled``."
    ]


def test_stale_entry_is_reported(tmp_path: Path) -> None:
    clean = tmp_path / "src" / "clean.py"
    clean.parent.mkdir(parents=True)
    clean.write_text("TIER = 'control_system.writes'\n", encoding="utf-8")
    allowed: dict[str, tuple[str, Scope]] = {
        "src/clean.py": ("no longer needed", None),
        "src/gone.py": ("deleted", None),
    }
    assert _stale_entries(tmp_path, allowed) == [
        "src/clean.py: no longer spells a retired write key",
        "src/gone.py: path is gone",
    ]
