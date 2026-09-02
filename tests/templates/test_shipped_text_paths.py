"""Shipped text names only files and pages that exist where it is read.

Three guards over the paths the shipped templates cite. Each one pins a class
of drift that has recurred: a prompt that sends the agent to a ``src/osprey/``
path (only a source checkout has one — a facility runs the installed package),
a ``config.yml`` comment that cites a docs page by a path the docs tree no
longer has, and a generated README that tells the operator to copy or edit a
file the renderer never produces.

All three scan template SOURCE, not a render: the offending text is literal in
the template, and a scan over the source names the line to fix.
"""

from __future__ import annotations

import re
from pathlib import Path

import osprey

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_SOURCE = REPO_ROOT / "docs" / "source"
PACKAGE_ROOT = Path(osprey.__file__).resolve().parent
TEMPLATES = PACKAGE_ROOT / "templates"
CLAUDE_CODE_TEMPLATES = TEMPLATES / "claude_code"

_SKIP_DIRS = {"__pycache__", "node_modules"}
_SKIP_SUFFIXES = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf"}


def _text_files(root: Path) -> list[Path]:
    """Every readable text file under *root*, skipping caches and binaries."""
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if _SKIP_DIRS & set(path.relative_to(root).parts):
            continue
        if path.suffix.lower() in _SKIP_SUFFIXES:
            continue
        found.append(path)
    return found


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _describe(hits: list[tuple[str, int, str]]) -> str:
    return "\n".join(f"  {path}:{lineno}: {line}" for path, lineno, line in hits)


# ---------------------------------------------------------------------------
# 1. Prompts cite the installed package, never a source checkout
# ---------------------------------------------------------------------------


def test_no_shipped_prompt_or_hook_cites_a_source_checkout_path() -> None:
    """Nothing under ``templates/claude_code/`` names a ``src/osprey/`` path.

    The agent reads these files on a facility's deployment, where OSPREY is an
    installed package: there is no ``src/`` tree to open. A worked example is
    named either by its dotted module (``osprey.services.bluesky_bridge...``)
    or by joining onto ``Path(osprey.__file__).parent``.
    """
    offenders: list[tuple[str, int, str]] = []
    for path in _text_files(CLAUDE_CODE_TEMPLATES):
        text = _read(path)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "src/osprey/" in line:
                offenders.append((_rel(path), lineno, line.strip()))
    assert offenders == [], "Shipped prompt text cites a source-checkout path:\n" + _describe(
        offenders
    )


# ---------------------------------------------------------------------------
# 2. Every docs page a template cites is a page the docs tree has
# ---------------------------------------------------------------------------

#: The top-level sections of the OSPREY docs tree. A ``docs/`` path is only an
#: OSPREY docs citation when its first segment is one of these: ``docs/runbook.md``
#: in a deployment repo, or a design note under ``docs/superpowers/``, is not.
_DOCS_SECTIONS = sorted(
    p.name for p in DOCS_SOURCE.iterdir() if p.is_dir() and not p.name.startswith("_")
)

#: A docs page named by tree path: ``docs/how-to/x.rst`` (the retired short
#: form) or ``docs/source/how-to/x.rst`` (the real path). Both map to
#: ``docs/source/<path>``.
_DOCS_TREE_PATH = re.compile(
    r"\bdocs/(?:source/)?((?:"
    + "|".join(map(re.escape, _DOCS_SECTIONS))
    + r")/[\w./-]+?\.(?:rst|md))\b"
)

#: A docs page named by its published URL, which is the form a deployment can
#: actually open. ``how-to/x.html`` maps to ``docs/source/how-to/x.rst``.
_DOCS_PUBLISHED_URL = re.compile(r"https://als-apg\.github\.io/osprey/([\w./-]+?)\.html\b")


def _docs_page_exists(stem: str) -> bool:
    """``stem`` is a docs-relative path without extension (``how-to/x``)."""
    return any((DOCS_SOURCE / f"{stem}{suffix}").is_file() for suffix in (".rst", ".md"))


def _docs_citations(text: str) -> list[tuple[int, str, str]]:
    """``(lineno, cited text, docs-relative stem)`` for every citation in *text*."""
    cited: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in _DOCS_TREE_PATH.finditer(line):
            stem = re.sub(r"\.(?:rst|md)$", "", match.group(1))
            cited.append((lineno, match.group(0), stem))
        for match in _DOCS_PUBLISHED_URL.finditer(line):
            cited.append((lineno, match.group(0), match.group(1)))
    return cited


def test_every_docs_page_the_package_cites_exists() -> None:
    """A cited docs page, by tree path or by published URL, resolves under ``docs/source/``.

    Covers the whole installed package rather than the templates alone: a
    runtime message that cites a page reaches the operator by the same route a
    ``config.yml`` comment does.
    """
    assert DOCS_SOURCE.is_dir(), f"docs tree not found at {DOCS_SOURCE}"
    offenders: list[tuple[str, int, str]] = []
    seen_any = False
    for path in _text_files(PACKAGE_ROOT):
        text = _read(path)
        if text is None:
            continue
        for lineno, cited, stem in _docs_citations(text):
            seen_any = True
            if not _docs_page_exists(stem):
                offenders.append((_rel(path), lineno, cited))
    assert seen_any, "found no docs citations at all — the scan has stopped working"
    assert offenders == [], "Shipped text cites a docs page that does not exist:\n" + _describe(
        offenders
    )


def test_docs_citation_patterns_still_match_their_examples() -> None:
    """Each pattern proves it can fire, so a clean sweep is a result and not a bug."""
    assert _DOCS_TREE_PATH.search("see docs/how-to/monitor-agent.rst for more")
    assert _DOCS_TREE_PATH.search("(docs/source/how-to/deploy-project/index.rst)")
    assert _DOCS_PUBLISHED_URL.search(
        "https://als-apg.github.io/osprey/how-to/deploy-a-facility.html"
    )
    assert _docs_citations("x docs/how-to/a/b.rst y")[0][2] == "how-to/a/b"
    # Not OSPREY docs: a deployment repo's own docs/, a repo-internal design note.
    assert _docs_citations("mirrors project/docs/runbook.md") == []
    assert _docs_citations("see docs/superpowers/frontend-foundation/PROGRAM.md") == []
    assert _docs_citations("https://als-apg.github.io/osprey/reference/ports.html#x")[0][2] == (
        "reference/ports"
    )


# ---------------------------------------------------------------------------
# 3. A generated README names only files the renderer produces
# ---------------------------------------------------------------------------

#: Text a shipped README or ``.env.example`` may not carry. Each entry is a
#: file or command that has sent an operator to something the renderer does not
#: produce: the example env file is ``.env.example`` (never ``env.example``),
#: and a generated project has no Python package of its own, so no
#: ``registry.py`` and no ``context_classes.py`` exist to edit.
_NOT_RENDERED = (
    "cp env.example",
    "registry.py",
    "context_classes.py",
)


def _shipped_readmes_and_env_examples() -> list[Path]:
    files = sorted(TEMPLATES.glob("apps/*/README.md.j2"))
    files.append(TEMPLATES / "project" / "README.md.j2")
    files.append(TEMPLATES / "project" / "env.example.j2")
    return files


def test_shipped_readmes_name_only_files_the_renderer_produces() -> None:
    files = _shipped_readmes_and_env_examples()
    assert all(path.is_file() for path in files), [str(p) for p in files if not p.is_file()]
    offenders: list[tuple[str, int, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for needle in _NOT_RENDERED:
                if needle in line:
                    offenders.append((_rel(path), lineno, line.strip()))
    assert offenders == [], "A shipped README names a file the renderer never produces:\n" + (
        _describe(offenders)
    )
