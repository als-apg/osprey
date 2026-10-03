"""The facility token names nothing a container uses.

``facility.prefix`` has one reader under ``src/``: ``osprey knowledge``, which
mints the knowledge graph's identifiers with it. Every container is named by
``resolve_project_name()``. These checks keep it that way: a new reader of the
key anywhere else under ``src/`` fails here, naming the file and line.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

#: The spellings of the key and its template variable.
TOKEN_RE = re.compile(r"facility_prefix|facility\.prefix")

#: A lookup of the ``facility`` section, with or without a default.
FACILITY_GET_RE = re.compile(r"""get\(\s*["']facility["']\s*[,)]""")

#: A lookup of ``prefix`` on whatever the statement holds.
PREFIX_GET_RE = re.compile(r"""\.get\(\s*["']prefix["']""")

#: Files allowed to spell the key, relative to ``src/``: its one reader, a
#: docstring that names it as an example dotted key, the manifest that
#: declares it, and the presets that set it.
TOKEN_ALLOWED = frozenset(
    {
        "osprey/cli/knowledge_cmd.py",
        "osprey/deployment/web_terminals/lint.py",
        "osprey/profiles/config_key_manifest.yml",
        "osprey/profiles/presets/ariel-standalone.yml",
        "osprey/profiles/presets/channel-finder-standalone.yml",
        "osprey/profiles/presets/control-assistant.yml",
    }
)

#: Files allowed to read ``prefix`` off the ``facility`` section.
READER_ALLOWED = frozenset({"osprey/cli/knowledge_cmd.py"})


def _text_files(root: Path):
    """Yield ``(relative path, text)`` for every non-binary file under *root*, sorted."""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        yield path.relative_to(root).as_posix(), data.decode("utf-8", errors="replace")


def token_hits(root: Path, allowed: frozenset[str]) -> list[str]:
    """Return ``path:line`` for every spelling of the key outside *allowed*."""
    hits = []
    for rel, text in _text_files(root):
        if rel in allowed:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if TOKEN_RE.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()}")
    return hits


def _statement_spans(tree: ast.AST) -> list[tuple[int, int]]:
    """Return ``(first line, last line)`` of every statement's own text.

    A compound statement's span stops before its first nested statement, so the
    span of an ``if`` is its header, not its body.
    """
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        end = node.end_lineno or node.lineno
        for field in ("body", "orelse", "finalbody", "handlers", "cases"):
            children = getattr(node, field, None) or []
            if children:
                end = min(end, children[0].lineno - 1)
        spans.append((node.lineno, max(end, node.lineno)))
    return spans


def reader_hits(root: Path, allowed: frozenset[str]) -> list[str]:
    """Return ``path:line`` for every statement that reads ``prefix`` off ``facility``."""
    hits = []
    for rel, text in _text_files(root):
        if not rel.endswith(".py") or rel in allowed or not FACILITY_GET_RE.search(text):
            continue
        lines = text.splitlines()
        spans = _statement_spans(ast.parse(text, filename=rel))
        for lineno, line in enumerate(lines, start=1):
            match = FACILITY_GET_RE.search(line)
            if not match:
                continue
            enclosing = [span for span in spans if span[0] <= lineno <= span[1]]
            last = min(enclosing, key=lambda s: s[1] - s[0])[1] if enclosing else lineno
            rest = "\n".join([line[match.start() :], *lines[lineno:last]])
            if PREFIX_GET_RE.search(rest):
                hits.append(f"{rel}:{lineno}: {line.strip()}")
    return hits


def test_only_the_knowledge_graph_spells_the_facility_prefix():
    assert token_hits(SRC, TOKEN_ALLOWED) == []


def test_only_the_knowledge_graph_reads_prefix_off_the_facility_section():
    assert reader_hits(SRC, READER_ALLOWED) == []


@pytest.mark.parametrize(
    "source",
    [
        'token = (config.get("facility") or {}).get("prefix")\n',
        'token = config.get("facility", {}).get("prefix", "")\n',
        'token = (\n    config.get("facility")\n    or {}\n).get(\n    "prefix"\n)\n',
        'if (config.get("facility") or {}).get("prefix"):\n    pass\n',
    ],
    ids=["one-line", "with-default", "multi-line", "if-header"],
)
def test_a_planted_reader_is_caught(tmp_path: Path, source: str):
    (tmp_path / "planted.py").write_text(source)

    assert reader_hits(tmp_path, READER_ALLOWED) == [
        f"planted.py:{_first_facility_line(source)}: "
        f"{source.splitlines()[_first_facility_line(source) - 1].strip()}"
    ]


def test_a_facility_lookup_without_prefix_is_not_a_reader(tmp_path: Path):
    (tmp_path / "other.py").write_text(
        'name = config.get("facility")\nif name:\n    value = other.get("prefix")\n'
    )

    assert reader_hits(tmp_path, READER_ALLOWED) == []


def test_a_planted_spelling_is_caught(tmp_path: Path):
    (tmp_path / "template.j2").write_text("container_name: {{ facility_prefix }}-nginx\n")

    assert token_hits(tmp_path, TOKEN_ALLOWED) == [
        "template.j2:1: container_name: {{ facility_prefix }}-nginx"
    ]


def _first_facility_line(source: str) -> int:
    return next(
        i for i, line in enumerate(source.splitlines(), start=1) if FACILITY_GET_RE.search(line)
    )
