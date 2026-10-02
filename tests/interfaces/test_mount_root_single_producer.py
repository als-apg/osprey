"""The per-user mount root is spelled once, and every producer derives from it."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import osprey
from osprey.deployment.web_terminals.render import _user_card, terminal_login_url
from osprey.interfaces import common_middleware
from osprey.interfaces.common_middleware import (
    TERMINAL_USER_ENV,
    URL_MOUNT_ROOT,
    compute_url_prefix,
    url_mount_prefix,
)
from osprey.services.auth_sidecar.return_to import safe_return_to


def test_the_mount_prefix_is_the_root_and_one_user_segment():
    assert URL_MOUNT_ROOT == "/u"
    assert url_mount_prefix("alice") == "/u/alice"


def test_the_container_prefix_derives_from_the_root(monkeypatch):
    monkeypatch.setattr(common_middleware, "URL_MOUNT_ROOT", "/m")
    monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
    assert compute_url_prefix() == "/m/alice"


def test_the_login_url_and_the_landing_card_derive_from_the_mount_root(monkeypatch):
    monkeypatch.setattr(common_middleware, "URL_MOUNT_ROOT", "/m")
    config = {"modules": {"web_terminals": {"external_origin": "https://ops.example.org"}}}
    assert (
        terminal_login_url(config, "alice", "s/t") == "https://ops.example.org/m/alice/?token=s%2Ft"
    )
    assert _user_card({"name": "alice", "persona": None}, frozenset())["url"] == "/m/alice/"


def test_the_sidecars_default_return_to_derives_from_the_root(monkeypatch):
    monkeypatch.setattr(common_middleware, "URL_MOUNT_ROOT", "/m")
    assert safe_return_to("", "alice") == "/m/alice/"
    assert safe_return_to("https://evil.example/", "alice") == "/m/alice/"


#: The file that holds the one spelling, relative to the scanned package.
_DEFINING_MODULE = Path("interfaces/common_middleware.py")

_JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HASH_COMMENT = re.compile(r"(^|(?<=\s))#.*$", re.MULTILINE)

#: Characters after which a ``/`` opens a regex literal rather than a division.
_REGEX_PRECEDERS = set("(,=:[!&|?{};+-*%<>~^")


def _blank(text: str) -> str:
    """``text`` with every character but newlines replaced by a space."""
    return re.sub(r"[^\n]", " ", text)


def _drop(pattern: re.Pattern[str], text: str) -> str:
    return pattern.sub(lambda m: _blank(m.group(0)), text)


def _strip_js(text: str) -> tuple[str, list[str]]:
    """Drop ``//`` and ``/* */`` comments from JS, keeping strings and regexes intact.

    Returns the text with comments blanked (newlines kept, so line numbers
    hold) and the contents of every quoted string token.
    """
    out: list[str] = []
    strings: list[str] = []
    i, n = 0, len(text)
    prev = ""
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            end = text.find("\n", i)
            end = n if end == -1 else end
            out.append(_blank(text[i:end]))
            i = end
            continue
        if ch == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            end = n if end == -1 else end + 2
            out.append(_blank(text[i:end]))
            i = end
            continue
        if ch in "'\"`":
            j = i + 1
            while j < n and text[j] != ch and not (ch != "`" and text[j] == "\n"):
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            token = text[i:j]
            strings.append(token[1:-1])
            out.append(token)
            prev = ch
            i = j
            continue
        if ch == "/" and (prev == "" or prev in _REGEX_PRECEDERS):
            j, in_class = i + 1, False
            while j < n and text[j] != "\n":
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "[":
                    in_class = True
                elif text[j] == "]":
                    in_class = False
                elif text[j] == "/" and not in_class:
                    break
                j += 1
            j = min(j + 1, n)
            out.append(text[i:j])
            prev = "/"
            i = j
            continue
        out.append(ch)
        if not ch.isspace():
            prev = ch
        i += 1
    return "".join(out), strings


def _hits_in_text(rel: Path, text: str, needle: str) -> list[str]:
    return [f"{rel}:{no}" for no, line in enumerate(text.splitlines(), 1) if needle in line]


def _python_hits(rel: Path, source: str, root: str) -> list[str]:
    tree = ast.parse(source)
    needle = f"{root}/"
    inside_fstring: set[int] = set()
    docstrings: set[int] = set()
    definition: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            inside_fstring.update(id(child) for child in ast.walk(node) if child is not node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            docstrings.add(id(node.value))
        elif (
            rel == _DEFINING_MODULE
            and isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "URL_MOUNT_ROOT" for t in node.targets)
        ):
            definition.add(id(node.value))

    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr) and id(node) not in inside_fstring:
            text = "".join(
                part.value if isinstance(part, ast.Constant) else "{}" for part in node.values
            )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in inside_fstring
            and id(node) not in docstrings
        ):
            text = node.value
            if text == root and id(node) not in definition:
                hits.append(f"{rel}:{node.lineno}")
                continue
        else:
            continue
        for offset, line in enumerate(text.splitlines()):
            stripped = line.strip()
            if stripped.startswith(("#", "//")):
                continue
            if needle in line:
                hits.append(f"{rel}:{node.lineno + offset}")
    return hits


def _js_hits(rel: Path, text: str, root: str) -> list[str]:
    code, strings = _strip_js(text)
    hits = _hits_in_text(rel, code, f"{root}/")
    if root in strings:
        hits.append(f"{rel}: string {root!r}")
    return hits


def _scan(package: Path) -> list[str]:
    """Every place under ``package`` that spells the mount root outside prose."""
    root = common_middleware.URL_MOUNT_ROOT
    needle = f"{root}/"
    hits: list[str] = []
    for path in sorted(package.rglob("*")):
        rel = path.relative_to(package)
        if not path.is_file() or "vendor" in rel.parts or path.name.endswith(".min.js"):
            continue
        suffix = path.suffix
        if suffix not in {".py", ".js", ".mjs", ".ts", ".html", ".j2"} and (
            path.name != "Dockerfile"
        ):
            continue
        text = path.read_text(encoding="utf-8")
        if suffix == ".py":
            hits += _python_hits(rel, text, root)
        elif suffix in {".js", ".mjs", ".ts"}:
            hits += _js_hits(rel, text, root)
        elif suffix == ".html":
            text = _drop(_HTML_COMMENT, _drop(_JINJA_COMMENT, text))
            hits += _js_hits(rel, text, root)
        else:
            text = _drop(_HASH_COMMENT, _drop(_JINJA_COMMENT, text))
            hits += _hits_in_text(rel, text, needle)
    return hits


def test_the_mount_root_is_spelled_once_in_the_package():
    hits = _scan(Path(osprey.__file__).resolve().parent)
    assert hits == [], (
        "the per-user mount root is spelled outside "
        "osprey.interfaces.common_middleware.URL_MOUNT_ROOT; build the URL with "
        "url_mount_prefix(user) or read URL_MOUNT_ROOT instead: " + ", ".join(hits)
    )


@pytest.mark.parametrize(
    ("name", "content", "flagged"),
    [
        ("producer.py", 'def f(user):\n    return f"/u/{user}/"\n', True),
        ("split.py", 'def f(user):\n    return (f"{user}" "/u/" f"{user}")\n', True),
        ("app.js", "export const m = (p) => p.startsWith('/u/');\n", True),
        ("root.js", "export const r = '/u';\n", True),
        ("nginx.conf.j2", "location /u/{{ svc.user }}/ {\n}\n", True),
        ("doc.py", 'def f():\n    """Served at /u/<user>/."""\n', False),
        ("attr.py", "#: served at /u/<user>/\nX = 1\n", False),
        ("emit.py", 'HEADER = "# mounted at /u/<user>/\\n"\n', False),
        ("app2.js", "// mounted at /u/<user>\nexport const x = 'http://host/';\n", False),
        ("tpl.j2", "{# mounted at /u/<user> #}\nlisten 80;\n", False),
    ],
)
def test_the_guard_flags_a_spelling_and_spares_prose(tmp_path, name, content, flagged):
    (tmp_path / name).write_text(content, encoding="utf-8")
    assert bool(_scan(tmp_path)) is flagged
