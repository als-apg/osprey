"""The upstream image catalogue is the one home of every image OSPREY pulls.

Every default reference to an image OSPREY pulls from an upstream lives in
:data:`~osprey.deployment.upstream_images.UPSTREAM_IMAGES`, and every reader
moves it onto ``images.base_registry`` through
:func:`~osprey.deployment.upstream_images.upstream_image`. The guard here fails
on an image reference spelled anywhere else in the deployment and agent-record
packages or in the shipped templates; Dockerfile ``FROM`` lines belong to the
recipe guard in ``test_service_dockerfiles.py``.
"""

import ast
import pathlib
import re

import osprey

_SRC = pathlib.Path(osprey.__file__).parent
_CATALOGUE = _SRC / "deployment" / "upstream_images.py"
_PYTHON_SCOPES = (_SRC / "deployment", _SRC / "agent_record")
_TEMPLATES = _SRC / "templates"

#: An image reference: an optional registry host (with port), a lowercase path
#: and a tag.
_REFERENCE = re.compile(
    r"(?:[a-z0-9.-]+(?::[0-9]+)?/)?[a-z0-9]+(?:[._/-][a-z0-9]+)*:[A-Za-z0-9_][A-Za-z0-9._-]*"
)
#: A ``uid:gid`` pair, which the grammar above also matches.
_UID_PAIR = re.compile(r"\d+:\d+")
#: A host and a port of four or five digits, which the grammar above also matches.
_HOST_PORT = re.compile(r"[\w.-]+:\d{4,5}")

_JINJA_EXPR = re.compile(r"\{\{(.*?)\}\}")
_QUOTED = re.compile(r"""'([^']*)'|"([^"]*)\"""")
_BARE_VALUE = re.compile(r"^\s*(?:-\s+name|image)\s*:\s*['\"]?([^'\"\s]+)['\"]?\s*$")
_JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.DOTALL)


def _is_reference(text: str) -> bool:
    return bool(
        _REFERENCE.fullmatch(text)
        and not _UID_PAIR.fullmatch(text)
        and not _HOST_PORT.fullmatch(text)
    )


def _docstring_nodes(tree: ast.AST) -> set[int]:
    owners = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, owners) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


def _python_hits(path: pathlib.Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_nodes(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and _is_reference(node.value)
    ]


def _blank_jinja_comments(text: str) -> str:
    return _JINJA_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def _template_hits(path: pathlib.Path) -> list[tuple[int, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    hits = []
    for lineno, line in enumerate(_blank_jinja_comments(text).splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        for expr in _JINJA_EXPR.findall(line):
            for single, double in _QUOTED.findall(expr):
                value = single or double
                if _is_reference(value):
                    hits.append((lineno, value))
        bare = _BARE_VALUE.match(line)
        if bare and _is_reference(bare.group(1)):
            hits.append((lineno, bare.group(1)))
    return hits


def _hits_outside_the_catalogue() -> list[str]:
    found = []
    for scope in _PYTHON_SCOPES:
        for path in sorted(scope.rglob("*.py")):
            if path == _CATALOGUE:
                continue
            for lineno, value in _python_hits(path):
                found.append(f"{path.relative_to(_SRC)}:{lineno}: {value}")
    for path in sorted(_TEMPLATES.rglob("*")):
        if not path.is_file() or path.name.startswith("Dockerfile"):
            continue
        for lineno, value in _template_hits(path):
            found.append(f"{path.relative_to(_SRC)}:{lineno}: {value}")
    return found


def test_no_image_literal_outside_the_catalogue():
    assert _hits_outside_the_catalogue() == []


def test_the_catalogue_itself_is_seen_by_the_scan():
    assert _python_hits(_CATALOGUE)


def test_every_entry_mirrors_or_names_another_registry():
    from osprey.deployment.upstream_images import UPSTREAM_IMAGES, upstream_image

    for name, reference in UPSTREAM_IMAGES.items():
        assert upstream_image(name, "") == reference
        mirrored = upstream_image(name, "m")
        if name == "tiled":
            assert mirrored == reference
        else:
            assert mirrored.startswith("m/"), name
