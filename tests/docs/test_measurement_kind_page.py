"""The register-a-measurement-kind how-to shows the fixture kind and quotes the build's refusals.

Every ``literalinclude`` on the page shows a file of the deployment package that
``tests/facility/test_fixture_kind_deployment.py`` builds end to end, and each
``:pyobject:`` or ``:start-at:`` it selects is in that file. Every refusal in the
page's table is spelled by the code that raises it, or by the fixture test that
pins its full text: each fragment between the placeholders (``<word>`` and
``…``) appears there.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "docs/source/how-to/register-a-measurement-kind.rst"
PACKAGE_DIR = REPO_ROOT / "tests/facility/fixtures/fixture_kind"

#: Where a refusal on the page is spelled: the code that raises it and the test pinning it.
REFUSAL_SOURCES = (
    REPO_ROOT / "src/osprey/facility/kinds.py",
    REPO_ROOT / "src/osprey/facility/validate.py",
    REPO_ROOT / "src/osprey/deployment/wheel_build.py",
    PACKAGE_DIR / "fixture_kind_deployment/__init__.py",
    REPO_ROOT / "tests/facility/test_fixture_kind_deployment.py",
)

#: The fixture test's f-string names, as the page spells them.
FIXTURE_NAMES = {"{KIND}": "fixture_kind", "{DISTRIBUTION}": "fixture-kind-deployment"}

_INCLUDE = re.compile(r"^\.\. literalinclude:: (\S+)\n((?:   :\S+:.*\n)*)", re.MULTILINE)


def _includes() -> list[tuple[Path, dict[str, str]]]:
    found = []
    for target, options in _INCLUDE.findall(PAGE.read_text(encoding="utf-8")):
        opts = dict(re.findall(r"^   :(\S+):[ \t]*(.*)$", options, re.MULTILINE))
        found.append(((PAGE.parent / target).resolve(), opts))
    return found


def _source_text() -> str:
    """The sources, with adjacent string literals joined and the fixture names filled in."""
    text = "\n".join(path.read_text(encoding="utf-8") for path in REFUSAL_SOURCES)
    text = re.sub(r"\"\s*\n\s*f?\"", "", text)
    for name, value in FIXTURE_NAMES.items():
        text = text.replace(name, value)
    return text


def _refusals() -> list[str]:
    table = PAGE.read_text(encoding="utf-8").split("What a refusal means", 1)[1]
    cells = re.findall(r"^   \* - ``(.+?)``\n", re.sub(r"\n {7}", " ", table), re.MULTILINE)
    return cells


def test_the_page_includes_only_the_fixture_kinds_files() -> None:
    includes = _includes()

    assert includes
    for path, _options in includes:
        assert path.is_file(), path
        assert path.is_relative_to(PACKAGE_DIR), path


def test_each_selected_object_is_in_its_file() -> None:
    for path, options in _includes():
        text = path.read_text(encoding="utf-8")
        if "pyobject" in options:
            names = {
                node.name
                for node in ast.parse(text).body
                if isinstance(node, ast.ClassDef | ast.FunctionDef)
            }
            assert options["pyobject"] in names, (path, options)
        if "start-at" in options:
            assert options["start-at"] in text, (path, options)


def test_the_table_lists_the_refusals() -> None:
    assert len(_refusals()) == 10


@pytest.mark.parametrize("refusal", _refusals())
def test_each_refusal_is_spelled_by_the_code(refusal: str) -> None:
    source = _source_text()
    fragments = [part for part in re.split(r"<[a-z]+>|…", refusal) if part.strip()]

    assert fragments
    missing = [part for part in fragments if part not in source]
    assert missing == [], refusal
