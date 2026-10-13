"""The VA image under test is read in one module, and nothing else spells it."""

from __future__ import annotations

import ast
import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.va.e2e import _image

REPO_ROOT = Path(__file__).resolve().parents[2]
OWNER = REPO_ROOT / "tests" / "va" / "e2e" / "_image.py"
SCANNED_DIRECTORIES = (REPO_ROOT / "tests" / "va" / "e2e", REPO_ROOT / "tests" / "e2e")
IMAGE_FAMILY = "osprey-va-full"

#: The modules that boot the image or name it; the scan must see every one.
CONSUMERS = frozenset(
    {
        "tests/va/e2e/conftest.py",
        "tests/va/e2e/_guarded_measure.py",
        "tests/va/e2e/test_bench_ioc_live.py",
        "tests/va/e2e/test_live_standin.py",
        "tests/va/e2e/test_model_status_va.py",
        "tests/va/e2e/test_per_target_posture.py",
        "tests/va/e2e/test_planted_journal.py",
        "tests/va/e2e/test_target_switch.py",
        "tests/va/e2e/test_va_put_surface.py",
        "tests/va/e2e/test_mml_trees_boot.py",
        "tests/va/e2e/test_bluesky_lanes.py",
        "tests/va/e2e/test_substrate_parity.py",
        "tests/e2e/test_target_switch_agentic.py",
    }
)


def _docstrings(tree: ast.Module) -> set[int]:
    """The ids of the module, class and function docstring constants in ``tree``."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            found.add(id(body[0].value))
    return found


def own_spellings(source: str) -> list[tuple[int, str]]:
    """Each line of ``source`` that reads the override or spells the image itself.

    The override's name may not appear at all; the image's family name may
    appear only in a docstring, so no string that reaches docker can name it.
    """
    findings = [
        (number, f"names {_image.IMAGE_ENV}")
        for number, line in enumerate(source.splitlines(), start=1)
        if _image.IMAGE_ENV in line
    ]
    tree = ast.parse(source)
    docstrings = _docstrings(tree)
    findings.extend(
        (node.lineno, f"spells {IMAGE_FAMILY!r} in a string")
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and IMAGE_FAMILY in node.value
    )
    return sorted(findings)


def _scanned_modules() -> list[Path]:
    return sorted(
        path
        for directory in SCANNED_DIRECTORIES
        for path in directory.rglob("*.py")
        if path != OWNER
    )


def test_no_module_reads_or_spells_the_image_itself() -> None:
    modules = _scanned_modules()
    seen = {path.relative_to(REPO_ROOT).as_posix() for path in modules}
    assert CONSUMERS <= seen, f"the scan did not see {sorted(CONSUMERS - seen)}"

    offences = [
        f"{path.relative_to(REPO_ROOT).as_posix()}:{line}: {reason}"
        for path in modules
        for line, reason in own_spellings(path.read_text(encoding="utf-8"))
    ]
    assert not offences, (
        "the VA image under test and its override are read in one place; "
        "import it from tests/va/e2e/_image.py:\n" + "\n".join(offences)
    )


_DOCSTRING_ONLY = (
    '"""Boots an ``osprey-va-full`` container."""\n'
    "\n"
    "class Lane:\n"
    '    """Serves from ``osprey-va-full``."""\n'
    "\n"
    "    def boot(self):\n"
    '        """Start one ``osprey-va-full`` container."""\n'
)


@pytest.mark.parametrize(
    ("source", "flagged"),
    [
        ('import os\nIMAGE = os.environ.get("OSPREY_VA_E2E_IMAGE", "x")\n', True),
        ('IMAGE = "osprey-va-full:latest"\n', True),
        ('def run(tag):\n    return f"osprey-va-full{tag}"\n', True),
        (_DOCSTRING_ONLY, False),
    ],
    ids=["environment-read", "tagged-literal", "bare-name-in-f-string", "docstring-only"],
)
def test_the_scan_catches_a_planted_read(source: str, flagged: bool) -> None:
    assert bool(own_spellings(source)) is flagged


@pytest.fixture
def reload_image_after() -> Iterator[None]:
    """Reload the owner once the test's environment patch is undone."""
    yield
    importlib.reload(_image)


@pytest.mark.usefixtures("reload_image_after")
def test_image_follows_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_image.IMAGE_ENV, "side-build:probe")
    assert importlib.reload(_image).IMAGE == "side-build:probe"

    monkeypatch.delenv(_image.IMAGE_ENV)
    assert importlib.reload(_image).IMAGE == _image.DEFAULT_IMAGE
