"""The connectors package's public names check: a listed name has a user outside its module.

The check is the ``connectors-public-names`` record of the package's own checks
module. Each fixture is a small tree under ``tmp_path`` with the repository's
layout: the package's source under ``packages/osprey-connectors/src`` and the
framework under ``src``. The last test holds the real tree to it.
"""

from __future__ import annotations

from pathlib import Path

from tests.scripts._check_script import check

REPO_ROOT = Path(__file__).resolve().parents[2]

names = check.DECLARED.modules["packages/osprey-connectors/repo_checks.py"]

PACKAGE = Path(names.PACKAGE_SRC) / "osprey_connectors"


def _tree(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _strays(root: Path, extension_points: dict[str, str] | None = None) -> list[str]:
    return names.stray_exports(root, extension_points=extension_points or {})


def test_an_unused_listed_name_fails_with_its_line(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        {
            f"{PACKAGE}/__init__.py": "",
            f"{PACKAGE}/a.py": '__all__ = [\n    "used",\n    "unused",\n]\nused = unused = 1\n',
            "src/osprey/x.py": "from osprey_connectors.a import used\n",
        },
    )
    [line] = _strays(root)
    assert line.startswith(f"{PACKAGE.as_posix()}/a.py:3: osprey_connectors.a.unused ")


def test_a_used_or_declared_name_passes(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        {
            f"{PACKAGE}/__init__.py": "",
            f"{PACKAGE}/a.py": '__all__ = ["absolute", "relative", "aliased", "declared"]\n',
            f"{PACKAGE}/b.py": "from .a import relative\n",
            "src/osprey/x.py": (
                "import osprey_connectors.a as conn\n"
                "from osprey_connectors.a import absolute\n"
                "conn.aliased\n"
            ),
        },
    )
    [line] = _strays(root)
    assert "osprey_connectors.a.declared " in line
    assert _strays(root, {"osprey_connectors.a.declared": "a hook for other packages"}) == []


def test_a_test_or_docstring_use_does_not_count(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        {
            f"{PACKAGE}/__init__.py": "",
            f"{PACKAGE}/a.py": '__all__ = ["only_tested"]\n',
            "src/osprey/x.py": '"""Calls osprey_connectors.a.only_tested."""\n',
            "tests/test_a.py": "from osprey_connectors.a import only_tested\n",
        },
    )
    [line] = _strays(root)
    assert "osprey_connectors.a.only_tested " in line


def test_a_private_module_list_is_not_judged(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        {
            f"{PACKAGE}/__init__.py": "",
            f"{PACKAGE}/_impl.py": '__all__ = ["internal"]\n',
            f"{PACKAGE}/_inner/mod.py": '__all__ = ["internal"]\n',
        },
    )
    assert _strays(root) == []


def test_a_stale_extension_point_fails(tmp_path: Path) -> None:
    root = _tree(tmp_path, {f"{PACKAGE}/__init__.py": "", f"{PACKAGE}/a.py": "__all__ = []\n"})
    [line] = _strays(root, {"osprey_connectors.a.gone": "was a hook"})
    assert "osprey_connectors.a.gone is in EXTENSION_POINTS and no __all__ lists it" in line


def test_the_tree_passes() -> None:
    assert names.stray_exports(REPO_ROOT) == []
