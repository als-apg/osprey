"""No module in the optimizer distribution imports ``osprey``.

``osprey-optimizer`` runs in its own image, where the framework is not
installed, and the framework imports its base layer, so an import back into
``osprey`` would be both a missing module there and a cycle here. As in the
connectors guard this reads the source rather than running it, so an import
inside a function or a rarely taken branch is caught too. ``osprey_connectors``
and ``osprey_optimizer`` are distributions of their own, not the framework.
"""

from __future__ import annotations

from pathlib import Path

from tests.connectors.test_package_never_imports_osprey import framework_imports

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "packages" / "osprey-optimizer" / "src" / "osprey_optimizer"

# Below the package's module count and above that of any one subpackage, so a
# walk rooted at the wrong directory, or one that finds nothing, fails here.
MODULE_FLOOR = 60


def _package_modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_no_module_in_the_optimizer_package_imports_osprey() -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}: {module}"
        for path in _package_modules()
        for line, module in framework_imports(path.read_text(encoding="utf-8"), str(path))
    ]
    assert not offenders, "the optimizer package imports osprey:\n" + "\n".join(offenders)


def test_the_walk_covers_the_whole_package() -> None:
    modules = _package_modules()
    assert PACKAGE / "__init__.py" in modules, f"no package at {PACKAGE}"
    assert len(modules) >= MODULE_FLOOR, (
        f"walked {len(modules)} modules under {PACKAGE}, expected at least {MODULE_FLOOR}"
    )
