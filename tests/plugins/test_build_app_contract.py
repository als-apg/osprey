"""The build-app skill's review record against the app contract it describes.

``plugins/osprey/skills/build-app/SKILL.md`` states facts about the app contract:
manifest keys, rule names, the SDK, the ``osprey app`` command, the scaffold. The
skill's ``contract-fingerprint.json`` records a hash of each file those facts come
from, taken when the skill was last read against them. A change to one of those
files without a refreshed record means the skill may now say something false, and
this module fails until someone has looked.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from osprey.apps.contract import APP_CONTRACT, SURFACE_SOURCES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SKILL = "plugins/osprey/skills/build-app"
_SCRIPT = _REPO_ROOT / _SKILL / "scripts" / "app_contract.py"
_PACKAGE = _REPO_ROOT / "src" / "osprey"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_app_contract", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # No __pycache__ beside the script: the plugin tree ships as it stands on disk.
    writes_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = writes_bytecode
    return module


@pytest.fixture(scope="module")
def contract() -> ModuleType:
    return _load_script()


def test_skill_was_reviewed_against_the_current_contract(contract: ModuleType) -> None:
    differing, _record = contract.drift(_PACKAGE)
    assert not differing, (
        "The app contract changed since the build-app skill was last reviewed against it:\n  "
        + "\n  ".join(differing)
        + "\n\nDoes this change break apps? Then raise `APP_CONTRACT` and add a history row "
        "(docs/source/reference/contracts/app-contract.rst).\n"
        f"\n1. Read {_SKILL}/SKILL.md against the change (the script's WATCHED table "
        "names the sections that rest on each file) and correct what is now wrong.\n"
        "2. Record the review:\n"
        f"     uv run python {_SKILL}/scripts/app_contract.py --refresh\n"
        "3. Commit the refreshed contract-fingerprint.json with the change. A change under "
        "plugins/ also needs the plugin version bump and a changelog fragment."
    )


def test_every_watched_file_ships_in_the_package(contract: ModuleType) -> None:
    """A watched path outside ``src/osprey`` would be absent from an installed OSPREY."""
    missing = [
        entry
        for entry, _defines, _sections in contract.WATCHED
        if not (_PACKAGE / entry).exists() or (_PACKAGE / entry).is_dir() != entry.endswith("/")
    ]
    assert not missing, f"WATCHED names paths that are not in src/osprey: {missing}"
    recorded = json.loads(contract.FINGERPRINT.read_text(encoding="utf-8"))["files"]
    assert all((_PACKAGE / path).is_file() for path in recorded)


def test_drift_names_a_changed_added_and_removed_file(
    contract: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same package laid out as an install, with three watched files touched."""
    package = tmp_path / "site-packages" / "osprey"
    for entry, _defines, _sections in contract.WATCHED:
        source, target = _PACKAGE / entry, package / entry
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, target)
    (package / "__init__.py").touch()
    assert contract.drift(package)[0] == []

    with (package / "apps" / "manifest.py").open("a", encoding="utf-8") as handle:
        handle.write("\nNEW_KEY = 1\n")
    (package / "apps" / "scaffold-app" / "pyproject.toml").unlink()
    (package / "apps" / "scaffold-app" / "panel" / "extra.js").write_text("", encoding="utf-8")

    assert contract.drift(package)[0] == [
        "apps/manifest.py  (changed)",
        "apps/scaffold-app/panel/extra.js  (not in the reviewed set)",
        "apps/scaffold-app/pyproject.toml  (missing here)",
    ]
    assert contract.main(["--root", str(package), "--strict"]) == 1
    report = capsys.readouterr().out
    assert "contract CHANGED since review in 3 file(s)" in report
    assert "NEW_KEY = 1" not in report  # the summary lists schema constants it knows
    assert "== app.yml schema" in report


def test_no_osprey_is_reported_not_raised(
    contract: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert contract.main(["--root", str(tmp_path)]) == 2
    assert "no OSPREY with an app contract found" in capsys.readouterr().out


def _module_path(module: str) -> str | None:
    """The package-relative file of an ``osprey.*`` *module*, or ``None`` if it is not one."""
    if module != "osprey" and not module.startswith("osprey."):
        return None
    parts = module.split(".")[1:]
    candidates = [
        "/".join(parts) + ".py" if parts else "__init__.py",
        "/".join([*parts, "__init__.py"]),
    ]
    return next((c for c in candidates if (_PACKAGE / c).is_file()), None)


def _module_level_imports(relative: str) -> set[str]:
    """The ``osprey.*`` modules *relative*'s module-level imports name, as files."""
    tree = ast.parse((_PACKAGE / relative).read_text(encoding="utf-8"))
    package = ["osprey", *Path(relative).parent.parts]
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            module = ".".join([*base, *([node.module] if node.module else [])])
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return {path for name in names if (path := _module_path(name)) is not None}


def _watched(contract: ModuleType, relative: str) -> bool:
    return any(
        relative == entry or (entry.endswith("/") and relative.startswith(entry))
        for entry, _defines, _sections in contract.WATCHED
    )


def test_every_module_the_app_check_judges_with_is_watched_or_named_not_contract(
    contract: ModuleType,
) -> None:
    """A rule module the check reaches is either one the skill rests on or named as not."""
    not_contract = {path for path, _reason in contract.NOT_CONTRACT}
    reached: set[str] = set()
    walk = ["cli/app_cmd.py"]
    while walk:
        for module in _module_level_imports(walk.pop()):
            if module in reached:
                continue
            reached.add(module)
            if _watched(contract, module):
                walk.append(module)
    unnamed = sorted(m for m in reached if not _watched(contract, m) and m not in not_contract)
    assert not unnamed, (
        "osprey app check reaches modules app_contract.py neither watches nor names as "
        f"not contract: {unnamed}"
    )
    stale = sorted(not_contract - reached)
    assert not stale, f"NOT_CONTRACT names modules the check no longer reaches: {stale}"


def test_every_surface_source_in_the_package_is_watched(contract: ModuleType) -> None:
    """A file the app contract's surface is read from is one the skill rests on too."""
    prefix = "src/osprey/"
    sources = {
        path.removeprefix(prefix)
        for paths in SURFACE_SOURCES.values()
        for path in paths
        if path.startswith(prefix)
    }
    unwatched = sorted(path for path in sources if not _watched(contract, path))
    assert not unwatched, f"SURFACE_SOURCES files app_contract.py does not watch: {unwatched}"


def test_the_printed_contract_number_is_the_one_osprey_implements(
    contract: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert contract.main(["--root", str(_PACKAGE), "--drift-only"]) == 0
    printed = re.findall(r"^contract: (\d+)\b", capsys.readouterr().out, re.MULTILINE)
    assert printed == [str(APP_CONTRACT)]

    assert contract.main(["--root", str(_PACKAGE), "--summary-only"]) == 0
    printed = re.findall(r"^contract: (\d+)\b", capsys.readouterr().out, re.MULTILINE)
    assert printed == [str(APP_CONTRACT)]


def test_an_osprey_before_the_contract_number_prints_none(
    contract: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = tmp_path / "osprey"
    (package / "apps").mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "apps" / "manifest.py").touch()
    assert contract.main(["--root", str(package), "--drift-only"]) == 0
    out = capsys.readouterr().out
    assert "contract: none (this OSPREY predates the contract number)" in out


def test_skill_states_no_contract_number_and_no_bump_rule() -> None:
    """The number is printed by the script and the bump rule lives on the reference page."""
    skill = (_REPO_ROOT / _SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert not re.search(r"contract:\s*\d", skill), "SKILL.md states a literal contract number"
    assert "raises the number by one" not in skill, "SKILL.md restates the bump rule"
