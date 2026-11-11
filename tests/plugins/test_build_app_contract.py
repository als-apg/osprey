"""The build-app skill's review record against the app contract it describes.

``plugins/osprey/skills/build-app/SKILL.md`` states facts about the app contract:
manifest keys, rule names, the SDK, the ``osprey app`` command, the scaffold. The
skill's ``contract-fingerprint.json`` records a hash of each file those facts come
from, taken when the skill was last read against them. A change to one of those
files without a refreshed record means the skill may now say something false, and
this module fails until someone has looked.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

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
        + f"\n\n1. Read {_SKILL}/SKILL.md against the change (the script's WATCHED table "
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
