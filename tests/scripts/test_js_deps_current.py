"""``scripts/js_deps_current.py`` says whether ``node_modules`` holds exactly what
``package-lock.json`` pins, and names the first package that differs.

Each case runs the script from a repository copy in ``tmp_path``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"


_LOCK = {
    "name": "fixture",
    "lockfileVersion": 3,
    "packages": {
        "": {"name": "fixture", "devDependencies": {"a": "1.0.0"}},
        "node_modules/a": {"version": "1.0.0", "integrity": "sha512-a", "dev": True},
        "node_modules/a-darwin": {"version": "1.0.0", "dev": True, "optional": True},
        "node_modules/a-linux": {"version": "1.0.0", "dev": True, "optional": True},
    },
}


def _js_tree(root: Path, *, hidden: dict | None) -> Path:
    """A repository copy holding the helper, a lockfile and optionally npm's
    hidden lockfile."""
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPTS / "js_deps_current.py", root / "scripts")
    (root / "package-lock.json").write_text(json.dumps(_LOCK), encoding="utf-8")
    if hidden is not None:
        (root / "node_modules").mkdir(exist_ok=True)
        (root / "node_modules/.package-lock.json").write_text(json.dumps(hidden), encoding="utf-8")
    return root


def _hidden(*, drop: tuple[str, ...] = (), version: dict[str, str] | None = None) -> dict:
    packages = {
        key: dict(value) for key, value in _LOCK["packages"].items() if key and key not in drop
    }
    for key, moved in (version or {}).items():
        packages[key]["version"] = moved
    return {"name": "fixture", "lockfileVersion": 3, "packages": packages}


def _js_current(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / "scripts/js_deps_current.py")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_js_deps_current_when_equal(tmp_path: Path) -> None:
    completed = _js_current(_js_tree(tmp_path, hidden=_hidden()))
    assert completed.returncode == 0, completed.stdout


def test_js_deps_current_with_an_optional_package_left_out(tmp_path: Path) -> None:
    completed = _js_current(_js_tree(tmp_path, hidden=_hidden(drop=("node_modules/a-linux",))))
    assert completed.returncode == 0, completed.stdout


def test_js_deps_current_is_stale_when_a_version_moved(tmp_path: Path) -> None:
    hidden = _hidden(version={"node_modules/a": "0.9.0"})
    completed = _js_current(_js_tree(tmp_path, hidden=hidden))
    assert completed.returncode == 1
    assert completed.stdout.strip().splitlines() == [completed.stdout.strip()]
    assert "node_modules/a" in completed.stdout


def test_js_deps_current_is_stale_when_a_required_package_is_missing(tmp_path: Path) -> None:
    completed = _js_current(_js_tree(tmp_path, hidden=_hidden(drop=("node_modules/a",))))
    assert completed.returncode == 1
    assert "node_modules/a" in completed.stdout


def test_js_deps_current_is_stale_without_the_hidden_lockfile(tmp_path: Path) -> None:
    completed = _js_current(_js_tree(tmp_path, hidden=None))
    assert completed.returncode == 1
    assert "node_modules/.package-lock.json" in completed.stdout


#: A linux glibc platform package as ``package-lock.json`` pins it, and as a
#: linux ``npm ci`` records it in the hidden lockfile: the same install, with
#: ``libc`` left out of the installed copy.
_LINUX_GNU_KEY = "node_modules/@rolldown/binding-linux-x64-gnu"
_LINUX_GNU_LOCKED = {
    "version": "1.2.12",
    "resolved": "https://registry.npmjs.org/@rolldown/binding-linux-x64-gnu/-/binding-linux-x64-gnu-1.2.12.tgz",
    "integrity": "sha512-linux-x64-gnu",
    "cpu": ["x64"],
    "dev": True,
    "libc": ["glibc"],
    "license": "MIT",
    "optional": True,
    "os": ["linux"],
    "engines": {"node": "^20.19.0 || >=22.12.0"},
}
_LINUX_GNU_INSTALLED = {key: value for key, value in _LINUX_GNU_LOCKED.items() if key != "libc"}


def _linux_tree(root: Path, installed: dict) -> Path:
    tree = _js_tree(root, hidden=None)
    lock = json.loads((tree / "package-lock.json").read_text(encoding="utf-8"))
    lock["packages"][_LINUX_GNU_KEY] = _LINUX_GNU_LOCKED
    (tree / "package-lock.json").write_text(json.dumps(lock), encoding="utf-8")
    hidden = _hidden()
    hidden["packages"][_LINUX_GNU_KEY] = installed
    (tree / "node_modules").mkdir()
    (tree / "node_modules/.package-lock.json").write_text(json.dumps(hidden), encoding="utf-8")
    return tree


def test_js_deps_current_with_a_linux_platform_package_recorded_without_libc(
    tmp_path: Path,
) -> None:
    completed = _js_current(_linux_tree(tmp_path, _LINUX_GNU_INSTALLED))
    assert completed.returncode == 0, completed.stdout


def test_js_deps_current_is_stale_when_a_linux_platform_package_version_moved(
    tmp_path: Path,
) -> None:
    completed = _js_current(_linux_tree(tmp_path, {**_LINUX_GNU_INSTALLED, "version": "1.2.11"}))
    assert completed.returncode == 1
    assert _LINUX_GNU_KEY in completed.stdout


def test_js_deps_current_is_stale_when_a_linux_platform_package_integrity_moved(
    tmp_path: Path,
) -> None:
    installed = {**_LINUX_GNU_INSTALLED, "integrity": "sha512-other"}
    completed = _js_current(_linux_tree(tmp_path, installed))
    assert completed.returncode == 1
    assert _LINUX_GNU_KEY in completed.stdout


def test_js_deps_current_is_stale_when_a_package_is_installed_but_not_locked(
    tmp_path: Path,
) -> None:
    hidden = _hidden()
    hidden["packages"]["node_modules/stray"] = {"version": "1.0.0", "integrity": "sha512-s"}
    completed = _js_current(_js_tree(tmp_path, hidden=hidden))
    assert completed.returncode == 1
    assert "node_modules/stray" in completed.stdout
