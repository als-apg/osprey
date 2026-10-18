"""The gate environment step: every gate script runs on the tree's pins or names
what to install.

``scripts/pinned_claude.py`` finds the pinned Claude Code, ``scripts/js_deps_current.py``
decides whether ``node_modules`` matches the lockfile, and ``scripts/_gate_env.sh``
is the one sourced step that uses both. Each is driven here against stubs in
``tmp_path``; nothing on the host's ``PATH`` or under its ``HOME`` is consulted.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
PIN = _DEFAULT_CLAUDE_CLI_VERSION


def _stub(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _claude(path: Path, version: str) -> Path:
    return _stub(path, f'echo "{version} (Claude Code)"')


def _resolve(tmp_path: Path, *path_dirs: Path) -> subprocess.CompletedProcess[str]:
    env = {
        "HOME": str(tmp_path / "home"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "PATH": os.pathsep.join(str(d) for d in path_dirs),
    }
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "pinned_claude.py")],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def test_resolver_prints_the_pin() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / "pinned_claude.py"), "--pin"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert completed.stdout.strip() == PIN


def test_resolver_prefers_a_path_claude_at_the_pin(tmp_path: Path) -> None:
    stale, pinned, later = tmp_path / "stale", tmp_path / "pinned", tmp_path / "later"
    _claude(stale / "claude", "0.0.1")
    _claude(pinned / "claude", PIN)
    _claude(later / "claude", PIN)
    completed = _resolve(tmp_path, stale, pinned, later)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(pinned)


def test_resolver_shims_the_versions_dir_binary(tmp_path: Path) -> None:
    stale = tmp_path / "stale"
    _claude(stale / "claude", "0.0.1")
    versioned = _claude(tmp_path / "home/.local/share/claude/versions" / PIN, PIN)
    completed = _resolve(tmp_path, stale)
    assert completed.returncode == 0, completed.stderr
    shim_dir = tmp_path / "cache/osprey/claude" / PIN
    assert completed.stdout.strip() == str(shim_dir)
    shim = shim_dir / "claude"
    assert shim.is_symlink() and shim.resolve() == versioned.resolve()
    # A second run replaces the link in place.
    again = _resolve(tmp_path, stale)
    assert again.returncode == 0, again.stderr
    assert sorted(p.name for p in shim_dir.iterdir()) == ["claude"]


def test_resolver_refuses_with_the_install_command(tmp_path: Path) -> None:
    stale = tmp_path / "stale"
    _claude(stale / "claude", "0.0.1")
    completed = _resolve(tmp_path, stale)
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert f"npm install -g @anthropic-ai/claude-code@{PIN}" in completed.stderr


def test_resolver_reads_a_probe_timeout_as_a_refusal_naming_the_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = importlib.util.spec_from_file_location("pinned_claude", SCRIPTS / "pinned_claude.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    slow = _claude(tmp_path / "slow/claude", PIN)

    def _timeout(executable: str) -> str:
        raise subprocess.TimeoutExpired([executable, "--version"], 15)

    monkeypatch.setattr(module, "_probe_cli_version", _timeout)
    monkeypatch.setenv("PATH", str(slow.parent))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert module.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(slow) in captured.err


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
