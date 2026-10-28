"""``scripts/pinned_claude.py`` finds the Claude Code the tree pins, or refuses
with the install command, and is the one file that reads the pin.

The resolver is driven against stub ``claude`` executables in ``tmp_path``;
nothing on the host's ``PATH`` or under its ``HOME`` is consulted.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
PIN = _DEFAULT_CLAUDE_CLI_VERSION


def _script_source(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


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


#: Every file that provisions or verifies the pinned CLI outside the resolver.
_PIN_READERS = (
    "scripts/repo_checks.py",
    ".github/actions/install-pinned-claude-code/action.yml",
)


def _restated_pin(source: str) -> list[str]:
    found = []
    if PIN in source:
        found.append(f"the literal {PIN}")
    if re.search(r"claude-code@[0-9]", source):
        found.append("a claude-code@<version> spelling")
    if "_DEFAULT_CLAUDE_CLI_VERSION" in source:
        found.append("the pin constant")
    return found


def _assert_no_restated_pin(path: str, source: str) -> None:
    assert _restated_pin(source) == [], f"{path} restates the pin"


@pytest.mark.parametrize("path", _PIN_READERS)
def test_no_gate_file_restates_the_pin(path: str) -> None:
    """``scripts/pinned_claude.py`` is the one reader of the pin; every other file
    asks it."""
    _assert_no_restated_pin(path, _script_source(path))


@pytest.mark.parametrize("path", _PIN_READERS)
def test_no_gate_file_restates_the_pin__mutation_plants_the_literal(path: str) -> None:
    planted = _script_source(path) + f"\n# claude {PIN}\n"
    with pytest.raises(AssertionError):
        _assert_no_restated_pin(path, planted)
