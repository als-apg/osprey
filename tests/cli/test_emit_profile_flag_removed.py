"""Removal gate: ``osprey build --emit-profile`` is gone, with no alias.

The verb that materializes a deployment repo is ``osprey init``. The old flag was
removed outright rather than aliased, so two things must hold and keep holding:

* the CLI rejects ``--emit-profile`` as an unknown option, and
* no shipped source or documentation page still tells someone to run it.

The sweep covers ``src/osprey/`` (including the bundled skills and app README
templates, which are shipped instructions) and ``docs/source/``. ``CHANGELOG.md``
is out of scope by design — it records the removal and the historical entries
that introduced the flag, and rewriting released history to satisfy a grep would
be worse than the grep failing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build

# Spelled once. This file is not itself swept — the roots below are the shipped
# package and the docs tree, not tests/ — so the literal is safe here.
REMOVED_FLAG = "--emit-profile"

ROOTS = ("src/osprey", "docs/source")
SCAN_SUFFIXES = (
    ".py",
    ".md",
    ".rst",
    ".txt",
    ".j2",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".sh",
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _shipped_sources() -> list[Path]:
    files: list[Path] = []
    for root in ROOTS:
        base = _REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def test_no_live_reference_to_the_removed_flag() -> None:
    offenders: list[str] = []
    for path in _shipped_sources():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if REMOVED_FLAG not in content:
            continue
        hits = [
            f"{i}: {line.strip()}"
            for i, line in enumerate(content.splitlines(), start=1)
            if REMOVED_FLAG in line
        ][:3]
        offenders.append(f"{path.relative_to(_REPO_ROOT)}\n  " + "\n  ".join(hits))

    assert not offenders, (
        f"{REMOVED_FLAG} was removed — use `osprey init DIR --preset X`. "
        "Stale references remain:\n" + "\n".join(offenders)
    )


def test_the_sweep_would_catch_a_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard is only worth having if it fails on a reintroduced reference."""
    fake_root = tmp_path / "repo"
    (fake_root / "src" / "osprey").mkdir(parents=True)
    (fake_root / "src" / "osprey" / "regress.md").write_text(
        f"run osprey build {REMOVED_FLAG} p --preset hello-world\n", encoding="utf-8"
    )
    monkeypatch.setattr(f"{__name__}._REPO_ROOT", fake_root, raising=True)

    with pytest.raises(AssertionError, match="Stale references remain"):
        test_no_live_reference_to_the_removed_flag()


def test_cli_rejects_the_removed_flag(tmp_path: Path) -> None:
    """No silent alias: the option does not exist on ``osprey build``."""
    result = CliRunner().invoke(
        build, [REMOVED_FLAG, str(tmp_path / "p"), "--preset", "hello-world"]
    )

    assert result.exit_code == 2, result.output
    assert "No such option" in result.output


def test_build_help_does_not_mention_the_removed_flag() -> None:
    result = CliRunner().invoke(build, ["--help"])

    assert result.exit_code == 0, result.output
    assert REMOVED_FLAG not in result.output
