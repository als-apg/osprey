"""Tests for osprey.interfaces.design_system.generator.build.

Hermetic: every test writes its own tiny token tree under ``tmp_path``
(this task's file ownership doesn't include fixtures/**) rather than
depending on the real ``tokens/`` tree, and every artifact is written
under ``tmp_path`` too — never the real ``static/`` directory (task #9
owns committing those generated files).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.interfaces.design_system.generator.build import (
    CSS_RELATIVE_PATH,
    THEME_BOOT_JS_RELATIVE_PATH,
    TOKENS_JS_RELATIVE_PATH,
    BuildError,
    build_artifacts,
    check_artifacts,
    main,
    write_artifacts,
)
from osprey.interfaces.design_system.generator.emit_css import emit_css
from osprey.interfaces.design_system.generator.emit_js import render_theme_boot_js, render_tokens_js
from osprey.interfaces.design_system.generator.model import load_token_tree


def _write_minimal_valid_tokens(tokens_dir: Path) -> None:
    """Write a tiny, complete, WCAG-passing tokens/ tree to tokens_dir."""
    (tokens_dir / "themes").mkdir(parents=True)
    (tokens_dir / "interfaces").mkdir(parents=True)

    (tokens_dir / "core.json").write_text(
        json.dumps(
            {
                "color": {
                    "teal": {"500": {"$value": "#14b8a6", "$type": "color"}},
                    "slate": {
                        "50": {"$value": "#f8fafc", "$type": "color"},
                        "900": {"$value": "#0a0f1a", "$type": "color"},
                    },
                },
                "font": {
                    "display": {"$value": "'Outfit', sans-serif", "$type": "fontFamily"},
                    "mono": {"$value": "'JetBrains Mono', monospace", "$type": "fontFamily"},
                },
            }
        ),
        encoding="utf-8",
    )

    dark = {
        "$extensions": {"id": "dark", "label": "Dark", "mode": "dark"},
        "bg": {"primary": {"$value": "{color.slate.900}", "$type": "color"}},
        "text": {"primary": {"$value": "#ffffff", "$type": "color"}},
        "accent": {"base": {"$value": "{color.teal.500}", "$type": "color"}},
        "terminal": {"cursor": {"$value": "{color.teal.500}", "$type": "color"}},
    }
    light = {
        "$extensions": {"id": "light", "label": "Light", "mode": "light"},
        "bg": {"primary": {"$value": "{color.slate.50}", "$type": "color"}},
        "text": {"primary": {"$value": "#000000", "$type": "color"}},
        "accent": {"base": {"$value": "#065f5c", "$type": "color"}},
        "terminal": {"cursor": {"$value": "{color.teal.500}", "$type": "color"}},
    }
    (tokens_dir / "themes" / "dark.json").write_text(json.dumps(dark), encoding="utf-8")
    (tokens_dir / "themes" / "light.json").write_text(json.dumps(light), encoding="utf-8")

    demo = {
        "dark": {"wt-crt": {"opacity": {"$value": "1", "$type": "number"}}},
        "light": {"wt-crt": {"opacity": {"$value": "0", "$type": "number"}}},
    }
    (tokens_dir / "interfaces" / "demo.json").write_text(json.dumps(demo), encoding="utf-8")


def _write_invalid_tokens(tokens_dir: Path) -> None:
    """Write a tokens/ tree with a dangling alias AND a theme-completeness gap."""
    (tokens_dir / "themes").mkdir(parents=True)

    (tokens_dir / "core.json").write_text(
        json.dumps({"color": {"slate": {"900": {"$value": "#0a0f1a", "$type": "color"}}}}),
        encoding="utf-8",
    )
    dark = {
        "$extensions": {"id": "dark", "label": "Dark", "mode": "dark"},
        "bg": {"primary": {"$value": "{color.slate.900}", "$type": "color"}},
        "accent": {"base": {"$value": "{color.nonexistent}", "$type": "color"}},
    }
    light = {
        "$extensions": {"id": "light", "label": "Light", "mode": "light"},
        "bg": {"primary": {"$value": "#ffffff", "$type": "color"}},
        # Missing "accent.base" entirely -> theme-completeness error too.
    }
    (tokens_dir / "themes" / "dark.json").write_text(json.dumps(dark), encoding="utf-8")
    (tokens_dir / "themes" / "light.json").write_text(json.dumps(light), encoding="utf-8")


# --- build_artifacts ---------------------------------------------------------------


def test_build_artifacts_returns_expected_relative_paths(tmp_path: Path) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    _write_minimal_valid_tokens(tokens_dir)

    artifacts = build_artifacts(tokens_dir)

    assert [a.relative_path for a in artifacts] == [
        CSS_RELATIVE_PATH,
        TOKENS_JS_RELATIVE_PATH,
        THEME_BOOT_JS_RELATIVE_PATH,
    ]


def test_build_artifacts_content_matches_direct_emitter_calls(tmp_path: Path) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    _write_minimal_valid_tokens(tokens_dir)

    artifacts = build_artifacts(tokens_dir)
    tree = load_token_tree(tokens_dir)

    by_path = {a.relative_path: a.content for a in artifacts}
    assert by_path[CSS_RELATIVE_PATH] == emit_css(tree)
    assert by_path[TOKENS_JS_RELATIVE_PATH] == render_tokens_js(tree)
    assert by_path[THEME_BOOT_JS_RELATIVE_PATH] == render_theme_boot_js(tree)


def test_build_artifacts_raises_on_missing_core_json(tmp_path: Path) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()

    with pytest.raises(BuildError, match="failed to load token sources"):
        build_artifacts(tokens_dir)


def test_build_artifacts_raises_with_every_validation_error(tmp_path: Path) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    _write_invalid_tokens(tokens_dir)

    with pytest.raises(BuildError) as excinfo:
        build_artifacts(tokens_dir)

    message = str(excinfo.value)
    # Both the dangling alias AND the theme-completeness gap must be
    # reported together, not just the first one found.
    assert "dangling" in message.lower() or "does not resolve" in message.lower()
    assert "accent.base" in message
    assert "validation error(s)" in message


# --- write_artifacts ----------------------------------------------------------------


def test_write_artifacts_creates_directories_and_files(tmp_path: Path) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    _write_minimal_valid_tokens(tokens_dir)
    static_dir = tmp_path / "static"

    artifacts = build_artifacts(tokens_dir)
    write_artifacts(artifacts, static_dir)

    for artifact in artifacts:
        target = static_dir / artifact.relative_path
        assert target.is_file()
        assert target.read_text(encoding="utf-8") == artifact.content


def test_write_artifacts_overwrites_existing_content(tmp_path: Path) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    _write_minimal_valid_tokens(tokens_dir)
    static_dir = tmp_path / "static"
    (static_dir / "css").mkdir(parents=True)
    (static_dir / CSS_RELATIVE_PATH).write_text("stale content", encoding="utf-8")

    artifacts = build_artifacts(tokens_dir)
    write_artifacts(artifacts, static_dir)

    assert (static_dir / CSS_RELATIVE_PATH).read_text(encoding="utf-8") != "stale content"


# --- check_artifacts -----------------------------------------------------------------


def test_check_artifacts_empty_when_disk_matches(tmp_path: Path) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    _write_minimal_valid_tokens(tokens_dir)
    static_dir = tmp_path / "static"

    artifacts = build_artifacts(tokens_dir)
    write_artifacts(artifacts, static_dir)

    assert check_artifacts(artifacts, static_dir) == []


def test_check_artifacts_reports_missing_files_as_drift(tmp_path: Path) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    _write_minimal_valid_tokens(tokens_dir)
    static_dir = tmp_path / "static"  # never written

    artifacts = build_artifacts(tokens_dir)
    diffs = check_artifacts(artifacts, static_dir)

    assert {diff.relative_path for diff in diffs} == {
        CSS_RELATIVE_PATH,
        TOKENS_JS_RELATIVE_PATH,
        THEME_BOOT_JS_RELATIVE_PATH,
    }
    for diff in diffs:
        assert diff.unified_diff  # non-empty unified diff text


def test_check_artifacts_reports_stale_content_as_drift(tmp_path: Path) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    _write_minimal_valid_tokens(tokens_dir)
    static_dir = tmp_path / "static"

    artifacts = build_artifacts(tokens_dir)
    write_artifacts(artifacts, static_dir)
    (static_dir / CSS_RELATIVE_PATH).write_text("hand-edited\n", encoding="utf-8")

    diffs = check_artifacts(artifacts, static_dir)

    assert [diff.relative_path for diff in diffs] == [CSS_RELATIVE_PATH]
    assert "hand-edited" in diffs[0].unified_diff


# --- main() / CLI ---------------------------------------------------------------------


def test_main_build_mode_writes_files_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    _write_minimal_valid_tokens(tokens_dir)
    static_dir = tmp_path / "static"

    exit_code = main([], tokens_dir=tokens_dir, static_dir=static_dir)

    assert exit_code == 0
    assert (static_dir / CSS_RELATIVE_PATH).is_file()
    assert (static_dir / TOKENS_JS_RELATIVE_PATH).is_file()
    assert (static_dir / THEME_BOOT_JS_RELATIVE_PATH).is_file()
    out = capsys.readouterr().out
    assert "wrote" in out


def test_main_check_mode_clean_returns_zero_and_does_not_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    _write_minimal_valid_tokens(tokens_dir)
    static_dir = tmp_path / "static"
    write_artifacts(build_artifacts(tokens_dir), static_dir)
    before = (static_dir / CSS_RELATIVE_PATH).read_text(encoding="utf-8")

    exit_code = main(["--check"], tokens_dir=tokens_dir, static_dir=static_dir)

    assert exit_code == 0
    assert (static_dir / CSS_RELATIVE_PATH).read_text(encoding="utf-8") == before
    assert "up to date" in capsys.readouterr().out


def test_main_check_mode_drift_returns_one_and_does_not_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    _write_minimal_valid_tokens(tokens_dir)
    static_dir = tmp_path / "static"  # deliberately never written

    exit_code = main(["--check"], tokens_dir=tokens_dir, static_dir=static_dir)

    assert exit_code == 1
    assert not (static_dir / CSS_RELATIVE_PATH).exists()
    err = capsys.readouterr().err
    assert "drift" in err


def test_main_validation_failure_returns_one_and_prints_every_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    _write_invalid_tokens(tokens_dir)
    static_dir = tmp_path / "static"

    exit_code = main([], tokens_dir=tokens_dir, static_dir=static_dir)

    assert exit_code == 1
    assert not static_dir.exists()
    err = capsys.readouterr().err
    assert "accent.base" in err
    assert "validation error(s)" in err


def test_main_validation_failure_also_fails_under_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    _write_invalid_tokens(tokens_dir)
    static_dir = tmp_path / "static"

    exit_code = main(["--check"], tokens_dir=tokens_dir, static_dir=static_dir)

    assert exit_code == 1


# --- `python -m ...` entrypoint smoke test --------------------------------------------


def test_module_is_runnable_as_python_dash_m() -> None:
    # A light smoke test of the __main__ guard + argparse wiring against
    # the real repo tokens/static dirs. --check never writes, so this is
    # safe regardless of whether the real static/ artifacts exist yet
    # (that's task #9's freshness gate, not this test's concern) — we only
    # assert the module runs cleanly to completion (exit 0 or 1) rather
    # than crashing (any other code, or a traceback on stderr).
    result = subprocess.run(
        [sys.executable, "-m", "osprey.interfaces.design_system.generator.build", "--check"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode in (0, 1)
    assert "Traceback" not in result.stderr
