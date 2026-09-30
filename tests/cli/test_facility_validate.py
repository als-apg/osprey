"""``osprey facility validate``: every facility check and every view, nothing written.

The verb builds the facility file from ``data/facility/``, renders the repo's
main profile into a temporary directory and hands that render to
``render_facility_outputs``, then discards it. A clean repo exits 0 and prints
nothing; a failing stage prints every one of its lines, sorted, to stderr and
exits 1. The repo's tree, ``build/`` included, is byte-unchanged either way.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.init_cmd import init
from osprey.cli.main import cli
from osprey.facility.render import FACILITY_FILE
from osprey.utils.workspace import BUILD_DIR_NAME

pytestmark = pytest.mark.slow

#: A demo channel with a ``unit`` its authored layer states as ``mm``.
CHANNEL = "BR:DIAG:BPM:01:POSITION:X"


@pytest.fixture(scope="module")
def initialised(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A control-assistant repo, initialised once for this module and never edited."""
    repo = tmp_path_factory.mktemp("ca") / "demo"
    result = CliRunner().invoke(init, [str(repo), "--preset", "control-assistant", "--no-git"])
    assert result.exit_code == 0, result.output
    return repo


@pytest.fixture
def repo(initialised: Path, tmp_path: Path) -> Iterator[Path]:
    """A copy of the initialised repo this test may edit."""
    copy = tmp_path / initialised.name
    shutil.copytree(initialised, copy, symlinks=True)
    yield copy


def _validate(repo: Path) -> Result:
    return CliRunner().invoke(cli, ["facility", "validate", "--repo", str(repo)])


def _snapshot(root: Path) -> dict[str, bytes | None]:
    """Every path under *root* with its bytes; ``None`` for a directory."""
    return {
        path.relative_to(root).as_posix(): (
            None if path.is_dir() and not path.is_symlink() else path.read_bytes()
        )
        for path in sorted(root.rglob("*"))
    }


def _fixes(repo: Path, *entries: dict[str, Any]) -> None:
    document = {"schema": "osprey.facility.fixes/1", "fixes": list(entries)}
    (repo / "data" / "facility" / "fixes.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def test_a_clean_repo_exits_0_and_prints_nothing(repo: Path) -> None:
    result = _validate(repo)

    assert result.exit_code == 0, result.output
    assert (result.stdout, result.stderr) == ("", "")


def test_a_clean_run_leaves_the_repo_byte_unchanged(repo: Path) -> None:
    before = _snapshot(repo)

    result = _validate(repo)

    assert result.exit_code == 0, result.output
    assert _snapshot(repo) == before
    assert not (repo / BUILD_DIR_NAME).exists()


def test_a_failing_run_leaves_the_repo_byte_unchanged(repo: Path) -> None:
    _fixes(repo, {"op": "drop", "kind": "device", "id": "SR/Q9", "why": "gone"})
    before = _snapshot(repo)

    result = _validate(repo)

    assert result.exit_code == 1, result.output
    assert _snapshot(repo) == before


def test_the_views_are_reached_only_through_render_facility_outputs(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One call, on a render outside the repo that is gone once the verb returns."""
    from osprey.facility import render

    calls: list[tuple[Path, list[Path]]] = []
    real = render.render_facility_outputs

    def spy(render_dir: Path, doc: Any, rendered_config: Any) -> list[Path]:
        written = real(render_dir, doc, rendered_config)
        assert (render_dir / "config.yml").is_file()
        calls.append((render_dir, written))
        return written

    monkeypatch.setattr(render, "render_facility_outputs", spy)

    result = _validate(repo)

    assert result.exit_code == 0, result.output
    ((render_dir, written),) = calls
    assert written == [render_dir / FACILITY_FILE]
    assert repo not in render_dir.parents
    assert not render_dir.exists()


def test_a_stale_fix_prints_the_block_to_paste(repo: Path) -> None:
    imported = repo / "data" / "facility" / "imported" / "mml" / "channels.yaml"
    imported.parent.mkdir(parents=True)
    imported.write_text(yaml.safe_dump([{"id": CHANNEL, "unit": "mm"}]), encoding="utf-8")
    _fixes(
        repo,
        {
            "op": "set",
            "kind": "channel",
            "id": CHANNEL,
            "fields": {"unit": "um"},
            "was": {"unit": {"mml": "m"}},
            "why": "The unit is micrometres.",
        },
    )

    result = _validate(repo)

    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert result.stderr == (
        f"facility: fix-stale: channel {CHANNEL} — `was` does not match the layers; fix: "
        f"replace the fix with {{fields: {{unit: um}}, id: '{CHANNEL}', kind: channel, "
        "op: set, was: {unit: {authored: mm, mml: mm}}, why: The unit is micrometres.}\n"
    )


def test_every_error_of_the_failing_stage_is_printed_sorted(repo: Path) -> None:
    _fixes(
        repo,
        {"op": "drop", "kind": "device", "id": "SR/Q9", "why": "gone"},
        {"op": "drop", "kind": "device", "id": "SR/Q8", "why": "gone"},
    )

    result = _validate(repo)

    assert result.exit_code == 1, result.output
    assert result.stderr.splitlines() == [
        f"facility: fix-missing: device SR/{name} — no device SR/{name} exists; fix: remove "
        "the fix from fixes.yaml"
        for name in ("Q8", "Q9")
    ]


def test_validate_has_no_json_flag(repo: Path) -> None:
    result = CliRunner().invoke(cli, ["facility", "validate", "--json", "--repo", str(repo)])

    assert result.exit_code == 2
