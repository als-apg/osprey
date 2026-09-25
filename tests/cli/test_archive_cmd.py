"""``osprey archive``: one pass or a loop, over the trees it is handed."""

from __future__ import annotations

import os
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from osprey.cli import archive_cmd
from osprey.cli.archive_cmd import archive


def _put(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)


@pytest.fixture
def trees(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    os.chmod(dest, 0o700)
    return sources, dest


def test_archive_is_listed_and_resolves():
    from osprey.cli.main import cli

    ctx = click.Context(cli)
    assert "archive" in cli.list_commands(ctx)
    command = cli.get_command(ctx, "archive")
    assert isinstance(command, click.Command)
    assert command.name == "archive"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_once_exit_codes(trees, tmp_path):
    sources, dest = trees
    runner = CliRunner()
    _put(sources / "terminals/alice/projects/p/s.jsonl", "x\n")

    ok = runner.invoke(archive, ["--once", "--sources", str(sources), "--dest", str(dest)])
    assert ok.exit_code == 0, ok.output

    bad = sources / "terminals/alice/projects/p/unreadable.jsonl"
    _put(bad, "y\n")
    os.chmod(bad, 0)
    try:
        errored = runner.invoke(archive, ["--sources", str(sources), "--dest", str(dest)])
    finally:
        os.chmod(bad, 0o600)
    assert errored.exit_code == 1, errored.output

    missing = runner.invoke(
        archive, ["--once", "--sources", str(sources), "--dest", str(tmp_path / "absent")]
    )
    assert missing.exit_code == 2, missing.output


def test_watch_exits_after_five_passes_that_cannot_run(tmp_path, monkeypatch):
    sleeps: list[int] = []
    monkeypatch.setattr(archive_cmd, "_sleep", sleeps.append)

    result = CliRunner().invoke(
        archive,
        [
            "--watch",
            "--sources",
            str(tmp_path),
            "--dest",
            str(tmp_path / "absent"),
            "--interval",
            "60",
        ],
    )

    assert result.exit_code == 2, result.output
    assert sleeps == [60] * (archive_cmd.WATCH_FAILURE_CAP - 1)


def test_sources_and_dest_read_their_environment_variables(trees):
    sources, dest = trees
    _put(sources / "audit/ident/ledger.jsonl", "{}\n")

    result = CliRunner().invoke(
        archive,
        ["--once"],
        env={"OSPREY_ARCHIVE_SOURCES": str(sources), "OSPREY_ARCHIVE_DEST": str(dest)},
    )

    assert result.exit_code == 0, result.output
    copies = list(dest.glob("*/audit/ident/ledger.jsonl"))
    assert len(copies) == 1
