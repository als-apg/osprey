"""Both execution wrappers inject the one shared ``save_artifact()``.

The python executor and the visualization sandbox each generate a subprocess
script. A wrapper that re-inlined its own ``save_artifact()`` would compile and
run fine while quietly growing a second serializer — so the guard here is on
the generated text: the shared source must appear in both scripts verbatim.

The parent reads back what an execution saved as untrusted input: only regular
files sitting directly in ``artifacts/`` are collected, and none is read through
a link.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from osprey.mcp_server.workspace.execution.sandbox_executor import _create_sandbox_wrapper
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
from osprey.stores.artifact_manifest import (
    SAVE_ARTIFACT_SOURCE,
    CollectedFileRefused,
    collect_artifacts,
    is_collectable_file,
    read_collected_file,
)


def test_shared_source_is_valid_python():
    ast.parse(SAVE_ARTIFACT_SOURCE)
    assert "def save_artifact(" in SAVE_ARTIFACT_SOURCE


def test_python_executor_wrapper_injects_the_shared_source():
    script = ExecutionWrapper().create_wrapper("x = 1")

    assert SAVE_ARTIFACT_SOURCE.strip() in script
    assert script.count("def save_artifact(") == 1


def test_visualization_sandbox_injects_the_shared_source(tmp_path: Path):
    script = _create_sandbox_wrapper(
        "x = 1",
        execution_folder=tmp_path / "exec",
        workspace_root=tmp_path / "ws",
        project_root=tmp_path,
        secret_roots=(),
    )

    assert SAVE_ARTIFACT_SOURCE in script
    assert script.count("def save_artifact(") == 1


# ---------------------------------------------------------------------------
# Collection reads only regular files inside the artifacts folder
# ---------------------------------------------------------------------------

SECRET = b"API_KEY=do-not-read\n"


def _execution(tmp_path: Path) -> tuple[Path, Path]:
    """An execution folder with its ``artifacts/`` and a secret file outside it."""
    outside = tmp_path / "project"
    outside.mkdir()
    (outside / ".env").write_bytes(SECRET)
    folder = outside / "_agent_data" / "executions" / "run1"
    (folder / "artifacts").mkdir(parents=True)
    return folder, outside / ".env"


def _manifest(folder: Path, *filenames: str) -> None:
    entries = [{"filename": name, "title": name} for name in filenames]
    (folder / "artifacts" / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")


def _collected(folder: Path) -> list[str]:
    return [artifact["path"].name for artifact in collect_artifacts(folder)]


def test_a_filename_that_climbs_out_of_the_folder_is_not_collected(tmp_path: Path):
    folder, secret = _execution(tmp_path)
    (folder / "artifacts" / "kept.json").write_bytes(b"{}")
    _manifest(folder, "../../../../.env", "kept.json")

    assert (folder / "artifacts" / "../../../../.env").resolve() == secret.resolve()
    assert _collected(folder) == ["kept.json"]


def test_an_absolute_filename_is_not_collected(tmp_path: Path):
    folder, secret = _execution(tmp_path)
    (folder / "artifacts" / "kept.json").write_bytes(b"{}")
    _manifest(folder, str(secret), "kept.json")

    assert _collected(folder) == ["kept.json"]


@pytest.mark.parametrize(
    "filename",
    ["", ".", "..", "sub/inner.json", "sub\\inner.json", "manifest.json", "nul\x00.json"],
)
def test_a_name_that_is_not_a_file_of_the_folder_is_not_collected(tmp_path: Path, filename: str):
    folder, _ = _execution(tmp_path)
    (folder / "artifacts" / "sub").mkdir()
    (folder / "artifacts" / "sub" / "inner.json").write_bytes(b"{}")
    (folder / "artifacts" / "kept.json").write_bytes(b"{}")
    _manifest(folder, filename, "kept.json")

    assert _collected(folder) == ["kept.json"]


def test_a_link_in_the_folder_to_a_file_outside_is_neither_collected_nor_read(tmp_path: Path):
    folder, secret = _execution(tmp_path)
    link = folder / "artifacts" / "leak.txt"
    link.symlink_to(secret)
    (folder / "artifacts" / "kept.json").write_bytes(b"{}")
    _manifest(folder, "leak.txt", "kept.json")

    assert _collected(folder) == ["kept.json"]
    with pytest.raises(CollectedFileRefused):
        read_collected_file(link)


def test_a_hard_link_to_a_file_outside_is_neither_collected_nor_read(tmp_path: Path):
    folder, secret = _execution(tmp_path)
    link = folder / "artifacts" / "leak.txt"
    os.link(secret, link)
    (folder / "artifacts" / "kept.json").write_bytes(b"{}")
    _manifest(folder, "leak.txt", "kept.json")

    assert _collected(folder) == ["kept.json"]
    with pytest.raises(CollectedFileRefused):
        read_collected_file(link)


def test_an_artifacts_folder_that_is_a_link_collects_nothing(tmp_path: Path):
    folder, secret = _execution(tmp_path)
    (folder / "artifacts").rmdir()
    elsewhere = secret.parent / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "kept.json").write_bytes(b"{}")
    (elsewhere / "manifest.json").write_text(json.dumps([{"filename": "kept.json"}]))
    (folder / "artifacts").symlink_to(elsewhere, target_is_directory=True)

    assert collect_artifacts(folder) == []
    with pytest.raises(CollectedFileRefused):
        read_collected_file(folder / "artifacts" / "kept.json")


def test_a_manifest_that_is_a_link_collects_nothing(tmp_path: Path):
    folder, secret = _execution(tmp_path)
    (folder / "artifacts" / "kept.json").write_bytes(b"{}")
    real = secret.parent / "manifest.json"
    real.write_text(json.dumps([{"filename": "kept.json"}]), encoding="utf-8")
    (folder / "artifacts" / "manifest.json").symlink_to(real)

    assert collect_artifacts(folder) == []


def test_a_directory_or_a_fifo_under_an_artifact_name_is_not_collected(tmp_path: Path):
    folder, _ = _execution(tmp_path)
    (folder / "artifacts" / "folder.json").mkdir()
    os.mkfifo(folder / "artifacts" / "pipe.json")
    (folder / "artifacts" / "kept.json").write_bytes(b"{}")
    _manifest(folder, "folder.json", "pipe.json", "kept.json")

    assert _collected(folder) == ["kept.json"]


def test_a_file_over_the_size_cap_is_not_read(tmp_path: Path):
    folder, _ = _execution(tmp_path)
    big = folder / "artifacts" / "big.bin"
    big.write_bytes(b"x" * 9)

    assert read_collected_file(big, max_bytes=9) == b"x" * 9
    with pytest.raises(CollectedFileRefused):
        read_collected_file(big, max_bytes=8)
    assert not is_collectable_file(big, max_bytes=8)


def test_a_collected_file_reads_back_its_bytes(tmp_path: Path):
    folder, _ = _execution(tmp_path)
    (folder / "artifacts" / "kept.json").write_bytes(b'{"a": 1}')
    _manifest(folder, "kept.json")

    (artifact,) = collect_artifacts(folder)

    assert read_collected_file(artifact["path"]) == b'{"a": 1}'


def test_files_past_the_entry_cap_are_not_collected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import osprey.stores.artifact_manifest as artifact_manifest

    monkeypatch.setattr(artifact_manifest, "MAX_COLLECTED_ENTRIES", 2)
    folder, _ = _execution(tmp_path)
    names = ["a.json", "b.json", "c.json"]
    for name in names:
        (folder / "artifacts" / name).write_bytes(b"{}")
    _manifest(folder, *names)

    assert _collected(folder) == ["a.json", "b.json"]


def test_files_past_the_run_byte_cap_are_not_collected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import osprey.stores.artifact_manifest as artifact_manifest

    monkeypatch.setattr(artifact_manifest, "MAX_COLLECTED_TOTAL_BYTES", 10)
    folder, _ = _execution(tmp_path)
    (folder / "artifacts" / "a.bin").write_bytes(b"x" * 6)
    (folder / "artifacts" / "big.bin").write_bytes(b"x" * 6)
    (folder / "artifacts" / "small.bin").write_bytes(b"x" * 4)
    _manifest(folder, "a.bin", "big.bin", "small.bin")

    assert _collected(folder) == ["a.bin", "small.bin"]


def test_a_file_swapped_for_another_between_the_check_and_the_open_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    folder, secret = _execution(tmp_path)
    target = folder / "artifacts" / "kept.json"
    target.write_bytes(b"{}")
    real_open = os.open
    # The checked file stays open across the swap, so its inode cannot be
    # freed and handed to the replacement: Linux file systems reuse a freed
    # inode number at once, which would make the swap indistinguishable.
    held = real_open(target, os.O_RDONLY)

    def swapping_open(path, flags, *args, **kwargs):
        if path == "kept.json":
            target.unlink()
            target.write_bytes(b"swapped")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swapping_open)

    try:
        with pytest.raises(CollectedFileRefused):
            read_collected_file(target)
    finally:
        os.close(held)
