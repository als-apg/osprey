"""A knowledge page linked to a device the facility file does not hold warns.

``device_id`` in a page's frontmatter is the link. The reader names every page
under ``data/facility/knowledge`` whose id is no device id of the facility
file; the build states each once and never stops on one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from osprey.cli import build_cmd
from osprey.facility.knowledge_links import DanglingLink, dangling_links
from tests._builds import BuiltProject, run_build

DOC = {"devices": [{"id": "SR/BPM01"}, {"id": "SR/QF1"}]}


def _page(facility_dir: Path, relative: str, frontmatter: str) -> None:
    path = facility_dir / "knowledge" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{frontmatter}---\n\n# Body\n", encoding="utf-8")


def test_a_page_linked_to_a_missing_device_is_named(tmp_path: Path) -> None:
    _page(tmp_path, "devices/gone.md", "type: device\ndevice_id: SR/BPM99\n")

    assert dangling_links(tmp_path, DOC) == [DanglingLink("knowledge/devices/gone.md", "SR/BPM99")]


def test_a_page_linked_to_a_held_device_is_not_named(tmp_path: Path) -> None:
    _page(tmp_path, "devices/bpm.md", "type: device\ndevice_id: SR/BPM01\n")

    assert dangling_links(tmp_path, DOC) == []


@pytest.mark.parametrize(
    "frontmatter",
    ["type: concept\n", "device_id:\n", "device_id: ''\n", "device_id: [SR/BPM99]\n"],
)
def test_a_page_without_the_key_is_unlinked(tmp_path: Path, frontmatter: str) -> None:
    _page(tmp_path, "notes.md", frontmatter)

    assert dangling_links(tmp_path, DOC) == []


def test_a_page_the_reader_cannot_parse_is_unlinked(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "plain.md").write_text("# No frontmatter\n", encoding="utf-8")
    (knowledge / "open.md").write_text("---\ndevice_id: SR/BPM99\n", encoding="utf-8")
    (knowledge / "bad.md").write_text("---\ndevice_id: [\n---\n", encoding="utf-8")
    (knowledge / "binary.md").write_bytes(b"\xff\xfe---\n")

    assert dangling_links(tmp_path, DOC) == []


def test_pages_are_named_in_sorted_order(tmp_path: Path) -> None:
    _page(tmp_path, "z.md", "device_id: B\n")
    _page(tmp_path, "a/b.md", "device_id: A\n")
    _page(tmp_path, "a.md", "device_id: C\n")

    assert [link.page for link in dangling_links(tmp_path, DOC)] == [
        "knowledge/a.md",
        "knowledge/a/b.md",
        "knowledge/z.md",
    ]


def test_a_tree_without_a_knowledge_directory_has_no_links(tmp_path: Path) -> None:
    assert dangling_links(tmp_path, DOC) == []
    assert dangling_links(tmp_path / "missing", {}) == []


def test_the_build_warns_once_per_page(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _page(tmp_path, "devices/gone.md", "device_id: SR/BPM99\n")
    _page(tmp_path, "devices/bpm.md", "device_id: SR/BPM01\n")
    reported: set[str] = set()

    build_cmd._warn_knowledge_links(tmp_path, DOC, reported, tmp_path.parent)
    first = " ".join(capsys.readouterr().err.split())
    build_cmd._warn_knowledge_links(tmp_path, DOC, reported, tmp_path.parent)
    second = capsys.readouterr().err

    assert f"{tmp_path.name}/knowledge/devices/gone.md" in first
    assert "SR/BPM99" in first
    assert "bpm.md" not in first
    assert second == ""


def test_a_single_render_names_every_page_it_finds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _page(tmp_path, "devices/gone.md", "device_id: SR/BPM99\n")

    build_cmd._warn_knowledge_links(tmp_path, DOC, None, tmp_path)
    build_cmd._warn_knowledge_links(tmp_path, DOC, None, tmp_path)

    assert capsys.readouterr().err.count("SR/BPM99") == 2


def test_the_page_is_spelled_from_the_profile_data_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    facility_dir = tmp_path / "sources" / "facility"
    _page(facility_dir, "devices/gone.md", "device_id: SR/BPM99\n")

    build_cmd._warn_knowledge_links(facility_dir, DOC, None, tmp_path)
    err = " ".join(capsys.readouterr().err.split())

    assert "sources/facility/knowledge/devices/gone.md" in err
    assert "data/facility" not in err


def test_the_build_names_a_dangling_page_and_still_writes_the_facility_file(
    built_control_assistant: BuiltProject, tmp_path: Path
) -> None:
    repo = tmp_path / built_control_assistant.repo.name
    shutil.copytree(built_control_assistant.repo, repo, symlinks=True)
    facility_file = (
        built_control_assistant.build_dir.relative_to(built_control_assistant.repo)
        / "facility.json"
    )
    (repo / facility_file).unlink()
    knowledge = repo / "data" / "facility" / "knowledge"
    held = built_control_assistant.facility["devices"][0]["id"]
    (knowledge / "linked.md").write_text(
        f"---\ntype: device\ntitle: Linked\ndescription: Held.\ndevice_id: {held}\n---\n",
        encoding="utf-8",
    )
    (knowledge / "gone.md").write_text(
        "---\ntype: device\ntitle: Gone\ndescription: Missing.\ndevice_id: NO/SUCH9\n---\n",
        encoding="utf-8",
    )

    result = run_build(repo)

    assert result.exit_code == 0, result.output
    lines = [line for line in result.stderr.splitlines() if "NO/SUCH9" in line]
    assert len(lines) == 1
    assert "data/facility/knowledge/gone.md" in lines[0]
    assert "linked.md" not in result.stderr
    assert (repo / facility_file).is_file()


def test_stubs_seeded_from_the_graph_view_link_held_devices(
    built_control_assistant: BuiltProject, tmp_path: Path
) -> None:
    from click.testing import CliRunner

    from osprey.cli.knowledge_cmd import knowledge

    graph_view = built_control_assistant.build_dir / "data" / "graph" / "facility.ttl"
    bundle = tmp_path / "knowledge"
    bundle.mkdir()

    result = CliRunner().invoke(knowledge, ["seed-from-ttl", str(graph_view), str(bundle)])

    assert result.exit_code == 0, result.output
    held = {device["id"] for device in built_control_assistant.facility["devices"]}
    linked = {
        line.removeprefix("device_id: ").strip("'\"")
        for page in bundle.rglob("*.md")
        for line in page.read_text(encoding="utf-8").splitlines()
        if line.startswith("device_id: ")
    }
    assert linked and linked <= held
    assert dangling_links(tmp_path, built_control_assistant.facility) == []
