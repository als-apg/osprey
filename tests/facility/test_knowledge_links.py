"""A knowledge page linking what the facility file does not hold warns.

A page links the facility two ways: ``device_id`` in its frontmatter names a
device, and a ``[label](facility:ADDRESS)`` link in its body names a channel or
a device. The reader names every link under ``data/facility/knowledge`` that
the facility file does not resolve; the facility build states each once, as a
warning, and never stops on one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from osprey.facility.build import LATER_STAGES
from osprey.facility.knowledge_links import (
    DanglingLink,
    check_knowledge_links,
    dangling_links,
)
from osprey.facility.validate import Validated
from tests._builds import BuiltProject, run_build

DOC = {
    "channels": [{"id": "SR:DIAG:BPM:01:POSITION:X"}],
    "devices": [{"id": "SR/BPM01"}, {"id": "SR/QF1"}],
}


def _page(facility_dir: Path, relative: str, frontmatter: str, body: str = "# Body\n") -> None:
    path = facility_dir / "knowledge" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{frontmatter}---\n\n{body}", encoding="utf-8")


def test_a_page_linked_to_a_missing_device_is_named(tmp_path: Path) -> None:
    _page(tmp_path, "devices/gone.md", "type: device\ndevice_id: SR/BPM99\n")

    assert dangling_links(tmp_path, DOC) == [
        DanglingLink("knowledge/devices/gone.md", "SR/BPM99", "frontmatter")
    ]


def test_a_page_linked_to_a_held_device_is_not_named(tmp_path: Path) -> None:
    _page(tmp_path, "devices/bpm.md", "type: device\ndevice_id: SR/BPM01\n")

    assert dangling_links(tmp_path, DOC) == []


def test_device_id_naming_a_channel_dangles(tmp_path: Path) -> None:
    _page(tmp_path, "devices/x.md", "type: device\ndevice_id: SR:DIAG:BPM:01:POSITION:X\n")

    assert dangling_links(tmp_path, DOC) == [
        DanglingLink("knowledge/devices/x.md", "SR:DIAG:BPM:01:POSITION:X", "frontmatter")
    ]


def test_a_body_link_to_a_missing_channel_warns(tmp_path: Path) -> None:
    _page(
        tmp_path,
        "devices/bpm.md",
        "type: device\ndevice_id: SR/BPM99\n",
        "Read [Y](facility:SR:DIAG:BPM:01:POSITION:Y), [X](facility:SR:DIAG:BPM:01:POSITION:X)"
        " and [gauge](<facility:SR GAUGE 1>).\n",
    )
    validated = _validated(tmp_path)

    assert dangling_links(tmp_path, DOC) == [
        DanglingLink("knowledge/devices/bpm.md", "SR/BPM99", "frontmatter"),
        DanglingLink("knowledge/devices/bpm.md", "SR:DIAG:BPM:01:POSITION:Y", "body"),
        DanglingLink("knowledge/devices/bpm.md", "SR GAUGE 1", "body"),
    ]
    assert check_knowledge_links(validated) == []
    assert [warning.line for warning in validated.warnings][1:] == [
        "facility: link-dangling: page knowledge/devices/bpm.md — links "
        "facility:SR:DIAG:BPM:01:POSITION:Y, which the facility file does not hold; fix: link "
        "a channel id or a device id of the facility file, or remove the link",
        "facility: link-dangling: page knowledge/devices/bpm.md — links "
        "facility:SR GAUGE 1, which the facility file does not hold; fix: link "
        "a channel id or a device id of the facility file, or remove the link",
    ]


def test_a_body_link_to_a_held_channel_or_device_is_quiet(tmp_path: Path) -> None:
    _page(
        tmp_path,
        "notes.md",
        "type: concept\n",
        "[X](facility:SR:DIAG:BPM:01:POSITION:X) on [the monitor](facility:SR/BPM01);"
        " `[code](facility:NO/SUCH)` and [a page](/devices/bpm.md) are not facility links.\n",
    )

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


def _validated(facility_dir: Path) -> Validated:
    validated = Validated(facility_dir, "demo")
    validated.document = {key: [dict(record) for record in DOC[key]] for key in DOC}
    return validated


def test_a_dangling_page_is_a_build_warning(tmp_path: Path) -> None:
    _page(tmp_path, "devices/gone.md", "type: device\ndevice_id: SR/BPM99\n")
    validated = _validated(tmp_path)

    assert check_knowledge_links(validated) == []
    assert [warning.line for warning in validated.warnings] == [
        "facility: link-dangling: page knowledge/devices/gone.md — links device SR/BPM99, "
        "which the facility file does not hold; fix: set device_id to a device id of the "
        "facility file, or remove the key"
    ]


def test_a_held_page_adds_no_warning(tmp_path: Path) -> None:
    _page(tmp_path, "devices/bpm.md", "type: device\ndevice_id: SR/BPM01\n")
    validated = _validated(tmp_path)

    assert check_knowledge_links(validated) == []
    assert validated.warnings == []


def test_every_dangling_page_warns_in_page_order(tmp_path: Path) -> None:
    _page(tmp_path, "z.md", "device_id: SR/BPM98\n")
    _page(tmp_path, "a.md", "device_id: SR/BPM99\n")
    validated = _validated(tmp_path)

    assert check_knowledge_links(validated) == []
    assert [(warning.kind, warning.record_id) for warning in validated.warnings] == [
        ("link-dangling", "knowledge/a.md"),
        ("link-dangling", "knowledge/z.md"),
    ]


def test_the_knowledge_stage_runs_after_every_stop() -> None:
    assert LATER_STAGES[-1] == ("knowledge", check_knowledge_links)


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
    joined = " ".join(result.stderr.split())
    assert joined.count("link-dangling: page knowledge/gone.md") == 1
    assert "links device NO/SUCH9" in joined
    assert "linked.md" not in result.stderr
    assert (repo / facility_file).is_file()


def test_validate_and_show_name_a_dangling_page_on_one_stderr_line(
    built_control_assistant: BuiltProject, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    from click.testing import CliRunner

    from osprey.cli.main import cli

    repo = tmp_path / built_control_assistant.repo.name
    shutil.copytree(built_control_assistant.repo, repo, symlinks=True)
    knowledge = repo / "data" / "facility" / "knowledge"
    (knowledge / "gone.md").write_text(
        "---\ntype: device\ntitle: Gone\ndescription: Missing.\ndevice_id: NO/SUCH9\n---\n",
        encoding="utf-8",
    )
    expected = (
        "facility: link-dangling: page knowledge/gone.md — links device NO/SUCH9, which the "
        "facility file does not hold; fix: set device_id to a device id of the facility file, "
        "or remove the key"
    )

    capfd.readouterr()
    for verb in ("validate", "show"):
        result = CliRunner().invoke(cli, ["facility", verb, "--repo", str(repo)])
        stderr = result.stderr + capfd.readouterr().err

        assert result.exit_code == 0, result.output
        assert [line for line in stderr.splitlines() if "NO/SUCH9" in line] == [expected]
        assert "NO/SUCH9" not in result.stdout


def test_stubs_seeded_from_the_graph_view_link_held_devices(
    built_control_assistant: BuiltProject, tmp_path: Path
) -> None:
    from click.testing import CliRunner

    from osprey.cli.knowledge_cmd import knowledge

    graph_view = built_control_assistant.build_dir / "data" / "graph" / "facility.ttl"
    bundle = tmp_path / "knowledge"
    bundle.mkdir()

    result = CliRunner().invoke(knowledge, ["seed-from-ttl", str(bundle), "--ttl", str(graph_view)])

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


#: The example pages that name facility things, listed so the check below cannot
#: pass on an empty scan.
LINKING_PAGES: tuple[str, ...] = (
    "devices/bpm.md",
    "devices/ion-pump.md",
    "physics/beam-current-calibration.md",
    "physics/quadrupole-scan.md",
    "procedures/orbit-correction.md",
    "procedures/ps-startup.md",
    "procedures/pss-reset.md",
    "procedures/sample-scan.md",
    "procedures/vacuum-recovery.md",
    "references/epics-channel-access.md",
    "references/safety-rules.md",
    "subsystems/experimental-stations.md",
    "subsystems/primary-source.md",
    "subsystems/pss.md",
    "subsystems/timing.md",
    "subsystems/transport-delivery.md",
    "subsystems/vacuum.md",
)


def test_the_example_pages_link_only_held_records(built_control_assistant: BuiltProject) -> None:
    validated = Validated(built_control_assistant.facility_dir, "demo")
    validated.document = built_control_assistant.facility

    assert dangling_links(built_control_assistant.facility_dir, validated.document) == []
    assert check_knowledge_links(validated) == []
    assert [w for w in validated.warnings if w.kind == "link-dangling"] == []


def test_every_linking_example_page_links_a_held_record(
    built_control_assistant: BuiltProject,
) -> None:
    from osprey.facility.records import FacilityRecords
    from osprey.facility.references import find_facility_links
    from osprey.services.facility_knowledge.okf.document import OKFDocument

    records = FacilityRecords(built_control_assistant.facility)
    knowledge = built_control_assistant.facility_dir / "knowledge"
    unlinked = []
    for page in LINKING_PAGES:
        body = OKFDocument.parse((knowledge / page).read_text(encoding="utf-8")).body
        if not any(found.kinds for found in find_facility_links(body, records)):
            unlinked.append(page)
    assert unlinked == []
