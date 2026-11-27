"""Tests for ``osprey scaffold pull`` and the packaged-template lookups it uses.

The lookups come first because everything else here depends on them: a preset
name has to become an app template, and an app template has to become a
directory on disk, before there is anything to copy out of the installation.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import click
import pytest
from click.testing import CliRunner

from osprey.cli.build_profile import list_presets
from osprey.cli.main import cli
from osprey.cli.profile_cmd import (
    _app_template_root,
    _preset_data,
    _resolve_preset_bundle,
)
from osprey.cli.scaffold_cmd import scaffold
from osprey.cli.scaffold_pull import (
    PullAction,
    apply_pull,
    list_pullable_paths,
    plan_pull,
)
from osprey.cli.templates.manager import TemplateManager
from osprey.cli.templates.preset_data import PresetData
from osprey.services.facility_knowledge.okf.index import check_indexes
from osprey_connectors.errors import BuildProfileError

# The forward check the emitted CI files already get, borrowed rather than
# rewritten: one extraction and one resolver, so a verb named in help text and a
# verb named in a pipeline are held to the same CLI.
from tests.cli.test_scaffold_ci import named_commands, unresolvable


@pytest.fixture
def manager() -> TemplateManager:
    """A manager pointed at the installed template root."""
    return TemplateManager()


# ---------------------------------------------------------------------------
# App-template resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["hello-world", "demo"])
def test_app_template_root_resolver_locates_a_shipped_bundle(
    manager: TemplateManager, preset: str
) -> None:
    """Every preset's app template is a real directory in the installation."""
    _name, data_bundle = _resolve_preset_bundle(preset)

    root = _app_template_root(manager, data_bundle)

    assert root.is_dir()
    assert root == Path(manager.template_root) / "apps" / data_bundle


def test_app_template_root_resolver_rejects_an_absent_bundle(manager: TemplateManager) -> None:
    """A bundle that ships nothing is a packaging fault, named as one."""
    with pytest.raises(BuildProfileError) as excinfo:
        _app_template_root(manager, "no_such_app_template")

    message = str(excinfo.value)
    assert "no_such_app_template" in message
    assert "reinstall" in message


def test_preset_data_composes_the_app_template_and_its_facility(
    manager: TemplateManager,
) -> None:
    """The packaged data is the app template's ``web/`` plus the named facility."""
    composed = _preset_data(manager, "demo")

    assert composed.app_root == _app_template_root(manager, "demo")
    assert composed.app_web == composed.app_root / "web"
    assert composed.facility_root == Path(manager.template_root) / "facilities" / "example"
    assert not (composed.app_web / "facility").exists()
    assert composed.placed_files()["facility/identity.yaml"] == (
        composed.facility_root / "identity.yaml"
    )


def test_preset_data_composes_an_app_template_with_no_data_of_its_own(
    manager: TemplateManager,
) -> None:
    """hello-world's app template ships no ``web/``; its facility and the starter are all of it."""
    composed = _preset_data(manager, "hello-world")

    assert not composed.app_web.exists()
    assert composed.facility_root == Path(manager.template_root) / "facilities" / "hello_world"
    assert {
        relative for relative in composed.placed_files() if not relative.startswith("facility/")
    } == {"agent/code_review/reviewer.md"}


# ---------------------------------------------------------------------------
# Preset -> app template
# ---------------------------------------------------------------------------


def test_resolve_preset_bundle_returns_the_name_and_its_app_template() -> None:
    """The preset's own bundle is what comes back beside the name.

    Read back from the preset chain the way the build reads it: the bundle is a
    preset-side fact, and the resolved profile carries no such key.
    """
    from osprey.cli.build_cmd import _profile_data_bundle
    from osprey.cli.build_profile import resolve_build_profile

    expected, _preset_dir = resolve_build_profile(None, "demo")

    assert _resolve_preset_bundle("demo") == (
        "demo",
        _profile_data_bundle(expected),
    )


def test_resolve_preset_bundle_normalizes_the_underscore_spelling() -> None:
    """Both CLI spellings of a preset resolve to the same answer."""
    assert _resolve_preset_bundle("hello_world") == _resolve_preset_bundle("hello-world")
    assert _resolve_preset_bundle("hello_world")[0] == "hello-world"


def test_resolve_preset_bundle_rejects_an_unknown_preset() -> None:
    """An unknown name is a usage error listing every preset that does exist."""
    with pytest.raises(click.UsageError) as excinfo:
        _resolve_preset_bundle("not-a-preset")

    message = str(excinfo.value)
    assert "not-a-preset" in message
    presets = list_presets()
    assert presets
    for name in presets:
        assert name in message


# ---------------------------------------------------------------------------
# Pullable-path catalog
# ---------------------------------------------------------------------------

#: Every path the demo template offers, pinned exactly. Packaged
#: data is fixed for a release, so a diff here is a real change to what an
#: operator can pull — a bundle gaining or losing content, or the exclusions
#: drifting — and is meant to be read before it is accepted.
DEMO_PULLABLE = [
    "agent/",
    "agent/code_review/",
    "agent/context/",
    "facility/",
    "facility/ariel/",
    "facility/benchmarks/",
    "facility/benchmarks/channel_finder/",
    "facility/knowledge/",
    "facility/knowledge/devices/",
    "facility/knowledge/devices/figures/",
    "facility/knowledge/physics/",
    "facility/knowledge/procedures/",
    "facility/knowledge/references/",
    "facility/knowledge/subsystems/",
    "facility/provenance/",
    "facility/records/",
    "facility/simulation/",
    "facility/simulation/decks/",
    "facility/simulation/measurement/",
    "facility/simulation/scenarios/",
    "facility/simulation/scenarios/bpm-polarity/",
    "facility/simulation/scenarios/bpm-polarity/plots/",
    "facility/simulation/scenarios/nominal/",
    "facility/simulation/scenarios/nominal/plots/",
    "facility/simulation/scenarios/rf-thermal/",
    "facility/simulation/scenarios/rf-thermal/plots/",
    "facility/vocabulary/",
    "web/",
    "web/canopy/",
    "web/canopy/collections/",
    "web/canopy/library/",
    "web/canopy/library/screens/",
    "web/canopy/library/screens/br/",
    "web/canopy/library/screens/bts/",
    "web/canopy/library/screens/common/",
    "web/canopy/library/screens/common/embedded/",
    "web/canopy/library/screens/common/images/",
    "web/canopy/library/screens/facility/",
    "web/canopy/library/screens/gtl/",
    "web/canopy/library/screens/sr/",
    "web/canopy/theme/",
    "web/landing/",
    "agent/code_review/reviewer.md",
    "agent/context/base.md",
    "facility/ariel/README.md",
    "facility/ariel/vocabulary.yml",
    "facility/benchmarks/channel_finder/in_context_queries.json",
    "facility/benchmarks/channel_finder/tree_queries.json",
    "facility/identity.yaml",
    "facility/knowledge/devices/bpm.md",
    "facility/knowledge/devices/figures/bpm-button-pickups.png",
    "facility/knowledge/devices/index.md",
    "facility/knowledge/devices/ion-pump.md",
    "facility/knowledge/index.md",
    "facility/knowledge/physics/beam-current-calibration.md",
    "facility/knowledge/physics/index.md",
    "facility/knowledge/physics/quadrupole-scan.md",
    "facility/knowledge/procedures/index.md",
    "facility/knowledge/procedures/orbit-correction.md",
    "facility/knowledge/procedures/ps-startup.md",
    "facility/knowledge/procedures/pss-reset.md",
    "facility/knowledge/procedures/sample-scan.md",
    "facility/knowledge/procedures/vacuum-recovery.md",
    "facility/knowledge/references/epics-channel-access.md",
    "facility/knowledge/references/index.md",
    "facility/knowledge/references/safety-rules.md",
    "facility/knowledge/subsystems/experimental-stations.md",
    "facility/knowledge/subsystems/index.md",
    "facility/knowledge/subsystems/primary-source.md",
    "facility/knowledge/subsystems/pss.md",
    "facility/knowledge/subsystems/timing.md",
    "facility/knowledge/subsystems/transport-delivery.md",
    "facility/knowledge/subsystems/vacuum.md",
    "facility/limits.yaml",
    "facility/provenance/documents.yaml",
    "facility/provenance/policy.yaml",
    "facility/records/channels.yaml",
    "facility/records/devices.yaml",
    "facility/records/groups.yaml",
    "facility/records/places.yaml",
    "facility/records/relations.yaml",
    "facility/simulation/decks/BR.json",
    "facility/simulation/decks/BTS.json",
    "facility/simulation/decks/LTB.json",
    "facility/simulation/decks/SR.json",
    "facility/simulation/measurement/BR.yaml",
    "facility/simulation/measurement/BTS.yaml",
    "facility/simulation/measurement/LTB.yaml",
    "facility/simulation/measurement/SR.yaml",
    "facility/simulation/models.yaml",
    "facility/simulation/scenarios/bpm-polarity.yaml",
    "facility/simulation/scenarios/bpm-polarity/plots/corrector_bump_test.png",
    "facility/simulation/scenarios/bts-server-stopped.yaml",
    "facility/simulation/scenarios/demo-history.yaml",
    "facility/simulation/scenarios/nominal.yaml",
    "facility/simulation/scenarios/nominal/plots/orbit_rms.json",
    "facility/simulation/scenarios/orm-dual-fault.yaml",
    "facility/simulation/scenarios/rf-thermal-live.yaml",
    "facility/simulation/scenarios/rf-thermal.yaml",
    "facility/simulation/scenarios/rf-thermal/plots/cavity_temperatures.json",
    "facility/simulation/scenarios/vacuum-burst.yaml",
    "facility/simulation/seeds.yaml",
    "facility/vocabulary/classes.yaml",
    "web/canopy/collections/demo.json",
    "web/canopy/library/LIBRARY-REPORT.json",
    "web/canopy/library/LIBRARY-REPORT.md",
    "web/canopy/library/screens/br/overview.screen.json",
    "web/canopy/library/screens/bts/BTS.screen.json",
    "web/canopy/library/screens/common/embedded/_KnobDevice.screen.json",
    "web/canopy/library/screens/common/embedded/_bpm.screen.json",
    "web/canopy/library/screens/common/embedded/_tv.screen.json",
    "web/canopy/library/screens/common/images/Light off.jpg",
    "web/canopy/library/screens/common/images/Light on.jpg",
    "web/canopy/library/screens/common/images/Paddle in.jpg",
    "web/canopy/library/screens/common/images/Paddle out.jpg",
    "web/canopy/library/screens/common/images/TV off.jpg",
    "web/canopy/library/screens/common/images/TV on.jpg",
    "web/canopy/library/screens/facility/overview.screen.json",
    "web/canopy/library/screens/gtl/GTL.screen.json",
    *(f"web/canopy/library/screens/sr/_absorbers-{n:02d}.screen.json" for n in (2, *range(4, 13))),
    *(f"web/canopy/library/screens/sr/_vacuum-{n:02d}.screen.json" for n in range(1, 13)),
    "web/canopy/library/screens/sr/orbit-feedback.screen.json",
    *(f"web/canopy/library/screens/sr/sector-{n:02d}.screen.json" for n in range(1, 13)),
    "web/canopy/theme/osprey.json",
    "web/landing/working-safely.md",
]


@pytest.fixture
def demo(manager: TemplateManager) -> PresetData:
    """The demo preset's packaged content: app template + facility."""
    return _preset_data(manager, "demo")


def packaged(source: PresetData, relative: str | Path) -> Path:
    """Where the template-relative *relative* is read from in the installation."""
    parts = PurePosixPath(Path(relative).as_posix()).parts
    placed = source.placed_files().get("/".join(parts))
    if placed is not None:
        return placed
    if source.facility_root is not None and parts[:1] == ("facility",):
        return source.facility_root.joinpath(*parts[1:])
    return source.app_root.joinpath(*parts)


def _stage_template(root: Path, relative_paths: list[str]) -> None:
    """Write an empty file at each path under ``root``, creating parents."""
    for relative in relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def test_list_pullable_paths_pins_the_demo_catalog(
    demo: PresetData,
) -> None:
    """The whole listing, directories first and each group sorted."""
    assert list_pullable_paths(demo) == DEMO_PULLABLE


def test_list_pullable_paths_offers_the_content_an_operator_edits(
    demo: PresetData,
) -> None:
    """The two trees a facility rewrites first are both named, machinery is not."""
    listed = list_pullable_paths(demo)

    assert "facility/knowledge/" in listed
    assert "agent/context/" in listed
    assert [entry for entry in listed if entry.endswith(".j2")] == []
    assert "__init__.py" not in listed


def test_list_pullable_paths_keeps_a_nested_package_marker(manager: TemplateManager) -> None:
    """A bundled MCP server is content, so its own ``__init__.py`` survives."""
    listed = list_pullable_paths(_preset_data(manager, "hello-world"))

    assert "agent/mcp_servers/example_server/__init__.py" in listed
    assert "__init__.py" not in listed


def test_list_pullable_paths_never_lists_a_bytecode_cache(tmp_path: Path) -> None:
    """Importing a template's modules leaves ``__pycache__`` beside them; it is never content."""
    root = tmp_path / "app_template"
    _stage_template(
        root,
        [
            "__init__.py",
            "__pycache__/__init__.cpython-313.pyc",
            "web/README.md",
            "web/tools/__pycache__/x.cpython-313.pyc",
            "web/tools/x.py",
        ],
    )

    listed = list_pullable_paths(PresetData(root))

    assert listed == ["web/", "web/tools/", "web/README.md", "web/tools/x.py"]


def test_list_pullable_paths_restricts_to_a_subtree(demo: PresetData) -> None:
    """A directory subtree yields itself and everything below it, and nothing else."""
    listed = list_pullable_paths(demo, "facility/knowledge")

    assert "facility/knowledge/" in listed
    assert "facility/knowledge/procedures/orbit-correction.md" in listed
    assert all(entry.startswith("facility/knowledge/") for entry in listed)
    assert listed == [entry for entry in DEMO_PULLABLE if entry.startswith("facility/knowledge/")]


def test_list_pullable_paths_restricts_to_a_single_file(demo: PresetData) -> None:
    """A file subtree is just that file."""
    assert list_pullable_paths(demo, "web/canopy/theme/osprey.json") == [
        "web/canopy/theme/osprey.json"
    ]


def test_list_pullable_paths_rejects_an_unknown_subtree(demo: PresetData) -> None:
    """A miss names the top-level entries, which is what corrects the path."""
    with pytest.raises(ValueError) as excinfo:
        list_pullable_paths(demo, "facility_knowlege")

    message = str(excinfo.value)
    assert "facility_knowlege" in message
    assert "facility/" in message
    assert "agent/" in message
    assert ".j2" not in message


# ---------------------------------------------------------------------------
# Pull plan
# ---------------------------------------------------------------------------

#: The knowledge base as the demo template ships it, pinned so a
#: skeleton pull and a full pull are both anchored to real numbers: 23 Markdown
#: documents, of which 6 are the ``index.md`` files that carry the structure,
#: and 1 figure a concept embeds. Knowledge content is not Markdown only: a
#: bundle image is content like the concept that shows it.
KNOWLEDGE_MARKDOWN_FILES = 23
KNOWLEDGE_INDEX_FILES = 6
KNOWLEDGE_IMAGE_FILES = 1
KNOWLEDGE_FILES = KNOWLEDGE_MARKDOWN_FILES + KNOWLEDGE_IMAGE_FILES


def _by_action(actions: list[PullAction]) -> dict[str, list[PullAction]]:
    """The plan grouped by outcome, so a test can count one kind at a time."""
    grouped: dict[str, list[PullAction]] = {}
    for action in actions:
        grouped.setdefault(action.action, []).append(action)
    return grouped


def test_plan_pull_counts_match_the_packaged_knowledge_base(
    demo: PresetData,
) -> None:
    """The pinned totals above are what the template actually ships."""
    listed = list_pullable_paths(demo, "facility/knowledge")
    markdown = [entry for entry in listed if entry.endswith(".md")]

    assert len(markdown) == KNOWLEDGE_MARKDOWN_FILES
    assert (
        len([entry for entry in markdown if entry.endswith("/index.md")]) == KNOWLEDGE_INDEX_FILES
    )
    files = [entry for entry in listed if not entry.endswith("/")]
    images = [entry for entry in files if entry.endswith(".png")]
    assert len(images) == KNOWLEDGE_IMAGE_FILES
    # Nothing else is in there: every file is a document or a figure.
    assert sorted(files) == sorted(markdown + images)
    assert len(files) == KNOWLEDGE_FILES


def test_plan_pull_of_the_knowledge_base_writes_only_the_indexes(
    demo: PresetData, tmp_path: Path
) -> None:
    """A plain pull produces the structure and leaves the demo documents behind."""
    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge",
        force=False,
        with_content=False,
    )

    grouped = _by_action(plan)
    assert sorted(grouped) == ["skipped", "written"]
    assert len(grouped["written"]) == KNOWLEDGE_INDEX_FILES
    assert all(action.source.name == "index.md" for action in grouped["written"])
    # The figure is content too, so it stays behind with the demo concepts.
    assert len(grouped["skipped"]) == KNOWLEDGE_FILES - KNOWLEDGE_INDEX_FILES
    assert any(action.source.suffix == ".png" for action in grouped["skipped"])
    assert all("--with-content" in action.reason for action in grouped["skipped"])


def test_plan_pull_with_content_writes_every_knowledge_document(
    demo: PresetData, tmp_path: Path
) -> None:
    """The flag turns the skeleton into the whole worked example."""
    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge",
        force=False,
        with_content=True,
    )

    assert sorted(_by_action(plan)) == ["written"]
    assert len(plan) == KNOWLEDGE_FILES
    suffixes = sorted(action.target.suffix for action in plan)
    assert suffixes == [".md"] * KNOWLEDGE_MARKDOWN_FILES + [".png"] * KNOWLEDGE_IMAGE_FILES


def test_plan_pull_mirrors_the_template_relative_path_under_the_repo(
    demo: PresetData, tmp_path: Path
) -> None:
    """Targets are the source path again, rooted at the repo."""
    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge/index.md",
        force=False,
        with_content=False,
    )

    assert [action.target for action in plan] == [tmp_path / "facility" / "knowledge" / "index.md"]
    assert plan[0].source == packaged(demo, "facility/knowledge/index.md")


def test_plan_pull_refuses_a_single_filtered_knowledge_file(
    demo: PresetData, tmp_path: Path
) -> None:
    """Skipping the only file asked for would do nothing, so it is refused instead."""
    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge/devices/bpm.md",
        force=False,
        with_content=False,
    )

    assert len(plan) == 1
    assert plan[0].action == "refused"
    assert "--with-content" in plan[0].reason


def test_plan_pull_refuses_an_existing_destination(demo: PresetData, tmp_path: Path) -> None:
    """A file already in the repo stops the pull and the reason names the flag."""
    _stage_template(tmp_path, ["facility/knowledge/index.md"])

    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge",
        force=False,
        with_content=False,
    )

    refused = _by_action(plan)["refused"]
    assert [action.target for action in refused] == [
        tmp_path / "facility" / "knowledge" / "index.md"
    ]
    assert "--force" in refused[0].reason


def test_plan_pull_updates_an_existing_destination_under_force(
    demo: PresetData, tmp_path: Path
) -> None:
    """With the flag, a differing file is an update rather than a refusal."""
    _stage_template(tmp_path, ["facility/knowledge/index.md"])

    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge",
        force=True,
        with_content=False,
    )

    grouped = _by_action(plan)
    assert "refused" not in grouped
    assert [action.target for action in grouped["updated"]] == [
        tmp_path / "facility" / "knowledge" / "index.md"
    ]
    assert len(grouped["written"]) == KNOWLEDGE_INDEX_FILES - 1


def test_plan_pull_reports_an_identical_destination_as_unchanged(
    demo: PresetData, tmp_path: Path
) -> None:
    """Re-pulling what is already there is a no-op, reported as one."""
    source = packaged(demo, "facility/knowledge/index.md")
    target = tmp_path / "facility" / "knowledge" / "index.md"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())

    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge/index.md",
        force=True,
        with_content=False,
    )

    assert [(action.action, action.target) for action in plan] == [("unchanged", target)]


def test_plan_pull_refuses_a_symlinked_target_directory(demo: PresetData, tmp_path: Path) -> None:
    """A link on the way to the target would land the copy somewhere else."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    repo_root = tmp_path / "repo"
    (repo_root / "facility").mkdir(parents=True)
    (repo_root / "facility" / "knowledge").symlink_to(elsewhere, target_is_directory=True)

    plan = plan_pull(
        demo,
        repo_root,
        "facility/knowledge",
        force=True,
        with_content=True,
    )

    assert sorted(_by_action(plan)) == ["refused"]
    assert len(plan) == KNOWLEDGE_FILES
    assert "symlink" in plan[0].reason
    # Planning is pure: nothing reached the directory the link points at.
    assert list(elsewhere.iterdir()) == []


def test_plan_pull_refuses_a_directory_where_the_file_goes(
    demo: PresetData, tmp_path: Path
) -> None:
    """A kind mismatch is never resolved on the facility's behalf."""
    (tmp_path / "facility" / "knowledge" / "index.md").mkdir(parents=True)

    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge/index.md",
        force=True,
        with_content=False,
    )

    assert plan[0].action == "refused"
    assert "directory" in plan[0].reason


def test_plan_pull_refuses_a_file_where_a_directory_is_needed(
    demo: PresetData, tmp_path: Path
) -> None:
    """The mismatch counts anywhere between the repo root and the target."""
    _stage_template(tmp_path, ["facility/knowledge/devices"])

    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge/devices/bpm.md",
        force=True,
        with_content=True,
    )

    assert plan[0].action == "refused"
    assert "facility/knowledge/devices" in plan[0].reason


def test_plan_pull_keeps_a_nested_package_marker(manager: TemplateManager, tmp_path: Path) -> None:
    """A bundled MCP server pulls with its own ``__init__.py``."""
    plan = plan_pull(
        _preset_data(manager, "hello-world"),
        tmp_path,
        "agent/mcp_servers/example_server",
        force=False,
        with_content=False,
    )

    marker = tmp_path / "agent" / "mcp_servers" / "example_server" / "__init__.py"
    assert [action.action for action in plan if action.target == marker] == ["written"]


def test_plan_pull_rejects_an_unknown_path(demo: PresetData, tmp_path: Path) -> None:
    """An unknown path fails the same way the listing does, with the same message."""
    with pytest.raises(ValueError, match="facility_knowlege"):
        plan_pull(
            demo,
            tmp_path,
            "facility_knowlege",
            force=False,
            with_content=False,
        )


# ---------------------------------------------------------------------------
# Applying a pull
# ---------------------------------------------------------------------------


#: The one file name the knowledge base treats as structure rather than content.
_INDEX_NAME = "index.md"


def _tree(root: Path) -> list[tuple[str, bytes | None]]:
    """Every path under ``root``, with file contents — a snapshot to compare."""
    return [
        (
            path.relative_to(root).as_posix(),
            path.read_bytes() if path.is_file() and not path.is_symlink() else None,
        )
        for path in sorted(root.rglob("*"))
    ]


def _packaged_demo_titles(knowledge_root: Path) -> set[str]:
    """Every document title the packaged sub-directory indexes advertise.

    These are the demo concepts a facility replaces. None of them may survive a
    skeleton pull, where the documents themselves are left behind.
    """
    titles: set[str] = set()
    for index in knowledge_root.rglob(_INDEX_NAME):
        if index.parent == knowledge_root:
            continue  # The root index links directories, not documents.
        for line in index.read_text(encoding="utf-8").splitlines():
            if line.startswith("* [") and "](" in line:
                titles.add(line[3 : line.index("](")])
    return titles


def test_apply_pull_of_the_knowledge_base_writes_only_the_indexes(
    demo: PresetData, tmp_path: Path
) -> None:
    """A skeleton pull lands the structure and not one demo document."""
    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge",
        force=False,
        with_content=False,
    )

    applied = apply_pull(plan, repo_root=tmp_path, with_content=False)

    written = sorted(
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()
    )
    assert len(written) == KNOWLEDGE_INDEX_FILES
    assert all(entry.endswith(f"/{_INDEX_NAME}") for entry in written)
    assert len(applied) == KNOWLEDGE_INDEX_FILES


def test_apply_pull_leaves_no_demo_document_title_in_a_pulled_index(
    demo: PresetData, tmp_path: Path
) -> None:
    """The rebuilt indexes name what is on disk, so the demo concepts are gone."""
    knowledge_source = packaged(demo, "facility/knowledge")
    demo_titles = _packaged_demo_titles(knowledge_source)
    assert "Beam Position Monitor (BPM)" in demo_titles

    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge",
        force=False,
        with_content=False,
    )
    apply_pull(plan, repo_root=tmp_path, with_content=False)

    knowledge_target = tmp_path / "facility" / "knowledge"
    pulled = "\n".join(
        index.read_text(encoding="utf-8") for index in knowledge_target.rglob(_INDEX_NAME)
    )
    for title in demo_titles:
        assert title not in pulled
    assert "Beam Position Monitor" not in pulled
    # The structure itself survives: the root index still names its directories.
    root_index = (knowledge_target / _INDEX_NAME).read_text(encoding="utf-8")
    assert "okf_version" in root_index
    for subdirectory in ("devices", "physics", "procedures", "references", "subsystems"):
        assert f"[{subdirectory}](/{subdirectory}/)" in root_index


def test_a_skeleton_pull_leaves_indexes_that_validate(demo: PresetData, tmp_path: Path) -> None:
    """A skeleton pull leaves every index as regen-index would write it."""
    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge",
        force=False,
        with_content=False,
    )
    apply_pull(plan, repo_root=tmp_path, with_content=False)

    knowledge_target = tmp_path / "facility" / "knowledge"
    assert check_indexes(knowledge_target) == []


def test_apply_pull_with_content_writes_the_packaged_knowledge_base_verbatim(
    demo: PresetData, tmp_path: Path
) -> None:
    """With the documents in hand the packaged indexes are already true, so they stand."""
    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge",
        force=False,
        with_content=True,
    )

    applied = apply_pull(plan, repo_root=tmp_path, with_content=True)

    written = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    assert len(applied) == KNOWLEDGE_FILES
    assert len(written) == KNOWLEDGE_FILES
    assert len([path for path in written if path.suffix == ".png"]) == KNOWLEDGE_IMAGE_FILES
    for path in written:
        relative = path.relative_to(tmp_path)
        assert path.read_bytes() == packaged(demo, relative).read_bytes()
    assert check_indexes(tmp_path / "facility" / "knowledge") == []


def test_apply_pull_writes_nothing_when_the_plan_holds_a_refusal(
    demo: PresetData, tmp_path: Path
) -> None:
    """One refusal stops the whole pull, so there is no half-applied copy to undo."""
    _stage_template(tmp_path, ["facility/knowledge/index.md"])
    before = _tree(tmp_path)

    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge",
        force=False,
        with_content=False,
    )
    assert "refused" in _by_action(plan)

    returned = apply_pull(plan, repo_root=tmp_path, with_content=False)

    assert returned == plan
    assert _tree(tmp_path) == before


def test_apply_pull_overwrites_under_force_and_keeps_a_file_the_template_lacks(
    demo: PresetData, tmp_path: Path
) -> None:
    """A pull replaces what it names and removes nothing it does not."""
    _stage_template(
        tmp_path,
        ["facility/knowledge/index.md", "facility/knowledge/devices/local-note.md"],
    )

    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge",
        force=True,
        with_content=True,
    )
    apply_pull(plan, repo_root=tmp_path, with_content=True)

    target = tmp_path / "facility" / "knowledge" / "index.md"
    assert target.read_bytes() == (packaged(demo, "facility/knowledge/index.md")).read_bytes()
    assert (tmp_path / "facility" / "knowledge" / "devices" / "local-note.md").is_file()


def test_apply_pull_writes_a_nested_package_marker(
    manager: TemplateManager, tmp_path: Path
) -> None:
    """A bundled MCP server lands importable, marker and all."""
    source = _preset_data(manager, "hello-world")

    plan = plan_pull(
        source, tmp_path, "agent/mcp_servers/example_server", force=False, with_content=False
    )
    apply_pull(plan, repo_root=tmp_path, with_content=False)

    marker = tmp_path / "agent" / "mcp_servers" / "example_server" / "__init__.py"
    assert marker.is_file()
    assert (
        marker.read_bytes()
        == (source.app_root / "agent/mcp_servers/example_server/__init__.py").read_bytes()
    )


def test_apply_pull_returns_the_actions_it_applied(demo: PresetData, tmp_path: Path) -> None:
    """The command reports from what came back, and those are the planned objects."""
    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge",
        force=False,
        with_content=False,
    )

    applied = apply_pull(plan, repo_root=tmp_path, with_content=False)

    assert [action.action for action in applied] == ["written"] * KNOWLEDGE_INDEX_FILES
    assert all(any(action is planned for planned in plan) for action in applied)
    assert _by_action(plan)["skipped"]  # the skipped documents were quietly left alone


def test_apply_pull_leaves_an_unchanged_target_alone(demo: PresetData, tmp_path: Path) -> None:
    """Re-pulling an identical file writes nothing and rebuilds nothing."""
    source = packaged(demo, "facility/knowledge/index.md")
    target = tmp_path / "facility" / "knowledge" / "index.md"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())

    plan = plan_pull(
        demo,
        tmp_path,
        "facility/knowledge/index.md",
        force=True,
        with_content=False,
    )

    assert apply_pull(plan, repo_root=tmp_path, with_content=False) == []
    assert target.read_bytes() == source.read_bytes()


# ---------------------------------------------------------------------------
# The verb
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A deployment repo with nothing in it but the marker that makes it one.

    ``find_repo_root`` walks up to a ``profile.yml`` and reads nothing out of
    it, and a pull renders nothing from the profile either — so an empty marker
    is the whole repo this verb needs, and a fixture that built more would be
    testing the builder.
    """
    root = tmp_path / "deployment"
    root.mkdir()
    (root / "profile.yml").write_text("", encoding="utf-8")
    return root


def run(runner: CliRunner, repo: Path, *args: str):
    """Run ``osprey scaffold pull`` against *repo* through the real CLI."""
    return runner.invoke(cli, ["scaffold", "pull", *args, "--repo", str(repo)])


def _markdown(repo: Path) -> list[str]:
    """Every Markdown file in the repo, repo-relative and sorted."""
    return sorted(path.relative_to(repo).as_posix() for path in repo.rglob("*.md"))


def test_the_listing_is_the_pinned_catalog(runner: CliRunner, repo: Path) -> None:
    """``--list`` prints the catalog and nothing else on stdout."""
    result = run(runner, repo, "demo", "--list")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == DEMO_PULLABLE


def test_the_listing_names_the_content_and_not_the_machinery(runner: CliRunner, repo: Path) -> None:
    """The two trees a facility rewrites are named; the build machinery is not."""
    listed = run(runner, repo, "demo", "--list").stdout.splitlines()

    assert "facility/knowledge/" in listed
    assert "agent/context/" in listed
    assert [entry for entry in listed if entry.endswith(".j2")] == []
    assert "__init__.py" not in listed


def test_the_listing_restricts_to_a_subtree_and_writes_nothing(
    runner: CliRunner, repo: Path
) -> None:
    """A path after the colon narrows the listing, and listing is read-only."""
    result = run(runner, repo, "demo:facility/knowledge", "--list")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        entry for entry in DEMO_PULLABLE if entry.startswith("facility/knowledge/")
    ]
    assert [path.name for path in repo.iterdir()] == ["profile.yml"]


def test_pulling_the_knowledge_base_writes_only_the_indexes(runner: CliRunner, repo: Path) -> None:
    """A plain pull lands the structure, says what it left, and rebuilds the indexes."""
    result = run(runner, repo, "demo:facility/knowledge")

    assert result.exit_code == 0, result.output
    written = _markdown(repo)
    assert len(written) == KNOWLEDGE_INDEX_FILES
    assert all(entry.endswith(f"/{_INDEX_NAME}") for entry in written)
    assert list(repo.rglob("*.png")) == []
    assert result.output.count("✓") == KNOWLEDGE_INDEX_FILES
    assert f"{KNOWLEDGE_FILES - KNOWLEDGE_INDEX_FILES} knowledge documents skipped" in result.output
    assert "--with-content" in result.output
    assert "rebuilt" in result.output


def test_pulling_the_knowledge_base_with_content_writes_every_document(
    runner: CliRunner, repo: Path
) -> None:
    """The flag turns the skeleton into the whole worked example."""
    result = run(runner, repo, "demo:facility/knowledge", "--with-content")

    assert result.exit_code == 0, result.output
    assert len(_markdown(repo)) == KNOWLEDGE_MARKDOWN_FILES
    figures = sorted(path.relative_to(repo).as_posix() for path in repo.rglob("*.png"))
    assert figures == ["facility/knowledge/devices/figures/bpm-button-pickups.png"]
    assert "skipped" not in result.output


def test_pulling_one_knowledge_document_names_the_flag_that_brings_it(
    runner: CliRunner, repo: Path
) -> None:
    """Asking for a single filtered document is a refusal, not a silent no-op."""
    result = run(runner, repo, "demo:facility/knowledge/devices/bpm.md")

    assert result.exit_code == 1
    assert "--with-content" in result.output
    assert _markdown(repo) == []


def test_pulling_a_bundled_server_keeps_its_package_marker(runner: CliRunner, repo: Path) -> None:
    """A bundled MCP server lands importable, marker and all."""
    result = run(runner, repo, "hello-world:agent/mcp_servers/example_server")

    assert result.exit_code == 0, result.output
    assert (repo / "agent" / "mcp_servers" / "example_server" / "__init__.py").is_file()


def test_an_existing_destination_stops_the_whole_pull(runner: CliRunner, repo: Path) -> None:
    """One file already here refuses the pull, and nothing else is written."""
    target = repo / "facility" / "knowledge" / _INDEX_NAME
    target.parent.mkdir(parents=True)
    target.write_text("ours", encoding="utf-8")

    result = run(runner, repo, "demo:facility/knowledge")

    assert result.exit_code == 1
    assert "--force" in result.output
    assert _markdown(repo) == ["facility/knowledge/index.md"]
    assert target.read_text(encoding="utf-8") == "ours"


def test_force_overwrites_and_keeps_a_file_the_template_lacks(
    runner: CliRunner, repo: Path, demo: PresetData
) -> None:
    """A pull replaces what it names and removes nothing it does not."""
    knowledge = repo / "facility" / "knowledge"
    (knowledge / "devices").mkdir(parents=True)
    (knowledge / _INDEX_NAME).write_text("ours", encoding="utf-8")
    (knowledge / "devices" / "local-note.md").write_text("kept", encoding="utf-8")

    result = run(runner, repo, "demo:facility/knowledge", "--force", "--with-content")

    assert result.exit_code == 0, result.output
    assert "(updated)" in result.output
    assert (knowledge / _INDEX_NAME).read_bytes() == (
        packaged(demo, "facility/knowledge/index.md")
    ).read_bytes()
    assert (knowledge / "devices" / "local-note.md").read_text(encoding="utf-8") == "kept"


def test_an_identical_file_is_reported_as_unchanged(
    runner: CliRunner, repo: Path, demo: PresetData
) -> None:
    """Re-pulling what is already there says so rather than claiming a write."""
    source = packaged(demo, "facility/knowledge/index.md")
    target = repo / "facility" / "knowledge" / _INDEX_NAME
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())

    result = run(runner, repo, "demo:facility/knowledge/index.md", "--force")

    assert result.exit_code == 0, result.output
    assert "(unchanged)" in result.output
    assert "✓" not in result.output


def test_a_symlinked_target_is_refused(runner: CliRunner, repo: Path, tmp_path: Path) -> None:
    """A link on the way to the target would land the copy somewhere else."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (repo / "facility").mkdir(parents=True)
    (repo / "facility" / "knowledge").symlink_to(elsewhere, target_is_directory=True)

    result = run(runner, repo, "demo:facility/knowledge", "--force")

    assert result.exit_code == 1
    assert "symlink" in result.output
    assert list(elsewhere.iterdir()) == []


def test_an_unknown_preset_lists_the_presets_that_exist(runner: CliRunner, repo: Path) -> None:
    """The correction is the list of names, so the message carries it."""
    result = run(runner, repo, "not-a-preset:facility", "--list")

    assert result.exit_code != 0
    presets = list_presets()
    assert presets
    for name in presets:
        assert name in result.output


def test_an_unknown_path_lists_the_top_level_entries(runner: CliRunner, repo: Path) -> None:
    """A mistyped path is a usage error naming where to look instead."""
    result = run(runner, repo, "demo:facility_knowlege")

    assert result.exit_code != 0
    assert "facility/" in result.output
    assert "agent/" in result.output
    assert _markdown(repo) == []


def test_outside_any_repo_the_failure_is_about_the_repo(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repo is resolved before the preset is, ``--list`` included."""
    elsewhere = tmp_path / "not-a-repo"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = runner.invoke(cli, ["scaffold", "pull", "demo", "--list"])

    assert result.exit_code == 1
    assert "No OSPREY deployment repo found" in result.output
    assert not any(elsewhere.iterdir())


def test_the_docstrings_name_only_commands_that_exist() -> None:
    """Every ``osprey`` chain in the verb's help resolves against the live CLI.

    The same forward check the emitted CI files get, turned on the help text an
    operator reads: a docstring is where a verb that was planned and never added
    survives longest, and ``--help`` is where it is believed.
    """
    for command in (scaffold, scaffold.get_command(click.Context(cli), "pull")):
        assert command is not None
        text = command.__doc__ or ""
        chains = named_commands(text)
        assert chains, f"no osprey command found in {command.name}'s help"
        for chain in sorted(chains):
            problem = unresolvable(chain)
            assert problem is None, f"{command.name}: {problem}"


def test_the_verb_is_registered_on_the_real_cli() -> None:
    """``osprey scaffold pull`` resolves the way an operator's shell walks it."""
    assert unresolvable(("scaffold", "pull")) is None
