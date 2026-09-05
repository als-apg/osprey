"""Tests for ``osprey scaffold pull`` and the packaged-template lookups it uses.

The lookups come first because everything else here depends on them: a preset
name has to become an app template, and an app template has to become a
directory on disk, before there is anything to copy out of the installation.
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from osprey.cli.build_profile import list_presets
from osprey.cli.main import cli
from osprey.cli.profile_cmd import (
    _app_template_root,
    _packaged_data_source,
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
from osprey.errors import BuildProfileError

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


@pytest.mark.parametrize("preset", ["hello-world", "control-assistant"])
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


def test_packaged_data_source_resolver_returns_the_data_subtree(
    manager: TemplateManager,
) -> None:
    """The data tree is still the ``data/`` directory inside the app template."""
    _name, data_bundle = _resolve_preset_bundle("control-assistant")

    data_source = _packaged_data_source(manager, data_bundle)

    assert data_source == _app_template_root(manager, data_bundle) / "data"
    assert data_source.is_dir()


def test_packaged_data_source_resolver_shares_the_absent_bundle_check(
    manager: TemplateManager,
) -> None:
    """One check, so the data tree fails on a missing template like everything else."""
    with pytest.raises(BuildProfileError, match="reinstall"):
        _packaged_data_source(manager, "no_such_app_template")


# ---------------------------------------------------------------------------
# Preset -> app template
# ---------------------------------------------------------------------------


def test_resolve_preset_bundle_returns_the_name_and_its_app_template() -> None:
    """The resolved profile's ``data_bundle`` is what comes back beside the name."""
    from osprey.cli.build_profile import resolve_build_profile

    expected, _preset_dir = resolve_build_profile(None, "control-assistant")

    assert _resolve_preset_bundle("control-assistant") == (
        "control-assistant",
        expected.data_bundle,
    )


def test_resolve_preset_bundle_normalizes_the_underscore_spelling() -> None:
    """Both CLI spellings of a preset resolve to the same answer."""
    assert _resolve_preset_bundle("control_assistant") == _resolve_preset_bundle(
        "control-assistant"
    )


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

#: Every path the control-assistant template offers, pinned exactly. Packaged
#: data is fixed for a release, so a diff here is a real change to what an
#: operator can pull — a bundle gaining or losing content, or the exclusions
#: drifting — and is meant to be read before it is accepted.
CONTROL_ASSISTANT_PULLABLE = [
    "data/",
    "data/ariel/",
    "data/benchmarks/",
    "data/benchmarks/cross_paradigm/",
    "data/benchmarks/cross_paradigm/queries/",
    "data/channel_databases/",
    "data/channel_databases/examples/",
    "data/channel_databases/tiers/",
    "data/channel_databases/tiers/tier1/",
    "data/channel_databases/tiers/tier3/",
    "data/facility_knowledge/",
    "data/facility_knowledge/devices/",
    "data/facility_knowledge/physics/",
    "data/facility_knowledge/procedures/",
    "data/facility_knowledge/references/",
    "data/facility_knowledge/subsystems/",
    "data/landing/",
    "data/lattice/",
    "data/raw/",
    "data/simulation/",
    "data/simulation/scenarios/",
    "data/simulation/scenarios/bpm-polarity/",
    "data/simulation/scenarios/nominal/",
    "data/simulation/scenarios/orm-dual-fault/",
    "data/simulation/scenarios/rf-thermal/",
    "data/simulation/scenarios/vacuum-burst/",
    "web-terminal-context/",
    "data/README.md",
    "data/ariel/README.md",
    "data/ariel/vocabulary.yml",
    "data/benchmarks/cross_paradigm/queries/tier1_queries.json",
    "data/benchmarks/cross_paradigm/queries/tier3_queries.json",
    "data/channel_databases/TEMPLATE_EXAMPLE.json",
    "data/channel_databases/examples/README.md",
    "data/channel_databases/examples/consecutive_instances.json",
    "data/channel_databases/examples/hierarchical_jlab_style.json",
    "data/channel_databases/examples/instance_first.json",
    "data/channel_databases/examples/mixed_hierarchy.json",
    "data/channel_databases/examples/optional_levels.json",
    "data/channel_databases/tiers/tier1/in_context.json",
    "data/channel_databases/tiers/tier3/hierarchical.json",
    "data/channel_databases/tiers/tier3/in_context.json",
    "data/channel_databases/tiers/tier3/middle_layer.json",
    "data/channel_limits.json",
    "data/demo_machine.ttl",
    "data/facility_knowledge/devices/bpm.md",
    "data/facility_knowledge/devices/index.md",
    "data/facility_knowledge/devices/ion-pump.md",
    "data/facility_knowledge/index.md",
    "data/facility_knowledge/physics/beam-current-calibration.md",
    "data/facility_knowledge/physics/index.md",
    "data/facility_knowledge/physics/quadrupole-scan.md",
    "data/facility_knowledge/procedures/index.md",
    "data/facility_knowledge/procedures/orbit-correction.md",
    "data/facility_knowledge/procedures/ps-startup.md",
    "data/facility_knowledge/procedures/pss-reset.md",
    "data/facility_knowledge/procedures/sample-scan.md",
    "data/facility_knowledge/procedures/vacuum-recovery.md",
    "data/facility_knowledge/references/epics-channel-access.md",
    "data/facility_knowledge/references/index.md",
    "data/facility_knowledge/references/safety-rules.md",
    "data/facility_knowledge/subsystems/experimental-stations.md",
    "data/facility_knowledge/subsystems/index.md",
    "data/facility_knowledge/subsystems/primary-source.md",
    "data/facility_knowledge/subsystems/pss.md",
    "data/facility_knowledge/subsystems/timing.md",
    "data/facility_knowledge/subsystems/transport-delivery.md",
    "data/facility_knowledge/subsystems/vacuum.md",
    "data/facility_ontology.json",
    "data/landing/working-safely.md",
    "data/lattice/als_u_ar.mat",
    "data/machine_state_channels.json",
    "data/raw/CSV_EXAMPLE.csv",
    "data/raw/address_list.csv",
    "data/simulation/machine.json",
    "data/simulation/scenarios/bpm-polarity/logbook.json",
    "data/simulation/scenarios/bpm-polarity/scenario.json",
    "data/simulation/scenarios/nominal/logbook.json",
    "data/simulation/scenarios/nominal/scenario.json",
    "data/simulation/scenarios/orm-dual-fault/scenario.json",
    "data/simulation/scenarios/rf-thermal/logbook.json",
    "data/simulation/scenarios/rf-thermal/scenario.json",
    "data/simulation/scenarios/vacuum-burst/scenario.json",
    "web-terminal-context/base.md",
]


@pytest.fixture
def control_assistant_root(manager: TemplateManager) -> Path:
    """The installed control-assistant app template."""
    _name, data_bundle = _resolve_preset_bundle("control-assistant")
    return _app_template_root(manager, data_bundle)


def _stage_template(root: Path, relative_paths: list[str]) -> None:
    """Write an empty file at each path under ``root``, creating parents."""
    for relative in relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def test_list_pullable_paths_pins_the_control_assistant_catalog(
    control_assistant_root: Path,
) -> None:
    """The whole listing, directories first and each group sorted."""
    assert list_pullable_paths(control_assistant_root) == CONTROL_ASSISTANT_PULLABLE


def test_list_pullable_paths_offers_the_content_an_operator_edits(
    control_assistant_root: Path,
) -> None:
    """The two trees a facility rewrites first are both named, machinery is not."""
    listed = list_pullable_paths(control_assistant_root)

    assert "data/facility_knowledge/" in listed
    assert "web-terminal-context/" in listed
    assert [entry for entry in listed if entry.endswith(".j2")] == []
    assert "__init__.py" not in listed


def test_list_pullable_paths_keeps_a_nested_package_marker(manager: TemplateManager) -> None:
    """A bundled MCP server is content, so its own ``__init__.py`` survives."""
    _name, data_bundle = _resolve_preset_bundle("hello-world")

    listed = list_pullable_paths(_app_template_root(manager, data_bundle))

    assert "mcp_servers/example_server/__init__.py" in listed
    assert "__init__.py" not in listed


def test_list_pullable_paths_drops_what_the_packaged_data_copy_drops(tmp_path: Path) -> None:
    """Build exhaust a checkout holds but a wheel never ships stays out of the listing."""
    root = tmp_path / "app_template"
    _stage_template(
        root,
        [
            "__init__.py",
            "config.yml.j2",
            "data/benchmarks/results/x.json",
            "data/benchmarks/cross_paradigm/queries/q.json",
            "data/results/kept.json",
        ],
    )

    listed = list_pullable_paths(root)

    assert "data/benchmarks/results/" not in listed
    assert "data/benchmarks/results/x.json" not in listed
    # Positional, not by name: a ``results/`` elsewhere in the tree is content.
    assert "data/results/kept.json" in listed
    assert listed == [
        "data/",
        "data/benchmarks/",
        "data/benchmarks/cross_paradigm/",
        "data/benchmarks/cross_paradigm/queries/",
        "data/results/",
        "data/benchmarks/cross_paradigm/queries/q.json",
        "data/results/kept.json",
    ]


def test_list_pullable_paths_restricts_to_a_subtree(control_assistant_root: Path) -> None:
    """A directory subtree yields itself and everything below it, and nothing else."""
    listed = list_pullable_paths(control_assistant_root, "data/facility_knowledge")

    assert "data/facility_knowledge/" in listed
    assert "data/facility_knowledge/procedures/orbit-correction.md" in listed
    assert all(entry.startswith("data/facility_knowledge/") for entry in listed)
    assert listed == [
        entry
        for entry in CONTROL_ASSISTANT_PULLABLE
        if entry.startswith("data/facility_knowledge/")
    ]


def test_list_pullable_paths_restricts_to_a_single_file(control_assistant_root: Path) -> None:
    """A file subtree is just that file."""
    assert list_pullable_paths(control_assistant_root, "data/README.md") == ["data/README.md"]


def test_list_pullable_paths_rejects_an_unknown_subtree(control_assistant_root: Path) -> None:
    """A miss names the top-level entries, which is what corrects the path."""
    with pytest.raises(ValueError) as excinfo:
        list_pullable_paths(control_assistant_root, "data/facility_knowlege")

    message = str(excinfo.value)
    assert "data/facility_knowlege" in message
    assert "data/" in message
    assert "web-terminal-context/" in message
    assert ".j2" not in message


# ---------------------------------------------------------------------------
# Pull plan
# ---------------------------------------------------------------------------

#: The knowledge base as the control-assistant template ships it, pinned so a
#: skeleton pull and a full pull are both anchored to real numbers: 23 Markdown
#: documents, of which 6 are the ``index.md`` files that carry the structure.
KNOWLEDGE_MARKDOWN_FILES = 23
KNOWLEDGE_INDEX_FILES = 6


def _by_action(actions: list[PullAction]) -> dict[str, list[PullAction]]:
    """The plan grouped by outcome, so a test can count one kind at a time."""
    grouped: dict[str, list[PullAction]] = {}
    for action in actions:
        grouped.setdefault(action.action, []).append(action)
    return grouped


def test_plan_pull_counts_match_the_packaged_knowledge_base(
    control_assistant_root: Path,
) -> None:
    """The pinned totals above are what the template actually ships."""
    listed = list_pullable_paths(control_assistant_root, "data/facility_knowledge")
    markdown = [entry for entry in listed if entry.endswith(".md")]

    assert len(markdown) == KNOWLEDGE_MARKDOWN_FILES
    assert (
        len([entry for entry in markdown if entry.endswith("/index.md")]) == KNOWLEDGE_INDEX_FILES
    )
    # Nothing else is in there, so "every file" and "every .md" are the same set.
    assert [entry for entry in listed if not entry.endswith("/")] == markdown


def test_plan_pull_of_the_knowledge_base_writes_only_the_indexes(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """A plain pull produces the structure and leaves the demo documents behind."""
    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge",
        force=False,
        with_content=False,
    )

    grouped = _by_action(plan)
    assert sorted(grouped) == ["skipped", "written"]
    assert len(grouped["written"]) == KNOWLEDGE_INDEX_FILES
    assert all(action.source.name == "index.md" for action in grouped["written"])
    assert len(grouped["skipped"]) == KNOWLEDGE_MARKDOWN_FILES - KNOWLEDGE_INDEX_FILES
    assert all("--with-content" in action.reason for action in grouped["skipped"])


def test_plan_pull_with_content_writes_every_knowledge_document(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """The flag turns the skeleton into the whole worked example."""
    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge",
        force=False,
        with_content=True,
    )

    assert sorted(_by_action(plan)) == ["written"]
    assert len(plan) == KNOWLEDGE_MARKDOWN_FILES
    assert all(action.target.suffix == ".md" for action in plan)


def test_plan_pull_mirrors_the_template_relative_path_under_the_repo(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """Targets are the source path again, rooted at the repo."""
    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge/index.md",
        force=False,
        with_content=False,
    )

    assert [action.target for action in plan] == [
        tmp_path / "data" / "facility_knowledge" / "index.md"
    ]
    assert plan[0].source == control_assistant_root / "data/facility_knowledge/index.md"


def test_plan_pull_refuses_a_single_filtered_knowledge_file(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """Skipping the only file asked for would do nothing, so it is refused instead."""
    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge/devices/bpm.md",
        force=False,
        with_content=False,
    )

    assert len(plan) == 1
    assert plan[0].action == "refused"
    assert "--with-content" in plan[0].reason


def test_plan_pull_refuses_an_existing_destination(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """A file already in the repo stops the pull and the reason names the flag."""
    _stage_template(tmp_path, ["data/facility_knowledge/index.md"])

    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge",
        force=False,
        with_content=False,
    )

    refused = _by_action(plan)["refused"]
    assert [action.target for action in refused] == [
        tmp_path / "data" / "facility_knowledge" / "index.md"
    ]
    assert "--force" in refused[0].reason


def test_plan_pull_updates_an_existing_destination_under_force(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """With the flag, a differing file is an update rather than a refusal."""
    _stage_template(tmp_path, ["data/facility_knowledge/index.md"])

    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge",
        force=True,
        with_content=False,
    )

    grouped = _by_action(plan)
    assert "refused" not in grouped
    assert [action.target for action in grouped["updated"]] == [
        tmp_path / "data" / "facility_knowledge" / "index.md"
    ]
    assert len(grouped["written"]) == KNOWLEDGE_INDEX_FILES - 1


def test_plan_pull_reports_an_identical_destination_as_unchanged(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """Re-pulling what is already there is a no-op, reported as one."""
    source = control_assistant_root / "data/facility_knowledge/index.md"
    target = tmp_path / "data" / "facility_knowledge" / "index.md"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())

    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge/index.md",
        force=True,
        with_content=False,
    )

    assert [(action.action, action.target) for action in plan] == [("unchanged", target)]


def test_plan_pull_refuses_a_symlinked_target_directory(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """A link on the way to the target would land the copy somewhere else."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    repo_root = tmp_path / "repo"
    (repo_root / "data").mkdir(parents=True)
    (repo_root / "data" / "facility_knowledge").symlink_to(elsewhere, target_is_directory=True)

    plan = plan_pull(
        control_assistant_root,
        repo_root,
        "data/facility_knowledge",
        force=True,
        with_content=True,
    )

    assert sorted(_by_action(plan)) == ["refused"]
    assert len(plan) == KNOWLEDGE_MARKDOWN_FILES
    assert "symlink" in plan[0].reason
    # Planning is pure: nothing reached the directory the link points at.
    assert list(elsewhere.iterdir()) == []


def test_plan_pull_refuses_a_directory_where_the_file_goes(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """A kind mismatch is never resolved on the facility's behalf."""
    (tmp_path / "data" / "facility_knowledge" / "index.md").mkdir(parents=True)

    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge/index.md",
        force=True,
        with_content=False,
    )

    assert plan[0].action == "refused"
    assert "directory" in plan[0].reason


def test_plan_pull_refuses_a_file_where_a_directory_is_needed(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """The mismatch counts anywhere between the repo root and the target."""
    _stage_template(tmp_path, ["data/facility_knowledge/devices"])

    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge/devices/bpm.md",
        force=True,
        with_content=True,
    )

    assert plan[0].action == "refused"
    assert "data/facility_knowledge/devices" in plan[0].reason


def test_plan_pull_keeps_a_nested_package_marker(manager: TemplateManager, tmp_path: Path) -> None:
    """A bundled MCP server pulls with its own ``__init__.py``."""
    _name, data_bundle = _resolve_preset_bundle("hello-world")

    plan = plan_pull(
        _app_template_root(manager, data_bundle),
        tmp_path,
        "mcp_servers/example_server",
        force=False,
        with_content=False,
    )

    marker = tmp_path / "mcp_servers" / "example_server" / "__init__.py"
    assert [action.action for action in plan if action.target == marker] == ["written"]


def test_plan_pull_rejects_an_unknown_path(control_assistant_root: Path, tmp_path: Path) -> None:
    """An unknown path fails the same way the listing does, with the same message."""
    with pytest.raises(ValueError, match="data/facility_knowlege"):
        plan_pull(
            control_assistant_root,
            tmp_path,
            "data/facility_knowlege",
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
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """A skeleton pull lands the structure and not one demo document."""
    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge",
        force=False,
        with_content=False,
    )

    applied = apply_pull(plan, repo_root=tmp_path, with_content=False)

    written = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.md"))
    assert len(written) == KNOWLEDGE_INDEX_FILES
    assert all(entry.endswith(f"/{_INDEX_NAME}") for entry in written)
    assert len(applied) == KNOWLEDGE_INDEX_FILES


def test_apply_pull_leaves_no_demo_document_title_in_a_pulled_index(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """The rebuilt indexes name what is on disk, so the demo concepts are gone."""
    knowledge_source = control_assistant_root / "data" / "facility_knowledge"
    demo_titles = _packaged_demo_titles(knowledge_source)
    assert "Beam Position Monitor (BPM)" in demo_titles

    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge",
        force=False,
        with_content=False,
    )
    apply_pull(plan, repo_root=tmp_path, with_content=False)

    knowledge_target = tmp_path / "data" / "facility_knowledge"
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


def test_apply_pull_with_content_writes_the_packaged_knowledge_base_verbatim(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """With the documents in hand the packaged indexes are already true, so they stand."""
    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge",
        force=False,
        with_content=True,
    )

    applied = apply_pull(plan, repo_root=tmp_path, with_content=True)

    written = sorted(tmp_path.rglob("*.md"))
    assert len(applied) == KNOWLEDGE_MARKDOWN_FILES
    assert len(written) == KNOWLEDGE_MARKDOWN_FILES
    for path in written:
        relative = path.relative_to(tmp_path)
        assert path.read_bytes() == (control_assistant_root / relative).read_bytes()


def test_apply_pull_writes_nothing_when_the_plan_holds_a_refusal(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """One refusal stops the whole pull, so there is no half-applied copy to undo."""
    _stage_template(tmp_path, ["data/facility_knowledge/index.md"])
    before = _tree(tmp_path)

    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge",
        force=False,
        with_content=False,
    )
    assert "refused" in _by_action(plan)

    returned = apply_pull(plan, repo_root=tmp_path, with_content=False)

    assert returned == plan
    assert _tree(tmp_path) == before


def test_apply_pull_overwrites_under_force_and_keeps_a_file_the_template_lacks(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """A pull replaces what it names and removes nothing it does not."""
    _stage_template(
        tmp_path,
        ["data/facility_knowledge/index.md", "data/facility_knowledge/devices/local-note.md"],
    )

    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge",
        force=True,
        with_content=True,
    )
    apply_pull(plan, repo_root=tmp_path, with_content=True)

    target = tmp_path / "data" / "facility_knowledge" / "index.md"
    assert (
        target.read_bytes()
        == (control_assistant_root / "data/facility_knowledge/index.md").read_bytes()
    )
    assert (tmp_path / "data" / "facility_knowledge" / "devices" / "local-note.md").is_file()


def test_apply_pull_writes_a_nested_package_marker(
    manager: TemplateManager, tmp_path: Path
) -> None:
    """A bundled MCP server lands importable, marker and all."""
    _name, data_bundle = _resolve_preset_bundle("hello-world")
    app_root = _app_template_root(manager, data_bundle)

    plan = plan_pull(
        app_root, tmp_path, "mcp_servers/example_server", force=False, with_content=False
    )
    apply_pull(plan, repo_root=tmp_path, with_content=False)

    marker = tmp_path / "mcp_servers" / "example_server" / "__init__.py"
    assert marker.is_file()
    assert marker.read_bytes() == (app_root / "mcp_servers/example_server/__init__.py").read_bytes()


def test_apply_pull_returns_the_actions_it_applied(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """The command reports from what came back, and those are the planned objects."""
    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge",
        force=False,
        with_content=False,
    )

    applied = apply_pull(plan, repo_root=tmp_path, with_content=False)

    assert [action.action for action in applied] == ["written"] * KNOWLEDGE_INDEX_FILES
    assert all(any(action is planned for planned in plan) for action in applied)
    assert _by_action(plan)["skipped"]  # the skipped documents were quietly left alone


def test_apply_pull_leaves_an_unchanged_target_alone(
    control_assistant_root: Path, tmp_path: Path
) -> None:
    """Re-pulling an identical file writes nothing and rebuilds nothing."""
    source = control_assistant_root / "data/facility_knowledge/index.md"
    target = tmp_path / "data" / "facility_knowledge" / "index.md"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())

    plan = plan_pull(
        control_assistant_root,
        tmp_path,
        "data/facility_knowledge/index.md",
        force=True,
        with_content=False,
    )

    assert apply_pull(plan, repo_root=tmp_path, with_content=False) == []
    assert target.read_bytes() == source.read_bytes()


# ---------------------------------------------------------------------------
# The verb
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


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
    result = run(runner, repo, "control-assistant", "--list")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == CONTROL_ASSISTANT_PULLABLE


def test_the_listing_names_the_content_and_not_the_machinery(runner: CliRunner, repo: Path) -> None:
    """The two trees a facility rewrites are named; the build machinery is not."""
    listed = run(runner, repo, "control-assistant", "--list").stdout.splitlines()

    assert "data/facility_knowledge/" in listed
    assert "web-terminal-context/" in listed
    assert [entry for entry in listed if entry.endswith(".j2")] == []
    assert "__init__.py" not in listed


def test_the_listing_restricts_to_a_subtree_and_writes_nothing(
    runner: CliRunner, repo: Path
) -> None:
    """A path after the colon narrows the listing, and listing is read-only."""
    result = run(runner, repo, "control-assistant:data/facility_knowledge", "--list")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        entry
        for entry in CONTROL_ASSISTANT_PULLABLE
        if entry.startswith("data/facility_knowledge/")
    ]
    assert [path.name for path in repo.iterdir()] == ["profile.yml"]


def test_pulling_the_knowledge_base_writes_only_the_indexes(runner: CliRunner, repo: Path) -> None:
    """A plain pull lands the structure, says what it left, and rebuilds the indexes."""
    result = run(runner, repo, "control-assistant:data/facility_knowledge")

    assert result.exit_code == 0, result.output
    written = _markdown(repo)
    assert len(written) == KNOWLEDGE_INDEX_FILES
    assert all(entry.endswith(f"/{_INDEX_NAME}") for entry in written)
    assert result.output.count("✓") == KNOWLEDGE_INDEX_FILES
    assert (
        f"{KNOWLEDGE_MARKDOWN_FILES - KNOWLEDGE_INDEX_FILES} knowledge documents skipped"
        in result.output
    )
    assert "--with-content" in result.output
    assert "rebuilt" in result.output


def test_pulling_the_knowledge_base_with_content_writes_every_document(
    runner: CliRunner, repo: Path
) -> None:
    """The flag turns the skeleton into the whole worked example."""
    result = run(runner, repo, "control-assistant:data/facility_knowledge", "--with-content")

    assert result.exit_code == 0, result.output
    assert len(_markdown(repo)) == KNOWLEDGE_MARKDOWN_FILES
    assert "skipped" not in result.output


def test_pulling_one_knowledge_document_names_the_flag_that_brings_it(
    runner: CliRunner, repo: Path
) -> None:
    """Asking for a single filtered document is a refusal, not a silent no-op."""
    result = run(runner, repo, "control-assistant:data/facility_knowledge/devices/bpm.md")

    assert result.exit_code == 1
    assert "--with-content" in result.output
    assert _markdown(repo) == []


def test_pulling_a_bundled_server_keeps_its_package_marker(runner: CliRunner, repo: Path) -> None:
    """A bundled MCP server lands importable, marker and all."""
    result = run(runner, repo, "hello-world:mcp_servers/example_server")

    assert result.exit_code == 0, result.output
    assert (repo / "mcp_servers" / "example_server" / "__init__.py").is_file()


def test_an_existing_destination_stops_the_whole_pull(runner: CliRunner, repo: Path) -> None:
    """One file already here refuses the pull, and nothing else is written."""
    target = repo / "data" / "facility_knowledge" / _INDEX_NAME
    target.parent.mkdir(parents=True)
    target.write_text("ours", encoding="utf-8")

    result = run(runner, repo, "control-assistant:data/facility_knowledge")

    assert result.exit_code == 1
    assert "--force" in result.output
    assert _markdown(repo) == ["data/facility_knowledge/index.md"]
    assert target.read_text(encoding="utf-8") == "ours"


def test_force_overwrites_and_keeps_a_file_the_template_lacks(
    runner: CliRunner, repo: Path, control_assistant_root: Path
) -> None:
    """A pull replaces what it names and removes nothing it does not."""
    knowledge = repo / "data" / "facility_knowledge"
    (knowledge / "devices").mkdir(parents=True)
    (knowledge / _INDEX_NAME).write_text("ours", encoding="utf-8")
    (knowledge / "devices" / "local-note.md").write_text("kept", encoding="utf-8")

    result = run(
        runner, repo, "control-assistant:data/facility_knowledge", "--force", "--with-content"
    )

    assert result.exit_code == 0, result.output
    assert "(updated)" in result.output
    assert (knowledge / _INDEX_NAME).read_bytes() == (
        control_assistant_root / "data/facility_knowledge/index.md"
    ).read_bytes()
    assert (knowledge / "devices" / "local-note.md").read_text(encoding="utf-8") == "kept"


def test_an_identical_file_is_reported_as_unchanged(
    runner: CliRunner, repo: Path, control_assistant_root: Path
) -> None:
    """Re-pulling what is already there says so rather than claiming a write."""
    source = control_assistant_root / "data/facility_knowledge/index.md"
    target = repo / "data" / "facility_knowledge" / _INDEX_NAME
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())

    result = run(runner, repo, "control-assistant:data/facility_knowledge/index.md", "--force")

    assert result.exit_code == 0, result.output
    assert "(unchanged)" in result.output
    assert "✓" not in result.output


def test_a_symlinked_target_is_refused(runner: CliRunner, repo: Path, tmp_path: Path) -> None:
    """A link on the way to the target would land the copy somewhere else."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (repo / "data").mkdir()
    (repo / "data" / "facility_knowledge").symlink_to(elsewhere, target_is_directory=True)

    result = run(runner, repo, "control-assistant:data/facility_knowledge", "--force")

    assert result.exit_code == 1
    assert "symlink" in result.output
    assert list(elsewhere.iterdir()) == []


def test_an_unknown_preset_lists_the_presets_that_exist(runner: CliRunner, repo: Path) -> None:
    """The correction is the list of names, so the message carries it."""
    result = run(runner, repo, "not-a-preset:data", "--list")

    assert result.exit_code != 0
    presets = list_presets()
    assert presets
    for name in presets:
        assert name in result.output


def test_an_unknown_path_lists_the_top_level_entries(runner: CliRunner, repo: Path) -> None:
    """A mistyped path is a usage error naming where to look instead."""
    result = run(runner, repo, "control-assistant:data/facility_knowlege")

    assert result.exit_code != 0
    assert "data/" in result.output
    assert "web-terminal-context/" in result.output
    assert _markdown(repo) == []


def test_outside_any_repo_the_failure_is_about_the_repo(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repo is resolved before the preset is, ``--list`` included."""
    elsewhere = tmp_path / "not-a-repo"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = runner.invoke(cli, ["scaffold", "pull", "control-assistant", "--list"])

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
