"""Preset updates: what a preset changed since the repo last took it, and nothing else.

``osprey init`` saves what the presets produced under ``.osprey/preset-base/``.
The delta between that saved copy and what the presets produce today is what
they changed. ``osprey init --revalidate`` applies it where the operator left
the preset's value alone and rewrites the saved copy; ``osprey validate`` and
``osprey build`` only say, in one line, that there is something to apply, and
never refuse an edit.

A preset change is staged here by editing the saved copy back to what an older
preset would have said: the delta is the same either way, and the installed
preset stays the one the package ships.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile_updates import (
    APP_PROVENANCE_FILENAME,
    PRESET_BASE_DIR,
    Entry,
    Member,
    PresetRecordError,
    PresetUpdateState,
    covered_files,
    flatten,
    preset_delta,
    preset_update_status,
    read_saved_copy,
)
from osprey.cli.init_cmd import init
from osprey.cli.main import cli
from osprey.cli.set_cmd import set as set_cmd
from osprey.cli.validate_cmd import validate
from osprey.deployment import staleness

PROFILE = "profile.yml"
ROOT_BASE = "presets/demo.yml"
#: The one line validate prints when the preset has changed.
NOTE = "The demo preset has changed since this repo last took it"
PANEL = (
    "  - jupyter         # JUPYTER tab, JupyterLab with kernels that follow the terminal session\n"
)
THEME = "  web.theme: light\n"


@pytest.fixture(scope="module")
def materialized(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real materialization to edit copies of."""
    root = tmp_path_factory.mktemp("preset-updates") / "demo"
    result = CliRunner().invoke(init, [str(root), "--preset", "demo", "--no-git"])
    assert result.exit_code == 0, result.output
    return root


@pytest.fixture
def repo(materialized: Path, tmp_path: Path) -> Path:
    target = tmp_path / "demo"
    shutil.copytree(materialized, target)
    return target


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{old!r} must occur exactly once in {path}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _older_preset(repo: Path, key: str, edit: Callable[[dict[str, Any]], None]) -> None:
    """Edit the saved copy at *key* into what an older preset would have said."""
    path = repo / PRESET_BASE_DIR / key
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(document)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _revalidate(repo: Path, *extra: str):
    result = CliRunner().invoke(init, [str(repo), "--revalidate", *extra], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def _validate(repo: Path):
    result = CliRunner().invoke(validate, ["--repo", str(repo)])
    assert result.exit_code == 0, result.output
    return result


def _profile(repo: Path) -> dict[str, Any]:
    return yaml.safe_load((repo / PROFILE).read_text(encoding="utf-8"))


def _without_panel(document: dict[str, Any]) -> None:
    document["web_panels"].remove("jupyter")


# ── init writes the saved copy ───────────────────────────────────────────────


def test_init_saves_what_each_covered_file_came_from(materialized: Path) -> None:
    saved = read_saved_copy(materialized)

    assert {ROOT_BASE, "providers.yml", "triggers.yml", "tiers/readonly.yml"} <= set(saved)
    # Plain data: no header, no comments, none of the keys the repo writes itself.
    text = (materialized / PRESET_BASE_DIR / ROOT_BASE).read_text(encoding="utf-8")
    assert "#" not in text.split("\n", 1)[0]
    assert not {"name", "project_name", "provenance", "data"} & set(saved[ROOT_BASE])
    assert (materialized / PRESET_BASE_DIR / "README").is_file()


# ── validate and build: one line, never a refusal ────────────────────────────


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (THEME, ""),
        (PANEL, ""),
        (THEME, f"{THEME}  web.no_such_knob: 1\n"),
        (THEME, "  web.theme: dark\n"),
    ],
    ids=["a-dropped-key", "a-dropped-member", "an-extra-key", "a-changed-value"],
)
def test_an_edit_is_neither_refused_nor_mentioned(repo: Path, old: str, new: str) -> None:
    _edit(repo / PROFILE, old, new)

    result = _validate(repo)

    assert "Profile is valid" in result.output
    assert "preset" not in result.output.lower().replace("preset demo", "")
    assert "DEVIATION" not in result.output


def test_drift_is_not_an_option_any_more(repo: Path) -> None:
    result = CliRunner().invoke(validate, ["--repo", str(repo), "--drift=warn"])

    assert result.exit_code == 2
    assert "No such option" in result.output


def test_set_knob_then_validate_green(repo: Path) -> None:
    setting = CliRunner().invoke(set_cmd, ["--repo", str(repo), "config.web.theme=dark"])
    assert setting.exit_code == 0, setting.output

    assert "Profile is valid" in _validate(repo).output


def test_a_changed_preset_is_one_line_pointing_at_revalidate(repo: Path) -> None:
    _older_preset(repo, ROOT_BASE, _without_panel)

    result = _validate(repo)

    assert NOTE in result.output
    assert "osprey init --revalidate" in result.output
    assert "jupyter" not in result.output


def test_build_prints_the_same_line_and_lists_nothing(repo: Path) -> None:
    _older_preset(repo, ROOT_BASE, _without_panel)
    _edit(repo / PROFILE, THEME, "  web.theme: dark\n")
    args = ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"]

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0, result.output
    assert NOTE in result.output
    assert "web.theme" not in result.output


def test_a_new_osprey_version_alone_changes_nothing(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.cli import build_profile_emit

    monkeypatch.setattr(build_profile_emit, "__version__", "2099.1.0")
    before = (repo / PROFILE).read_text(encoding="utf-8")

    assert NOTE not in _validate(repo).output
    assert "already current" in _revalidate(repo).output
    assert (repo / PROFILE).read_text(encoding="utf-8") == before


# ── --revalidate applies what the preset changed ─────────────────────────────


def test_a_heavily_edited_profile_with_an_unchanged_preset_is_left_alone(repo: Path) -> None:
    _edit(repo / PROFILE, THEME, "  web.theme: dark\n")
    _edit(repo / PROFILE, PANEL, "")
    _edit(repo / PROFILE, "  cli.theme: default\n", "")
    before = (repo / PROFILE).read_text(encoding="utf-8")

    assert NOTE not in _validate(repo).output
    result = _revalidate(repo)

    assert "already current" in result.output
    assert (repo / PROFILE).read_text(encoding="utf-8") == before


def test_a_panel_the_preset_added_is_added(repo: Path) -> None:
    _older_preset(repo, ROOT_BASE, _without_panel)
    _edit(repo / PROFILE, PANEL, "")

    result = _revalidate(repo)

    assert "jupyter" in _profile(repo)["web_panels"]
    assert "profile.yml: added web_panels: jupyter" in result.output
    assert NOTE not in _validate(repo).output


def test_a_panel_the_preset_added_that_the_profile_has_is_left_alone(repo: Path) -> None:
    _older_preset(repo, ROOT_BASE, _without_panel)
    before = (repo / PROFILE).read_text(encoding="utf-8")

    _revalidate(repo)

    assert (repo / PROFILE).read_text(encoding="utf-8") == before
    assert NOTE not in _validate(repo).output


@pytest.mark.parametrize(("kept", "removed"), [("1", True), ("2", False)])
def test_a_key_the_preset_removed_goes_only_if_the_profile_kept_it(
    repo: Path, kept: str, removed: bool
) -> None:
    _older_preset(repo, ROOT_BASE, lambda d: d["config"].update({"web.retired_knob": 1}))
    _edit(repo / PROFILE, THEME, f"{THEME}  web.retired_knob: {kept}\n")

    _revalidate(repo)

    assert ("web.retired_knob" not in _profile(repo)["config"]) is removed


@pytest.mark.parametrize(
    ("yours", "after"), [("dark", "light"), ("high-contrast", "high-contrast")]
)
def test_a_changed_default_moves_only_if_the_profile_kept_the_old_one(
    repo: Path, yours: str, after: str
) -> None:
    _older_preset(repo, ROOT_BASE, lambda d: d["config"].update({"web.theme": "dark"}))
    _edit(repo / PROFILE, THEME, f"  web.theme: {yours}\n")

    _revalidate(repo)

    assert _profile(repo)["config"]["web.theme"] == after
    # Taken once: the saved copy now says what the preset says.
    assert "already current" in _revalidate(repo).output


def test_a_named_entry_the_preset_added_leaves_the_operators_own_entry_alone(repo: Path) -> None:
    def older(document: dict[str, Any]) -> None:
        document["agent_record"]["readers"] = [{"name": "dave", "role": "viewer"}]

    _older_preset(repo, ROOT_BASE, older)
    _edit(
        repo / PROFILE,
        "  readers:\n    - name: carol\n",
        "  readers:\n    - name: dave\n      role: viewer\n      extra: mine\n",
    )

    _revalidate(repo)

    readers = _profile(repo)["agent_record"]["readers"]
    assert {"name": "carol"} in readers
    # Removed by the preset, but the operator changed it: theirs to keep.
    assert {"name": "dave", "role": "viewer", "extra": "mine"} in readers


def test_a_step_the_preset_added_is_inserted_at_the_presets_position(repo: Path) -> None:
    _older_preset(repo, ROOT_BASE, lambda d: d.pop("lifecycle"))
    _edit(
        repo / PROFILE,
        "    - name: canopy-payload\n      timeout: 30\n",
        "    - name: mine\n      timeout: 30\n",
    )

    _revalidate(repo)

    names = [step["name"] for step in _profile(repo)["lifecycle"]["pre_build"]]
    assert names == ["canopy-payload", "mine"]


def test_dotted_and_nested_spellings_compare_equal() -> None:
    dotted = {"config": {"web.theme": "light", "approval.tools.write": "always"}}
    nested = {"config": {"web": {"theme": "light"}, "approval": {"tools.write": "always"}}}

    assert flatten(dotted) == flatten(nested)
    assert preset_delta(dotted, nested) == []


def test_lists_compare_by_member_and_by_name() -> None:
    flat = flatten({"web_panels": ["a", "b"], "readers": [{"name": "x", "role": "r"}]})

    assert flat[("web_panels", Member("a"))] is True
    assert flat[("readers", Entry("x"), "role")] == "r"
    assert flatten({"web_panels": ["b", "a"]}) == flatten({"web_panels": ["a", "b"]})


def test_a_key_moved_into_a_tier_counts_as_present(repo: Path) -> None:
    _older_preset(repo, ROOT_BASE, lambda d: d["config"].pop("web.theme"))
    _edit(repo / PROFILE, THEME, "")
    _edit(
        repo / "tiers" / "readonly.yml",
        "  control_system.writes: never\n",
        "  control_system.writes: never\n  web.theme: light\n",
    )

    _revalidate(repo)

    assert "web.theme" not in _profile(repo)["config"]


def test_a_tier_copied_under_a_new_name_pairs_with_its_bundled_source(repo: Path) -> None:
    tiers = repo / "tiers"
    (tiers / "logbook.yml").rename(tiers / "night-shift.yml")

    paired = {item.rel: item.key for item in covered_files(repo)}
    assert paired["tiers/night-shift.yml"] == "tiers/logbook.yml"

    _older_preset(repo, "tiers/logbook.yml", lambda d: d["skills"].remove("session-report"))
    _edit(tiers / "night-shift.yml", "  - session-report", "  - session-reports-removed")

    _revalidate(repo)

    skills = yaml.safe_load((tiers / "night-shift.yml").read_text(encoding="utf-8"))["skills"]
    assert "session-report" in skills


def test_a_change_dropped_before_commit_is_not_offered_again(repo: Path) -> None:
    _older_preset(repo, ROOT_BASE, _without_panel)
    _edit(repo / PROFILE, PANEL, "")
    declined = (repo / PROFILE).read_text(encoding="utf-8")
    _revalidate(repo)
    # The operator drops that hunk and commits the rest, the saved copy included.
    (repo / PROFILE).write_text(declined, encoding="utf-8")

    assert NOTE not in _validate(repo).output
    assert "already current" in _revalidate(repo).output
    assert "jupyter" not in _profile(repo)["web_panels"]


def test_a_list_changed_on_both_sides_is_left_for_you_once(repo: Path) -> None:
    """A list of mappings with no ``name`` is one value: the operator decides."""

    def older(document: dict[str, Any]) -> None:
        document["watchdogs"]["guard-circumvention"]["options"].append({"id": "older"})

    _older_preset(repo, ROOT_BASE, older)
    _edit(
        repo / PROFILE,
        "questions about the guards themselves.\n        flag: false\n",
        "questions about the guards themselves.\n        flag: true\n",
    )
    before = (repo / PROFILE).read_text(encoding="utf-8")

    first = _revalidate(repo)

    assert "Preset changes left for you" in first.output
    assert "watchdogs.guard-circumvention.options" in first.output
    assert (repo / PROFILE).read_text(encoding="utf-8") == before
    assert "Preset changes left for you" not in _revalidate(repo).output


def test_two_checkouts_of_one_profile_get_the_same_answer(
    materialized: Path, tmp_path: Path
) -> None:
    outputs = []
    for name in ("assistant", "assistant-wt"):
        copy = tmp_path / name
        shutil.copytree(materialized, copy)
        _older_preset(copy, ROOT_BASE, _without_panel)
        _edit(copy / PROFILE, PANEL, "")
        result = _revalidate(copy)
        outputs.append((result.output.replace(str(copy), "<repo>"), _profile(copy)))

    assert outputs[0] == outputs[1]


# ── --preset, --force, legacy repos ──────────────────────────────────────────


def test_revalidate_with_another_preset_saves_that_presets_output(repo: Path) -> None:
    _revalidate(repo, "--preset", "hello-world")

    saved = read_saved_copy(repo)
    assert "presets/hello-world.yml" in saved
    assert ROOT_BASE not in saved
    assert _profile(repo)["provenance"]["preset"] == "hello-world"


def test_init_force_holds_the_saved_copy_aside_and_restores_it(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _older_preset(repo, ROOT_BASE, _without_panel)
    held = (repo / PRESET_BASE_DIR / ROOT_BASE).read_text(encoding="utf-8")

    def explode(*_args: object, **_kwargs: object) -> list[str]:
        raise OSError("No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr("osprey.cli.profile_cmd._write_secret_channel", explode)
        failed = CliRunner().invoke(init, [str(repo), "--preset", "demo", "--force", "--no-git"])
    assert failed.exit_code != 0
    assert (repo / PRESET_BASE_DIR / ROOT_BASE).read_text(encoding="utf-8") == held

    forced = CliRunner().invoke(init, [str(repo), "--preset", "demo", "--force", "--no-git"])
    assert forced.exit_code == 0, forced.output
    assert "jupyter" in read_saved_copy(repo)[ROOT_BASE]["web_panels"]


def test_a_repo_from_before_the_saved_copy_with_its_preset_unchanged(repo: Path) -> None:
    shutil.rmtree(repo / ".osprey")
    _edit(repo / PROFILE, PANEL, "")
    before = (repo / PROFILE).read_text(encoding="utf-8")

    assert "predates preset tracking" in _validate(repo).output
    _revalidate(repo)

    # The hash matches: today's preset is the starting point, so the panel the
    # operator dropped is not put back.
    assert (repo / PROFILE).read_text(encoding="utf-8") == before
    assert ROOT_BASE in read_saved_copy(repo)
    assert "predates" not in _validate(repo).output


def test_a_repo_from_before_the_saved_copy_with_its_preset_moved_on(repo: Path) -> None:
    shutil.rmtree(repo / ".osprey")
    _edit(repo / PROFILE, "  preset_hash: sha256:", "  preset_hash: sha256:0ld")
    _edit(repo / PROFILE, THEME, "")

    result = _revalidate(repo)

    assert _profile(repo)["config"]["web.theme"] == "light"
    assert "predates preset tracking" in result.output
    assert "added config.web.theme" in result.output
    assert NOTE not in _validate(repo).output


# ── a saved copy that does not parse ─────────────────────────────────────────

CONFLICT = "<<<<<<< HEAD\nwatch: []\n=======\nwatch: [a]\n>>>>>>> theirs\n"
TRIGGERS_BASE = f"{PRESET_BASE_DIR.as_posix()}/triggers.yml"


def test_an_unreadable_saved_copy_is_a_note_naming_the_record_not_a_refusal(repo: Path) -> None:
    (repo / TRIGGERS_BASE).write_text(CONFLICT, encoding="utf-8")

    result = _validate(repo)

    assert "Profile is valid" in result.output
    assert TRIGGERS_BASE in result.output
    assert "Invalid YAML in triggers.yml" not in result.output
    assert "osprey init --revalidate" in result.output
    assert NOTE not in result.output


def test_build_goes_on_past_an_unreadable_saved_copy(repo: Path) -> None:
    (repo / TRIGGERS_BASE).write_text(CONFLICT, encoding="utf-8")
    args = ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"]

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0, result.output
    assert TRIGGERS_BASE in result.output
    assert "Build error" not in result.output


def test_the_saved_copy_loader_names_the_record_it_could_not_read(repo: Path) -> None:
    (repo / TRIGGERS_BASE).write_text(CONFLICT, encoding="utf-8")

    with pytest.raises(PresetRecordError) as caught:
        read_saved_copy(repo)

    assert caught.value.rel == TRIGGERS_BASE
    assert f"Invalid YAML in {TRIGGERS_BASE}" in str(caught.value)
    assert "Invalid YAML in triggers.yml" not in str(caught.value)


def test_an_unreadable_application_sidecar_is_a_note_too(repo: Path) -> None:
    sidecar = repo / "apps" / "knob" / APP_PROVENANCE_FILENAME
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(CONFLICT, encoding="utf-8")

    status = preset_update_status(repo / PROFILE)

    assert status.state is PresetUpdateState.UNREADABLE
    assert status.note is not None
    assert f"apps/knob/{APP_PROVENANCE_FILENAME}" in status.note
    assert "\n" not in status.note


# ── the state behind the note ────────────────────────────────────────────────


def _presets_changed(repo: Path) -> bool:
    manifest = {"build_args": {"profile_path_abs": str(repo / PROFILE)}}
    return staleness._presets_changed(manifest)


def test_only_a_changed_preset_sends_osprey_up_to_revalidate(repo: Path) -> None:
    assert preset_update_status(repo / PROFILE).state is PresetUpdateState.CURRENT
    assert not _presets_changed(repo)

    _older_preset(repo, ROOT_BASE, _without_panel)

    status = preset_update_status(repo / PROFILE)
    assert status.state is PresetUpdateState.CHANGED
    assert status.note is not None and NOTE in status.note
    assert _presets_changed(repo)


def test_a_preset_this_osprey_does_not_bundle_has_nothing_to_take(repo: Path) -> None:
    _edit(repo / PROFILE, "  preset: demo\n", "  preset: elsewhere\n")

    status = preset_update_status(repo / PROFILE)

    assert status.state is PresetUpdateState.NOT_BUNDLED
    assert status.note is not None and "does not bundle" in status.note
    assert not _presets_changed(repo)


def test_a_repo_from_before_the_saved_copy_has_nothing_to_take(repo: Path) -> None:
    shutil.rmtree(repo / ".osprey")

    status = preset_update_status(repo / PROFILE)

    assert status.state is PresetUpdateState.UNTRACKED
    assert status.note is not None and "predates preset tracking" in status.note
    assert not _presets_changed(repo)


@pytest.mark.parametrize(
    ("rel", "stamp"),
    [
        (PROFILE, "preset_hash"),
        (PROFILE, "providers_hash"),
        ("tiers/admin.yml", "preset_hash"),
    ],
)
def test_a_repo_from_before_the_saved_copy_whose_preset_moved_on_has_changes_to_take(
    repo: Path, rel: str, stamp: str
) -> None:
    shutil.rmtree(repo / ".osprey")
    _edit(repo / rel, f"  {stamp}: sha256:", f"  {stamp}: sha256:0ld")

    status = preset_update_status(repo / PROFILE)

    assert status.state is PresetUpdateState.UNTRACKED_CHANGED
    assert status.note is not None and "predates preset tracking" in status.note
    assert _presets_changed(repo)


def test_an_unreadable_saved_copy_has_nothing_to_take(repo: Path) -> None:
    (repo / TRIGGERS_BASE).write_text(CONFLICT, encoding="utf-8")

    assert preset_update_status(repo / PROFILE).state is PresetUpdateState.UNREADABLE
    assert not _presets_changed(repo)


# ── a repo from before the saved copy: what cannot be told apart is listed ───

LEFT = "Preset changes left for you"


def _left(output: str) -> str:
    """The lines ``--revalidate`` printed under "left for you"."""
    return output.split(LEFT, 1)[1] if LEFT in output else ""


def test_a_tier_from_before_the_saved_copy_whose_login_moved_on_is_listed(repo: Path) -> None:
    """Nothing says what the bundled login changed: what the tier lacks or states differently."""
    shutil.rmtree(repo / ".osprey")
    tier = repo / "tiers" / "admin.yml"
    _edit(tier, "  preset_hash: sha256:", "  preset_hash: sha256:0ld")
    _edit(tier, "  web.scaffold_gallery.write_enabled: true\n", "")
    _edit(tier, "  web.config_panel.enabled: true\n", "  web.config_panel.enabled: false\n")
    _edit(tier, "  - bluesky ", "  - bluesky-removed ")
    before = yaml.safe_load(tier.read_text(encoding="utf-8"))

    first = _revalidate(repo)

    left = _left(first.output)
    assert "tiers/admin.yml: config.web.scaffold_gallery.write_enabled" in left
    assert "tiers/admin.yml: config.web.config_panel.enabled" in left
    assert "tiers/admin.yml: web_panels: bluesky:" in left
    assert "tiers/readonly.yml" not in first.output
    # Listed, never written: a key the tier lacks is as likely one the operator removed.
    after = yaml.safe_load(tier.read_text(encoding="utf-8"))
    assert {k: v for k, v in after.items() if k != "provenance"} == {
        k: v for k, v in before.items() if k != "provenance"
    }
    assert "tiers/admin.yml" in read_saved_copy(repo)
    assert LEFT not in _revalidate(repo).output


def test_triggers_from_before_the_saved_copy_that_differ_from_the_presets_are_listed(
    repo: Path,
) -> None:
    """``triggers.yml`` carries no hash: the file itself is the only record of what was copied."""
    shutil.rmtree(repo / ".osprey")
    triggers = repo / "triggers.yml"
    document = yaml.safe_load(triggers.read_text(encoding="utf-8"))
    document["triggers"] = [t for t in document["triggers"] if t["name"] != "hello-dispatch"]
    document["dispatcher"]["max_queue_depth"] = 7
    triggers.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    before = triggers.read_text(encoding="utf-8")

    first = _revalidate(repo)

    left = _left(first.output)
    assert "triggers.yml: triggers[hello-dispatch]" in left
    assert "triggers.yml: dispatcher.max_queue_depth" in left
    assert triggers.read_text(encoding="utf-8") == before
    assert LEFT not in _revalidate(repo).output


def test_the_one_time_fill_lists_what_the_profile_lacks_outside_config(repo: Path) -> None:
    """The fill writes ``config:`` keys only: a panel the profile lacks is listed, not dropped."""
    shutil.rmtree(repo / ".osprey")
    _edit(repo / PROFILE, "  preset_hash: sha256:", "  preset_hash: sha256:0ld")
    _edit(repo / PROFILE, PANEL, "")
    _edit(repo / PROFILE, THEME, "")

    first = _revalidate(repo)

    assert "added config.web.theme" in first.output
    assert "profile.yml: web_panels: jupyter:" in _left(first.output)
    assert "jupyter" not in _profile(repo)["web_panels"]
    assert LEFT not in _revalidate(repo).output


def test_a_provider_missing_from_before_the_saved_copy_is_listed_not_added(repo: Path) -> None:
    """With no saved copy, an entry the file lacks is as likely one the operator removed."""
    from osprey.cli.profile_cmd import _catalog_entry_blocks

    shutil.rmtree(repo / ".osprey")
    _edit(repo / PROFILE, "  providers_hash: sha256:", "  providers_hash: sha256:0ld")
    catalog = repo / "providers.yml"
    blocks = _catalog_entry_blocks(catalog.read_text(encoding="utf-8"))
    # One entry and one key an earlier release's marker claims absent, and one of each unclaimed.
    _edit(catalog, blocks["openrouter"], "  # DEVIATION: providers.openrouter: no route out.\n")
    _edit(catalog, blocks["vllm"], "")
    _edit(
        catalog,
        "    health_model: claude-haiku-4-5\n",
        "    # DEVIATION: providers.cborg.health_model: probed with the default model.\n",
    )
    _edit(catalog, "    health_model: gpt-5.6-luna\n", "")
    before = catalog.read_text(encoding="utf-8")

    first = _revalidate(repo)

    assert catalog.read_text(encoding="utf-8") == before
    left = _left(first.output)
    assert "providers.yml: providers.vllm:" in left
    assert "providers.yml: providers.openai.health_model:" in left
    assert "openrouter" not in first.output
    assert "cborg" not in first.output
    second = _revalidate(repo)
    assert catalog.read_text(encoding="utf-8") == before
    assert LEFT not in second.output


def test_revalidate_refuses_an_unreadable_saved_copy_and_says_how_to_recover(repo: Path) -> None:
    (repo / TRIGGERS_BASE).write_text(CONFLICT, encoding="utf-8")
    before = (repo / PROFILE).read_text(encoding="utf-8")

    result = CliRunner().invoke(init, [str(repo), "--revalidate"])

    assert result.exit_code != 0
    assert f"Invalid YAML in {TRIGGERS_BASE}" in result.output
    assert "Restore it from version control, or delete" in result.output
    assert (repo / PROFILE).read_text(encoding="utf-8") == before
    assert (repo / TRIGGERS_BASE).read_text(encoding="utf-8") == CONFLICT


def test_revalidate_digests_the_copied_applications_once(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.cli import build_profile_updates

    calls: list[Path] = []
    real = build_profile_updates._bundled_app_digests

    def counted(root: Path) -> dict[str, tuple[str, str]]:
        calls.append(root)
        return real(root)

    monkeypatch.setattr(build_profile_updates, "_bundled_app_digests", counted)

    _revalidate(repo)

    assert len(calls) == 1
