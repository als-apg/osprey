"""``osprey init --revalidate`` rewrites the retired write-posture keys.

The retired ``control_system.writes_enabled`` and every
``control_system.connector.<type>.writes_enabled`` become the tier key
``control_system.writes`` by the parse point's rule, in the profile, its
overlays and every tier or app file the rewrite keeps. The previous release's
comment stanzas about them take the bundled preset's current words; anything
else written about them is left for the operator and printed.

The repo is ``tests/fixtures/previous_release_repo/`` laid over the exemplar,
as in ``test_init_revalidate.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile_resolve import resolve_build_document
from osprey.cli.init_cmd import init
from osprey.cli.init_revalidate import WRITE_STANZAS, _preset_stanza
from osprey.utils.config_writer import config_update_fields
from osprey_connectors.types import configured_targets, tier_writes
from tests._retired_layout import RETIRED_LAYER_DIRNAME
from tests.cli.test_init_revalidate import (
    _hermetic_git,  # noqa: F401  (autouse: keeps the developer's git config out)
    _previous_release_repo,
    _repo_files,
    _revalidate,
    revalidated_exemplar,
)

_FLAT = "  control_system.writes_enabled: true\n"


def _overlay(root: Path, text: str) -> Path:
    path = root / "profiles" / "night.yml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _render_section(path: Path, scratch: Path) -> dict:
    """The ``control_system`` section the build writes for the render at *path*."""
    loaded = resolve_build_document(path, None, (), ())
    target = scratch / "rendered-config.yml"
    target.write_text("{}\n", encoding="utf-8")
    config_update_fields(target, loaded.profile.config)
    section = (yaml.safe_load(target.read_text(encoding="utf-8")) or {}).get("control_system")
    return section if isinstance(section, dict) else {}


# ── The gate ────────────────────────────────────────────────────────────────


def test_the_previous_release_repo_revalidates_to_the_exemplar_and_then_rests(
    tmp_path: Path,
) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar", git=True)

    first = _revalidate(root)

    assert first.exit_code == 0, first.output
    assert _repo_files(root) == revalidated_exemplar()
    assert "Left for you to do by hand" not in first.output
    profile = (root / "profile.yml").read_text(encoding="utf-8")
    assert "writes_enabled:" not in profile
    # The hooks-list end-of-line comment is a stanza of its own (D7): it is
    # brought to the switch wording, and the retired key leaves the comment too.
    assert "while its machine's write switch is off\n" in profile
    assert "writes_enabled" not in profile

    second = _revalidate(root)

    assert second.exit_code == 0
    assert "nothing to rewrite" in second.output
    assert _repo_files(root) == revalidated_exemplar()


def test_logbook_and_knowledge_become_shared_logins(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")

    assert _revalidate(root).exit_code == 0

    for login in ("logbook", "knowledge"):
        text = (root / "tiers" / f"{login}.yml").read_text(encoding="utf-8")
        assert "control_system.writes: never\n" in text
        assert "\nlists: own\n" in text
    assert not (root / "apps").exists()
    assert not (root / RETIRED_LAYER_DIRNAME).exists()


#: Each retired layer: its successor's kind, the tier key it read, the one its
#: successor reads. An unedited layer is re-emitted from today's preset, so the
#: readwrite tier moves to the shipped `asked` (FR4); every other value holds.
_SUCCESSORS = {
    "readonly": ("tiers", "never", "never"),
    "readwrite": ("tiers", "allowed", "asked"),
    "admin": ("tiers", "allowed", "allowed"),
    "logbook": ("tiers", "never", "never"),
    "knowledge": ("tiers", "never", "never"),
}


@pytest.mark.parametrize("name", sorted(_SUCCESSORS))
def test_each_old_tier_and_its_successor_resolve_alike(tmp_path: Path, name: str) -> None:
    """The shipped tier key and the same configured targets after the rewrite."""
    root = _previous_release_repo(tmp_path / "als-exemplar")
    old = _render_section(root / RETIRED_LAYER_DIRNAME / f"{name}.yml", tmp_path)

    assert _revalidate(root).exit_code == 0
    kind, before, after = _SUCCESSORS[name]
    new = _render_section(root / kind / f"{name}.yml", tmp_path)

    assert tier_writes(old) == before
    assert tier_writes(new) == after
    assert new.get("writes") == after
    assert configured_targets(old) == configured_targets(new)
    assert configured_targets(new)


# ── The fill ────────────────────────────────────────────────────────────────


def test_a_silent_root_is_not_given_the_tier_key(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    profile = root / "profile.yml"
    profile.write_text(profile.read_text(encoding="utf-8").replace(_FLAT, ""), encoding="utf-8")

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    text = profile.read_text(encoding="utf-8")
    assert "control_system.writes:" not in text
    assert "control_system.writes: a root that does not state it reads as `never`" in (
        " ".join(result.output.split())
    )
    assert tier_writes(_render_section(profile, tmp_path)) == "never"


# ── Spellings and placement ─────────────────────────────────────────────────


def test_a_nested_flat_key_is_renamed_in_place_with_its_comments(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    overlay = _overlay(
        root,
        "config:\n"
        "  control_system:\n"
        "    # the live machine stays off\n"
        "    writes_enabled: false  # night\n"
        "    connector:\n"
        "      virtual_accelerator:\n"
        "        writes_enabled: true\n"
        "        timeout_s: 3.0\n",
    )

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert overlay.read_text(encoding="utf-8") == (
        "config:\n"
        "  control_system:\n"
        "    # the live machine stays off\n"
        "    writes: allowed  # night\n"
        "    connector:\n"
        "      virtual_accelerator:\n"
        "        timeout_s: 3.0\n"
    )
    assert "# the live machine stays off" in result.output


def test_per_type_keys_alone_put_the_tier_key_where_the_first_stood(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    overlay = _overlay(
        root,
        "config:\n"
        "  web.ui_mode: simple\n"
        "  control_system.connector.epics.writes_enabled: false\n"
        "  control_system.connector.virtual_accelerator.writes_enabled: true\n"
        "  web.control_target_picker: true\n",
    )

    assert _revalidate(root).exit_code == 0

    assert overlay.read_text(encoding="utf-8") == (
        "config:\n"
        "  web.ui_mode: simple\n"
        "  control_system.writes: allowed\n"
        "  web.control_target_picker: true\n"
    )


def test_a_connector_block_the_rewrite_empties_is_reported_not_removed(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    overlay = _overlay(
        root,
        "config:\n  control_system:\n    connector:\n      epics:\n        writes_enabled: true\n",
    )

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    data = yaml.safe_load(overlay.read_text(encoding="utf-8"))
    assert data == {"config": {"control_system": {"connector": {"epics": {}}, "writes": "allowed"}}}
    assert "profiles/night.yml: `control_system.connector.epics` held only" in " ".join(
        result.output.split()
    )


def test_both_spellings_in_one_file_are_refused_and_nothing_is_touched(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    _overlay(
        root,
        "config:\n  control_system.writes: never\n  control_system.writes_enabled: true\n",
    )
    before = _repo_files(root)

    result = CliRunner().invoke(init, [str(root), "--revalidate"])

    assert result.exit_code == 2
    assert "control_system.writes_enabled" in result.output
    assert "control_system.writes`" in result.output
    assert _repo_files(root) == before


def test_a_login_kept_in_apps_is_moved_and_rewritten_to_never_whatever_it_spelled(
    tmp_path: Path,
) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    app = root / "apps" / "extra.yml"
    app.parent.mkdir(exist_ok=True)
    app.write_text(
        "name: Extra\nconfig:\n  control_system.writes_enabled: true\n  web.ui_mode: expert\n",
        encoding="utf-8",
    )

    assert _revalidate(root).exit_code == 0

    # Moved to tiers/ as a `lists: own` login, which also gets the rules list it
    # selects by path (#2002); the retired key that armed reads `never`.
    assert not app.exists()
    assert (root / "tiers" / "extra.yml").read_text(encoding="utf-8") == (
        "name: Extra\nlists: own\nrules:\n  - rules/facility\nconfig:\n"
        "  control_system.writes: never\n  web.ui_mode: expert\n"
    )


# ── Comments ────────────────────────────────────────────────────────────────


def test_edited_stanzas_are_reported_and_examples_still_dropped(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    profile = root / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    edited = text.replace(
        "approval prompt is consulted. On here because the baseline is the",
        "approval prompt is consulted. ON here because the baseline is the",
    ).replace(
        "# Write posture for the simulator alone. Uncomment to arm writes here while",
        "# Our simulator posture. Uncomment to arm writes here while",
    )
    assert edited != text
    profile.write_text(edited, encoding="utf-8")

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    after = profile.read_text(encoding="utf-8")
    assert "control_system.writes: allowed\n" in after
    # The edited flat stanza stays as the operator left it, and is reported.
    assert "# Master write switch, the FIRST guard" in after
    # Every commented retired-key example is gone, edited stanza or not.
    assert "# control_system.connector.virtual_accelerator.writes_enabled" not in after
    assert "# control_system.connector.epics.writes_enabled" not in after
    # The untouched stanzas took today's words.
    assert "# Device attribute the target switch reads" in after
    output = " ".join(result.output.split())
    assert "Left for you to do by hand" in output
    assert "profile.yml: comments written for the retired `writes_enabled` keys remain" in output
    assert "the comment from `# Master write switch, the FIRST guard" in output
    assert (
        "`# `readwrite` and `admin` tiers in tiers/ write this key.`, above the dropped" in output
    )

    before = _repo_files(root)
    again = _revalidate(root)
    assert again.exit_code == 0
    assert _repo_files(root) == before


def test_every_frozen_stanza_has_its_successor_in_the_bundled_preset() -> None:
    """Tripwire: a preset edit that moves a successor stanza breaks this, not a repo."""
    assert WRITE_STANZAS
    for stanza in WRITE_STANZAS:
        current = _preset_stanza(stanza)
        assert current is not None, stanza
        assert len(current) == stanza.new_lines
        assert not any("writes_enabled" in line for line in current), stanza


# ── The retired per-tool policies of the governed tools ─────────────────────


def test_revalidate_drops_the_governed_policies_and_reports_them(tmp_path: Path) -> None:
    from osprey.cli import build_profile_document as doc
    from osprey.cli.build_profile_document import _read_profile_document
    from osprey.cli.init_revalidate import revalidate_repo

    root = _previous_release_repo(tmp_path / "als-exemplar")
    before = (root / "profile.yml").read_text(encoding="utf-8")
    assert "approval.tools.channel_write: always\n" in before
    assert "approval.tools.execute: selective\n" in before

    result = revalidate_repo(root)

    text = (root / "profile.yml").read_text(encoding="utf-8")
    assert "approval.tools.channel_write" not in text
    assert "approval.tools.execute:" not in text
    assert "approval.tools.entry_create: always\n" in text
    assert "approval.tools.setup_patch: always\n" in text
    assert len(result.retired_policies) == 1
    report = result.retired_policies[0]
    assert report.startswith("profile.yml: ")
    assert "`approval.tools.channel_write`" in report
    assert "`approval.tools.execute`" in report
    assert "slider" in report
    assert not any("approval.tools" in step for step in result.manual_steps)

    # The rewritten profile parses silently: nothing is left to drop.
    doc._WARNED_RETIRED_KEYS.clear()
    raw = _read_profile_document(root / "profile.yml", "rewritten.yml")
    assert not any(
        source == "rewritten.yml" and key == "approval.tools"
        for source, key in doc._WARNED_RETIRED_KEYS
    )
    assert raw["config"]["approval.tools.entry_create"] == "always"

    again = revalidate_repo(root)
    assert again.retired_policies == []
    assert not again.changed


def test_a_stated_tier_word_unlike_the_presets_is_listed_as_skipped(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    said = " ".join(result.output.split())
    assert "Left for you: control_system.writes: this profile states `allowed`" in said
    assert "preset `control-assistant` ships `asked`" in said
    assert "preset-drift NOTE, not a refusal" in said
    assert "`# DEVIATION:`" in said
    assert "control_system.writes: allowed\n" in (root / "profile.yml").read_text(encoding="utf-8")


def test_a_kept_tier_file_loses_its_governed_policy_too(tmp_path: Path) -> None:
    from osprey.cli.init_revalidate import revalidate_repo

    root = _previous_release_repo(tmp_path / "als-exemplar")
    assert revalidate_repo(root).changed
    tier = root / "tiers" / "readwrite.yml"
    text = tier.read_text(encoding="utf-8")
    assert text.count("\nconfig:\n") == 1
    tier.write_text(
        text.replace(
            "\nconfig:\n",
            "\nconfig:\n  approval.tools.queue_add: skip\n  approval.tools.entry_create: skip\n",
        ),
        encoding="utf-8",
    )

    result = revalidate_repo(root)

    rewritten = tier.read_text(encoding="utf-8")
    assert "approval.tools.queue_add" not in rewritten
    assert "  approval.tools.entry_create: skip\n" in rewritten
    assert [line.split(":", 1)[0] for line in result.retired_policies] == ["tiers/readwrite.yml"]
    assert "`approval.tools.queue_add`" in result.retired_policies[0]
