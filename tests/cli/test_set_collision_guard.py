"""Tests for the ``--set config.*`` spelling-collision guard.

A ``config:`` block is a flat bag of literal dotted keys, each applied verbatim
to one addressed leaf. ``--set config.a.b=1`` instead merges a nested ``a:``
mapping in beside them, and the deep merge never notices: ``'a.b'`` and ``'a'``
are different dict keys, so both survive and which one reaches the rendered
``config.yml`` is decided by key order. One of the two values the user gave is
then discarded with nothing said. The guard refuses that pair up front, after
``extends`` resolution so a literal key inherited from a parent counts too, and
names the spelling that would actually have worked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_profile_load import CONNECTOR_CONFIG_KEY
from osprey.cli.build_profile_presets import list_presets
from osprey.cli.build_profile_resolve import (
    _set_config_paths,
    merge_cli_overrides,
    resolve_build_profile,
)
from osprey.cli.init_cmd import init
from osprey.errors import BuildProfileError


def _flat(text: str) -> str:
    """Collapse whitespace so assertions survive terminal line wrapping."""
    return " ".join(text.split())


# ── recording the --set provenance ───────────────────────────────────────────


def test_set_config_paths_records_the_addressed_path() -> None:
    """The segments after ``config`` are recorded exactly as the user typed them."""
    assert _set_config_paths(("config.control_system.type=epics",)) == [("control_system", "type")]


def test_non_config_set_pairs_are_not_recorded() -> None:
    """Only ``config.*`` paths address the rendered config."""
    assert _set_config_paths(("provider=anthropic", "connector=epics", "model=sonnet")) == []


def test_bare_config_set_pair_is_not_recorded() -> None:
    """``--set config=...`` replaces the block wholesale and addresses no path."""
    assert _set_config_paths(("config={}",)) == []


def test_merge_cli_overrides_fills_the_collector() -> None:
    """The layering step is where the provenance exists, so it records there."""
    collected: list[tuple[str, ...]] = []
    merge_cli_overrides(
        {"name": "x"},
        (),
        ("config.modules.web_terminals.enabled=false", "provider=anthropic"),
        set_config_paths=collected,
    )

    assert collected == [("modules", "web_terminals", "enabled")]


def test_merge_cli_overrides_without_a_collector_is_unchanged() -> None:
    """The collector is opt-in — callers that only want the merge pass nothing."""
    merged = merge_cli_overrides({"name": "x"}, (), ("config.system.timezone=UTC",))

    assert merged == {"name": "x", "config": {"system": {"timezone": "UTC"}}}


# ── the collision, on a preset that spells the literal key itself ────────────


def test_nested_connector_set_collides_with_the_preset_literal_key() -> None:
    """``--set config.control_system.type=`` never silently outranks the preset."""
    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_profile(
            None, "control-assistant", set_pairs=("config.control_system.type=epics",)
        )

    message = _flat(str(excinfo.value))
    assert "--set config.control_system.type=..." in message
    assert repr(CONNECTOR_CONFIG_KEY) in message


def test_the_connector_collision_names_the_shorthand_as_the_remedy() -> None:
    """For this one key there is a ``--set`` spelling that works; it is named."""
    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_profile(
            None, "control-assistant", set_pairs=("config.control_system.type=epics",)
        )

    assert "--set connector=<type>" in _flat(str(excinfo.value))


def test_a_non_connector_collision_points_at_an_override_file() -> None:
    """No ``--set`` spelling reaches a literal dotted key, so ``-O`` is the remedy."""
    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_profile(
            None,
            "control-assistant",
            set_pairs=("config.modules.web_terminals.enabled=false",),
        )

    message = _flat(str(excinfo.value))
    assert "'modules.web_terminals'" in message
    assert "-O override file" in message
    assert "modules.web_terminals.enabled: <value>" in message


def test_every_colliding_pair_is_named_at_once() -> None:
    """Two bad spellings are two problems reported together, not one at a time."""
    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_profile(
            None,
            "control-assistant",
            set_pairs=(
                "config.control_system.type=epics",
                "config.web.panels.plan.label=SCAN",
            ),
        )

    message = _flat(str(excinfo.value))
    assert "--set config.control_system.type=..." in message
    assert "--set config.web.panels.plan.label=..." in message


# ── extends parents are not an escape ────────────────────────────────────────


@pytest.mark.parametrize("preset", ["control-assistant-readonly", "control-assistant-readwrite"])
def test_a_literal_key_inherited_from_a_parent_still_collides(preset: str) -> None:
    """The check runs post-``extends``: the parent's key counts as the profile's."""
    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_profile(None, preset, set_pairs=("config.control_system.type=epics",))

    assert repr(CONNECTOR_CONFIG_KEY) in _flat(str(excinfo.value))


@pytest.mark.parametrize("preset", ["control-assistant-readonly", "control-assistant-readwrite"])
def test_the_persona_tier_key_collides_in_its_own_spelling(preset: str) -> None:
    """``writes_enabled`` is the tier boundary — overriding it nested is refused."""
    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_profile(
            None, preset, set_pairs=("config.control_system.writes_enabled=true",)
        )

    message = _flat(str(excinfo.value))
    assert "'control_system.writes_enabled'" in message
    # The nested mapping's blast radius is the whole subtree, so the sibling
    # key it would have wiped is named too.
    assert repr(CONNECTOR_CONFIG_KEY) in message


def test_a_parent_connector_shorthand_counts_as_the_literal_key(tmp_path: Path) -> None:
    """A parent spelling ``connector:`` sets ``control_system.type`` just as surely."""
    parent = tmp_path / "parent.yml"
    parent.write_text("name: Parent\nconnector: mock\n", encoding="utf-8")
    child = tmp_path / "child.yml"
    child.write_text("name: Child\nextends: parent.yml\n", encoding="utf-8")

    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_profile(child, None, set_pairs=("config.control_system.type=epics",))

    assert repr(CONNECTOR_CONFIG_KEY) in _flat(str(excinfo.value))


# ── connector shorthand given alongside the literal key ──────────────────────


def test_connector_with_a_nested_type_set_is_refused() -> None:
    """One command line stating the connector twice is a usage error, not a race."""
    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_profile(
            None,
            "hello-world",
            set_pairs=("connector=epics", "config.control_system.type=doocs"),
        )

    message = _flat(str(excinfo.value))
    assert "Conflicting connector overrides" in message
    assert "connector='epics'" in message
    assert f"config.{CONNECTOR_CONFIG_KEY}='doocs'" in message


def test_connector_with_a_literal_key_in_an_override_file_is_refused(tmp_path: Path) -> None:
    """The ``-O`` file spelling is the same statement, so it conflicts the same way."""
    override = tmp_path / "over.yml"
    override.write_text(f"config:\n  {CONNECTOR_CONFIG_KEY}: doocs\n", encoding="utf-8")

    with pytest.raises(BuildProfileError, match="Conflicting connector overrides"):
        resolve_build_profile(None, "hello-world", (override,), ("connector=epics",))


def test_the_conflict_is_caught_at_merge_time() -> None:
    """Before extends resolution — the two spellings are both CLI-layer facts."""
    with pytest.raises(BuildProfileError, match="Conflicting connector overrides"):
        merge_cli_overrides({}, (), ("connector=epics", "config.control_system.type=doocs"))


# ── what must keep working ───────────────────────────────────────────────────


def test_the_shorthand_alone_still_overrides_the_preset_literal_key() -> None:
    """The preset's own literal key is what the shorthand exists to override.

    ``mock`` rather than ``epics`` because this preset ships a live stand-in,
    and a stand-in on an ``epics`` baseline is refused on its own terms: going
    live starts by deleting the stand-in, not by keeping it beside the gateways.
    """
    profile, _profile_dir = resolve_build_profile(
        None, "control-assistant", set_pairs=("connector=mock",)
    )

    assert profile.config[CONNECTOR_CONFIG_KEY] == "mock"


def test_a_nested_set_with_no_literal_key_beneath_it_resolves() -> None:
    """The guard fires on collisions only — an unrelated path is still a ``--set``."""
    profile, _profile_dir = resolve_build_profile(
        None, "hello-world", set_pairs=("config.system.timezone=UTC",)
    )

    assert profile.config["system"] == {"timezone": "UTC"}


def test_a_sibling_segment_is_not_a_collision() -> None:
    """Root-segment match is by segment, not by string prefix."""
    profile, _profile_dir = resolve_build_profile(
        None, "hello-world", set_pairs=("config.control_system_extras.debug=true",)
    )

    assert profile.config["control_system_extras"] == {"debug": True}
    assert profile.config[CONNECTOR_CONFIG_KEY] == "mock"


@pytest.mark.parametrize("preset", list_presets())
def test_every_shipped_preset_still_resolves(preset: str) -> None:
    """No preset trips its own guard — the check is about CLI layers only."""
    profile, _profile_dir = resolve_build_profile(None, preset)

    assert profile.name


# ── the CLI surface ──────────────────────────────────────────────────────────


def test_init_exits_non_zero_on_a_collision(tmp_path: Path) -> None:
    """The collision stops ``osprey init`` before the source zone is written."""
    target = tmp_path / "my-deployment"
    result = CliRunner().invoke(
        init,
        [
            str(target),
            "--preset",
            "control-assistant",
            "--no-git",
            "--set",
            "config.control_system.type=epics",
        ],
    )

    assert result.exit_code != 0
    assert "--set connector=<type>" in _flat(result.output)
    assert not (target / "profile.yml").exists()
