"""The preset-side ``facility:`` key and the packaged data it composes.

A preset names a bundled facility (``templates/facilities/<name>/``) with
``facility:``, beside the app template it names with ``app_template:``. Both
keys are preset-side only: the resolver follows ``extends:`` to find them, the
preset read consumes them, an emitted profile carries neither, and a repo
``profile.yml`` spelling one is refused. ``osprey init`` copies the app
template's ``data/`` plus the facility at ``data/facility/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile import _load_preset_raw, resolve_build_profile
from osprey.cli.build_profile_emit import emit_standalone_profile_yaml
from osprey.cli.build_profile_presets import (
    PRESET_FACILITY_KEY,
    _read_preset_document,
    list_presets,
    list_tiers,
    preset_facility,
)
from osprey.cli.profile_cmd import _preset_data
from osprey.cli.templates.manager import TemplateManager
from osprey.cli.templates.preset_data import compose_preset_data
from osprey_connectors.errors import BuildProfileError

FACILITIES = Path(TemplateManager().template_root) / "facilities"


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    """Write a profile document; a repo profile also gets the ``data:`` it must name."""
    if path.parent.name != "presets":
        body = {**body, "data": "data"}
        (path.parent / "data").mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("preset", "facility"),
    [
        ("control-assistant", "example"),
        ("control_assistant", "example"),
        ("hello-world", "hello_world"),
        ("tiers/readonly", None),
        ("tiers/logbook", None),
        (None, None),
        ("no-such-preset", None),
    ],
)
def test_the_resolver_names_each_shipped_preset_s_facility(
    preset: str | None, facility: str | None
) -> None:
    assert preset_facility(preset) == facility


def test_every_facility_a_shipped_preset_names_is_shipped() -> None:
    named = {preset_facility(preset) for preset in list_presets()} - {None}

    assert named == {"example", "hello_world"}
    assert all((FACILITIES / name).is_dir() for name in named)


@pytest.mark.parametrize(
    "layer",
    [f"tiers/{name}" for name in list_tiers()],
)
def test_no_shipped_tier_or_app_names_a_facility(layer: str) -> None:
    """The facility is the deployment's, named once on the root preset.

    A tier or app renders on the data bundle of the root it rides on, so a
    ``facility:`` of its own would name a bundle nothing composes.
    """
    assert PRESET_FACILITY_KEY not in _read_preset_document(layer)[0]
    assert preset_facility(layer) is None


def test_the_resolver_follows_extends(fake_presets: Path) -> None:
    _write_yaml(fake_presets / "base.yml", {"name": "base", "facility": "example"})
    _write_yaml(fake_presets / "child.yml", {"name": "child", "extends": "base"})

    assert preset_facility("child") == "example"


def test_the_nearest_facility_on_the_chain_wins(fake_presets: Path) -> None:
    _write_yaml(fake_presets / "base.yml", {"name": "base", "facility": "example"})
    _write_yaml(
        fake_presets / "child.yml", {"name": "child", "extends": "base", "facility": "hello_world"}
    )

    assert preset_facility("child") == "hello_world"


def test_a_chain_naming_no_facility_resolves_to_none(fake_presets: Path) -> None:
    _write_yaml(fake_presets / "base.yml", {"name": "base", "app_template": "hello_world"})

    assert preset_facility("base") is None


# ---------------------------------------------------------------------------
# Consumed by the preset read; never part of a profile
# ---------------------------------------------------------------------------


def test_the_preset_read_consumes_the_facility_key(fake_presets: Path) -> None:
    _write_yaml(fake_presets / "base.yml", {"name": "base", "facility": "example"})

    raw, _path = _load_preset_raw("base")

    assert PRESET_FACILITY_KEY not in raw


def test_an_inherited_facility_key_is_consumed_too(fake_presets: Path) -> None:
    _write_yaml(fake_presets / "base.yml", {"name": "base", "facility": "example"})
    _write_yaml(fake_presets / "child.yml", {"name": "child", "extends": "base"})

    profile, _dir = resolve_build_profile(None, "child")

    assert profile.name == "child"


def test_a_repo_profile_extending_a_preset_naming_a_facility_resolves(
    fake_presets: Path, tmp_path: Path
) -> None:
    _write_yaml(fake_presets / "base.yml", {"name": "base", "facility": "example"})
    profile = _write_yaml(tmp_path / "p.yml", {"name": "p", "extends": "base"})

    resolved, _dir = resolve_build_profile(profile, None)

    assert resolved.name == "p"


def test_a_profile_naming_a_facility_is_refused(tmp_path: Path) -> None:
    profile = _write_yaml(tmp_path / "p.yml", {"name": "p", "facility": "example"})

    with pytest.raises(BuildProfileError, match="facility is a preset key"):
        resolve_build_profile(profile, None)


@pytest.mark.parametrize("preset", ["control-assistant", "hello-world"])
def test_the_emitted_profile_carries_neither_the_key_nor_its_comment(preset: str) -> None:
    text = emit_standalone_profile_yaml(preset, (), "Emitted", project_name="emitted")

    assert PRESET_FACILITY_KEY not in (yaml.safe_load(text) or {})
    assert "Which bundled facility this shows" not in text


# ---------------------------------------------------------------------------
# The composition: app template data/ + facility at data/facility/
# ---------------------------------------------------------------------------


def test_the_composition_lands_the_facility_under_facility(tmp_path: Path) -> None:
    composed = _preset_data(TemplateManager(), "control-assistant")
    target = tmp_path / "data"

    composed.copy_into(target)

    assert (target / "facility" / "identity.yaml").read_bytes() == (
        FACILITIES / "example" / "identity.yaml"
    ).read_bytes()
    assert (target / "README.md").is_file()


def test_the_example_facility_states_its_display_name() -> None:
    identity = yaml.safe_load((FACILITIES / "example" / "identity.yaml").read_text())

    assert identity == {"code": "ca", "name": "Example Research Facility"}


def test_a_preset_naming_no_facility_composes_the_app_template_alone(tmp_path: Path) -> None:
    root = _template_root(tmp_path, app_ships_facility=False)

    composed = compose_preset_data(root, "app", None)

    assert composed.facility_root is None
    assert not any(relative.startswith("facility/") for relative in composed.placed_files())


def _template_root(tmp_path: Path, *, app_ships_facility: bool) -> Path:
    root = tmp_path / "templates"
    (root / "apps" / "app" / "data").mkdir(parents=True)
    if app_ships_facility:
        (root / "apps" / "app" / "data" / "facility").mkdir()
    (root / "facilities" / "plant").mkdir(parents=True)
    (root / "facilities" / "plant" / "identity.yaml").write_text("code: plant\n")
    return root


def test_an_app_template_shipping_a_facility_beside_a_named_one_is_refused(
    tmp_path: Path,
) -> None:
    root = _template_root(tmp_path, app_ships_facility=True)

    with pytest.raises(BuildProfileError, match="ships data/facility/"):
        compose_preset_data(root, "app", "plant")


def test_an_app_template_shipping_a_facility_composes_when_none_is_named(
    tmp_path: Path,
) -> None:
    root = _template_root(tmp_path, app_ships_facility=True)

    assert compose_preset_data(root, "app", None).facility_root is None


def test_an_absent_facility_is_a_packaging_fault(tmp_path: Path) -> None:
    root = _template_root(tmp_path, app_ships_facility=False)

    with pytest.raises(BuildProfileError, match="reinstall"):
        compose_preset_data(root, "app", "no_such_facility")
