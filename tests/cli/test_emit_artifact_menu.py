"""An emitted profile is menu-complete: every selectable artifact is either
active in one of the six lists or offered as a commented entry."""

from osprey.cli.build_profile_emit import (
    _MENU_HEADER,
    artifact_menu_catalog,
    emit_standalone_profile_yaml,
)


def _emit(preset: str) -> str:
    return emit_standalone_profile_yaml(preset, (), (), "Emitted")


def test_catalog_covers_the_six_profile_list_keys() -> None:
    menu = artifact_menu_catalog()
    assert set(menu) == {"hooks", "rules", "skills", "agents", "output_styles", "web_panels"}
    # Panels that are universal (always served) are not offered as opt-ins.
    panel_names = [name for name, _ in menu["web_panels"]]
    assert "artifacts" not in panel_names
    assert "lattice" in panel_names


def test_menu_offers_unselected_panels_in_control_assistant() -> None:
    # control-assistant selects ariel/channel-finder/okf/system-health; lattice is the gap.
    text = _emit("control-assistant")
    assert "# - lattice" in text


def test_menu_dedupes_against_preset_comment_mentions() -> None:
    # The preset's skills list already discusses setup-mode in a hand-written
    # comment; the generated menu must not offer it a second time.
    text = _emit("control-assistant")
    assert text.count("setup-mode") == 1


def test_every_selectable_artifact_is_active_or_offered() -> None:
    text = _emit("control-assistant")
    for entries in artifact_menu_catalog().values():
        for name, _description in entries:
            assert name in text, f"{name} is neither active nor offered"


def test_empty_lists_still_offer_the_menu() -> None:
    # hello-world's skills list is inline-empty (`skills: []`); the menu still
    # lands beneath it.
    text = _emit("hello-world")
    assert _MENU_HEADER in text
    assert "# - diagnose" in text


def test_menu_insertion_is_deterministic() -> None:
    assert _emit("control-assistant") == _emit("control-assistant")
