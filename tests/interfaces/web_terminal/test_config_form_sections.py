"""The Config panel's Form view offers only sections a config file really has.

``_AGENT_CONFIG_SECTIONS`` is an allowlist, and an entry that matches no
top-level key renders nothing — silently. That is how ``python_execution`` (the
section is called ``execution``) kept the execution settings out of the form
until someone noticed by eye. The shipped presets carry the whole configuration
a build renders, so they are what an entry is checked against.

The list stays an ALLOWLIST. Inverting it to a denylist would make every new
config block editable in the Form view by default, and the Raw YAML view
already reaches everything.
"""

from typing import Any

import yaml

from osprey.cli.build_profile_presets import _presets_dir, list_presets
from osprey.interfaces.web_terminal.routes.config import _AGENT_CONFIG_SECTIONS


def _shipped_top_level_keys() -> set[str]:
    """Every top-level config section the presets OSPREY ships can render."""
    keys: set[str] = set()
    for name in list_presets():
        document: dict[str, Any] = (
            yaml.safe_load((_presets_dir() / f"{name}.yml").read_text(encoding="utf-8")) or {}
        )
        for dotted in document.get("config") or {}:
            keys.add(str(dotted).split(".", 1)[0])
    return keys


def test_every_form_section_is_a_real_config_section() -> None:
    shipped = _shipped_top_level_keys()
    assert shipped, "no preset carried a config block — the authority moved"
    phantom = [name for name in _AGENT_CONFIG_SECTIONS if name not in shipped]
    assert not phantom, (
        f"Form view sections matching no shipped config key: {phantom} — such an "
        "entry renders nothing at all, with no error"
    )


def test_the_form_view_stays_an_allowlist() -> None:
    """Narrower than the file on purpose: infra/build blocks stay in Raw YAML."""
    assert set(_AGENT_CONFIG_SECTIONS) < _shipped_top_level_keys()
