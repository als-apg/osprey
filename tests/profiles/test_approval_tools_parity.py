"""The ``approval.tools.<tool>`` rows the root presets ship, against the registry.

Every row is a hand-written policy for one tool. The set is deliberately SMALLER
than the set of tools the approval hook governs — a curated starter subset under
a fail-closed ``approval.default_policy: always``, so a governed tool with no row
prompts every time rather than slipping through. The manifest states that
asymmetry outright and says not to "fix" one side to match the other, and
``scripts/check_config_keys.py``'s own governed-set checks already go red when a
newly gated tool appears.

What is NOT protected by any of that is the other direction: a row naming a tool
the preset's resolved servers do not gate. Such a row governs nothing, reads as
posture the deployment does not have, and nothing errors. That is what these
tests pin, along with the one other surface the same set has to appear on: the
manifest documents every shipped row, and documents no row that is unshipped.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESET_DIR = REPO_ROOT / "src" / "osprey" / "profiles" / "presets"
MANIFEST_PATH = REPO_ROOT / "src" / "osprey" / "profiles" / "config_key_manifest.yml"
GUARD_PATH = REPO_ROOT / "scripts" / "check_config_keys.py"

#: The four documents an operator starts from. Persona presets extend
#: control-assistant and carry deltas, not blocks of their own.
ROOT_PRESETS = ("control-assistant", "hello-world", "ariel-standalone", "channel-finder-standalone")

#: Rows that name a tool their preset's servers do not gate, and the reason.
#: Quoted from the manifest's own entry for these keys: "Inert in hello_world:
#: that template disables the ariel server, so the tool never exists. Harmless
#: (fail-closed either way); recorded so it is not mistaken for drift." The
#: manifest is pinned to this set below, so the two cannot part company.
_INERT_ROWS = {("hello-world", "entry_create"), ("hello-world", "entry_publish")}

#: The manifest phrase that records the exemption above. Matched on the shape
#: rather than the exact spelling: the manifest writes the preset's name both
#: ways ("hello_world" beside "hello-world"), and which separator a note happens
#: to use is not the fact being pinned.
_INERT_NOTE_RE = re.compile(r"inert in hello[-_]world", re.IGNORECASE)

_ROW_RE = re.compile(r"^\s*approval\.tools\.([a-z_]+):", re.MULTILINE)

pytestmark = pytest.mark.unit


def _load_guard():
    """Import ``scripts/check_config_keys.py`` by path, as its own suite does."""
    spec = importlib.util.spec_from_file_location("check_config_keys", GUARD_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def guard():
    module = _load_guard()
    return module.ConfigKeyGuard(REPO_ROOT, module.load_manifest(module.DEFAULT_MANIFEST))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text())


def _preset_rows(preset: str) -> set[str]:
    """The tool short names ``approval.tools.<name>`` names in *preset*."""
    return set(_ROW_RE.findall((PRESET_DIR / f"{preset}.yml").read_text()))


@pytest.mark.parametrize("preset", ROOT_PRESETS)
def test_no_row_names_a_tool_the_preset_does_not_gate(preset: str, guard) -> None:
    """A policy for an ungated tool is posture the deployment does not have."""
    governed = guard.governed_tools(preset)
    unexplained = {
        tool for tool in _preset_rows(preset) - governed if (preset, tool) not in _INERT_ROWS
    }

    assert not unexplained, (
        f"{preset}: approval.tools names {sorted(unexplained)}, which its resolved "
        f"servers do not attach the approval hook to"
    )


def test_the_inert_rows_are_still_recorded_in_the_manifest(manifest) -> None:
    """The exemption above is the manifest's, not this test's."""
    keys = manifest["keys"]
    for _preset, tool in _INERT_ROWS:
        note = keys[f"approval.tools.{tool}"].get("note", "")
        assert _INERT_NOTE_RE.search(note), tool


def test_the_manifest_rows_are_the_union_of_what_the_presets_ship(manifest) -> None:
    """Every shipped row is documented, and no documented row is unshipped."""
    documented = {
        key.removeprefix("approval.tools.")
        for key in manifest["keys"]
        if key.startswith("approval.tools.")
    }
    shipped: set[str] = set()
    for preset in ROOT_PRESETS:
        shipped |= _preset_rows(preset)

    assert documented == shipped
