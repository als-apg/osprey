"""The presets that declare ``facility.ontology`` ship the table it names.

``facility.ontology`` is the single source of the device vocabulary the
channel-finder subagent's terminology table renders, and a declared path that
does not resolve is a hard build error. Two presets declare it, so two things
have to hold for a fresh ``osprey init`` to render a working table: the key has
to name a file the render actually copies, and that file has to be the compiled
ontology the rest of the demo machine was generated from.

The second half is the same dual-copy problem ``demo_machine.ttl`` already has,
one file further up the pipeline. ``demo_ontology.json`` is generated —
``osprey knowledge compile-ontology`` writes it from ``demo_ontology.yaml`` —
and each preset carries its own committed copy so the preset's ``data/`` tree
is self-contained. A regeneration that reached the package copy and not the
presets would leave the shipped agents naming classes and family tokens the
shipped corpus no longer has, which is exactly the drift the vocabulary was
centralised to end.

``ariel_standalone`` is asserted the other way round: it must NOT declare the
key. Its graph agent reads the vocabulary out of the seeded store rather than
out of a file, so a live key there would be a second source of truth for a fact
the store already holds.
"""

from pathlib import Path

import pytest
import yaml

from osprey.cli.templates.manager import TemplateManager

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The compiled table as the package ships it — the copies' authority.
PACKAGED_ONTOLOGY = (
    _REPO_ROOT / "src/osprey/services/facility_knowledge/ttl_generator/demo_ontology.json"
)

#: Where each preset's copy lands in its own data tree, and the value its
#: ``config.yml`` must declare for it. Spelled here rather than read back from
#: the rendered config so the test states the expected path instead of agreeing
#: with whatever is configured.
DECLARED_PATH = "data/facility_ontology.json"

#: The presets that declare the key and therefore ship the table.
PRESETS_WITH_ONTOLOGY = ("control_assistant", "channel_finder_standalone")


def _preset_copy(preset: str) -> Path:
    return _REPO_ROOT / "src/osprey/templates/apps" / preset / DECLARED_PATH


def _rendered_config(tmp_path: Path, preset: str) -> dict:
    """The preset's ``config.yml``, rendered the way the CLI renders it."""
    manager = TemplateManager()
    output = tmp_path / f"{preset}-config.yml"
    manager.render_config(
        project_name=f"ontology-{preset}",
        project_dir=tmp_path / preset,
        output_path=output,
        data_bundle=preset,
        # Required of any bundle that selects the channel-finder agent; the
        # paradigm is irrelevant to the facility block, so one is pinned rather
        # than parametrised.
        context={"channel_finder_mode": "hierarchical"},
    )
    return yaml.safe_load(output.read_text(encoding="utf-8")) or {}


@pytest.mark.parametrize("preset", PRESETS_WITH_ONTOLOGY)
def test_preset_copy_is_the_packaged_table_byte_for_byte(preset):
    """Each preset's copy is the generated table, not a hand-edited variant."""
    copy = _preset_copy(preset)
    assert copy.is_file(), f"{preset} declares {DECLARED_PATH} but ships no such file"
    assert copy.read_bytes() == PACKAGED_ONTOLOGY.read_bytes(), (
        f"{copy} has drifted from {PACKAGED_ONTOLOGY}. Re-copy it after every "
        "`osprey knowledge compile-ontology` run — both presets and the package "
        "carry the same generated file."
    )


@pytest.mark.parametrize("preset", PRESETS_WITH_ONTOLOGY)
def test_preset_declares_the_path_it_ships(tmp_path, preset):
    """The rendered key names the file the render puts on disk."""
    config = _rendered_config(tmp_path, preset)
    facility = config.get("facility") or {}
    assert facility.get("ontology") == DECLARED_PATH, (
        f"{preset} must declare facility.ontology as {DECLARED_PATH!r} — the "
        "path its data tree ships the compiled table at."
    )


def test_ariel_standalone_leaves_the_key_commented(tmp_path):
    """ARIEL's agent reads the vocabulary from the store, not from a file."""
    config = _rendered_config(tmp_path, "ariel_standalone")
    facility = config.get("facility") or {}
    assert "ontology" not in facility, (
        "ariel_standalone must keep facility.ontology as a commented example: "
        "its graph agent captures the vocabulary from the seeded store, and a "
        "live key would be a second source of truth for it."
    )
    assert not _preset_copy("ariel_standalone").exists()


def test_hello_world_declares_no_facility_block(tmp_path):
    """The minimal preset is untouched by the vocabulary work."""
    config = _rendered_config(tmp_path, "hello_world")
    assert config.get("facility") is None
    assert not _preset_copy("hello_world").exists()
