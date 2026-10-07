"""The channel-finder-standalone preset ships control-assistant's demo channel database.

Both presets serve the same demo storage ring. ``generate_from_spec`` writes the
tier-3 ``hierarchical.json`` under control-assistant's data tree, and the
standalone preset carries its own committed copy so its ``data/`` tree is
self-contained. The standalone preset also ships ``data/facility_ontology.json``,
which is pinned to the packaged ontology table, and that table names exactly the
families of the tier-3 database. A standalone copy that fell behind the tier-3
file would leave the standalone agent naming device families its own channel
database does not hold.
"""

from pathlib import Path

_APPS = Path(__file__).resolve().parents[2] / "src/osprey/templates/apps"

#: The tier-3 database as ``generate_from_spec`` writes it: the copy's authority.
TIER3_HIERARCHICAL = (
    _APPS / "control_assistant/data/channel_databases/tiers/tier3/hierarchical.json"
)

#: The standalone preset's copy, at the path its hierarchical pipeline reads.
STANDALONE_HIERARCHICAL = (
    _APPS / "channel_finder_standalone/data/channel_databases/hierarchical.json"
)


def test_standalone_copy_is_the_tier3_database_byte_for_byte():
    """The standalone preset's database is the generated tier-3 file, not a variant."""
    assert STANDALONE_HIERARCHICAL.is_file()
    assert STANDALONE_HIERARCHICAL.read_bytes() == TIER3_HIERARCHICAL.read_bytes(), (
        f"{STANDALONE_HIERARCHICAL} has drifted from {TIER3_HIERARCHICAL}. The two are "
        "regenerated together: after `python -m "
        "osprey.services.channel_finder.tools.generate_from_spec`, copy the tier-3 "
        "hierarchical.json over the standalone preset's copy."
    )
