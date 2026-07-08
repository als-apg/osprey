"""Filesystem locations for the paradigm channel databases and scenario data.

All paths are resolved relative to this file so the generator works
regardless of the caller's current working directory.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

_CONTROL_ASSISTANT_DATA = (
    REPO_ROOT / "src" / "osprey" / "templates" / "apps" / "control_assistant" / "data"
)

# Build-resolved default tier for the control-assistant preset. The preset's
# `channel_finder_mode` defaults to "hierarchical"
# (src/osprey/profiles/presets/control-assistant.yml), and
# BuildProfile.resolved_tier() maps that to tier 3 (in_context -> tier 1,
# hierarchical/middle_layer -> tier 3; see src/osprey/cli/build_profile.py).
# All three paradigm DBs are address-identical at tier 3 (verified by
# build.build_manifest()'s cross-paradigm check), so tier 3 is the single
# build-resolved tier this generator expands.
DEFAULT_TIER = 3

TIER_DIR = _CONTROL_ASSISTANT_DATA / "channel_databases" / "tiers" / f"tier{DEFAULT_TIER}"

HIERARCHICAL_DB = TIER_DIR / "hierarchical.json"
IN_CONTEXT_DB = TIER_DIR / "in_context.json"
MIDDLE_LAYER_DB = TIER_DIR / "middle_layer.json"

MACHINE_JSON = _CONTROL_ASSISTANT_DATA / "simulation" / "machine.json"
MACHINE_STATE_TEMPLATE = _CONTROL_ASSISTANT_DATA / "machine_state_channels.json.j2"

MANIFEST_OUTPUT = Path(__file__).resolve().parent / "channel_manifest.json"
