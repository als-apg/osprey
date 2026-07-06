"""Render/reconciliation test for ``machine_state_channels.json.j2`` (task 5.1).

Before this task the template branched on ``default_pipeline`` into three
separate channel lists (fictional FEL names for ``in_context``, ``SR01C:``-
style names for ``middle_layer``, obsolete bracket names ``MAG:DIPOLE[B01]``
for ``hierarchical``) -- none of which exist in any channel-finder DB or the
namespace-union manifest (see ``tests/va/test_manifest.py`` /
``docker/virtual-accelerator/manifest``). The template now emits ONE
canonical channel list, independent of pipeline mode, drawn from real
``RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD`` addresses that the manifest
actually contains.

This is the CC-4 regression guard: every rendered machine_state channel must
be a real address in the manifest's namespace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from osprey.cli.templates.manager import TemplateManager

REPO_ROOT = Path(__file__).resolve().parents[2]

# docker/virtual-accelerator/manifest is not an importable dotted package
# (its parent directory name contains a hyphen); see tests/va/test_manifest.py
# for the same import shim.
_MANIFEST_PARENT = REPO_ROOT / "docker" / "virtual-accelerator"
if str(_MANIFEST_PARENT) not in sys.path:
    sys.path.insert(0, str(_MANIFEST_PARENT))

from manifest import build_manifest  # noqa: E402

TEMPLATE_PATH = "apps/control_assistant/data/machine_state_channels.json.j2"

# Fictional/broken names from the pre-fix template, pinned so they never
# silently reappear.
FICTIONAL_ADDRESSES = (
    "TerminalVoltageReadBack",
    "GunPressure",
    "CollectorPressure",
    "IP41Pressure",
    "CoronaTriodCurrent",
    "SR01C:BPM1:X",
    "SR01C:BPM1:Y",
    "SR01C:HCM1:Current",
    "SR01C:VCM1:Current",
    "SR:DCCT:Current",
    "SR:RF1:Freq",
    "MAG:DIPOLE[B01]:CURRENT:RB",
    "MAG:QF[QF01]:CURRENT:RB",
    "MAG:QD[QD01]:CURRENT:RB",
    "MAG:HCM[H01]:CURRENT:RB",
    "MAG:VCM[V01]:CURRENT:RB",
    "RF:CAVITY[C1]:VOLTAGE:RB",
)


def _render_machine_state(**context) -> dict:
    manager = TemplateManager()
    template = manager.jinja_env.get_template(TEMPLATE_PATH)
    rendered = template.render(**context)
    return json.loads(rendered)


def _channels(rendered: dict) -> dict:
    """Rendered entries minus the underscore-prefixed metadata keys (the
    real MachineStateReader skips these too; see
    src/osprey/services/machine_state/reader.py)."""
    return {k: v for k, v in rendered.items() if not k.startswith("_")}


@pytest.fixture(scope="module")
def manifest_addresses() -> set[str]:
    manifest = build_manifest()
    return {c["address"] for c in manifest["channels"]}


class TestRendersAsOneCanonicalList:
    def test_renders_as_valid_json_with_no_context(self):
        rendered = _render_machine_state()
        assert isinstance(rendered, dict)
        assert _channels(rendered), "expected at least one machine_state channel"

    @pytest.mark.parametrize(
        "pipeline_mode", ["in_context", "hierarchical", "middle_layer", None]
    )
    def test_output_is_identical_regardless_of_pipeline_mode(self, pipeline_mode):
        """The template no longer branches on default_pipeline -- passing any
        (or no) pipeline mode must render the same canonical channel set."""
        baseline = _channels(_render_machine_state())
        context = {"default_pipeline": pipeline_mode} if pipeline_mode is not None else {}
        rendered = _channels(_render_machine_state(**context))
        assert rendered == baseline


class TestManifestConsistency:
    """CC-4 regression guard: every rendered channel must be a real address."""

    def test_every_channel_is_in_the_manifest(self, manifest_addresses):
        channels = _channels(_render_machine_state())
        for address in channels:
            assert address in manifest_addresses, f"{address!r} not in manifest namespace"

    def test_no_fictional_addresses_remain(self):
        channels = _channels(_render_machine_state())
        for fictional in FICTIONAL_ADDRESSES:
            assert fictional not in channels, f"fictional address {fictional!r} reappeared"

    def test_addresses_follow_the_real_naming_grammar(self):
        channels = _channels(_render_machine_state())
        for address in channels:
            parts = address.split(":")
            assert len(parts) == 6, f"{address!r} does not match RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD"
            assert "[" not in address and "]" not in address


class TestChannelShape:
    def test_every_entry_has_label_and_group(self):
        channels = _channels(_render_machine_state())
        for address, defn in channels.items():
            assert defn.get("label"), f"{address!r} missing a non-empty label"
            assert defn.get("group"), f"{address!r} missing a non-empty group"

    def test_covers_the_representative_categories(self):
        """One representative channel per category named in the task:
        DCCT current, RF cavity voltage, a BPM pair, one corrector RB, and a
        representative vacuum pressure."""
        channels = _channels(_render_machine_state())
        groups = {defn["group"] for defn in channels.values()}
        assert {"beam", "rf", "orbit", "magnets", "vacuum"} <= groups

        addresses = set(channels)
        assert any("DCCT" in a and a.endswith(":CURRENT:RB") for a in addresses)
        assert any(":RF:CAVITY:" in a and a.endswith(":VOLTAGE:RB") for a in addresses)
        assert any(":DIAG:BPM:" in a and a.endswith(":POSITION:X") for a in addresses)
        assert any(":DIAG:BPM:" in a and a.endswith(":POSITION:Y") for a in addresses)
        assert any(":MAG:" in a and a.endswith(":CURRENT:RB") for a in addresses)
        assert any(":VAC:GAUGE:" in a and a.endswith(":PRESSURE:RB") for a in addresses)
