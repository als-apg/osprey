"""What the band script decides about a supply's polarity, and what it says.

``scripts/va/derive_bands.py`` floors a band minimum at zero for a family the
tree states runs on a unipolar supply, and it reads that statement off the
nominal currents the tree carries. The rule is evidence about supplies, not
about ring stability, and the evidence has edges: a bipolar family that
happens to be parked non-negative looks exactly like a unipolar one, and a
single stated nominal a hair below zero takes the floor away from every device
of its family.

Neither case can be decided from the nominals alone, so the run reports what
it decided and what decided it. These cases hold that report to saying so --
they are about what a reader is told, which is the only defence a silent
decision has.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

DERIVE_BANDS = Path(__file__).resolve().parents[2] / "scripts" / "va" / "derive_bands.py"


@pytest.fixture(scope="module")
def script():
    """The band script, loaded from the checkout as a module."""
    spec = importlib.util.spec_from_file_location("_derive_bands_floor", DERIVE_BANDS)
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses resolve their own module while the class body runs, so
    # the module has to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: One family whose stated nominals are all above zero, one parked entirely at
#: zero, one openly bipolar, and one whose single negative nominal is small
#: enough to be arithmetic rather than a polarity.
NOMINALS = {
    "ONE_POLARITY": [271.856, 288.222, 280.0],
    "AT_REST": [0.0, 0.0],
    "BOTH_POLARITIES": [12.0, -12.0],
    "ROUNDED_BELOW_ZERO": [100.0, -1e-12],
}


class TestWhichFamiliesTheFloorCovers:
    def test_a_family_stated_only_above_zero_is_floored(self, script):
        assert "ONE_POLARITY" in script.unipolar_families(NOMINALS)

    def test_a_family_stated_on_both_polarities_is_not(self, script):
        assert "BOTH_POLARITIES" not in script.unipolar_families(NOMINALS)

    def test_a_family_parked_entirely_at_zero_is_not(self, script):
        """Zero is the same number on either polarity, so a family at rest
        states nothing for the rule to read."""
        assert "AT_REST" not in script.unipolar_families(NOMINALS)

    def test_one_nominal_of_round_off_size_below_zero_drops_the_floor(self, script):
        """The rule reads the sign it is given. A nominal of -1e-12 is a
        polarity to it, and the whole family loses its floor over it."""
        assert "ROUNDED_BELOW_ZERO" not in script.unipolar_families(NOMINALS)


class TestTheRunSaysWhatItDecidedAndWhy:
    """A floor that is applied, and a floor that is dropped, are both reported
    with the stated nominals behind them -- the two cases a reader cannot
    otherwise tell from the bands alone."""

    @staticmethod
    def _reported(script, family: str) -> str:
        floored = script.unipolar_families(NOMINALS)
        decisions = script.floor_decisions(NOMINALS, [family], floored)
        lines = script.format_floor_decisions(decisions)
        assert len(lines) == 1
        return lines[0]

    def test_a_floored_family_is_reported_with_its_nominals(self, script):
        line = self._reported(script, "ONE_POLARITY")
        assert "ONE_POLARITY" in line
        assert "floored at 0 A" in line
        assert "3 stated nominal(s)" in line
        assert "271.856" in line and "288.222" in line

    def test_an_unfloored_family_is_reported_with_the_reason_it_lost_the_floor(self, script):
        line = self._reported(script, "BOTH_POLARITIES")
        assert "not floored" in line
        assert "below zero" in line

    def test_the_round_off_negative_is_visible_in_the_report(self, script):
        """The dangerous case: the band keeps a negative minimum and nothing
        else in the output would say the floor had been dropped."""
        line = self._reported(script, "ROUNDED_BELOW_ZERO")
        assert "not floored" in line
        assert "a stated nominal is below zero" in line
        assert "-1e-12" in line

    def test_a_family_named_on_the_command_line_is_reported_as_floored(self, script):
        """An override is reported as the run applied it, not as the tree's
        own nominals would have had it."""
        decisions = script.floor_decisions(
            NOMINALS, ["BOTH_POLARITIES"], frozenset({"BOTH_POLARITIES"})
        )
        assert decisions[0].floored
        assert "floored at 0 A" in script.format_floor_decisions(decisions)[0]

    def test_a_family_the_tree_states_no_nominal_for_says_so(self, script):
        decisions = script.floor_decisions(NOMINALS, ["UNSTATED"], frozenset())
        assert decisions[0].stated == 0
        line = script.format_floor_decisions(decisions)[0]
        assert "states no nominal" in line
        assert "no stated nominal" in line

    def test_every_swept_family_gets_exactly_one_line(self, script):
        families = list(NOMINALS)
        decisions = script.floor_decisions(NOMINALS, families + families, frozenset())
        assert [decision.family for decision in decisions] == sorted(families)
