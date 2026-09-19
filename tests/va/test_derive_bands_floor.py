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


class _StubBinding:
    """One binding, carrying only what the floor report reads off it."""

    def __init__(self, family: str, nominal: float) -> None:
        self.family = family
        self.nominal = nominal


class _StubDocument:
    def __init__(self, bindings: list[_StubBinding]) -> None:
        self.bindings = bindings


class _StubSweeper:
    """A tree stated in nominals alone, which is all the floor rule reads.

    The sweep itself is physics over a real ring and is exercised elsewhere;
    what is under test here is whether a mode reports the decision it applied,
    so the sweep is stubbed out and the nominals are the whole input.
    """

    def __init__(self, nominals: dict[str, list[float]]) -> None:
        self._by_address = {
            f"{family}:{ordinal}": _StubBinding(family, value)
            for family, values in nominals.items()
            for ordinal, value in enumerate(values, start=1)
        }
        self.document = _StubDocument(list(self._by_address.values()))

    @property
    def addresses(self) -> list[str]:
        return list(self._by_address)

    def binding(self, address: str):
        return self._by_address[address]


class TestEveryModeSaysIt:
    """Each of the script's three modes applies the floor, so each prints it.

    A mode that applied the floor without saying so would hand its reader a
    band whose lower edge has two possible meanings and no way to tell them
    apart -- and a comparison against committed bands is exactly where that
    matters, because a floored edge agreeing with a committed one is a
    different fact from two derived edges agreeing.
    """

    @pytest.fixture
    def sweeper(self):
        return _StubSweeper({"ONE_POLARITY": [271.856, 288.222], "BOTH_POLARITIES": [12.0, -12.0]})

    @pytest.fixture
    def no_sweep(self, script, monkeypatch):
        """Stand in for the physics sweep; the modes are under test, not it."""
        monkeypatch.setattr(script, "derive_bands", lambda *args, **kwargs: {})

    @staticmethod
    def _run(script, mode, sweeper, tmp_path):
        floored = script.unipolar_families(script.nominals_by_family(sweeper.document))
        if mode == "verify":
            committed = tmp_path / "channel_limits.json"
            committed.write_text("{}", encoding="utf-8")
            return script._run_verify(
                sweeper, sweeper.addresses, floored, committed, tol=script.DEFAULT_VERIFY_TOL
            )
        if mode == "check":
            return script._run_check(sweeper, sweeper.addresses, floored)
        return script._run_derive_all(sweeper, sweeper.addresses, floored, str(tmp_path / "out"))

    @pytest.mark.parametrize("mode", ["derive", "check", "verify"])
    def test_the_mode_prints_a_floor_decision_for_every_swept_family(
        self, script, sweeper, no_sweep, tmp_path, capsys, mode
    ):
        assert self._run(script, mode, sweeper, tmp_path) == 0

        reported = capsys.readouterr().err
        assert "ONE_POLARITY" in reported
        assert "floored at 0 A" in reported
        assert "BOTH_POLARITIES" in reported
        assert "not floored" in reported
