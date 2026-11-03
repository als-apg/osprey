"""Tests for the write time budget, :mod:`osprey.runtime.budget`."""

from __future__ import annotations

import logging

import pytest

import osprey.runtime as runtime
from osprey.runtime.budget import (
    MAX_WRITE_BOUND_S,
    WRITE_MARGIN_S,
    WriteBudget,
    bound_budget,
    write_budget,
)


def config(**blocks):
    return {"control_system": {"connector": blocks}}


def as_tuple(budget: WriteBudget) -> tuple[float, float, float, float, float]:
    return (budget.T, budget.S, budget.D, budget.grace_s, budget.relay_timeout_s)


class TestWriteBudget:
    @pytest.mark.parametrize(
        "cfg",
        [
            {},
            config(),
            {"control_system": "nonsense"},
            {"control_system": {"connector": "nonsense"}},
            config(epics={}),
        ],
    )
    def test_the_defaults_give_5_2_12_14_15(self, cfg):
        assert as_tuple(write_budget(cfg)) == (5.0, 2.0, 12.0, 14.0, 15.0)

    def test_a_block_raises_the_call_bound(self):
        budget = write_budget(config(epics={"timeout_s": 3}, virtual_accelerator={"timeout_s": 8}))
        assert as_tuple(budget) == (8.0, 2.0, 18.0, 20.0, 21.0)

    def test_the_maxima_come_from_different_blocks(self):
        budget = write_budget(
            config(
                epics={"timeout_s": 3, "step_read_timeout_s": 4},
                virtual_accelerator={"timeout_s": 7},
            )
        )
        assert as_tuple(budget) == (7.0, 4.0, 18.0, 20.0, 21.0)

    def test_a_block_below_the_defaults_does_not_lower_them(self):
        budget = write_budget(config(epics={"timeout_s": 1, "step_read_timeout_s": 0.5}))
        assert as_tuple(budget) == (5.0, 2.0, 12.0, 14.0, 15.0)

    def test_the_call_bound_is_clamped_so_d_stays_within_24(self):
        budget = write_budget(config(epics={"timeout_s": 60, "step_read_timeout_s": 4}))
        assert as_tuple(budget) == (10.0, 4.0, 24.0, 26.0, 27.0)

    def test_the_step_read_is_capped_so_the_call_bound_keeps_one_second(self):
        budget = write_budget(config(epics={"timeout_s": 30, "step_read_timeout_s": 90}))
        assert as_tuple(budget) == (1.0, 22.0, 24.0, 26.0, 27.0)

    def test_a_long_step_read_alone_clamps_the_default_call_bound(self):
        budget = write_budget(config(epics={"step_read_timeout_s": 90}))
        assert (budget.T, budget.S, budget.D) == (1.0, 22.0, 24.0)

    @pytest.mark.parametrize(
        "blocks",
        [
            {"epics": {"timeout_s": 6}},
            {"epics": {"timeout_s": 60, "step_read_timeout_s": 4}},
            {"epics": {"timeout_s": 30, "step_read_timeout_s": 90}},
            {"epics": {"timeout_s": 0.25}},
        ],
    )
    def test_d_plus_the_margin_fits_the_grace_and_the_relay(self, blocks):
        budget = write_budget(config(**blocks))
        assert budget.D == budget.S + 2 * budget.T
        assert budget.D <= MAX_WRITE_BOUND_S
        assert budget.D + WRITE_MARGIN_S <= budget.grace_s
        assert budget.D + WRITE_MARGIN_S <= budget.relay_timeout_s <= 27.0

    def test_a_block_that_is_not_a_mapping_is_skipped_with_a_warning(self, caplog):
        caplog.set_level(logging.WARNING)
        budget = write_budget(config(tango="nonsense"))
        assert as_tuple(budget) == (5.0, 2.0, 12.0, 14.0, 15.0)
        assert "control_system.connector.tango" in caplog.text

    def test_a_block_with_an_unreadable_call_bound_is_skipped_with_a_warning(self, caplog):
        caplog.set_level(logging.WARNING)
        budget = write_budget(
            config(
                epics={"timeout": 30, "step_read_timeout_s": 9},
                tango={"timeout_s": -1},
            )
        )
        assert as_tuple(budget) == (5.0, 2.0, 12.0, 14.0, 15.0)
        assert "control_system.connector.epics.timeout is renamed" in caplog.text
        assert "control_system.connector.tango.timeout_s" in caplog.text

    def test_a_malformed_block_leaves_a_good_one_counting(self, caplog):
        caplog.set_level(logging.WARNING)
        budget = write_budget(config(tango="nonsense", epics={"timeout_s": 8}))
        assert as_tuple(budget) == (8.0, 2.0, 18.0, 20.0, 21.0)

    def test_the_budget_is_frozen(self):
        budget = write_budget({})
        with pytest.raises(AttributeError):
            budget.T = 1.0  # type: ignore[misc]

    def test_the_budget_stays_off_the_script_surface(self):
        """A panel's session derives its bounds; a script's ``osprey.runtime`` does not offer it."""
        for name in ("write_budget", "bound_budget", "WriteBudget"):
            assert name not in runtime.__all__
            assert not hasattr(runtime, name)


class TestBoundBudget:
    def test_a_block_below_the_budget_keeps_its_own_values(self):
        budget = write_budget(config(epics={"timeout_s": 8, "step_read_timeout_s": 4}))
        assert bound_budget({"timeout_s": 3, "step_read_timeout_s": 1}, budget) == (
            3.0,
            1.0,
            7.0,
        )

    def test_a_block_above_the_budget_is_bounded_by_it(self):
        budget = write_budget(config(epics={"timeout_s": 60, "step_read_timeout_s": 4}))
        assert bound_budget({"timeout_s": 60, "step_read_timeout_s": 4}, budget) == (
            10.0,
            4.0,
            24.0,
        )

    def test_a_step_read_beyond_the_cap_is_bounded_by_s(self):
        budget = write_budget(config(epics={"timeout_s": 30, "step_read_timeout_s": 90}))
        assert bound_budget({"timeout_s": 30, "step_read_timeout_s": 90}, budget) == (
            1.0,
            22.0,
            24.0,
        )

    @pytest.mark.parametrize("block", [None, {}])
    def test_a_block_that_declares_nothing_takes_the_defaults(self, block):
        assert bound_budget(block, write_budget({})) == (5.0, 2.0, 12.0)

    def test_the_bound_never_exceeds_the_budget(self):
        budget = write_budget(config(epics={"timeout_s": 6, "step_read_timeout_s": 3}))
        t_b, s_b, d_b = bound_budget({"timeout_s": 9, "step_read_timeout_s": 1}, budget)
        assert (t_b, s_b, d_b) == (6.0, 1.0, 13.0)
        assert d_b <= budget.D

    def test_an_unreadable_call_bound_is_refused(self):
        with pytest.raises(ValueError, match=r"control_system\.connector\.epics\.timeout"):
            bound_budget({"timeout": 3}, write_budget({}), "epics")

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize(
        ("key", "match"),
        [
            pytest.param(
                "step_read_timeout_s",
                r"control_system\.connector\.epics\.step_read_timeout_s must be a finite",
                id="step-read",
            ),
            pytest.param(
                "timeout_s", r"control_system\.connector\.epics\.timeout_s", id="call-bound"
            ),
        ],
    )
    def test_a_non_finite_bound_is_refused_naming_its_key(self, key, match, value):
        with pytest.raises(ValueError, match=match):
            bound_budget({key: value}, write_budget({}), "epics")

    def test_a_refused_step_read_names_a_placeholder_without_a_type(self):
        with pytest.raises(
            ValueError, match=r"control_system\.connector\.<type>\.step_read_timeout_s"
        ):
            bound_budget({"step_read_timeout_s": float("nan")}, write_budget({}))

    @pytest.mark.parametrize("field", ["T", "S"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_budget_bound_is_refused(self, field, value):
        bounds = {"T": 5.0, "S": 2.0}
        bounds[field] = value
        budget = WriteBudget(
            T=bounds["T"], S=bounds["S"], D=12.0, grace_s=14.0, relay_timeout_s=15.0
        )
        with pytest.raises(ValueError, match=rf"the write budget's {field} must be a finite"):
            bound_budget({}, budget, "epics")

    def test_finite_values_are_still_bounded(self):
        assert bound_budget({"timeout_s": 3, "step_read_timeout_s": 1.5}, write_budget({})) == (
            3.0,
            1.5,
            7.5,
        )
