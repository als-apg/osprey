"""The precondition of every live lane that compares a reading: nothing it reads moves.

A lane that measures a response, or holds a served reading to a model, starts
its deployment in the built-in ``still`` scenario and asserts, before it
measures, that every reading it compares has motion envelope 0 under the
active set. Without the assertion the rule is unverifiable: a seed or a
scenario that gains motion would turn a parity failure into noise, and a lane
that then widened a tolerance would stop measuring the model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from osprey_connectors.simulation.state import STILL_SCENARIO


def assert_readings_still(view: Any, addresses: Iterable[str], active: Sequence[str]) -> None:
    """Fail unless every address in ``addresses`` serves without motion under ``active``.

    Args:
        view: The :class:`~osprey_connectors.simulation.view.SimulatorView` the
            served machine is rendered from.
        addresses: The readings the lane compares.
        active: The active scenario names, in order; ``still`` must be one.
    """
    names = {str(scenario["name"]) for scenario in view.scenarios()}
    assert STILL_SCENARIO in names, (
        f"the view lists no {STILL_SCENARIO!r} scenario: rebuild the render"
    )
    assert STILL_SCENARIO in active, (
        f"the active set is {list(active)}, not one that names {STILL_SCENARIO!r}"
    )
    moving = {
        address: band
        for address in sorted(set(addresses))
        if (band := view.motion_envelope(address, active=tuple(active))) != 0.0
    }
    assert not moving, (
        f"readings this lane compares still move under {list(active)}: {moving}; "
        "the comparison would be measuring motion, not the model"
    )
