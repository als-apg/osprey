"""Facility questions the e2e lanes ask of a built render.

A lane asks the built facility for a kind of channel -- a seeded reading, a
bounded writable setpoint, a corrector a limits record bounds -- and never
names an address. Every answer is read from the simulator view the build
rendered: each channel's role, ``writable``, ``value_range`` (set only when its
limits record states both ends), its seed, and the bindings of the served
models. A facility change therefore moves a lane's channel with it instead of
leaving the lane holding a stale address.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey_connectors.simulation.view import SimulatorView


@cache
def _view_of_resolved(path: Path) -> SimulatorView:
    """The memoized open behind :func:`repo_view`, keyed on a resolved view directory."""
    from osprey_connectors.simulation.view import SimulatorView

    return SimulatorView.open(path)


def repo_view(repo: Path) -> SimulatorView:
    """The simulator view the build rendered into ``repo``.

    The one authority on which channels the accelerator model drives and what
    each of them does to it: every binding of a served model names the
    address, its role and the transverse plane it steers or reads. A lane
    choosing devices with a plane or a role in mind reads them from here.

    Opened once per view directory per process, keyed on the resolved path: a
    lane that rebuilds a repo already read here must build it under a fresh
    path to be served the new view.

    Lives HERE rather than in the product because choosing a physics-appropriate
    subset of a facility's channels is a harness concern: the staged view
    holds every channel and says which way each points, and which of them a
    given plan can do physics with is the lane's own question.
    """
    from osprey_connectors.simulation.view import SimulatorView

    path = SimulatorView.path_for_project(repo)
    if not path.is_dir():
        raise AssertionError(f"the build rendered no simulator view at {path}")
    return _view_of_resolved(path.resolve())


def corrector_addresses(
    view: SimulatorView, *, plane: str | None = None, model: str | None = None
) -> frozenset[str]:
    """Every corrector setpoint the view's bindings serve, optionally of one
    plane and of one model.

    A corrector is a setpoint binding that steers a transverse plane. Grouping
    by what a binding DOES -- steer a plane, read one -- rather than by the
    family name its address spells is what keeps a lane's device choice the
    same question on every facility.
    """
    return frozenset(
        binding.address
        for binding in view.bindings(role="setpoint", model=model)
        if binding.plane is not None and (plane is None or binding.plane == plane)
    )


def monitor_addresses(
    view: SimulatorView, *, plane: str | None = None, model: str | None = None
) -> frozenset[str]:
    """Every monitor reading the view's bindings serve, optionally of one plane
    and of one model."""
    return frozenset(
        binding.address for binding in view.bindings(role="monitor", plane=plane, model=model)
    )


def claimed_addresses(view: SimulatorView) -> frozenset[str]:
    """Every corrector setpoint and monitor reading the view's bindings serve.

    The single spelling of the channels a corrector-sweeping plan can do
    physics with, so a lane narrowing the staged devices and a lane selecting
    them read the rule from here.
    """
    return corrector_addresses(view) | monitor_addresses(view)


def bounded_corrector(repo: Path) -> tuple[str, str, str]:
    """The first corrector, in address order, that a limits record bounds at
    both ends and that names its own readback, as ``(setpoint, readback, model)``.

    The device a lane names when its request depends on a corrector having a
    range: the limits table holds only the records the facility tree authors,
    so a corrector without one has no range to sweep across. ``model`` is the
    model whose binding wires the setpoint, so a lane reads the sweep's
    response on that model's monitors.
    """
    view = repo_view(repo)
    for address in sorted(corrector_addresses(view)):
        channel = view.channel(address)
        readback = channel.pair
        if channel.value_range is not None and readback is not None and readback != address:
            binding = view.binding(address)
            assert binding is not None, f"corrector {address} has no served binding"
            return address, readback, binding.model
    raise AssertionError(
        "no corrector has a limits record with both min_value and max_value "
        f"in the limits table of {repo / 'data' / 'facility' / 'limits.yaml'}"
    )
