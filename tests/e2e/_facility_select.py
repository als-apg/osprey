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


def bounded_corrector_of(view: SimulatorView) -> tuple[str, str, str] | None:
    """The first corrector of ``view``, in address order, that a limits record
    bounds at both ends and that names its own readback, as
    ``(setpoint, readback, model)``; ``None`` when no corrector qualifies.

    ``model`` is the model whose binding wires the setpoint: a lane that
    drives correctors drives that model's, and reads their response on that
    model's monitors.
    """
    for address in sorted(corrector_addresses(view)):
        channel = view.channel(address)
        readback = channel.pair
        if channel.value_range is not None and readback is not None and readback != address:
            binding = view.binding(address)
            assert binding is not None, f"corrector {address} has no served binding"
            return address, readback, binding.model
    return None


def bounded_corrector(repo: Path) -> tuple[str, str, str]:
    """:func:`bounded_corrector_of` the view the build rendered into ``repo``.

    The device a lane names when its request depends on a corrector having a
    range: the limits table holds only the records the facility tree authors,
    so a corrector without one has no range to sweep across.
    """
    found = bounded_corrector_of(repo_view(repo))
    if found is None:
        raise AssertionError(
            "no corrector has a limits record with both min_value and max_value "
            f"in the limits table of {repo / 'data' / 'facility' / 'limits.yaml'}"
        )
    return found


def seeded_readback(view: SimulatorView) -> str:
    """The first channel, in address order, that is not a setpoint, has a seed
    and is wired by no served model.

    Reading it returns a number near the seed's nominal: no model moves it.
    """
    for address in view.channels():
        if (
            view.channel(address).role != "setpoint"
            and view.seed(address) is not None
            and view.binding(address) is None
        ):
            return address
    raise AssertionError("the view has no seeded channel that no served model wires")


def bounded_setpoint(view: SimulatorView) -> tuple[str, float]:
    """The first writable setpoint, in address order, that a limits record
    bounds at both ends, with the midpoint of that band.

    The midpoint is a value the limits hook passes on, derived from the band
    rather than typed.
    """
    for address in view.channels():
        channel = view.channel(address)
        if channel.role == "setpoint" and channel.writable and channel.value_range is not None:
            low, high = channel.value_range
            return address, (float(low) + float(high)) / 2
    raise AssertionError("the view has no writable setpoint with a band at both ends")
