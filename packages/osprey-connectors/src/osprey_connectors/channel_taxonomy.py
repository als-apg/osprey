"""Shared procedural channel taxonomy for the mock connectors.

Both the mock control-system connector and the mock archiver connector need to
guess plausible defaults for channels that the data-driven simulation engine does
*not* serve. They classify a channel by substring conventions in its address and
map it to a physical base value and engineering units. The conventions happen to
read like EPICS names, but nothing here is protocol-specific: a DOOCS property
address classifies by the same substrings.

Neither connector may hold its own copy of this keyword ladder: two copies drift
apart branch by branch, and the same channel then classifies one way for live
reads and another for archived ones. This module is the single source of truth:
:func:`classify_channel` returns a :class:`ChannelKind`, and each connector layers
its own behaviour on top — the archiver shapes a time series per
:attr:`ChannelKind.name`, the control connector seeds an initial value and reports
units.

The kind also carries :attr:`ChannelKind.noise_scale`, the absolute floor under
the otherwise purely relative sigma, so a kind whose base value is legitimately
``0.0`` is not silently dead-flat. :meth:`ChannelKind.noise_sigma` combines the
two and is the single implementation both noise-applying fallbacks (the mock
control connector and the Virtual Accelerator's engine source) call.
"""

from dataclasses import dataclass

__all__ = ["POSITION_NOISE_SCALE", "ChannelKind", "classify_channel"]

POSITION_NOISE_SCALE = 0.005
"""Absolute noise floor for the ``position`` kind, in mm.

Roughly a BPM's single-shot resolution. Position is the one kind whose
:attr:`ChannelKind.base_value` is legitimately ``0.0`` (a centred orbit), which
makes the relative noise the fallback paths apply — ``abs(base) * noise_level`` —
identically zero, so the channel reads back dead-flat however noisy it is
declared to be. The floor is deliberately per-kind rather than global: base
values span 1e-9 Torr to 5000 V, and any single absolute number would be
overwhelming noise at one end of that range and invisible at the other.
"""


@dataclass(frozen=True)
class ChannelKind:
    """A procedural channel classification.

    Attributes:
        name: Canonical kind, used by the archiver to pick a synthesis shape.
        base_value: Plausible steady-state value for the kind.
        units: Engineering units string (empty for the generic default).
        noise_scale: Absolute lower bound on the kind's noise sigma, in the
            kind's own units. Non-zero only for kinds whose ``base_value`` can
            legitimately be ``0.0``; every other kind leaves it at ``0.0``, so
            ``max(abs(base) * noise_level, noise_scale)`` is arithmetically
            identical to the plain relative sigma there.
    """

    name: str
    base_value: float
    units: str
    noise_scale: float = 0.0

    def noise_sigma(self, base_value: float, noise_level: float) -> float:
        """Sigma a noise-applying fallback should draw with for this kind.

        The relative term ``abs(base_value) * noise_level`` is dead at a ``0.0``
        base, so kinds whose base can legitimately be zero floor it with
        :attr:`noise_scale`; every other kind's floor is ``0.0``, leaving the
        sigma at the plain relative one. A ``noise_level`` of exactly ``0.0`` is
        an explicit request for determinism, though, so it disables the floor
        rather than being overridden by it.

        Args:
            base_value: The value the noise will be added to. Callers pass the
                value they actually hold, which need not be :attr:`base_value`
                (the mock control connector's may have been written to).
            noise_level: Relative noise fraction.

        Returns:
            The sigma to draw with; ``0.0`` when ``noise_level`` is not positive.
        """
        if noise_level <= 0.0:
            return 0.0
        return max(abs(base_value) * noise_level, self.noise_scale)


def classify_channel(channel: str) -> ChannelKind:
    """Classify a channel address into a :class:`ChannelKind` by naming convention.

    The checks are ordered: more specific kinds (beam current) are tested before
    more general ones (current). A name matching nothing falls through to the
    generic ``default`` kind.
    """
    lower = channel.lower()

    if ("beam" in lower and "current" in lower) or "dcct" in lower:
        return ChannelKind("beam_current", 500.0, "mA")
    if "current" in lower:
        return ChannelKind("current", 150.0, "A")
    if "voltage" in lower:
        return ChannelKind("voltage", 5000.0, "V")
    if "power" in lower:
        return ChannelKind("power", 50.0, "kW")
    if "pressure" in lower:
        return ChannelKind("pressure", 1e-9, "Torr")
    if "temp" in lower:
        return ChannelKind("temperature", 25.0, "°C")
    if "lifetime" in lower:
        return ChannelKind("lifetime", 10.0, "hours")
    if "position" in lower or "pos" in lower:
        return ChannelKind("position", 0.0, "mm", noise_scale=POSITION_NOISE_SCALE)
    if "energy" in lower:
        return ChannelKind("energy", 1900.0, "MeV")
    return ChannelKind("default", 100.0, "")
