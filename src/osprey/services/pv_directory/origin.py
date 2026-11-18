"""Join directory names to the curated roster: the origin mark on every All PVs row.

``curated`` means the facility file declares the channel, ``directory`` means the
control system knows it and the facility file does not, and ``unknown`` means there
is no roster to say which. The roster comes only from
:func:`osprey.channel_roster.registered_channels`. The join is on the normalised
name as spelled, with no alias resolution, and it never writes to the roster.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from osprey.channel_roster import registered_channels

__all__ = ["NormalisedAddress", "Origin", "mark", "normalise_address"]

Origin = Literal["curated", "directory", "unknown"]

_PROTOCOL_PREFIXES = ("ca://", "pva://")

#: An EPICS field suffix: upper-case, with the ``$`` long-string modifier allowed. Any
#: other dot belongs to the name, as a facility's own dotted address does.
_FIELD_SUFFIX = re.compile(r"(?P<name>.+)\.(?P<field>[A-Z][A-Z0-9_]*\$?)")


@dataclass(frozen=True, slots=True)
class NormalisedAddress:
    """An address as the join reads it.

    Attributes:
        name: The PV name: no protocol prefix, no field suffix.
        field: The ``.FIELD`` suffix without its dot, kept for display, or ``None``.
    """

    name: str
    field: str | None = None

    @property
    def display(self) -> str:
        """The name with its field, without the protocol prefix."""
        return self.name if self.field is None else f"{self.name}.{self.field}"


def normalise_address(address: str) -> NormalisedAddress:
    """Split *address* into its PV name and field: ``pva://X.VAL`` -> ``X``, ``VAL``."""
    bare = address.strip()
    for prefix in _PROTOCOL_PREFIXES:
        bare = bare.removeprefix(prefix)
    suffixed = _FIELD_SUFFIX.fullmatch(bare)
    if suffixed is None:
        return NormalisedAddress(bare)
    return NormalisedAddress(suffixed["name"], suffixed["field"])


def mark(names: Iterable[str], config: dict[str, Any]) -> dict[str, Origin]:
    """Mark each of *names* by whether this project's roster declares it.

    Args:
        names: Directory names, in any spelling :func:`normalise_address` reads.
        config: Full project configuration dictionary.

    Returns:
        One origin per name, keyed by the name as given. Every name is ``unknown``
        when the roster is absent, because an absent roster cannot tell a curated
        name from an uncurated one.
    """
    roster = registered_channels(config)
    if roster.absence is not None:
        return dict.fromkeys(names, "unknown")
    curated = {normalise_address(address).name for address in roster.addresses}
    return {
        name: "curated" if normalise_address(name).name in curated else "directory"
        for name in names
    }
