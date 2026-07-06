"""Generic SR PyAT lattice for the virtual accelerator.

Builds a storage-ring lattice whose device inventory (dipoles, quadrupole
families, correctors, BPMs) matches the namespace-union manifest's
pyat-coupled partition exactly (see inventory.py) -- the channel-finder DBs
fix the lattice, not vice versa. `orbit_response` provides the synchronous
corrector-kick -> BPM-readback contract (FR3) that the future IOC's SP write
handler calls into.
"""

from .response import AMPS_PER_RADIAN_KICK, orbit_response
from .ring import N_ARC_CELLS, RING_ENERGY_EV, build_ring

__all__ = [
    "build_ring",
    "orbit_response",
    "N_ARC_CELLS",
    "RING_ENERGY_EV",
    "AMPS_PER_RADIAN_KICK",
]
