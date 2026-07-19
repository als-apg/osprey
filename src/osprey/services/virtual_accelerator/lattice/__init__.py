"""ALS-U AR ring for the virtual accelerator.

`build_ring` adapts the hand-ported real ring (see `ring.py`) for this
service, validating its device inventory against the manifest's
pyat-coupled partition (see `inventory.py`) -- the facility spec fixes the
expected inventory, not vice versa. `orbit_response` provides the
synchronous corrector-kick -> BPM-readback contract (FR3) that the IOC's SP
write handler calls into.
"""

from .response import AMPS_PER_RADIAN_KICK, orbit_response
from .ring import build_ring

__all__ = [
    "build_ring",
    "orbit_response",
    "AMPS_PER_RADIAN_KICK",
]
