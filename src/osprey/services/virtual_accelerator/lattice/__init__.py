"""The lattice the virtual accelerator serves, and what it is driven with.

`build_ring` (see `ring.py`) loads the ring a served tree carries and checks
it against the bindings document derived against it -- the facility's own
exported lattice, not a ring this package describes. `orbit_response` (see
`response.py`) sweeps one bound actuator on a served model and reads the
monitors the bindings name, and `orbit_responses` sweeps several in one pass:
they are the verify oracle the facility's exported response matrix is checked
against, and no serving path calls into them.
"""

from .response import orbit_response, orbit_responses
from .ring import build_ring

__all__ = [
    "build_ring",
    "orbit_response",
    "orbit_responses",
]
