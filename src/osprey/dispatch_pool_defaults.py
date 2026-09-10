"""Single source of truth for the event dispatcher's pool limits.

Three places state how many agent runs a dispatcher carries and how deep its
queue goes: the runtime dataclass, the ``triggers.yml`` loader's fallback, and
the build profile's ``dispatch:`` block. A hand-run dispatcher and a built one
must start at the same posture, so all three read the constants here.

This leaf imports nothing from ``osprey`` and nothing outside the standard
library. That is what lets the CLI's profile schema — which the ``build``,
``config`` and ``init`` commands import on every invocation — share the numbers
without pulling ``osprey.dispatch`` and its HTTP worker client into the profile
import graph.
"""

#: Agent runs the dispatcher will carry at once when nothing says otherwise.
DEFAULT_MAX_CONCURRENT_RUNS: int = 2

#: Events the dispatcher will hold waiting for a slot before it refuses more.
DEFAULT_MAX_QUEUE_DEPTH: int = 50
