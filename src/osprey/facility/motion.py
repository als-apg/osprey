"""How far a channel's declared motion can carry its value from the value it holds.

A simulated channel's seed may declare two kinds of motion: ``noise``, the
additive Gaussian sigma in the channel's unit, and ``drift``, a slow wander
whose ``amplitude`` bounds it (``|wander| <= amplitude`` by construction). A
served value is therefore the held value plus at most the drift amplitude plus
a noise draw, and a readback counts as settled on a demand when::

    |readback - demand| <= |drift.amplitude| + MOTION_SIGMAS x |noise|

using the readback's own seed. A seed that declares no motion has a band of
0.0, so a caller's own floor governs that channel.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["MOTION_SIGMAS", "settle_band"]

#: Gaussian headroom on a seed's ``noise``: a per-draw miss rate near 1e-9.
MOTION_SIGMAS = 6.0


def settle_band(seed: Mapping[str, Any] | None) -> float:
    """The band a value moving as ``seed`` declares stays within around its held value.

    Args:
        seed: A channel's ``simulation`` seed, or None.

    Returns:
        ``|drift.amplitude| + MOTION_SIGMAS x |noise|``; 0.0 for an absent seed
        or a seed with neither key.
    """
    if not seed:
        return 0.0
    drift = seed.get("drift") or {}
    amplitude = abs(float(drift.get("amplitude") or 0.0))
    return amplitude + MOTION_SIGMAS * abs(float(seed.get("noise") or 0.0))
