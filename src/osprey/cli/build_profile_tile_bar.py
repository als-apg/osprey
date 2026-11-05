"""``web.tile_bar.controls`` judged on the profile's merged ``config:`` block.

:meth:`osprey.cli.build_profile_model.BuildProfile.validate` calls
:func:`tile_bar_control_errors` beside the ``web.panels`` checks, so
``osprey build`` and ``osprey validate`` refuse a value the tile bar cannot
draw while it is still a line in the profile. The terminal reads the same key
and drops such a value with a warning; the ids and the verdict both come from
:mod:`osprey.interfaces.web_terminal.tile_bar`, so the two cannot disagree.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osprey.interfaces.web_terminal.tile_bar import (
    TILE_BAR_CONTROLS_KEY,
    tile_bar_controls_refusal,
)


def tile_bar_control_errors(config: Any) -> list[str]:
    """One message per spelling of ``web.tile_bar.controls`` the bar cannot draw.

    Every spelling of the key is read — dotted, a dotted prefix over a mapping,
    nested — so the message names the line to fix however it was written.

    Args:
        config: The profile's merged ``config:`` block, whatever shape it parsed as.

    Returns:
        Messages naming the key as written, the ids the bar does not draw and
        the ids to use. Empty when every spelling is good.
    """
    if not isinstance(config, Mapping):
        return []

    # Imported in-function: the reach registry behind `spelled_values` pulls
    # the whole service-resolution package in with it.
    from .build_profile_reach import spelled_values

    return [
        f"`{spelling}` {refusal}"
        for spelling, value in spelled_values(config, TILE_BAR_CONTROLS_KEY)
        if (refusal := tile_bar_controls_refusal(value)) is not None
    ]
