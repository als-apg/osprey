"""The shared controls of a service tile's bar: the ids the build and the terminal share.

``web.tile_bar.controls`` lists the shared controls each service tile's bar
shows after the panel's own icons, left to right. The terminal reads the key
(:func:`osprey.interfaces.web_terminal.app.tile_bar_controls`) and stamps the
list on ``<html data-tile-bar-controls>``. ``osprey build`` and ``osprey
validate`` refuse a value the bar cannot draw
(:func:`osprey.cli.build_profile_tile_bar.tile_bar_control_errors`). Both take
the ids and the verdict from this module, so the build accepts exactly the ids
the page draws.

The module imports nothing, so the build reaches it without the web
application.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_TILE_BAR_CONTROLS",
    "TILE_BAR_CONTROLS",
    "TILE_BAR_CONTROLS_KEY",
    "tile_bar_controls_refusal",
]

#: The dotted config key.
TILE_BAR_CONTROLS_KEY = "web.tile_bar.controls"

#: Every shared control the tile bar draws, by id. ``ask_agent`` opens the
#: Agent pane with a question about the panel, ``reload`` reloads the panel,
#: and ``help`` opens the panel's page in the deployment's documentation.
TILE_BAR_CONTROLS: tuple[str, ...] = ("ask_agent", "reload", "help")

#: The controls of a deployment that does not set the key.
DEFAULT_TILE_BAR_CONTROLS: tuple[str, ...] = ("ask_agent",)


def tile_bar_controls_refusal(value: object) -> str | None:
    """Why *value* cannot be ``web.tile_bar.controls``, or ``None`` when it can.

    A list of ids from :data:`TILE_BAR_CONTROLS` passes in any order, and the
    empty list passes too: it gives a bar without shared controls. ``None``
    (the key written with no value) reads as absent, so the default applies.
    A repeated id passes, and the reader keeps its first place.

    Args:
        value: The configured value, as YAML parsed it.

    Returns:
        The rest of a sentence that starts with the key: what is wrong, and the
        ids to use. ``None`` when the value is good.
    """
    if value is None:
        return None
    allowed = ", ".join(TILE_BAR_CONTROLS)
    if not isinstance(value, list):
        return f"is {value!r}, not a list. Write a list of ids from {allowed}."
    unknown = [entry for entry in value if entry not in TILE_BAR_CONTROLS]
    if not unknown:
        return None
    named = ", ".join(repr(entry) for entry in unknown)
    return f"names {named}, which the tile bar does not draw. Use ids from {allowed}."
