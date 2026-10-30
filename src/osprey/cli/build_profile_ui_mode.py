"""``web.expert_view`` against ``web.ui_mode``: one deployment, one answer.

``web.ui_mode`` picks the layout a terminal opens in; ``web.expert_view: false``
says the deployment offers no Expert view at all, and the web terminal then
resolves the mode to ``simple`` whatever ``web.ui_mode`` says
(``osprey.interfaces.web_terminal.app.resolve_ui_mode``). A profile that states
both ``web.expert_view: false`` and ``web.ui_mode: expert`` — whether in one
file or a tier or app laid over its deployment — is two spellings of one fact
disagreeing, so the build refuses it naming both lines, the rule
:func:`osprey.cli.build_profile_panels.panel_selection_errors` applies to a
panel's ``enabled``.
"""

from __future__ import annotations

from typing import Any

from .build_profile_reach import spelled_values

__all__ = ["EXPERT_VIEW_KEY", "UI_MODE_KEY", "expert_view_errors"]

#: The deployment switch that rules the Expert view out when ``false``.
EXPERT_VIEW_KEY = "web.expert_view"
#: The layout a terminal opens in.
UI_MODE_KEY = "web.ui_mode"

#: The quoted spellings the web terminal reads as ``false``
#: (``app.coerce_config_flag``), so the build refuses what the server would honour.
_FALSE_WORDS = frozenset({"false", "no", "off", "0"})


def _says_false(value: Any) -> bool:
    if value is False:
        return True
    return isinstance(value, str) and value.strip().lower() in _FALSE_WORDS


def expert_view_errors(config: Any) -> list[str]:
    """Refuse ``web.ui_mode: expert`` beside ``web.expert_view: false``.

    Every spelling of either key (dotted, prefix-over-mapping, nested) is
    found, so the refusal names the exact lines. An absent ``web.ui_mode`` is
    not refused: the default it stands for is simply resolved to ``simple``.

    Args:
        config: The profile's ``config:`` block, whatever shape it parsed as.

    Returns:
        One error per ``expert`` spelling while the Expert view is ruled out.
    """
    ruled_out = [
        spelling
        for spelling, value in spelled_values(config, EXPERT_VIEW_KEY)
        if _says_false(value)
    ]
    if not ruled_out:
        return []
    return [
        f"{spelling}: 'expert' but {ruled_out[0]}: false rules the Expert view out on "
        f"this deployment. Set {UI_MODE_KEY}: simple, or drop the line."
        for spelling, value in spelled_values(config, UI_MODE_KEY)
        if value == "expert"
    ]
