"""The ``web_panels`` selection, projected onto a render's ``web.panels`` blocks.

A render's tab strip is read from the rendered ``web.panels.<id>`` mapping and
nothing else (``interfaces/web_terminal/app.py``), while what a profile
*selects* is its ``web_panels:`` list. The two are one fact in two places, and
they only stay one fact if the build carries the selection across: every block
the render ends up with — the template's own ``enabled: true`` for a selected
builtin, a ``config:`` fact about a custom panel's address, a fact a persona
inherits for a tab it excluded, a block an injector writes into a deploying
render for a tab only its personas select — is told whether this render shows
it. Facts stay where they are (a persona cannot subtract ``config:``, and the
Reach Contract copies addresses from the host's render), so nothing about
inheritance moves; only ``enabled`` does, and it says exactly one thing:
*selected here*.

:func:`panel_selection_overrides` is that projection, keyed by id.
:func:`apply_panel_selection` writes it into the rendered document after the
injectors have written their blocks. It indexes each block by its id rather
than through a dotted key, because an id may carry a dot and a dotted key is
split at every one.
:func:`panel_selection_errors` is the model-time refusal of an authored
``enabled`` that contradicts the selection — the same rule
:func:`osprey.cli.build_profile_reach.reach_override_errors` applies to a pinned
port: a spelling that agrees is the same fact, one that disagrees is refused.
The predicate every reader of a block then shares is
:func:`osprey.profiles.web_panels.panel_spec_enabled`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from osprey.profiles.web_panels import (
    BAR_ITEM_PANEL_GATES,
    UNIVERSAL_PANELS,
    panel_id_refusal,
)

from .build_profile_reach import spelled_values

__all__ = [
    "apply_panel_selection",
    "bar_items_selection_warnings",
    "panel_id_errors",
    "panel_selection_errors",
    "panel_selection_overrides",
]

#: The two bars a ``web.bar_items`` block arranges, as ``BAR_HOSTS`` names them.
_BAR_HOSTS: tuple[str, ...] = ("header", "status")


def panel_selection_overrides(
    selected_panels: Iterable[str], rendered_config: Mapping[str, Any]
) -> dict[str, bool]:
    """The ``enabled`` every block the render carries is told, keyed by its id.

    ``True`` for a selected (or universal) panel, ``False`` for any other block
    — whatever wrote it and whatever else it says. Only blocks that exist are
    annotated: a selected panel with no block is the template's or an
    injector's to write (and, for a Reach tab, ``selected_panel_errors``'s to
    refuse), not this function's. The id is the block's key as the render
    wrote it, dots included.

    Args:
        selected_panels: The profile's resolved ``web_panels`` selection.
        rendered_config: The render's ``config.yml``, after the injectors.

    Returns:
        Panel id to ``enabled``, in block order, for :func:`apply_panel_selection`.
    """
    web = rendered_config.get("web") if isinstance(rendered_config, Mapping) else None
    panels = web.get("panels") if isinstance(web, Mapping) else None
    if not isinstance(panels, Mapping):
        return {}
    shown = set(selected_panels) | UNIVERSAL_PANELS
    return {str(pid): str(pid) in shown for pid in panels}


def apply_panel_selection(config_path: Path, projection: Mapping[str, bool]) -> None:
    """Write each block's ``enabled`` into the rendered ``config.yml``.

    Each block is indexed by its id as one key, so ``beam.viewer`` is told
    under ``web.panels["beam.viewer"]`` and nowhere else. A block that is not a
    mapping, such as ``okf: true``, carries no other fact, so it becomes
    ``{enabled: <shown>}``. The write goes through
    :func:`osprey.utils.config_writer.anchored_put`, the public writer with the
    same trailing-section-comment handling a new dotted key gets.

    Args:
        config_path: The render's ``config.yml``.
        projection: Panel id to ``enabled``, from :func:`panel_selection_overrides`.
    """
    from osprey.utils.config_writer import (
        anchored_put,
        load_config_document,
        save_config_document,
    )

    if not projection:
        return
    document = load_config_document(config_path)
    web = document.get("web") if isinstance(document, Mapping) else None
    panels = web.get("panels") if isinstance(web, Mapping) else None
    if not isinstance(panels, dict):
        return
    for pid, shown in projection.items():
        if pid not in panels:
            continue
        block = panels[pid]
        if isinstance(block, dict):
            anchored_put(block, "enabled", shown)
        else:
            anchored_put(panels, pid, {"enabled": shown})
    save_config_document(config_path, document)


def panel_selection_errors(config: Any, selected_panels: Iterable[str]) -> list[str]:
    """Refuse an authored ``web.panels.<id>.enabled`` that contradicts the selection.

    ``web_panels`` is what shows a tab; an ``enabled`` under ``config:`` that
    says otherwise is two spellings of one fact disagreeing. Every spelling of
    the leaf is found (dotted, prefix-over-mapping, nested, mixed) so the
    refusal can name the line to remove — a spelling missed here would be a
    contradiction the projection silently overwrote.

    Args:
        config: The profile's ``config:`` block, whatever shape it parsed as.
        selected_panels: The profile's resolved ``web_panels`` selection.

    Returns:
        One error per contradicting spelling, naming the fix.
    """
    shown = set(selected_panels) | UNIVERSAL_PANELS
    errors: list[str] = []
    for pid in sorted(_spelled_panel_ids(config)):
        for spelling, value in spelled_values(config, f"web.panels.{pid}.enabled"):
            if value is True and pid not in shown:
                errors.append(
                    f"{spelling}: {value!r} but {pid!r} is not in web_panels — the selection "
                    f"is what shows a tab. Add {pid!r} to web_panels, or drop the line."
                )
            elif value is False and pid in shown:
                errors.append(
                    f"{spelling}: {value!r} but {pid!r} is in web_panels — a selected tab is on. "
                    f"Drop {pid!r} from web_panels, or keep it off-screen with "
                    f"web.panels.{pid}.hidden: true."
                )
    return errors


def panel_id_errors(config: Any) -> list[str]:
    """Refuse a ``web.panels.<id>`` the terminal could never serve.

    An id is a URL path segment and a header value on every proxied hop, and
    the terminal refuses to start on one outside that class — so a profile
    carrying such an id builds a deployment that cannot run. The refusal is
    worth having here as well as there, because a build is where the id is
    still a line in a file the operator has open, and the verdict is the same
    one: :func:`osprey.profiles.web_panels.panel_id_refusal`, not a second
    charset that could drift from the one the proxy actually spells.

    Every spelling of the mapping is read, so an id reaches this check however
    its block was authored.

    Args:
        config: The profile's ``config:`` block, whatever shape it parsed as.

    Returns:
        One refusal per unservable id, id order.
    """
    return [
        refusal
        for pid in sorted(_spelled_panel_ids(config))
        if (refusal := panel_id_refusal(pid)) is not None
    ]


def _spelled_panel_ids(config: Any) -> set[str]:
    """Every panel id a ``config:`` block names under ``web.panels``, as the render keys it.

    The render splits a top-level key at every dot, so ``web.panels.<id>.<leaf>``
    names its id in the third segment. A key inside a mapping is written as one
    key, dots included, so each key of a mapping under ``web.panels`` is an id
    taken whole. Only the mapping spellings can carry an id with a dot in it.
    """
    ids: set[str] = set()
    if not isinstance(config, Mapping):
        return ids
    for key, value in config.items():
        parts = str(key).split(".")
        if parts[:2] == ["web", "panels"]:
            if len(parts) > 2:
                ids.add(parts[2])
            elif isinstance(value, Mapping):
                ids.update(str(sub) for sub in value)
        elif parts == ["web"] and isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                sub_parts = str(sub_key).split(".")
                if sub_parts[0] != "panels":
                    continue
                if len(sub_parts) > 1:
                    ids.add(sub_parts[1])
                elif isinstance(sub_value, Mapping):
                    ids.update(str(leaf) for leaf in sub_value)
    return ids


def bar_items_selection_warnings(config: Any, selected_panels: Iterable[str]) -> list[str]:
    """Name each authored ``web.bar_items`` entry this deployment cannot show.

    The SYSTEM panel's status item renders only where that panel is selected
    and ``bluesky-queue`` only where the Bluesky panel is; an authored default that
    places either on a deployment without the panel is filtered by the server
    before it is served, so the item never appears and nothing on screen says
    why. A warning, not an error — the deployment still builds and renders,
    minus that one item — but the operator wrote the entry and will look for it,
    so the build says which one it was and what it needs.

    Every spelling of the list is found (dotted, prefix-over-mapping, nested),
    and each entry is reported by position, the way the server's own log line
    reports it.

    Args:
        config: The profile's ``config:`` block, whatever shape it parsed as.
        selected_panels: The profile's resolved ``web_panels`` selection.

    Returns:
        One warning per unrenderable entry, in bar order then list order.
    """
    shown = set(selected_panels) | UNIVERSAL_PANELS
    warnings: list[str] = []
    for host in _BAR_HOSTS:
        for spelling, entries in spelled_values(config, f"web.bar_items.{host}"):
            if not isinstance(entries, list):
                continue
            for index, entry in enumerate(entries):
                item_type = entry.get("type") if isinstance(entry, Mapping) else entry
                panel = BAR_ITEM_PANEL_GATES.get(str(item_type))
                if panel is None or panel in shown:
                    continue
                warnings.append(
                    f"{spelling}[{index}] places {item_type!r}, which needs the {panel!r} panel "
                    f"and {panel!r} is not in web_panels — the item leaves the default this "
                    f"deployment serves. Add {panel!r} to web_panels, or drop the entry."
                )
    return warnings
