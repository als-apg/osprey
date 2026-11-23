"""The flat-name rule for place ids.

A place id is a ``/`` path. A name naming a place is either such a path, which
must equal an id, or a flat name, which matches every id equal to it or ending
in ``/<name>``: whole segments only, so ``S01`` matches ``SR/S01`` and never
``SR/XS01``.

The rule is pure and imports nothing from the build or the importers, so both
the relations importer and the read model resolve names the same way.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["resolve_place_name"]


def resolve_place_name(name: str, ids: Iterable[str]) -> str | list[str] | None:
    """Resolve a place name against place ids.

    Args:
        name: A path (holding ``/``) or a flat name.
        ids: The place ids to match.

    Returns:
        The id for one match, the sorted candidates for several, ``None``
        for none.
    """
    if "/" in name:
        return name if name in set(ids) else None
    suffix = "/" + name
    matches = sorted({i for i in ids if i == name or i.endswith(suffix)})
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return matches
