"""Fold the duplicate probes of one shared resource into one row.

Several categories answer the same question about one resource from their own
angle: the graph store's bolt connection is read by the ``graphdb`` category,
restated by ``channel_finder`` in its graph paradigm, and knocked on by
``reach`` for the graph MCP server. Reported side by side, one fault shows up
three times and one fact is counted three times. :func:`fold_shared` keeps one
row per resource and records the consumers whose probes it absorbed in
:attr:`~osprey.health.models.CheckResult.used_by`.

The equivalences are :data:`SHARED_PROBES`: groups of row names that read the
same fact. The first name present in a group is the row kept; when the probes
disagree the worst one is kept instead, since a consumer that cannot reach the
resource from where it runs is the fact that matters. A group with a single
present row is left untouched, so a ``--category`` run of one category reports
exactly what it did before.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from osprey.health.models import CheckResult, Status

#: Row names that read the same fact about one resource, canonical row first.
SHARED_PROBES: tuple[tuple[str, ...], ...] = (
    ("graphdb_connection", "channel_finder_store", "reach.graphdb"),
    ("graphdb_resources", "channel_finder_resources"),
    ("graphdb_seed", "channel_finder_seed"),
    ("qmd_okf", "reach.qmd-okf"),
    ("qmd_ariel", "reach.qmd-ariel"),
    ("web_panels.bluesky", "reach.bluesky_web"),
)

#: How a category names itself as a consumer when the row's message does not.
_CONSUMER_LABELS = {
    "graphdb": "graph store service",
    "channel_finder": "channel finder",
    "qmd": "qmd index",
    "web_panels": "panel",
}

_RANK = {Status.SKIP: 0, Status.OK: 1, Status.WARNING: 2, Status.ERROR: 3}


def _consumer(row: CheckResult) -> str:
    """The consumer a row speaks for: the ``reach`` prefix, else its category."""
    if row.category == "reach" and ":" in row.message:
        return row.message.split(":", 1)[0].strip()
    return _CONSUMER_LABELS.get(row.category, row.category.replace("_", " "))


def fold_shared(results: Sequence[CheckResult]) -> list[CheckResult]:
    """Return *results* with each :data:`SHARED_PROBES` group folded into one row.

    The kept row takes the position of the group's first row in *results*; the
    others are dropped. Order is otherwise preserved.
    """
    by_name = {r.name: r for r in results}
    keep: dict[str, CheckResult] = {}
    drop: set[str] = set()
    for group in SHARED_PROBES:
        present = [by_name[name] for name in group if name in by_name]
        if len(present) < 2:
            continue
        kept = present[0]
        for row in present[1:]:
            if _RANK[row.status] > _RANK[kept.status]:
                kept = row
        consumers = tuple(dict.fromkeys(_consumer(r) for r in present))
        first = present[0].name
        keep[first] = replace(kept, used_by=consumers)
        drop.update(r.name for r in present if r.name != first)
    if not keep:
        return list(results)
    return [keep.get(r.name, r) for r in results if r.name not in drop]
