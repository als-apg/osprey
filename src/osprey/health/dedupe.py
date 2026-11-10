"""Fold the duplicate probes of one shared resource into one row.

Several categories answer the same question about one resource from their own
angle: the graph store's bolt connection is read by the ``graphdb`` category,
restated by ``channel_finder`` in its graph paradigm, and knocked on by
``reach`` for the graph MCP server. Reported side by side, one fault shows up
three times and one fact is counted three times. :func:`fold_shared` keeps one
row per resource and records the consumers whose probes it absorbed in
:attr:`~osprey.health.models.CheckResult.used_by`.

The equivalences are :data:`SHARED_PROBES`: groups of row names that read the
same fact. The first name present in a group is the row kept, under its own
name and category; when the probes disagree it carries the worst one's status,
message and value, since a consumer that cannot reach the resource from where
it runs is the fact that matters. Folding a folded row again keeps every
consumer it already named. A group with a single
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
    ("web_panels.optimizer", "reach.optimizer_web"),
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
    others are dropped. Order is otherwise preserved. Idempotent.
    """
    by_name = {r.name: r for r in results}
    keep: dict[str, CheckResult] = {}
    first_of: dict[str, str] = {}
    for group in SHARED_PROBES:
        present = [by_name[name] for name in group if name in by_name]
        if len(present) < 2:
            continue
        canonical = present[0]
        worst = canonical
        for row in present[1:]:
            if _RANK[row.status] > _RANK[worst.status]:
                worst = row
        # A row folded earlier already names its consumers; keep them all.
        consumers = tuple(dict.fromkeys(c for r in present for c in (r.used_by or (_consumer(r),))))
        keep[canonical.name] = replace(
            canonical,
            status=worst.status,
            message=worst.message,
            value=worst.value,
            used_by=consumers,
        )
        first_of.update(dict.fromkeys((r.name for r in present), canonical.name))
    out: list[CheckResult] = []
    placed: set[str] = set()
    for r in results:
        canonical_name = first_of.get(r.name)
        if canonical_name is None:
            out.append(r)
        elif canonical_name not in placed:
            # The folded row takes the place of the group's first row in *results*.
            out.append(keep[canonical_name])
            placed.add(canonical_name)
    return out
