"""Inventory helpers: the rows, channels and form totals of a set of artifacts.

Shared by every "this session" inventory (the agent's listing, the web
terminal's session page, the feedback bundle) so each lists a set of entries
the way the gallery sidebar does: one row per run, a run's preview and
notebook folded under its head.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from osprey.stores.artifact_forms import form_label
from osprey.stores.artifact_store import ArtifactEntry, listed_form, registry_or_default

__all__ = ["entry_channels", "form_totals", "listed_rows"]

#: Keys of ``summary`` / ``access_details`` that carry channel names.
_CHANNEL_KEYS = ("channels", "columns", "channel_names", "pvs")


def listed_rows(
    store: Any, entries: Iterable[ArtifactEntry], registry: Any = None
) -> list[tuple[ArtifactEntry, str, str]]:
    """``(entry, form key, label)`` for each of *entries* that is listed.

    Runs are classified over the whole index (``store.run_groups()``), as the
    sidebar does; *entries* only selects the rows kept. An entry whose
    :func:`listed_form` is ``None`` (a run's preview, notebook or second result)
    is dropped; a run's head and its outputs stay.
    """
    registry = registry_or_default(registry)
    groups = store.run_groups()
    rows: list[tuple[ArtifactEntry, str, str]] = []
    for entry in entries:
        key = listed_form(entry, groups.get(entry.exec_id), registry)
        if key is not None:
            rows.append((entry, key, form_label(key, registry)))
    return rows


def entry_channels(entry: ArtifactEntry) -> list[str]:
    """Channel names an entry's ``summary`` or ``access_details`` names, de-duplicated in order."""
    channels: list[str] = []
    for source in (entry.summary or {}, entry.access_details or {}):
        for key in _CHANNEL_KEYS:
            val = source.get(key)
            if isinstance(val, list):
                channels.extend(str(c) for c in val)
    return list(dict.fromkeys(channels))


def form_totals(pairs: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    """``[{key, label, count}]`` for the ``(form, label)`` *pairs*, in first-seen order."""
    counts = Counter(pairs)
    return [{"key": key, "label": label, "count": n} for (key, label), n in counts.items()]
