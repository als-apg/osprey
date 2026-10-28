"""Resolve a write-surface row the way the emitted guards do."""

from __future__ import annotations

import importlib


def resolve_target(dotted):
    """Resolve a write-surface row to the object that carries its attributes.

    Deliberately a copy of the resolver the guards emit rather than an import of
    it: an emitted guard has to be self-contained (it runs before the user code
    in a subprocess that may not be able to import OSPREY at all), so there is
    no function to share.
    """
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            try:
                obj = getattr(obj, attr)
            except AttributeError:
                return None
        return obj
    return None
