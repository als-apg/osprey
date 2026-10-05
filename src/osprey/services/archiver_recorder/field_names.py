"""Which channel addresses the archive can hold as a field name.

A tick is stored as one flat document keyed by channel address, and the
archiver connector projects the same names back out. MongoDB reads ``.`` in a
field name as a path separator and a leading ``$`` as an operator, and holds no
NUL byte at all, so an address carrying one of those is not a field this store
can round-trip.

The module imports nothing beyond the standard library, so the build's
simulator view can apply the same check without loading the recorder's store or
its config readers.
"""

from __future__ import annotations

__all__ = ["_unstorable_field_name"]


def _unstorable_field_name(address: str) -> str | None:
    """Why ``address`` cannot be an archive field name, or ``None`` if it can.

    Args:
        address: A channel address.

    Returns:
        A clause naming the offending character and how the archive reads it,
        phrased to follow ``it``; ``None`` for an address the archive stores.
    """
    if "." in address:
        return "contains '.', which the archive reads as a document path separator"
    if address.startswith("$"):
        return "starts with '$', which the archive reads as an operator"
    if "\x00" in address:
        return "contains a NUL byte, which a field name cannot hold"
    return None
