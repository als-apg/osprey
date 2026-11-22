"""One order for every list of the panels that is ordered by name.

A name that holds numbers is ordered by their values, not as text: ``BO 2``
before ``BO 10``, never ``1, 10, 11, 2``. It is an order only; it groups nothing.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"(\d+)")


def name_order(name: str) -> tuple[tuple[int, int, str], ...]:
    """The sort key of *name*: its runs of digits by value, the text between them
    as text. Keys of any two names compare, whatever their shape."""
    return tuple(
        (0, int(part), "") if part.isdigit() else (1, 0, part) for part in _NUMBER.split(name)
    )
