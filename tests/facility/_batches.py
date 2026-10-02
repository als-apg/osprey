"""The ordered delivery stages the facility guards key their entries on.

``BATCHES`` is the one source of the order; ``CURRENT_BATCH`` indexes the last
closed stage. An entry tagged with a stage at or before ``CURRENT_BATCH`` must
already hold, so each stage's close moves ``CURRENT_BATCH`` forward by one.
"""

from __future__ import annotations

BATCHES: tuple[str, ...] = (
    "1a",
    "1b",
    "1c",
    "6",
    "2",
    "3a",
    "3b",
    "4a",
    "4b",
    "5",
    "7a0",
    "7a",
    "7b",
    "7c",
    "7d",
    "7d2",
    "7e",
    "8",
    "9",
    "10",
    "11a",
    "11b",
    "11c",
    "12",
)

CURRENT_BATCH: int = BATCHES.index("3b")
