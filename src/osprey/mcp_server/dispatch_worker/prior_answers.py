"""Which earlier runs a dispatched agent may read back.

The one definition shared by the worker that receives the list from a dispatch
and the workspace tool ``prior_answer_read`` that enforces it. A chat bridge
that replays a long earlier answer shortened names that answer's run; the
worker stamps those run ids into the agent's environment at spawn, and the tool
reads only records whose id is in that set.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

PRIOR_ANSWER_RUNS_ENV = "OSPREY_DISPATCH_PRIOR_ANSWER_RUNS"
"""The variable ``run_dispatch`` stamps and ``prior_answer_read`` reads."""

PRIOR_ANSWERS_CAPABILITY = "prior_answers"
"""The ``/health`` capability the dispatcher and the worker both advertise."""

MAX_PRIOR_ANSWER_RUNS = 100
"""Equal to the bridge history's turn cap, so a bridge never sends more."""


def is_run_id(value: object) -> bool:
    """Return ``True`` only for a ``str`` that is a canonical UUID.

    Worker run ids are ``str(uuid.uuid4())``, and the tool joins an accepted id
    into a path, so nothing else may pass.
    """
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def keep_run_ids(values: Iterable[object]) -> list[str]:
    """Keep the valid run ids, first occurrence of each, newest ``MAX_PRIOR_ANSWER_RUNS``.

    History order is oldest first, so keeping the last entries keeps the newest.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for value in values:
        if isinstance(value, str) and is_run_id(value) and value not in seen:
            seen.add(value)
            kept.append(value)
    return kept[-MAX_PRIOR_ANSWER_RUNS:]


def format_run_ids(ids: Iterable[str]) -> str:
    """Join run ids into the environment-variable form."""
    return ",".join(ids)


def parse_run_ids(raw: str | None) -> frozenset[str]:
    """Parse the environment-variable form; blank or ``None`` gives the empty set."""
    if not raw:
        return frozenset()
    return frozenset(part.strip() for part in raw.split(",") if is_run_id(part.strip()))
