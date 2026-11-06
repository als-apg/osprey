"""The ``agentic`` marker: which tier an agentic test is in, and what it asks.

An agentic test is one where the model under test drives the agent and can fail
the task. Each carries one marker, on the test itself::

    @pytest.mark.agentic(
        tier=1,
        title="Read one channel",
        asks="What is the beam current?",
        passes_if="answers with the live value; no writes",
    )

The marker is the only place a test's tier is written down. The benchmark
board runs the tests that carry it, records the tier with each result, and
shows the title, the ask and the pass condition in place of a file name. The
``tier2/`` directory is a subset of tier 2, checked at collection, not a second
source.

* Tier 1, basic: the stack works, the model calls the right tools, a simple
  workflow goes through. Expected to pass; the board compares time and tokens.
* Tier 2, benchmark: harder tasks a capable model may still miss. The board
  shows each test's first-attempt result per seed.

The CI job names that say "Tier 0" to "Tier 3" number something else (the cost
of a proof about a preset) and are not read from this marker.
"""

from collections.abc import Iterable
from pathlib import Path

import pytest

#: The marker's name.
AGENTIC_MARKER = "agentic"

#: The tiers a test can carry, and the word each goes by.
TIERS = {1: "basic", 2: "benchmark"}

#: The marker's keyword arguments, all required.
FIELDS = ("tier", "title", "asks", "passes_if")

#: The lane marker for tests that run an agent but grade OSPREY, not the model.
HARNESS_MARKER = "harness_benchmark"

#: The directory whose tests must all be tier 2.
TIER2_DIR = "tier2"

#: How a skip reason says the test does not apply to the model under test. The
#: board shows ``n/a: <label>`` in the model's cell and counts the result
#: against neither its passes nor its all-green check.
NOT_APPLICABLE = "n/a: "


def not_applicable(label: str, detail: str) -> str:
    """The skip reason for a test that does not apply to this model, such as a
    vision task on a model without vision."""
    return f"{NOT_APPLICABLE}{label} ({detail})"


def marker_problem(marker: pytest.Mark) -> str | None:
    """What is wrong with one ``agentic`` marker, or None when it is complete."""
    if marker.args:
        return "takes keyword arguments only"
    missing = [f for f in FIELDS if f not in marker.kwargs]
    if missing:
        return "missing " + ", ".join(missing)
    unknown = sorted(set(marker.kwargs) - set(FIELDS))
    if unknown:
        return "unknown argument " + ", ".join(unknown)
    tier = marker.kwargs["tier"]
    if isinstance(tier, bool) or not isinstance(tier, int) or tier not in TIERS:
        return f"tier={tier!r} is not one of {sorted(TIERS)}"
    blank = [
        f
        for f in FIELDS[1:]
        if not isinstance(marker.kwargs[f], str) or not marker.kwargs[f].strip()
    ]
    if blank:
        return "empty " + ", ".join(blank)
    return None


def item_problem(item: pytest.Item, e2e_root: Path) -> str | None:
    """What is wrong with one collected test's tier, or None."""
    marker = item.get_closest_marker(AGENTIC_MARKER)
    path = Path(str(getattr(item, "path", "") or item.fspath)).resolve()
    in_tier2 = (e2e_root / TIER2_DIR) in path.parents
    if marker is None:
        return f"sits in {TIER2_DIR}/ without an agentic(tier=2) marker" if in_tier2 else None
    problem = marker_problem(marker)
    if problem:
        return f"agentic marker {problem}"
    if item.get_closest_marker(HARNESS_MARKER) is not None:
        return f"carries both agentic and {HARNESS_MARKER}"
    if in_tier2 and marker.kwargs["tier"] != 2:
        return f"sits in {TIER2_DIR}/ but carries tier={marker.kwargs['tier']}"
    return None


def collection_refusal(problems: Iterable[tuple[str, str]]) -> str | None:
    """The message that stops a run whose agentic markers are incomplete."""
    problems = sorted(problems)
    if not problems:
        return None
    lines = [f"  {nodeid}: {problem}" for nodeid, problem in problems]
    return (
        f"{len(problems)} agentic test(s) carry an incomplete tier marker. Each needs "
        f"@pytest.mark.agentic(tier=1|2, title=..., asks=..., passes_if=...) "
        f"(see tests/e2e/agentic_tiers.py):\n" + "\n".join(lines)
    )


def manifest_entry(item: pytest.Item) -> dict:
    """The board's record of one collected test: its lane, and for an agentic
    test its tier, title, ask and pass condition."""
    marker = item.get_closest_marker(AGENTIC_MARKER)
    harness = item.get_closest_marker(HARNESS_MARKER) is not None
    if marker is not None and harness:
        return {"lane": "both"}  # check_e2e_coverage --check-lanes rejects it
    if marker is not None:
        return {"lane": "agentic", **{f: marker.kwargs.get(f) for f in FIELDS}}
    return {"lane": "harness" if harness else "unmarked"}
