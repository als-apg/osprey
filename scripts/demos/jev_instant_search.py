#!/usr/bin/env python3
"""Show what the ARIEL ``jev`` search mode changes, without a database.

The mode is two stages: a lexical retrieval that decides *membership*, and a
Jev call that decides *order*. This script keeps the second stage exactly as it
ships — the real ``jev_search``, the real state and questions, the real
blending — and replaces only the first with an in-memory ranker over a dozen
hand-written logbook entries, so the whole path runs with no Postgres, no
sidecar and no deployment.

It prints the two orderings side by side: what the lexical stage alone would
have put in front of the operator, and what came back after the model reviewed
the pool. The queries are chosen so that the difference is the point --- each
one is worded the way an operator would word it rather than the way the
logbook does.

Run it::

    export TYPESAFE_API_KEY=...            # or it falls back to a recorded reply
    uv run python scripts/demos/jev_instant_search.py
    uv run python scripts/demos/jev_instant_search.py "your own query"

With no key the script still runs, against a recorded reply for its first
query. That is enough to show the mechanism and not enough to show the model:
treat a keyless run as a smoke test, not as evidence about ranking quality.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

# Run from a checkout without installing: src/ is where the package lives.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from osprey.services.ariel_search.config import SearchModuleConfig  # noqa: E402
from osprey.services.ariel_search.search import jev as jev_module  # noqa: E402
from osprey.services.ariel_search.search.base import ModuleOutput  # noqa: E402
from osprey.services.jev import DEFAULT_API_KEY_ENV, JevAnswer, JevResult  # noqa: E402


@dataclass(frozen=True)
class Entry:
    """One logbook entry in the demo corpus."""

    entry_id: str
    day: int
    author: str
    text: str


#: Day 0 of the demo corpus; each entry's ``day`` is an offset from it.
_DEMO_EPOCH = datetime(2026, 4, 1, 9, 0, tzinfo=UTC)

#: A small corpus with the property that makes the mode worth having: several
#: entries share the operator's vocabulary without being about what was asked,
#: and the one that answers the question uses the logbook's words instead.
CORPUS: tuple[Entry, ...] = (
    Entry(
        "2026-0412",
        12,
        "rhodes",
        "SR02C BPM electronics swapped; beam position readback restored after the new module was calibrated.",
    ),
    Entry(
        "2026-0418",
        18,
        "okafor",
        "Beam dumped at 14:02. Interlock chain traced to the RF cavity 3 arc detector; cavity conditioning restarted.",
    ),
    Entry(
        "2026-0421",
        21,
        "rhodes",
        "Routine vacuum survey of the storage ring straight sections. All gauges nominal, no action.",
    ),
    Entry(
        "2026-0423",
        23,
        "lindqvist",
        "Injection efficiency down to 62%. Septum timing adjusted, recovered to 88% by end of shift.",
    ),
    Entry(
        "2026-0425",
        25,
        "okafor",
        "Lost stored beam during user run. Post-mortem shows a fast orbit excursion in sector 7 preceding the loss by 3 ms.",
    ),
    Entry(
        "2026-0427",
        27,
        "chen",
        "Quadrupole QF14 power supply tripped on overcurrent. Supply reset, no further trips this shift.",
    ),
    Entry(
        "2026-0429",
        29,
        "chen",
        "Discussion note: proposal to add an interlock on the sector 7 corrector magnets after last week's excursion.",
    ),
    Entry(
        "2026-0501",
        31,
        "lindqvist",
        "Beam current dropped from 500 mA to 410 mA over two hours. Lifetime degradation traced to a vacuum leak near the ID straight.",
    ),
    Entry(
        "2026-0503",
        33,
        "rhodes",
        "RF cavity 3 returned to service at full gradient. No arcing observed over an 8-hour soak.",
    ),
    Entry(
        "2026-0505",
        35,
        "okafor",
        "Scheduled machine physics shift: chromaticity scan, orbit response matrix measurement. No faults.",
    ),
    Entry(
        "2026-0507",
        37,
        "chen",
        "Sudden beam loss at 03:41. Orbit interlock fired in sector 7 again; the corrector interlock proposal is now urgent.",
    ),
    Entry(
        "2026-0509",
        39,
        "lindqvist",
        "Front-end shutter stuck closed on beamline 8. Mechanical fault, beamline down for the shift.",
    ),
)

#: The queries the demo runs by default.
DEMO_QUERIES: tuple[str, ...] = (
    "why did we lose the beam",
    "rf cavity trouble",
    "the most recent vacuum problem",
)

#: A reply recorded from one live run of the first demo query, so a machine with
#: no API key still exercises the whole path. It is a fixture, not a result:
#: only the numbers for the first query are real, and none of them are claims
#: about how the model scores anything else.
RECORDED_REPLY: dict[str, tuple[int, float]] = {
    "cand_0": (2, 0.81),
    "cand_1": (4, 0.77),
    "cand_2": (1, 0.93),
    "cand_3": (3, 0.68),
    "cand_4": (5, 0.88),
    "cand_5": (2, 0.74),
    "cand_6": (3, 0.62),
    "cand_7": (4, 0.71),
    "cand_8": (1, 0.90),
    "cand_9": (1, 0.85),
    "cand_10": (5, 0.86),
    "cand_11": (1, 0.92),
}


class RecordedClient:
    """A JevClient stand-in replaying :data:`RECORDED_REPLY`."""

    def is_available(self) -> bool:
        """Always ready: the reply is in this file."""
        return True

    async def ask(self, state: Any, questions: Any) -> JevResult:
        """Answer every candidate question the caller asked.

        Args:
            state: Ignored — the recording is fixed.
            questions: Used only for which candidate ids to answer.

        Returns:
            A reply shaped exactly like a live one.
        """
        answers: dict[str, JevAnswer] = {}
        legend = {str(index + 1): "" for index in range(len(jev_module.RELEVANCE_LEVELS))}
        for key in questions:
            if not key.startswith("cand_"):
                continue
            level, confidence = RECORDED_REPLY.get(key, (1, 0.5))
            answers[key] = JevAnswer(
                type="score", value=level, confidence=confidence, legend=legend
            )
        return JevResult(
            answers=answers,
            model="jev-latest (recorded)",
            input_tokens=7800,
            output_tokens=96,
            latency_ms=0,
        )


def _entry_row(entry: Entry) -> dict[str, Any]:
    """Shape one corpus entry the way the repository would."""
    return {
        "entry_id": entry.entry_id,
        "source_system": "demo",
        "author": entry.author,
        "timestamp": _DEMO_EPOCH + timedelta(days=entry.day),
        "raw_text": entry.text,
        "attachments": [],
        "metadata": {},
        "summary": "",
        "keywords": [],
    }


def _lexical_rank(query: str) -> list[tuple[dict[str, Any], float, list[str]]]:
    """Rank the corpus by term overlap, standing in for PostgreSQL full-text.

    Crude on purpose. The point of the demo is what the second stage does with
    a merely adequate first stage, and a first stage good enough to need no
    help would make the comparison meaningless.

    Args:
        query: The operator's query.

    Returns:
        ``(entry, score, highlights)`` tuples, best first, matches only.
    """
    terms = {token.strip(".,;:'\"").lower() for token in query.split() if len(token) > 2}
    scored: list[tuple[dict[str, Any], float, list[str]]] = []
    for entry in CORPUS:
        words = {token.strip(".,;:'\"").lower() for token in entry.text.split()}
        overlap = len(terms & words)
        if overlap:
            scored.append((_entry_row(entry), overlap / max(1, len(terms)), []))
    scored.sort(key=lambda row: row[1], reverse=True)
    return scored


def _config() -> Any:
    """A config carrying only what the module reads."""
    return SimpleNamespace(
        search_modules={
            "jev": SearchModuleConfig(
                enabled=True, settings={"candidate_limit": len(CORPUS), "min_relevance": 0.0}
            )
        }
    )


def _install_lexical_stage() -> None:
    """Point the module's lexical stage at the in-memory corpus."""

    async def _keyword_search(query, repository, config, **kwargs):
        rows = _lexical_rank(query)[: kwargs.get("max_results", 10)]
        return ModuleOutput(entries=rows)

    jev_module.keyword_search = _keyword_search  # type: ignore[assignment]


def _label(row: Any) -> str:
    """One line describing a result row."""
    entry = row[0]
    return f"{entry['entry_id']}  {entry['raw_text'][:68]}"


async def _run_query(query: str, client: Any) -> None:
    """Print the two orderings for one query.

    Args:
        query: The operator's query.
        client: The Jev client (live or recorded).
    """
    config = _config()
    before = await jev_module.jev_search(
        query, None, config, max_results=5, rerank=False, client=client
    )
    after = await jev_module.jev_search(
        query, None, config, max_results=5, rerank=True, client=client
    )

    print(f"\n\033[1m{query}\033[0m")
    print("  keyword only")
    for position, row in enumerate(before.entries, start=1):
        print(f"    {position}. {_label(row)}")
    print("  reranked by Jev")
    for position, row in enumerate(after.entries, start=1):
        print(f"    {position}. {_label(row)}")
    for diagnostic in after.diagnostics:
        print(f"    [{diagnostic.level.value}] {diagnostic.message}")


async def _main(queries: list[str]) -> int:
    """Run every query and report which client answered.

    Args:
        queries: Queries to run.

    Returns:
        Process exit code.
    """
    _install_lexical_stage()

    if os.environ.get(DEFAULT_API_KEY_ENV):
        client = None  # jev_search builds a live one from the settings block
        print(f"Asking the live endpoint ({DEFAULT_API_KEY_ENV} is set).")
    else:
        client = RecordedClient()
        print(
            textwrap.dedent(
                f"""\
                {DEFAULT_API_KEY_ENV} is unset, so this run replays a recorded reply.
                The path is real; the scores are a fixture recorded against one query,
                so only that query is run. Set the key to see the model rank the rest."""
            )
        )
        if queries == list(DEMO_QUERIES):
            queries = [DEMO_QUERIES[0]]

    for query in queries:
        await _run_query(query, client)
    print()
    return 0


def main() -> int:
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", nargs="*", help="Queries to run (default: three built-in ones)")
    args = parser.parse_args()
    return asyncio.run(_main(list(args.query) or list(DEMO_QUERIES)))


if __name__ == "__main__":
    raise SystemExit(main())
