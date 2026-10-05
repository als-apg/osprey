#!/usr/bin/env python3
"""Calibrate the picture lane's fusion margin and floor on a labelled store export.

Picture search admits an image-only hit when its cosine similarity reaches
``max(best - relative_margin, min_similarity)``
(:func:`osprey.services.ariel_search.search.fusion.fuse_lanes`). This script
measures how often such admitted hits are relevant, on a store export whose
queries carry labelled relevant entries:

1. The set is checked first: at least 100 embedded pictures in the image table
   and at least 30 queries with labelled relevant entries. A smaller set is
   reported and nothing is measured.
2. Each query is embedded on llama-server through the adapter OSPREY itself
   calls, and each entry's nearest picture is read with the picture lane's own
   SQL.
3. For every ``(margin, floor)`` point of the grid, the image-only hits the
   shipped rule admits are counted against the labels. Precision is relevant
   admitted hits over admitted hits, summed over all queries.
4. Among the points whose precision reaches ``--threshold`` (stated before the
   run), the chosen point admits the most relevant hits, then has the highest
   precision, then is the shipped default. When no point passes, the shipped
   defaults stand.

The set file is JSON::

    {"export": "<name of the store export>",
     "queries": [{"query": "...", "relevant": ["<entry id>", ...],
                  "text_hits": ["<entry id>", ...]}]}

``text_hits`` (optional) are the entries the text lane returns for the query;
they are never image-only.

Usage::

    python scripts/benchmark/fusion_calibration.py --set set.json \\
        --dsn postgresql://ariel@localhost/ariel --threshold 0.8 --json out.json

Exit status: 0 when the run completed (pass or fail); 1 when the set is too
small; 2 when the server or the database could not be measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey.services.ariel_search.search import fusion

MIN_PICTURES = 100
MIN_LABELLED_QUERIES = 30
DEFAULT_CAP = 4


@dataclass(frozen=True)
class LabelledQuery:
    """One query of the set, with its measured nearest pictures."""

    query: str
    relevant: frozenset[str]
    text_hits: tuple[str, ...] = ()
    image_hits: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelledSet:
    """A named store export's labelled queries."""

    export: str
    queries: list[LabelledQuery]


def load_set(path: Path) -> LabelledSet:
    """Read the set file; every query must name at least one relevant entry."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = []
    for index, item in enumerate(data.get("queries", [])):
        relevant = frozenset(str(entry) for entry in item.get("relevant") or ())
        if not relevant:
            raise ValueError(f"query {index} ({item.get('query')!r}) has no relevant entries")
        queries.append(
            LabelledQuery(
                query=str(item["query"]),
                relevant=relevant,
                text_hits=tuple(str(entry) for entry in item.get("text_hits") or ()),
            )
        )
    return LabelledSet(export=str(data.get("export", "")), queries=queries)


def set_problems(*, pictures: int, labelled_queries: int) -> list[str]:
    """Why the set is too small to calibrate on; empty when it is large enough."""
    problems = []
    if pictures < MIN_PICTURES:
        problems.append(f"{pictures} embedded pictures, at least {MIN_PICTURES} needed")
    if labelled_queries < MIN_LABELLED_QUERIES:
        problems.append(
            f"{labelled_queries} labelled queries, at least {MIN_LABELLED_QUERIES} needed"
        )
    return problems


def parse_grid(spec: str) -> list[float]:
    """``"start:stop:step"`` (inclusive) or a single value, as a list of floats."""
    parts = spec.split(":")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"bad grid {spec!r}") from exc
    if len(values) == 1:
        return values
    if len(values) != 3:
        raise ValueError(f"bad grid {spec!r}: want start:stop:step")
    start, stop, step = values
    if step <= 0 or stop < start:
        raise ValueError(f"bad grid {spec!r}: need step > 0 and stop >= start")
    count = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 6) for i in range(count)]


def admitted_image_only(
    text_hits: Iterable[str],
    image_hits: Mapping[str, float],
    *,
    margin: float,
    floor: float,
    cap: int,
) -> list[str]:
    """The image-only entries :func:`fusion.fuse_lanes` admits, closest first."""
    hits = {entry: fusion.ImageHit(entry, similarity) for entry, similarity in image_hits.items()}
    fused = fusion.fuse_lanes(
        [(entry, 1.0) for entry in dict.fromkeys(text_hits)],
        hits,
        relative_margin=margin,
        min_similarity=floor,
        cap=cap,
    )
    admitted = [hit.entry_id for hit in fused if hit.matched_via == ["image"]]
    return sorted(admitted, key=lambda entry: (-image_hits[entry], entry))


def evaluate(
    queries: Sequence[LabelledQuery], *, margin: float, floor: float, cap: int
) -> dict[str, Any]:
    """Precision of admitted image-only hits at one ``(margin, floor)`` point."""
    admitted = relevant = 0
    for query in queries:
        entries = admitted_image_only(
            query.text_hits, query.image_hits, margin=margin, floor=floor, cap=cap
        )
        admitted += len(entries)
        relevant += sum(1 for entry in entries if entry in query.relevant)
    return {
        "relative_margin": margin,
        "min_similarity": floor,
        "admitted": admitted,
        "relevant_admitted": relevant,
        "precision": relevant / admitted if admitted else None,
    }


def _is_default(row: Mapping[str, Any]) -> bool:
    return (
        abs(row["relative_margin"] - fusion.RELATIVE_MARGIN) < 1e-9
        and abs(row["min_similarity"] - fusion.MIN_SIMILARITY) < 1e-9
    )


def sweep(
    queries: Sequence[LabelledQuery],
    *,
    margins: Sequence[float],
    floors: Sequence[float],
    cap: int,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate the grid and choose a point against the pre-stated threshold."""
    grid = [
        evaluate(queries, margin=margin, floor=floor, cap=cap)
        for margin in margins
        for floor in floors
    ]
    for row in grid:
        row["passes"] = row["precision"] is not None and row["precision"] >= threshold
    passing = [row for row in grid if row["passes"]]
    if passing:
        chosen = max(
            passing,
            key=lambda row: (row["relevant_admitted"], row["precision"], _is_default(row)),
        )
    else:
        chosen = next((row for row in grid if _is_default(row)), None) or {
            **evaluate(
                queries,
                margin=fusion.RELATIVE_MARGIN,
                floor=fusion.MIN_SIMILARITY,
                cap=cap,
            ),
            "passes": False,
        }
    return {"threshold": threshold, "grid": grid, "chosen": dict(chosen)}


def count_pictures(dsn: str, table: str) -> int:
    """Embedded pictures in the image table."""
    import psycopg

    with psycopg.connect(dsn) as conn:
        row = conn.execute(f"SELECT count(*) FROM {table} WHERE embedding IS NOT NULL").fetchone()
    return int(row[0]) if row else 0


def measure(
    labelled: LabelledSet,
    *,
    dsn: str,
    table: str,
    base_url: str,
    model: str,
    dimensions: int,
    pictures: int,
    timeout: float,
) -> list[LabelledQuery]:
    """Each query's nearest picture per entry, as the picture lane reads it."""
    import psycopg

    from osprey.models.providers.base import TextInput
    from osprey.models.providers.llama_cpp import LlamaCppProviderAdapter
    from osprey.services.ariel_search.database.vector_literal import vector_literal
    from osprey.services.ariel_search.search.image_lane import _NEAREST_SQL, _ef_search

    adapter = LlamaCppProviderAdapter()
    measured = []
    with psycopg.connect(dsn) as conn:
        for query in labelled.queries:
            (vector,) = adapter.execute_image_embedding(
                [TextInput(query.query)],
                model,
                base_url=base_url,
                dimensions=dimensions,
                timeout=timeout,
            )
            with conn.transaction():
                conn.execute(
                    "SELECT set_config('hnsw.ef_search', %(ef)s, true)",
                    {"ef": _ef_search(pictures)},
                )
                rows = conn.execute(
                    _NEAREST_SQL.format(table=table),
                    {"q": vector_literal(vector), "k": pictures},
                ).fetchall()
            hits: dict[str, float] = {}
            for entry_id, _attachment_id, similarity in rows:
                hits.setdefault(str(entry_id), float(similarity))
            measured.append(
                LabelledQuery(query.query, query.relevant, query.text_hits, image_hits=hits)
            )
    return measured


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--set", dest="set_path", required=True, help="the labelled set file")
    parser.add_argument("--dsn", required=True, help="the store's PostgreSQL connection string")
    parser.add_argument(
        "--threshold", type=float, required=True, help="precision a point must reach"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama-server root")
    parser.add_argument("--model", default=None, help="image embedding model (server alias)")
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--margins", default="0.02:0.16:0.02", help="start:stop:step or value")
    parser.add_argument("--floors", default="0.35:0.60:0.01", help="start:stop:step or value")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="image-only admissions")
    parser.add_argument("--pictures", type=int, default=150, help="pictures fetched per query")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per embedding")
    parser.add_argument("--json", dest="json_path", help="also write the result here")
    args = parser.parse_args(argv)

    from osprey.models.providers.llama_cpp import LLAMA_CPP_DEFAULT_MODEL
    from osprey.services.ariel_search.database.migrations import image_embedding_target

    model = args.model or LLAMA_CPP_DEFAULT_MODEL
    table = image_embedding_target({"model": model, "dimensions": args.dimensions}).table
    labelled = load_set(Path(args.set_path))
    result: dict[str, Any] = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "export": labelled.export,
        "model": model,
        "dimensions": args.dimensions,
        "table": table,
        "labelled_queries": len(labelled.queries),
        "threshold": args.threshold,
        "cap": args.cap,
        "defaults": {
            "relative_margin": fusion.RELATIVE_MARGIN,
            "min_similarity": fusion.MIN_SIMILARITY,
        },
    }

    def emit(code: int) -> int:
        text = json.dumps(result, indent=2)
        print(text)
        if args.json_path:
            Path(args.json_path).write_text(text + "\n", encoding="utf-8")
        return code

    try:
        result["pictures"] = count_pictures(args.dsn, table)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return emit(2)
    problems = set_problems(pictures=result["pictures"], labelled_queries=len(labelled.queries))
    if problems:
        result["problems"] = problems
        return emit(1)

    try:
        queries = measure(
            labelled,
            dsn=args.dsn,
            table=table,
            base_url=args.base_url,
            model=model,
            dimensions=args.dimensions,
            pictures=args.pictures,
            timeout=args.timeout,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return emit(2)

    result.update(
        sweep(
            queries,
            margins=parse_grid(args.margins),
            floors=parse_grid(args.floors),
            cap=args.cap,
            threshold=args.threshold,
        )
    )
    return emit(0)


if __name__ == "__main__":
    sys.exit(main())
