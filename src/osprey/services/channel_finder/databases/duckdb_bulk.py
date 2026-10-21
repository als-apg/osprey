"""Columnar bulk loading of rows into a DuckDB table.

Importing this module is cheap: ``pandas`` is imported only when rows are
loaded, so the health check and other light callers can reach the modules
that use it without pulling a dataframe library into the process.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

#: The name the bulk loader registers its frame under. It is unregistered again
#: before :func:`bulk_insert` returns, so one name serves every table.
BULK_VIEW = "osprey_index_rows"


def bulk_insert(con: Any, table: str, columns: tuple[str, ...], rows: Iterable[Any]) -> int:
    """Bulk-load ``rows`` into ``table``, returning how many were written.

    The rows are pivoted into one column per table column and handed to DuckDB
    as a single registered ``pandas`` frame, which the ``INSERT ... SELECT``
    then reads columnwise. The obvious spelling — one ``executemany`` over the
    row tuples — costs about a quarter of a millisecond per row whatever the
    batch size, because it walks DuckDB's prepared-statement path once per row;
    a hundred thousand rows took nearly forty seconds that way against under a
    second here. ``pandas`` is a runtime dependency already (the archiver
    connectors return frames), so this buys the order of magnitude without
    adding one.

    DuckDB reads each object column's Python values directly, so a list stays a
    list, ``None`` stays SQL ``NULL`` and an empty list stays an empty list. A
    column of nothing but ``None`` arrives typed as the target column by the
    ``INSERT``'s own cast. The target columns are always named, so a table
    column not in ``columns`` takes its default.

    Raises:
        ValueError: When a row does not have one value per column. The
            columnar path would otherwise pad the short row out with nulls and
            load it.
    """
    import pandas as pd

    tuples = [tuple(row) for row in rows]
    if not tuples:
        return 0
    for position, row in enumerate(tuples):
        if len(row) != len(columns):
            raise ValueError(
                f"Row {position} of {table} carries {len(row)} values, but the table "
                f"has {len(columns)} columns: {', '.join(columns)}."
            )

    frame = pd.DataFrame(
        {
            name: list(values)
            for name, values in zip(columns, zip(*tuples, strict=True), strict=True)
        },
        columns=list(columns),
        dtype=object,
    )
    names = ", ".join(columns)
    con.register(BULK_VIEW, frame)
    try:
        con.execute(f"INSERT INTO {table} ({names}) SELECT {names} FROM {BULK_VIEW}")
    finally:
        con.unregister(BULK_VIEW)
    return len(tuples)
