"""PV directory search routes: ``search``, ``export.csv`` and ``count``.

Mounted under ``/api/directory`` by :mod:`.directory_api`. Every route is a GET
that passes its filters to the lifespan's ``DirectoryService`` as keywords and
answers the service's payload; none builds a contract type.
"""

from __future__ import annotations

import csv
import io
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from osprey.interfaces.opaque_origin import document_headers

router = APIRouter()

#: The largest page a search answers.
MAX_PAGE_SIZE = 50
#: The CSV columns: the ``DirectoryRow`` fields, in Contracts order, then the alarm
#: severity and whether the archive records the PV.
_COLUMNS = (
    "name",
    "origin",
    "status",
    "description",
    "record_type",
    "ioc",
    "host",
    "last_announced",
    "alarm",
    "archived",
)
#: The CSV text of a row's ``archived`` field; unknown is empty.
_ARCHIVED_TEXT = {True: "yes", False: "no"}
_CSV = "text/csv"
#: The leading characters that make a spreadsheet read a cell as a formula.
_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")

_Words = Annotated[str, Query()]
_Many = Annotated[list[str] | None, Query()]
_Page = Annotated[int, Query(ge=0)]
_PageSize = Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)]


def _filters(
    q: str,
    name: str | None,
    origin: list[str] | None,
    ioc: list[str] | None,
    record_type: list[str] | None,
    alarm: list[str] | None,
    archived: list[str] | None,
) -> dict[str, Any]:
    """The search's filters, as the service's keywords."""
    return {
        "words": q,
        "name": name,
        "origin": tuple(origin or ()),
        "ioc": tuple(ioc or ()),
        "record_type": tuple(record_type or ()),
        "alarm": tuple(alarm or ()),
        "archived": tuple(archived or ()),
    }


def _service(request: Request) -> Any:
    return request.app.state.directory_service


@router.get("/search")
async def search(
    request: Request,
    q: _Words = "",
    name: str | None = None,
    origin: _Many = None,
    ioc: _Many = None,
    record_type: _Many = None,
    alarm: _Many = None,
    archived: _Many = None,
    page: _Page = 0,
    page_size: _PageSize = MAX_PAGE_SIZE,
) -> dict[str, Any]:
    """One page of the search, as the ``DirectoryPage`` payload."""
    filters = _filters(q, name, origin, ioc, record_type, alarm, archived)
    payload: dict[str, Any] = await _service(request).search(
        **filters, page=page, page_size=page_size
    )
    return payload


@router.get("/count")
async def count(
    request: Request, q: _Words = "", alarm: _Many = None, archived: _Many = None
) -> dict[str, Any]:
    """How many PVs match *q*, *alarm* and *archived*; without them, the target's directory."""
    payload: dict[str, Any] = await _service(request).count(
        words=q, alarm=tuple(alarm or ()), archived=tuple(archived or ())
    )
    return payload


@router.get("/export.csv")
async def export_csv(
    request: Request,
    q: _Words = "",
    name: str | None = None,
    origin: _Many = None,
    ioc: _Many = None,
    record_type: _Many = None,
    alarm: _Many = None,
    archived: _Many = None,
    page: _Page = 0,
    page_size: _PageSize = MAX_PAGE_SIZE,
) -> Response:
    """The search as CSV: the whole set within the filter cap, else pages 0 to *page*.

    The fetched pages stop at the filter cap's rows and at the first short page. The
    first line names what was searched, the second the columns.
    """
    service = _service(request)
    filters = _filters(q, name, origin, ioc, record_type, alarm, archived)
    counted = await service.count(**filters)
    total = counted.get("total") if counted.get("outcome") != "unavailable" else None
    cap = service.filter_cap
    if isinstance(total, int) and 0 < total <= cap:
        answer = await service.search(**filters, page=0, page_size=total)
        _raise_unavailable(answer)
        searched, rows = answer.get("searched", ""), list(answer.get("rows", ()))
    else:
        limit = max(cap, page_size)
        searched, rows = "", []
        for number in range(min(page, (limit - 1) // page_size) + 1):
            answer = await service.search(**filters, page=number, page_size=page_size)
            _raise_unavailable(answer)
            fetched = list(answer.get("rows", ()))
            searched = searched or answer.get("searched", "")
            rows.extend(fetched)
            if len(fetched) < page_size:
                break
        rows = rows[:limit]

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([_cell(searched)])
    writer.writerow(_COLUMNS)
    for row in rows:
        writer.writerow([_cell(_column(row, column)) for column in _COLUMNS])
    return Response(content=out.getvalue(), media_type=_CSV, headers=document_headers(_CSV))


def _raise_unavailable(answer: dict[str, Any]) -> None:
    if answer.get("outcome") == "unavailable":
        raise HTTPException(status_code=503, detail=answer.get("reason") or "unavailable")


def _column(row: dict[str, Any], column: str) -> Any:
    """*row*'s value in *column*; ``alarm`` is the feed's severity, empty for OK or none.

    ``archived`` is ``yes`` or ``no``, empty when unknown or the target has no archive.
    """
    value = row.get(column)
    if column == "archived":
        return _ARCHIVED_TEXT.get(value) if isinstance(value, bool) else None
    if column != "alarm":
        return value
    severity = value.get("severity") if isinstance(value, dict) else None
    return None if severity == "OK" else severity


def _cell(value: Any) -> str:
    """*value* as CSV text; text a spreadsheet would read as a formula gains a leading quote."""
    text = "" if value is None else str(value)
    return f"'{text}" if text.startswith(_FORMULA_STARTS) else text
