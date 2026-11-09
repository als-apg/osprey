"""PV directory search routes: ``search``, ``export.csv`` and ``count``.

Mounted under ``/api/directory`` by :mod:`.directory_api`. Every route is a GET
that passes its filters to the lifespan's ``DirectoryService`` as keywords and
answers the service's payload; none builds a contract type.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from osprey.interfaces.opaque_origin import document_headers

router = APIRouter()

#: The largest page a search answers.
MAX_PAGE_SIZE = 50
#: The largest set the export fetches whole when ``display.filter_cap`` is unset; the
#: service's default, so the export fetches whole exactly the sets the service facets.
_DEFAULT_FILTER_CAP = 5000
#: The CSV columns: the ``DirectoryRow`` fields, in Contracts order.
_COLUMNS = (
    "name",
    "origin",
    "status",
    "description",
    "record_type",
    "ioc",
    "host",
    "last_announced",
)
_CSV = "text/csv"

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
) -> dict[str, Any]:
    """The search's filters, as the service's keywords."""
    return {
        "words": q,
        "name": name,
        "origin": tuple(origin or ()),
        "ioc": tuple(ioc or ()),
        "record_type": tuple(record_type or ()),
    }


def _service(request: Request) -> Any:
    return request.app.state.directory_service


def _filter_cap(config: Any) -> int:
    section = config.get("pv_directory") if isinstance(config, Mapping) else None
    display = section.get("display") if isinstance(section, Mapping) else None
    value = display.get("filter_cap") if isinstance(display, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return _DEFAULT_FILTER_CAP
    return value


@router.get("/search")
async def search(
    request: Request,
    q: _Words = "",
    name: str | None = None,
    origin: _Many = None,
    ioc: _Many = None,
    record_type: _Many = None,
    page: _Page = 0,
    page_size: _PageSize = MAX_PAGE_SIZE,
) -> dict[str, Any]:
    """One page of the search, as the ``DirectoryPage`` payload."""
    filters = _filters(q, name, origin, ioc, record_type)
    payload: dict[str, Any] = await _service(request).search(
        **filters, page=page, page_size=page_size
    )
    return payload


@router.get("/count")
async def count(request: Request, q: _Words = "") -> dict[str, Any]:
    """How many PVs match *q*; without it, the size of the current target's directory."""
    payload: dict[str, Any] = await _service(request).count(words=q)
    return payload


@router.get("/export.csv")
async def export_csv(
    request: Request,
    q: _Words = "",
    name: str | None = None,
    origin: _Many = None,
    ioc: _Many = None,
    record_type: _Many = None,
    page: _Page = 0,
    page_size: _PageSize = MAX_PAGE_SIZE,
) -> Response:
    """The search as CSV: the whole set within the filter cap, else pages 0 to *page*.

    The first line names what was searched, the second the columns.
    """
    service = _service(request)
    filters = _filters(q, name, origin, ioc, record_type)
    counted = await service.count(**filters)
    total = counted.get("total") if counted.get("outcome") != "unavailable" else None
    cap = _filter_cap(getattr(request.app.state, "config", None))
    if isinstance(total, int) and 0 < total <= cap:
        pages = [await service.search(**filters, page=0, page_size=total)]
    else:
        pages = [
            await service.search(**filters, page=number, page_size=page_size)
            for number in range(page + 1)
        ]
    for answer in pages:
        if answer.get("outcome") == "unavailable":
            raise HTTPException(status_code=503, detail=answer.get("reason") or "unavailable")

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([pages[0].get("searched", "")])
    writer.writerow(_COLUMNS)
    for answer in pages:
        for row in answer.get("rows", ()):
            writer.writerow(["" if row.get(c) is None else row.get(c) for c in _COLUMNS])
    return Response(content=out.getvalue(), media_type=_CSV, headers=document_headers(_CSV))
