"""The page's one way to the telemetry store: a view-contract query as a view user.

Every call is a POST as the store's ``history`` account whose body is
``EXECUTE AS <view user> <query>`` (GET implies ``readonly``, which refuses
``IMPERSONATE``), with the query's values as the store's typed ``param_<name>``
parameters and three settings on the URL of **every** call:

* ``log_comment=<person>:<view user>``: who asked, in the store's query log;
* ``quota_key=<person>``: the view users' per-person quota key;
* ``http_wait_end_of_query=1``: the store buffers the whole answer before it
  sends a status, so a limit hit part way through is a non-200 carrying the
  store's error and never a 200 with a cut-short body.

At most :data:`~osprey.deployment.telemetry_config.HISTORY_CONCURRENCY` calls
run at once: the page's queries are internal to the store and escape every
per-user counter, so this semaphore is their only bound.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from osprey.deployment.telemetry_config import HISTORY_CONCURRENCY
from osprey.deployment.telemetry_store.accounts import HISTORY_SECONDS, HISTORY_USER
from osprey.history import queries
from osprey.history.viewer import Viewer

__all__ = [
    "HistoryStore",
    "StoreRefusal",
    "store_params",
    "store_body",
]

#: The settings sent on the URL of every call, beside the query's parameters.
WAIT_END_OF_QUERY = "http_wait_end_of_query"

_ERROR_NAME = re.compile(r"\(([A-Z][A-Z0-9_]+)\)")
_ERROR_CODE = re.compile(r"Code:\s*(\d+)")


@dataclass(frozen=True)
class StoreRefusal(Exception):
    """The store answered a query with an error (or could not be reached)."""

    status: int
    name: str
    code: str
    detail: str

    def __str__(self) -> str:
        label = self.name or "error"
        return f"{label} ({self.code})" if self.code else label


def store_params(viewer: Viewer, name: str, values: Mapping[str, Any]) -> dict[str, str]:
    """The URL parameters of one call: the typed values and the page's settings."""
    return {
        **queries.query_parameters(name, values),
        "log_comment": f"{viewer.person}:{viewer.view_user}",
        "quota_key": viewer.person,
        WAIT_END_OF_QUERY: "1",
    }


def store_body(viewer: Viewer, name: str) -> str:
    """The body of one call: the contract query, run as the viewer's view user."""
    sql = queries.load(name).strip().rstrip(";")
    return f"EXECUTE AS `{viewer.view_user}` {sql}\nFORMAT JSONEachRow"


def _refusal(response: httpx.Response) -> StoreRefusal:
    text = response.text
    code = response.headers.get("X-ClickHouse-Exception-Code", "")
    if not code:
        match = _ERROR_CODE.search(text)
        code = match.group(1) if match else ""
    names = _ERROR_NAME.findall(text)
    return StoreRefusal(
        status=response.status_code,
        name=names[-1] if names else "",
        code=code,
        detail=text.strip()[:600],
    )


class HistoryStore:
    """Runs view-contract queries on the store as the ``history`` account."""

    def __init__(
        self,
        url: str,
        password: str,
        *,
        client: httpx.AsyncClient,
        concurrency: int = HISTORY_CONCURRENCY,
    ) -> None:
        self._url = url.rstrip("/") + "/"
        self._password = password
        self._client = client
        self._slots = asyncio.Semaphore(concurrency)

    async def rows(
        self, viewer: Viewer, name: str, values: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """The rows query *name* returns for *viewer*.

        Raises:
            StoreRefusal: The store refused the query or could not be reached.
        """
        params = store_params(viewer, name, values)
        body = store_body(viewer, name)
        async with self._slots:
            try:
                response = await self._client.post(
                    self._url,
                    params=params,
                    content=body.encode("utf-8"),
                    headers={"X-ClickHouse-User": HISTORY_USER, "X-ClickHouse-Key": self._password},
                    timeout=HISTORY_SECONDS + 5,
                )
            except httpx.HTTPError as exc:
                raise StoreRefusal(
                    status=0, name="STORE_UNREACHABLE", code="", detail=str(exc)
                ) from exc
        if response.status_code != 200 or response.headers.get("X-ClickHouse-Exception-Code"):
            raise _refusal(response)
        return [json.loads(line) for line in response.text.splitlines() if line.strip()]
