"""Tests for the data route's Tiled branch — `_from_tiled` and the shared
`_tiled_run_snapshot` it windows.

`get_run_data` falls back to Tiled once a run's live buffer is gone
(post-restart, or evicted past `live_rows._MAX_RUNS`). This environment has
`tiled[client]` but not `tiled[server]` (no `sqlalchemy`), so there is no
in-process Tiled server to run against — these tests fake the client boundary
(`tiled.client.from_uri`) instead, which is exactly the seam the read path is
built around. Real-server coverage is the e2e round trip.

Exercised here:

- `BLUESKY_TILED_URI` unset: `_from_tiled` returns `None` without importing
  `tiled` at all (logged, not an error) — matches "Tiled not configured"
  everywhere else in this phase.
- The search query is built as `Key("start.osprey_run_id") == run_id`, NOT a
  bare `Key("osprey_run_id")` — the start doc lives under `metadata["start"]`
  on the run container `TiledWriter.start` creates.
- No matching run: `_from_tiled` returns `None` (caller raises 404).
- A matching run with recorded events: the stored table is read and projected
  onto the live-buffer column set — `seq_num`, `time`, and every `ts_*` column
  dropped — then windowed via `_window`, with `run_uid` taken from the start
  doc's `uid`, never from the caller's `run_id`.
- A matching run whose start doc landed but which never got a `"primary"`
  event stream (e.g. it errored before its first point): treated as a real,
  empty run — not a 404.
- A matching run whose `"primary"` stream exists but whose `.base` holds no
  table: NOT the empty-run shape. That is a broken catalog (or a broken
  traversal), and it must surface rather than be laundered into a successful
  empty read.
- Pagination (`max_rows`/`offset`/`tail`) flows through unchanged, proving
  `_from_tiled` delegates to `_window` rather than reimplementing it.
- **Honesty (SC-4):** `partial` is the stop document's ABSENCE, and rows come
  back sorted by `seq_num` however the store ordered them — the same two rules
  the figure route reads by, because both routes read through one snapshot.
- **Cost:** an external array part (a waveform, an image stack) is never
  fetched. Naming it in the read would download the whole payload only for the
  row filter to discard it.
- **No silent empties:** a stream whose parts disagree about length still
  serves its rows. A stream with no table part at all reads as empty rather
  than raising.
- **Order:** the served column order is the stored table's declared order, not
  whatever order the composite read's internal `set` happens to yield.
- **Cell shape:** a waveform reading is a JSON array of plain floats, and a
  non-finite reading is `null` at whatever depth it sits — the same spelling a
  scalar gets, and the only legal JSON one.
- **Outage:** a catalog that is configured but cannot be reached is a 503
  naming the outage, never the 404 that would tell a caller the run does not
  exist and never the bare 500 the route used to answer with.

The fakes below model the real Tiled 0.2.12 client surface, which is the whole
point of this file. Against a live server — with `bluesky-tiled-plugins`
installed, as it is here — a run's `"primary"` child is a stream client whose
keys are the *flattened part names* — `seq_num`, `time`, `det1-value`,
`ts_det1-value` — NOT the table. The read path names the table parts' columns
in a `read()`, and the stream client resolves them; the appendable
`DataFrameClient` they come from lives under `primary.base`, which is where the
fake's `read()` resolves them from too. Asking the stream itself for
`"internal"` raises::

    KeyError: "Key 'internal' not found. If it refers to a table, access it
    via the base Container client using `.base['internal']` instead."

`_FakeCompositeClient.read` is a faithful port of
`tiled.client.composite.CompositeClient.read`'s projection and dimension rules,
not a convenience shim: it selects a table's columns through a `set` (so column
order out of the read really is nondeterministic), it fetches array parts it is
asked for (so an unnecessary fetch is *observable* here rather than only in
production), and it hands every variable a private dimension name when the
parts' leading lengths disagree. A fake that always returned one tidy shared
dimension would make every one of those behaviors structurally untestable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osprey.services.bluesky_bridge import analysis
from osprey.services.bluesky_bridge import app as app_module

_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_TILED_URI_ENV, raising=False)
    monkeypatch.delenv(_TILED_API_KEY_ENV, raising=False)
    # A settled run's statistics are held per run id; a stale entry would let a
    # test read the previous one's answer instead of computing its own.
    analysis._clear()
    yield
    analysis._clear()


# =========================================================================
# Fakes for the `tiled.client.from_uri` boundary
# =========================================================================


class _FakeTablePart:
    """Stands in for a Tiled `DataFrameClient` — a table-family part.

    `columns` comes off the stored structure, so naming this part's columns in
    a read costs nothing; `read_count` records the payload fetches that do.
    """

    structure_family = "table"

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.read_count = 0

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        self.read_count += 1
        return self._df if columns is None else self._df[list(columns)]


class _FakeArrayPart:
    """Stands in for an external array part: a waveform, an image stack.

    `TiledWriter` registers non-scalar data keys as array nodes beside the
    internal table. Its rows are not table rows — one "row" here is a whole
    frame — and `read()` is a full payload fetch, which is why `read_count`
    exists: the read path must never touch it.
    """

    structure_family = "array"

    def __init__(self, values: Any) -> None:
        self._values = np.asarray(values)
        self.read_count = 0

    @property
    def dims(self) -> None:
        return None  # TiledWriter declares no `dims` on external array nodes

    def read(self) -> Any:
        self.read_count += 1
        return self._values


class _FakeBaseContainer:
    """Stands in for `CompositeClient.base` — a plain Container of part nodes.

    Raises a bare `KeyError` for a missing child, exactly as a real Container
    does. The read path must NOT swallow that: a `"primary"` with no table is
    a broken catalog, not an empty run.
    """

    def __init__(self, parts: dict) -> None:
        self._parts = parts

    def __getitem__(self, key: str):
        return self._parts[key]

    def items(self):
        return list(self._parts.items())

    def keys(self):
        return list(self._parts)


class _FakeCompositeClient:
    """Stands in for the stream client a run's `"primary"` child really is.

    Iterating it yields the stream's flattened *part* names — table columns
    flattened out, array parts under their own name — and `read()` resolves
    the named variables off `.base`, returning an `xarray.Dataset`, exactly as
    a real composite read does.

    Its `__getitem__` exposes those flattened names and raises `KeyError` —
    with the real 0.2.12 message — for any other key, `"internal"` included:
    the table is reachable only via `.base`, and a fake that answered
    `primary["internal"]` would encode the bug it is meant to catch.
    """

    # Verbatim from `tiled.client.composite.CompositeClient.__getitem__` (0.2.12).
    _NOT_FOUND_TEMPLATE = (
        "Key '{key}' not found. If it refers to a table, access it via "
        "the base Container client using `.base['{key}']` instead."
    )

    def __init__(self, base: _FakeBaseContainer) -> None:
        self.base = base

    @property
    def _flat_keys(self) -> list[str]:
        keys: list[str] = []
        for name, part in self.base.items():
            keys.extend(part.columns if part.structure_family == "table" else [name])
        return keys

    def __getitem__(self, key: str):
        if key in self._flat_keys:
            return object()  # a column client; the read path never asks for one
        raise KeyError(self._NOT_FOUND_TEMPLATE.format(key=key))

    def __iter__(self):
        return iter(self._flat_keys)

    def keys(self):
        return list(self._flat_keys)

    def read(self, variables=None) -> xr.Dataset:
        """A port of `CompositeClient.read`'s projection and dimension rules.

        Two fidelities carry the weight. Table columns are selected through a
        `set`, so the order variables land in `data_vars` is genuinely
        unstable — anything downstream that wants a stable column order has to
        impose it. And when the parts' leading lengths disagree, every variable
        gets a PRIVATE dimension name instead of a shared `dim0`, which is what
        turns a naive "same dims as seq_num" row filter into a silent empty.
        """
        data_vars: dict[str, Any] = {}
        for name, part in self.base.items():
            if part.structure_family == "table":
                requested = set(part.columns if variables is None else variables)
                df = part.read(list(requested & set(part.columns)))
                for column in set(df.columns):
                    data_vars[column] = df[column].values
            elif variables is None or name in variables:
                data_vars[name] = part.read()

        consistent = len({arr.shape[0] for arr in data_vars.values() if arr.ndim > 0}) <= 1
        return xr.Dataset(
            data_vars={
                name: (
                    (
                        ("dim0", *(f"{name}_dim{i + 1}" for i in range(arr.ndim - 1)))
                        if consistent
                        else tuple(f"{name}_dim{i}" for i in range(arr.ndim))
                    ),
                    arr,
                )
                for name, arr in data_vars.items()
            }
        )


class _FakeRunNode:
    """Stands in for a Tiled run container: `.metadata` plus stream-node lookup.

    `__contains__` mirrors `Mapping.__contains__` (which is what tiled's
    `Container` inherits): a missing key is a `KeyError` from `__getitem__`,
    reported as `False`. The `"primary" not in run_node` guard rides on
    exactly this.
    """

    def __init__(self, metadata: dict, streams: dict | None = None) -> None:
        self.metadata = metadata
        self._streams = streams or {}

    def __getitem__(self, key: str):
        return self._streams[key]

    def __contains__(self, key: str) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True


class _FakeSearchResult:
    def __init__(self, nodes: list[_FakeRunNode]) -> None:
        self._nodes = nodes

    def values(self) -> list[_FakeRunNode]:
        return self._nodes


class _FakeTiledClient:
    """Stands in for the client `tiled.client.from_uri(...)` returns.

    Records every `.search()` query and every `from_uri(...)` call's kwargs
    so tests can assert on the exact query shape and auth used, not just the
    end result.
    """

    def __init__(self, nodes_by_run_id: dict[str, _FakeRunNode]) -> None:
        self._nodes_by_run_id = nodes_by_run_id
        self.search_queries: list[object] = []

    def search(self, query) -> _FakeSearchResult:
        self.search_queries.append(query)
        node = self._nodes_by_run_id.get(query.value)
        return _FakeSearchResult([node] if node is not None else [])


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, client: _FakeTiledClient) -> list[tuple]:
    """Patch `tiled.client.from_uri` to return `client`; return a list `from_uri` calls append to."""
    pytest.importorskip("tiled")

    from tiled.client import from_uri as real_from_uri  # noqa: F401 (import sanity only)

    calls: list[tuple] = []

    def fake_from_uri(uri, **kwargs):
        calls.append((uri, kwargs))
        return client

    monkeypatch.setattr("tiled.client.from_uri", fake_from_uri)
    return calls


def _configured(
    monkeypatch: pytest.MonkeyPatch, run_id: str, node: _FakeRunNode
) -> _FakeTiledClient:
    """Point the bridge at a fake catalog holding exactly *node* under *run_id*."""
    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")
    monkeypatch.setenv(_TILED_API_KEY_ENV, "test-api-key")
    client = _FakeTiledClient(nodes_by_run_id={run_id: node})
    _install_fake_client(monkeypatch, client)
    return client


def _internal_frame(rows: list[dict], *, row_order: list[int] | None = None) -> pd.DataFrame:
    """A table shaped like `TiledWriter`'s internal one: `seq_num`, `time`, the
    real data columns, then `ts_<key>` per data column.

    ``row_order`` stores the rows in that physical order while ``seq_num``
    still records emission order — the read path must sort, so a shuffled
    store and an ordered one must serve the same rows.
    """
    table_rows = []
    for i, row in enumerate(rows):
        table_row = {"seq_num": i + 1, "time": 1000.0 + i, **row}
        table_row.update({f"ts_{k}": 1000.0 + i for k in row})
        table_rows.append(table_row)
    if row_order is not None:
        table_rows = [table_rows[i] for i in row_order]
    return pd.DataFrame(table_rows)


def _run_node_with_events(
    run_uid: str,
    run_id: str,
    rows: list[dict],
    *,
    stopped: bool = True,
    row_order: list[int] | None = None,
    extra_parts: dict | None = None,
) -> _FakeRunNode:
    """A run container whose `"primary"` composite wraps `TiledWriter`-shaped parts."""
    metadata: dict[str, Any] = {"start": {"uid": run_uid, "osprey_run_id": run_id}}
    if stopped:
        metadata["stop"] = {"exit_status": "success"}

    parts: dict[str, Any] = {"internal": _FakeTablePart(_internal_frame(rows, row_order=row_order))}
    parts.update(extra_parts or {})
    primary = _FakeCompositeClient(base=_FakeBaseContainer(parts))
    return _FakeRunNode(metadata=metadata, streams={"primary": primary})


# =========================================================================
# BLUESKY_TILED_URI unset: None, no client built
#
# (Whether `tiled` itself stays unimported in this branch is a module-level
# import-cleanliness property this in-process test cannot observe reliably —
# this venv already has `tiled` loaded from other tests. That invariant is
# enforced by `test_app_import_clean.py`'s subprocess check.)
# =========================================================================


def test_tiled_uri_unset_returns_none() -> None:
    assert app_module._from_tiled("run-1", max_rows=100, offset=None, tail=False) is None


# =========================================================================
# Query shape: Key("start.osprey_run_id") == run_id
# =========================================================================


def test_search_query_targets_start_dot_osprey_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # Imported from the same module `app.py` imports it from. The plugin
    # re-exports Tiled's own `Key` today, so both spellings are the identical
    # object and this assertion would hold either way — but the day the plugin
    # ships its own, an oracle built from the other module stops testing the
    # query the route actually sends.
    from bluesky_tiled_plugins.queries import Key

    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")
    monkeypatch.setenv(_TILED_API_KEY_ENV, "test-api-key")

    client = _FakeTiledClient(nodes_by_run_id={})
    _install_fake_client(monkeypatch, client)

    result = app_module._from_tiled("run-xyz", max_rows=100, offset=None, tail=False)

    assert result is None  # no match seeded
    assert len(client.search_queries) == 1
    assert client.search_queries[0] == (Key("start.osprey_run_id") == "run-xyz")


def test_from_uri_called_with_configured_uri_and_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")
    monkeypatch.setenv(_TILED_API_KEY_ENV, "test-api-key")

    client = _FakeTiledClient(nodes_by_run_id={})
    calls = _install_fake_client(monkeypatch, client)

    app_module._from_tiled("run-xyz", max_rows=100, offset=None, tail=False)

    assert calls == [("http://tiled:8000", {"api_key": "test-api-key"})]


# =========================================================================
# No match -> None (caller raises 404)
# =========================================================================


def test_no_matching_run_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")
    monkeypatch.setenv(_TILED_API_KEY_ENV, "test-api-key")

    client = _FakeTiledClient(nodes_by_run_id={})
    _install_fake_client(monkeypatch, client)

    assert app_module._from_tiled("does-not-exist", max_rows=100, offset=None, tail=False) is None


# =========================================================================
# Matching run with events: projection + windowing
# =========================================================================


def test_matching_run_projects_away_seq_num_time_and_ts_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_node = _run_node_with_events(
        run_uid="bluesky-uid-1",
        run_id="run-1",
        rows=[{"motor": 1.0, "det": 10}, {"motor": 2.0, "det": 20}, {"motor": 3.0, "det": 30}],
    )
    _configured(monkeypatch, "run-1", run_node)

    result = app_module._from_tiled("run-1", max_rows=100, offset=None, tail=False)

    assert result is not None
    assert result["run_uid"] == "bluesky-uid-1"
    assert set(result["columns"]) == {"motor", "det"}
    assert "seq_num" not in result["columns"]
    assert "time" not in result["columns"]
    assert not any(c.startswith("ts_") for c in result["columns"])
    assert result["row_count"] == 3
    assert len(result["rows"]) == 3
    assert result["truncated"] is False
    assert "partial" not in result  # this run's stop document landed


def test_matching_run_pagination_delegates_to_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same `_window` semantics as the live path — proven here by reusing
    max_rows/offset/tail rather than re-deriving truncation logic.
    """
    rows = [{"motor": float(i)} for i in range(10)]
    run_node = _run_node_with_events(run_uid="bluesky-uid-2", run_id="run-2", rows=rows)
    _configured(monkeypatch, "run-2", run_node)

    result = app_module._from_tiled("run-2", max_rows=3, offset=2, tail=False)

    assert result["row_count"] == 10
    assert result["rows"] == [[2.0], [3.0], [4.0]]
    assert result["truncated"] is True

    tail_result = app_module._from_tiled("run-2", max_rows=3, offset=0, tail=True)
    assert tail_result["rows"] == [[7.0], [8.0], [9.0]]
    assert tail_result["truncated"] is True


# =========================================================================
# Matching run, no "primary" stream: real empty run, not a 404
# =========================================================================


def test_matching_run_with_no_primary_stream_reports_empty_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_node = _FakeRunNode(
        metadata={
            "start": {"uid": "bluesky-uid-3", "osprey_run_id": "run-3"},
            "stop": {"exit_status": "abort"},
        },
        streams={},  # no "primary" child at all
    )
    _configured(monkeypatch, "run-3", run_node)

    result = app_module._from_tiled("run-3", max_rows=100, offset=None, tail=False)

    assert result == {
        "run_uid": "bluesky-uid-3",
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        # Settled, but stamped with no plan: nothing declares what this run's
        # columns would have meant, so its statistics are absent with that
        # reason rather than missing from the shape.
        "analysis": analysis.absent(analysis.REASON_PLAN_IDENTITY_UNAVAILABLE),
    }


# =========================================================================
# The composite traversal itself
# =========================================================================


def test_fake_primary_rejects_direct_internal_lookup_like_a_real_composite() -> None:
    """Guards the fake, not the read path.

    If this ever passes by returning a table, the fake has drifted back to the
    wrong shape and every test around it stops proving anything about the real
    traversal.
    """
    run_node = _run_node_with_events("uid", "run", [{"det1-value": 1.0}])
    primary = run_node["primary"]

    with pytest.raises(KeyError, match=r"access it via the base Container client"):
        primary["internal"]

    assert primary.base["internal"].read() is not None


def test_real_tiled_column_set_projects_to_the_live_buffer_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact column set a live Tiled 0.2.12 server returns for a one-detector
    `count` — `['seq_num', 'time', 'det1-value', 'ts_det1-value']` — must project
    down to the single real column `['det1-value']`.
    """
    run_node = _run_node_with_events(
        run_uid="bluesky-uid-real",
        run_id="repro-run-id-123",
        rows=[{"det1-value": 1.0}, {"det1-value": 2.0}, {"det1-value": 3.0}],
    )
    assert run_node["primary"].keys() == ["seq_num", "time", "det1-value", "ts_det1-value"]

    _configured(monkeypatch, "repro-run-id-123", run_node)

    result = app_module._from_tiled("repro-run-id-123", max_rows=100, offset=None, tail=False)

    assert result is not None
    assert result["columns"] == ["det1-value"]
    assert result["rows"] == [[1.0], [2.0], [3.0]]
    assert result["row_count"] == 3


def test_a_table_part_that_cannot_be_read_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream whose table part is listed but cannot be READ is a broken
    catalog, and must NOT be laundered into the empty-but-real 200 that the
    *missing-`"primary"`* case legitimately produces.

    This is the regression that a `try`/`except KeyError` wrapped around the
    whole traversal caused: it returned `{"columns": [], "rows": [],
    "row_count": 0}` for a run holding persisted data. Silent data loss, served
    as HTTP 200. `match` pins that the part's own failure propagates unchanged
    rather than being re-raised as something the caller has to decode.
    """

    class _VanishedTablePart(_FakeTablePart):
        """Listed by the container, gone by the time its payload is fetched."""

        def read(self, columns: list[str] | None = None):
            raise KeyError("internal")

    run_node = _FakeRunNode(
        metadata={"start": {"uid": "bluesky-uid-4", "osprey_run_id": "run-4"}, "stop": {}},
        streams={
            "primary": _FakeCompositeClient(
                base=_FakeBaseContainer(
                    {"internal": _VanishedTablePart(_internal_frame([{"motor": 1.0}]))}
                )
            )
        },
    )
    _configured(monkeypatch, "run-4", run_node)

    with pytest.raises(KeyError, match=r"^'internal'$"):
        app_module._from_tiled("run-4", max_rows=100, offset=None, tail=False)


# =========================================================================
# SC-4: partial from the stop document's absence, rows in seq_num order
# =========================================================================


def test_run_without_a_stop_document_is_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run still executing after a bridge restart has no live buffer and no
    stop document. The data route must say so — the same word, from the same
    signal, the figure route uses.
    """
    run_node = _run_node_with_events(
        run_uid="bluesky-uid-live",
        run_id="run-live",
        rows=[{"motor": 1.0}, {"motor": 2.0}],
        stopped=False,
    )
    _configured(monkeypatch, "run-live", run_node)

    result = app_module._from_tiled("run-live", max_rows=100, offset=None, tail=False)

    assert result is not None
    assert result["partial"] is True
    assert result["row_count"] == 2


def test_settled_run_omits_partial_exactly_as_the_live_path_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`partial` is written only when true. The key's ABSENCE is how a caller
    reads "settled" on the live path, so the Tiled path must not spell the same
    verdict as `partial: false`.
    """
    run_node = _run_node_with_events(
        run_uid="bluesky-uid-done", run_id="run-done", rows=[{"motor": 1.0}]
    )
    _configured(monkeypatch, "run-done", run_node)

    result = app_module._from_tiled("run-done", max_rows=100, offset=None, tail=False)

    assert "partial" not in result


def test_rows_are_sorted_by_seq_num_when_the_store_returns_them_shuffled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The catalog makes no ordering promise; `seq_num` is the emission order.

    Without the sort the route serves a run's points out of order — a plot
    that zig-zags and a window (`tail=True` especially) that is not the run's
    last points at all.
    """
    run_node = _run_node_with_events(
        run_uid="bluesky-uid-shuffled",
        run_id="run-shuffled",
        rows=[{"motor": float(i)} for i in range(5)],
        row_order=[3, 0, 4, 1, 2],
    )
    _configured(monkeypatch, "run-shuffled", run_node)

    result = app_module._from_tiled("run-shuffled", max_rows=100, offset=None, tail=False)

    assert result["rows"] == [[0.0], [1.0], [2.0], [3.0], [4.0]]

    tail_result = app_module._from_tiled("run-shuffled", max_rows=2, offset=0, tail=True)
    assert tail_result["rows"] == [[3.0], [4.0]]


def test_data_and_figure_routes_read_one_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """The anti-drift pin: both Tiled-backed routes go through
    `_tiled_run_snapshot`, so the same run cannot be ordered on one route and
    shuffled on the other, or settled on one and in-flight on the other.
    """
    run_node = _run_node_with_events(
        run_uid="bluesky-uid-parity",
        run_id="run-parity",
        rows=[{"motor": float(i)} for i in range(4)],
        stopped=False,
        row_order=[2, 3, 0, 1],
    )
    _configured(monkeypatch, "run-parity", run_node)

    snapshot = app_module._tiled_run_snapshot("run-parity")
    windowed = app_module._from_tiled("run-parity", max_rows=100, offset=None, tail=False)

    assert snapshot is not None
    assert windowed["columns"] == snapshot["columns"]
    assert windowed["rows"] == snapshot["rows"]
    assert windowed["row_count"] == snapshot["total_seen"]
    assert windowed["partial"] is snapshot["partial"] is True


# =========================================================================
# Reading a run must not cost the frames it carries
# =========================================================================


def test_external_array_parts_are_never_fetched(monkeypatch: pytest.MonkeyPatch) -> None:
    """A waveform or image-stack part is not a column of a row window, and
    naming it in the read downloads the whole payload before the row filter
    discards it. The selection has to happen from structure metadata, which
    costs no fetch — so the array part's `read()` must never be called.
    """
    frames = _FakeArrayPart(np.zeros((3, 8, 8)))
    run_node = _run_node_with_events(
        run_uid="bluesky-uid-waveform",
        run_id="run-waveform",
        rows=[{"motor": 1.0}, {"motor": 2.0}, {"motor": 3.0}],
        extra_parts={"camera-image": frames},
    )
    _configured(monkeypatch, "run-waveform", run_node)

    result = app_module._from_tiled("run-waveform", max_rows=100, offset=None, tail=False)

    assert frames.read_count == 0, "the image part was downloaded only to be discarded"
    assert result["columns"] == ["motor"]
    assert result["rows"] == [[1.0], [2.0], [3.0]]


# =========================================================================
# No silent empties
# =========================================================================


def test_parts_of_differing_length_still_serve_their_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a composite read's parts disagree about length it gives EVERY
    variable a private dimension name instead of a shared `dim0`.

    A row filter that asked "same dims as seq_num" would then match `seq_num`
    alone, and the route would answer `columns: []` with a nonzero row count —
    an empty-but-successful response a caller cannot tell apart from a genuinely
    column-less run. Row-shapedness is decided per variable against the row
    axis's LENGTH instead, so the run's real columns still arrive.
    """
    baseline = _FakeTablePart(pd.DataFrame({"baseline-temp": [20.0, 20.1, 20.2, 20.3, 20.4]}))
    run_node = _run_node_with_events(
        run_uid="bluesky-uid-ragged",
        run_id="run-ragged",
        rows=[{"motor": 1.0}, {"motor": 2.0}, {"motor": 3.0}],
        extra_parts={"baseline": baseline},
    )
    _configured(monkeypatch, "run-ragged", run_node)

    result = app_module._from_tiled("run-ragged", max_rows=100, offset=None, tail=False)

    assert result["columns"] == ["motor"], "the run's own columns collapsed away"
    assert result["rows"] == [[1.0], [2.0], [3.0]]
    assert result["row_count"] == 3


def test_primary_stream_with_no_table_part_reads_empty_rather_than_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream holding only array parts has no rows to serve. That is an empty
    read — the honest answer — not an exception escaping into a 500 from a path
    whose only declared failure is "no run matched".
    """
    run_node = _FakeRunNode(
        metadata={"start": {"uid": "bluesky-uid-arrays", "osprey_run_id": "run-arrays"}},
        streams={
            "primary": _FakeCompositeClient(
                base=_FakeBaseContainer({"camera-image": _FakeArrayPart(np.zeros((2, 4, 4)))})
            )
        },
    )
    _configured(monkeypatch, "run-arrays", run_node)

    result = app_module._from_tiled("run-arrays", max_rows=100, offset=None, tail=False)

    assert result == {
        "run_uid": "bluesky-uid-arrays",
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "partial": True,  # no stop document landed
        # Statistics are computed once a run settles; this one has not.
        "analysis": analysis.absent(analysis.REASON_RUN_IN_PROGRESS),
    }


# =========================================================================
# Column order is the stored order, not the read's set order
# =========================================================================


def test_served_column_order_follows_the_stored_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """`CompositeClient.read` selects a table's columns through a `set`, so the
    order variables come back in varies with the process's string hash salt.

    The order a caller sees must not: a table rendered for an operator would
    reorder its columns across bridge restarts for the same run, and would not
    match the order the live path served. The declared table order is carried
    through the read instead.
    """
    columns = ["alpha", "bravo", "charlie", "delta", "echo"]
    run_node = _run_node_with_events(
        run_uid="bluesky-uid-order",
        run_id="run-order",
        rows=[dict.fromkeys(columns, 1.0), dict.fromkeys(columns, 2.0)],
    )
    _configured(monkeypatch, "run-order", run_node)

    result = app_module._from_tiled("run-order", max_rows=100, offset=None, tail=False)

    assert result["columns"] == columns


# =========================================================================
# A waveform cell is a list of numbers, never text describing one
# =========================================================================


def _waveform_run(
    run_id: str, waves: list[Any], *, scalars: list[Any] | None = None
) -> _FakeRunNode:
    """A run whose internal table carries a list-valued column beside a scalar one.

    This is what an arrow ``list<double>`` column reads back as: an object-dtype
    column whose every cell is its own array. `_FakeArrayPart` is the other,
    unrelated shape — an EXTERNAL array node beside the table, which the read
    path deliberately never fetches. Both exist in a real catalog and only this
    one reaches a served row.
    """
    scalars = [1.0 * (i + 1) for i in range(len(waves))] if scalars is None else scalars
    return _run_node_with_events(
        run_uid=f"bluesky-uid-{run_id}",
        run_id=run_id,
        rows=[
            {"motor": motor, "det_wave": wave} for motor, wave in zip(scalars, waves, strict=True)
        ],
    )


def test_a_waveform_cell_is_served_as_a_json_array_of_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wire shape for a non-scalar reading: a list, made of plain floats.

    An object-dtype column survives ``frame.values.tolist()`` with its cells
    still `numpy.ndarray`, and an ndarray reaching the response encoder came out
    as the TEXT ``'[0. 1. 2.]'`` — not parseable as data, and indistinguishable
    from a signal that genuinely recorded that string. A caller asking for the
    run's data gets the readings, so the cell is the array.

    ``float`` exactly, not "something numeric": a `numpy.float64` would satisfy
    an equality assertion here and still be a type the encoder has to guess at.
    """
    run_node = _waveform_run("run-wave", [np.arange(3.0), np.arange(3.0) + 10.0])
    _configured(monkeypatch, "run-wave", run_node)

    result = app_module._from_tiled("run-wave", max_rows=100, offset=None, tail=False)

    assert result["columns"] == ["motor", "det_wave"]
    assert result["rows"] == [[1.0, [0.0, 1.0, 2.0]], [2.0, [10.0, 11.0, 12.0]]]

    wave_cell = result["rows"][0][1]
    assert type(wave_cell) is list
    assert all(type(value) is float for value in wave_cell)


def test_a_run_of_only_scalars_is_untouched_by_the_waveform_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The common case pays nothing and looks the same as it always did.

    The conversion is decided per column from the frame's dtypes, so a run with
    no object column takes the plain ``tolist()`` path. Pinned because this
    route is polled at 1 Hz over runs that reach six figures of rows: a per-cell
    check would be a real cost paid by every run that has no waveform at all.
    """
    run_node = _run_node_with_events(
        run_uid="bluesky-uid-scalars", run_id="run-scalars", rows=[{"motor": 1.0}, {"motor": 2.0}]
    )
    _configured(monkeypatch, "run-scalars", run_node)

    result = app_module._from_tiled("run-scalars", max_rows=100, offset=None, tail=False)

    assert result["rows"] == [[1.0], [2.0]]
    assert all(type(cell) is float for row in result["rows"] for cell in row)


def test_a_non_finite_reading_is_null_whether_it_is_scalar_or_inside_a_waveform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NaN is handled the same way at both depths, and the body stays legal JSON.

    A detector that could not measure writes NaN, and it can write one into a
    scalar column or into one frame of a waveform. Both come back ``null`` —
    which is the ONLY legal JSON spelling; a bare ``NaN`` token in the body is
    something a strict parser on the other end rejects outright. Asserted
    against the response text rather than the parsed body because Python's own
    ``json.loads`` accepts ``NaN`` and would hide exactly the bug being pinned.
    """
    from fastapi.testclient import TestClient

    wave = np.array([float("nan"), 1.0])
    run_node = _waveform_run("run-nan", [wave], scalars=[float("nan")])
    _configured(monkeypatch, "run-nan", run_node)

    response = TestClient(app_module.app).get("/runs/run-nan/data")

    assert response.status_code == 200, response.text
    assert "NaN" not in response.text
    assert response.json()["rows"] == [[None, [None, 1.0]]]


def test_the_figure_path_leaves_a_waveform_column_out_rather_than_plotting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both Tiled routes read one snapshot, so the new cell shape reaches the figure too.

    A list is not a point on an axis, and the default view's numeric filter
    (`figure._numeric_value`) drops it for the same reason it drops a string
    status: ``float()`` of a list raises, so the column contributes no panel.
    The scalar column beside it still plots, which is what makes this a filter
    rather than a run the figure route gave up on.

    This is a no-change claim, deliberately: an ndarray cell was dropped by the
    same ``float()`` and the same filter before the wire shape was fixed. It is
    pinned anyway because the two routes read ONE snapshot — the data route's
    cell shape is the figure route's cell shape, and nothing else in the file
    would notice if a later change to it started feeding the fit lists.
    """
    run_node = _waveform_run("run-wave-fig", [np.arange(3.0) + i for i in range(4)])
    _configured(monkeypatch, "run-wave-fig", run_node)

    figure = app_module.get_run_figure("run-wave-fig")

    plotted = {
        series["label"]
        for panel in figure["panels"]
        for series in panel["mark"].get("series") or []
    }
    assert "det_wave" not in plotted
    assert plotted == {"motor"}


# =========================================================================
# A catalog that is configured but cannot be reached
# =========================================================================


def test_a_configured_but_unreachable_catalog_is_a_503_not_a_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordinary outage, driven against a real closed port.

    No fake client here on purpose: this is the one case a fake cannot make,
    because the failure happens inside `from_uri`'s handshake GET before any
    catalog object exists to fake. The route answered a bare
    ``500 Internal Server Error`` for it — a body a caller cannot branch on, for
    the routine states of the tiled container restarting, the bridge starting
    first, a rotated key, or a network blip.

    503, never the 404: a 404 says the run does not exist, and saying that about
    a run whose catalog merely blinked sends a caller looking for a mistake it
    did not make. The detail has to say the catalog was configured and could not
    be reached, so an operator is not left deciding between "no Tiled here" and
    "Tiled is down" from a status code alone.
    """
    import socket

    from fastapi.testclient import TestClient
    from tiled.client import utils as tiled_utils

    # Tiled's own retry loop would spend most of a minute backing off before the
    # route ever sees the exception. What refusal the route composes once the
    # client gives up is the whole claim here, so bounding the wait weakens it
    # in no way.
    monkeypatch.setattr(tiled_utils, "TILED_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(tiled_utils, "TILED_RETRY_TIMEOUT", 1.0)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]

    monkeypatch.setenv(_TILED_URI_ENV, f"http://127.0.0.1:{closed_port}")

    response = TestClient(app_module.app).get("/runs/run-unreachable/data")

    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert "could not be reached" in detail
    assert "run-unreachable" in detail


def test_a_catalog_that_raises_mid_read_is_refused_rather_than_crashed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard covers the whole read, not only the client handshake.

    The search, the ``"primary"`` membership test and the part reads all resolve
    against the server too, so any of them can throw when the catalog goes away
    mid-request. This drives the failure from the search — the step AFTER the
    one the closed-port test reaches — so a guard narrowed to client
    construction could not pass both.
    """
    from fastapi.testclient import TestClient

    class _VanishingClient(_FakeTiledClient):
        def search(self, query):
            raise OSError("connection reset by peer")

    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")
    _install_fake_client(monkeypatch, _VanishingClient(nodes_by_run_id={}))

    response = TestClient(app_module.app).get("/runs/run-vanishing/data")

    assert response.status_code == 503, response.text
    assert "connection reset by peer" in response.json()["detail"]


def test_a_run_no_catalog_knows_is_still_a_404_not_an_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must not turn "no such run" into "come back later".

    A reachable catalog that simply holds no match is an answer, and it keeps
    the 404 the MCP tool maps to `unknown_run`. Pinned beside the 503 because a
    guard that swallowed this distinction would make every unknown id look like
    an outage and every outage look retryable in the same breath.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")
    _install_fake_client(monkeypatch, _FakeTiledClient(nodes_by_run_id={}))

    response = TestClient(app_module.app).get("/runs/run-missing/data")

    assert response.status_code == 404
    assert "unknown run" in response.json()["detail"]
