"""Tests for `_from_tiled` (task 3.2) — the durable read source `read_run_data`

falls back to once a run's live buffer is gone (post-restart, or evicted past
`live_rows._MAX_RUNS`). This environment has `tiled[client]` but not
`tiled[server]` (no `sqlalchemy`), so there is no in-process Tiled server to
run against — these tests fake the client boundary (`tiled.client.from_uri`)
instead, which is exactly the seam `_from_tiled` itself is built around.
Real-server coverage is the e2e round trip (task 4.1).

Exercised here:

- `BLUESKY_TILED_URI` unset: `_from_tiled` returns `None` without importing
  `tiled` at all (logged, not an error) — matches "Tiled not configured"
  everywhere else in this phase (`_build_tiled_writer_factory`, task 2.5).
- The search query is built as `Key("start.osprey_run_id") == run_id`, NOT a
  bare `Key("osprey_run_id")` — the start doc lives under `metadata["start"]`
  on the run container `TiledWriter.start` creates.
- No matching run: `_from_tiled` returns `None` (caller raises 404).
- A matching run with recorded events: the internal appendable table is read
  and projected onto the live-buffer column set — `seq_num`, `time`, and
  every `ts_*` column dropped — then windowed via `_window` (task 3.1), with
  `run_uid` taken from the start doc's `uid`, never from the caller's
  `run_id`.
- A matching run whose start doc landed but which never got a `"primary"`
  event stream (e.g. it errored before its first point): treated as a real,
  empty run — `{"columns": [], "rows": [], "row_count": 0, "truncated":
  False}` — not a 404.
- A matching run whose `"primary"` stream exists but whose `.base` holds no
  `"internal"` table: NOT the empty-run shape. That is a broken catalog (or
  a broken traversal), and it must surface rather than be laundered into a
  successful empty read.
- Pagination (`max_rows`/`offset`/`tail`) flows through unchanged, proving
  `_from_tiled` delegates to `_window` rather than reimplementing it.

The fakes below model the real Tiled 0.2.12 client surface, which is the
whole point of this file. Against a live server, a run's `"primary"` child is
a `CompositeClient` whose keys are the *flattened column names* — `seq_num`,
`time`, `det1-value`, `ts_det1-value` — NOT the table. Asking it for
`"internal"` raises::

    KeyError: "Key 'internal' not found. If it refers to a table, access it
    via the base Container client using `.base['internal']` instead."

The appendable `DataFrameClient` lives at `primary.base["internal"]`. A fake
that answers `primary["internal"]` directly would let a wrong traversal pass
here and fail only in production — and because `_from_tiled`'s empty-run
branch keys off `"primary"` membership rather than catching `KeyError`, a
wrong traversal now raises instead of returning a plausible empty result.
"""

from __future__ import annotations

import pandas as pd
import pytest

from osprey.services.bluesky_bridge import app as app_module

_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_TILED_URI_ENV, raising=False)
    monkeypatch.delenv(_TILED_API_KEY_ENV, raising=False)
    yield


# =========================================================================
# Fakes for the `tiled.client.from_uri` boundary
# =========================================================================


class _FakeInternalTable:
    """Stands in for a Tiled `DataFrameClient` (the "internal" appendable table)."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def read(self) -> pd.DataFrame:
        return self._df


class _FakeBaseContainer:
    """Stands in for `CompositeClient.base` — a plain Container of table nodes.

    Raises a bare `KeyError` for a missing child, exactly as a real Container
    does. `_from_tiled` must NOT swallow that: an absent `"internal"` here is
    a broken catalog, not an empty run.
    """

    def __init__(self, tables: dict) -> None:
        self._tables = tables

    def __getitem__(self, key: str):
        return self._tables[key]

    def keys(self):
        return list(self._tables)


class _FakeCompositeClient:
    """Stands in for the `CompositeClient` a run's `"primary"` child really is.

    The fidelity that matters: its `__getitem__` exposes the table's flattened
    *columns*, and raises `KeyError` — with the real 0.2.12 message — for any
    other key, `"internal"` included. The table is reachable only via `.base`.

    Modelling this is the entire reason this fake exists. A fake that answered
    `primary["internal"]` would encode the bug it is meant to catch.
    """

    # Verbatim from `tiled.client.composite.CompositeClient.__getitem__` (0.2.12).
    _NOT_FOUND_TEMPLATE = (
        "Key '{key}' not found. If it refers to a table, access it via "
        "the base Container client using `.base['{key}']` instead."
    )

    def __init__(self, columns: list[str], base: _FakeBaseContainer) -> None:
        self._columns = list(columns)
        self.base = base

    def __getitem__(self, key: str):
        if key in self._columns:
            return object()  # a column client; `_from_tiled` never asks for one
        raise KeyError(self._NOT_FOUND_TEMPLATE.format(key=key))

    def keys(self):
        return list(self._columns)


class _FakeRunNode:
    """Stands in for a Tiled run container: `.metadata` plus stream-node lookup.

    `__contains__` mirrors `Mapping.__contains__` (which is what tiled's
    `Container` inherits): a missing key is a `KeyError` from `__getitem__`,
    reported as `False`. `_from_tiled`'s `"primary" not in run_node` guard
    rides on exactly this.
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


def _run_node_with_events(run_uid: str, run_id: str, rows: list[dict]) -> _FakeRunNode:
    """A run container whose `"primary"` composite wraps a table shaped like TiledWriter's
    output: `seq_num`, `time`, the real data columns, then `ts_<key>` per data column.

    The composite's own keys are those column names (matching a live server);
    the table hangs off `.base["internal"]`.
    """
    metadata = {"start": {"uid": run_uid, "osprey_run_id": run_id}, "stop": {}}
    table_rows = []
    for i, row in enumerate(rows):
        table_row = {"seq_num": i + 1, "time": 1000.0 + i, **row}
        table_row.update({f"ts_{k}": 1000.0 + i for k in row})
        table_rows.append(table_row)
    df = pd.DataFrame(table_rows)
    primary = _FakeCompositeClient(
        columns=list(df.columns),
        base=_FakeBaseContainer({"internal": _FakeInternalTable(df)}),
    )
    return _FakeRunNode(metadata=metadata, streams={"primary": primary})


# =========================================================================
# BLUESKY_TILED_URI unset: None, no client built
#
# (Whether `tiled` itself stays unimported in this branch is a module-level
# import-cleanliness property this in-process test cannot observe reliably —
# this venv already has `tiled` loaded from other tests. That invariant is
# enforced by `test_app_import_clean.py`'s subprocess check, task 3.4.)
# =========================================================================


def test_tiled_uri_unset_returns_none() -> None:
    assert app_module._from_tiled("run-1", max_rows=100, offset=None, tail=False) is None


# =========================================================================
# Query shape: Key("start.osprey_run_id") == run_id
# =========================================================================


def test_search_query_targets_start_dot_osprey_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from tiled.queries import Key

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
    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")
    monkeypatch.setenv(_TILED_API_KEY_ENV, "test-api-key")

    run_node = _run_node_with_events(
        run_uid="bluesky-uid-1",
        run_id="run-1",
        rows=[{"motor": 1.0, "det": 10}, {"motor": 2.0, "det": 20}, {"motor": 3.0, "det": 30}],
    )
    client = _FakeTiledClient(nodes_by_run_id={"run-1": run_node})
    _install_fake_client(monkeypatch, client)

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
    assert "partial" not in result  # Tiled-sourced data is always for a completed run


def test_matching_run_pagination_delegates_to_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same `_window` semantics as the live path (task 3.1) — proven here by
    reusing max_rows/offset/tail rather than re-deriving truncation logic.
    """
    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")
    monkeypatch.setenv(_TILED_API_KEY_ENV, "test-api-key")

    rows = [{"motor": float(i)} for i in range(10)]
    run_node = _run_node_with_events(run_uid="bluesky-uid-2", run_id="run-2", rows=rows)
    client = _FakeTiledClient(nodes_by_run_id={"run-2": run_node})
    _install_fake_client(monkeypatch, client)

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
    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")
    monkeypatch.setenv(_TILED_API_KEY_ENV, "test-api-key")

    run_node = _FakeRunNode(
        metadata={"start": {"uid": "bluesky-uid-3", "osprey_run_id": "run-3"}, "stop": {}},
        streams={},  # no "primary" child at all
    )
    client = _FakeTiledClient(nodes_by_run_id={"run-3": run_node})
    _install_fake_client(monkeypatch, client)

    result = app_module._from_tiled("run-3", max_rows=100, offset=None, tail=False)

    assert result == {
        "run_uid": "bluesky-uid-3",
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
    }


# =========================================================================
# The composite traversal itself
# =========================================================================


def test_fake_primary_rejects_direct_internal_lookup_like_a_real_composite() -> None:
    """Guards the fake, not `_from_tiled`.

    If this ever passes by returning a table, the fake has drifted back to the
    wrong shape and every test below it stops proving anything about the real
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
    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")
    monkeypatch.setenv(_TILED_API_KEY_ENV, "test-api-key")

    run_node = _run_node_with_events(
        run_uid="bluesky-uid-real",
        run_id="repro-run-id-123",
        rows=[{"det1-value": 1.0}, {"det1-value": 2.0}, {"det1-value": 3.0}],
    )
    assert run_node["primary"].keys() == ["seq_num", "time", "det1-value", "ts_det1-value"]

    client = _FakeTiledClient(nodes_by_run_id={"repro-run-id-123": run_node})
    _install_fake_client(monkeypatch, client)

    result = app_module._from_tiled("repro-run-id-123", max_rows=100, offset=None, tail=False)

    assert result is not None
    assert result["columns"] == ["det1-value"]
    assert result["rows"] == [[1.0], [2.0], [3.0]]
    assert result["row_count"] == 3


def test_primary_present_but_base_internal_missing_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `"primary"` composite with no `"internal"` table under `.base` is a bug —
    a broken catalog, or a broken traversal — and must NOT be laundered into the
    empty-but-real 200 that the *missing-`"primary"`* case legitimately produces.

    This is the regression that a `try`/`except KeyError` wrapped around the whole
    traversal caused: it returned `{"columns": [], "rows": [], "row_count": 0}` for
    a run holding persisted data. Silent data loss, served as HTTP 200.
    """
    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")
    monkeypatch.setenv(_TILED_API_KEY_ENV, "test-api-key")

    run_node = _FakeRunNode(
        metadata={"start": {"uid": "bluesky-uid-4", "osprey_run_id": "run-4"}, "stop": {}},
        streams={"primary": _FakeCompositeClient(columns=["seq_num"], base=_FakeBaseContainer({}))},
    )
    client = _FakeTiledClient(nodes_by_run_id={"run-4": run_node})
    _install_fake_client(monkeypatch, client)

    # `match` pins *which* KeyError. A traversal that regressed to
    # `primary["internal"]` also raises `KeyError` — with the composite's
    # "Key 'internal' not found. ... use `.base[...]`" guidance — so a bare
    # `pytest.raises(KeyError)`, or even `match=r"internal"`, would pass for
    # the wrong reason. Anchoring to the bare `KeyError("internal")` that a
    # plain Container raises is what makes this test self-sufficient.
    with pytest.raises(KeyError, match=r"^'internal'$"):
        app_module._from_tiled("run-4", max_rows=100, offset=None, tail=False)
