"""``GET /runs/{id}/export`` against a REAL Tiled server, at the pinned image tag.

Every other test of the bridge's Tiled surface runs against fakes that model the
SHAPE of the client boundary. That is enough to pin how the bridge composes its
answer, and structurally unable to say anything about what the catalog actually
does: a fake serializes whatever it is asked to, and a fake stream hands back a
tidy all-scalar table with one shared dimension. This module is the other half —
the only place in the suite where the answers come from
``ghcr.io/bluesky/tiled:0.2.12`` itself, written by the same `TiledWriter` the
queueserver worker subscribes.

The tag is pinned, deliberately and exactly, to the one
``docker-compose.yml.j2`` renders for the ``tiled`` service. An export route
delegates its entire output to the server's serializers, so "which formats does
this bridge offer" is a fact about that image and not about the `tiled` library:
the client library registers serializers the server image answers 500 for. The
first test in this file is that verification, run rather than asserted from the
registry.

The server is started with the production command from the compose template,
including its ``duckdb://`` writable target — `TiledWriter` appends event tables
through ``create_appendable_table``/``append_partition``, which need SQL-family
storage, and a filesystem-only catalog fails the write with an opaque 409 on the
run's metadata POST.

Two runs are written, and the second one is the point: it carries a waveform
signal, so its stored table has a list-valued column. A non-scalar data key is
exactly what the fakes cannot model, and it is the case where the image's format
support stops being uniform — ``application/json`` serializes a scalar-only
table and fails on this one.

Skips cleanly when no container runtime is reachable (`is_docker_available`).
"""

from __future__ import annotations

import io
import time
from collections.abc import Iterator
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests._container_support import is_docker_available, start_or_skip, stop_quietly

# xdist_group("docker"): this file starts a real container, and every
# container-starting file in the suite shares one group so that only one xdist
# worker ever has a testcontainers session open — each worker runs its own
# session and its own ryuk reaper, and two concurrent starts race the Docker
# daemon's port mapper (see tests/README.md, "Containers").
pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("docker")]

# The tag the compose template pins for the `tiled` service. Pinned here too,
# and not read from the template: this module's assertions are statements about
# THIS image's serializers, so it has to fail loudly if it is ever pointed at
# another one rather than quietly re-verify a different server.
TILED_IMAGE = "ghcr.io/bluesky/tiled:0.2.12"

# Tiled refuses a non-alphanumeric single-user API key at startup, with the
# failure surfacing only as "application startup failed" in the container log.
API_KEY = "ospreyexportkey0123456789abcdef"

_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"

# The two runs written into the catalog once per module.
SCALAR_RUN = "export-scalar-run"
WAVE_RUN = "export-wave-run"
BARE_RUN = "export-bare-run"

WAVEFORM = np.arange(8, dtype=float)


@pytest.fixture(scope="module")
def tiled_uri() -> Iterator[str]:
    """A live Tiled server at the pinned tag, serving a writable catalog.

    The command is the compose template's, verbatim in shape: a catalog DB under
    ``/storage`` (a path the image ships pre-owned by the uid it runs as) plus
    both writable targets, the filesystem one and the ``duckdb://`` one whose
    four slashes make the path absolute.
    """
    if not is_docker_available():
        pytest.skip("No container runtime reachable — start Docker or Podman to run this test.")

    from testcontainers.core.container import DockerContainer

    container = start_or_skip(
        lambda: (
            DockerContainer(TILED_IMAGE)
            .with_command(
                "tiled serve catalog /storage/catalog.db --init "
                f"--host 0.0.0.0 --port 8000 --api-key {API_KEY} "
                "-w /storage/files -w duckdb:////storage/data.duckdb"
            )
            # No fixed host port: a host running an OSPREY demo stack already
            # holds 8091, and a fixed port would turn that into a flake.
            .with_exposed_ports(8000)
        ),
        label="tiled",
    )
    try:
        uri = _serving_uri(container)
        yield uri
    finally:
        stop_quietly(container)


def _serving_uri(container: Any, timeout: float = 90.0) -> str:
    """Block until the server answers ``/healthz``, then return its base URI.

    The port mapping and the application's own startup are two separate waits,
    and only the second one proves the catalog opened: Tiled exits during
    lifespan startup for a bad key or an unwritable storage path, so a mapped
    port is not evidence that anything is serving behind it.
    """
    deadline = time.monotonic() + timeout
    uri = ""
    while time.monotonic() < deadline:
        try:
            port = container.get_exposed_port(8000)
            uri = f"http://{container.get_container_host_ip()}:{port}"
            if httpx.get(f"{uri}/healthz", timeout=2.0).status_code == 200:
                return uri
        except Exception:  # noqa: S110 - any failure here is "not up yet"
            pass
        time.sleep(0.5)
    logs = ""
    try:
        logs = container.get_logs()[1].decode()[-2000:]
    except Exception:
        pass
    pytest.fail(f"Tiled at {uri or '<unmapped>'} never served /healthz within {timeout}s.\n{logs}")


@pytest.fixture(scope="module")
def stored_runs(tiled_uri: str) -> dict[str, Any]:
    """Three real runs in the catalog, written by `TiledWriter` off a RunEngine.

    One scalar-only run, one carrying a waveform signal, and one that opens and
    closes without ever emitting an event — the "start doc landed, nothing was
    recorded" shape the export route has to refuse by name rather than 500 on.

    Returns the declared column order of each written table, read back off the
    catalog: the assertions about what the bridge serves are made against what
    the catalog actually stored, never against a list written out here by hand.
    """
    from bluesky import RunEngine
    from bluesky import plan_stubs as bps
    from bluesky.callbacks.tiled_writer import TiledWriter
    from bluesky.plans import count
    from ophyd.sim import SynSignal, motor
    from tiled.client import from_uri

    client = from_uri(tiled_uri, api_key=API_KEY)

    det_scalar = SynSignal(func=lambda: 1.23, name="det_scalar")
    det_wave = SynSignal(func=lambda: WAVEFORM.copy(), name="det_wave")

    engine = RunEngine({})
    engine.subscribe(TiledWriter(client))

    engine(count([det_scalar, motor], num=3), osprey_run_id=SCALAR_RUN)
    engine(count([det_scalar, det_wave, motor], num=3), osprey_run_id=WAVE_RUN)

    def _open_and_close() -> Any:
        yield from bps.open_run(md={"osprey_run_id": BARE_RUN})
        yield from bps.close_run()

    engine(_open_and_close())

    return {
        "client": client,
        "columns": {run_id: _stored_columns(client, run_id) for run_id in (SCALAR_RUN, WAVE_RUN)},
    }


def _stored_columns(client: Any, run_id: str) -> list[str]:
    """The column order the catalog itself declares for one run's stored table."""
    return list(_table_node(client, run_id).columns)


def _table_node(client: Any, run_id: str) -> Any:
    """The ``primary`` stream's table part for one run, straight off the catalog."""
    from bluesky_tiled_plugins.queries import Key

    run = max(
        client.search(Key("start.osprey_run_id") == run_id).values(),
        key=lambda r: dict(r.metadata).get("start", {}).get("time", 0),
    )
    return run["primary"].base["internal"]


@pytest.fixture
def impatient_tiled_client(monkeypatch: Any) -> None:
    """Bound Tiled's own retry loop for the tests that make it fail on purpose.

    `retry_context` reads ``TILED_RETRY_ATTEMPTS`` (10) and
    ``TILED_RETRY_TIMEOUT`` (45s) off its module at call time, so a request to a
    catalog that is down or that cannot serialize spends most of a minute backing
    off before the route ever sees the exception. That backoff is Tiled's
    behavior and worth having in production; in a test it is 25 seconds of
    sleeping per case on a lane that runs on every PR.

    Bounding it weakens nothing under test. The assertions are about which
    refusal the ROUTE composes once the client gives up — how many times the
    client tried first is not a claim either test makes.
    """
    from tiled.client import utils as tiled_utils

    monkeypatch.setattr(tiled_utils, "TILED_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(tiled_utils, "TILED_RETRY_TIMEOUT", 1.0)


@pytest.fixture
def bridge(tiled_uri: str, stored_runs: dict[str, Any], monkeypatch: Any) -> Iterator[TestClient]:
    """The bridge app, pointed at the live catalog through the compose env vars.

    `app` is imported inside the fixture, after the container is up, so a module
    with no runtime available skips without paying the bridge's import.
    """
    monkeypatch.setenv(_TILED_URI_ENV, tiled_uri)
    monkeypatch.setenv(_TILED_API_KEY_ENV, API_KEY)

    from osprey.services.bluesky_bridge.app import app

    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# What the pinned image actually serves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run_id", [SCALAR_RUN, WAVE_RUN])
@pytest.mark.parametrize("media_type", ["text/csv", "application/x-parquet"])
def test_pinned_image_serves_every_offered_format_for_every_stored_table(
    stored_runs: dict[str, Any], run_id: str, media_type: str
) -> None:
    """Each format the bridge offers is one the server answers, for both run shapes.

    This is the verification behind `_EXPORT_FORMATS`, run rather than read off
    Tiled's serializer registry. The registry lists what the library CAN
    register; this asserts what the pinned image DOES answer, and it asserts it
    for a table with a list-valued column as well as an all-scalar one, because
    that is where the image's support stops being uniform.
    """
    node = _table_node(stored_runs["client"], run_id)
    response = httpx.get(
        node.item["links"]["full"],
        params={"format": media_type},
        headers={"Authorization": f"Apikey {API_KEY}"},
        timeout=60.0,
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith(media_type)
    assert response.content


@pytest.mark.parametrize("media_type", ["application/x-hdf5", "application/json"])
def test_offering_more_formats_would_ship_a_broken_download(
    stored_runs: dict[str, Any], media_type: str
) -> None:
    """The formats `_EXPORT_FORMATS` leaves out are left out for a reason.

    Both of these are registered for the table family in the shipped image and
    both fail it: ``application/x-hdf5`` for every table, ``application/json``
    for any table holding a list-valued column — which is every run with a
    waveform signal. Pinning the failure is what keeps the two-entry map a
    recorded measurement rather than a guess; if a future image tag fixes one of
    them, this test is the thing that says the map can grow.
    """
    node = _table_node(stored_runs["client"], WAVE_RUN)
    response = httpx.get(
        node.item["links"]["full"],
        params={"format": media_type},
        headers={"Authorization": f"Apikey {API_KEY}"},
        timeout=60.0,
    )

    assert response.status_code != 200, (
        f"{TILED_IMAGE} now serves {media_type}; _EXPORT_FORMATS may be able to offer it."
    )


def test_the_run_container_itself_serves_no_tabular_format(stored_runs: dict[str, Any]) -> None:
    """Why the export resolves down to a table part instead of exporting the run.

    A run node and a stream node are both containers, and the image's container
    family registers ``application/json`` alone — a CSV or parquet request
    against either is refused with 406. `_stored_table_node` walking down to the
    stream's table part is that constraint, not a preference.
    """
    from bluesky_tiled_plugins.queries import Key

    client = stored_runs["client"]
    run = next(iter(client.search(Key("start.osprey_run_id") == WAVE_RUN).values()))

    for node in (run, run["primary"]):
        response = httpx.get(
            node.item["links"]["full"],
            params={"format": "text/csv"},
            headers={"Authorization": f"Apikey {API_KEY}"},
            timeout=60.0,
        )
        assert response.status_code == 406, response.text


# ---------------------------------------------------------------------------
# The route, round-tripped
# ---------------------------------------------------------------------------


def test_csv_export_round_trips_a_completed_run(
    bridge: TestClient, stored_runs: dict[str, Any]
) -> None:
    """SC-9: a completed run downloads as CSV that parses back to what was stored.

    Every stored column, in the catalog's declared order, and one row per point
    — including the ``seq_num``/``time``/``ts_*`` columns the data route projects
    away. An export is the archive, not the live-comparable view.
    """
    import pandas as pd

    response = bridge.get(f"/runs/{SCALAR_RUN}/export", params={"format": "csv"})

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["content-disposition"] == (
        f"attachment; filename=\"{SCALAR_RUN}.csv\"; filename*=UTF-8''{SCALAR_RUN}.csv"
    )

    frame = pd.read_csv(io.BytesIO(response.content))
    assert list(frame.columns) == stored_runs["columns"][SCALAR_RUN]
    assert len(frame) == 3
    assert frame["det_scalar"].tolist() == [1.23, 1.23, 1.23]
    assert frame["seq_num"].tolist() == [1, 2, 3]


def test_parquet_export_round_trips_a_completed_run(
    bridge: TestClient, stored_runs: dict[str, Any]
) -> None:
    """The binary format the image serves, round-tripped through its own reader.

    Asserted on the waveform run: parquet is the format that keeps a list-valued
    column a list, so this is where the binary offering earns its place over
    re-parsing CSV's bracketed cell.
    """
    import pandas as pd

    response = bridge.get(f"/runs/{WAVE_RUN}/export", params={"format": "parquet"})

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-parquet")
    assert response.headers["content-disposition"] == (
        f"attachment; filename=\"{WAVE_RUN}.parquet\"; filename*=UTF-8''{WAVE_RUN}.parquet"
    )
    assert response.content[:4] == b"PAR1"

    frame = pd.read_parquet(io.BytesIO(response.content))
    assert list(frame.columns) == stored_runs["columns"][WAVE_RUN]
    assert len(frame) == 3
    assert np.array_equal(np.asarray(frame["det_wave"].iloc[0], dtype=float), WAVEFORM)


def test_csv_defaults_when_no_format_is_asked_for(bridge: TestClient) -> None:
    """``?format=`` is optional; the text format is what a caller gets by default."""
    response = bridge.get(f"/runs/{SCALAR_RUN}/export")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")


# ---------------------------------------------------------------------------
# The non-scalar data key, against the internal table
# ---------------------------------------------------------------------------


def test_a_waveform_column_survives_the_export_and_the_data_route(
    bridge: TestClient, stored_runs: dict[str, Any]
) -> None:
    """What a non-scalar data key does to both Tiled-backed answers, for real.

    The stored table declares ``det_wave`` as an arrow list column, and neither
    Tiled-backed route drops it: the export carries it because the server
    serializes it, and the data route carries it because a list column reads
    back as a one-dimensional object array, so `_primary_table`'s row-shape rule
    admits the column alongside the scalars.

    Both routes' column lists are asserted against the order the CATALOG
    declares — the export's is that list entire, the data route's is that list
    minus exactly the ``seq_num``/``time``/``ts_*`` projection. That is the
    claim the fakes structurally cannot make: they build their dataset from a
    frame whose column order they chose themselves, so any ordering they agree
    with is their own.
    """
    import pandas as pd

    declared = stored_runs["columns"][WAVE_RUN]
    assert "det_wave" in declared

    exported = pd.read_csv(io.BytesIO(bridge.get(f"/runs/{WAVE_RUN}/export").content))
    assert list(exported.columns) == declared

    data = bridge.get(f"/runs/{WAVE_RUN}/data")
    assert data.status_code == 200, data.text
    body = data.json()

    projected = [c for c in declared if c not in ("seq_num", "time") and not c.startswith("ts_")]
    assert body["columns"] == projected
    assert "det_wave" in body["columns"]


def test_the_read_path_keeps_a_waveform_a_waveform_through_the_client(
    bridge: TestClient,
) -> None:
    """The upstream half of the same defect, measured at the seam it happens on.

    Tiled's table client assembles its Arrow partitions through dask, and dask's
    ``dataframe.convert-string`` (on by default) casts every object-dtype column
    to a string dtype on the way out — which for a column whose cells are arrays
    means ``str`` of each array. The readings are destroyed inside the client,
    before any code in this repo touches them, so no amount of care in the row
    serialization could have recovered them.

    Driven both ways against the real catalog so the claim is a measurement and
    not a citation: with the option on, the cell is text; with it off — which is
    how `_primary_table` reads — it is the array the run wrote.
    """
    import dask

    from osprey.services.bluesky_bridge import app as app_module

    part = app_module._newest_run_node(app_module._tiled_client(), WAVE_RUN)["primary"].base[
        "internal"
    ]

    with dask.config.set({"dataframe.convert-string": True}):
        assert part.read(["det_wave"])["det_wave"].iloc[0] == str(WAVEFORM)

    with dask.config.set({"dataframe.convert-string": False}):
        kept = part.read(["det_wave"])["det_wave"].iloc[0]
    assert np.array_equal(kept, WAVEFORM)


def test_the_data_route_serves_a_waveform_cell_as_a_json_array(
    bridge: TestClient,
) -> None:
    """The readings themselves reach a caller, against a real stored list column.

    ``/runs/{id}/data`` used to build its rows with a bare
    ``frame.values.tolist()``. For a stored list column that array is
    object-dtype, so each cell survived as a `numpy.ndarray` and reached a
    caller as the TEXT ``'[0. 1. 2. 3. 4. 5. 6. 7.]'``: not a JSON array, not
    round-trippable, and indistinguishable from a signal that genuinely recorded
    that string. `_json_rows` converts the object columns' cells now, so the
    waveform arrives as what it is.

    This case is exactly the class of thing only a real server can show — every
    fake in the unit suite builds its rows from a frame it composed itself, so
    the arrow ``list<double>`` round trip through the catalog is not one they
    can make. It reads the value straight back out of the JSON body and compares
    it to the array the run actually wrote.

    The export route is the agreeing half in the same file: the same value comes
    back through Tiled's own serializers as a bracketed CSV cell and as a native
    parquet list. All three now carry the numbers.
    """
    body = bridge.get(f"/runs/{WAVE_RUN}/data").json()
    wave_cell = body["rows"][0][body["columns"].index("det_wave")]

    assert wave_cell == WAVEFORM.tolist()
    assert type(wave_cell) is list
    assert all(type(value) is float for value in wave_cell)


# ---------------------------------------------------------------------------
# Refusals — a download route that fails still has to say why
# ---------------------------------------------------------------------------


def test_an_unsupported_format_is_named_here_not_relayed_from_the_catalog(
    bridge: TestClient,
) -> None:
    """A format this bridge does not offer is refused before any catalog call.

    400 with the offered set in the sentence, never the server's own 406: the
    caller asked this process for something it does not serve, and the list of
    what it does serve is this process's to state.
    """
    response = bridge.get(f"/runs/{SCALAR_RUN}/export", params={"format": "hdf5"})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "unsupported_format"
    assert "csv" in detail["detail"] and "parquet" in detail["detail"]


def test_an_unknown_run_is_a_structured_404(bridge: TestClient) -> None:
    """A run id no catalog entry matches — refused by name, not by a bare status."""
    response = bridge.get("/runs/no-such-run/export")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "unknown_run"


def test_a_run_that_recorded_nothing_is_refused_as_such(bridge: TestClient) -> None:
    """A run whose start doc landed but which emitted no event.

    The run is real and the id resolves, so this is not a 404 — there is simply
    no stored table to serialize, and a caller has to be able to tell the two
    apart. The data route answers the same run with a 200 and an empty column
    list, which is why the export cannot: there is no empty file to hand back
    that would mean the same thing.
    """
    response = bridge.get(f"/runs/{BARE_RUN}/export")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "nothing_stored"

    assert bridge.get(f"/runs/{BARE_RUN}/data").status_code == 200


def test_an_unconfigured_catalog_is_refused_before_any_read(
    tiled_uri: str, stored_runs: dict[str, Any], monkeypatch: Any
) -> None:
    """A deployment with no Tiled at all answers 503, not a 500 out of the client.

    Deliberately not parameterized onto the `bridge` fixture: this is the one
    test that needs the environment variable ABSENT, and it asserts the branch
    every browse-only deployment takes on every export request.
    """
    monkeypatch.delenv(_TILED_URI_ENV, raising=False)

    from osprey.services.bluesky_bridge.app import app

    with TestClient(app) as client:
        response = client.get(f"/runs/{SCALAR_RUN}/export")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "tiled_unavailable"


def test_a_configured_but_unreachable_catalog_is_refused_not_a_500(
    stored_runs: dict[str, Any], impatient_tiled_client: None, monkeypatch: Any
) -> None:
    """The ordinary outage: Tiled is configured, and nothing is listening.

    This is the case the unset-variable test above cannot reach — that one takes
    a plain ``if client is None`` branch that has no way to fail. Here the whole
    catalog resolution runs and throws: `from_uri` performs a real handshake GET
    before it returns a client, so the connection is refused before any run is
    ever looked up.

    It is the routine operational state, not an exotic one — the tiled container
    restarting, the bridge coming up before tiled is healthy, a rotated key, a
    network blip — and the route promises a caller can branch on ``detail.code``
    for all of them. Regression pin: this answered a bare
    ``500 Internal Server Error`` with no body to branch on before the catalog
    resolution was guarded.

    The port is bound and then closed, so it is a port nothing can be listening
    on rather than one that merely looked free.
    """
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]

    monkeypatch.setenv(_TILED_URI_ENV, f"http://127.0.0.1:{closed_port}")
    monkeypatch.setenv(_TILED_API_KEY_ENV, API_KEY)

    from osprey.services.bluesky_bridge.app import app

    with TestClient(app) as client:
        response = client.get(f"/runs/{SCALAR_RUN}/export")

    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "tiled_unavailable"
    # The sentence has to describe THIS failure. The unset-variable branch says
    # "not configured for this deployment", which would be a lie here.
    assert "could not be reached" in detail["detail"]


def test_a_catalog_that_cannot_serialize_is_a_502_not_a_503(
    bridge: TestClient,
    stored_runs: dict[str, Any],
    impatient_tiled_client: None,
    monkeypatch: Any,
) -> None:
    """The other half of the network ladder: reached, and it failed on the format.

    Driven with a real server-side failure rather than a stub, using the fact the
    format probe above already establishes: this image's table family REGISTERS
    ``application/x-hdf5`` and answers 500 for it. Widening `_EXPORT_FORMATS` for
    the length of this test is enough to send a genuine request down the real
    export path and have the real catalog fail it — no fake server, and nothing
    about the failure invented here.

    The status is the point. 503 means "could not reach the catalog, retry
    later"; 502 means "reached it, this run will not serialize", which retrying
    will not fix. A caller that cannot tell those apart cannot decide whether to
    come back.
    """
    from osprey.services.bluesky_bridge import app as app_module

    monkeypatch.setitem(app_module._EXPORT_FORMATS, "hdf5", "application/x-hdf5")

    response = bridge.get(f"/runs/{SCALAR_RUN}/export", params={"format": "hdf5"})

    assert response.status_code == 502, response.text
    assert response.json()["detail"]["code"] == "export_failed"


def test_a_run_id_reaches_the_download_header_sanitized(
    bridge: TestClient, stored_runs: dict[str, Any], monkeypatch: Any
) -> None:
    """A run id is untrusted input on its way into ``Content-Disposition``.

    OSPREY mints run ids as uuid4 hex, so nothing this deployment enqueues can
    carry a quote, a CRLF, or a non-ASCII character — but the catalog is shared,
    and anything else writing through `TiledWriter` can stamp an arbitrary
    string. The header is built from a sanitized derivative for that reason, and
    this asserts the sanitizing rather than the mint.

    The catalog lookup is stubbed to a table node the fixture already wrote:
    what is under test is the header construction, and reaching it otherwise
    would mean writing a hostile run id into a real catalog just to read one
    header back.
    """
    from osprey.services.bluesky_bridge import app as app_module

    node = _table_node(stored_runs["client"], SCALAR_RUN)
    monkeypatch.setattr(app_module, "_stored_table_node", lambda _run_id: node)

    hostile = 'a"; filename="evil.csv'
    response = bridge.get(f"/runs/{hostile}/export")

    assert response.status_code == 200, response.text
    disposition = response.headers["content-disposition"]

    # Every character the injection needed — the quote that would close the
    # parameter, the semicolon that would start another, the `=` that would name
    # it — is an underscore before it reaches the header.
    assert 'filename="a___filename__evil.csv.csv"' in disposition
    # Exactly two parameters, `filename` and `filename*`, so no attacker-chosen
    # third one slipped in. Split on `;` rather than counting a substring:
    # `filename*=` does not contain `filename=`.
    assert [p.split("=")[0].strip() for p in disposition.split(";")] == [
        "attachment",
        "filename",
        "filename*",
    ]
    # The true id survives only in the percent-encoded form, where its quote and
    # semicolon are %22 and %3B and cannot act as syntax.
    assert "filename*=UTF-8''a%22%3B%20filename%3D%22evil.csv.csv" in disposition


def test_a_non_ascii_run_id_does_not_crash_the_header_encoder(
    bridge: TestClient, stored_runs: dict[str, Any], monkeypatch: Any
) -> None:
    """A non-latin-1 run id used to raise inside Starlette's header encoding.

    The sanitized ``filename`` is pure ASCII by construction, and the true id
    rides along percent-encoded in the RFC 5987 ``filename*`` form — so a client
    that understands it still saves the real name, and one that does not gets a
    valid one instead of a 500.
    """
    from osprey.services.bluesky_bridge import app as app_module

    node = _table_node(stored_runs["client"], SCALAR_RUN)
    monkeypatch.setattr(app_module, "_stored_table_node", lambda _run_id: node)

    response = bridge.get("/runs/résumé-run/export")

    assert response.status_code == 200, response.text
    disposition = response.headers["content-disposition"]
    assert 'filename="r_sum_-run.csv"' in disposition
    assert "filename*=UTF-8''r%C3%A9sum%C3%A9-run.csv" in disposition
    assert disposition.isascii()
