"""The bridge's document plane: CurveZMQ-gated proxy -> dispatcher -> live rows.

Two things are load-bearing here and both are exercised against a real 0MQ
proxy rather than a mock, because both are properties of the sockets and not
of our Python:

1. **Documents published by an authenticated worker become live rows** under
   the OSPREY run id the enqueue path threaded through item metadata — the
   identity the read path looks runs up by — not the RunEngine uid, which the
   bridge never chose and cannot map back to an enqueued item.
2. **Documents published without the client key never arrive.** The
   queueserver container is dual-homed onto the shared network, so key
   authentication is the only thing protecting this socket; a regression that
   quietly dropped `in_curve` would be invisible to any test that only checks
   the happy path.

Proving an *absence* over 0MQ takes care, because a freshly connected PUB
socket drops everything it publishes before its handshake finishes — so "I
published a forgery and nothing showed up" is true whether or not the key was
ever checked. Two things make the negative assertion real: the forged
publisher republishes in a loop for seconds (far longer than an authorized
publisher needs to get a document through, which every test measures for
itself in `_establish_link`), and an authenticated document published
afterwards must still arrive, proving the plane never stopped delivering.
Swapping the forged publisher's config for a valid one must make this test
fail; it does.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import zmq.auth

from osprey.services.bluesky_bridge import document_plane, live_rows
from osprey.services.bluesky_bridge.document_plane import (
    DocumentPlane,
    DocumentPlaneConfig,
    DocumentPlaneError,
    RunDocumentRouter,
    expected_points_from_params,
)

pytestmark = pytest.mark.unit

# Generous relative to the ~0.1 s a local 0MQ hop actually takes; these are
# upper bounds on a *failure*, not sleeps — the waits return as soon as the
# condition holds.
_DELIVERY_TIMEOUT = 15.0
_LINK_TIMEOUT = 15.0
_POLL_INTERVAL = 0.05

# How long an unauthenticated publisher is given to prove itself rejected. This
# one IS spent in full — the assertion is an absence, so the cost buys the
# test its teeth. Orders of magnitude longer than the single probe an
# authorized publisher needs to get through.
_FORGED_GRACE = 3.0


@pytest.fixture(autouse=True)
def _clean_buffers():
    live_rows._clear()
    document_plane._clear()
    yield
    live_rows._clear()
    document_plane._clear()


@pytest.fixture
def certificates(tmp_path: Path) -> dict[str, Path]:
    """CURVE keypairs for the proxy, an authorized publisher, and an intruder.

    The intruder's keypair is every bit as valid as the publisher's — the only
    difference is that its public certificate is never copied into the
    accepted-client directory. Both keypairs are minted here, before the plane
    starts, because pyzmq's authenticator reads that directory once when the
    socket is configured (see `test_documents_from_an_unpinned_keypair_are_rejected`).
    """
    server_dir = tmp_path / "server"
    client_dir = tmp_path / "client"
    intruder_dir = tmp_path / "intruder"
    allowed_dir = tmp_path / "allowed"
    for directory in (server_dir, client_dir, intruder_dir, allowed_dir):
        directory.mkdir()

    server_public, server_secret = zmq.auth.create_certificates(server_dir, "proxy")
    client_public, client_secret = zmq.auth.create_certificates(client_dir, "publisher")
    _, intruder_secret = zmq.auth.create_certificates(intruder_dir, "intruder")
    # ServerCurve pins a *directory* of accepted client public keys. Only the
    # publisher's goes in; the intruder's stays out, which is the whole point.
    (allowed_dir / Path(client_public).name).write_bytes(Path(client_public).read_bytes())

    return {
        "server_public": Path(server_public),
        "server_secret": Path(server_secret),
        "client_public": Path(client_public),
        "client_secret": Path(client_secret),
        "intruder_secret": Path(intruder_secret),
        "allowed": allowed_dir,
    }


@pytest.fixture
def plane(certificates: dict[str, Path]):
    """A started document plane on kernel-assigned loopback ports."""
    started = DocumentPlane(
        DocumentPlaneConfig(
            publish_address="tcp://127.0.0.1",
            subscribe_address="tcp://127.0.0.1",
            server_secret_key=certificates["server_secret"],
            client_public_keys=certificates["allowed"],
        )
    ).start()
    try:
        yield started
    finally:
        started.stop()


def _publisher(plane: DocumentPlane, certificates: dict[str, Path] | None = None):
    """A ``Publisher`` aimed at *plane*, authenticated only if given certificates."""
    from bluesky.callbacks.zmq import ClientCurve, Publisher

    curve = None
    if certificates is not None:
        curve = ClientCurve(
            secret_path=certificates["client_secret"],
            server_public_key=certificates["server_public"],
        )
    return Publisher(plane.publish_address, curve_config=curve)


def _wait_until(predicate, timeout: float = _DELIVERY_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(_POLL_INTERVAL)
    return predicate()


def _start_doc(uid: str, run_id: str | None = None, **extra: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {"uid": uid, "time": 1.0, "scan_id": 1, **extra}
    if run_id is not None:
        doc["osprey_run_id"] = run_id
    return doc


def _descriptor_doc(uid: str, run_uid: str) -> dict[str, Any]:
    return {"uid": uid, "run_start": run_uid, "time": 1.0, "name": "primary", "data_keys": {}}


def _event_doc(descriptor_uid: str, seq_num: int, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "uid": f"event-{descriptor_uid}-{seq_num}",
        "descriptor": descriptor_uid,
        "seq_num": seq_num,
        "time": 1.0,
        "data": data,
        "timestamps": dict.fromkeys(data, 1.0),
    }


def _stop_doc(run_uid: str, num_events: int) -> dict[str, Any]:
    return {
        "uid": f"stop-{run_uid}",
        "run_start": run_uid,
        "time": 2.0,
        "exit_status": "success",
        "num_events": {"primary": num_events},
    }


def _establish_link(publisher, plane: DocumentPlane) -> None:
    """Publish probe runs until one lands, so a later silence means rejection.

    0MQ PUB sockets drop messages published before the subscriber's handshake
    completes (the "slow joiner" problem), so a test that publishes once and
    waits would be flaky in the happy case and *vacuous* in the forged case.
    Probing until a document round-trips proves the link is live; the probe's
    own buffer is then dropped so it cannot be mistaken for real data.
    """
    probe_uid = "probe-run-uid"
    landed = _wait_until(
        lambda: (
            (publisher("start", _start_doc(probe_uid)), live_rows.get(probe_uid))[1] is not None
        ),
        timeout=_LINK_TIMEOUT,
    )
    assert landed, "the authenticated publisher never reached the proxy"
    live_rows._clear()


def test_rows_land_under_the_osprey_run_id(plane, certificates):
    """A worker's documents become live rows keyed by the enqueued run id."""
    publisher = _publisher(plane, certificates)
    _establish_link(publisher, plane)

    document_plane.record_run_params(
        "osprey-run-1", {"detectors": ["bpm"], "axes": [{"num_points": 3}]}
    )

    run_uid = "run-engine-uid-1"
    descriptor_uid = "descriptor-1"
    publisher("start", _start_doc(run_uid, run_id="osprey-run-1"))
    publisher("descriptor", _descriptor_doc(descriptor_uid, run_uid))
    for seq, value in enumerate([1.0, 2.0, 3.0], start=1):
        publisher("event", _event_doc(descriptor_uid, seq, {"corrector": value, "bpm": -value}))
    publisher("stop", _stop_doc(run_uid, 3))

    assert _wait_until(lambda: (live_rows.get("osprey-run-1") or {}).get("partial") is False), (
        "the run's stop document never arrived"
    )

    buffer = live_rows.get("osprey-run-1")
    assert buffer is not None
    assert buffer["columns"] == ["corrector", "bpm"]
    assert buffer["rows"] == [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]]
    assert buffer["total_seen"] == 3
    assert buffer["partial"] is False
    # The RunEngine's own uid is not an identity the bridge records under.
    assert live_rows.get(run_uid) is None

    assert document_plane.progress("osprey-run-1") == {
        "rows_seen": 3,
        "expected_points": 3,
        "fraction": 1.0,
        "complete": True,
    }


def _assert_refused(forged, authorized, label: str) -> None:
    """Assert *forged* never gets a document through, then fence with *authorized*.

    The forged publisher gets the *same* grace an authenticated one gets — it
    republishes in a loop for :data:`_FORGED_GRACE` seconds, far longer than
    the single probe an authorized link needs. Publishing once and looking
    immediately would pass whether or not the key was checked, since a fresh
    0MQ socket drops what it publishes before its handshake completes.
    """
    run_uid = f"{label}-run-uid"
    run_id = f"{label}-run"
    deadline = time.monotonic() + _FORGED_GRACE
    while time.monotonic() < deadline:
        forged("start", _start_doc(run_uid, run_id=run_id))
        forged("descriptor", _descriptor_doc(f"{label}-descriptor", run_uid))
        forged("event", _event_doc(f"{label}-descriptor", 1, {"corrector": 99.0}))
        assert live_rows.get(run_id) is None, f"the {label} publisher was accepted"
        time.sleep(_POLL_INTERVAL)

    # Fence: a document published down the authenticated socket AFTER the
    # forgery, proving the plane was still delivering the whole time.
    fence_id = f"fence-after-{label}"
    authorized("start", _start_doc(f"{fence_id}-uid", run_id=fence_id))
    assert _wait_until(lambda: live_rows.get(fence_id) is not None), (
        "the authenticated publisher stopped being delivered"
    )

    assert live_rows.get(run_id) is None
    assert live_rows.get(run_uid) is None
    assert document_plane.progress(run_id) is None


def test_documents_without_the_client_key_are_rejected(plane, certificates):
    """CurveZMQ, not network placement, is what protects the in-side socket."""
    authorized = _publisher(plane, certificates)
    _establish_link(authorized, plane)

    _assert_refused(_publisher(plane, certificates=None), authorized, "unkeyed")


def test_documents_from_an_unpinned_keypair_are_rejected(plane, certificates):
    """A well-formed key the accepted-client directory doesn't list is still a stranger.

    This is the difference the pinned directory buys over ``CURVE_ALLOW_ANY``:
    the intruder holds a perfectly valid CURVE keypair and knows the proxy's
    public key — everything an allow-any socket asks for — and is refused
    solely because its public certificate is not in the accepted-client
    directory.

    Both keypairs exist before the plane starts, which matters: pyzmq's
    authenticator reads the accepted-client directory when the socket is
    configured, so dropping a new key in afterwards changes nothing until the
    bridge restarts. Copying the intruder's public certificate into that
    directory in the fixture — i.e. before startup — must make this test fail;
    it does.
    """
    authorized = _publisher(plane, certificates)
    _establish_link(authorized, plane)

    intruder = _publisher(
        plane,
        {
            "client_secret": certificates["intruder_secret"],
            "server_public": certificates["server_public"],
        },
    )
    _assert_refused(intruder, authorized, "unpinned")


def test_a_run_without_an_osprey_run_id_falls_back_to_its_run_uid(plane, certificates):
    """A worker driven outside the queue is still recorded — under its only identity."""
    publisher = _publisher(plane, certificates)
    _establish_link(publisher, plane)

    run_uid = "unmanaged-run-uid"
    publisher("start", _start_doc(run_uid))
    publisher("descriptor", _descriptor_doc("unmanaged-descriptor", run_uid))
    publisher("event", _event_doc("unmanaged-descriptor", 1, {"bpm": 0.5}))

    assert _wait_until(lambda: (live_rows.get(run_uid) or {}).get("total_seen") == 1)
    buffer = live_rows.get(run_uid)
    assert buffer is not None
    assert buffer["rows"] == [[0.5]]


def test_start_documents_declaring_num_points_seed_progress(plane, certificates):
    """A plan that declares its own point count gives progress a denominator."""
    publisher = _publisher(plane, certificates)
    _establish_link(publisher, plane)

    run_uid = "declared-run-uid"
    publisher("start", _start_doc(run_uid, run_id="declared-run", num_points=4))
    publisher("descriptor", _descriptor_doc("declared-descriptor", run_uid))
    publisher("event", _event_doc("declared-descriptor", 1, {"bpm": 0.5}))

    assert _wait_until(
        lambda: (document_plane.progress("declared-run") or {}).get("rows_seen") == 1
    )
    assert document_plane.progress("declared-run") == {
        "rows_seen": 1,
        "expected_points": 4,
        "fraction": 0.25,
        "complete": False,
    }


def test_the_worker_publisher_and_this_proxy_speak_the_same_configuration(plane, certificates):
    """The two halves are built by different modules from the same env contract.

    Every other test in this file builds its publisher by hand, which proves
    the socket works but not that the *worker* would produce a matching one:
    the two sides live in separate containers and are wired by env vars whose
    names could drift apart silently. This one builds the client half through
    ``qserver_startup.build_zmq_publisher`` — the real code the RE worker
    runs — from the worker-side env, aimed at the address this plane bound.
    """
    from osprey.services.bluesky_bridge import qserver_startup

    publisher = qserver_startup.build_zmq_publisher(
        {
            "BLUESKY_ZMQ_PUBLISH_ADDR": plane.publish_address,
            "BLUESKY_ZMQ_CURVE_SECRET_KEY": str(certificates["client_secret"]),
            "BLUESKY_ZMQ_CURVE_SERVER_PUBLIC_KEY": str(certificates["server_public"]),
        }
    )
    assert publisher is not None
    _establish_link(publisher, plane)

    run_uid = "worker-built-run-uid"
    publisher("start", _start_doc(run_uid, run_id="worker-built-run"))
    publisher("descriptor", _descriptor_doc("worker-descriptor", run_uid))
    publisher("event", _event_doc("worker-descriptor", 1, {"bpm": 4.2}))

    # Wait for the EVENT doc, not just the start doc: the run appears in
    # live_rows as soon as "start" is processed, while the row from "event"
    # can still be in flight behind it on the same socket.
    assert _wait_until(lambda: bool((live_rows.get("worker-built-run") or {}).get("rows")))
    buffer = live_rows.get("worker-built-run")
    assert buffer is not None
    assert buffer["rows"] == [[4.2]]


# --------------------------------------------------------------- configuration


def test_from_env_returns_none_when_no_document_plane_is_deployed():
    assert DocumentPlaneConfig.from_env({}) is None


def test_from_env_refuses_to_bind_without_a_server_key():
    with pytest.raises(DocumentPlaneError, match="unauthenticated document socket"):
        DocumentPlaneConfig.from_env({"BLUESKY_ZMQ_PUBLISH_ADDR": "tcp://*:5567"})


def test_from_env_refuses_a_missing_server_key_file(tmp_path: Path):
    with pytest.raises(DocumentPlaneError, match="readable file"):
        DocumentPlaneConfig.from_env(
            {
                "BLUESKY_ZMQ_PUBLISH_ADDR": "tcp://*:5567",
                "BLUESKY_ZMQ_CURVE_SECRET_KEY": str(tmp_path / "absent.key_secret"),
            }
        )


def test_from_env_refuses_a_missing_client_key_directory(certificates, tmp_path: Path):
    with pytest.raises(DocumentPlaneError, match="not a directory"):
        DocumentPlaneConfig.from_env(
            {
                "BLUESKY_ZMQ_PUBLISH_ADDR": "tcp://*:5567",
                "BLUESKY_ZMQ_CURVE_SECRET_KEY": str(certificates["server_secret"]),
                "BLUESKY_ZMQ_CURVE_CLIENT_PUBLIC_KEYS": str(tmp_path / "absent-dir"),
            }
        )


def test_from_env_refuses_to_accept_any_client_key(certificates):
    """No accepted-client directory means CURVE_ALLOW_ANY, which is not authentication."""
    with pytest.raises(DocumentPlaneError, match="well-formed"):
        DocumentPlaneConfig.from_env(
            {
                "BLUESKY_ZMQ_PUBLISH_ADDR": "tcp://*:5567",
                "BLUESKY_ZMQ_CURVE_SECRET_KEY": str(certificates["server_secret"]),
            }
        )


def test_from_env_reads_the_full_deployed_configuration(certificates):
    config = DocumentPlaneConfig.from_env(
        {
            "BLUESKY_ZMQ_PUBLISH_ADDR": "tcp://*:5567",
            "BLUESKY_ZMQ_SUBSCRIBE_ADDR": "tcp://127.0.0.1:5568",
            "BLUESKY_ZMQ_CURVE_SECRET_KEY": str(certificates["server_secret"]),
            "BLUESKY_ZMQ_CURVE_CLIENT_PUBLIC_KEYS": str(certificates["allowed"]),
        }
    )
    assert config == DocumentPlaneConfig(
        publish_address="tcp://*:5567",
        subscribe_address="tcp://127.0.0.1:5568",
        server_secret_key=certificates["server_secret"],
        client_public_keys=certificates["allowed"],
    )


def test_start_from_env_is_a_no_op_without_a_document_plane():
    assert document_plane.start_from_env({}) is None
    document_plane.shutdown()  # idempotent on a plane that never started


def test_start_from_env_propagates_a_refusal_to_bind_unauthenticated():
    """A misconfigured key is the bridge's problem, not something to degrade past."""
    with pytest.raises(DocumentPlaneError):
        document_plane.start_from_env({"BLUESKY_ZMQ_PUBLISH_ADDR": "tcp://*:5567"})


def test_start_from_env_degrades_when_the_socket_cannot_be_brought_up(
    monkeypatch, certificates, caplog
):
    """A taken port costs the operator live rows, never the whole bridge."""

    def _explode(self):
        raise OSError("address already in use")

    monkeypatch.setattr(DocumentPlane, "start", _explode)
    env = {
        "BLUESKY_ZMQ_PUBLISH_ADDR": "tcp://*:5567",
        "BLUESKY_ZMQ_CURVE_SECRET_KEY": str(certificates["server_secret"]),
        "BLUESKY_ZMQ_CURVE_CLIENT_PUBLIC_KEYS": str(certificates["allowed"]),
    }
    with caplog.at_level("ERROR"):
        assert document_plane.start_from_env(env) is None
    assert "without live rows" in caplog.text


# ------------------------------------------------------------- expected points


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        # grid_scan: the product of every axis's own point count.
        ({"detectors": ["bpm"], "axes": [{"num_points": 5}]}, 5),
        (
            {"detectors": ["bpm"], "axes": [{"num_points": 5}, {"num_points": 4}]},
            20,
        ),
        # orm: `num` points per swept corrector; detectors are read, not swept.
        ({"correctors": ["ch1", "ch2"], "detectors": ["bpm1"], "num": 7}, 14),
        # A scalar count with no swept device list is the count itself.
        ({"detectors": ["bpm"], "num_points": 9}, 9),
        # Unrecognized shapes are honestly unknown, never guessed.
        ({"detectors": ["bpm"]}, None),
        ({"correctors": ["ch1"], "steps": ["a", "b"], "num": 3}, None),
        ({"detectors": ["bpm"], "axes": [{"start": 0.0}]}, None),
        ({"detectors": ["bpm"], "num_points": 0}, None),
        # `snake_axes: True` is a bool, not a point count.
        ({"detectors": ["bpm"], "num_points": True}, None),
        (None, None),
    ],
)
def test_expected_points_from_params(params, expected):
    assert expected_points_from_params(params) == expected


def test_progress_is_unknown_for_an_unknown_run():
    assert document_plane.progress("never-heard-of-it") is None


def test_progress_reports_rows_without_a_fraction_when_the_total_is_unknown():
    router = RunDocumentRouter()
    router("start", _start_doc("uid-a", run_id="run-a"))
    router("descriptor", _descriptor_doc("desc-a", "uid-a"))
    router("event", _event_doc("desc-a", 1, {"bpm": 1.0}))

    assert document_plane.progress("run-a") == {
        "rows_seen": 1,
        "expected_points": None,
        "fraction": None,
        "complete": False,
    }


def test_progress_never_exceeds_one():
    document_plane.record_expected_points("run-b", 2)
    router = RunDocumentRouter()
    router("start", _start_doc("uid-b", run_id="run-b"))
    router("descriptor", _descriptor_doc("desc-b", "uid-b"))
    for seq in range(1, 6):
        router("event", _event_doc("desc-b", seq, {"bpm": float(seq)}))

    progress = document_plane.progress("run-b")
    assert progress is not None
    assert progress["rows_seen"] == 5
    assert progress["fraction"] == 1.0
    assert progress["complete"] is False


def test_record_run_params_clears_a_stale_count():
    document_plane.record_expected_points("run-c", 10)
    assert document_plane.record_run_params("run-c", {"detectors": ["bpm"]}) is None
    assert document_plane.get_expected_points("run-c") is None


# ---------------------------------------------------------------- doc routing


def test_router_keeps_interleaved_runs_apart():
    """Events are routed by the document graph, not by arrival order."""
    router = RunDocumentRouter()
    router("start", _start_doc("uid-1", run_id="run-1"))
    router("start", _start_doc("uid-2", run_id="run-2"))
    router("descriptor", _descriptor_doc("desc-1", "uid-1"))
    router("descriptor", _descriptor_doc("desc-2", "uid-2"))

    router("event", _event_doc("desc-2", 1, {"bpm": 2.0}))
    router("event", _event_doc("desc-1", 1, {"bpm": 1.0}))
    router("event", _event_doc("desc-2", 2, {"bpm": 2.5}))
    router("stop", _stop_doc("uid-1", 1))

    first = live_rows.get("run-1")
    second = live_rows.get("run-2")
    assert first is not None and second is not None
    assert first["rows"] == [[1.0]]
    assert first["partial"] is False
    assert second["rows"] == [[2.0], [2.5]]
    assert second["partial"] is True


def test_router_drops_documents_it_cannot_place():
    """An event for an unknown descriptor is ignored, not attributed to a neighbour."""
    router = RunDocumentRouter()
    router("start", _start_doc("uid-1", run_id="run-1"))
    router("descriptor", _descriptor_doc("desc-1", "uid-1"))
    router("event", _event_doc("desc-orphan", 1, {"bpm": 9.0}))
    router("event", _event_doc("desc-1", 1, {"bpm": 1.0}))

    buffer = live_rows.get("run-1")
    assert buffer is not None
    assert buffer["rows"] == [[1.0]]


def test_router_never_raises_on_a_malformed_document():
    router = RunDocumentRouter()
    router("start", {})  # no uid at all
    router("descriptor", {"uid": "d"})  # no run_start
    router("event", {"seq_num": 1})  # no descriptor
    router("stop", {})  # no run_start
    assert live_rows.get("") is None
