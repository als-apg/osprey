"""Read-time attribution: each entry's person comes only from trusted sources.

A terminal, OTLP login or host entry belongs to its source identity; the
sidecar's ``on_behalf_of`` on any surface and a queueserver's on ``preflight``
only name a person; a dispatch session belongs to the requester of the first
``run:<run_id>:accept`` entry naming it, whichever landed first. Anything an
entry claims about itself (body fields, attributes, an untrusted
``on_behalf_of``) is ignored, and nothing is stored.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from osprey.agent_record import attribution as at
from osprey.agent_record import contract as c
from osprey.agent_record import identities as ids
from tests.agent_record.test_identities import _config

T0 = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


def _entry(
    identity: str,
    *,
    kind: str = c.KIND_ANSWER,
    key: str | None = None,
    body: dict | None = None,
    on_behalf_of: str = "",
    session_id: str = "",
    run_id: str = "",
    arrived: int | None = 0,
    ts: int = 0,
) -> c.Entry:
    return c.Entry(
        key=key or f"sp:{identity}:e{arrived}-{ts}-{session_id or 'x'}",
        kind=kind,
        ts=T0 + timedelta(seconds=ts),
        identity=identity,
        body=body or {},
        on_behalf_of=on_behalf_of,
        session_id=session_id,
        run_id=run_id,
        inserted_at=None if arrived is None else T0 + timedelta(seconds=arrived),
    )


def _accept(
    identity: str,
    run_id: str,
    session_id: str,
    requester: str | None,
    *,
    arrived: int = 0,
) -> c.Entry:
    body: dict = {"phase": c.RUN_ACCEPT, "run_id": run_id, "session_id": session_id}
    if requester is not None:
        body["requester"] = requester
    return _entry(
        identity,
        kind=c.KIND_RUN,
        key=c.run_key(run_id, c.RUN_ACCEPT),
        body=body,
        session_id=session_id,
        run_id=run_id,
        arrived=arrived,
    )


def _terminal(identity: str, run_id: str, session_id: str = "", *, arrived: int = 5) -> c.Entry:
    return _entry(
        identity,
        kind=c.KIND_RUN,
        key=c.run_key(run_id, c.RUN_TERMINAL),
        body={"phase": c.RUN_TERMINAL, "status": "done", "requester": "mallory"},
        session_id=session_id,
        run_id=run_id,
        arrived=arrived,
    )


# --- source identities -------------------------------------------------------------


def test_terminal_entries_belong_to_their_source_identity():
    attribution = at.Attribution.build(_config(), [])
    assert attribution.person(_entry("alice")) == "alice"
    assert attribution.person(_entry("bob", kind=c.KIND_USAGE)) == "bob"


def test_otlp_login_entries_belong_to_the_login():
    config = _config()
    identity = ids.login_source_identity(config, "alice")
    entry = _entry(identity, kind=c.KIND_DECISION, key=f"otel:{identity}:s1:tool_decision:t1")
    assert at.Attribution.build(config, []).person(entry) == "alice"


def test_host_mount_entries_belong_to_the_host_identity():
    config = _config(host="builder")
    identity = ids.transcript_source_identity(config, ids.HOST_TRANSCRIPT_ROOT)
    assert at.Attribution.build(config, []).person(_entry(identity)) == "builder"


def test_no_host_identity_means_a_host_named_entry_is_no_person():
    # "builder" is not a terminal and no host identity is configured.
    assert at.Attribution.build(_config(host=None), []).person(_entry("builder")) is None


@pytest.mark.parametrize(
    "identity",
    ["teams-bridge", "history", "operator", "bluesky-web", "queueserver", "sidecar"],
)
def test_service_entries_without_a_trusted_on_behalf_have_no_person(identity):
    config = _config(bridges=("teams_bridge",), lanes=True)
    assert at.Attribution.build(config, []).person(_entry(identity, kind=c.KIND_AUDIT)) is None


# --- spoofed claims ----------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"identity": "bob"},
        {"person": "bob", "user": "bob", "requester": "bob"},
        {"attributes": {"user.id": "bob", "osprey.identity": "bob", "user.email": "bob@x"}},
        {"actor": "bob", "on_behalf_of": "bob", "surface": "preflight"},
    ],
)
def test_a_spoofed_body_or_attribute_identity_is_ignored(body):
    attribution = at.Attribution.build(_config(), [])
    assert attribution.person(_entry("alice", body=body)) == "alice"
    assert attribution.person(_entry("teams-bridge", kind=c.KIND_AUDIT, body=body)) is None


def test_a_spoofed_on_behalf_of_from_an_untrusted_sender_is_ignored():
    attribution = at.Attribution.build(_config(lanes=True, bridges=("teams_bridge",)), [])
    # A terminal that names someone else stays its own person.
    assert attribution.person(_entry("alice", kind=c.KIND_AUDIT, on_behalf_of="bob")) == "alice"
    # A bridge or any other service is never trusted.
    for sender in ("teams-bridge", "bluesky-web", "history"):
        spoof = _entry(sender, kind=c.KIND_AUDIT, on_behalf_of="bob", body={"surface": "preflight"})
        assert attribution.person(spoof) is None
    # A queueserver off the preflight surface is not trusted.
    off = _entry("queueserver", kind=c.KIND_AUDIT, on_behalf_of="bob", body={"surface": "http"})
    assert attribution.person(off) is None
    bare = _entry("queueserver", kind=c.KIND_AUDIT, on_behalf_of="bob")
    assert attribution.person(bare) is None


def test_a_dispatch_worker_on_behalf_of_is_ignored():
    config = _config(workers=1)
    attribution = at.Attribution.build(config, [])
    spoof = _entry("dispatch-worker-1", kind=c.KIND_AUDIT, on_behalf_of="bob", session_id="s1")
    assert attribution.person(spoof) is None


def test_the_sidecar_on_behalf_of_is_trusted_on_any_surface():
    attribution = at.Attribution.build(_config(), [])
    for surface in ("login", "http", None):
        body = {} if surface is None else {"surface": surface}
        row = _entry("sidecar", kind=c.KIND_AUDIT, on_behalf_of="alice", body=body)
        assert attribution.person(row) == "alice"


def test_a_queueserver_on_behalf_of_is_trusted_on_preflight_only():
    attribution = at.Attribution.build(_config(lanes=True), [])
    row = _entry(
        "queueserver", kind=c.KIND_AUDIT, on_behalf_of="bob", body={"surface": "preflight"}
    )
    assert attribution.person(row) == "bob"


def test_a_queueserver_not_rendered_is_not_trusted():
    attribution = at.Attribution.build(_config(lanes=False), [])
    row = _entry(
        "queueserver", kind=c.KIND_AUDIT, on_behalf_of="bob", body={"surface": "preflight"}
    )
    assert attribution.person(row) is None


# --- dispatch sessions ---------------------------------------------------------------


def test_a_dispatch_session_belongs_to_its_requester():
    config = _config(workers=2)
    accept = _accept("dispatch-worker-1", "r1", "s1", "alice")
    attribution = at.Attribution.build(config, [accept])
    assert attribution.person(_entry("dispatch-worker-1", session_id="s1")) == "alice"
    assert attribution.person(accept) == "alice"
    # The run's terminal entry belongs to the same person; its body is ignored.
    assert attribution.person(_terminal("dispatch-worker-1", "r1")) == "alice"
    # An entry naming only the run joins through it.
    assert attribution.person(_entry("dispatch-worker-1", run_id="r1")) == "alice"
    # Another session, or the same session id under another worker, is no one's.
    assert attribution.person(_entry("dispatch-worker-1", session_id="s2")) is None
    assert attribution.person(_entry("dispatch-worker-2", session_id="s1")) is None
    assert attribution.person(_entry("dispatch-worker-1")) is None


def test_shared_mode_dispatch_workers_resolve_through_the_identity_map():
    config = _config(workers=3, shared=True)
    shared = ids.SHARED_DISPATCH_IDENTITY
    # The API's accept line was found in worker 2's spool directory; the
    # session's transcript is on the shared volume.
    accept = _accept(ids.spool_source_identity(config, "dispatch-worker-2"), "r1", "s1", "alice")
    transcript = _entry(
        ids.transcript_source_identity(config, ids.DISPATCH_TRANSCRIPT_VOLUME), session_id="s1"
    )
    attribution = at.Attribution.build(config, [accept])
    assert accept.identity == transcript.identity == shared
    assert attribution.person(transcript) == "alice"
    # An entry still named by a worker's own name maps to the shared identity.
    accept_raw = _accept("dispatch-worker-3", "r2", "s2", "bob")
    attribution = at.Attribution.build(config, [accept_raw])
    assert attribution.person(_entry(shared, session_id="s2")) == "bob"
    assert attribution.person(_entry("dispatch-worker-1", session_id="s2")) == "bob"


def test_isolated_workers_do_not_share_requesters():
    config = _config(workers=2)
    attribution = at.Attribution.build(config, [_accept("dispatch-worker-2", "r1", "s1", "alice")])
    assert attribution.person(_entry("dispatch-worker-1", session_id="s1")) is None
    assert attribution.person(_entry("dispatch-worker-2", session_id="s1")) == "alice"


def test_a_run_entry_that_arrives_after_its_session_entries_is_handled():
    config = _config(workers=1)
    early = _entry("dispatch-worker-1", session_id="s1", arrived=0)
    # Read before the accept entry landed: no person yet, and nothing stored.
    assert at.Attribution.build(config, []).person(early) is None
    late_accept = _accept("dispatch-worker-1", "r1", "s1", "alice", arrived=60)
    assert at.Attribution.build(config, [late_accept]).person(early) == "alice"
    assert early.inserted_at == T0  # the entry itself is unchanged


def test_the_first_accept_naming_a_session_wins():
    config = _config(workers=1)
    first = _accept("dispatch-worker-1", "r1", "s1", "alice", arrived=1)
    later = _accept("dispatch-worker-1", "r2", "s1", "mallory", arrived=2)
    for order in ([first, later], [later, first]):
        attribution = at.Attribution.build(config, order)
        assert attribution.person(_entry("dispatch-worker-1", session_id="s1")) == "alice"
        # The refused later run names no one.
        assert attribution.person(later) is None
        assert attribution.person(_terminal("dispatch-worker-1", "r2")) is None


def test_a_cron_run_has_no_person():
    config = _config(workers=1)
    for requester in (None, ""):
        cron = _accept("dispatch-worker-1", "r1", "s1", requester)
        attribution = at.Attribution.build(config, [cron])
        assert attribution.person(cron) is None
        assert attribution.person(_entry("dispatch-worker-1", session_id="s1")) is None
        assert attribution.person(_terminal("dispatch-worker-1", "r1", "s1")) is None


def test_a_cron_first_accept_is_not_overridden_by_a_later_requester():
    config = _config(workers=1)
    cron = _accept("dispatch-worker-1", "r1", "s1", None, arrived=1)
    later = _accept("dispatch-worker-1", "r2", "s1", "mallory", arrived=2)
    attribution = at.Attribution.build(config, [later, cron])
    assert attribution.person(_entry("dispatch-worker-1", session_id="s1")) is None


def test_run_entries_from_a_non_dispatch_source_name_no_one():
    config = _config(workers=1)
    forged = _accept("alice", "r1", "s1", "bob")
    attribution = at.Attribution.build(config, [forged])
    assert attribution.person(_entry("dispatch-worker-1", session_id="s1")) is None
    # alice's own entries stay hers.
    assert attribution.person(forged) == "alice"


def test_a_session_id_only_in_the_accept_body_still_joins():
    config = _config(workers=1)
    accept = c.Entry(
        key=c.run_key("r1", c.RUN_ACCEPT),
        kind=c.KIND_RUN,
        ts=T0,
        identity="dispatch-worker-1",
        body={"phase": "accept", "run_id": "r1", "session_id": "s1", "requester": "alice"},
        run_id="r1",
        inserted_at=T0,
    )
    attribution = at.Attribution.build(config, [accept])
    assert attribution.person(_entry("dispatch-worker-1", session_id="s1")) == "alice"


def test_build_ignores_entries_that_are_not_run_accepts():
    config = _config(workers=1)
    noise = [
        _entry("dispatch-worker-1", session_id="s1", body={"requester": "mallory"}),
        _terminal("dispatch-worker-1", "r1", "s1"),
    ]
    attribution = at.Attribution.build(config, noise)
    assert attribution.person(_entry("dispatch-worker-1", session_id="s1")) is None


def test_accepts_without_arrival_order_by_ts():
    config = _config(workers=1)
    a = _accept("dispatch-worker-1", "r1", "s1", "alice")
    b = _accept("dispatch-worker-1", "r2", "s1", "mallory")
    a = c.Entry(**{**a.__dict__, "inserted_at": None, "ts": T0})
    b = c.Entry(**{**b.__dict__, "inserted_at": None, "ts": T0 + timedelta(seconds=1)})
    attribution = at.Attribution.build(config, [b, a])
    assert attribution.person(_entry("dispatch-worker-1", session_id="s1")) == "alice"


def test_is_run_accept():
    assert at.is_run_accept(_accept("dispatch-worker-1", "r1", "s1", "alice"))
    assert not at.is_run_accept(_terminal("dispatch-worker-1", "r1"))
    assert not at.is_run_accept(_entry("dispatch-worker-1", body={"phase": "accept"}))


def test_attribution_pairs_every_entry_with_its_person():
    config = _config(workers=1)
    entries = [
        _entry("alice"),
        _accept("dispatch-worker-1", "r1", "s1", "bob"),
        _entry("dispatch-worker-1", session_id="s1"),
        _entry("teams-bridge", kind=c.KIND_AUDIT),
    ]
    attribution = at.Attribution.build(config, entries)
    people = [attribution.person(entry) for entry in entries]
    assert people == ["alice", "bob", "bob", None]
