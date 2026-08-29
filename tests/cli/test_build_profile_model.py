"""Tests for the port ledger and the worker-band refusal in ``BuildProfile``.

Two rules live here, and they are two halves of one contract.

``_claimed_ports`` is the profile-altitude ledger of every host port a build
spends, keyed by the line an author would edit. Since the framework's ports are
DERIVED from :data:`~osprey.port_layout.LAYOUT` at the deployment's own base
rather than pinned in a template, most of them appear in no profile at all — so
the ledger seeds them itself, for the slots the build actually deploys and at
the indices the roster actually allocates.

``BuildProfile.validate`` then refuses the one fan-out that would walk over
them: on the host network the dispatch workers bind real host ports, and a
``worker_count`` or ``worker_port_stride`` whose last worker lands past the end
of the layout's worker band takes a port the same deployment publishes for a
service of its own. The refusal names that service, because a message that only
said "too many workers" would leave the author to find out which one at deploy
time.

Every expected number is derived from the layout rather than written down: a
test that pinned 10070 would pass a layout that had quietly moved ``tiled``
somewhere else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.cli.build_profile import BuildProfile, _parse_profile
from osprey.errors import BuildProfileError
from osprey.port_layout import DEFAULT_PORT_BASE, WORKER_MAX, default_port

#: The bundled trigger file every dispatch block below points at. The dispatch
#: stanza requires a resolvable one, and which file it is has nothing to do
#: with ports.
_TRIGGERS = "tutorial_triggers.yml"


def _profile(**raw: Any) -> BuildProfile:
    """Parse a minimal profile carrying whatever blocks a test needs."""
    return _parse_profile({"name": "ports", **raw})


def _dispatch_profile(**dispatch: Any) -> BuildProfile:
    """Parse a minimal profile whose ``dispatch:`` block is under test."""
    return _profile(dispatch={"triggers": _TRIGGERS, **dispatch})


def _errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """Validate ``profile`` and return the individual accumulated failures."""
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    header, _, body = str(exc.value).partition(":\n  - ")
    assert header == "Build profile validation failed"
    return body.split("\n  - ")


def _worker_ports(profile: BuildProfile) -> dict[str, int]:
    """Only the worker entries of a profile's ledger."""
    return {
        key: port
        for key, port in profile._claimed_ports().items()
        if key.startswith("dispatch.worker_port_base")
    }


# --- the worker band -------------------------------------------------------


def test_bridge_mode_runs_as_many_workers_as_the_machine_allows(tmp_path: Path) -> None:
    """A bridge-mode worker owns its own namespace, so the band does not bind it.

    Sixty workers is well past the layout's window, and under the default
    topology that is not the layout's business: nothing is published on the
    host, so nothing can collide with the services above the band.
    """
    _dispatch_profile(worker_count=60).validate(tmp_path)


def test_bridge_mode_workers_leave_the_stand_in_and_its_neighbours_alone(
    tmp_path: Path,
) -> None:
    """The big bridge fan-out does not manufacture collisions for the ledger.

    ``live_standin: true`` resolves to the layout's stand-in slot, which sits
    inside the span sixty workers would sweep if the ledger walked them
    regardless of topology. It builds clean instead, alongside the three
    services that span crosses.
    """
    profile = _profile(
        dispatch={"triggers": _TRIGGERS, "worker_count": 60},
        virtual_accelerator={"port": 5064, "live_standin": True},
        bluesky={"port": default_port("bluesky"), "tiled_enabled": True},
        config={"services.openobserve.port": default_port("openobserve")},
    )
    profile.validate(tmp_path)


def test_host_mode_refuses_a_fan_out_that_reaches_the_tiled_slot(tmp_path: Path) -> None:
    """Sixty host-network workers end on the slot the layout gives to tiled."""
    top = default_port("worker", 1) + 59
    reported = _errors(_dispatch_profile(worker_count=60, network="host"), tmp_path)

    assert len(reported) == 1
    assert f"would publish worker 60 on port {top}" in reported[0]
    assert (
        f"past the end of the layout's worker band at {default_port('worker', WORKER_MAX)}"
        in (reported[0])
    )
    assert f"the 'tiled' slot ({default_port('tiled')})" in reported[0]


def test_host_mode_refuses_a_stride_that_reaches_the_qmd_slot(tmp_path: Path) -> None:
    """A widened stride overruns the band with far fewer workers.

    Six workers ten apart end one port above the qmd slot, and the slot named
    is the one whose ground the derived port sits in — the greatest offset at
    or below it, not the nearest.
    """
    top = default_port("worker", 1) + 50
    reported = _errors(
        _dispatch_profile(worker_count=6, worker_port_stride=10, network="host"), tmp_path
    )

    assert len(reported) == 1
    assert f"would publish worker 6 on port {top}" in reported[0]
    assert f"dispatch.worker_port_stride {10}" in reported[0]
    assert f"the 'qmd' slot ({default_port('qmd')})" in reported[0]
    assert top > default_port("qmd")


def test_a_fan_out_that_fills_the_band_exactly_is_built(tmp_path: Path) -> None:
    """The band's last port is inside it: the refusal is past the end, not at it."""
    _dispatch_profile(worker_count=WORKER_MAX, network="host").validate(tmp_path)


def test_workers_moved_off_the_band_are_not_held_to_it(tmp_path: Path) -> None:
    """An absolute ``worker_port_base`` is the layout's documented escape.

    A profile that placed worker 1 outside the band has said the workers do not
    live in the block, so the window is not what bounds them — the host-port
    preflight is. Refusing here would contradict the escape the layout offers.
    """
    _dispatch_profile(worker_count=60, worker_port_base=40000, network="host").validate(tmp_path)


def test_worker_port_stride_must_be_at_least_one(tmp_path: Path) -> None:
    """A stride of zero stacks every worker on one port, so it is refused by name."""
    reported = _errors(
        _dispatch_profile(worker_count=2, worker_port_stride=0, network="host"), tmp_path
    )

    assert reported == [
        "dispatch.worker_port_stride must be >= 1 (got 0): worker i publishes on "
        "worker_port_base + (i - 1) * worker_port_stride, so a stride of 0 puts every "
        "worker on one port and a negative one runs them backwards out of their band. "
        "Leave the key unset for the layout's own spacing of 1."
    ]


# --- what the workers claim ------------------------------------------------


def test_host_mode_worker_claims_land_on_the_strided_ports() -> None:
    """The ledger multiplies by the stride, the way the compose render does.

    Worker 6 of a stride-ten fan-out is fifty ports up, not five — a ledger
    that walked one port per worker would clear a live_standin against ports
    nothing binds and miss the ones something does.
    """
    first = default_port("worker", 1)
    claimed = _worker_ports(
        _dispatch_profile(worker_count=4, worker_port_stride=10, network="host")
    )

    assert claimed == {
        "dispatch.worker_port_base": first,
        "dispatch.worker_port_base + 10": first + 10,
        "dispatch.worker_port_base + 20": first + 20,
        "dispatch.worker_port_base + 30": first + 30,
    }
    # The ports a stride-one walk would have claimed are NOT spent: the band
    # between two workers belongs to whatever the facility widened it for.
    assert not {first + 1, first + 2, first + 3} & set(claimed.values())


def test_bridge_mode_workers_claim_no_host_ports() -> None:
    """Bridge-mode workers publish nothing on the host, so they spend nothing."""
    assert _worker_ports(_dispatch_profile(worker_count=4)) == {}


def test_the_dispatcher_is_claimed_in_either_topology() -> None:
    """Only the WORKER band is topology-gated; the dispatcher's port is not."""
    for network in ("bridge", "host"):
        claimed = _dispatch_profile(network=network)._claimed_ports()
        assert claimed["dispatch.dispatcher_port"] == default_port("dispatcher")


def test_the_second_lane_is_derived_at_the_deployments_own_base() -> None:
    """Lane 2's bridge port is re-checked against the layout at THIS base.

    ``second_lane_port`` takes the resolved base and falls back to the layout's
    own default when it is handed none — so the ledger passes the base it
    resolved rather than letting a deployment that lives at 20000 be described
    in terms of a block it does not occupy.
    """
    base = DEFAULT_PORT_BASE + 10000
    claimed = _profile(
        config={"deployment.port_base": base},
        bluesky={"port": default_port("bluesky", base=base), "second_lane": True},
    )._claimed_ports()

    lane_two = [port for key, port in claimed.items() if "lane 2" in key]
    assert lane_two == [default_port("bluesky_second_lane", base=base)]


# --- the layout seed -------------------------------------------------------


def test_the_ledger_carries_the_layout_slots_the_build_deploys() -> None:
    """A profile that spells no ports still spends the framework's own.

    The app template renders these blocks and derives their ports from the
    layout, so the numbers appear nowhere in the profile — and a ledger that
    only read the profile would clear a colliding port as free.
    """
    claimed = _profile()._claimed_ports()

    assert claimed["services.openobserve.port"] == default_port("openobserve")
    assert claimed["services.qmd.port"] == default_port("qmd")
    assert claimed["services.postgresql.port_host"] == default_port("postgres")
    assert claimed["services.graphdb.port_host"] == default_port("graphdb_bolt")
    assert claimed["services.graphdb.http_port_host"] == default_port("graphdb_http")


def test_the_seed_follows_the_profiles_own_port_base() -> None:
    """A deployment on a second base is described in terms of that base.

    This is the layout's one rule at the ledger's altitude: never the module's
    default base when the profile resolved one of its own.
    """
    base = DEFAULT_PORT_BASE + 10000
    claimed = _profile(config={"deployment.port_base": base})._claimed_ports()

    assert claimed["services.qmd.port"] == default_port("qmd", base=base)
    assert claimed["services.qmd.port"] != default_port("qmd")


def test_a_port_the_profile_moves_keeps_the_profiles_own_number() -> None:
    """The seed fills gaps; it never overwrites what an author wrote."""
    moved = default_port("qmd") + 4321
    claimed = _profile(config={"services.qmd.port": moved})._claimed_ports()

    assert claimed["services.qmd.port"] == moved


def test_a_slot_the_profile_removes_is_not_seeded() -> None:
    """The bare removal spelling takes the service — and its port — away."""
    claimed = _profile(config={"services.graphdb": None})._claimed_ports()

    assert "services.graphdb.port_host" not in claimed
    assert "services.graphdb.http_port_host" not in claimed


def test_an_attached_project_seeds_no_service_slots() -> None:
    """An attached project scaffolds no stack, so it publishes none of its ports."""
    claimed = _profile(deploy_services=False)._claimed_ports()

    assert not [key for key in claimed if key.startswith("services.")]


def test_the_stand_ins_own_slot_is_never_seeded(tmp_path: Path) -> None:
    """Seeding ``va_standin`` would make ``live_standin: true`` collide with itself.

    The stand-in's port IS that slot, and the ledger exists to be checked
    against it, so the one slot the seed must leave empty is that one.
    """
    profile = _profile(virtual_accelerator={"port": 5064, "live_standin": True})

    assert default_port("va_standin") not in profile._claimed_ports().values()
    profile.validate(tmp_path)


# --- the web stack ---------------------------------------------------------


def test_the_web_stack_is_seeded_at_the_roster_indices_only() -> None:
    """One port per family per USER, not the hundred each band could hold.

    A bare roster entry takes its list position and an object entry takes its
    own index, the same rule the render normalises by — so the ledger describes
    the ports this deployment binds and not the span they were cut from.
    """
    claimed = _profile(
        config={
            "modules.web_terminals": {
                "enabled": True,
                "users": ["alice", {"name": "bob", "index": 3}],
            }
        }
    )._claimed_ports()

    assert claimed["modules.web_terminals.nginx_port"] == default_port("nginx")
    assert claimed["modules.web_terminals.web_base_port"] == default_port("web", 0)
    assert claimed["modules.web_terminals.web_base_port + 3"] == default_port("web", 3)
    assert claimed["modules.web_terminals.okf_base_port + 3"] == default_port("okf", 3)
    assert "modules.web_terminals.web_base_port + 1" not in claimed
    assert "modules.web_terminals.web_base_port + 99" not in claimed


def test_the_auth_sidecars_port_is_claimed_only_when_it_is_deployed() -> None:
    """``token`` and ``none`` put no sidecar in front of the terminals.

    The gateway's second port is spent by the auth container, and under the
    default magic-link posture there is no auth container — so claiming its slot
    would refuse a stand-in over a port nothing binds.
    """

    def claimed(method: str | None) -> dict[str, int]:
        auth = {"method": method} if method else {}
        return _profile(
            config={"modules.web_terminals": {"enabled": True, "auth": auth, "users": ["alice"]}}
        )._claimed_ports()

    key = "modules.web_terminals.auth.port"
    assert key not in claimed(None)
    assert key not in claimed("token")
    assert key not in claimed("none")
    assert claimed("password")[key] == default_port("auth")
    assert claimed("oidc")[key] == default_port("auth")


def test_a_web_terminal_port_override_wins_over_the_layout() -> None:
    """An override is an absolute port, and the ledger records where it lands."""
    moved = default_port("nginx") + 5000
    claimed = _profile(
        config={
            "modules.web_terminals": {
                "enabled": True,
                "nginx_port": moved,
                "web_base_port": moved + 100,
                "users": ["alice"],
            }
        }
    )._claimed_ports()

    assert claimed["modules.web_terminals.nginx_port"] == moved
    assert claimed["modules.web_terminals.web_base_port"] == moved + 100


def test_a_profile_with_no_web_stack_seeds_no_web_ports() -> None:
    """Nothing to publish, nothing to claim."""
    claimed = _profile()._claimed_ports()

    assert not [key for key in claimed if key.startswith("modules.web_terminals")]
