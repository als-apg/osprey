"""Tests for the ``virtual_accelerator.live_standin:`` refusals in ``BuildProfile.validate``.

The stand-in is a SECOND soft-IOC standing up a THIRD control target, so it can
be wrong in ways the baseline soft-IOC cannot: it can land on a port some other
block already spends, it can land on the very gateway the simulation is dialed
through, and it can be named as a deployment's baseline without being built.

What it can no longer be is refused for standing beside a facility's own
machine: ``live`` keeps meaning the authored ``epics`` block, so an ``epics``
baseline with a stand-in is the ordinary shape and is pinned here as one.

Each refusal is pinned by its exact message, because the message is the whole
deliverable — a refusal an author cannot act on is a build that fails twice.
The suite also pins the accumulation contract the rest of ``validate`` keeps:
several stand-in faults arrive in ONE
:class:`~osprey_connectors.errors.BuildProfileError`, never one rebuild per typo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.cli.build_profile import BuildProfile, _parse_profile
from tests.cli._profiles import validation_errors

pytestmark = pytest.mark.usefixtures("facility_data_tree")


def _standin_profile(live_standin: int = 5074, **extra: Any) -> BuildProfile:
    """A minimal profile whose VA block asks for a live stand-in."""
    raw: dict[str, Any] = {
        "name": "standin",
        "data": "data",
        "virtual_accelerator": {"port": 5064, "live_standin": live_standin},
        **extra,
    }
    return _parse_profile(raw)


# --- the port itself -------------------------------------------------------


@pytest.mark.parametrize(
    ("port", "message"),
    [
        # The stand-in's Channel Access port must be a usable TCP port.
        (70000, "virtual_accelerator.live_standin must be in 1..65535 (got 70000)"),
        # Sandbox and stand-in are two containers; one port cannot serve both.
        (
            5064,
            "virtual_accelerator.live_standin must differ from virtual_accelerator.port (both 5064)",
        ),
    ],
    ids=["out-of-range", "baseline-ioc-port"],
)
def test_live_standin_validate_rejects_an_unusable_port(
    tmp_path: Path, port: int, message: str
) -> None:
    assert validation_errors(_standin_profile(port), tmp_path) == [message]


def test_live_standin_validate_rejects_a_port_another_block_claims(tmp_path: Path) -> None:
    """A collision names the block key that has to move, not just the number."""
    profile = _standin_profile(5074, bluesky={"port": 5074})
    assert validation_errors(profile, tmp_path) == [
        "virtual_accelerator.live_standin (5074) collides with bluesky.port (5074)"
    ]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        # The ``config:`` block is a port-authoring surface too, and is swept the same.
        (
            {"services.postgresql.port_host": 5074},
            "virtual_accelerator.live_standin (5074) collides with services.postgresql.port_host (5074)",
        ),
        # A hand-authored gateway port equal to the stand-in's is one endpoint for two targets.
        (
            {"control_system.connector.virtual_accelerator.gateways.write_access.port": 5074},
            "virtual_accelerator.live_standin (5074) collides with the profile's `config:` "
            "control_system.connector.virtual_accelerator.gateways.write_access.port (5074) — "
            "the virtual accelerator and its live stand-in are two endpoints, never one",
        ),
    ],
    ids=["config-service", "dotted-gateway"],
)
def test_live_standin_validate_rejects_a_port_the_config_block_claims(
    tmp_path: Path, config: dict[str, int], message: str
) -> None:
    profile = _standin_profile(5074, config=config)
    assert validation_errors(profile, tmp_path) == [message]


# --- the VA gateways the simulation is dialed through ----------------------


def test_live_standin_validate_reads_a_nested_gateway_port_the_same(tmp_path: Path) -> None:
    """Same refusal through the nested spelling — either one reaches the same leaf."""
    profile = _standin_profile(
        5074,
        config={
            "control_system": {
                "connector": {"virtual_accelerator": {"gateways": {"read_only": {"port": 5074}}}}
            }
        },
    )
    assert validation_errors(profile, tmp_path) == [
        "virtual_accelerator.live_standin (5074) collides with the profile's `config:` "
        "control_system.connector.virtual_accelerator.gateways.read_only.port (5074) — "
        "the virtual accelerator and its live stand-in are two endpoints, never one"
    ]


# --- the third target, beside the facility's own machine -------------------


def test_live_standin_validate_accepts_an_epics_baseline(tmp_path: Path) -> None:
    """A deployment on its own machine may stand the rehearsal up beside it.

    The refusal this replaces read the stand-in as a claim on ``live``. It is
    not one: ``live`` is the facility's authored ``epics`` block before and
    after, and the stand-in is the third target beside it.
    """
    _standin_profile(5074, config={"control_system.type": "epics"}).validate(tmp_path)


def test_live_standin_validate_refuses_a_standin_baseline_with_no_standin(
    tmp_path: Path,
) -> None:
    """A baseline naming the stand-in on a build that stands none up."""
    profile = _parse_profile(
        {
            "name": "standin",
            "data": "data",
            "virtual_accelerator": {"port": 5064},
            "config": {
                "control_system.type": "live_standin",
                # An honest pairing, so the one thing reported is the missing
                # stand-in: a baseline the deployment stands up for itself with
                # the synthesizing archiver is its own refusal.
                "archiver.type": "mongodb_archiver",
            },
        }
    )
    assert validation_errors(profile, tmp_path) == [
        "control_system.type: live_standin with no virtual_accelerator.live_standin — "
        "the baseline names a machine this deployment does not stand up, so every "
        "session would dial a port nothing serves. Set "
        "`virtual_accelerator.live_standin` to the port the stand-in should serve, or "
        "set `control_system.type` back to the connector that reaches this machine "
        "(`epics` for a facility's own)."
    ]


def test_live_standin_validate_refuses_a_standin_baseline_with_no_va_block(
    tmp_path: Path,
) -> None:
    """No ``virtual_accelerator:`` block at all is the same fault, reached differently."""
    profile = _parse_profile(
        {
            "name": "standin",
            "data": "data",
            "config": {
                "control_system.type": "live_standin",
                "archiver.type": "mongodb_archiver",
            },
        }
    )
    reported = validation_errors(profile, tmp_path)
    assert len(reported) == 1
    assert reported[0].startswith("control_system.type: live_standin with no ")


def test_live_standin_validate_accepts_a_standin_baseline_that_builds_one(
    tmp_path: Path,
) -> None:
    """Baselined on the stand-in, with the stand-in built: the shape the key is for."""
    _standin_profile(
        5074,
        config={
            "control_system.type": "live_standin",
            "archiver.type": "mongodb_archiver",
        },
    ).validate(tmp_path)


# --- the archive is the virtual accelerator's, whatever the baseline --------


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"control_system.type": "epics"}, id="epics"),
        pytest.param(
            {"control_system.type": "virtual_accelerator", "archiver.type": "mongodb_archiver"},
            id="virtual_accelerator",
        ),
        pytest.param(
            {"control_system.type": "live_standin", "archiver.type": "mongodb_archiver"},
            id="live_standin",
        ),
    ],
)
def test_live_standin_validate_accepts_a_recorded_archive_on_any_baseline(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    """A stand-in beside a ``va_archiver`` block is no fault on any baseline.

    The store the block stands up is the virtual accelerator's — the recorder
    samples the simulator, never the stand-in — and the stand-in is shown the
    facility's own archive. Whether the store is honest to serve is the
    archiver pairing's question (hence ``archiver.type`` on the simulated
    baselines), asked of the baseline with or without a stand-in; the stand-in
    adds nothing to it.
    """
    _standin_profile(5074, config=config, va_archiver={}).validate(tmp_path)


# --- the lattice -----------------------------------------------------------


def test_live_standin_validate_accepts_a_fault_set_with_no_lattice(
    tmp_path: Path,
) -> None:
    """A stand-in serving no lattice is a legal stand-in, whatever its chain sets.

    The stand-in carries no readout errors of its own, so nothing it serves
    depends on a lattice being there to displace.
    """
    (tmp_path / ".env").write_text("VA_LATTICE=none\nVA_STANDIN_BPM_ERRORS=BPM01:offset_x=1e-4\n")
    _standin_profile(5074).validate(tmp_path)


# --- accumulation, and the clean case --------------------------------------


def test_live_standin_validate_reports_every_violation_in_one_error(tmp_path: Path) -> None:
    """Two unrelated stand-in faults arrive in one raise, not one rebuild each."""
    profile = _standin_profile(
        5074,
        config={
            "control_system.type": "epics",
            "services.postgresql.port_host": 5074,
            "control_system.connector.virtual_accelerator.gateways.read_only.port": 5074,
        },
    )
    reported = validation_errors(profile, tmp_path)
    assert len(reported) == 2
    assert reported[0] == (
        "virtual_accelerator.live_standin (5074) collides with services.postgresql.port_host (5074)"
    )
    assert reported[1].startswith(
        "virtual_accelerator.live_standin (5074) collides with the profile's `config:` "
        "control_system.connector.virtual_accelerator.gateways.read_only.port (5074)"
    )


def test_live_standin_validate_accepts_a_clean_profile(tmp_path: Path) -> None:
    """The shipped shape — a stand-in on its own port, nothing else claiming it."""
    _standin_profile(5074).validate(tmp_path)


def test_live_standin_validate_leaves_a_profile_without_the_key_alone(tmp_path: Path) -> None:
    """Absent means no stand-in, and none of these rules have anything to say."""
    _parse_profile({"name": "x", "data": "data", "virtual_accelerator": {"port": 5064}}).validate(
        tmp_path
    )
