"""Tests for the ``virtual_accelerator.live_standin:`` refusals in ``BuildProfile.validate``.

The stand-in is a SECOND soft-IOC standing up a THIRD control target, so it can
be wrong in ways the baseline soft-IOC cannot: it can land on a port some other
block already spends, it can land on the very gateway the simulation is dialed
through, it can be named as a deployment's baseline without being built, it can
record a store that would be read as the real machine's past, and it can be
built on a tree with no lattice behind the readout perturbation it ships.

What it can no longer be is refused for standing beside a facility's own
machine: ``live`` keeps meaning the authored ``epics`` block, so an ``epics``
baseline with a stand-in is the ordinary shape and is pinned here as one.

Each refusal is pinned by its exact message, because the message is the whole
deliverable — a refusal an author cannot act on is a build that fails twice.
The suite also pins the accumulation contract the rest of ``validate`` keeps:
several stand-in faults arrive in ONE
:class:`~osprey.errors.BuildProfileError`, never one rebuild per typo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.cli.build_profile import BuildProfile, _parse_profile
from osprey.cli.build_profile_va_faults import shipped_bpm_errors_field_errors
from osprey.errors import BuildProfileError


def _errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """Validate ``profile`` and return the individual accumulated failures."""
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    header, _, body = str(exc.value).partition(":\n  - ")
    assert header == "Build profile validation failed"
    return body.split("\n  - ")


def _standin_profile(live_standin: int = 5074, **extra: Any) -> BuildProfile:
    """A minimal profile whose VA block asks for a live stand-in."""
    raw: dict[str, Any] = {
        "name": "standin",
        "virtual_accelerator": {"port": 5064, "live_standin": live_standin},
        **extra,
    }
    return _parse_profile(raw)


# --- the port itself -------------------------------------------------------


def test_live_standin_validate_rejects_a_port_out_of_range(tmp_path: Path) -> None:
    """The stand-in's Channel Access port must be a usable TCP port."""
    assert _errors(_standin_profile(70000), tmp_path) == [
        "virtual_accelerator.live_standin must be in 1..65535 (got 70000)"
    ]


def test_live_standin_validate_rejects_the_baseline_ioc_port(tmp_path: Path) -> None:
    """Sandbox and stand-in are two containers; one port cannot serve both."""
    assert _errors(_standin_profile(5064), tmp_path) == [
        "virtual_accelerator.live_standin must differ from virtual_accelerator.port (both 5064)"
    ]


def test_live_standin_validate_rejects_a_port_another_block_claims(tmp_path: Path) -> None:
    """A collision names the block key that has to move, not just the number."""
    profile = _standin_profile(5074, bluesky={"port": 5074})
    assert _errors(profile, tmp_path) == [
        "virtual_accelerator.live_standin (5074) collides with bluesky.port (5074)"
    ]


def test_live_standin_validate_rejects_a_port_a_config_service_claims(tmp_path: Path) -> None:
    """The ``config:`` block is a port-authoring surface too, and is swept the same."""
    profile = _standin_profile(5074, config={"services.postgresql.port_host": 5074})
    assert _errors(profile, tmp_path) == [
        "virtual_accelerator.live_standin (5074) collides with services.postgresql.port_host (5074)"
    ]


# --- the VA gateways the simulation is dialed through ----------------------


def test_live_standin_validate_rejects_a_dotted_gateway_port(tmp_path: Path) -> None:
    """A hand-authored gateway port equal to the stand-in's is one endpoint for two targets."""
    profile = _standin_profile(
        5074,
        config={"control_system.connector.virtual_accelerator.gateways.write_access.port": 5074},
    )
    assert _errors(profile, tmp_path) == [
        "virtual_accelerator.live_standin (5074) collides with the profile's `config:` "
        "control_system.connector.virtual_accelerator.gateways.write_access.port (5074) — "
        "the virtual accelerator and its live stand-in are two endpoints, never one"
    ]


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
    assert _errors(profile, tmp_path) == [
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
    assert _errors(profile, tmp_path) == [
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
            "config": {
                "control_system.type": "live_standin",
                "archiver.type": "mongodb_archiver",
            },
        }
    )
    reported = _errors(profile, tmp_path)
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


# --- the archive belongs to the machine it records -------------------------


def test_live_standin_validate_refuses_a_recorded_archive_on_a_live_baseline(
    tmp_path: Path,
) -> None:
    """A store recorded off the stand-in, served as the real machine's past."""
    profile = _standin_profile(5074, config={"control_system.type": "epics"}, va_archiver={})
    assert _errors(profile, tmp_path) == [
        "virtual_accelerator.live_standin with a va_archiver block on a "
        "control_system.type: epics baseline — the archive belongs to the machine it "
        "records. The recorder writes THIS deployment's store from the stand-in, and "
        "on a baseline naming the facility's own machine that store is served as the "
        "real machine's past. Either delete `va_archiver` and let the deployment read "
        "the facility's own archiver, or baseline this deployment on the machine "
        "being recorded (`control_system.type: live_standin`, or "
        "`virtual_accelerator` for the simulation)."
    ]


def test_live_standin_validate_accepts_a_recorded_archive_on_a_va_baseline(
    tmp_path: Path,
) -> None:
    """A simulated baseline records its own machine, which is what the store holds."""
    profile = _standin_profile(
        5074,
        config={
            "control_system.type": "virtual_accelerator",
            "archiver.type": "mongodb_archiver",
        },
        va_archiver={},
    )
    profile.validate(tmp_path)


def test_live_standin_validate_accepts_a_recorded_archive_on_a_standin_baseline(
    tmp_path: Path,
) -> None:
    """The stand-in as its own baseline is the other machine a deployment stands up."""
    profile = _standin_profile(
        5074,
        config={
            "control_system.type": "live_standin",
            "archiver.type": "mongodb_archiver",
        },
        va_archiver={},
    )
    profile.validate(tmp_path)


def test_live_standin_validate_leaves_an_archive_without_a_standin_alone(
    tmp_path: Path,
) -> None:
    """No stand-in, no rule: the store records whatever the baseline names."""
    profile = _parse_profile(
        {
            "name": "standin",
            "virtual_accelerator": {"port": 5064},
            "config": {"control_system.type": "epics"},
            "va_archiver": {},
        }
    )
    profile.validate(tmp_path)


# --- the lattice the shipped perturbation needs ---------------------------


def test_live_standin_validate_accepts_a_lattice_pinned_off_on_its_own(
    tmp_path: Path,
) -> None:
    """``VA_LATTICE=none`` alone is a stand-in without faults, not a broken one.

    The shipped perturbation is the builtin lattice's — it names offsets on a
    PyAT model — so a deployment that pinned the lattice off and asked for
    nothing else gets the empty fault set from the render and has nothing to
    refuse. A facility on a file-backed channel set can still rehearse.
    """
    (tmp_path / ".env").write_text("VA_LATTICE=none\n")
    _standin_profile(5074).validate(tmp_path)


def test_live_standin_validate_refuses_an_authored_fault_set_with_no_lattice(
    tmp_path: Path,
) -> None:
    """The one shape that cannot boot: faults asked for, no model to apply them to."""
    (tmp_path / ".env").write_text("VA_LATTICE=none\nVA_STANDIN_BPM_ERRORS=BPM01:offset_x=1e-4\n")
    assert _errors(_standin_profile(5074), tmp_path) == [
        "virtual_accelerator.live_standin ships a readout perturbation, but this "
        "deployment's env chain resolves VA_LATTICE='none'. There is no PyAT model to "
        "displace, so the stand-in's IOC exits at boot rather than serving a machine "
        "that ignores the faults it was configured with. Set VA_LATTICE=builtin, or "
        "turn the perturbation off with VA_STANDIN_BPM_ERRORS= (empty)."
    ]


def test_live_standin_validate_accepts_no_perturbation_with_no_lattice(
    tmp_path: Path,
) -> None:
    """Turning the fault set off is the second way out, and the message names it.

    An explicit empty value is honored as an empty fault set rather than
    rounded back up to the shipped default, which is what makes the refusal's
    advertised fix a real one.
    """
    (tmp_path / ".env").write_text("VA_LATTICE=none\nVA_STANDIN_BPM_ERRORS=\n")
    _standin_profile(5074).validate(tmp_path)


def test_live_standin_validate_reads_the_whole_chain_for_the_lattice(
    tmp_path: Path,
) -> None:
    """``.env`` wins over ``.env.shared``, the precedence the chain defines."""
    (tmp_path / ".env.shared").write_text("VA_LATTICE=none\n")
    (tmp_path / ".env").write_text("VA_LATTICE=builtin\n")
    _standin_profile(5074).validate(tmp_path)


def test_live_standin_validate_accepts_a_lattice_pinned_to_builtin(tmp_path: Path) -> None:
    """The pin the build would have written itself is not a fault."""
    (tmp_path / ".env").write_text("VA_LATTICE=builtin\n")
    _standin_profile(5074).validate(tmp_path)


# --- accumulation, and the clean case --------------------------------------


def test_live_standin_validate_reports_every_violation_in_one_error(tmp_path: Path) -> None:
    """Two unrelated stand-in faults arrive in one raise, not one rebuild each."""
    profile = _standin_profile(
        5074,
        config={"control_system.type": "epics", "services.postgresql.port_host": 5074},
        va_archiver={},
    )
    reported = _errors(profile, tmp_path)
    assert len(reported) == 2
    assert reported[0].startswith("virtual_accelerator.live_standin (5074) collides with")
    assert reported[1].startswith("virtual_accelerator.live_standin with a va_archiver block on a")


def test_live_standin_validate_accepts_a_clean_profile(tmp_path: Path) -> None:
    """The shipped shape — a stand-in on its own port, nothing else claiming it."""
    _standin_profile(5074).validate(tmp_path)


def test_live_standin_validate_leaves_a_profile_without_the_key_alone(tmp_path: Path) -> None:
    """Absent means no stand-in, and none of these rules have anything to say."""
    _parse_profile({"name": "x", "virtual_accelerator": {"port": 5064}}).validate(tmp_path)


# --- the shipped perturbation's grammar ------------------------------------


def test_live_standin_validate_shipped_bpm_errors_accepts_offsets_only() -> None:
    """Static transverse offsets are the whole of what the shipped default may perturb."""
    spec = "BPM01:offset_x=50e-6,offset_y=-30e-6;BPM07:offset_x=10e-6"
    assert shipped_bpm_errors_field_errors(spec) == []


def test_live_standin_validate_shipped_bpm_errors_names_every_other_field() -> None:
    """One failure per non-offset field, each naming the entry it came from."""
    spec = "BPM01:offset_x=50e-6,gain_y=1.05;BPM07:roll=0.01"
    assert shipped_bpm_errors_field_errors(spec) == [
        "VA_STANDIN_BPM_ERRORS entry 'BPM01:offset_x=50e-6,gain_y=1.05' perturbs 'gain_y'; "
        "the shipped stand-in default is offset_x/offset_y only",
        "VA_STANDIN_BPM_ERRORS entry 'BPM07:roll=0.01' perturbs 'roll'; "
        "the shipped stand-in default is offset_x/offset_y only",
    ]


def test_live_standin_validate_shipped_bpm_errors_ignores_entries_naming_no_field() -> None:
    """Empty and device-only entries are the IOC's to refuse; this check is about fields."""
    assert shipped_bpm_errors_field_errors("") == []
    assert shipped_bpm_errors_field_errors(";; ;") == []
    assert shipped_bpm_errors_field_errors("BPM01;BPM07:") == []
