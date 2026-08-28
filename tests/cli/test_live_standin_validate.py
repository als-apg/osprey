"""Tests for the ``virtual_accelerator.live_standin:`` refusals in ``BuildProfile.validate``.

The stand-in is a SECOND soft-IOC wired in as the deployment's ``live`` target,
so it can be wrong in ways the baseline soft-IOC cannot: it can land on a port
some other block already spends, it can land on the very gateway the simulation
is dialed through, it can be asked for by a deployment that is already pointed
at the real machine, and it can be built on a tree with no lattice behind the
readout perturbation it ships.

Each of those is pinned here by its exact message, because the message is the
whole deliverable — a refusal an author cannot act on is a build that fails
twice. The suite also pins the accumulation contract the rest of ``validate``
keeps: several stand-in faults arrive in ONE
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


# --- going live is three steps --------------------------------------------


def test_live_standin_validate_refuses_an_epics_baseline(tmp_path: Path) -> None:
    """A deployment already on the real machine has nothing to stand in for."""
    profile = _standin_profile(5074, config={"control_system.type": "epics"})
    assert _errors(profile, tmp_path) == [
        "control_system.type: epics with virtual_accelerator.live_standin — the stand-in "
        "IS this deployment's live target, and a deployment already pointed at the real "
        "machine has nothing to stand in for. Going live is three steps: delete `virtual_accelerator.live_standin`, point `control_system.connector.epics.gateways` at your facility, and replace `control_system.target_switch.live_gateway_acknowledged` with your own live gateway's hostname."
    ]


# --- the lattice the shipped perturbation needs ---------------------------


def test_live_standin_validate_refuses_a_dotenv_pinning_the_lattice_off(tmp_path: Path) -> None:
    """``VA_LATTICE=none`` on file wins over the build, and the stand-in exits at boot."""
    env_path = tmp_path / ".env"
    env_path.write_text("VA_LATTICE=none\n")
    assert _errors(_standin_profile(5074), tmp_path) == [
        f"virtual_accelerator.live_standin needs a lattice-backed virtual accelerator, "
        f"but {env_path} pins VA_LATTICE='none'. The build appends to that file and never "
        f"overwrites it, so the stand-in would boot with no lattice behind the "
        f"perturbation it ships and exit. Remove the line, or delete "
        f"`virtual_accelerator.live_standin`."
    ]


def test_live_standin_validate_accepts_a_lattice_pinned_to_builtin(tmp_path: Path) -> None:
    """The pin the build would have written itself is not a fault."""
    (tmp_path / ".env").write_text("VA_LATTICE=builtin\n")
    _standin_profile(5074).validate(tmp_path)


# --- accumulation, and the clean case --------------------------------------


def test_live_standin_validate_reports_every_violation_in_one_error(tmp_path: Path) -> None:
    """Three unrelated stand-in faults arrive in one raise, not one rebuild each."""
    (tmp_path / ".env").write_text("VA_LATTICE=none\n")
    profile = _standin_profile(
        5074,
        config={"control_system.type": "epics", "services.postgresql.port_host": 5074},
    )
    reported = _errors(profile, tmp_path)
    assert len(reported) == 3
    assert reported[0].startswith("virtual_accelerator.live_standin (5074) collides with")
    assert reported[1].startswith("control_system.type: epics")
    assert reported[2].startswith(
        "virtual_accelerator.live_standin needs a lattice-backed virtual accelerator"
    )


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
