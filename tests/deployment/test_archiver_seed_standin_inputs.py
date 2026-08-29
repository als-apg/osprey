"""Where the seed gets the stand-in's offsets from, and what it refuses to invent.

The archive belongs to the machine it records, and a model has no past. A
deployment that records its own store beside a stand-in has the recorder
sampling that stand-in — so the deploy-time seed must lay down a past carrying
the same systematic BPM offsets, or the store shows a clean machine's history
under a displaced machine's present and every trend across the seam has a step
no operator caused.

This is the *host* half of that: deciding whether there is a stand-in, finding
the offsets it will actually run with, and turning them into the callable and
the fingerprint description the seeder takes. The arithmetic itself lives in
``osprey_connectors.simulation.archiver_seed`` and is pinned in
``tests/simulation/test_archiver_seed_transform.py``.

Two rules carry most of the weight:

* **Not the stand-in's archive, no transform.** The pair comes back
  ``(None, None)`` unless both halves of ``archive_belongs_to_standin`` hold —
  a stand-in was stood up AND this deployment runs the recorder that samples it
  — so a deployment without one seeds and fingerprints exactly as it did before
  the hook existed. That is asserted directly rather than assumed.

* **Only offsets, and say so when there is more.** With the readout chain at
  identity a reading is ``truth - offset``, which the seed can reproduce by
  subtraction. A gain, a roll or a noise term cannot be reproduced that way at
  all, so an operator override carrying one is applied by the container, skipped
  here, and warned about by name.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle

# The shipped stand-in default, as the four devices and axes it perturbs. Read
# from the module that owns it rather than restated, so a change to the shipped
# perturbation moves this test with it instead of leaving it asserting history.
from osprey.services.virtual_accelerator.manifest.standin_defaults import parse_standin_default

STANDIN_PORT = 5164

# Every address the shipped default reaches, mapped to the offset on it.
SHIPPED_OFFSETS = {
    f"SR:DIAG:BPM:{fam_name.removeprefix('BPM')}:POSITION:{field[-1].upper()}": value
    for fam_name, fields in parse_standin_default().items()
    for field, value in fields.items()
}

UNPERTURBED = "SR:VAC:IP07:PRESSURE"


def _standin_config() -> dict:
    """A rendered config whose archive is its stand-in's.

    Only the two keys the predicate reads: ``archive_belongs_to_standin`` is the
    single place "whose past is in this store" is answered — the deployment
    stood a stand-in up, and it runs the recorder that samples it — and a test
    that built the whole gateway table would be asserting a second answer.
    """
    return {
        "services": {"live_standin": {"port": STANDIN_PORT}},
        "deployed_services": ["mongodb", "archiver_recorder", "live_standin"],
    }


def _manifest_channel(address: str) -> dict:
    """One manifest entry carrying the full per-channel schema the loader demands."""
    return {
        "address": address,
        "ring": "SR",
        "system": "diagnostics",
        "family": "BPM",
        "device": "1",
        "field": "X",
        "subfield": "",
        "partition": "static-noisy",
        "record_type": "ai",
        "noise": 0.01,
    }


def _project(tmp_path: Path, *, env: str = "") -> Path:
    """A project directory whose manifest serves every shipped-default address."""
    simulation_dir = tmp_path / "build" / "data" / "simulation"
    simulation_dir.mkdir(parents=True)
    addresses = [*SHIPPED_OFFSETS, UNPERTURBED]
    (simulation_dir / "channel_manifest.json").write_text(
        json.dumps({"channels": [_manifest_channel(address) for address in addresses]})
    )
    (tmp_path / ".env").write_text(f"VA_CHANNELS_FILE=channel_manifest.json\n{env}")
    return tmp_path


def _seed_transform(config: dict, project_dir: Path):
    """The pair the seeder is handed, resolved the way the deploy resolves it."""
    _channels, _engine, _boot, transform, fingerprint = container_lifecycle._archiver_seed_inputs(
        config, project_dir
    )
    return transform, fingerprint


# ---------------------------------------------------------------------------
# No stand-in
# ---------------------------------------------------------------------------


def test_no_standin_means_no_seed_transform(tmp_path: Path) -> None:
    """A deployment without a stand-in seeds and fingerprints as it always did.

    Both halves are ``None``, which is what makes the base series and the
    manifest byte-identical to a run predating the hook — a store seeded before
    this feature must not read as MISMATCH and rebuild for nothing.
    """
    transform, fingerprint = _seed_transform({}, _project(tmp_path))

    assert transform is None
    assert fingerprint is None


def test_a_leftover_standin_block_without_a_port_means_no_seed_transform(tmp_path: Path) -> None:
    """A key that names no port is not a deployment saying where its stand-in is,
    and the seed follows the same predicate the recorder does.

    The recorder is deployed here, so the port is the only conjunct left to
    fail: without it this would pass on the half of the predicate the test is
    not about."""
    config = {
        "services": {"live_standin": {"port": ""}},
        "deployed_services": ["mongodb", "archiver_recorder"],
    }

    transform, fingerprint = _seed_transform(config, _project(tmp_path))

    assert transform is None
    assert fingerprint is None


# --- begin: recorder-and-seed-bind-predicate -------------------------------
# Whose past the store holds is `archive_belongs_to_standin`, and both of its
# conjuncts gate the seed. Each is checked with the other one satisfied, so a
# transform that came back for one alone would fail here rather than pass by
# accident.
# ---------------------------------------------------------------------------


def test_a_standin_with_no_recorder_is_not_the_archives_machine(tmp_path: Path) -> None:
    """A stand-in nothing records leaves the seed alone.

    The store this deploy seeds is never sampled from the stand-in, so laying a
    displaced machine's past into it would describe a machine no half of this
    deployment ever writes. The port alone is not the question — whose history
    the store holds is — and the seed answers it with the same predicate the
    recorder's compose entry and enablement gate do.
    """
    config = {
        "services": {"live_standin": {"port": STANDIN_PORT}},
        "deployed_services": ["mongodb", "live_standin"],
    }

    transform, fingerprint = _seed_transform(config, _project(tmp_path))

    assert transform is None
    assert fingerprint is None


def test_a_recorder_with_no_standin_is_not_the_archives_machine(tmp_path: Path) -> None:
    """The recorder half alone is the ordinary VA deployment, seeded as ever.

    Nothing was stood up to displace the readings, so the recorded present is
    the sandbox machine's and the seeded past has to be the same machine's.
    """
    config = {"deployed_services": ["mongodb", "archiver_recorder", "virtual_accelerator"]}

    transform, fingerprint = _seed_transform(config, _project(tmp_path))

    assert transform is None
    assert fingerprint is None


def test_the_seed_never_reads_the_epics_block_for_the_recorded_machine(tmp_path: Path) -> None:
    """``live`` always means the facility's authored ``epics`` block, and the
    archive question is not asked of it.

    An ``epics`` block pointed at a real facility does not stop this deployment
    recording its own stand-in into its own store, and one pointed at the
    stand-in's loopback port cannot conjure a stand-in that was never stood up.
    Both directions are pinned, because a seed that consulted ``epics`` would
    move under exactly these two edits.
    """
    project = _project(tmp_path)
    real_machine = {
        "control_system": {
            "connector": {
                "epics": {"gateways": {"read_only": {"address": "cagw.example.org", "port": 5064}}}
            }
        }
    }

    transform, fingerprint = _seed_transform(_standin_config() | real_machine, project)
    assert transform is not None
    assert fingerprint == {"kind": "bpm_offsets", "offsets": SHIPPED_OFFSETS}

    pointed_at_the_standin = {
        "control_system": {
            "connector": {
                "epics": {"gateways": {"read_only": {"address": "127.0.0.1", "port": STANDIN_PORT}}}
            }
        },
        "deployed_services": ["mongodb", "archiver_recorder"],
    }

    transform, fingerprint = _seed_transform(pointed_at_the_standin, project)
    assert transform is None
    assert fingerprint is None


# --- end: recorder-and-seed-bind-predicate ---------------------------------


# ---------------------------------------------------------------------------
# The shipped default
# ---------------------------------------------------------------------------


def test_the_shipped_standin_default_is_the_seed_transform_source(tmp_path: Path) -> None:
    """With no override, the offsets are the ones the compose template renders as
    the stand-in's own default — the values the container will actually run."""
    transform, fingerprint = _seed_transform(_standin_config(), _project(tmp_path))

    assert transform is not None
    assert fingerprint == {"kind": "bpm_offsets", "offsets": SHIPPED_OFFSETS}
    # Four devices, two of them single-axis: six readbacks, not eight.
    assert len(SHIPPED_OFFSETS) == 6


def test_the_seed_transform_subtracts_on_the_perturbed_addresses_only(tmp_path: Path) -> None:
    """``reading = truth - offset`` where the stand-in is displaced, and the value
    this module computed everywhere else."""
    transform, _fingerprint = _seed_transform(_standin_config(), _project(tmp_path))
    assert transform is not None

    for address, offset in SHIPPED_OFFSETS.items():
        assert transform(address, [1.0, 2.0]) == [1.0 - offset, 2.0 - offset]
    assert transform(UNPERTURBED, [1.0, 2.0]) == [1.0, 2.0]


def test_the_seed_transform_ignores_a_device_the_manifest_does_not_serve(tmp_path: Path) -> None:
    """A fam_name this lattice has no BPM for perturbs nothing on the live half
    either (the physics bridge warns and carries on), so the fingerprint
    describes what the store holds rather than what the spec asked for."""
    project = _project(tmp_path, env="VA_STANDIN_BPM_ERRORS=BPM99:offset_x=1e-4\n")

    transform, fingerprint = _seed_transform(_standin_config(), project)

    assert transform is None
    assert fingerprint is None


# ---------------------------------------------------------------------------
# The operator override
# ---------------------------------------------------------------------------


def test_a_dotenv_override_wins_over_the_shipped_seed_transform_default(tmp_path: Path) -> None:
    """The project's ``.env`` is what the compose interpolation reads, so it is
    what the stand-in runs — and therefore what the seeded past must carry."""
    address = next(iter(SHIPPED_OFFSETS))
    device = address.split(":")[3]
    project = _project(tmp_path, env=f"VA_STANDIN_BPM_ERRORS=BPM{device}:offset_x=7.5e-4\n")

    transform, fingerprint = _seed_transform(_standin_config(), project)

    assert fingerprint == {"kind": "bpm_offsets", "offsets": {f"{address[:-1]}X": 7.5e-4}}
    assert transform is not None
    assert transform(f"{address[:-1]}X", [1.0]) == [1.0 - 7.5e-4]


def test_an_empty_override_seeds_an_unperturbed_seed_transform_standin(
    tmp_path: Path,
) -> None:
    """``${VA_STANDIN_BPM_ERRORS-<default>}`` substitutes only when UNSET, so an
    explicitly empty value is an unperturbed stand-in and not the shipped
    default. The seed mirrors the interpolation because the container is the
    authority on what it is running — and seeding the shipped offsets under a
    machine serving none would put a step in every trend across the seam."""
    project = _project(tmp_path, env="VA_STANDIN_BPM_ERRORS=\n")

    transform, fingerprint = _seed_transform(_standin_config(), project)

    assert transform is None
    assert fingerprint is None


def test_an_ambient_override_is_the_fallback_for_the_seed_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same order, and for the same reason, as ``VA_CHANNELS_FILE``: the build
    writes the project's value into its ``.env``, and an exported one covers a
    deploy whose environment carries it instead."""
    monkeypatch.setenv("VA_STANDIN_BPM_ERRORS", "BPM03:offset_y=-9.0e-4")

    _transform, fingerprint = _seed_transform(_standin_config(), _project(tmp_path))

    assert fingerprint == {
        "kind": "bpm_offsets",
        "offsets": {"SR:DIAG:BPM:03:POSITION:Y": -9.0e-4},
    }


def test_a_non_offset_override_field_is_skipped_and_warned_about(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A gain, a roll or a noise term is not a subtraction of the synthesized
    value, so the seed cannot reproduce it. The container still applies it — the
    honest report is that the seeded past and the recorded present will differ by
    exactly those fields, named."""
    project = _project(
        tmp_path, env="VA_STANDIN_BPM_ERRORS=BPM03:offset_x=1e-4,gain_y=1.05;BPM21:roll=0.01\n"
    )

    with caplog.at_level(logging.WARNING):
        transform, fingerprint = _seed_transform(_standin_config(), project)

    assert fingerprint == {"kind": "bpm_offsets", "offsets": {"SR:DIAG:BPM:03:POSITION:X": 1e-4}}
    assert transform is not None
    assert "gain_y" in caplog.text
    assert "roll" in caplog.text


# --- begin: compose-standin-default-conditional ----------------------------
# Which default the seed falls back to when the chain names no perturbation.
# The shipped offsets displace the builtin PyAT model, so they are the default
# for the builtin lattice ONLY: a deployment whose chain resolves VA_LATTICE
# elsewhere is rendered the empty set and serves its facility manifest
# unperturbed, and a past seeded with the shipped offsets under it would
# describe a machine this deployment never runs. Read through the one function
# that owns the rule (`build_profile_va_faults.effective_standin_bpm_errors`),
# so the refusal, the render and the seed cannot answer it three ways.
# ---------------------------------------------------------------------------


def test_a_non_builtin_lattice_seeds_an_unperturbed_seed_transform_standin(
    tmp_path: Path,
) -> None:
    """``VA_LATTICE=none`` has no model to displace, so there is nothing to seed.

    The render hands that stand-in an empty fault set rather than refusing the
    build, and the seeded past has to match the machine that will actually be
    serving — clean, on the facility's own manifest.
    """
    project = _project(tmp_path, env="VA_LATTICE=none\n")

    transform, fingerprint = _seed_transform(_standin_config(), project)

    assert transform is None
    assert fingerprint is None


def test_a_builtin_lattice_pin_keeps_the_shipped_seed_transform_default(tmp_path: Path) -> None:
    """The conditional turns on the lattice and nothing else.

    A chain that pins the lattice the shipped offsets were written for gets
    exactly what an unpinned one gets — the resolver's default IS ``builtin``,
    so naming it changes no answer.
    """
    project = _project(tmp_path, env="VA_LATTICE=builtin\n")

    _transform, fingerprint = _seed_transform(_standin_config(), project)

    assert fingerprint == {"kind": "bpm_offsets", "offsets": SHIPPED_OFFSETS}


def test_a_non_builtin_lattice_still_seeds_an_explicit_seed_transform_override(
    tmp_path: Path,
) -> None:
    """A chain that names its own perturbation is seeded with it, lattice aside.

    The lattice only decides the DEFAULT. This shape — a non-empty override on a
    lattice that cannot apply it — is refused at validation, so it is not a
    deployment the seed will meet; what is pinned here is that the two halves
    are independent, and the seed never quietly drops an override it was given.
    """
    address = next(iter(SHIPPED_OFFSETS))
    device = address.split(":")[3]
    project = _project(
        tmp_path,
        env=f"VA_LATTICE=none\nVA_STANDIN_BPM_ERRORS=BPM{device}:offset_x=3.0e-4\n",
    )

    _transform, fingerprint = _seed_transform(_standin_config(), project)

    assert fingerprint == {"kind": "bpm_offsets", "offsets": {f"{address[:-1]}X": 3.0e-4}}


def test_an_empty_ambient_override_seeds_an_unperturbed_seed_transform_standin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Presence, not truthiness, on the ambient rung too.

    An exported ``VA_STANDIN_BPM_ERRORS=`` is the same request as the one in the
    project's ``.env``, and compose reads it the same way. Falling through to
    the shipped default because the value is falsy would seed a displaced past
    for a stand-in the operator asked to run clean.
    """
    monkeypatch.setenv("VA_STANDIN_BPM_ERRORS", "")

    transform, fingerprint = _seed_transform(_standin_config(), _project(tmp_path))

    assert transform is None
    assert fingerprint is None


# --- end: compose-standin-default-conditional ------------------------------
