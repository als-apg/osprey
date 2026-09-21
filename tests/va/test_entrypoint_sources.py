"""Tests for entrypoint.py's facility-neutral source configuration.

Exercises the resolution helpers directly (not ``main()``, which also needs
a real ``machine.json`` and softioc) -- same shape as
``test_entrypoint_fault_env.py``. Each helper reads ``os.environ`` itself,
so tests set env vars via ``monkeypatch``.

The one thing neither helper will do is choose a namespace on its own. A
container that picks the framework's bundled demo channels when it was told
nothing serves addresses that belong to one particular facility, under
whatever name the deployment gave it -- and from a client's side that is
indistinguishable from serving the facility. So the demo namespace is a
committed file like any other manifest, reachable only by being named, and
the tests below pin both halves: the refusal, and that naming it still
yields exactly the namespace the generator produces.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import at
import pytest

from osprey.services.virtual_accelerator import entrypoint
from osprey.services.virtual_accelerator.bindings import load_bindings
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    READBACK_SUBFIELD,
    RECORD_TYPE_ANALOG,
    SETPOINT_SUBFIELD,
)
from osprey.services.virtual_accelerator.manifest.build import (
    MANIFEST_FILENAME,
    build_manifest,
)
from osprey.services.virtual_accelerator.manifest.loaders import load_manifest_file
from osprey.services.virtual_accelerator.manifest.paths import (
    MANIFEST_OUTPUT,
    PACKAGE_PATHS,
    ManifestPaths,
)
from tests.va.test_pyat_ring_model import (
    BPM_X,
    CORR_SP,
    MONITOR_ELEMENT,
    QUAD_RB,
    QUAD_SP,
    _kick,
    _limits,
    _machine,
    _manifest,
    _monitor,
    _ring,
    _strength,
    _tree,
)
from tests.va.test_serving_entrypoint import VA_ENV_VARS, _boot

#: The file name a facility tree's lattice carries -- the one ``VA_LATTICE``
#: names when a lattice is served, derived from the layout rather than spelled
#: again here.
DEMO_LATTICE = PACKAGE_PATHS.lattice_json.name


@pytest.fixture(autouse=True)
def _clean_va_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No VA_* variable a developer's shell exports reaches these tests."""
    for name in VA_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class TestResolveChannelsFile:
    """Backs VA_CHANNELS_FILE -- the channel source, which is required."""

    def test_unset_is_fatal(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VA_CHANNELS_FILE", raising=False)
        with pytest.raises(SystemExit) as excinfo:
            entrypoint._resolve_channels_file(tmp_path)
        assert "VA_CHANNELS_FILE" in str(excinfo.value)

    def test_empty_is_fatal_too(self, monkeypatch, tmp_path):
        # The compose passthrough sends "" when the host var is absent --
        # empty must behave exactly like unset, and unset is a refusal.
        monkeypatch.setenv("VA_CHANNELS_FILE", "")
        with pytest.raises(SystemExit, match="VA_CHANNELS_FILE"):
            entrypoint._resolve_channels_file(tmp_path)

    def test_refusal_names_both_ways_out(self, monkeypatch, tmp_path):
        """The message has to leave an operator somewhere to go.

        Two audiences reach it, and each needs a different sentence: a project
        deployment whose build should have written the pointer, and a hand-run
        container that wanted the demo. So the refusal names ``osprey build``
        for the first and the packaged manifest's own resolved path -- plus
        the lattice that goes with it -- for the second.
        """
        monkeypatch.delenv("VA_CHANNELS_FILE", raising=False)
        with pytest.raises(SystemExit) as excinfo:
            entrypoint._resolve_channels_file(tmp_path)
        message = str(excinfo.value)
        assert "osprey build" in message
        assert str(MANIFEST_OUTPUT) in message
        assert f"VA_LATTICE={DEMO_LATTICE}" in message
        # And it says what it will not do, so the refusal reads as a decision
        # rather than as a missing default.
        assert "never falls back" in message

    def test_relative_path_resolves_against_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VA_CHANNELS_FILE", "channels_manifest.json")
        assert entrypoint._resolve_channels_file(tmp_path) == tmp_path / "channels_manifest.json"

    def test_absolute_path_is_kept(self, monkeypatch, tmp_path):
        absolute = tmp_path / "elsewhere" / "manifest.json"
        monkeypatch.setenv("VA_CHANNELS_FILE", str(absolute))
        assert entrypoint._resolve_channels_file(tmp_path / "data") == absolute


class TestResolveLattice:
    """Backs VA_LATTICE -- the lattice file served, or none at all."""

    def test_default_is_no_lattice(self, monkeypatch, tmp_path):
        # The only model this process could build unasked is the framework's
        # own tutorial ring, which is the physics half of the same
        # substitution the channel source refuses. So it is asked for.
        monkeypatch.delenv("VA_LATTICE", raising=False)
        assert entrypoint._resolve_lattice(tmp_path) is None

    def test_explicit_none_is_honoured(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VA_LATTICE", entrypoint.LATTICE_NONE)
        assert entrypoint._resolve_lattice(tmp_path) is None

    def test_empty_behaves_like_unset(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VA_LATTICE", "")
        assert entrypoint._resolve_lattice(tmp_path) is None

    def test_a_relative_name_resolves_against_the_data_dir(self, monkeypatch, tmp_path):
        (tmp_path / DEMO_LATTICE).write_text("{}")
        monkeypatch.setenv("VA_LATTICE", DEMO_LATTICE)
        assert entrypoint._resolve_lattice(tmp_path) == tmp_path / DEMO_LATTICE

    def test_an_absolute_name_is_kept(self, monkeypatch, tmp_path):
        absolute = tmp_path / "elsewhere.json"
        absolute.write_text("{}")
        monkeypatch.setenv("VA_LATTICE", str(absolute))
        assert entrypoint._resolve_lattice(tmp_path / "data") == absolute

    def test_the_case_of_the_name_is_preserved(self, monkeypatch, tmp_path):
        """The served tree is searched for that file verbatim: a case-folding
        resolver finds a file on one host and misses it on another."""
        (tmp_path / "Lattice.JSON").write_text("{}")
        monkeypatch.setenv("VA_LATTICE", "Lattice.JSON")
        assert entrypoint._resolve_lattice(tmp_path).name == "Lattice.JSON"

    def test_a_name_the_tree_does_not_carry_is_fatal(self, monkeypatch, tmp_path):
        """Refused here, against the tree this process was handed, rather than
        surfacing later as a decoding failure from inside the lattice loader.
        The refusal leaves somewhere to go: the value that serves no lattice
        at all."""
        monkeypatch.setenv("VA_LATTICE", "als.json")
        with pytest.raises(SystemExit) as excinfo:
            entrypoint._resolve_lattice(tmp_path)
        message = str(excinfo.value)
        assert str(tmp_path / "als.json") in message
        assert f"VA_LATTICE={entrypoint.LATTICE_NONE}" in message


class TestPackagedDemoManifest:
    """The demo namespace the refusal points at, and what naming it buys.

    Removing the implicit fallback only holds if the explicit route lands in
    the same place: the tutorial quick-start (``scripts/va/run_va.sh``) and
    the image boot gate both name this file, and both are written against the
    channel set the generator produces. A committed manifest that had drifted
    from ``build_manifest()`` would move them off it silently -- same script,
    same readiness line, a different machine.
    """

    def test_the_packaged_manifest_is_the_generated_namespace(self):
        assert load_manifest_file(MANIFEST_OUTPUT) == build_manifest()["channels"]

    def test_naming_it_resolves_to_that_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VA_CHANNELS_FILE", str(MANIFEST_OUTPUT))
        assert entrypoint._resolve_channels_file(tmp_path) == MANIFEST_OUTPUT


class TestParameterizedDataLoads:
    """Every boot reads facility data from the mount, never from the bundled
    tutorial files.

    The write bands have no default at all: they belong to one tree, and a
    process serving a facility was handed that facility's directory. The boot
    values keep theirs for the bundled-template reads the build-time suites
    are written against, and no boot takes it."""

    def test_boot_values_from_explicit_machine_json(self, tmp_path):
        machine = tmp_path / "machine.json"
        machine.write_text(
            json.dumps(
                {
                    "channels": {
                        "ZZEXP:LASER:ENERGY:RB": {"value": 1.25},
                        "ZZEXP:LASER:MODE:RB": {"expr": "derived, no value"},
                    }
                }
            )
        )
        assert entrypoint._load_boot_values(machine) == {"ZZEXP:LASER:ENERGY:RB": 1.25}

    def test_drive_limits_from_explicit_limits_file(self, tmp_path):
        limits = tmp_path / "channel_limits.json"
        limits.write_text(
            json.dumps(
                {
                    "defaults": {"writable": False},
                    "ZZEXP:JET:STAGE:SP": {
                        "writable": True,
                        "min_value": -5.0,
                        "max_value": 5.0,
                    },
                    "ZZEXP:LOCKED:DOWN:SP": {"min_value": 0, "max_value": 1},
                }
            )
        )
        # The non-writable default suppresses the second entry; only the
        # explicitly writable one yields a clamp band.
        assert entrypoint._load_drive_limits(
            limits, setpoints={"ZZEXP:JET:STAGE:SP", "ZZEXP:LOCKED:DOWN:SP"}
        ) == {"ZZEXP:JET:STAGE:SP": (-5.0, 5.0)}

    def test_the_boot_value_default_still_reads_the_bundled_data(self):
        # Calling with no argument must keep returning the bundled tutorial
        # data (non-empty, known shape) for the callers that still do.
        assert entrypoint._load_boot_values()

    def test_the_write_bands_have_to_be_named(self):
        """A band map is a statement about one tree, so the file it comes from
        is always the caller's to name -- there is no tree a serving process
        could fall back to that is not some other facility's."""
        with pytest.raises(TypeError):
            entrypoint._load_drive_limits(setpoints=frozenset())


# --- the boot, against a served tree and a real model ----------------------
#
# Every test above drives a resolution helper. These drive `main()` with the
# model the tree actually describes -- the one place the served files, the
# layout that finds them and the model built from them meet. The runner and
# the threads are faked (the sibling suite's harness); the manifest, the
# bindings document, the lattice, the nominals and the bands are the real
# files, because they are the thing under test.


@pytest.fixture(scope="module")
def served_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A whole facility tree, and the directory a deployment mounts from it.

    Written by the model suite's own builder, so what boots here is the tree
    shape that suite pins rather than a second idea of one. The manifest and
    the scenario-state file are added because the boot reads them and the
    model does not.
    """
    root = _tree(
        tmp_path_factory.mktemp("boot") / "data",
        machine=_machine(),
        limits=_limits(),
    )
    served = ManifestPaths(data_root=root).machine_json.parent
    (served / MANIFEST_FILENAME).write_text(json.dumps({"channels": _manifest()}))
    (served / "active_scenarios").write_text("[]")
    return served


def _copied(served: Path, destination: Path) -> Path:
    """The served tree copied whole, and the served directory inside the copy.

    The data root and everything under it, because the tree is what the model
    reads: the bands sit at the root and the model beside the manifest.
    """
    shutil.copytree(served.parent, destination)
    return destination / served.name


def _boot_served(monkeypatch: pytest.MonkeyPatch, served: Path, *, fake_bridge: bool = True) -> Any:
    """Assemble the process against ``served``, with the real model.

    ``fake_bridge=False`` keeps the deployed physics bridge as well, so the
    boot is the one a container performs: the tree's own monitors, read
    through the tree's own bindings.
    """
    return _boot(
        monkeypatch,
        served,
        lattice=DEMO_LATTICE,
        channels_file=MANIFEST_FILENAME,
        fake_model=False,
        fake_bridge=fake_bridge,
    )


class TestTheModelComesFromTheMountedTree:
    """The boot builds the model the mounted files describe, and no other."""

    def test_a_served_tree_boots_and_binds_its_own_addresses(
        self, monkeypatch: pytest.MonkeyPatch, served_tree: Path
    ) -> None:
        """The model's variables are the manifest's coupled addresses, named
        exactly as the manifest carries them -- nothing between the served
        files and the model translates an address."""
        boot = _boot_served(monkeypatch, served_tree)
        served_addresses = {
            channel["address"]
            for channel in _manifest()
            if channel["partition"] == PARTITION_PYAT_COUPLED
            and channel["subfield"] != READBACK_SUBFIELD
        }
        assert served_addresses <= set(boot.runner.model.supported_variables)

    def test_the_readback_rules_the_tree_exported_reach_the_runner(
        self, monkeypatch: pytest.MonkeyPatch, served_tree: Path
    ) -> None:
        """Read off the tree's own bindings document: which addresses serve a
        readback at all, and which of them serve the reverse curve."""
        bound = _boot_served(monkeypatch, served_tree).runner.kwargs["bound_setpoints"]
        assert bound[QUAD_SP].rule == "inverse"
        assert callable(bound[QUAD_SP].value)


class TestTheBandComesFromTheMountedFile:
    """A nominal outside its band refuses the boot, on the served files.

    The point is not that the refusal exists -- the model suite pins that --
    but that a deployment's own copy of the tree is what decides it. So the
    same tree is booted twice: copied and untouched, then copied with the band
    its own ``channel_limits.json`` states narrowed below the nominal standing
    beside it in ``machine.json``. Narrowing the band rather than moving the
    nominal is what makes the second boot's refusal provenance: no file but
    the copy's own limits states that band, so only a read of it can produce
    the refusal. If the bands were read from anywhere but the mount, both
    boots would behave alike.
    """

    def test_the_untouched_copy_boots(
        self, monkeypatch: pytest.MonkeyPatch, served_tree: Path, tmp_path: Path
    ) -> None:
        copy = _copied(served_tree, tmp_path / "unchanged")
        assert _boot_served(monkeypatch, copy).runner.model.supported_variables

    def test_an_out_of_band_nominal_in_the_copy_refuses_the_boot(
        self, monkeypatch: pytest.MonkeyPatch, served_tree: Path, tmp_path: Path
    ) -> None:
        copy = _copied(served_tree, tmp_path / "out-of-band")
        limits_json = ManifestPaths(data_root=copy.parent).channel_limits
        limits = json.loads(limits_json.read_text())
        # The band's upper edge brought a whole ampere below the nominal the
        # copy's own machine.json states, so no rounding argument reaches it
        # and the machine file is left exactly as every other tree carries it.
        nominal = _machine()[QUAD_SP]["value"]
        limits[QUAD_SP]["max_value"] = nominal - 1.0
        limits_json.write_text(json.dumps(limits))

        with pytest.raises(SystemExit) as excinfo:
            _boot_served(monkeypatch, copy)
        message = str(excinfo.value)
        assert QUAD_SP in message
        # And it names the file whose band was violated, in the copy: an
        # operator reading this has to know which of the two it is.
        assert str(limits_json) in message


class TestARefusalNamesTheCauseItHas:
    """A tree the model refuses is not always a tree that contradicts itself.

    The band refusal is one validation among the many a facility tree can
    fail: a slice weight, an energy table, a file that is not JSON at all.
    Each of those names its own file and key in the text it carries, so the
    boot's headline over them stays neutral -- a headline announcing a nominal
    outside the band ``channel_limits.json`` states sends an operator to a file
    that is fine, with the real cause buried in the parenthetical.
    """

    def test_a_limits_file_that_is_not_json_is_not_reported_as_a_band(
        self, monkeypatch: pytest.MonkeyPatch, served_tree: Path, tmp_path: Path
    ) -> None:
        copy = _copied(served_tree, tmp_path / "unreadable-limits")
        ManifestPaths(data_root=copy.parent).channel_limits.write_text("{ not json")

        with pytest.raises(SystemExit) as excinfo:
            _boot_served(monkeypatch, copy)
        message = str(excinfo.value)
        assert "contradicts itself" not in message
        # It still says which tree refused, because that is the one thing the
        # exception text from a parser does not carry.
        assert str(copy.parent) in message


class TestACavityLessTreeIsNotABootRefusal:
    """A deck with no cavity is a four-dimensional model, not a broken tree.

    ``enable_6d(at.RFCavity)`` touches cavities alone, so a ring that carries
    none stays 4D and its closed orbit is solved by the 4D solver. That is a
    coherent served model -- a sub-machine, or a line exported without its rf
    -- and refusing it at boot would refuse trees that work.

    What a cavity-less tree cannot do is carry an ``rf`` binding, and that is
    already refused where it is sharp: the binding names a cavity element, and
    the ring loader refuses a bound element the lattice does not have, by
    name. A blanket cavity check here would add nothing to that refusal and
    would take the working case away.
    """

    def test_a_tree_whose_deck_carries_no_cavity_still_boots(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        ring = _ring()
        four_dimensional = at.Lattice(
            [element for element in ring if not isinstance(element, at.RFCavity)],
            name=ring.name,
            energy=ring.energy,
            periodicity=1,
        )
        root = _tree(
            tmp_path / "data",
            ring=four_dimensional,
            bindings=[_strength(), _kick(), _monitor()],
            machine=_machine(),
            limits=_limits(),
        )
        served = ManifestPaths(data_root=root).machine_json.parent
        (served / MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "channels": [
                        channel
                        for channel in _manifest()
                        if channel["address"] in {QUAD_SP, QUAD_RB, CORR_SP, BPM_X}
                    ]
                }
            )
        )
        (served / "active_scenarios").write_text("[]")

        boot = _boot_served(monkeypatch, served)
        assert not boot.runner.model.lattice.is_6d
        assert QUAD_SP in boot.runner.model.supported_variables


class TestTheCoupledPartitionIsServedWhole:
    """Every channel the manifest calls coupled comes up with a record.

    The boot's promise to a facility: the partition the manifest states is the
    partition the wire carries. A channel dropped between the two is invisible
    from inside the process -- the model has its variables, the bridge has its
    monitors, and only a client asking for the missing address finds out.
    """

    def test_every_coupled_channel_of_the_mounted_manifest_gets_a_record(
        self, monkeypatch: pytest.MonkeyPatch, served_tree: Path
    ) -> None:
        boot = _boot_served(monkeypatch, served_tree)
        coupled = {
            channel["address"]
            for channel in _manifest()
            if channel["partition"] == PARTITION_PYAT_COUPLED
        }
        assert set(boot.runner.records.pyat_coupled) == coupled

    def test_the_same_tree_served_without_physics_serves_the_same_namespace(
        self, monkeypatch: pytest.MonkeyPatch, served_tree: Path
    ) -> None:
        """The lattice decides what happens after a write, never what is
        served: a deployment that cannot run PyAT serves its facility's whole
        namespace, coupled channels included, with nothing behind them."""
        with_physics = _boot_served(monkeypatch, served_tree)
        without = _boot(
            monkeypatch,
            served_tree,
            lattice=entrypoint.LATTICE_NONE,
            channels_file=MANIFEST_FILENAME,
        )
        assert set(without.runner.records.pvdb) == set(with_physics.runner.records.pvdb)
        assert set(without.runner.records.pyat_coupled) == set(
            with_physics.runner.records.pyat_coupled
        )
        assert without.runner.kwargs["on_setpoint"] is None


class TestTheDeployedBridgeReadsTheTreesOwnMonitors:
    """The boot a container performs, with nothing about the physics replaced.

    Every other boot here stands a fake in for the bridge, because what those
    tests ask about is the assembly. This one asks the question the fake
    cannot answer: whether the bridge a deployment actually runs can read the
    monitors of a tree it has never seen -- which it can only do by going
    through the served bindings for both the element each reading comes from
    and the curve it comes out in.
    """

    def test_a_tree_boots_with_the_bridge_a_deployment_runs(
        self, monkeypatch: pytest.MonkeyPatch, served_tree: Path
    ) -> None:
        boot = _boot_served(monkeypatch, served_tree, fake_bridge=False)
        bridge = boot.runner.kwargs["on_setpoint"].__self__
        readings = bridge.bpm_positions()
        assert readings, "the boot state is the orbit the server comes up serving"
        assert set(readings) <= set(boot.runner.records.pyat_coupled)
        # And the reading reached the record the server copies its spec from,
        # which is what makes it the value a client sees at boot.
        assert boot.runner.records.pvdb[BPM_X]["value"] == pytest.approx(readings[BPM_X])

    def test_a_readout_error_seeded_by_address_reaches_that_monitor(
        self, monkeypatch: pytest.MonkeyPatch, served_tree: Path
    ) -> None:
        """The whole chain in one assertion: a device named the way an
        operator reads it off the control system, resolved through the served
        document into the element the bridge keys by, and applied to the
        reading that element publishes -- which is the served value and not
        the model's truth, because the truth is what the error is applied to.
        """
        offset = 1.0e-3
        unperturbed = _boot_served(monkeypatch, served_tree, fake_bridge=False)
        before = unperturbed.runner.records.pvdb[BPM_X]["value"]

        monkeypatch.setenv("VA_BPM_ERRORS", f"{BPM_X}:offset_x={offset}")
        perturbed = _boot_served(monkeypatch, served_tree, fake_bridge=False)

        assert perturbed.runner.records.pvdb[BPM_X]["value"] == pytest.approx(before - offset)

    def test_the_element_spelling_of_the_same_monitor_does_the_same_thing(
        self, monkeypatch: pytest.MonkeyPatch, served_tree: Path
    ) -> None:
        """Two names for one device, and the boot cannot tell them apart:
        whichever a facility's people use, the same monitor is perturbed."""
        offset = 1.0e-3
        monkeypatch.setenv("VA_BPM_ERRORS", f"{MONITOR_ELEMENT}:offset_x={offset}")
        by_element = _boot_served(monkeypatch, served_tree, fake_bridge=False)

        monkeypatch.setenv("VA_BPM_ERRORS", f"{BPM_X}:offset_x={offset}")
        by_address = _boot_served(monkeypatch, served_tree, fake_bridge=False)

        assert by_element.runner.records.pvdb[BPM_X]["value"] == pytest.approx(
            by_address.runner.records.pvdb[BPM_X]["value"]
        )

    def test_a_device_the_tree_publishes_under_neither_name_ends_the_boot(
        self, monkeypatch: pytest.MonkeyPatch, served_tree: Path
    ) -> None:
        monkeypatch.setenv("VA_BPM_ERRORS", "NOSUCHBPM:offset_x=1e-3")
        with pytest.raises(SystemExit) as excinfo:
            _boot_served(monkeypatch, served_tree, fake_bridge=False)
        message = str(excinfo.value)
        assert "VA_BPM_ERRORS" in message
        assert "NOSUCHBPM" in message


# --- the emitted facility trees -------------------------------------------
#
# The same boot, against the trees the export and emit lanes write. Discovery
# rather than a list of facilities: a tree is a data root carrying
# `simulation/va_bindings.json` where the layout resolves one, so a facility
# whose fixture lands later is booted here without a line of code.

_MML_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "mml"


def _emitted_trees() -> list[Path]:
    """Every emitted served tree under the MML fixtures, by data root."""
    return sorted(
        path.parent.parent for path in _MML_FIXTURE_ROOT.rglob("simulation/va_bindings.json")
    )


def _document_channels(data_root: Path) -> list[dict[str, Any]]:
    """A channel list for a tree that ships no manifest, read off its bindings.

    An emitted tree carries no channel database -- a deployment's namespace
    comes from the facility's own databases at build time -- so what the
    document binds is the namespace to serve. Nothing here reads a family, a
    subsystem or a unit out of an address: what says a channel is written is
    the binding's kind, and every other manifest field is a value the serving
    database needs and this tree does not state.
    """
    document = load_bindings(ManifestPaths(data_root=data_root).va_bindings)
    channels: list[dict[str, Any]] = []
    for index, binding in enumerate(document.bindings):
        shared = {
            "ring": "TREE",
            "system": "VA",
            "family": binding.owner,
            # One device per binding, so a setpoint and the readback beside it
            # pair with each other and with no other binding's.
            "device": f"{index:04d}",
            "field": "VALUE",
            "partition": PARTITION_PYAT_COUPLED,
            "record_type": RECORD_TYPE_ANALOG,
            "noise": 0.0,
        }
        channels.append(
            {
                **shared,
                "address": binding.setpoint_address,
                "subfield": SETPOINT_SUBFIELD if binding.is_writable else READBACK_SUBFIELD,
            }
        )
        if binding.readback_address is not None:
            channels.append(
                {**shared, "address": binding.readback_address, "subfield": READBACK_SUBFIELD}
            )
    return channels


def _prepared_copy(source: Path, destination: Path) -> Path:
    """A copy of the tree at ``source``, and the directory a deployment mounts.

    Copied because a boot reads a tree the way a container does and a fixture
    is committed: the manifest a tree does not ship and the scenario-state
    file the engine polls are written into the copy, never beside the fixture.
    A tree that does ship a manifest keeps it -- that is the namespace its
    facility serves, and deriving a second one from the document would test
    this file's idea of the tree instead of the tree.
    """
    shutil.copytree(source, destination)
    paths = ManifestPaths(data_root=destination)
    served = paths.machine_json.parent
    manifest = served / MANIFEST_FILENAME
    if not manifest.is_file():
        manifest.write_text(json.dumps({"channels": _document_channels(destination)}))
    (served / "active_scenarios").write_text("[]")
    return served


@pytest.mark.skipif(
    not _emitted_trees(),
    reason=(
        f"no emitted 2.0 tree under {_MML_FIXTURE_ROOT}: the re-export writes a "
        f"facility's va.json, the import reader and the emit lane turn it into a "
        f"served tree (simulation/va_bindings.json beside its machine.json and "
        f"lattice), and these boot it the moment one lands"
    ),
)
@pytest.mark.parametrize(
    "tree", _emitted_trees() or [None], ids=lambda path: getattr(path, "name", "none")
)
class TestAnEmittedFacilityTreeBoots:
    """The whole boot against the trees the MML chain emits, facility by facility.

    Facility-agnostic by construction: every name is read out of the tree
    under test, so a third facility's fixture adds a third case and no code.
    """

    def test_every_coupled_channel_of_the_tree_is_served(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tree: Path
    ) -> None:
        served = _prepared_copy(tree, tmp_path / tree.name)
        boot = _boot_served(monkeypatch, served, fake_bridge=False)
        coupled = {
            channel["address"]
            for channel in load_manifest_file(served / MANIFEST_FILENAME)
            if channel["partition"] == PARTITION_PYAT_COUPLED
        }
        assert set(boot.runner.records.pyat_coupled) == coupled

    def test_every_writable_binding_of_the_tree_is_a_model_variable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tree: Path
    ) -> None:
        """A coupled setpoint with no variable behind it is a channel that
        accepts writes the ring never sees."""
        served = _prepared_copy(tree, tmp_path / tree.name)
        boot = _boot_served(monkeypatch, served, fake_bridge=False)
        document = load_bindings(ManifestPaths(data_root=served.parent).va_bindings)
        writable = {
            binding.setpoint_address for binding in document.bindings if binding.is_writable
        }
        assert writable <= set(boot.runner.model.supported_variables)

    def test_the_same_tree_serves_its_namespace_without_a_lattice(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tree: Path
    ) -> None:
        served = _prepared_copy(tree, tmp_path / tree.name)
        with_physics = _boot_served(monkeypatch, served)
        without = _boot(
            monkeypatch,
            served,
            lattice=entrypoint.LATTICE_NONE,
            channels_file=MANIFEST_FILENAME,
        )
        assert set(without.runner.records.pvdb) == set(with_physics.runner.records.pvdb)
        assert without.runner.kwargs["on_setpoint"] is None
