"""A harvested facility tree, served by a real container, over Channel Access.

The container half of success criterion 1: a facility export goes through the
whole install -- ``init``, ``facility import mml`` under the tree's reviewed
mapping, ``mml import``, the reviewed mapping, ``mml emit``, ``osprey set``,
``validate``, ``osprey build`` -- and the simulator view the build rendered is
handed to the virtual accelerator image, whose composite serves that
facility's own channels with that facility's own lattice behind them. Every earlier task in this
feature proves a step of that chain against files; this module is the only one
that proves the chain ends in a machine a control system can talk to.

What each lane asserts, and why it is not vacuous:

* **Every wired channel is served.** The simulator view's wiring says which
  addresses the physics models drive -- every setpoint they take and every
  reading they answer -- and every one of them answers a Channel Access read
  from the host. A mount the container could not resolve a model over serves
  none of them with physics behind it. Reading the whole wired set is what
  separates those from a machine.
* **A write comes back the way the served model says it does.** A driven
  channel either answers on the address it was written to or pairs with a
  readback of its own, which the model computes from the element field the
  write set, through that readback's own calibration. The expected value is
  read off an in-process composite over the same simulator view, driven
  through the same writes, never by restating a number: a lane that pasted
  one would pass against a calibration nobody exported.
* **The model is really behind the channels.** A corrector write moves the
  monitors. Nothing else in this file could distinguish a served model from a
  well-formed echo.
* **The harvest passed the stops its tree plants.** The imported limits hold
  each band as the export states it, so the build stops ``seed-invalid`` while
  a setpoint starts outside its band; the harvest widens exactly the records
  ``facility validate`` names, and the lane holds that set against the tree's
  own.

The trees. Naming a facility in ``CRITERION_TREES`` is the claim that its
export reaches a served machine, so a tree named there that commits no 2.0
export fails rather than stands aside. Whether it carries one is still
DISCOVERED -- a directory holding a ``*.va.json`` sibling is a 2.0 tree -- so
the claim is checked against the tree on disk rather than restated here. The
facilities are named in one tuple because what pytest parametrises over is read
at collection time, and a directory scan there would fail this whole directory
rather than one lane.

Nothing below skips. A facility exports the knobs it has, and the lanes differ
in which of them they drive, so a lane can find no device of its kind -- but
which kinds a tree couples is the reviewed mapping's answer, and that makes an
absence a fact to ASSERT against the mapping rather than a reason to stop
measuring. ``_agrees_with_the_mapping`` is where that is done, and it is the
last thing a lane does before ending empty-handed. In a report a skip and a
clean pass are told apart only by reading the reason; an assertion needs no
such reading.

Sequencing. One container at a time: the harvest and the boot are module-scoped
and parametrised over the trees, so pytest tears the previous tree's container
down before the next one starts, and the module's ``xdist_group`` mark keeps
every lane on one worker so a parallel run cannot boot the same tree twice
over. Each container takes an ephemeral host port of its own and a name unique
to the run, so a concurrent run of another module in this directory cannot
collide with, or force-remove, this one's.

Process-boundary note (the directory conftest's rule): every Channel Access
operation here runs in a SUBPROCESS. pyepics' libca contexts are per-thread and
a client that has used CA from a worker thread can wedge this process at
interpreter exit, so the client lives and dies in a process of its own, exactly
as the sibling lanes do.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _worker(request: dict) -> dict:
    """Run one scripted Channel Access exchange and report what it read.

    Two operations cover this module:

    ``read``
        Connect each address and report its value.
    ``write_read``
        Write each ``[address, value]`` pair with put-completion, then read the
        requested addresses. Put-completion is what makes the following read
        meaningful: the server ends the asynchronous write only once the model
        has taken the value and every readback it owes has been posted.

    Values are read off the wire (``use_monitor=False``). A subscribed PV's
    cached value can lag a completed put, and an assertion that reads a cache
    which never moves passes for free.
    """
    import epics

    op = request["op"]
    timeout = float(request.get("timeout", 30.0))

    def connect(address: str) -> Any:
        pv = epics.PV(address, connection_timeout=timeout)
        if not pv.wait_for_connection(timeout=timeout):
            raise RuntimeError(f"{address} never connected")
        return pv

    if op == "write_read":
        for address, value in request["writes"]:
            connect(address).put(value, wait=True, timeout=timeout)
    elif op != "read":
        raise RuntimeError(f"unknown worker op {op!r}")

    values: dict[str, Any] = {}
    failed: dict[str, str] = {}
    for address in request["addresses"]:
        try:
            values[address] = connect(address).get(use_monitor=False, timeout=timeout)
        except Exception as error:  # reported to the test, not raised here
            failed[address] = f"{type(error).__name__}: {error}"
    return {"values": values, "failed": failed}


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--worker":
    # Before the heavy imports below, so the CA client process carries pyepics
    # and nothing else.
    print(json.dumps(_worker(json.loads(sys.argv[2])), default=str), flush=True)
    raise SystemExit(0)

import pytest  # noqa: E402
import yaml  # noqa: E402
from click.testing import CliRunner  # noqa: E402

from tests.e2e._orm_stack import physics_wiring  # noqa: E402
from tests.va.e2e import conftest as e2e_conftest  # noqa: E402

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    # One worker runs every lane here. The boots must not overlap -- each is a
    # whole install recipe plus an emulated linux/amd64 container -- and a
    # module-scoped fixture only serializes them inside ONE worker process, so
    # under `--dist loadgroup` this mark is what keeps the group together.
    pytest.mark.xdist_group("mml-trees-boot"),
]

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
# Eight lanes over three trees; the guard test itself is the twenty-fifth
# item, so a floor of 24 reds on the loss of a single lane.
MIN_COLLECTED_TESTS = 24

#: The image under test -- the same one the rest of this directory serves from.
IMAGE = e2e_conftest.IMAGE

#: Container-name prefix. The run's own port and a random suffix are appended,
#: because a name shared with a concurrent run is destructive rather than tidy:
#: each run force-removes its own name as stale cleanup.
CONTAINER_PREFIX = "osprey-va-e2e-mml-tree"

BOOT_TIMEOUT_S = 240.0

#: Bound on ONE readiness probe. A probe worker that wedges is killed at this
#: bound and the next probe dials afresh; without it a single hung worker holds
#: the loop past the boot deadline and a server that is up is never asked again.
PROBE_TIMEOUT_S = 45.0

#: The facilities success criterion 1 names, and the trees this module opens a
#: lane for. A literal tuple, because parametrisation is read at COLLECTION: a
#: directory scan here fails the whole ``tests/va/e2e`` directory rather than
#: one lane. Every tree named here is CLAIMED to reach a served machine, and
#: ``served`` fails one that commits no ``*.va.json``; a facility not named
#: here joins by being added to this tuple.
CRITERION_TREES = ("nsls2", "spear3", "synthetic")


def _recipes():
    """The CLI recipe module, imported on first use rather than at collection.

    That module reads the fixture and packaged-knowledge directories while it
    imports. Importing it at module level would make an unreadable directory a
    collection error for every module in this directory; behind a call it fails
    only the lanes that need the recipe.
    """
    from tests.cli import test_mml_build_recipes

    return test_mml_build_recipes


#: How far a write moves a device from its nominal, as a fraction of the
#: nominal. Small on purpose: the point of every write lane is what the served
#: machine does with a value, and a large excursion invites the drive-limit
#: clamp -- which would make the readback assertions pass against a number the
#: test never chose. The band is not the only thing that can stop a write: a
#: value inside it is still refused when the ring loses its closed orbit, and
#: the one lane whose step of this size reaches that has a fraction of its
#: own below.
WRITE_FRACTION = 1e-3

#: The same, for the RF frequency alone. A frequency step moves the whole beam
#: off momentum: the ring answers a fractional step with a momentum deviation
#: larger by one over the momentum compaction, so what a ring takes is its
#: momentum acceptance times the momentum compaction itself -- tens of parts
#: per million on a real machine, and a part per thousand loses the closed
#: orbit outright. The model rolls such a write back and withholds its echo,
#: so the readback would then be about the refusal and not about the write.
#: The band does not stand in for this: a part per thousand sits well inside a
#: cavity's band and is refused anyway. A ring whose momentum compaction is
#: orders larger turns the same frequency step into a far smaller momentum
#: deviation and takes it, which is why the general fraction does not notice.
#: Measured on the served real-machine trees: both take one part in a hundred
#: thousand, and the tighter of them refuses two.
RF_WRITE_FRACTION = 1e-6

#: Tolerance for a served value against the in-process composite's. Generous
#: against the wire (Channel Access serves a double, and the IOC's display
#: precision does not enter a ``caget``), tight against the thing being tested:
#: a readback served through the wrong curve, or through no curve at all, is
#: wrong by orders of magnitude, not by parts in a billion.
READBACK_RTOL = 1e-9

#: The kind of device a write wiring record drives, by the engine attribute it
#: writes. The kinds are the reviewed mapping's own coupling kinds, so a served
#: view and its mapping can be held against each other kind by kind.
KIND_OF_ATTRIBUTE = {
    "KickAngle": "kick",
    "PolynomA": "strength",
    "PolynomB": "strength",
    "Frequency": "rf",
    "energy": "energy",
}

#: The kind of a read wiring record that reads one axis of the orbit.
MONITOR = "monitor"


# ===================================================================
# The harvest
# ===================================================================


@dataclass(frozen=True)
class Device:
    """One driven channel of the served view.

    Attributes:
        kind: What the write drives, from :data:`KIND_OF_ATTRIBUTE`.
        setpoint: The address a client writes.
        readback: The address the written field is read back on; the setpoint
            itself for a channel that pairs with no readback of its own.
    """

    kind: str
    setpoint: str
    readback: str


@dataclass(frozen=True)
class BuiltTree:
    """One facility, installed from its export and built.

    Attributes:
        name: The fixture the export came from.
        repo: The deployment repo the recipe built.
        view: The ``variables.json`` of the simulator view the build rendered.
        declared_kinds: The coupling kinds the reviewed mapping declares.
        stopped: What the first build printed to stderr before any remedy.
        remedied: The setpoints whose limits records the harvest widened, in
            the order the build's stops named them.
    """

    name: str
    repo: Path
    view: dict[str, Any]
    declared_kinds: frozenset[str]
    stopped: str
    remedied: tuple[str, ...]

    @property
    def data_root(self) -> Path:
        """The data root the container mounts: the simulator view sits under it."""
        return self.repo / "build" / "data"

    @property
    def simulator_dir(self) -> Path:
        """The simulator view the container's composite serves."""
        return self.data_root / "simulator"

    def channel(self, address: str) -> dict[str, Any]:
        """The view's record of ``address``."""
        for channel in self.view["channels"]:
            if channel["address"] == address:
                return channel
        raise AssertionError(f"{self.name}: the served view lists no channel {address}")

    def wired(self) -> list[str]:
        """Every address a physics model of the view wires, in wiring order."""
        return list(dict.fromkeys(str(record["address"]) for record in physics_wiring(self.view)))

    def devices(self, kind: str) -> tuple[Device, ...]:
        """The view's driven channels of one kind, in wiring order.

        Wiring order is the facility file's record order, so a lane naming a
        slot names one device on every run against a given tree -- which is
        how two lanes driving the same tree are kept off each other's device.
        A monitor is a reading rather than a write, so it is its own readback.
        """
        devices: list[Device] = []
        for record in physics_wiring(self.view):
            address = str(record["address"])
            engine = record.get("engine") or {}
            if record.get("direction") == "read":
                if kind == MONITOR and "axis" in engine and "attribute" not in engine:
                    devices.append(Device(kind=MONITOR, setpoint=address, readback=address))
                continue
            if KIND_OF_ATTRIBUTE.get(str(engine.get("attribute"))) == kind:
                pair = self.channel(address).get("pair") or address
                devices.append(Device(kind=kind, setpoint=address, readback=str(pair)))
        return tuple(devices)

    def band(self, address: str) -> tuple[float, float]:
        """The drive band the served view gives ``address``."""
        bounds = self.channel(address).get("value_range")
        assert bounds is not None and None not in bounds, (
            f"{address} carries no write band in the served view"
        )
        low, high = bounds
        return float(low), float(high)

    def oracle(self) -> Any:
        """A fresh in-process composite over the view the container serves.

        No state directory and no model log: it starts at the view's baseline,
        as the container does, and writes nothing outside this process.
        """
        from osprey_connectors.simulation.composite import Composite

        return Composite(self.simulator_dir, state_dir=None, model_log=False)

    def target(self, device: Device, booted: float) -> float:
        """A hardware value inside ``device``'s band and away from ``booted``.

        Derived from the view's own band and the value the device serves
        rather than chosen here, because a value outside the band is clamped:
        the readback would then be about the limit and not about the write.
        The step is taken in whichever direction the band has room for, and
        it is the RF fraction for a cavity, whose step the lattice bounds more
        tightly than the band does (see ``RF_WRITE_FRACTION``).
        """
        low, high = self.band(device.setpoint)
        fraction = RF_WRITE_FRACTION if device.kind == "rf" else WRITE_FRACTION
        step = abs(booted) * fraction or (high - low) * fraction
        for candidate in (booted + step, booted - step):
            if low < candidate < high:
                return candidate
        raise AssertionError(
            f"{device.setpoint}: neither {booted + step} nor {booted - step} "
            f"fits inside its band [{low}, {high}], so no write can be made that the "
            f"drive limits would not clamp"
        )


def harvest_and_build(name: str, destination: Path) -> BuiltTree:
    """Install fixture ``name`` into ``destination`` and build it.

    The literal recipe the install skill tells an operator to type, driven
    through the real verbs: the deployment is a control-assistant one because
    that is the preset a facility harvest lands on, every refusal is obeyed as
    printed, and the build is the ordinary one -- no flag here tells it to
    treat a served tree differently.

    The exports enter the facility description before ``mml emit`` runs: the
    stop ``facility import mml`` makes over the preset's authored sources is
    obeyed line by line, the tree's reviewed ``imported/mml/mapping.yaml`` is
    installed, and every export goes in one call. The import lists the demo
    scenarios it leaves stale, and exactly those are removed. The first build
    then stops on a setpoint that starts outside its seeded band; ``facility
    validate`` names every such setpoint, the limits records those lines name
    are widened in the deployment, never in the fixture, and the set is held
    against the tree's own.
    """
    recipes = _recipes()
    fixture = recipes.FIXTURES / name
    exports = sorted(str(path) for path in fixture.glob("*.ao.json"))
    assert exports, f"{name} commits a virtual accelerator but no export to harvest"

    runner = CliRunner()
    repo = destination / "deployment"
    recipes.invoke(runner, "init", str(repo), "--preset", "control-assistant", "--no-git")
    recipes.clear_authored(runner, repo, exports)
    imported = recipes.import_facility(runner, repo, exports, recipes.facility_mapping(fixture))
    recipes.remove_stale_scenarios(repo, imported)
    recipes.invoke(runner, "mml", "import", *exports, "--repo", str(repo))
    shutil.copy(fixture / "mapping.yaml", repo / "data" / "mml" / "mapping.yaml")
    recipes.drive_emit(runner, repo)
    recipes.invoke(
        runner,
        "set",
        "--repo",
        str(repo),
        *recipes.MIDDLE_LAYER_SETTINGS,
    )
    recipes.invoke(runner, "validate", "--repo", str(repo), "--drift=warn")
    stopped, remedied, _ = recipes.build_past_the_seed_stops(
        runner, repo, responses=recipes.expected_response_lines(name)
    )
    expected = recipes.expected_seed_stops(name)
    assert len(remedied) == len(expected) and set(remedied) == expected, (
        f"{name}: the build stopped on {sorted(remedied)}, and the tree plants {sorted(expected)}"
    )

    view = json.loads(
        (repo / "build" / "data" / "simulator" / "variables.json").read_text(encoding="utf-8")
    )
    return BuiltTree(
        name=name,
        repo=repo,
        view=view,
        declared_kinds=_declared_kinds(repo / "data" / "mml" / "mapping.yaml"),
        stopped=stopped.stderr,
        remedied=remedied,
    )


def _declared_kinds(mapping: Path) -> frozenset[str]:
    """The coupling kinds the reviewed mapping declares for the served system.

    Read from the copy the install left in the deployment rather than from the
    fixture beside the export, so this is the same document ``mml emit`` bound
    from.

    Only a family whose verdict is ``couple`` contributes its ``kind``. A
    latched family carries a ``slot.kind`` too, but that names the QUESTION
    that was asked about the family -- an unknown ATType, an escape hatch --
    and a question binds nothing.
    """
    block = yaml.safe_load(mapping.read_text(encoding="utf-8"))["virtual_accelerator"]
    return frozenset(
        str(family["kind"])
        for family in block["families"].values()
        if isinstance(family, dict) and family.get("verdict") == "couple" and "kind" in family
    )


# ===================================================================
# The served machine
# ===================================================================


def _docker(*arguments: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *arguments], capture_output=True, text=True, timeout=timeout)


@dataclass(frozen=True)
class ServedTree:
    """A built tree with a container serving it, the port it answers on, and its oracle.

    ``oracle`` is an in-process composite over the same view, given every
    write the container is given, in the same order: what it holds is what
    the served machine owes.
    """

    tree: BuiltTree
    port: int
    oracle: Any

    def call(self, request: dict, *, timeout: float = 600.0) -> dict:
        """Run one Channel Access exchange against this container.

        Out of process, and pointed at this container by PORT through
        name-server mode: a host client dials a port, and the stale address
        variables a developer's shell may carry would otherwise send it to
        whatever else is serving Channel Access on this machine.
        """
        environment = {
            **os.environ,
            "EPICS_CA_NAME_SERVERS": f"localhost:{self.port}",
            "EPICS_CA_AUTO_ADDR_LIST": "NO",
        }
        for stale in ("EPICS_CA_ADDR_LIST", "EPICS_CA_SERVER_PORT", "EPICS_CAS_SERVER_PORT"):
            environment.pop(stale, None)
        result = subprocess.run(
            [sys.executable, __file__, "--worker", json.dumps(request)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"CA worker failed ({request['op']}):\n{result.stdout}\n{result.stderr}"
            )
        return json.loads(result.stdout.strip().splitlines()[-1])

    def read(self, *addresses: str) -> dict[str, Any]:
        answer = self.call({"op": "read", "addresses": list(addresses)})
        assert not answer["failed"], f"unreadable: {answer['failed']}"
        return answer["values"]

    def write_then_read(
        self, writes: list[tuple[str, float]], addresses: list[str]
    ) -> dict[str, Any]:
        """Write over Channel Access, then read; the oracle is given the same writes first."""
        for address, value in writes:
            self.oracle.set({address: value})
        answer = self.call(
            {"op": "write_read", "writes": [list(pair) for pair in writes], "addresses": addresses}
        )
        assert not answer["failed"], f"unreadable after the write: {answer['failed']}"
        return answer["values"]

    def owed(self, *addresses: str) -> dict[str, float]:
        """What the served machine owes at ``addresses``, by the oracle."""
        values = self.oracle.get(list(addresses))
        return {address: float(values[address]) for address in addresses}


@contextmanager
def _serving(tree: BuiltTree):
    """Boot one container over ``tree``'s published data root and wait for it.

    The container reads the instance and the simulator view from the env and
    data root the directory conftest's composite boot uses: the instance
    named, and the build's ``data`` root mounted as ``/data``, whose
    ``simulator/`` is the view the composite serves. Nothing here restates
    what the build derived from this tree, so the boot tests the harvest
    rather than this file's idea of it.
    """

    def container(port: int) -> tuple[str, list[str]]:
        # The port alone does not make the name unique: it is released when the
        # probe socket closes and only bound again by `docker run`, so two runs
        # can reserve the same number. The suffix is what keeps the
        # force-remove from reaching a concurrent run's container.
        name = f"{CONTAINER_PREFIX}-{port}-{uuid.uuid4().hex[:8]}"
        return name, [
            "run",
            "-d",
            "--name",
            name,
            "-e",
            f"EPICS_CA_SERVER_PORT={port}",
            *e2e_conftest.DEMO_NAMESPACE_RUN_ARGS,
            *e2e_conftest.data_root_run_args(tree.data_root),
            "-p",
            f"127.0.0.1:{port}:{port}/tcp",
            IMAGE,
        ]

    port, name = e2e_conftest.run_on_free_port(container)

    served = ServedTree(tree=tree, port=port, oracle=tree.oracle())
    try:
        _wait_until_ready(name, served)
        yield served
    finally:
        _docker("rm", "-f", name)


def _wait_until_ready(container: str, served: ServedTree) -> None:
    """Block until the container serves, or fail naming what it said.

    Readiness is a served ANSWER, not a log line: the readiness marker is
    printed before the first client has ever reached the server, and a
    container whose port never became reachable from the host would sail past
    a log check. The address probed is one the served view itself wires.
    """
    probe = served.tree.wired()[0]
    deadline = time.monotonic() + BOOT_TIMEOUT_S
    # Kept so a boot that never answers says what the client last saw.
    last_attempt = "no read completed"
    while (remaining := deadline - time.monotonic()) > 0:
        bound = min(PROBE_TIMEOUT_S, remaining)
        try:
            answer = served.call(
                {"op": "read", "addresses": [probe], "timeout": min(30.0, bound)}, timeout=bound
            )
            if not answer["failed"] and answer["values"].get(probe) is not None:
                return
            last_attempt = f"{probe} read back as None"
        except Exception as exc:  # "not up yet" is the expected case here
            last_attempt = f"{type(exc).__name__}: {exc}"
        time.sleep(2.0)

    logs = _docker("logs", "--tail", "60", container)
    raise AssertionError(
        f"{served.tree.name}: the container never served {probe} within {BOOT_TIMEOUT_S}s.\n"
        f"The client's last attempt: {last_attempt}\n"
        f"{e2e_conftest.boot_report(container, served.port)}\n"
        f"Container logs:\n{logs.stdout}\n{logs.stderr}"
    )


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="module", params=CRITERION_TREES)
def served(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
    """One tree, harvested, built and served.

    Module-scoped and parametrised, which is what keeps the boots sequential
    within a worker: the previous tree's container is torn down before the next
    tree's harvest begins. Across workers it is the module's
    ``xdist_group`` mark that keeps every lane on one of them, so no two boots
    of this module run at once.

    Whether a tree carries a machine is read off the CLI recipe's own
    discovery, so the boot lane and the recipe lane can never disagree about
    it -- and a tree named here that carries none FAILS. Naming a facility in
    ``CRITERION_TREES`` is the claim that its export reaches a served machine;
    a tree that stops meeting that claim has lost something this suite exists
    to notice.
    """
    name = str(request.param)
    assert name in _recipes().TWO_ZERO_TREES, (
        f"{name} commits no 2.0 export (no *.va.json beside its Accelerator Objects), so it "
        f"carries no machine to serve; re-export it with mml_export 2.0, or drop it from "
        f"CRITERION_TREES if this facility is no longer claimed to boot"
    )
    tree = harvest_and_build(name, tmp_path_factory.mktemp(f"mml-tree-{name}"))
    with _serving(tree) as running:
        yield running


def _agrees_with_the_mapping(tree: BuiltTree, kinds: tuple[str, ...]) -> None:
    """The served view wires each of ``kinds`` exactly when its mapping declares it.

    What every absence below rests on. Which families couple, and as what, is
    the reviewed mapping's answer, and the harvest turns exactly those answers
    into wiring. So a facility that exports no energy knob is a FACT about
    that facility, provable against its own mapping, rather than a lane that
    stops measuring -- and the reverse, a mapping that couples a family whose
    wiring never reached the served view, fails here rather than reading as
    a facility that simply has no such knob.
    """
    for kind in kinds:
        bound = bool(tree.devices(kind))
        declared = kind in tree.declared_kinds
        assert bound == declared, (
            f"{tree.name}: the served view wires {'a' if bound else 'no'} {kind} while its "
            f"reviewed mapping declares {'one' if declared else 'none'}; the mapping decides "
            f"what couples, so the two cannot disagree about a whole kind"
        )


def _pairs_are_coherent(tree: BuiltTree, kinds: tuple[str, ...]) -> None:
    """Every driven channel of ``kinds`` reads back where a model answers.

    A channel that pairs with itself answers on the address it was written
    to. One that names a readback of its own names a channel a physics model
    of the view wires as a reading: a readback the texture answered would
    echo the demand rather than report the field the model took.
    """
    reads = {
        str(record["address"])
        for record in physics_wiring(tree.view)
        if record.get("direction") == "read"
    }
    for kind in kinds:
        for device in tree.devices(kind):
            assert device.readback == device.setpoint or device.readback in reads, (
                f"{tree.name}: {device.setpoint} reads back on {device.readback}, which no "
                f"physics model of the served view wires as a reading"
            )


def _bound_or_absent(
    devices: tuple[Device, ...], tree: BuiltTree, kinds: tuple[str, ...]
) -> Device | None:
    """The first of ``devices``, or ``None`` once the absence is accounted for.

    Either way the tree is first held to its mapping over ``kinds``, so a lane
    that ends in ``None`` ends having proved something about the facility. The
    alternative -- stopping the lane -- reads in a report exactly like a lane
    that ran and found nothing wrong, which is the one thing a report of this
    suite must never be ambiguous about.
    """
    _agrees_with_the_mapping(tree, kinds)
    return devices[0] if devices else None


def _echoing(devices: tuple[Device, ...], *, echoes: bool) -> tuple[Device, ...]:
    """The devices that pair with themselves (``echoes``), or the ones that do not."""
    return tuple(device for device in devices if (device.readback == device.setpoint) == echoes)


def _write_and_hold(served: ServedTree, device: Device) -> None:
    """Write ``device`` once and hold both addresses to what the oracle owes.

    The setpoint must hold the value written; the readback must hold what the
    in-process composite computes for it after the same write -- the written
    value for a channel that pairs with itself, the model's reading of the
    field otherwise.
    """
    booted = float(served.read(device.setpoint)[device.setpoint])
    target = served.tree.target(device, booted)
    addresses = list(dict.fromkeys((device.setpoint, device.readback)))

    values = served.write_then_read([(device.setpoint, target)], addresses)

    assert values[device.setpoint] == pytest.approx(target, rel=READBACK_RTOL)
    owed = served.owed(*addresses)
    assert values[device.readback] == pytest.approx(owed[device.readback], rel=READBACK_RTOL), (
        f"{served.tree.name}: {device.readback} serves {values[device.readback]} after "
        f"{device.setpoint} took {target}, and the composite over the same view holds "
        f"{owed[device.readback]}"
    )


# ===================================================================
# The lanes
# ===================================================================


class TestTheServedTree:
    def test_the_harvest_passed_the_seed_stops_its_tree_plants(self, served: ServedTree) -> None:
        """The first build stopped on one of the tree's own setpoints.

        The build names the first stop and ``facility validate`` names them
        all; the harvest widened the record of each, and that set is the tree's.

        The synthetic tree plants one: the corrector its export starts outside
        its own ``Range``. That stop is the only line the first build printed
        about the facility, and the machine served here is the one built after
        its limits record was widened to hold the nominal.
        """
        recipes = _recipes()
        tree = served.tree
        expected = recipes.expected_seed_stops(tree.name)
        stops = recipes.seed_stops(tree.stopped)

        assert expected, f"{tree.name} plants no stop to pass"
        assert stops and set(stops) <= expected
        assert set(tree.remedied) == expected
        if tree.name == "synthetic":
            (address,) = expected
            facility_lines = [
                line for line in tree.stopped.splitlines() if line.startswith("facility: ")
            ]
            assert len(facility_lines) == 1, facility_lines
            assert facility_lines[0].startswith(f"facility: seed-invalid: channel {address} — ")
            assert stops[address] == ("max_value", 1.5)

    def test_every_wired_channel_is_served(self, served: ServedTree) -> None:
        """Every address the models drive answers a read from the host.

        The wired set is the view's own: every setpoint a physics model takes
        and every reading it answers. A channel in it that does not answer is
        a mount the container could not resolve a model over.
        """
        wired = sorted(served.tree.wired())
        assert wired, f"{served.tree.name} wires no channel at all"

        answer = served.call({"op": "read", "addresses": wired})

        assert not answer["failed"], (
            f"{served.tree.name}: {len(answer['failed'])} of {len(wired)} wired channels "
            f"are not served: {answer['failed']}"
        )
        missing = [address for address, value in answer["values"].items() if value is None]
        assert not missing, f"{served.tree.name}: served with no value: {missing}"

    def test_a_setpoint_carrying_its_own_readback_serves_what_was_written(
        self, served: ServedTree
    ) -> None:
        """A channel that pairs with itself holds the accepted value.

        The answer a read-modify-write client depends on -- a setpoint that
        answered with anything but the value it took would drift such a client
        one write at a time.

        Which channels pair with themselves is not the mapping's to state: it
        follows from the export's own channels, so the served view is the
        source of truth for it. A tree with no such channel therefore ends on
        what can be held against something independent -- its driven kinds
        against the mapping, and its readbacks against the models that answer
        them.

        The device is looked for among strengths, then correctors, then the RF
        frequency, which is the one device of this kind a tree may have. That
        device can be the only one of its kind, so a later lane may drive it
        too: the write is held to moving it, and the value it booted with is
        written back once the answer has been measured.
        """
        kinds = ("strength", "kick", "rf")
        _pairs_are_coherent(served.tree, kinds)
        device = _bound_or_absent(
            _echoing(served.tree.devices("strength"), echoes=True)
            or _echoing(served.tree.devices("kick"), echoes=True)
            or _echoing(served.tree.devices("rf"), echoes=True),
            served.tree,
            kinds,
        )
        if device is None:
            return
        address = device.setpoint
        booted = float(served.read(address)[address])
        target = served.tree.target(device, booted)
        assert booted != pytest.approx(target, rel=READBACK_RTOL), (
            f"{served.tree.name}: {address} already serves {target} before the write"
        )

        values = served.write_then_read([(address, target)], [address])

        assert values[address] == pytest.approx(target, rel=READBACK_RTOL)
        restored = served.write_then_read([(address, booted)], [address])
        assert restored[address] == pytest.approx(booted, rel=READBACK_RTOL)

    def test_a_strength_write_reads_back_through_the_model(self, served: ServedTree) -> None:
        """A strength's own readback is the model's reading of the field it set.

        This is the lane that would catch a readback served as a plain echo:
        the expected value is what the in-process composite computes through
        the readback's own calibration, which for a real export need not land
        near the number that was written.

        A tree whose strengths all pair with themselves ends at the same two
        checks as the lane above, for the same reason.
        """
        _pairs_are_coherent(served.tree, ("strength",))
        device = _bound_or_absent(
            _echoing(served.tree.devices("strength"), echoes=False), served.tree, ("strength",)
        )
        if device is None:
            return
        _write_and_hold(served, device)

    def test_a_kick_write_reads_back_on_the_address_the_view_pairs(
        self, served: ServedTree
    ) -> None:
        """A corrector, written and read back where its channel pairs.

        The readback is held to the composite rather than assumed to be an
        echo: what a facility's correctors read back is the export's answer,
        not this file's.
        """
        device = _bound_or_absent(served.tree.devices("kick"), served.tree, ("kick",))
        if device is None:
            return
        _write_and_hold(served, device)

    def test_a_corrector_write_moves_the_monitors(self, served: ServedTree) -> None:
        """The model is behind the channels.

        A corrector is written and the monitors are read before and after. A
        served echo -- or a container that resolved no model over its mount --
        leaves every monitor exactly where it was, which is the one thing no
        other lane in this file can tell apart from a machine.
        """
        _agrees_with_the_mapping(served.tree, (MONITOR, "kick"))
        monitors = served.tree.devices(MONITOR)
        correctors = served.tree.devices("kick")
        if not monitors or not correctors:
            return
        # The second corrector where the tree has one, so this lane and the
        # kick-readback lane above drive different devices. A tree with a
        # single corrector has none to spare, and measuring its orbit against
        # that one device is worth more than not measuring it at all.
        device = correctors[1] if len(correctors) > 1 else correctors[0]
        addresses = [monitor.setpoint for monitor in monitors]

        before = served.read(*addresses)
        booted = float(served.read(device.setpoint)[device.setpoint])
        after = served.write_then_read(
            [(device.setpoint, served.tree.target(device, booted))], addresses
        )

        moved = [address for address in addresses if before[address] != after[address]]
        assert moved, (
            f"{served.tree.name}: writing {device.setpoint} moved none of "
            f"{len(addresses)} monitors, so no model is behind those channels"
        )

    def test_the_rf_frequency_is_written_and_read_back(self, served: ServedTree) -> None:
        """The cavity knob, where the facility exports one.

        A tree wiring no ``rf`` ends at the mapping check: which knobs a
        facility has is its export's answer, and a lane that demanded one
        everywhere would fail a facility for a machine it does not run.
        """
        device = _bound_or_absent(served.tree.devices("rf"), served.tree, ("rf",))
        if device is None:
            return
        _write_and_hold(served, device)

    def test_the_energy_knob_is_written_and_read_back(self, served: ServedTree) -> None:
        """The dipole, which is the ring's energy rather than an element's field.

        Last in this class on purpose: an energy write rescales every
        rigidity-scaled family's physics value, so it changes what the ring
        does for every other lane's device. The readback lanes above are about
        a written hardware value and would survive it, but the ordering keeps
        the machine each of them ran against the one it booted in.
        """
        device = _bound_or_absent(served.tree.devices("energy"), served.tree, ("energy",))
        if device is None:
            return
        _write_and_hold(served, device)


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_mml_trees_boot.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
