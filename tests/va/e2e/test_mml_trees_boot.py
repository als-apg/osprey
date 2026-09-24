"""A harvested facility tree, served by a real container, over Channel Access.

The container half of success criterion 1: a facility export goes through the
whole install -- ``init``, ``mml import``, the reviewed mapping, ``mml emit``,
``osprey set``, ``validate``, ``osprey build`` -- and what the build published
is handed to the virtual accelerator image, which serves that facility's own
channels with that facility's own ring behind them. Every earlier task in this
feature proves a step of that chain against files; this module is the only one
that proves the chain ends in a machine a control system can talk to.

What each lane asserts, and why it is not vacuous:

* **Every coupled channel is served.** The bindings document says which
  addresses the model drives -- each binding's setpoint and the readback it
  serves where it serves one -- and every one of them answers a Channel Access
  read from the host. A tree whose manifest and bindings disagree serves a
  coupled channel nothing binds, which the entrypoint refuses; a tree the mount
  got wrong serves the channels with no physics behind them. Reading the whole
  claimed set is what separates those from a machine.
* **A write comes back the way the binding says it does.** The three readback
  rules are three different answers to one client's ``caput`` --
  ``same_as_setpoint`` serves the written value on the address it was written
  to, ``identity`` repeats it on a second address, and ``inverse`` serves it
  mapped through the facility's own ``monitor_inverse``. The expected value for
  an inverse is computed here through the product's own calibration functions
  over the served document, never by restating a number: a lane that pasted one
  would pass against a calibration nobody exported.
* **The ring is really behind the channels.** A corrector write moves the
  monitors. Nothing else in this file could distinguish a served model from a
  well-formed echo.

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
import socket
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

from osprey.services.virtual_accelerator.bindings import (  # noqa: E402
    Binding,
    BindingsDocument,
    load_bindings,
)
from osprey.services.virtual_accelerator.lattice.calibration import (  # noqa: E402
    to_hardware,
    to_physics,
)
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths  # noqa: E402
from tests.e2e._orm_stack import claimed_addresses  # noqa: E402
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
# Seven lanes over three trees; the guard test itself is the twenty-second
# item, so a floor of 21 reds on the loss of a single lane.
MIN_COLLECTED_TESTS = 21

#: The image under test -- the same one the rest of this directory serves from.
IMAGE = e2e_conftest.IMAGE

#: Container-name prefix. The run's own port and a random suffix are appended,
#: because a name shared with a concurrent run is destructive rather than tidy:
#: each run force-removes its own name as stale cleanup.
CONTAINER_PREFIX = "osprey-va-e2e-mml-tree"

BOOT_TIMEOUT_S = 240.0

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

#: Tolerance for a readback that came back through a calibration. Generous
#: against the wire (Channel Access serves a double, and the IOC's display
#: precision does not enter a ``caget``), tight against the thing being tested:
#: a readback served through the wrong curve, or through no curve at all, is
#: wrong by orders of magnitude, not by parts in a billion.
READBACK_RTOL = 1e-9

#: The three answers a served binding can give the client that writes to it.
#: A document naming anything else describes a machine the model cannot serve.
READBACK_RULES = ("same_as_setpoint", "identity", "inverse")


# ===================================================================
# The harvest
# ===================================================================


@dataclass(frozen=True)
class BuiltTree:
    """One facility, installed from its export and built.

    Attributes:
        name: The fixture the export came from.
        repo: The deployment repo the recipe built.
        env: The deployment ``.env`` the build appended its derived keys to.
        document: The bindings the build published into the served directory.
        limits: The write bands of that same directory.
        declared_kinds: The coupling kinds the reviewed mapping declares.
    """

    name: str
    repo: Path
    env: dict[str, str]
    document: BindingsDocument
    limits: dict[str, Any]
    declared_kinds: frozenset[str]

    @property
    def served_dir(self) -> Path:
        """The directory the container is pointed at."""
        return ManifestPaths(data_root=self.repo / "build" / "data").machine_json.parent

    def band(self, address: str) -> tuple[float, float]:
        """The drive band the served tree gives ``address``."""
        entry = self.limits.get(address)
        assert isinstance(entry, dict), f"{address} carries no write band in the served tree"
        return float(entry["min_value"]), float(entry["max_value"])

    def target(self, binding: Binding) -> float:
        """A hardware value inside ``binding``'s band and away from its nominal.

        Derived from the tree's own band and the device's own nominal rather
        than chosen here, because a value outside the band is clamped by the
        IOC: the readback would then be about the limit and not about the write.
        The step is taken in whichever direction the band has room for, and
        it is the RF fraction for a cavity, whose step the ring bounds more
        tightly than the band does (see ``RF_WRITE_FRACTION``).
        """
        low, high = self.band(binding.setpoint_address)
        nominal = binding.nominal
        assert nominal is not None, f"{binding.setpoint_address} carries no nominal"
        fraction = RF_WRITE_FRACTION if binding.kind == "rf" else WRITE_FRACTION
        step = abs(nominal) * fraction or (high - low) * fraction
        for candidate in (nominal + step, nominal - step):
            if low < candidate < high:
                return candidate
        raise AssertionError(
            f"{binding.setpoint_address}: neither {nominal + step} nor {nominal - step} "
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
    """
    recipes = _recipes()
    fixture = recipes.FIXTURES / name
    exports = sorted(str(path) for path in fixture.glob("*.ao.json"))
    assert exports, f"{name} commits a virtual accelerator but no export to harvest"

    runner = CliRunner()
    repo = destination / "deployment"
    recipes.invoke(runner, "init", str(repo), "--preset", "control-assistant", "--no-git")
    recipes.invoke(runner, "mml", "import", *exports, "--repo", str(repo))
    shutil.copy(fixture / "mapping.yaml", repo / "data" / "mml" / "mapping.yaml")
    recipes.drive_emit(runner, repo)
    recipes.invoke(
        runner,
        "set",
        "--repo",
        str(repo),
        *recipes.served_settings(recipes.facility_prefix(fixture)),
    )
    recipes.invoke(runner, "validate", "--repo", str(repo), "--drift=warn")
    recipes.invoke(runner, "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")

    paths = ManifestPaths(data_root=repo / "build" / "data")
    return BuiltTree(
        name=name,
        repo=repo,
        env=recipes.env_values(repo),
        document=load_bindings(paths.va_bindings),
        limits=json.loads(
            (paths.machine_json.parent / "channel_limits.json").read_text(encoding="utf-8")
        ),
        declared_kinds=_declared_kinds(repo / "data" / "mml" / "mapping.yaml"),
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


def _free_port() -> int:
    """A host port that was free at reservation time."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _docker(*arguments: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *arguments], capture_output=True, text=True, timeout=timeout)


@dataclass(frozen=True)
class ServedTree:
    """A built tree with a container serving it, and the port it answers on."""

    tree: BuiltTree
    port: int

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
        answer = self.call(
            {"op": "write_read", "writes": [list(pair) for pair in writes], "addresses": addresses}
        )
        assert not answer["failed"], f"unreadable after the write: {answer['failed']}"
        return answer["values"]


@contextmanager
def _serving(tree: BuiltTree):
    """Boot one container over ``tree``'s published data root and wait for it.

    The mount is the ROOT the build published, with ``VA_DATA_DIR`` naming the
    served directory inside it, because the model is resolved against the whole
    tree: the lattice and the bindings under the served directory, the write
    bands its variables are built from beside it at the root.

    The namespace and the ring are taken from the deployment's own ``.env``,
    which is where ``osprey build`` recorded what it derived from this tree.
    Naming them here instead would test this file's idea of the harvest rather
    than the harvest.
    """
    port = _free_port()
    # The port alone does not make the name unique: it is released when the
    # probe socket closes and only bound again by `docker run`, so two runs can
    # reserve the same number. The suffix is what keeps the force-remove below
    # from reaching a concurrent run's container.
    name = f"{CONTAINER_PREFIX}-{port}-{uuid.uuid4().hex[:8]}"
    _docker("rm", "-f", name)

    started = _docker(
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"EPICS_CA_SERVER_PORT={port}",
        "-e",
        f"VA_CHANNELS_FILE={tree.env['VA_CHANNELS_FILE']}",
        "-e",
        f"VA_LATTICE={tree.env['VA_LATTICE']}",
        *e2e_conftest.data_root_run_args(tree.served_dir),
        "-p",
        f"127.0.0.1:{port}:{port}/tcp",
        IMAGE,
    )
    if started.returncode != 0:
        raise RuntimeError(f"docker run failed: {started.stdout}\n{started.stderr}")

    served = ServedTree(tree=tree, port=port)
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
    a log check. The address probed is one the served tree itself names.
    """
    probe = served.tree.document.bindings[0].setpoint_address
    deadline = time.monotonic() + BOOT_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            if served.read(probe)[probe] is not None:
                return
        except Exception:  # "not up yet" is the expected case here
            pass
        time.sleep(2.0)

    logs = _docker("logs", "--tail", "60", container)
    raise AssertionError(
        f"{served.tree.name}: the container never served {probe} within {BOOT_TIMEOUT_S}s. "
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


def _of_kind(tree: BuiltTree, kind: str, readback: str | None = None) -> tuple[Binding, ...]:
    """The tree's bindings of one kind, in document order.

    Document order is the order the facility exported its devices in, so a lane
    naming a slot names one device on every run against a given tree -- which
    is how two lanes driving the same tree are kept off each other's device.
    """
    return tuple(
        binding
        for binding in tree.document.bindings
        if binding.kind == kind and (readback is None or binding.readback == readback)
    )


def _agrees_with_the_mapping(tree: BuiltTree, kinds: tuple[str, ...]) -> None:
    """The served tree binds each of ``kinds`` exactly when its mapping declares it.

    What every absence below rests on. Which families couple, and as what, is
    the reviewed mapping's answer; ``mml emit`` turns exactly those answers
    into bindings. So a facility that exports no energy knob is a FACT about
    that facility, provable against its own mapping, rather than a lane that
    stops measuring -- and the reverse, a mapping that couples a family whose
    bindings never reached the served tree, fails here rather than reading as
    a facility that simply has no such knob.
    """
    for kind in kinds:
        bound = bool(_of_kind(tree, kind))
        declared = kind in tree.declared_kinds
        assert bound == declared, (
            f"{tree.name}: the served tree binds {'a' if bound else 'no'} {kind} while its "
            f"reviewed mapping declares {'one' if declared else 'none'}; the mapping decides "
            f"what couples, so the two cannot disagree about a whole kind"
        )


def _rules_are_coherent(tree: BuiltTree, kinds: tuple[str, ...]) -> None:
    """Every binding of ``kinds`` serves a known rule, on the address it names.

    Two facts in one walk. A rule outside the three is a document no model
    could serve. And ``same_as_setpoint`` is the one rule that answers on the
    address it was written to, so it is also the one rule that owes no second
    address: a binding that carries one anyway serves its readback where no
    rule reaches, and one that carries none under another rule names nowhere
    to read.

    Written for the driven kinds only. A monitor reads a physics quantity and
    is served through its inverse on the address it was written to, so it is
    the one kind for which the second half does not hold.
    """
    for binding in tree.document.bindings:
        if binding.kind not in kinds:
            continue
        assert binding.readback in READBACK_RULES, (
            f"{tree.name}: {binding.setpoint_address} serves {binding.readback!r}, which is "
            f"none of the three rules a client can be written against ({READBACK_RULES})"
        )
        owes_an_address = binding.readback != "same_as_setpoint"
        assert (binding.readback_address is not None) == owes_an_address, (
            f"{tree.name}: {binding.setpoint_address} serves {binding.readback!r} and "
            f"{'names no' if owes_an_address else 'also names a'} readback address"
        )


def _bound_or_absent(
    bindings: tuple[Binding, ...], tree: BuiltTree, kinds: tuple[str, ...]
) -> Binding | None:
    """The first of ``bindings``, or ``None`` once the absence is accounted for.

    Either way the tree is first held to its mapping over ``kinds``, so a lane
    that ends in ``None`` ends having proved something about the facility. The
    alternative -- stopping the lane -- reads in a report exactly like a lane
    that ran and found nothing wrong, which is the one thing a report of this
    suite must never be ambiguous about.
    """
    _agrees_with_the_mapping(tree, kinds)
    return bindings[0] if bindings else None


def _expected_inverse(binding: Binding, written: float) -> float:
    """What an ``inverse`` readback owes for ``written``, through the real curves.

    ``monitor_inverse(calibration(written))``, evaluated by the product's own
    conversions over the served document. The two curves are independent
    exported data -- the inverse is not the calibration read backwards -- so the
    only way to state this expectation without restating the facility's
    numbers is to run the same two conversions the model runs.
    """
    assert binding.calibration is not None and binding.monitor_inverse is not None
    physics = to_physics(binding.calibration, written)
    return float(to_hardware(binding.monitor_inverse, physics))


# ===================================================================
# The lanes
# ===================================================================


class TestTheServedTree:
    def test_every_coupled_channel_the_bindings_claim_is_served(self, served: ServedTree) -> None:
        """Every address the model drives answers a read from the host.

        The claimed set is the document's own: each binding's setpoint plus the
        readback it serves where it serves one. A channel in it that does not
        answer is a tree whose manifest and bindings came from different runs,
        or a mount the container could not resolve a model over.
        """
        claimed = sorted(claimed_addresses(served.tree.document))
        assert claimed, f"{served.tree.name} claims no channel at all"

        answer = served.call({"op": "read", "addresses": claimed})

        assert not answer["failed"], (
            f"{served.tree.name}: {len(answer['failed'])} of {len(claimed)} coupled channels "
            f"are not served: {answer['failed']}"
        )
        missing = [address for address, value in answer["values"].items() if value is None]
        assert not missing, f"{served.tree.name}: served with no value: {missing}"

    def test_a_setpoint_carrying_its_own_readback_serves_what_was_written(
        self, served: ServedTree
    ) -> None:
        """``same_as_setpoint``: one address, and it holds the accepted value.

        The rule a read-modify-write client depends on -- a setpoint that
        answered with anything but the value it took would drift such a client
        one write at a time.

        Which rule a device serves is not the mapping's to state: it follows
        from the export's own channels and curves, so the emitted document is
        the source of truth for it. A tree serving this rule nowhere therefore
        ends on what can be held against something independent -- its driven
        kinds against the mapping, and its rules against the three a client can
        be written for.

        The device is looked for among strengths, then correctors, then the RF
        frequency, which is the one device of this rule a tree may have. That
        device can be the only one of its kind, so a later lane may drive it
        too: the write is held to moving it, and the value it booted with is
        written back once the rule has been measured.
        """
        kinds = ("strength", "kick", "rf")
        _rules_are_coherent(served.tree, kinds)
        binding = _bound_or_absent(
            _of_kind(served.tree, "strength", "same_as_setpoint")
            or _of_kind(served.tree, "kick", "same_as_setpoint")
            or _of_kind(served.tree, "rf", "same_as_setpoint"),
            served.tree,
            kinds,
        )
        if binding is None:
            return
        target = served.tree.target(binding)
        address = binding.setpoint_address
        booted = served.read(address)[address]
        assert booted != pytest.approx(target, rel=READBACK_RTOL), (
            f"{served.tree.name}: {address} already serves {target} before the write"
        )

        values = served.write_then_read([(address, target)], [address])

        assert values[address] == pytest.approx(target, rel=READBACK_RTOL)
        restored = served.write_then_read([(address, booted)], [address])
        assert restored[address] == pytest.approx(booted, rel=READBACK_RTOL)

    def test_a_strength_write_reads_back_through_the_exported_inverse(
        self, served: ServedTree
    ) -> None:
        """``inverse``: the readback is the written value through both curves.

        This is the lane that would catch a readback served as a plain echo:
        the expected value is computed through the facility's own calibration
        and its own ``monitor_inverse``, which for a real export land nowhere
        near the number that was written.

        A tree whose strengths all collapse to another rule ends at the same
        two checks as the lane above, for the same reason.
        """
        _rules_are_coherent(served.tree, ("strength",))
        binding = _bound_or_absent(
            _of_kind(served.tree, "strength", "inverse"), served.tree, ("strength",)
        )
        if binding is None:
            return
        assert binding.readback_address is not None
        target = served.tree.target(binding)

        values = served.write_then_read(
            [(binding.setpoint_address, target)],
            [binding.setpoint_address, binding.readback_address],
        )

        assert values[binding.setpoint_address] == pytest.approx(target, rel=READBACK_RTOL)
        assert values[binding.readback_address] == pytest.approx(
            _expected_inverse(binding, target), rel=READBACK_RTOL
        )

    def test_a_kick_write_reads_back_on_the_address_the_document_names(
        self, served: ServedTree
    ) -> None:
        """A corrector, written and read back per its own rule.

        The kick is asserted through the same three-rule branch as everything
        else rather than assumed to be an echo: which rule a facility's
        correctors use is the export's answer, not this file's.
        """
        binding = _bound_or_absent(_of_kind(served.tree, "kick"), served.tree, ("kick",))
        if binding is None:
            return
        target = served.tree.target(binding)
        addresses = [binding.setpoint_address]
        if binding.readback_address is not None:
            addresses.append(binding.readback_address)

        values = served.write_then_read([(binding.setpoint_address, target)], addresses)

        assert values[binding.setpoint_address] == pytest.approx(target, rel=READBACK_RTOL)
        expected = _expected_inverse(binding, target) if binding.readback == "inverse" else target
        served_at = binding.readback_address or binding.setpoint_address
        assert values[served_at] == pytest.approx(expected, rel=READBACK_RTOL)

    def test_a_corrector_write_moves_the_monitors(self, served: ServedTree) -> None:
        """The ring is behind the channels.

        A corrector is written and the monitors are read before and after. A
        served echo -- or a container that resolved no model over its mount --
        leaves every monitor exactly where it was, which is the one thing no
        other lane in this file can tell apart from a machine.
        """
        _agrees_with_the_mapping(served.tree, ("monitor", "kick"))
        monitors = _of_kind(served.tree, "monitor")
        correctors = _of_kind(served.tree, "kick")
        if not monitors or not correctors:
            return
        # The second corrector where the tree has one, so this lane and the
        # kick-readback lane above drive different devices. A tree with a
        # single corrector has none to spare, and measuring its orbit against
        # that one device is worth more than not measuring it at all.
        binding = correctors[1] if len(correctors) > 1 else correctors[0]
        addresses = [monitor.setpoint_address for monitor in monitors]

        before = served.read(*addresses)
        after = served.write_then_read(
            [(binding.setpoint_address, served.tree.target(binding))], addresses
        )

        moved = [address for address in addresses if before[address] != after[address]]
        assert moved, (
            f"{served.tree.name}: writing {binding.setpoint_address} moved none of "
            f"{len(addresses)} monitors, so nothing behind those channels is a ring"
        )

    def test_the_rf_frequency_is_written_and_read_back(self, served: ServedTree) -> None:
        """The cavity knob, where the facility exports one.

        A tree binding no ``rf`` ends at the mapping check: which knobs a
        facility has is its export's answer, and a lane that demanded one
        everywhere would fail a facility for a machine it does not run.
        """
        binding = _bound_or_absent(_of_kind(served.tree, "rf"), served.tree, ("rf",))
        if binding is None:
            return
        target = served.tree.target(binding)
        served_at = binding.readback_address or binding.setpoint_address

        values = served.write_then_read(
            [(binding.setpoint_address, target)], [binding.setpoint_address, served_at]
        )

        expected = _expected_inverse(binding, target) if binding.readback == "inverse" else target
        assert values[binding.setpoint_address] == pytest.approx(target, rel=READBACK_RTOL)
        assert values[served_at] == pytest.approx(expected, rel=READBACK_RTOL)

    def test_the_energy_knob_is_written_and_read_back(self, served: ServedTree) -> None:
        """The dipole, which is the ring's energy rather than an element's field.

        Last in this class on purpose: an energy write rescales every
        rigidity-scaled family's physics value, so it changes what the ring
        does for every other lane's device. The readback lanes above are about
        a written hardware value and would survive it, but the ordering keeps
        the machine each of them ran against the one it booted in.
        """
        binding = _bound_or_absent(_of_kind(served.tree, "energy"), served.tree, ("energy",))
        if binding is None:
            return
        target = served.tree.target(binding)
        served_at = binding.readback_address or binding.setpoint_address

        values = served.write_then_read(
            [(binding.setpoint_address, target)], [binding.setpoint_address, served_at]
        )

        expected = _expected_inverse(binding, target) if binding.readback == "inverse" else target
        assert values[binding.setpoint_address] == pytest.approx(target, rel=READBACK_RTOL)
        assert values[served_at] == pytest.approx(expected, rel=READBACK_RTOL)


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_mml_trees_boot.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
