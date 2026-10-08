"""The real-facility lane: one export imported, built and booted, outside the repo.

Every other lane in this suite runs on a committed fixture. A fixture is a
machine somebody invented to exercise a rule, so a fixture that passes says the
rule is self-consistent, not that it survives contact with a real middle layer.
That contact is what this file buys, and it buys it without bringing the
facility into the repository: the export and its reviewed mapping stay on the
machine that has them, and no facility constant, family name, count or address
enters osprey or its committed tests. So nothing here is pinned to a number or
names a line: every assertion runs over every model the facility file holds.

The lane is the install a deployer types, on a control-assistant deployment:

* ``init --preset control-assistant`` -> ``facility import mml`` past the stop
  it makes over the preset's authored sources -> the reviewed
  ``imported/mml/mapping.yaml`` -> ``facility import mml`` -> the stale
  scenarios it lists removed exactly as listed;
* the isolation settings below -> ``build`` past the ``seed-invalid`` stops the
  imported tree carries, widening exactly the limits records ``facility
  validate`` names;
* the isolation preflight over the rendered config -> the virtual accelerator
  image booted over the build's data root, on a loopback port -> every wired
  channel of every model read over Channel Access. Nothing is written.

Discovery, in order; the first variable that is set decides:

* ``OSPREY_ALS_PROFILES`` names the facility's profiles checkout:
  ``data/facility/identity.yaml`` and ``data/facility/imported/mml/mapping.yaml``
  under it, and every ``*.ao.json`` export anywhere in it;
* ``OSPREY_ALS_MML_EXPORT`` and ``OSPREY_ALS_MML_MAPPING`` name one export and
  the reviewed ``imported/mml/mapping.yaml`` file directly;
* ``OSPREY_ALS_LANE_STAND_IN`` names a directory holding exports and
  ``imported/mml/mapping.yaml``. Test-only: a green run proves this file's code
  path, never anything about the facility.

With none of them set the lane skips, naming them. A set variable that names a
path that is not there fails before anything is built, naming that path.

Isolation. The deployment is pointed at the virtual accelerator alone before it
is rendered: ``control_system.type: virtual_accelerator``,
``control_system.writes_enabled: false``, the connector table stated whole as
one ``virtual_accelerator`` block whose gateways are ``127.0.0.1:<VA port>``
(so no ``epics``, ``tango`` or ``doocs`` block survives), and no stand-in. The
rendered config is then held to it: every configured target resolves to a
simulated type, every gateway it states is loopback, and no target is armed.
Channel Access runs only in a client subprocess whose environment names the VA
alone, and the container publishes its port on 127.0.0.1.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def _worker(request: dict) -> dict:
    """Read each address over Channel Access and report what it served.

    Reads only: this process has no write operation to run. Values are read
    off the wire (``use_monitor=False``) so a cached value cannot stand in for
    a served one.
    """
    import epics

    timeout = float(request.get("timeout", 30.0))
    values: dict[str, Any] = {}
    failed: dict[str, str] = {}
    for address in request["addresses"]:
        try:
            pv = epics.PV(address, connection_timeout=timeout)
            if not pv.wait_for_connection(timeout=timeout):
                raise RuntimeError(f"{address} never connected")
            values[address] = pv.get(use_monitor=False, timeout=timeout)
        except Exception as error:  # reported to the test, not raised here
            failed[address] = f"{type(error).__name__}: {error}"
    return {"values": values, "failed": failed}


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--worker":
    # Before the heavy imports below, so the CA client process carries pyepics
    # and nothing else.
    print(json.dumps(_worker(json.loads(sys.argv[2])), default=str), flush=True)
    os._exit(0)

import ipaddress  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from collections.abc import Iterator, Sequence  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import yaml  # noqa: E402
from click.testing import CliRunner, Result  # noqa: E402

#: The facility's profiles checkout.
ALS_PROFILES_ENV = "OSPREY_ALS_PROFILES"

#: One export, named directly.
ALS_EXPORT_ENV = "OSPREY_ALS_MML_EXPORT"

#: The reviewed ``imported/mml/mapping.yaml`` for that export, named directly.
ALS_MAPPING_ENV = "OSPREY_ALS_MML_MAPPING"

#: A test-only stand-in directory: exports plus ``imported/mml/mapping.yaml``.
STAND_IN_ENV = "OSPREY_ALS_LANE_STAND_IN"

#: Every variable the lane reads, in discovery order.
LANE_VARIABLES = (ALS_PROFILES_ENV, ALS_EXPORT_ENV, ALS_MAPPING_ENV, STAND_IN_ENV)

#: Where a profiles checkout keeps its facility sources.
FACILITY_DIR = Path("data") / "facility"

#: The file whose presence makes a directory a facility checkout.
IDENTITY_FILE = "identity.yaml"

#: The suffix of an export's Accelerator Objects file, the one
#: ``facility import mml`` is handed.
AO_SUFFIX = ".ao.json"

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 25


def _mapping_file() -> str:
    """The reviewed mapping's place under a facility directory."""
    from osprey.facility.layers.mml.mapping import MAPPING_FILE

    return MAPPING_FILE


# ===================================================================
# Discovery
# ===================================================================


@dataclass(frozen=True)
class Lane:
    """The inputs one run of the lane installs.

    Attributes:
        label: What the tree is, for a failure to name.
        exports: The export files handed to ``facility import mml``.
        mapping: The reviewed mapping ``facility import mml`` reads.
    """

    label: str
    exports: tuple[str, ...]
    mapping: Path


@dataclass(frozen=True)
class Unset:
    """No lane variable is set: the lane skips with this reason."""

    reason: str


@dataclass(frozen=True)
class Broken:
    """A set variable names something that is not there: the lane fails."""

    reason: str


def _file(variable: str, path: Path) -> Broken | None:
    if path.is_file():
        return None
    return Broken(f"{variable} is set, and {path} is not a file")


def _from_profiles(named: str) -> Lane | Broken:
    root = Path(named).expanduser().resolve()
    facility = root / FACILITY_DIR
    for path in (facility / IDENTITY_FILE, facility / _mapping_file()):
        if (broken := _file(ALS_PROFILES_ENV, path)) is not None:
            return broken
    exports = sorted(str(path) for path in root.rglob(f"*{AO_SUFFIX}"))
    if not exports:
        return Broken(f"{ALS_PROFILES_ENV} is set, and {root} holds no *{AO_SUFFIX} export")
    return Lane(
        label=f"the profiles checkout {root}",
        exports=tuple(exports),
        mapping=facility / _mapping_file(),
    )


def _from_named(named_export: str | None, named_mapping: str | None) -> Lane | Broken:
    for variable, value in ((ALS_EXPORT_ENV, named_export), (ALS_MAPPING_ENV, named_mapping)):
        if not value:
            other = ALS_MAPPING_ENV if variable == ALS_EXPORT_ENV else ALS_EXPORT_ENV
            return Broken(f"{other} is set, and {variable} is not; the lane reads them as a pair")
    export = Path(str(named_export)).expanduser().resolve()
    mapping = Path(str(named_mapping)).expanduser().resolve()
    for variable, path in ((ALS_EXPORT_ENV, export), (ALS_MAPPING_ENV, mapping)):
        if (broken := _file(variable, path)) is not None:
            return broken
    return Lane(label=f"the export {export.name}", exports=(str(export),), mapping=mapping)


def _from_stand_in(named: str) -> Lane | Broken:
    tree = Path(named).expanduser().resolve()
    if not tree.is_dir():
        return Broken(f"{STAND_IN_ENV} is set, and {tree} is not a directory")
    mapping = tree / _mapping_file()
    if (broken := _file(STAND_IN_ENV, mapping)) is not None:
        return broken
    exports = sorted(str(path) for path in tree.glob(f"*{AO_SUFFIX}"))
    if not exports:
        return Broken(f"{STAND_IN_ENV} is set, and {tree} holds no *{AO_SUFFIX} export")
    return Lane(label=f"the stand-in tree {tree.name}", exports=tuple(exports), mapping=mapping)


def resolve(environ: Any = None) -> Lane | Unset | Broken:
    """The lane this machine can run, the reason it runs none, or what is missing.

    Args:
        environ: The environment to read; ``os.environ`` when omitted.
    """
    environ = os.environ if environ is None else environ
    if environ.get(ALS_PROFILES_ENV):
        return _from_profiles(environ[ALS_PROFILES_ENV])
    if environ.get(ALS_EXPORT_ENV) or environ.get(ALS_MAPPING_ENV):
        return _from_named(environ.get(ALS_EXPORT_ENV), environ.get(ALS_MAPPING_ENV))
    if environ.get(STAND_IN_ENV):
        return _from_stand_in(environ[STAND_IN_ENV])
    return Unset(
        f"{ALS_PROFILES_ENV}, {ALS_EXPORT_ENV} with {ALS_MAPPING_ENV}, and {STAND_IN_ENV} "
        "are not set; the lane installs the export one of them names"
    )


RESOLVED = resolve()

#: The mark on every test that installs the facility.
needs_lane = pytest.mark.skipif(
    isinstance(RESOLVED, Unset),
    reason=RESOLVED.reason if isinstance(RESOLVED, Unset) else "",
)

pytestmark = [pytest.mark.requires_als_profiles]


# ===================================================================
# Isolation
# ===================================================================

#: The connector types the lane accepts a configured target resolving to.
SIMULATED_TYPES = frozenset({"mock", "virtual_accelerator", "live_standin"})

#: The connector blocks that address a facility's own machine.
REAL_BLOCKS = ("epics", "tango", "doocs")

LOOPBACK_HOST = "127.0.0.1"


def isolation_settings(port: int, probe: str) -> list[str]:
    """The ``osprey set`` arguments that point a deployment at the VA alone.

    The connector table is stated whole, so every block the preset or the
    profile carried -- the facility's own among them -- is replaced by one
    ``virtual_accelerator`` block on ``127.0.0.1:<port>`` that probes *probe*,
    a channel of the imported facility. The stand-in is a second machine with
    an address the build derives, so it is switched off.
    """
    gateway = {"address": LOOPBACK_HOST, "port": port, "use_name_server": True}
    connector = {
        "virtual_accelerator": {
            "gateways": {"read_only": gateway, "write_access": gateway},
            "probe_channel": probe,
        }
    }
    return [
        "connector=virtual_accelerator",
        "config.control_system.writes_enabled=false",
        f"config.control_system.connector={json.dumps(connector)}",
        f"virtual_accelerator.port={port}",
        "virtual_accelerator.live_standin=null",
    ]


def _is_loopback(address: Any) -> bool:
    hosts = str(address or "").split()
    if not hosts:
        return False
    for host in hosts:
        name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
        if name == "localhost":
            continue
        try:
            if not ipaddress.ip_address(name).is_loopback:
                return False
        except ValueError:
            return False
    return True


def isolation_errors(section: Any) -> list[str]:
    """Everything in a rendered ``control_system`` section that could reach a real machine.

    Every target a session here can be pointed at is resolved the way the
    runtime resolves it, and each must land on a simulated type whose stated
    gateways -- ``read_only``, ``write_access`` and a ``pva_gateway`` beside
    them -- are all loopback. A Channel Access type must state both gateways:
    one that states none dials whatever the environment names. No target may
    be armed for writes.

    Returns:
        One line per violation; empty when the section is isolated.
    """
    from osprey_connectors.types import (
        any_target_writes_enabled,
        configured_targets,
        resolve_target,
    )

    if not isinstance(section, dict):
        return [f"control_system is {section!r}, not a mapping"]
    errors: list[str] = []
    if section.get("type") != "virtual_accelerator":
        errors.append(f"control_system.type is {section.get('type')!r}, not virtual_accelerator")
    connector = section.get("connector") or {}
    errors += [
        f"control_system.connector.{name} survived the isolation"
        for name in REAL_BLOCKS
        if name in connector
    ]
    for target in configured_targets(section):
        try:
            kind = resolve_target(section, target)
        except ValueError as error:
            errors.append(f"target {target} does not resolve: {error}")
            continue
        if kind not in SIMULATED_TYPES:
            errors.append(f"target {target} resolves to {kind}, a real machine")
            continue
        block = connector.get(kind) or {}
        gateways = dict(block.get("gateways") or {})
        if kind != "mock":
            errors += [
                f"control_system.connector.{kind}.gateways.{lane} is not stated"
                for lane in ("read_only", "write_access")
                if lane not in gateways
            ]
        if block.get("pva_gateway"):
            gateways["pva_gateway"] = block["pva_gateway"]
        errors += [
            f"control_system.connector.{kind} {lane} dials {gateway.get('address')!r}, "
            "which is not loopback"
            for lane, gateway in sorted(gateways.items())
            if not _is_loopback((gateway or {}).get("address"))
        ]
    if any_target_writes_enabled(section):
        errors.append("a configured target is armed for writes")
    return errors


def _isolated_section(port: int = 45000) -> dict[str, Any]:
    gateway = {"address": LOOPBACK_HOST, "port": port, "use_name_server": True}
    return {
        "type": "virtual_accelerator",
        "writes_enabled": False,
        "connector": {
            "virtual_accelerator": {"gateways": {"read_only": gateway, "write_access": gateway}}
        },
    }


# ===================================================================
# The install
# ===================================================================

#: The first line of the stop ``facility import mml`` prints over authored
#: record sources; one ``rm`` line per file follows it.
AUTHORED_PRESENT = "import mml: authored-present: "

#: The header ``facility import mml`` prints over the scenario files a clean
#: import leaves stale; one indented ``rm`` line per path follows it.
STALE_SCENARIOS = "these scenario files name channels that no longer exist:"

#: The line a build stage prints for a setpoint that starts outside its band.
SEED_INVALID = re.compile(
    r"^facility: seed-invalid: channel (?P<address>.+?) — nominal (?P<nominal>\S+) lies "
    r"(?P<side>above|below) `(?P<edge>min_value|max_value)` \S+; "
    r"fix: .*widen the limits record$"
)

BUILD = ("build", "--skip-deps", "--skip-lifecycle")


def _invoke(runner: CliRunner, *args: str) -> Result:
    from osprey.cli.main import cli

    return runner.invoke(cli, list(args), catch_exceptions=False)


def _ok(runner: CliRunner, *args: str) -> Result:
    result = _invoke(runner, *args)
    assert result.exit_code == 0, f"osprey {' '.join(args)} failed:\n{result.output}"
    return result


def _remove_named(repo: Path, line: str) -> list[str]:
    """Delete exactly the paths one printed ``rm`` line names."""
    named = [word for word in shlex.split(line) if word not in ("rm", "-r")]
    assert named, line
    for relative in named:
        target = repo / relative
        assert target.exists(), f"{relative} was named but is not there"
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    return named


def _widen(limits: Path, address: str, edge: str, nominal: float) -> None:
    """Move one edge of one limits record out to hold *nominal*; change no other line."""
    value = float(math.ceil(nominal) if edge == "max_value" else math.floor(nominal))
    lines = limits.read_text(encoding="utf-8").split("\n")
    starts = [
        index
        for index, line in enumerate(lines)
        if line.startswith("- address:") and yaml.safe_load(line[2:])["address"] == address
    ]
    assert len(starts) == 1, f"{address} has {len(starts)} limits records in {limits}"
    end = next(
        (index for index in range(starts[0] + 1, len(lines)) if not lines[index].startswith("  ")),
        len(lines),
    )
    edges = [index for index in range(starts[0] + 1, end) if lines[index].startswith(f"  {edge}:")]
    assert len(edges) == 1, f"{address} states {edge} {len(edges)} times in {limits}"
    lines[edges[0]] = f"  {edge}: {value}"
    limits.write_text("\n".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class Install:
    """One facility, imported, isolated and built.

    Attributes:
        lane: The inputs it was installed from.
        repo: The deployment repo.
        port: The loopback port the VA is configured on.
        cleared: The authored sources the first import named and were removed.
        stale: The scenario paths the clean import listed and were removed.
        widened: The setpoints whose limits records were widened.
        built: The build that passed.
    """

    lane: Lane
    repo: Path
    port: int
    cleared: tuple[str, ...]
    stale: tuple[str, ...]
    widened: tuple[str, ...]
    built: Result

    @property
    def config(self) -> dict[str, Any]:
        return yaml.safe_load((self.repo / "build" / "config.yml").read_text(encoding="utf-8"))

    @property
    def document(self) -> dict[str, Any]:
        from osprey.facility import FACILITY_FILE

        return json.loads((self.repo / "build" / FACILITY_FILE).read_text(encoding="utf-8"))

    @property
    def view(self) -> dict[str, Any]:
        path = self.repo / "build" / "data" / "simulator" / "variables.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def models(self) -> list[str]:
        """Every model of the facility file that wires channels: all but the texture."""
        from osprey.facility import TEXTURE

        return [str(model["name"]) for model in self.document["models"] if model["name"] != TEXTURE]

    def wired(self, model: str) -> list[str]:
        """Every address ``model`` wires in the simulator view, in wiring order."""
        (entry,) = [entry for entry in self.view["models"] if entry["name"] == model]
        return list(dict.fromkeys(str(record["address"]) for record in entry.get("wiring") or []))


def imported_probe(repo: Path) -> str:
    """The first readback by address of the import's channels.

    The rule the build's served-probe stop names its remedy by: a channel whose
    role is ``readback`` or states none.
    """
    channels = yaml.safe_load(
        (repo / FACILITY_DIR / "imported" / "mml" / "channels.yaml").read_text(encoding="utf-8")
    )
    return min(
        str(channel["id"]) for channel in channels if channel.get("role", "readback") == "readback"
    )


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK_HOST, 0))
        return int(sock.getsockname()[1])


def install(lane: Lane, repo: Path, port: int) -> Install:
    """Install *lane* into *repo* through ``facility import mml`` and build it."""
    runner = CliRunner()
    exports = list(lane.exports)
    _ok(runner, "init", str(repo), "--preset", "control-assistant", "--no-git")

    target = repo / FACILITY_DIR / _mapping_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(lane.mapping, target)

    arguments = ("facility", "import", "mml", *exports, "--repo", str(repo))
    imported = _invoke(runner, *arguments)
    cleared: list[str] = []
    if imported.exit_code != 0:
        lines = imported.stderr.splitlines()
        assert lines and lines[0].startswith(AUTHORED_PRESENT), imported.output
        assert all(line.startswith("rm ") for line in lines[1:]), imported.stderr
        for line in lines[1:]:
            cleared += _remove_named(repo, line)
        imported = _ok(runner, *arguments)

    stale: list[str] = []
    printed = imported.stderr.splitlines()
    if STALE_SCENARIOS in printed:
        listed = printed[printed.index(STALE_SCENARIOS) + 1 :]
        assert listed and all(line.startswith("  rm ") for line in listed), imported.stderr
        for line in listed:
            stale += _remove_named(repo, line.strip())

    _ok(runner, "set", "--repo", str(repo), *isolation_settings(port, imported_probe(repo)))

    where = ["--repo", str(repo)]
    first = _invoke(runner, *BUILD, *where)
    widened: list[str] = []
    if first.exit_code != 0:
        validate = _invoke(runner, "facility", "validate", *where)
        for line in validate.stderr.splitlines():
            if (match := SEED_INVALID.match(line)) is None:
                continue
            edge = "max_value" if match["side"] == "above" else "min_value"
            assert match["edge"] == edge, line
            _widen(
                repo / FACILITY_DIR / "limits.yaml", match["address"], edge, float(match["nominal"])
            )
            widened.append(match["address"])
        assert widened, (
            f"the build stopped on something other than a seed-invalid band:\n{first.output}"
        )
        built = _ok(runner, *BUILD, *where)
    else:
        built = first

    return Install(
        lane=lane,
        repo=repo,
        port=port,
        cleared=tuple(cleared),
        stale=tuple(stale),
        widened=tuple(widened),
        built=built,
    )


# ===================================================================
# The served machine
# ===================================================================

#: Bound on the container's boot, to the first served answer.
BOOT_TIMEOUT_S = 240.0

#: Bound on one readiness probe.
PROBE_TIMEOUT_S = 45.0

CONTAINER_PREFIX = "osprey-als-lane"


@dataclass(frozen=True)
class Served:
    """A booted container serving one install, and the CA endpoint the config names."""

    install: Install
    container: str
    endpoint: str

    def environment(self) -> dict[str, str]:
        """The CA client's environment: the VA and nothing else."""
        host = self.endpoint.rsplit(":", 1)[0]
        environment = {
            **os.environ,
            "EPICS_CA_NAME_SERVERS": self.endpoint,
            "EPICS_CA_ADDR_LIST": host,
            "EPICS_CA_AUTO_ADDR_LIST": "NO",
            "EPICS_PVA_ADDR_LIST": host,
            "EPICS_PVA_AUTO_ADDR_LIST": "NO",
        }
        for stale in ("EPICS_CA_SERVER_PORT", "EPICS_CAS_SERVER_PORT", "EPICS_PVA_NAME_SERVERS"):
            environment.pop(stale, None)
        return environment

    def read(self, addresses: Sequence[str], *, timeout: float = 30.0) -> dict[str, Any]:
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker",
                json.dumps({"addresses": list(addresses), "timeout": timeout}),
            ],
            capture_output=True,
            text=True,
            timeout=max(600.0, timeout * 4),
            env=self.environment(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"CA worker failed:\n{result.stdout}\n{result.stderr}")
        return json.loads(result.stdout.strip().splitlines()[-1])


def _endpoint(config: dict[str, Any]) -> str:
    """The Channel Access endpoint the rendered VA block dials."""
    gateway = config["control_system"]["connector"]["virtual_accelerator"]["gateways"]["read_only"]
    return f"{gateway['address']}:{gateway['port']}"


def _docker(*arguments: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *arguments], capture_output=True, text=True, timeout=timeout)


def boot(installed: Install) -> Iterator[Served]:
    """Boot the VA image over the build's data root, on the configured loopback port.

    The port is the one the rendered config names, published on 127.0.0.1
    only, so the client dials exactly what a session on this deployment would.
    """
    from tests.va.e2e import conftest as e2e

    endpoint = _endpoint(installed.config)
    port = installed.port
    name = f"{CONTAINER_PREFIX}-{port}-{uuid.uuid4().hex[:8]}"
    started = _docker(
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"EPICS_CA_SERVER_PORT={port}",
        *e2e.DEMO_NAMESPACE_RUN_ARGS,
        *e2e.data_root_run_args(installed.repo / "build" / "data"),
        "-p",
        f"{LOOPBACK_HOST}:{port}:{port}/tcp",
        e2e.IMAGE,
    )
    if started.returncode != 0:
        _docker("rm", "-f", name)
        pytest.fail(f"docker run failed: {started.stdout}\n{started.stderr}")
    served = Served(install=installed, container=name, endpoint=endpoint)
    try:
        _wait_until_ready(served)
        yield served
    finally:
        _docker("rm", "-f", name)


def _wait_until_ready(served: Served) -> None:
    """Block until the container answers a wired channel, or fail naming what it said."""
    from tests.va.e2e import conftest as e2e

    model = served.install.models()[0]
    probe = served.install.wired(model)[0]
    deadline = time.monotonic() + BOOT_TIMEOUT_S
    last_attempt = "no read completed"
    while (remaining := deadline - time.monotonic()) > 0:
        bound = min(PROBE_TIMEOUT_S, remaining)
        try:
            answer = served.read([probe], timeout=min(30.0, bound))
            if not answer["failed"] and answer["values"].get(probe) is not None:
                return
            last_attempt = f"{probe}: {answer['failed'] or 'read back as None'}"
        except Exception as exc:  # "not up yet" is the expected case here
            last_attempt = f"{type(exc).__name__}: {exc}"
        time.sleep(2.0)
    logs = _docker("logs", "--tail", "60", served.container)
    pytest.fail(
        f"{served.install.lane.label}: the container never served {probe} within "
        f"{BOOT_TIMEOUT_S}s.\nThe client's last attempt: {last_attempt}\n"
        f"{e2e.boot_report(served.container, served.install.port)}\n"
        f"Container logs:\n{logs.stdout}\n{logs.stderr}"
    )


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="module")
def lane() -> Lane:
    """The resolved lane; a set variable naming a missing path fails here."""
    if isinstance(RESOLVED, Broken):
        pytest.fail(RESOLVED.reason)
    assert isinstance(RESOLVED, Lane)
    return RESOLVED


@pytest.fixture(scope="module")
def installed(lane: Lane, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Install]:
    """The facility imported, isolated and built once for the module.

    The Channel Access search variables name loopback for the whole install,
    a second layer under the rendered config's own gateways.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("EPICS_CA_AUTO_ADDR_LIST", "NO")
        patch.setenv("EPICS_CA_ADDR_LIST", LOOPBACK_HOST)
        patch.setenv("EPICS_PVA_AUTO_ADDR_LIST", "NO")
        patch.setenv("EPICS_PVA_ADDR_LIST", LOOPBACK_HOST)
        repo = tmp_path_factory.mktemp("als-lane") / "deployment"
        yield install(lane, repo, _reserve_port())


@pytest.fixture(scope="module")
def served(installed: Install) -> Iterator[Served]:
    """The install booted, after its rendered config passed the isolation preflight."""
    errors = isolation_errors(installed.config.get("control_system"))
    if errors:
        pytest.fail(
            f"{installed.lane.label}: the rendered config is not isolated, so nothing boots:\n"
            + "\n".join(errors)
        )
    yield from boot(installed)


# ===================================================================
# Discovery: which variable decides, and what a set one must name
# ===================================================================


@pytest.fixture()
def nsls2() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "mml" / "nsls2"


def _profiles(root: Path, *, identity: bool = True, mapping: bool = True) -> Path:
    facility = root / FACILITY_DIR
    facility.mkdir(parents=True)
    if identity:
        (facility / IDENTITY_FILE).write_text("code: x\n", encoding="utf-8")
    if mapping:
        (facility / _mapping_file()).parent.mkdir(parents=True)
        (facility / _mapping_file()).write_text("models: {}\n", encoding="utf-8")
    (root / "exports").mkdir()
    (root / "exports" / f"x.sr{AO_SUFFIX}").write_text("{}", encoding="utf-8")
    return root


def test_no_variable_skips_naming_every_variable() -> None:
    resolved = resolve({})
    assert isinstance(resolved, Unset)
    for variable in LANE_VARIABLES:
        assert variable in resolved.reason


def test_the_profiles_checkout_is_read_for_its_mapping_and_exports(tmp_path: Path) -> None:
    root = _profiles(tmp_path / "profiles")
    resolved = resolve({ALS_PROFILES_ENV: str(root)})
    assert isinstance(resolved, Lane), resolved
    assert resolved.mapping == root.resolve() / FACILITY_DIR / _mapping_file()
    assert resolved.exports == (str(root.resolve() / "exports" / f"x.sr{AO_SUFFIX}"),)


@pytest.mark.parametrize("missing", ["identity", "mapping"])
def test_a_profiles_checkout_missing_a_file_fails_naming_it(tmp_path: Path, missing: str) -> None:
    root = _profiles(
        tmp_path / "profiles", identity=missing != "identity", mapping=missing != "mapping"
    )
    resolved = resolve({ALS_PROFILES_ENV: str(root)})
    assert isinstance(resolved, Broken), resolved
    name = IDENTITY_FILE if missing == "identity" else _mapping_file()
    assert str(root.resolve() / FACILITY_DIR / name) in resolved.reason


def test_the_profiles_checkout_wins_over_every_other_variable(tmp_path: Path, nsls2: Path) -> None:
    resolved = resolve({ALS_PROFILES_ENV: str(tmp_path / "absent"), STAND_IN_ENV: str(nsls2)})
    assert isinstance(resolved, Broken)
    assert ALS_PROFILES_ENV in resolved.reason


def test_a_named_export_and_mapping_are_installed_as_named(nsls2: Path) -> None:
    export = nsls2 / f"nsls2.ltb{AO_SUFFIX}"
    mapping = nsls2 / _mapping_file()
    resolved = resolve({ALS_EXPORT_ENV: str(export), ALS_MAPPING_ENV: str(mapping)})
    assert isinstance(resolved, Lane), resolved
    assert resolved.exports == (str(export.resolve()),)
    assert resolved.mapping == mapping.resolve()


def test_a_named_mapping_that_is_not_a_file_fails_naming_it(tmp_path: Path, nsls2: Path) -> None:
    missing = tmp_path / "mapping.yaml"
    resolved = resolve(
        {ALS_EXPORT_ENV: str(nsls2 / f"nsls2.ltb{AO_SUFFIX}"), ALS_MAPPING_ENV: str(missing)}
    )
    assert isinstance(resolved, Broken), resolved
    assert ALS_MAPPING_ENV in resolved.reason and str(missing) in resolved.reason


def test_a_named_export_that_is_not_a_file_fails_naming_it(tmp_path: Path, nsls2: Path) -> None:
    missing = tmp_path / f"x{AO_SUFFIX}"
    resolved = resolve(
        {ALS_EXPORT_ENV: str(missing), ALS_MAPPING_ENV: str(nsls2 / _mapping_file())}
    )
    assert isinstance(resolved, Broken), resolved
    assert ALS_EXPORT_ENV in resolved.reason and str(missing) in resolved.reason


@pytest.mark.parametrize("variable", [ALS_EXPORT_ENV, ALS_MAPPING_ENV])
def test_half_of_the_named_pair_fails_naming_the_other(variable: str, nsls2: Path) -> None:
    resolved = resolve({variable: str(nsls2)})
    assert isinstance(resolved, Broken), resolved
    other = ALS_MAPPING_ENV if variable == ALS_EXPORT_ENV else ALS_EXPORT_ENV
    assert f"{other} is not" in resolved.reason


def test_the_stand_in_reads_its_exports_and_mapping(nsls2: Path) -> None:
    resolved = resolve({STAND_IN_ENV: str(nsls2)})
    assert isinstance(resolved, Lane), resolved
    assert resolved.mapping == nsls2.resolve() / _mapping_file()
    assert [Path(export).name for export in resolved.exports] == sorted(
        path.name for path in nsls2.glob(f"*{AO_SUFFIX}")
    )


def test_a_stand_in_without_its_mapping_fails_naming_it(tmp_path: Path) -> None:
    resolved = resolve({STAND_IN_ENV: str(tmp_path)})
    assert isinstance(resolved, Broken), resolved
    assert str(tmp_path.resolve() / _mapping_file()) in resolved.reason


def test_a_stand_in_that_is_not_a_directory_fails_naming_it(tmp_path: Path) -> None:
    resolved = resolve({STAND_IN_ENV: str(tmp_path / "absent")})
    assert isinstance(resolved, Broken), resolved
    assert str(tmp_path / "absent") in resolved.reason


# ===================================================================
# The isolation preflight
# ===================================================================


def test_the_isolated_section_passes_the_preflight() -> None:
    assert isolation_errors(_isolated_section()) == []


def test_a_planted_real_block_with_a_remote_gateway_fails_the_preflight() -> None:
    section = _isolated_section()
    section["connector"]["epics"] = {
        "gateways": {"read_only": {"address": "10.0.0.1", "port": 5064}}
    }
    errors = isolation_errors(section)
    assert any("connector.epics survived" in error for error in errors), errors
    assert any("resolves to epics" in error for error in errors), errors


def test_a_profile_left_on_a_real_type_fails_the_preflight() -> None:
    section = _isolated_section()
    section["type"] = "epics"
    errors = isolation_errors(section)
    assert any("control_system.type is 'epics'" in error for error in errors), errors
    assert any("resolves to epics" in error for error in errors), errors


def test_a_remote_pva_gateway_fails_the_preflight() -> None:
    section = _isolated_section()
    section["connector"]["virtual_accelerator"]["pva_gateway"] = {"address": "pvagw.example"}
    errors = isolation_errors(section)
    assert any("pva_gateway dials 'pvagw.example'" in error for error in errors), errors


def test_an_armed_target_fails_the_preflight() -> None:
    section = _isolated_section()
    section["writes_enabled"] = True
    assert "a configured target is armed for writes" in isolation_errors(section)


def test_the_isolation_settings_state_the_connector_table_whole() -> None:
    settings = dict(argument.split("=", 1) for argument in isolation_settings(45123, "A:RB"))
    table = json.loads(settings["config.control_system.connector"])
    assert sorted(table) == ["virtual_accelerator"]
    assert table["virtual_accelerator"]["probe_channel"] == "A:RB"
    for lane_name in ("read_only", "write_access"):
        gateway = table["virtual_accelerator"]["gateways"][lane_name]
        assert (gateway["address"], gateway["port"]) == (LOOPBACK_HOST, 45123)
    assert settings["config.control_system.writes_enabled"] == "false"
    assert settings["connector"] == "virtual_accelerator"


# ===================================================================
# The lane
# ===================================================================


@needs_lane
def test_every_model_the_mapping_names_is_in_the_facility_file(installed: Install) -> None:
    mapping = yaml.safe_load(installed.lane.mapping.read_text(encoding="utf-8"))
    named = sorted(str(body.get("name", key)) for key, body in mapping["models"].items())
    assert named, f"{installed.lane.mapping} names no model"
    missing = sorted(set(named) - set(installed.models()))
    assert not missing, f"{installed.lane.label}: the facility file holds no model {missing}"


@needs_lane
def test_every_stale_scenario_the_import_listed_is_gone(installed: Install) -> None:
    scenarios = (FACILITY_DIR / "scenarios").as_posix()
    outside = [path for path in installed.stale if not path.startswith(f"{scenarios}/")]
    assert not outside, f"the import listed paths outside {scenarios}: {outside}"
    remaining = [path for path in installed.stale if (installed.repo / path).exists()]
    assert not remaining, remaining


@needs_lane
def test_every_model_in_the_facility_file_is_served_and_wired(installed: Install) -> None:
    path = installed.repo / "build" / "data" / "simulator" / "served_models.json"
    served = json.loads(path.read_text(encoding="utf-8"))["models"]
    models = installed.models()
    assert models, f"{installed.lane.label}: the facility file holds no model"
    unserved = sorted(set(models) - set(served))
    assert not unserved, f"{installed.lane.label}: models the view does not serve: {unserved}"
    unwired = [model for model in models if not installed.wired(model)]
    assert not unwired, f"{installed.lane.label}: models that wire no channel: {unwired}"


@needs_lane
def test_the_rendered_config_reaches_only_the_loopback_va(installed: Install) -> None:
    config = installed.config
    assert isolation_errors(config.get("control_system")) == []
    assert _endpoint(config) == f"{LOOPBACK_HOST}:{installed.port}"


@needs_lane
def test_the_simulator_target_probes_a_channel_of_the_facility_file(installed: Install) -> None:
    probe = installed.config["control_system"]["connector"]["virtual_accelerator"]
    channels = {str(channel["id"]) for channel in installed.document["channels"]}
    assert probe.get("probe_channel") in channels, probe


@needs_lane
def test_every_wired_channel_of_every_model_answers_from_the_booted_va(served: Served) -> None:
    unanswered: dict[str, dict[str, str]] = {}
    for model in served.install.models():
        wired = served.install.wired(model)
        answer = served.read(wired)
        missing = {
            address: "served with no value"
            for address, value in answer["values"].items()
            if value is None
        }
        if answer["failed"] or missing:
            unanswered[model] = {**answer["failed"], **missing}
    assert not unanswered, (
        f"{served.install.lane.label}: wired channels that did not answer, by model: "
        f"{ {model: sorted(addresses) for model, addresses in unanswered.items()} }"
    )


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_als_lane.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
