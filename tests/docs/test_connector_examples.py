"""The connector pages' example scripts run, and print what the pages show.

Every Python example on the connector pages is a file under
``docs/source/how-to/control-systems/examples/`` that the page includes, so the
code a reader copies is the code that runs here. Each script runs in a
subprocess of its own, in an empty directory, against a soft IOC, a stub
Archiver Appliance or a built simulator view. Where a page shows the output, it
includes the script's ``.out`` file, and the run must print exactly that.

A script that talks to a facility names a placeholder host in a module
constant (``CONFIG``, ``ARCHIVER``). That constant is the one thing replaced
here, with the test server's address; nothing else in the script is touched.
"""

from __future__ import annotations

import copy
import fnmatch
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import yaml

from tests._facility_file import demo_facility_document

pytest.importorskip("aioca", reason="the EPICS connector's Channel Access client")
pytest.importorskip("epicscorelibs.ioc", reason="serves the soft IOC")

from osprey_connectors import write_authorization as wa
from osprey_connectors_tests._harness._control_context_fixtures import (
    isolate_verdict_environment,
    write_control_context,
)
from osprey_connectors_tests.ipc._soft_ioc import SoftIOC

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "docs/source/how-to/control-systems/examples"
TEMPLATE_FACILITY = REPO_ROOT / "src/osprey/templates/facilities/example"

#: How long one example may run before the test gives up on it.
RUN_TIMEOUT_S = 120.0

#: The records the quickstart reads, as the page describes them.
QUICKSTART_RECORDS = """
record(ao, "DEMO:SETPOINT") { field(VAL, "1.5") field(PREC, "3") field(EGU, "A") }
record(ai, "DEMO:READBACK") { field(INP, "DEMO:SETPOINT CP") field(PREC, "3") field(EGU, "A") }
record(mbbo, "DEMO:MODE") {
    field(ZRST, "OFF") field(ONST, "STANDBY") field(TWST, "ON") field(VAL, "1")
}
"""

#: The correctors the write-authorization example writes.
CORRECTOR_RECORDS = """
record(ao, "SR:CORRECTOR:H1") { field(VAL, "0") field(PREC, "3") }
record(ao, "SR:CORRECTOR:H2") { field(VAL, "0") field(PREC, "3") }
"""

#: The samples the stub appliance answers with, the same for every channel.
#: Fixed instants, so the printed frame is the page's ``.out`` file exactly.
SAMPLE_SECS = [
    int(datetime(2026, 11, 4, 23, 33, 24, tzinfo=UTC).timestamp()) + 60 * i for i in range(3)
]
SAMPLE_VALUES = [1.5, 1.6, 1.7]

#: Runs the example at argv[1], with the module constants in argv[2] replaced.
_DRIVER = """
import asyncio, importlib.util, json, sys
spec = importlib.util.spec_from_file_location("example", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name, value in json.loads(sys.argv[2]).items():
    setattr(module, name, value)
asyncio.run(module.main())
"""


def _load(name: str) -> ModuleType:
    """The example module, imported here only to read its constants."""
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected(name: str) -> str:
    return (EXAMPLES / f"{name}.out").read_text()


def _clean_env(home: Path, **extra: str) -> dict[str, str]:
    """A reader's shell: no OSPREY or EPICS variables, no config, nothing of the suite's."""
    env = {"HOME": str(home), "PATH": os.defpath, "PYTHONNOUSERSITE": "1"}
    env.update(extra)
    return env


def _run(
    name: str,
    cwd: Path,
    env: Mapping[str, str],
    overrides: Mapping[str, Any] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the example *name* in *cwd*, its constants replaced by *overrides*."""
    return subprocess.run(
        [sys.executable, "-c", _DRIVER, str(EXAMPLES / f"{name}.py"), json.dumps(overrides or {})],
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        check=False,
    )


def _ok(result: subprocess.CompletedProcess[str]) -> str:
    assert result.returncode == 0, f"exit {result.returncode}\n{result.stdout}\n{result.stderr}"
    return result.stdout


def _pointed_at(config: Mapping[str, Any], port: int) -> dict[str, Any]:
    """An EPICS example's CONFIG with its read gateway moved to the soft IOC."""
    pointed = copy.deepcopy(dict(config))
    gateway = pointed["connector"]["epics"]["gateways"]["read_only"]
    gateway.update({"address": "127.0.0.1", "port": port, "use_name_server": False})
    return pointed


# -- the servers -------------------------------------------------------------


@pytest.fixture
def quickstart_ioc(tmp_path: Path) -> Iterator[SoftIOC]:
    ioc = SoftIOC(tmp_path, "quickstart", records=QUICKSTART_RECORDS)
    yield ioc
    ioc.stop()


class _Appliance(BaseHTTPRequestHandler):
    """The retrieval servlet's ``getData.json``, answering every channel with the samples."""

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path != "/retrieval/data/getData.json":
            self.send_error(404)
            return
        channel = parse_qs(url.query)["pv"][0]
        body = json.dumps(
            [
                {
                    "meta": {"name": channel, "EGU": "A", "PREC": "3"},
                    "data": [
                        {"secs": secs, "nanos": 0, "val": val, "severity": 0, "status": 0}
                        for secs, val in zip(SAMPLE_SECS, SAMPLE_VALUES, strict=True)
                    ],
                }
            ]
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet: the suite's output is no place for an access log."""


@pytest.fixture
def appliance() -> Iterator[str]:
    """The stub appliance's root URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Appliance)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _cf_channel(name: str, ioc: str, description: str) -> dict[str, Any]:
    """A channel as RecSync records it in ChannelFinder."""
    properties = {
        "iocName": ioc,
        "hostName": f"{ioc}.facility.edu",
        "pvStatus": "Active",
        "recordType": "ai",
        "recordDesc": description,
    }
    return {
        "name": name,
        "owner": "recceiver",
        "properties": [{"name": key, "value": value} for key, value in properties.items()],
        "tags": [],
    }


#: The channels the stub ChannelFinder lists.
CF_CHANNELS = [
    _cf_channel("SR:BPM1:X", "sr-bpm", "BPM 1 horizontal position"),
    _cf_channel("SR:BPM1:Y", "sr-bpm", "BPM 1 vertical position"),
    _cf_channel("SR:BPM2:X", "sr-bpm", "BPM 2 horizontal position"),
    _cf_channel("SR:RF:FWD", "sr-rf", "RF forward power"),
]


class _ChannelFinder(BaseHTTPRequestHandler):
    """ChannelFinder's channel and count routes, matching ``~name`` and properties as globs."""

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        prefix = "/ChannelFinder/resources/channels"
        if url.path not in (prefix, f"{prefix}/count"):
            self.send_error(404)
            return
        matched = [channel for channel in CF_CHANNELS if self._matches(channel, query)]
        if url.path.endswith("/count"):
            answer: Any = len(matched)
        else:
            start = int(query.get("~from", ["0"])[0])
            answer = matched[start : start + int(query.get("~size", ["100"])[0])]
        body = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _matches(channel: Mapping[str, Any], query: Mapping[str, list[str]]) -> bool:
        properties = {entry["name"]: entry["value"] for entry in channel["properties"]}
        for key, (pattern, *_) in query.items():
            if key == "~name":
                value = channel["name"]
            elif key.startswith("~"):
                continue
            else:
                value = properties.get(key, "")
            if not fnmatch.fnmatchcase(value, pattern):
                return False
        return True

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet: the suite's output is no place for an access log."""


@pytest.fixture
def channelfinder() -> Iterator[str]:
    """The stub ChannelFinder's service URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChannelFinder)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/ChannelFinder"
    server.shutdown()
    server.server_close()


# -- use-connectors-standalone ----------------------------------------------


def test_read_channels_prints_the_page_output(quickstart_ioc, tmp_path):
    module = _load("read_channels")
    result = _run(
        "read_channels",
        tmp_path,
        _clean_env(tmp_path),
        {"CONFIG": _pointed_at(module.CONFIG, quickstart_ioc.port)},
    )
    assert _ok(result) == _expected("read_channels")


def test_without_a_config_the_script_says_it_fell_back_to_utc(quickstart_ioc, tmp_path):
    """The page's claim: no ``config.yml`` reads the defaults and warns about UTC."""
    module = _load("read_channels")
    result = _run(
        "read_channels",
        tmp_path,
        _clean_env(tmp_path),
        {"CONFIG": _pointed_at(module.CONFIG, quickstart_ioc.port)},
    )
    _ok(result)
    assert "Falling back to UTC facility timezone" in result.stdout + result.stderr


def test_the_timezone_config_the_page_shows_ends_the_warning(quickstart_ioc, tmp_path):
    config = tmp_path / "config.yml"
    config.write_text((EXAMPLES / "system-timezone.yml").read_text())
    module = _load("read_channels")
    result = _run(
        "read_channels",
        tmp_path,
        _clean_env(tmp_path, CONFIG_FILE=str(config)),
        {"CONFIG": _pointed_at(module.CONFIG, quickstart_ioc.port)},
    )
    assert _ok(result) == _expected("read_channels")
    assert "Falling back to UTC" not in result.stdout + result.stderr


def test_the_timezone_config_the_page_shows_sets_the_zone(tmp_path):
    """Not only the warning: the zone the config states is the zone the package uses.

    Any config file ends the warning, so the warning alone cannot tell a key that
    is read from one that is silently ignored.
    """
    config = tmp_path / "config.yml"
    config.write_text((EXAMPLES / "system-timezone.yml").read_text())
    stated = yaml.safe_load(config.read_text())["system"]["timezone"]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from osprey_connectors.config import get_facility_timezone; "
                "print(get_facility_timezone())"
            ),
        ],
        cwd=tmp_path,
        env=_clean_env(tmp_path, CONFIG_FILE=str(config)),
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        check=False,
    )
    assert _ok(result).strip() == stated


def test_read_history_prints_the_page_output(appliance, tmp_path):
    module = _load("read_history")
    archiver = copy.deepcopy(module.ARCHIVER)
    archiver["settings"]["url"] = appliance
    result = _run("read_history", tmp_path, _clean_env(tmp_path), {"ARCHIVER": archiver})
    assert _ok(result) == _expected("read_history")


def test_read_directory_prints_the_page_output(channelfinder, tmp_path):
    pytest.importorskip("osprey_connectors.directory")
    module = _load("read_directory")
    directory = {**module.DIRECTORY, "url": channelfinder}
    result = _run("read_directory", tmp_path, _clean_env(tmp_path), {"DIRECTORY": directory})
    assert _ok(result) == _expected("read_directory")


def test_a_scripts_write_is_refused_as_the_page_shows(quickstart_ioc, tmp_path):
    module = _load("write_refused")
    result = _run(
        "write_refused",
        tmp_path,
        _clean_env(tmp_path),
        {"CONFIG": _pointed_at(module.CONFIG, quickstart_ioc.port)},
    )
    assert _ok(result) == _expected("write_refused")


def test_with_the_tier_armed_the_next_term_refuses(quickstart_ioc, tmp_path):
    """The page's claim: arming the tier in ``config.yml`` moves the refusal to ``target_unknown``."""
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump({"control_system": {"type": "epics", "writes": "allowed"}}))
    module = _load("write_refused")
    result = _run(
        "write_refused",
        tmp_path,
        _clean_env(tmp_path, CONFIG_FILE=str(config)),
        {"CONFIG": _pointed_at(module.CONFIG, quickstart_ioc.port)},
    )
    assert _ok(result).splitlines()[-1] == "[osprey-refusal: target_unknown]"


def test_scripts_leave_nothing_behind(quickstart_ioc, tmp_path):
    """A reader's working directory and home are as they were: nothing is created."""
    work, home = tmp_path / "work", tmp_path / "home"
    work.mkdir()
    home.mkdir()
    module = _load("write_refused")
    _ok(
        _run(
            "write_refused",
            work,
            _clean_env(home),
            {"CONFIG": _pointed_at(module.CONFIG, quickstart_ioc.port)},
        )
    )
    assert list(work.iterdir()) == []
    assert list(home.iterdir()) == []


# -- use-connectors: the simulator in process -------------------------------


@pytest.fixture(scope="module")
def simulator_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project holding the example facility's built simulator view and its config."""
    from osprey.facility.records import FacilityRecords
    from osprey.facility.served import resolve_served
    from osprey.facility.views import ViewInputs
    from osprey.facility.views.simulator import write_simulator_view
    from tests._simulator_view import write_scenarios_view

    config = {"control_system": {"type": "virtual_accelerator"}}
    root = tmp_path_factory.mktemp("project")
    records = FacilityRecords(demo_facility_document())
    write_simulator_view(
        root / "data" / "simulator",
        ViewInputs(
            records=records,
            rendered_config=config,
            facility_dir=TEMPLATE_FACILITY,
            served=resolve_served(config, records),
            reported=None,
        ),
    )
    write_scenarios_view(root, {"nominal": {}})
    (root / "config.yml").write_text(yaml.safe_dump(config))
    return root


def test_read_simulator_reads_the_beam_current(simulator_project, tmp_path):
    result = _run(
        "read_simulator",
        simulator_project,
        _clean_env(tmp_path, CONFIG_FILE=str(simulator_project / "config.yml")),
    )
    value, units = _ok(result).strip().removeprefix("Value: ").rsplit(" ", 1)
    assert float(value) > 0
    assert units == "mA"


# -- use-connectors: writing from a program ---------------------------------


@pytest.fixture
def standin_deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A deployment whose stand-in is a soft IOC serving the correctors.

    Its login's switch stays off, so only an approved authorization can write.
    Yields the deployment directory, which holds its ``config.yml``.
    """
    ioc = SoftIOC(tmp_path, "standin", records=CORRECTOR_RECORDS)
    try:
        isolate_verdict_environment(monkeypatch, tmp_path / "agent-data")
        for name in (wa.AUTHORIZATION_ENV_VAR, wa.LEDGER_DIR_ENV_VAR, "OSPREY_SESSION_ID"):
            monkeypatch.delenv(name, raising=False)
        write_control_context(tmp_path / "agent-data", target="standin", owned_by=None, switch={})
        deployment = tmp_path / "deployment"
        deployment.mkdir()
        endpoint = {"address": "127.0.0.1", "port": ioc.port, "use_name_server": False}
        section = {
            "type": "live_standin",
            "writes": "allowed",
            "connector": {"live_standin": {"timeout_s": 5.0, "gateways": {"read_only": endpoint}}},
        }
        (deployment / "config.yml").write_text(yaml.safe_dump({"control_system": section}))
        yield deployment
    finally:
        ioc.stop()


def _approve_when_asked(deadline_s: float) -> wa.Grant:
    """The person's half: approve the program's request from the CLI once it is there."""
    deadline = time.monotonic() + deadline_s
    directory = wa.requests_dir()
    assert directory is not None, "no requests directory resolves for this login"
    while time.monotonic() < deadline:
        requests, _unreadable = wa.list_requests(directory)
        if requests:
            (request,) = requests
            return wa.approve_request(request, surface="cli")
        time.sleep(0.2)
    raise AssertionError("the program never asked for an authorization")


def test_write_from_a_program_writes_what_was_approved(standin_deployment):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EPICS_CA", "EPICS_PVA"))}
    env["CONFIG_FILE"] = str(standin_deployment / "config.yml")
    program = subprocess.Popen(
        [sys.executable, "-c", _DRIVER, str(EXAMPLES / "write_from_a_program.py"), "{}"],
        cwd=standin_deployment,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        grant = _approve_when_asked(deadline_s=60.0)
        stdout, stderr = program.communicate(timeout=RUN_TIMEOUT_S)
    finally:
        if program.poll() is None:
            program.kill()
            program.communicate()
    assert program.returncode == 0, f"{stdout}\n{stderr}"
    assert stdout == _expected("write_from_a_program")
    assert grant.target == "standin"
    assert set(grant.channels) == {"SR:CORRECTOR:H1", "SR:CORRECTOR:H2"}
