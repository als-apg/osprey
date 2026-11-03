"""The CANOPY sidecar with the real servers behind it.

``python -m canopy_mcp --http --fd N`` and ``python -m uvicorn canopy.main:app
--fd N`` are started by a real :class:`CanopySidecar`, and everything is asked
for the way a browser behind the terminal's proxy asks: over HTTP and the
WebSocket relay on FRONT's port.

FRONT is given fixtures rather than a built payload: a viewer directory holding
a stub ``index.html`` and the shipped copy of CANOPY's ``ir-map.json`` (the widget
catalog), no staged spec and no Trace. The control system is the fake connector,
registered in the DATA process under the virtual accelerator's type by a
``sitecustomize`` on the children's module path, so a ``va`` record is served
with values and no IOC is needed.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

# Marks the module as a CANOPY contract test: the mark is derived from imports.
import canopy  # noqa: F401
import pytest
import yaml
from websockets.sync import client as websockets_client

from osprey.integrations.canopy.sidecar import CanopySidecar
from osprey_connectors import control_context, posture_store, types
from osprey_connectors.identity import AUDIT_IDENTITY_ENV

#: How long the two servers get to come up.
READY_TIMEOUT_S = 120.0

#: How long a test waits for the replacement's values before it fails: the hang
#: bound, the only time limit a test asserts.
REPLACEMENT_DEADLINE_S = 60.0

PV = "TEST:SCALAR"

#: The viewer's widget catalog, a copy of CANOPY's own at the proven commit.
IR_MAP = Path(__file__).parent / "fixtures" / "ir-map.json"

#: Runs in every child at interpreter start; only DATA carries the backend
#: variable. The session's own registration never replaces an existing one.
SITECUSTOMIZE = textwrap.dedent(
    """
    import os

    if os.environ.get("CANOPY_EPICS__BACKEND"):
        from osprey_connectors import types
        from osprey_connectors_tests._fake_connector import FakeConnector
        from osprey_connectors.factory import ConnectorFactory

        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, FakeConnector)
    """
)

CONFIG = {
    "control_system": {
        "type": types.VIRTUAL_ACCELERATOR,
        "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
    }
}


def write_record(state_dir: Path, *, target: str, generation: int) -> None:
    control_context.write_record(
        control_context.ControlContext(target=target, generation=generation, posture={}),
        path=state_dir / control_context.RECORD_FILENAME,
    )
    control_context.invalidate_cache()


def status_of(url: str, headers: dict[str, str] | None = None) -> int:
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return exc.code


def first_value(sidecar: CanopySidecar, deadline: float) -> float | None:
    """The first number FRONT's relay delivers for :data:`PV` before *deadline*."""
    address = sidecar.url.replace("http://", "ws://") + "/hare/v1/ws/pvs"
    while time.monotonic() < deadline:
        try:
            with websockets_client.connect(
                address,
                additional_headers=sidecar.auth_headers,
                open_timeout=max(0.1, deadline - time.monotonic()),
            ) as relay:
                relay.send(json.dumps({"type": "subscribe", "pvNames": [PV]}))
                while time.monotonic() < deadline:
                    frame = json.loads(relay.recv(timeout=max(0.05, deadline - time.monotonic())))
                    if frame.get("type") not in ("initial", "diff"):
                        continue
                    item = (frame.get("data") or {}).get(PV)
                    value = item.get("value") if isinstance(item, dict) else item
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        return float(value)
        except TimeoutError:
            break
        except Exception:  # the relay closed while DATA was being replaced
            time.sleep(0.05)
    return None


def refuses(url: str) -> bool:
    port = int(url.rsplit(":", 1)[1])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(2)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def wait_for(predicate, what: str, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    pytest.fail(f"{what} did not happen within {timeout} s")


@pytest.fixture
def launch(tmp_path, monkeypatch):
    """Build sidecars over a ``va`` deployment in ``tmp_path``; all are stopped afterwards."""
    for name in list(os.environ):
        if name.startswith(("CANOPY_", "OSPREY_CANOPY_", "OSPREY_CONTROL_TARGET", "EPICS_")):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)

    state_dir = tmp_path / "control_target"
    state_dir.mkdir()
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")

    payload = tmp_path / "canopy_payload"
    viewer = payload / "viewer"
    viewer.mkdir(parents=True)
    (viewer / "index.html").write_text("<!doctype html><title>stub</title>", encoding="utf-8")
    shutil.copy(IR_MAP, viewer / "ir-map.json")

    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    # The connectors' test package too: the sitecustomize imports the fake from it.
    root = Path(__file__).resolve().parents[3]
    paths = [str(site), str(root), str(root / "packages" / "osprey-connectors" / "tests")]
    inherited = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([*paths, inherited] if inherited else paths))

    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(state_dir))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "test-operator")
    # A stray file in the terminal's working directory must change nothing.
    (tmp_path / ".env").write_text("CANOPY_API_KEYS=stray:read+write\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    write_record(state_dir, target="va", generation=1)

    sidecars: list[CanopySidecar] = []

    def build() -> CanopySidecar:
        sidecar = CanopySidecar(tmp_path / "agent-data", "", None, payload_dir=payload)
        sidecars.append(sidecar)
        return sidecar

    build.state_dir = state_dir
    yield build
    for sidecar in sidecars:
        sidecar.stop()
    control_context.invalidate_cache()


@pytest.fixture
def running(launch):
    """A sidecar that is ready, and the state directory its record is in."""
    sidecar = launch()
    sidecar.preflight()
    sidecar.spawn()
    try:
        sidecar.wait_ready(READY_TIMEOUT_S)
    except RuntimeError as exc:
        pytest.fail(str(exc))
    return sidecar, launch.state_dir


def test_front_serves_behind_the_credential_and_relays_a_value(running):
    """``/health`` is open, everything else needs the header, and a value arrives."""
    sidecar, _ = running
    assert status_of(f"{sidecar.url}/health") == 200
    assert status_of(f"{sidecar.url}/") == 401
    assert status_of(f"{sidecar.url}/config.json") == 401
    assert status_of(f"{sidecar.url}/", sidecar.auth_headers) == 200
    assert status_of(f"{sidecar.data_url}/health") == 200
    assert status_of(f"{sidecar.data_url}/hare/v1/pvs/{PV}") == 401

    assert first_value(sidecar, time.monotonic() + 30.0) is not None


def test_a_record_flip_replaces_data_and_keeps_front_and_both_ports(running):
    """A new generation: another DATA pid, the same FRONT, the same ports, values back."""
    sidecar, state_dir = running
    assert first_value(sidecar, time.monotonic() + 30.0) is not None
    front_pid, data_pid = sidecar.front_pid, sidecar.data_pid
    url, data_url = sidecar.url, sidecar.data_url

    flipped = time.monotonic()
    write_record(state_dir, target="va", generation=2)
    wait_for(lambda: sidecar.data_starts >= 2, "DATA's replacement", REPLACEMENT_DEADLINE_S)
    value = first_value(sidecar, flipped + REPLACEMENT_DEADLINE_S)

    assert value is not None, f"no value within {REPLACEMENT_DEADLINE_S} s of the flip"
    assert sidecar.data_pid not in (None, data_pid)
    assert sidecar.front_pid == front_pid
    assert (sidecar.url, sidecar.data_url) == (url, data_url)
    assert sidecar.exit_status is None


def test_a_killed_data_is_replaced(running):
    """``kill -9`` is a crash: a replacement after the first backoff, values back."""
    sidecar, _ = running
    assert first_value(sidecar, time.monotonic() + 30.0) is not None
    front_pid, data_pid = sidecar.front_pid, sidecar.data_pid

    killed = time.monotonic()
    os.kill(data_pid, signal.SIGKILL)
    wait_for(lambda: sidecar.data_starts >= 2, "DATA's replacement", REPLACEMENT_DEADLINE_S)
    value = first_value(sidecar, killed + REPLACEMENT_DEADLINE_S)

    assert value is not None, f"no value within {REPLACEMENT_DEADLINE_S} s of the kill"
    assert sidecar.data_pid not in (None, data_pid)
    assert sidecar.front_pid == front_pid


def test_a_forbidden_variable_is_a_refused_start_and_the_panel_is_retracted(launch, monkeypatch):
    """DATA leaves with 78, is not started again, and the sidecar reports its end."""
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "someone")
    retracted = []
    sidecar = launch()
    sidecar.spawn()
    front_pid, url, data_url = sidecar.front_pid, sidecar.url, sidecar.data_url
    with pytest.raises(RuntimeError, match="exited with status 78"):
        sidecar.wait_ready(READY_TIMEOUT_S)
    sidecar.on_exit = lambda: retracted.append(True)
    wait_for(lambda: retracted, "the retraction", 30.0)

    assert sidecar.exit_status == 78
    assert f"{posture_store.CONTROL_OWNER_ENV_VAR} must not be set" in sidecar.stderr_tail
    assert sidecar.data_starts == 1
    assert refuses(url) and refuses(data_url)
    with pytest.raises(ProcessLookupError):
        os.kill(front_pid, 0)
    time.sleep(1.0)
    assert sidecar.data_starts == 1
    assert retracted == [True]
