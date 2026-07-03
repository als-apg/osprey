"""End-to-end test of the OSPREY-side Phoebus demo stack (no desktop required).

Proves the closed loop that the demo depends on, headlessly:

    phoebus MCP tool  --HTTP-->  bridge-faithful server  --caput-->  soft IOC
                                                                        |
    OSPREY EPICS view (pyepics caget)  <--------- same DEMO:* PV -------+

The real ``demo_ioc.py`` runs as a subprocess (a genuine Channel Access server).
A small in-process HTTP server speaks the *exact* bridge JSON contract the
``phoebus`` MCP tools expect, but backs ``/perceive`` and ``/drive`` with real
``pyepics`` reads/writes against the IOC — so a tool-driven "click" actually
moves the PV, and we verify it two independent ways (re-perceive + caget).

The live *Phoebus GUI* mile (real synthetic JavaFX events) is covered separately
by the phoebus repo's Monocle ``EndToEndHttpIT`` and by ``run_demo.sh`` on a
desktop; this test owns the OSPREY half.

Run explicitly (it is outside the default ``--ignore=tests/e2e`` fast run)::

    uv run pytest tests/e2e/test_phoebus_demo_e2e.py -v
"""

import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

epics = pytest.importorskip("epics")  # pyepics
pytest.importorskip("caproto")

_IOC = Path(__file__).resolve().parents[2] / "demos" / "phoebus" / "demo_ioc.py"
_CA_ENV = {
    "EPICS_CA_ADDR_LIST": "127.0.0.1",
    "EPICS_CA_AUTO_ADDR_LIST": "NO",
    "EPICS_CAS_INTF_ADDR_LIST": "127.0.0.1",
    "EPICS_CAS_BEACON_ADDR_LIST": "127.0.0.1",
}


@pytest.fixture(scope="module", autouse=True)
def _ca_env():
    for k, v in _CA_ENV.items():
        os.environ[k] = v
    yield


@pytest.fixture(scope="module")
def demo_ioc():
    """Start demos/phoebus/demo_ioc.py as a real CA server subprocess."""
    proc = subprocess.Popen(
        [sys.executable, str(_IOC)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, **_CA_ENV},
    )
    # Wait until the IOC answers.
    deadline = time.time() + 20
    while time.time() < deadline:
        if epics.caget("DEMO:Setpoint", timeout=1) is not None:
            break
        time.sleep(0.5)
    else:
        proc.terminate()
        pytest.fail("demo_ioc did not come up on Channel Access")
    yield
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


class _BridgeHandler(BaseHTTPRequestHandler):
    """Speaks the bridge JSON contract, backed by real pyepics IO on DEMO:*."""

    def log_message(self, *a):  # silence
        pass

    def _send(self, code, body, content_type="application/json"):
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if isinstance(body, bytes):  # snapshot registration headers
            self.send_header("X-Bridge-Origin-X", "100.0")
            self.send_header("X-Bridge-Origin-Y", "200.0")
            self.send_header("X-Bridge-Scale", "1.0")
        self.end_headers()
        self.wfile.write(payload)

    def _perceive(self):
        def pv(name, addr):
            return {
                "name": name,
                "type": "widget",
                "pv": {"name": addr, "value": str(epics.caget(addr, as_string=True))},
            }

        return {
            "display": {"name": "OSPREY Phoebus Demo"},
            "widgets": [
                pv("Setpoint", "DEMO:Setpoint"),
                pv("Readback", "DEMO:Readback"),
                pv("Current", "DEMO:Current"),
                pv("Status", "DEMO:Status"),
                {"name": "SetButton", "type": "action_button"},
            ],
        }

    def do_GET(self):
        if self.path.startswith("/displays"):
            self._send(200, [{"name": "OSPREY Phoebus Demo", "ready": True, "active": True}])
        elif self.path.startswith("/perceive"):
            self._send(200, self._perceive())
        elif self.path.startswith("/snapshot"):
            self._send(200, b"\x89PNG\r\n\x1a\nDEMO", content_type="image/png")
        else:
            self._send(404, {"error": f"Unknown route: {self.path}", "status": 404})

    def do_POST(self):
        if not self.path.startswith("/drive"):
            self._send(404, {"error": f"Unknown route: {self.path}", "status": 404})
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))
        widget, verb, value = req.get("widget"), req.get("verb"), req.get("value")
        # SetButton click writes 42; Setpoint type writes the typed value.
        if widget == "SetButton" and verb == "click":
            epics.caput("DEMO:Setpoint", 42, wait=True)
            self._send(200, {"fired": True, "detail": "ActionEvent fired via ButtonBase.fire()"})
        elif widget == "Setpoint" and verb == "type":
            epics.caput("DEMO:Setpoint", float(value), wait=True)
            self._send(200, {"fired": True, "detail": "text set + ENTER fired"})
        else:
            self._send(200, {"fired": False, "detail": "no interactive control resolved"})


@pytest.fixture(scope="module")
def bridge(demo_ioc):
    """Start the bridge-faithful HTTP server; point the tools at it."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BridgeHandler)
    port = server.server_address[1]
    Thread(target=server.serve_forever, daemon=True).start()
    os.environ["PHOEBUS_BRIDGE_URL"] = f"http://127.0.0.1:{port}"
    yield
    server.shutdown()
    os.environ.pop("PHOEBUS_BRIDGE_URL", None)


def _tool(name):
    from osprey.mcp_server.phoebus.tools import bridge_tools
    from tests.mcp_server.conftest import get_tool_fn

    return get_tool_fn(getattr(bridge_tools, name))


def _readback() -> float:
    return float(epics.caget("DEMO:Readback", timeout=3))


async def test_list_and_perceive(bridge):
    from tests.mcp_server.conftest import extract_response_dict

    displays = extract_response_dict(await _tool("phoebus_list_displays")())
    assert displays["displays"][0]["name"] == "OSPREY Phoebus Demo"

    perception = extract_response_dict(await _tool("phoebus_perceive")(display="active"))
    names = [w["name"] for w in perception["perception"]["widgets"]]
    assert {"Setpoint", "Readback", "Status", "SetButton"}.issubset(set(names))


async def test_synthetic_drive_moves_pv_verified_two_ways(bridge):
    from tests.mcp_server.conftest import extract_response_dict

    # baseline: drive to a known-not-42 value first
    epics.caput("DEMO:Setpoint", 5, wait=True)
    assert _readback() != pytest.approx(42)

    # phoebus_drive click the action button -> bridge -> caput 42
    out = extract_response_dict(await _tool("phoebus_drive")(widget="SetButton", verb="click"))
    assert out["status"] == "success" and out["fired"] is True

    time.sleep(0.5)
    # Verify way 1: OSPREY's own EPICS view (pyepics).
    assert _readback() == pytest.approx(42.0)
    # Verify way 2: the GUI's view, via a fresh perceive.
    perception = extract_response_dict(await _tool("phoebus_perceive")())
    rb = next(w for w in perception["perception"]["widgets"] if w["name"] == "Readback")
    assert rb["pv"]["value"].startswith("42")


async def test_semantic_type_and_status_derivation(bridge):
    from tests.mcp_server.conftest import extract_response_dict

    out = extract_response_dict(
        await _tool("phoebus_drive")(widget="Setpoint", verb="type", value="7", mode="semantic")
    )
    assert out["fired"] is True
    time.sleep(0.5)
    assert _readback() == pytest.approx(7.0)
    assert epics.caget("DEMO:Status", as_string=True) == "OK"

    # a high setpoint should derive FAULT, visible to OSPREY's EPICS view
    await _tool("phoebus_drive")(widget="Setpoint", verb="type", value="99", mode="semantic")
    time.sleep(0.5)
    assert epics.caget("DEMO:Status", as_string=True) == "FAULT"


async def test_snapshot_returns_png(bridge, tmp_path, monkeypatch):
    from osprey.mcp_server.phoebus.tools import bridge_tools
    from tests.mcp_server.conftest import extract_response_dict

    monkeypatch.setattr(bridge_tools, "_snapshot_dir", lambda: tmp_path)
    result = await _tool("phoebus_snapshot")(widget="Setpoint", display="active")
    data = extract_response_dict(result)
    assert data["status"] == "success"
    pngs = list(tmp_path.glob("phoebus_Setpoint_*.png"))
    assert len(pngs) == 1 and pngs[0].read_bytes().startswith(b"\x89PNG")
