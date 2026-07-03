"""End-to-end test against the REAL in-JVM Phoebus agent bridge (Task 4.1).

Unlike ``test_phoebus_demo_e2e.py`` (which backs the bridge contract with a
Python fake), this module drives a *running Phoebus product* through the same
MCP tools the agent uses, and confirms every drive with an independent
Channel Access read:

    phoebus MCP tool --HTTP--> in-JVM bridge --JavaFX/PV--> demo soft IOC
                                                                |
    independent CA readback (caproto sync client) <------------+

The full acceptance loop: ``open(name) → handle``, ``perceive(handle)`` returns
the widget tree, ``drive(handle)`` commits a setpoint, CA confirms the value.

Activation (skipped otherwise, so the default test run stays hermetic):

* ``OSPREY_REAL_BRIDGE_URL=http://127.0.0.1:7979`` — target a stack that is
  already running (e.g. started by ``demos/phoebus/run_demo.sh`` on appsdev2).
  Export the matching CA client env too when the stack uses dedicated ports
  (isolated mode): ``EPICS_CA_SERVER_PORT=5074 EPICS_CA_ADDR_LIST=127.0.0.1``.
* ``PHOEBUS_E2E_LAUNCH=1`` — this module launches ``run_demo.sh`` itself
  (honouring ``PHOEBUS_REPO`` / ``JAVA_HOME`` / ``ISOLATED`` / ``OSPREY_PYTHON``)
  and tears it down afterwards; CA env is derived from the script's defaults.

Only ``caproto`` is required for the readback (pyepics is not available on
appsdev2), and it honours the ``EPICS_CA_*`` environment variables.

Run explicitly (outside the default ``--ignore=tests/e2e`` fast run)::

    OSPREY_REAL_BRIDGE_URL=http://127.0.0.1:7979 EPICS_CA_SERVER_PORT=5074 \
        pytest tests/e2e/test_phoebus_real_bridge_e2e.py -v
"""

import json
import os
import platform
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

caproto_client = pytest.importorskip("caproto.sync.client")

from fastmcp.exceptions import ToolError  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_DEMO = _REPO_ROOT / "demos" / "phoebus" / "run_demo.sh"

_BRIDGE_READY_TIMEOUT = 240  # cold JVM + panel load on a shared box
_PV_SETTLE_TIMEOUT = 20


def _ca_get(pv: str, timeout: float = 5.0):
    """Independent CA read (bypasses Phoebus and the bridge entirely)."""
    res = caproto_client.read(pv, timeout=timeout)
    val = res.data[0]
    return val.decode() if isinstance(val, bytes) else val


def _bridge_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/displays", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _default_ca_env(isolated: bool) -> dict[str, str]:
    env = {
        "EPICS_CA_ADDR_LIST": "127.0.0.1",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
    }
    if isolated:
        # run_demo.sh isolated-mode defaults (appsdev2 recon-chosen ports).
        env["EPICS_CA_SERVER_PORT"] = os.environ.get("CA_SERVER_PORT", "5074")
        env["EPICS_CA_REPEATER_PORT"] = os.environ.get("CA_REPEATER_PORT", "5075")
    return env


@pytest.fixture(scope="module")
def real_bridge():
    """Yield the URL of a real bridge, launching the demo stack if asked to."""
    url = os.environ.get("OSPREY_REAL_BRIDGE_URL")
    saved = dict(os.environ)
    proc: subprocess.Popen | None = None

    if url:
        url = url.rstrip("/")
        # Fill in loopback CA defaults without clobbering caller-provided values.
        for k, v in _default_ca_env(isolated=False).items():
            os.environ.setdefault(k, v)
        if not _bridge_up(url):
            pytest.fail(
                f"OSPREY_REAL_BRIDGE_URL={url} does not answer GET /displays — "
                "is the demo stack (run_demo.sh) running?"
            )
    elif os.environ.get("PHOEBUS_E2E_LAUNCH") == "1":
        isolated = os.environ.get(
            "ISOLATED", "1" if platform.system() == "Linux" else "0"
        ) == "1"
        os.environ.update(_default_ca_env(isolated))
        url = f"http://127.0.0.1:{os.environ.get('BRIDGE_PORT', '7979')}"
        proc = subprocess.Popen(
            ["bash", str(_RUN_DEMO)],
            cwd=_REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + _BRIDGE_READY_TIMEOUT
        while time.time() < deadline:
            if _bridge_up(url):
                break
            if proc.poll() is not None:
                pytest.fail(
                    "run_demo.sh exited before the bridge came up — see "
                    "/tmp/osprey_demo_run.log and /tmp/osprey_demo_phoebus.log"
                )
            time.sleep(2)
        else:
            proc.terminate()
            pytest.fail(f"bridge at {url} not ready within {_BRIDGE_READY_TIMEOUT}s")
    else:
        pytest.skip(
            "real-bridge e2e needs OSPREY_REAL_BRIDGE_URL (running stack) "
            "or PHOEBUS_E2E_LAUNCH=1 (launch run_demo.sh here)"
        )

    os.environ["PHOEBUS_BRIDGE_URL"] = url
    yield url

    if proc is not None:
        proc.terminate()  # run_demo.sh traps TERM and tears down its children
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    os.environ.clear()
    os.environ.update(saved)


def _tool(name):
    from osprey.mcp_server.phoebus.tools import bridge_tools
    from tests.mcp_server.conftest import get_tool_fn

    return get_tool_fn(getattr(bridge_tools, name))


def _parse(result) -> dict:
    from tests.mcp_server.conftest import extract_response_dict

    return extract_response_dict(result)


async def _perceive_widgets(handle: str) -> list[dict]:
    out = _parse(await _tool("phoebus_perceive")(display=handle))
    assert out["status"] == "success"
    return out["perception"]["widgets"]


@pytest.fixture(scope="module")
async def demo_handle(real_bridge):
    """Open the demo panel on the REAL bridge and return its handle."""
    out = _parse(await _tool("phoebus_open_panel")(name="osprey_demo"))
    assert out["status"] == "success"
    assert out["handle"].startswith("handle:")
    assert out["ready"] is True, "display never reported ready=true after /open"
    return out["handle"]


async def test_reopen_is_stable_and_handle_resolves(real_bridge, demo_handle):
    # Phoebus reuses the existing DockItem when the same .bob is opened again,
    # so a re-open must return the SAME deterministic handle (distinct handles
    # for distinct displays are covered by the bridge's own unit/IT tests).
    out2 = _parse(await _tool("phoebus_open_panel")(name="osprey_demo"))
    assert out2["status"] == "success" and out2["ready"] is True
    assert out2["handle"] == demo_handle, (
        "re-opening an already-open .bob must resolve to the same handle"
    )
    ids = {d["id"] for d in _parse(await _tool("phoebus_list_displays")())["displays"]}
    assert demo_handle.split(":", 1)[1] in ids


async def test_perceive_by_handle_returns_live_widget_tree(demo_handle):
    # PVs may still be connecting right after ready=true — poll briefly.
    deadline = time.time() + _PV_SETTLE_TIMEOUT
    names: set = set()
    while time.time() < deadline:
        widgets = await _perceive_widgets(demo_handle)
        names = {w.get("name") for w in widgets}
        setpoint = next((w for w in widgets if w.get("name") == "Setpoint"), None)
        if setpoint and (setpoint.get("pv") or {}).get("value") not in (None, ""):
            break
        time.sleep(1)
    assert {"Setpoint", "Readback", "Status", "SetButton"}.issubset(names)
    setpoint = next(w for w in widgets if w["name"] == "Setpoint")
    # The bridge reports PV names scheme-qualified (e.g. "ca://DEMO:Setpoint").
    assert "DEMO:" in setpoint["pv"]["name"], "Setpoint must bind a DEMO:* PV"


async def test_drive_by_handle_moves_pv_ca_confirmed(demo_handle):
    current = float(_ca_get("DEMO:Setpoint"))
    target = 7.0 if abs(current - 7.0) > 0.01 else 8.0

    out = _parse(
        await _tool("phoebus_drive")(
            widget="Setpoint", verb="type", value=str(target), display=demo_handle
        )
    )
    assert out["status"] == "success" and out["fired"] is True, out

    # Verify way 1: independent CA readback (never touches Phoebus).
    deadline = time.time() + _PV_SETTLE_TIMEOUT
    rb = None
    while time.time() < deadline:
        rb = float(_ca_get("DEMO:Readback"))
        if rb == pytest.approx(target):
            break
        time.sleep(0.5)
    assert rb == pytest.approx(target), f"CA readback {rb} never reached {target}"
    # Enum reads may decode as index or string depending on the CA data type.
    assert _ca_get("DEMO:Status") in (0, "OK")  # "OK" (target < 80)

    # Verify way 2: the GUI's own view via a fresh perceive on the same handle.
    widgets = await _perceive_widgets(demo_handle)
    rb_widget = next(w for w in widgets if w["name"] == "Readback")
    assert str(rb_widget["pv"]["value"]).startswith(str(int(target)))


async def test_unknown_handle_is_clean_error_not_crash(real_bridge):
    with pytest.raises(ToolError):
        await _tool("phoebus_perceive")(display="handle:no-such-display")
    # The bridge must survive the bad request.
    assert _bridge_up(real_bridge), "bridge died after an unknown-handle request"


async def test_unknown_panel_name_is_clean_tool_error(real_bridge):
    with pytest.raises(ToolError, match="[Uu]nknown panel"):
        await _tool("phoebus_open_panel")(name="definitely-not-a-registered-panel")
