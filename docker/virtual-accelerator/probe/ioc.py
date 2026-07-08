"""Phase-1 probe IOC: toy PyAT line (1 corrector, 1 BPM) served over Channel
Access with pythonSoftIOC. Proves CA reachability from a container, not
physics fidelity -- see docker/virtual-accelerator/probe/README.md.
"""

import time

import at
import numpy as np
from softioc import asyncio_dispatcher, builder, softioc

# --- Minimal PyAT line: corrector -> drift -> BPM ---------------------------
hcm = at.elements.Corrector("HCM", length=0.0, kick_angle=[0.0, 0.0])
drift = at.elements.Drift("D1", length=2.0)
bpm = at.elements.Monitor("BPM")
ring = at.Lattice([hcm, drift, bpm], name="probe_ring", energy=1.5e9)
bpm_refpts = at.get_refpts(ring, "BPM")

CURRENT_TO_KICK = 1e-4  # rad per amp; arbitrary toy scale, not a real magnet


def bpm_x_mm(kick_rad: float) -> float:
    """Track a single on-axis particle through the line and return the
    horizontal position (mm) at the BPM for a given corrector kick angle."""
    hcm.KickAngle = [kick_rad, 0.0]
    r_in = np.zeros((6, 1))
    out = at.lattice_pass(ring, r_in, refpts=bpm_refpts)
    return float(out[0, 0, 0, 0]) * 1000.0


# --- EPICS records -----------------------------------------------------------
builder.SetDeviceName("PROBE")

hcm_rb = builder.aOut("HCM:CURRENT:RB", initial_value=0.0)
bpm_x = builder.aIn("BPM:POSITION:X", initial_value=bpm_x_mm(0.0))


def on_current_sp(value):
    hcm_rb.set(value)
    bpm_x.set(bpm_x_mm(value * CURRENT_TO_KICK))


builder.aOut("HCM:CURRENT:SP", initial_value=0.0, on_update=on_current_sp)

dispatcher = asyncio_dispatcher.AsyncioDispatcher()
builder.LoadDatabase()
softioc.iocInit(dispatcher)

print("probe IOC serving PVs: PROBE:HCM:CURRENT:SP/RB, PROBE:BPM:POSITION:X", flush=True)
while True:
    time.sleep(3600)
