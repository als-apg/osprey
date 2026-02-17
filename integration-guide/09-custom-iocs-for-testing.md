# Recipe 9: Creating Custom IOCs for Testing

## When You Need This

Your tool reads PVs from the machine (via `channel_read` or directly through the EPICS connector). To test end-to-end without real hardware, you create a **caproto soft IOC** that serves realistic simulated data over Channel Access — the same protocol the real machine uses.

OSPREY already has a complete IOC generation pipeline. This recipe shows how to use it for your tool.

## What You Get

```
Your Tool (MCP server)
    ↓ channel_read("SR01C:BPM1:X")
OSPREY Control System Connector
    ↓ epics.caget("SR01C:BPM1:X")
caproto Soft IOC (localhost:5064)
    ↓ pvproperty serves simulated value
Custom Backend (your physics model)
```

The soft IOC speaks real EPICS Channel Access. Your tool doesn't know the difference between it and the actual accelerator.

## Prerequisites

Both are already in OSPREY's dependencies (`pyproject.toml`):

```
caproto    — Python EPICS server (soft IOC)
pyepics    — Python EPICS client (PV reads/writes)
```

## The Three-Layer Architecture

OSPREY's IOC system has three decoupled layers:

### Layer 1: Channel Database (what PVs exist)

A JSON file listing all PV names, types, and metadata. Four formats are supported — for a new tool, **flat** is usually sufficient:

```json
{
  "_metadata": {
    "description": "TxT BPM channels for testing",
    "facility": "ALS",
    "pv_count": 12
  },
  "channels": [
    {
      "name": "SR01C:BPM1:X",
      "description": "Horizontal position, sector 1 BPM 1",
      "units": "mm",
      "data_type": "float",
      "precision": 6
    },
    {
      "name": "SR01C:BPM1:Y",
      "description": "Vertical position, sector 1 BPM 1",
      "units": "mm",
      "data_type": "float",
      "precision": 6
    },
    {
      "name": "SR:BEAM:CURRENT",
      "description": "Stored beam current",
      "units": "mA",
      "data_type": "float",
      "precision": 3
    },
    {
      "name": "SR:TUNE:H",
      "description": "Horizontal tune",
      "units": "",
      "data_type": "float",
      "precision": 6
    },
    {
      "name": "SR:TUNE:V",
      "description": "Vertical tune",
      "units": "",
      "data_type": "float",
      "precision": 6
    }
  ]
}
```

For TxT studies, you'd list all BPM PVs (X and Y for each BPM), plus tunes, beam current, and any corrector magnets you need.

**Supported data types:** `float`, `int`, `enum`, `string`, `array`

For array PVs (turn-by-turn data is naturally an array):

```json
{
  "name": "SR01C:BPM1:TBT:X",
  "description": "Turn-by-turn horizontal position, 1024 turns",
  "units": "mm",
  "data_type": "array",
  "count": 1024,
  "precision": 6
}
```

### Layer 2: Generated IOC (the caproto server)

Run the CLI command to generate the IOC Python file:

```bash
osprey generate soft-ioc \
    --channel-db data/my_channels.json \
    --name txt_test_ioc \
    --output-dir generated_iocs/
```

This produces `generated_iocs/txt_test_ioc_ioc.py` — a standalone Python script that:
- Defines a `caproto.server.PVGroup` with all your PVs as `pvproperty` members
- Includes a simulation loop (configurable update rate)
- Loads backends from config at runtime
- Handles setpoint → readback pairing
- Provides a heartbeat PV for liveness monitoring

**You rarely need to edit the generated file.** Behavior is controlled by backends.

### Layer 3: Simulation Backend (how PVs behave)

Backends define the physics. They're loaded at **runtime** from `config.yml`, so you can change behavior without regenerating the IOC.

#### Built-in Backends

**`mock_style`** — keyword-based heuristic simulation (good for quick testing):

| PV Name Contains | Simulated Value | Behavior |
|-----------------|-----------------|----------|
| `CURRENT` (beam) | 500 mA | Exponential decay with periodic refills |
| `BPM`, `POSITION` | 0.0 mm | Random equilibrium + slow drift + noise |
| `VOLTAGE` | 5000 V | Stable + small oscillation |
| `PRESSURE` | 1e-9 Torr | Gradual increase |
| `TEMPERATURE` | 25°C | Gradual increase |
| `TUNE` | 0.234 / 0.317 | Stable + tiny noise |
| `LIFETIME` | 10 hours | Linear decrease |
| (default) | 100.0 | Linear trend |

**`passthrough`** — PVs hold whatever value was last written. No simulation. Good for testing write operations.

#### Custom Backends

For realistic TxT testing, you'll want a custom backend that generates physically meaningful data. Create a Python file implementing the `SimulationBackend` protocol:

```python
"""Custom backend for TxT BPM simulation."""

import math
import numpy as np


class TxTBackend:
    """Generates turn-by-turn BPM data with realistic optics.

    Simulates betatron oscillation with:
    - Configurable tunes (fractional)
    - BPM-specific beta functions and phase advances
    - Amplitude decoherence
    - Noise floor
    """

    def __init__(self, pairings: dict, **params):
        self.pairings = pairings
        self.h_tune = params.get("h_tune", 0.234)
        self.v_tune = params.get("v_tune", 0.317)
        self.num_turns = params.get("num_turns", 1024)
        self.noise_rms = params.get("noise_rms", 0.010)  # mm
        self.kick_amplitude = params.get("kick_amplitude", 1.0)  # mm
        self.decoherence_turns = params.get("decoherence_turns", 500)
        self._values = {}
        self._turn_counter = 0

    def initialize(self, pv_definitions: dict) -> dict:
        """Set initial PV values."""
        initial = {}
        for pv_name in pv_definitions:
            if ":TBT:" in pv_name:
                # Generate turn-by-turn array
                initial[pv_name] = self._generate_tbt_data(pv_name)
            elif ":TUNE:" in pv_name:
                if ":H" in pv_name:
                    initial[pv_name] = self.h_tune
                elif ":V" in pv_name:
                    initial[pv_name] = self.v_tune
            elif ":BPM" in pv_name:
                # Static orbit position
                initial[pv_name] = np.random.normal(0.0, 0.1)
        self._values.update(initial)
        return initial

    def on_write(self, pv_name: str, value) -> dict:
        """Handle PV writes — update paired readbacks."""
        updates = {}
        if pv_name in self.pairings:
            rb_name = self.pairings[pv_name]
            updates[rb_name] = value
        self._values[pv_name] = value
        return updates

    def step(self, dt: float) -> dict:
        """Time-evolved updates (called at update_rate Hz)."""
        updates = {}
        self._turn_counter += 1

        # Regenerate TBT data periodically (simulates new acquisition)
        if self._turn_counter % 100 == 0:
            for pv_name in self._values:
                if ":TBT:" in pv_name:
                    updates[pv_name] = self._generate_tbt_data(pv_name)

        return updates

    def _generate_tbt_data(self, pv_name: str) -> list[float]:
        """Generate a realistic TBT trace for one BPM."""
        turns = np.arange(self.num_turns)

        # Select tune based on plane
        if ":X" in pv_name or ":H" in pv_name:
            tune = self.h_tune
        else:
            tune = self.v_tune

        # BPM-specific phase (hash PV name for reproducibility)
        phase = (hash(pv_name) % 1000) / 1000.0 * 2 * math.pi

        # Betatron oscillation with decoherence
        envelope = self.kick_amplitude * np.exp(-turns / self.decoherence_turns)
        signal = envelope * np.cos(2 * math.pi * tune * turns + phase)

        # Add noise floor
        noise = np.random.normal(0, self.noise_rms, self.num_turns)

        return (signal + noise).tolist()
```

Save this as e.g. `backends/txt_backend.py` and reference it in config:

```yaml
simulation:
  channel_database: "data/txt_channels.json"
  ioc:
    name: "txt_test_ioc"
    port: 5064
    output_dir: "generated_iocs/"
  base:
    type: "mock_style"
    noise_level: 0.01
    update_rate: 10.0
  overlays:
    - file_path: "backends/txt_backend.py"
      class_name: "TxTBackend"
      params:
        h_tune: 0.234
        v_tune: 0.317
        num_turns: 1024
        noise_rms: 0.010
        kick_amplitude: 1.0
        decoherence_turns: 500
```

The overlay wins over the base for any PV it handles. PVs not handled by the overlay fall through to `mock_style`.

### Backend Chaining

Multiple overlays are composed via `ChainedBackend`:

```yaml
overlays:
    - file_path: "backends/txt_backend.py"
      class_name: "TxTBackend"
      params: { h_tune: 0.234 }
    - file_path: "backends/noise_injector.py"
      class_name: "NoiseInjector"
      params: { level: 0.05 }
```

Priority: **last overlay wins** on conflicts. All `step()` methods run in order.

## Running the IOC

```bash
# Generate the IOC (one-time)
osprey generate soft-ioc --channel-db data/txt_channels.json --name txt_test

# Run it
python generated_iocs/txt_test_ioc.py

# Or with custom config
python generated_iocs/txt_test_ioc.py --config config.yml
```

The IOC starts listening on `localhost:5064` (or configured port).

## Connecting Your Tool to the IOC

### Option A: Switch Connector to `epics` (real Channel Access)

In `config.yml`, point the control system to your local IOC:

```yaml
control_system:
  type: epics
  connector:
    epics:
      timeout: 5.0
      gateways:
        read_only:
          address: localhost
          port: 5064
          use_name_server: false
```

Your MCP tools call `channel_read` → EPICS connector → real Channel Access → caproto IOC. This tests the full stack including network protocol.

### Option B: Use Mock Connector (no IOC needed)

For fast unit tests that don't need real CA:

```yaml
control_system:
  type: mock
```

The mock connector generates synthetic values from PV name heuristics. No IOC process needed. Fastest path for CI.

### Option C: Direct caproto Client (bypass OSPREY connector)

For integration tests that need fine-grained control:

```python
from caproto.threading.client import Context

ctx = Context()
pvs = ctx.get_pvs("SR01C:BPM1:X", "SR01C:BPM1:Y", timeout=5)
x_data = pvs[0].read()
y_data = pvs[1].read()
```

## Setpoint/Readback Pairings

For PVs with setpoint and readback variants (common in magnet control):

```json
{
  "channels": [
    { "name": "MAG:HCM01:CURRENT:SP", "description": "HCM setpoint" },
    { "name": "MAG:HCM01:CURRENT:RB", "description": "HCM readback" }
  ],
  "pairings": {
    "MAG:HCM01:CURRENT:SP": "MAG:HCM01:CURRENT:RB"
  }
}
```

When a value is written to `:SP`, the backend's `on_write()` automatically updates `:RB` (optionally with dynamics like first-order response).

## Using Previous Run Data

If you have real MudLab data from previous runs (e.g., saved BPM orbit data, tune measurements, TBT traces), you can create a **replay backend** that serves recorded data:

```python
"""Backend that replays recorded measurement data."""

import json
from pathlib import Path


class ReplayBackend:
    """Serves PV values from a recorded data file.

    Data file format (JSON):
    {
        "SR01C:BPM1:X": [0.123, 0.125, 0.121, ...],
        "SR01C:BPM1:Y": [-0.045, -0.043, ...],
        ...
    }
    """

    def __init__(self, pairings: dict, **params):
        self.pairings = pairings
        data_file = params.get("data_file", "data/recorded_orbits.json")
        self._data = json.loads(Path(data_file).read_text())
        self._index = 0

    def initialize(self, pv_definitions: dict) -> dict:
        initial = {}
        for pv_name in pv_definitions:
            if pv_name in self._data:
                values = self._data[pv_name]
                if isinstance(values, list) and len(values) > 0:
                    initial[pv_name] = values[0]
                else:
                    initial[pv_name] = values
        return initial

    def on_write(self, pv_name: str, value) -> dict:
        return {}  # Read-only replay

    def step(self, dt: float) -> dict:
        """Advance to next recorded sample."""
        self._index += 1
        updates = {}
        for pv_name, values in self._data.items():
            if isinstance(values, list) and len(values) > 0:
                idx = self._index % len(values)
                updates[pv_name] = values[idx]
        return updates
```

Config:

```yaml
overlays:
    - file_path: "backends/replay_backend.py"
      class_name: "ReplayBackend"
      params:
        data_file: "data/mudlab_recorded_orbits.json"
```

## Converting MudLab Data

MudLab likely stores data in MATLAB `.mat` files. Convert to JSON for the replay backend:

```python
"""Convert MudLab .mat files to OSPREY replay format."""

import json
from pathlib import Path
import scipy.io as sio
import numpy as np


def mat_to_replay_json(mat_path: str, output_path: str, pv_prefix: str = "SR"):
    """Convert a MudLab .mat file to OSPREY replay JSON.

    Adapts MATLAB struct field names to EPICS PV naming conventions.
    """
    data = sio.loadmat(mat_path, squeeze_me=True)

    replay = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue  # Skip MATLAB metadata

        # Convert MATLAB field name to EPICS PV name
        # Adapt this mapping to your MudLab naming convention
        pv_name = f"{pv_prefix}:{key.upper()}"

        if isinstance(value, np.ndarray):
            replay[pv_name] = value.tolist()
        elif isinstance(value, (int, float)):
            replay[pv_name] = value

    Path(output_path).write_text(json.dumps(replay, indent=2))
    print(f"Wrote {len(replay)} PVs to {output_path}")


if __name__ == "__main__":
    mat_to_replay_json(
        "mudlab_data/2024_run_03.mat",
        "data/mudlab_recorded_orbits.json",
        pv_prefix="SR"
    )
```

## Integration Test Pattern

```python
"""Integration test: TxT tool against caproto soft IOC."""

import asyncio
import subprocess
import time

import pytest


@pytest.fixture(scope="module")
def ioc_server():
    """Start soft IOC as subprocess for testing."""
    proc = subprocess.Popen(
        ["python", "generated_iocs/txt_test_ioc.py"],
        env={
            **os.environ,
            "EPICS_CA_SERVER_PORT": "5064",
            "EPICS_CA_ADDR_LIST": "localhost",
            "EPICS_CA_AUTO_ADDR_LIST": "NO",
        },
    )

    # Wait for IOC to be ready
    time.sleep(3)
    yield proc

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture
def epics_config(tmp_path, monkeypatch):
    """Config pointing to local IOC."""
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yml"
    config.write_text("""
control_system:
  type: epics
  writes_enabled: false
  connector:
    epics:
      timeout: 5.0
      gateways:
        read_only:
          address: localhost
          port: 5064
          use_name_server: false
""")
    return config


@pytest.mark.integration
@pytest.mark.asyncio
async def test_channel_read_from_ioc(ioc_server, epics_config):
    """Read BPM data from soft IOC via OSPREY connector."""
    from osprey.mcp_server.control_system.registry import (
        initialize_mcp_registry,
        get_mcp_registry,
    )

    initialize_mcp_registry()
    registry = get_mcp_registry()
    connector = await registry.control_system()

    result = await connector.read_channel("SR01C:BPM1:X")
    assert result.value is not None
    assert isinstance(result.value, float)
    assert result.metadata.units == "mm"
```

See `tests/integration/test_soft_ioc_server.py` (1653 lines) for comprehensive examples including:
- IOC lifecycle management (start, health check, shutdown)
- Backend verification (tracking calls to `initialize`, `on_write`, `step`)
- Multi-PV reads
- Write → readback verification
- Custom backend parameter passing
- Chained backend composition

## Facility Presets

OSPREY includes gateway presets for quick switching:

| Preset | Address | Port | Use Case |
|--------|---------|------|----------|
| **Simulation** | `localhost` | `5064` | caproto soft IOC (run `osprey generate soft-ioc` first) |
| **ALS** | `cagw-alsdmz.als.lbl.gov` | `5064` (read) / `5084` (write) | Production ALS accelerator |
| **APS** | `pvgatemain1.aps4.anl.gov` | `5064` | Argonne APS |

Switch between them by changing `control_system.type` and `connector.epics.gateways` in `config.yml`.

## Concrete Reference

- `src/osprey/generators/soft_ioc_template.py` — IOC code generator (523 lines)
- `src/osprey/generators/ioc_backends.py` — Backend protocol + built-in backends (368 lines)
- `src/osprey/connectors/control_system/mock_connector.py` — Mock data heuristics (466 lines)
- `src/osprey/connectors/control_system/epics_connector.py` — Production EPICS connector
- `src/osprey/connectors/control_system/base.py` — Abstract connector (ChannelValue, WriteVerification)
- `tests/integration/test_soft_ioc_server.py` — Full integration test suite (1653 lines)
- `docs/source/developer-guides/02_quick-start-patterns/05_soft-ioc-backends.rst` — Backend guide
- `src/osprey/templates/apps/control_assistant/data/channel_limits.json.j2` — Limits database template

## Checklist

- [ ] Channel database JSON file with all PVs your tool needs
- [ ] IOC generated via `osprey generate soft-ioc`
- [ ] Backend chosen: `mock_style` for quick tests, custom for realistic physics
- [ ] Config.yml updated: `control_system.type: epics` pointing to `localhost:5064`
- [ ] Setpoint/readback pairings defined (if your tool writes)
- [ ] Integration test with IOC subprocess fixture
- [ ] MudLab `.mat` data converted to replay JSON (if using recorded data)
- [ ] Channel limits database updated with your tool's PVs (if writes are involved)
