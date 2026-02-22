# Mock Archiver Reproducibility Issue

**Date identified:** 2026-02-20
**File:** `src/osprey/connectors/archiver/mock_archiver_connector.py`
**Status:** Not yet fixed

## Problem

The mock archiver generates different-looking plots every time the MCP server process restarts (e.g., after `osprey init` or project recreation), even for the same PV names and time ranges.

## Root Cause

### BPM channels (line 150)

```python
rng = np.random.default_rng(seed=hash(pv_name) % (2**32))
```

Python 3 randomizes `hash()` per process via `PYTHONHASHSEED`. Each new MCP server process produces a different hash for the same PV name string, which changes:
- `offset` — the BPM equilibrium position (line 159)
- `phase` — the sinusoidal phase (line 162)
- `frequency` — the oscillation frequency (line 165)

**Within** a single process, repeated calls for the same PV are reproducible. **Across** process restarts, they are not.

### Non-BPM channels (line 218)

```python
noise = np.random.normal(0, noise_amplitude, num_points)
```

Uses bare `np.random.normal()` with no seed at all — different on every call, even within the same process.

## Proposed Fix

### 1. Use `hashlib` instead of `hash()` for deterministic seeding

Replace (line 150):
```python
rng = np.random.default_rng(seed=hash(pv_name) % (2**32))
```

With:
```python
import hashlib
seed = int(hashlib.sha256(pv_name.encode()).hexdigest(), 16) % (2**32)
rng = np.random.default_rng(seed=seed)
```

`hashlib.sha256` is stable across processes, Python versions, and machines.

### 2. Seed the non-BPM paths too

Apply the same deterministic seeding to all PV types (lines 216-218), so that current, voltage, pressure, temperature, lifetime, etc. also produce reproducible noise across restarts.

### 3. Consider scope

Decide whether to also make the data stable across **time ranges** — currently `t = np.linspace(0, 1, num_points)` normalizes time to [0, 1], so requesting a different window for the same PV produces a completely different waveform shape mapped to different absolute times. A more realistic mock might anchor the sinusoidal phase to absolute time (e.g., use Unix timestamps) so that overlapping queries return consistent data.

## Testing

After fixing, verify:
1. Same PV + same time range → identical DataFrame across process restarts
2. Different PV names → visually distinct traces
3. BPM traces still look realistic (±100 µm equilibrium, ±10 µm oscillation)
4. Non-BPM traces still look realistic for their respective types
