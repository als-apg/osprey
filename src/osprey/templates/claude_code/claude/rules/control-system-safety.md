# Control System Safety — EPICS Channel Access

When writing Python code that interacts with the control system, you MUST use
the `osprey.runtime` API. Direct use of EPICS libraries is **prohibited** because
it bypasses safety layers (limits checking, write verification, approval).

### Allowed

```python
from osprey.runtime import read_channel, write_channel
value = read_channel("SR:BPM01:XPosition")
write_channel("SR:PS01:Current", 150.0)
```

### Prohibited

```python
# DO NOT use these — they bypass safety controls
import epics
epics.caget("SR:BPM01:XPosition")   # Bypasses audit logging
epics.caput("SR:PS01:Current", 150)  # Bypasses limits + approval
from epics import PV
pv = PV("SR:PS01:Current")
pv.put(150)  # Bypasses all safety layers
```

### Why This Matters

The `osprey.runtime` API enforces:
1. **Channel limits** — prevents out-of-range writes (channel_limits.json)
2. **Write verification** — confirms the control system accepted the value
3. **Human approval** — routes writes through the approval workflow
4. **Audit logging** — records all operations for traceability

Direct EPICS calls skip all of these protections.

## Write Operations

For scripts that write to the control system, use the `execute` MCP tool directly
with `execution_mode: "write"` —
the `execute` tool defaults to readonly mode, which blocks write operations.
