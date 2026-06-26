# Test IOC Safety — Mandatory Port Isolation for EPICS

> **Read this in full before starting, stopping, or configuring any test IOC.**
>
> This module operates an EPICS test IOC (Input/Output Controller) on a machine that may be on the same network as a real accelerator's Channel Access infrastructure. A misconfigured test IOC will contaminate the production CA namespace, and in the worst case, interfere with machine operations. The rules below are not optional.

This module is **only available when `control_system.type == "epics"` AND `modules.test_ioc.enabled == true`** in `facility-config.yml`. If either condition is false, refuse any test-IOC action — don't be clever about it.

For control systems other than EPICS (DOOCS, TANGO, etc.), the equivalent test-harness patterns are out of scope for this skill.

---

## Why this matters

EPICS Channel Access broadcasts on UDP by default. A `softIoc` started with no port configuration binds to UDP 5064 (server) and 5065 (beacon) and sends beacons to the broadcast address. Any real IOC on the same network sees these beacons and may log warnings, attempt PV resolution against your test IOC, or in rare cases have CA client software misroute writes.

The risk is asymmetric:
- **If you're right** about isolation: nothing bad happens. Cost: two env vars.
- **If you're wrong** about isolation: a test PV name collision with a real PV can route a well-meaning analysis tool to read the wrong value — or worse, a write that happens to match a real PV name gets sent to production hardware.

**Err on the side of paranoid isolation.** Always.

---

## The six rules (ALL must be satisfied)

1. **Use exotic, non-standard CAS ports.** Never in the range 5064–5076 (EPICS default server + common beacon variants). The interview defaults to `59064`/`59065` — high enough that no production IOC in any facility's known configuration will collide. If your facility uses an unusual port range for production, pick something further away.

2. **Set BOTH server and beacon ports.** Setting only one causes the IOC to fall back to defaults for the other. The two ports that must be set:
   - `EPICS_CAS_SERVER_PORT` — where the IOC listens for CA connections
   - `EPICS_CAS_BEACON_PORT` — where the IOC sends beacons announcing its existence

3. **CA clients connecting to the test IOC must override `EPICS_CA_SERVER_PORT`.** The client-side default is 5064; without the override, clients search production CA first. Setting it is the client's responsibility, but the skill must document this every time it hands the user a test PV.

4. **All test PVs must use the facility's test prefix.** The interview captures this as `${config.modules.test_ioc.pv_prefix}` (default: `OSPREY:TEST:`). No exceptions — not even for quick one-offs. A PV without the prefix is indistinguishable from a production PV by name; the prefix is the last line of defense if ports somehow fail to isolate.

5. **Never start `softIoc -d file.db` without explicit port env vars.** This is the most common accident. The pattern below wraps `softIoc` in a script that sets the ports first. Use the script; do not run `softIoc` directly at a shell prompt.

6. **DB file constraints:**
   - DESC fields ≤ 39 ASCII characters (EPICS limit; longer values get silently truncated).
   - No UTF-8 multibyte characters (EPICS DB parser is ASCII-only).
   - No `$(...)` in comments — the DB parser treats these as substitutions even inside `#` lines.

---

## Correct startup pattern

Substitute values from `facility-config.yml`:

```bash
cat > ${config.modules.test_ioc.startup_script_path} << 'EOF'
#!/bin/bash
# Test IOC — isolated from production CA by exotic ports + prefix
set -euo pipefail

# Ports (from facility-config.yml modules.test_ioc)
export EPICS_CAS_SERVER_PORT=${config.modules.test_ioc.cas_server_port}
export EPICS_CAS_BEACON_PORT=${config.modules.test_ioc.cas_beacon_port}

# Optional: restrict beacons to loopback to belt-and-suspenders isolation
export EPICS_CAS_INTF_ADDR_LIST="127.0.0.1"
export EPICS_CAS_BEACON_ADDR_LIST="127.0.0.1"

# Navigate to DB dir and launch
cd ${config.deploy.project_path}/$(dirname ${config.modules.test_ioc.db_path})
exec softIoc -d $(basename ${config.modules.test_ioc.db_path})
EOF
chmod +x ${config.modules.test_ioc.startup_script_path}
```

Launch via `at` (or similar) so the script survives the SSH session:

```bash
echo "${config.modules.test_ioc.startup_script_path} > /tmp/test-ioc.log 2>&1" | at now
```

To connect a CA client:

```bash
export EPICS_CA_SERVER_PORT=${config.modules.test_ioc.cas_server_port}
# Optional: also restrict client discovery to loopback
export EPICS_CA_ADDR_LIST="127.0.0.1"
export EPICS_CA_AUTO_ADDR_LIST=NO
caget ${config.modules.test_ioc.pv_prefix}EXAMPLE:PV
```

---

## Validation before any action

Before starting, stopping, or reconfiguring a test IOC, verify:

```bash
# 1. Config is internally consistent
grep -A1 "cas_server_port\|cas_beacon_port" facility-config.yml

# 2. Chosen ports are outside the production-CA range
python3 -c "
import yaml
c = yaml.safe_load(open('facility-config.yml'))
s = c['modules']['test_ioc']['cas_server_port']
b = c['modules']['test_ioc']['cas_beacon_port']
assert s not in range(5064, 5077), f'cas_server_port {s} is in production CA range — REFUSE'
assert b not in range(5064, 5077), f'cas_beacon_port {b} is in production CA range — REFUSE'
assert s != b, 'server and beacon ports must differ'
print(f'ports OK: server={s}, beacon={b}')
"

# 3. PV prefix is non-empty and ends with a colon
python3 -c "
import yaml
c = yaml.safe_load(open('facility-config.yml'))
p = c['modules']['test_ioc']['pv_prefix']
assert p, 'pv_prefix is empty — REFUSE'
assert p.endswith(':'), f'pv_prefix {p} should end with colon for EPICS convention'
print(f'prefix OK: {p}')
"
```

If any of these fail, **refuse the action and tell the user what to fix.** Do not try to be helpful by "working around" a port conflict — the entire purpose of this module is to catch port conflicts.

---

## Stopping a test IOC

```bash
# Find it
pgrep -fa "softIoc.*${config.modules.test_ioc.db_path}"

# Stop cleanly (SIGTERM first, SIGKILL if it doesn't exit)
pkill -f "softIoc.*${config.modules.test_ioc.db_path}" || true
sleep 1
pkill -9 -f "softIoc.*${config.modules.test_ioc.db_path}" 2>/dev/null || true
```

Always verify it's gone:
```bash
pgrep -fa "softIoc" && echo "⚠ IOC still running" || echo "✓ stopped"
```

A lingering test IOC on the wrong ports is the exact failure mode this module is designed to prevent.

---

## DB file authoring rules (recap)

When you're asked to add new test PVs to `${config.modules.test_ioc.db_path}`:

- Descriptive names with the test prefix: `${config.modules.test_ioc.pv_prefix}TEMP:READING`.
- DESC field ≤ 39 ASCII chars. Truncate before saving if longer.
- No UTF-8 multibyte chars anywhere (°, µ, etc. — write "deg" and "u").
- No `$(...)` in comments. Use plain prose.
- Prefer `ai` / `ao` records for numeric values; `stringin` / `stringout` for text; `bi` / `bo` for booleans.
- For simulated time-varying values, use the built-in `sim` device support on `ai` records rather than a separate simulation IOC.

---

## Decision tree for common requests

| User says... | You do... |
|--------------|-----------|
| "Start the test IOC" | Read this file → validate → run startup script |
| "Add a PV to the test IOC" | Edit DB file following rules above → restart IOC |
| "I'm getting weird CA errors" | Check `pgrep softIoc` for stray IOCs on default ports — that's almost always the root cause |
| "Can I use port 5064 just this once?" | **No.** Not negotiable. Refuse and explain |
| "I want to test against production PVs" | That's not a test IOC; use the read-only production connection. This module is for isolated simulated PVs only |
| "The test IOC works but production IOCs started warning" | Stop the test IOC immediately. Check its port config — something is misconfigured. Do not restart until you know why |
| "Our facility uses DOOCS / TANGO, not EPICS" | This module does not apply. Refuse and explain |

---

## Why the defaults are what they are

- **59064/59065**: high enough that no known EPICS production deployment uses them. Contiguous pair (convention: server and beacon differ by 1). Well outside the 5064–5076 production range.
- **`OSPREY:TEST:` prefix**: visually distinct from real PV names (most facilities use all-uppercase single-word prefixes without colons in the first segment). Easy to grep for. Matches ALS convention.
- **`EPICS_CAS_INTF_ADDR_LIST="127.0.0.1"`**: belt-and-suspenders — even if someone mistypes the server port, CA server only binds to loopback so nothing leaks onto the control network.

If the facility wants different defaults, the interview captures them in `facility-config.yml`. The rules above still apply regardless of the specific values chosen.
