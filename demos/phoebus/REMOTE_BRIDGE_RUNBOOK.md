# Remote Bridge Runbook — isolated Phoebus + OSPREY on appsdev2

Operate the **osprey-phoebus-remote-bridge** demo: an isolated, headless
Phoebus product on appsdev2 that the OSPREY agent opens, perceives, and drives
over the loopback bridge, with the live panel streamed into the OSPREY web
terminal via Xvnc → websockify → noVNC.

Everything below was validated live on appsdev2 on 2026-07-01: pre-flight
guard, launch, real-bridge e2e (5/5), the noVNC stream (visual framebuffer
check), the web-terminal `/panel/phoebus` embed path (HTTP + binary
WebSocket), and clean teardown with the production instances untouched.

---

## 0. Fixed values (recon-chosen, non-colliding)

| Resource | Value |
|---|---|
| Bridge HTTP | `127.0.0.1:7979` |
| X display / RFB | `:99` / `127.0.0.1:5977` (`:99`'s default 5999 is squatted by a host VNC) |
| Phoebus `-server` | `4079` (prod instances use 4001/4004/4005) |
| Demo CA server / repeater | `5074` / `5075` (UDP+TCP; host prod CA uses 5064/5065 — never touched) |
| websockify (noVNC HTTP+WS) | `127.0.0.1:6080` |
| Xvnc framebuffer | `1600x900` (`DEMO_GEOMETRY`); the launcher seeds a memento pinning the Phoebus window to fill it exactly, so the stream shows ONLY Phoebus |
| CA scrub | `EPICS_CA_ADDR_LIST=127.0.0.1`, `EPICS_CA_AUTO_ADDR_LIST=NO` (host default points at production 131.243.x subnets — the launcher overrides) |

Per-instance values shift by `INSTANCE-1` (§2b): instance 2 = bridge **7980**,
display `:100`, RFB **5978**, `-server` **4080**, websockify **6081**. The CA
ports (5074/5075) are shared — instance 2 is a plain CA client of instance 1's
soft IOC.

**Production baseline that must stay untouched** (verify with `ps -p`, NOT
`kill -0` — cross-user `kill -0` returns EPERM and looks like "gone"):

```bash
ps -p 49901,344076,1805779 -o pid,user,etime,args   # nusaqib/mjchin/hunt Phoebus
```

## 1. One-time staging layout (already in place)

```
~/phoebus-remote-bridge-demo/
├── phoebus/…/target/product-6.0.0-SNAPSHOT.jar + lib/   # bridge-enabled product (built on laptop)
│   └── lib-mac-aarch64-quarantine/                      # mac JavaFX jars moved OUT of lib/
├── osprey/            # git worktree of ~/projects/osprey @ feature/phoebus-demo-profile
├── novnc/             # vendored noVNC 1.6.0 client (vnc.html, app/, core/, vendor/)
└── pytools/           # pure-python wheels unzipped (pytest, pytest-asyncio, websockify)
```

* **JavaFX platform jars:** the product was built on macOS, so its `lib/`
  needed the seven `javafx-*-21.0.7-linux.jar` (from Maven Central) and the
  `*-mac-aarch64.jar` quarantined. `run_demo.sh` launches with
  `-cp "jar:lib/*"` (not `-jar`) precisely so *what is in `lib/` defines the
  runtime set* — the jar's manifest pins the build platform's jars and cannot
  be overridden with `-Djavafx.platform` at build time.
  Rebuild + reship after phoebus changes:
  `JAVA_HOME=/opt/homebrew/opt/openjdk@21 mvn -pl phoebus-product -am install -DskipTests`
  (laptop), then scp the product jar; `lib/` linux jars only need reshipping
  when `openjfx.version` changes.
* **Sync osprey code** (no shared remote involved): from the laptop
  `git push ssh://appsdev2/~/projects/osprey feature/phoebus-demo-profile:refs/heads/demo-incoming -f`,
  then on appsdev2 `cd ~/phoebus-remote-bridge-demo/osprey && git merge --ff-only demo-incoming`.
  (Direct push to the branch is refused once the worktree has it checked out.)

## 2. Launch

```bash
ssh appsdev2
cd ~/phoebus-remote-bridge-demo/osprey
env PHOEBUS_REPO=$HOME/phoebus-remote-bridge-demo/phoebus \
    JAVA_HOME=/home/als/alsbase/Phoebus/phoebus/jdk \
    ISOLATED=1 \
    OSPREY_PYTHON=$HOME/projects/osprey/.venv/bin/python \
    WEBSOCKIFY_CMD="env PYTHONPATH=$HOME/phoebus-remote-bridge-demo/pytools $HOME/projects/osprey/.venv/bin/python -m websockify" \
    bash demos/phoebus/run_demo.sh
```

* `ISOLATED=1` is the auto-default on Linux, but pin it anyway
  (belt-and-suspenders on a shared box).
* The **pre-flight collision guard** aborts if display `:99` or any of
  7979/4079/5977/5074/5075/6080 is occupied — that abort is the safety
  feature, not a bug. See "Troubleshooting" for the usual culprit.
* Success banner lists IOC, Phoebus, bridge, VNC, and noVNC endpoints.
  Logs: `/tmp/osprey_demo_{run,ioc→osprey_demo_ioc,phoebus,websockify}.log`.

## 2b. Optional second Phoebus instance

Same command as §2, prefixed with `INSTANCE=2` — in a second terminal, with
instance 1 already up (instance 2 skips the IOC and shares instance 1's demo
soft IOC; its pre-flight expects the CA ports to be busy and checks only its
own offset ports):

```bash
env INSTANCE=2 PHOEBUS_REPO=… JAVA_HOME=… ISOLATED=1 WEBSOCKIFY_CMD="…" \
    bash demos/phoebus/run_demo.sh     # no OSPREY_PYTHON needed — no IOC leg
```

Logs get a `.i2` suffix (`/tmp/osprey_demo_phoebus.i2.log`, …). To surface it
in the web terminal and give the agent a second toolset, uncomment the
`phoebus2` block in the project's `config.yml` (see the `phoebus-standalone`
preset; the server is declared as an `extends` clone of the framework server —
`claude_code.servers.phoebus2: {extends: phoebus, env: {PHOEBUS_BRIDGE_URL:
"${PHOEBUS2_BRIDGE_URL:-http://127.0.0.1:7980}"}}` — NOT the old
`servers.phoebus2.enabled: true` form, which no longer exists) and restart the
standalone terminal: PHOEBUS2 tab via `/panel/phoebus2`, tools via the
`phoebus2` MCP server (bridge 7980, `mcp__phoebus2__*`, drive approval-gated
like instance 1).

## 2c. Live machine data, read-only (`LIVE_READONLY=1`)

By default the sandbox sees ONLY the demo IOC — a real panel (e.g. GTLView)
renders with every PV stuck `CONNECTING`. To get live values, add
`LIVE_READONLY=1` to the §2/§2b launch env. This appends the ALS **read-only
CA gateway** (`cagw-alsdmz.als.lbl.gov`, the same host ALS's own
`readOnlyPhoebus.sh` uses) to the *client* search path:

```
EPICS_CA_ADDR_LIST="127.0.0.1:5074 cagw-alsdmz.als.lbl.gov:5064"
EPICS_PVA_ADDR_LIST="cagw-alsdmz.als.lbl.gov"
```

* The gateway grants `read, no write` (CA access security) on every channel —
  verified with `cainfo`. Writes are impossible **at the network layer**, so
  even a human clicking an ON/OFF button in the noVNC panel (which bypasses
  the MCP `phoebus_drive` gate) cannot actuate hardware. Phoebus renders such
  channels as non-writable.
* The demo IOC keeps working via the explicit `127.0.0.1:5074` entry.
* Server-side (`EPICS_CAS_*`) interfaces stay loopback in every mode.
* Override the gateway host with `READONLY_GATEWAY=<host>` if it ever moves.
* Quick sanity check of the resolved env without launching anything:
  `PRINT_EPICS_ENV=1 bash demos/phoebus/run_demo.sh`.

## 3. Verify (30 seconds)

```bash
curl -s http://127.0.0.1:7979/displays | jq .          # d-1 ready:true
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:6080/vnc.html   # 200
```

Full regression gate — the **real-bridge e2e** (open→perceive→drive→CA readback):

```bash
cd ~/phoebus-remote-bridge-demo/osprey
env PYTHONPATH=$PWD/src:$HOME/phoebus-remote-bridge-demo/pytools \
    OSPREY_REAL_BRIDGE_URL=http://127.0.0.1:7979 \
    EPICS_CA_ADDR_LIST=127.0.0.1 EPICS_CA_AUTO_ADDR_LIST=NO \
    EPICS_CA_SERVER_PORT=5074 EPICS_CA_REPEATER_PORT=5075 \
    ~/projects/osprey/.venv/bin/python -m pytest tests/e2e/test_phoebus_real_bridge_e2e.py -q
```

`PYTHONPATH=$PWD/src` is required: the shared `.venv` has osprey installed
editable from `~/projects/osprey` (branch `next`), and the worktree's demo
code must shadow it.

## 4. Watch the panel (operator view)

**Direct (quickest):** from the laptop
`ssh -f -N -L 16080:127.0.0.1:6080 appsdev2`, then open
`http://localhost:16080/vnc.html?autoconnect=1&resize=scale`.
(A SOCKS `DynamicForward` proxy works too — browse `http://127.0.0.1:6080/…`
through it.)

**Inside the OSPREY web terminal (PHOEBUS tab):** run a standalone terminal on
a free port (NOT the containerized 9090–9096) from a project built with the
`phoebus-standalone` preset — the preset ships the PHOEBUS panel wired to
`/panel/phoebus/vnc.html` with the RFB WebSocket riding the same proxy, so the
browser only needs to reach the web terminal port. Forward that one port and
open the PHOEBUS tab.

Launch it with `PYTHONPATH=~/phoebus-remote-bridge-demo/osprey/src` so the
demo-branch code shadows the shared `.venv`'s editable install (branch `next`)
— same trap as the e2e in §3. Without it the `/panel/...` proxy routes are
missing (404) and the WS relay is not binary-safe.

**Security note (accepted for the demo):** the VNC stream has **no
authentication** (`-SecurityTypes None`). It is bound to loopback only and
shows/controls only the demo panel + demo soft IOC (`DEMO:*`), but anyone with
a shell on appsdev2 can connect to it. Do not reuse this pattern for panels
bound to real PVs.

## 5. Drive it

Via the agent (in a built phoebus-demo project: `claude` →
"run the phoebus walkthrough"), or by hand:

```bash
curl -s -X POST http://127.0.0.1:7979/drive -H "Content-Type: application/json" \
  -d '{"display":"handle:d-1","widget":"Setpoint","verb":"type","value":"42.5","mode":"synthetic"}'
caget DEMO:Readback   # ..with EPICS_CA_SERVER_PORT=5074 EPICS_CA_ADDR_LIST=127.0.0.1
```

The typed value lands in the panel (visible in the stream), mirrors to
`DEMO:Readback`, and derives `DEMO:Status` (≥80 WARN, ≥95 FAULT).
`phoebus_drive` from the agent stays behind the approval hook.

## 6. Teardown + "untouched" verification

Ctrl-C the launcher (or `kill <runner pid>` — its trap kills only its own
children). Then verify:

```bash
# 1. nothing of ours is left listening
ss -tln | grep -E ':(7979|4079|5977|5074|6080)\b' || echo CLEAN
ls /tmp/.X11-unix/X99 2>/dev/null || echo "X99 gone"
# 2. the known stragglers (see Troubleshooting)
ss -ulnp | grep ':5075\b' || echo "no repeater leftover"
pgrep -u $USER -f phoebus-remote-bridge-demo || echo "no demo JVM leftover"
# 3. production untouched (ps -p, not kill -0)
ps -p 49901,344076,1805779 -o pid,user,etime
ss -tln | grep -E ':(4001|4004|4005)\b'      # three prod -server ports still bound
```

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Pre-flight abort: "CA repeater port 5075 already in use" | A demo **Phoebus JVM** or caproto repeater from a previous run lingers holding UDP 5075 (Phoebus's CA client becomes the repeater when none exists; it can survive teardown holding only that socket). `pid=$(ss -ulnp \| grep ':5075\b' \| grep -oP 'pid=\K[0-9]+')` → confirm it's yours (`ps -p $pid`) → `kill $pid`. |
| `Error initializing QuantumRenderer: no suitable pipeline` | `lib/` holds the wrong-platform JavaFX jars — see §1. |
| `NoClassDefFoundError: javafx/application/Application` | Launched with `-jar` (manifest classpath) instead of the script's `-cp` path, with mac jars quarantined. Use `run_demo.sh`. |
| "RVNC CONNECT" chrome in the stream, or a huge desktop (e.g. 5120×2880) with Phoebus in a corner | `Xvnc` on PATH is **RealVNC's** (its package symlinks `/usr/bin/Xvnc` → `Xvnc-realvnc` and renames TigerVNC's binary to `Xvnc.conflict`). RealVNC ignores `-geometry` in virtual mode and paints `vncserverui` overlays. The launcher auto-detects this and uses `Xvnc.conflict`; override with `XVNC_CMD=<path-to-TigerVNC-Xvnc>`. |
| noVNC tab shows connect error | The stream leg was skipped (websockify/noVNC assets missing — see the launch log line) or `WEB_VNC_PORT` collided. Bridge + agent still fully work. |
| e2e skipped | It's opt-in: set `OSPREY_REAL_BRIDGE_URL` (or `PHOEBUS_E2E_LAUNCH=1` to have the test launch the stack itself). |
| `pkill -f run_demo.sh` kills your own ssh session | The pattern matches your remote shell's own command line. Use `pkill -f 'run_demo[.]sh'` or kill the runner PID. |

## 8. Safety invariants (why this can't disturb production)

1. **Pre-flight guard** refuses to start on ANY resource collision.
2. All listeners are **loopback-only** (bridge, Xvnc RFB, websockify, CA).
3. **Dedicated CA ports** (5074/5075) + `EPICS_CA_ADDR_LIST=127.0.0.1` +
   `EPICS_PVA_ADDR_LIST=127.0.0.1` + `-Djca.use_env=true`: neither Phoebus nor
   the IOC can reach host CA (5064/5065), host PVA, or the production
   131.243.x broadcast lists. With `LIVE_READONLY=1` (§2c) the client search
   path additionally sees the read-only gateway — reads only; the gateway
   denies writes at the CA access-security layer.
4. Private `phoebus.user`/`user.home` under a mktemp dir — **no writes**
   outside it (nothing under `/home/als/alsbase/Phoebus`).
5. Dedicated `-server 4079`: `-resource` forwarding cannot target another
   Phoebus (prod uses 4001/4004/4005; default `-server` is disabled).
6. Teardown kills only the launcher's own children; §6 verifies.
