# OSPREY ↔ Phoebus Native Interaction Demo

This demo shows OSPREY interacting with a **live Phoebus control panel** as a
first-class capability: the agent perceives the widget tree, snapshots widgets,
and drives controls — via synthetic GUI events *or* the semantic PV path — while
also reading/writing the **same PVs** directly through its EPICS connector.

```
caproto soft IOC ──CA──► Phoebus (agent bridge :7979 + osprey_demo.bob)
   (DEMO:* PVs)               ▲
        ▲                     │ HTTP  /perceive /snapshot /drive
        │ pyepics caget/caput │
        └────────── OSPREY agent ──── phoebus MCP tools ──┘
              (epics mode + native Phoebus tools + walkthrough skill)
```

## Quickstart — run the full demo (`osprey web`)

**Prerequisites:** a logged-in macOS/Linux **desktop** session (Phoebus is a GUI app;
Linux headless needs Xvfb), **JDK 21**, **`uv`**, and the **`osprey`** CLI on PATH. The
`osprey` and `phoebus` checkouts are assumed to be siblings (e.g. both under
`~/LBL/ML/`). All commands below start from the OSPREY repo root unless noted.

1. **Build the Phoebus product once** (produces the jar `run_demo.sh` launches):
   ```bash
   cd ../phoebus        # your phoebus checkout; or set PHOEBUS_REPO later
   mvn -pl phoebus-product -am install -DskipTests
   cd ../osprey
   ```

2. **Build the OSPREY demo project** from the preset:
   ```bash
   osprey build phoebus-demo --preset phoebus-standalone
   ```
   Then put your API key in `phoebus-demo/.env` (e.g. `ANTHROPIC_API_KEY=…`, or whichever
   provider you configured).

3. **Launch the demo stack** (soft IOC + Phoebus + bridge) and leave it running:
   ```bash
   bash demos/phoebus/run_demo.sh
   ```
   Wait for `✓ Stack is UP`. A Phoebus window opens showing live `DEMO:*` values.
   Sanity check from another shell:
   ```bash
   curl -s http://127.0.0.1:7979/perceive?display=active | jq '.widgets[] | select(.pv) | {name, value: .pv.value}'
   ```

4. **Start OSPREY web** in a separate terminal:
   ```bash
   cd phoebus-demo
   osprey web                 # add --port 8088 if 8087 is in use
   ```
   Open the printed URL and select the **Terminal** tab.

5. **Run the guided walkthrough** — type into the OSPREY terminal:
   ```
   run the phoebus walkthrough
   ```
   It perceives the panel, snapshots a widget, clicks "Set 42" (synthetic) and verifies
   `DEMO:Readback` both via the GUI and via OSPREY's EPICS read, contrasts a semantic
   write, and shows a direct PV write surfacing on the panel's `Status` widget.

> Prefer the CLI to the browser? Step 4–5 also work as `cd phoebus-demo && claude`, then
> `run the phoebus walkthrough`.

## What's in here

| File | Purpose |
|------|---------|
| `demo_ioc.py` | caproto soft IOC serving `DEMO:*` Channel Access PVs (the shared control-system backend). |
| `panels/osprey_demo.bob` | Phoebus display bound to `ca://DEMO:*`, exercising every drive kind (text entry, action button, toggle/momentary bool buttons, live readouts). |
| `run_demo.sh` | Desktop launcher: starts the IOC + Phoebus (with the bridge) and waits. |

The framework pieces live in the OSPREY source tree:

| Component | Location |
|-----------|----------|
| `phoebus` MCP server (5 tools) | `src/osprey/mcp_server/phoebus/` |
| `phoebus-walkthrough` skill | `src/osprey/templates/.../skills/phoebus-walkthrough/` |
| `phoebus-standalone` preset | `src/osprey/profiles/presets/phoebus-standalone.yml` |

## The PVs (served by `demo_ioc.py`)

| PV | Access | Role |
|----|--------|------|
| `DEMO:Setpoint` | r/w | text entry (`type`) + "Set 42" action button (`click`) |
| `DEMO:Readback` | r | mirrors the committed setpoint |
| `DEMO:Current` | r | live (noisy) value for trend/meter widgets |
| `DEMO:Enable` | r/w | toggle bool button (OFF/ON) |
| `DEMO:Valve` | r/w | momentary bool button (CLOSED/OPEN) |
| `DEMO:Status` | r | derived OK/WARN/FAULT (≥80 → WARN, ≥95 → FAULT) |

Writing `DEMO:Setpoint` mirrors to `DEMO:Readback` and re-derives `DEMO:Status`.

## Automated tests (no desktop required)

Beyond the live demo above, the OSPREY half of the loop is covered headlessly —
MCP tools → HTTP → real PV writes, verified via re-perceive *and* pyepics:

```bash
uv run pytest tests/e2e/test_phoebus_demo_e2e.py -v     # 4 e2e tests (real soft IOC)
uv run pytest tests/mcp_server/test_phoebus_tools.py -v # 16 unit tests
```

The live *Phoebus GUI* mile (real synthetic JavaFX events) is what the Quickstart
exercises; its in-JVM equivalent is covered by the phoebus repo's Monocle
`EndToEndHttpIT`.

## Configuration knobs

- **Bridge location:** `PHOEBUS_BRIDGE_URL` (full URL) or `phoebus.host`/`phoebus.port`
  in the project `config.yml` (default `127.0.0.1:7979`).
- **CA on loopback:** the preset sets `EPICS_CA_ADDR_LIST=127.0.0.1` /
  `EPICS_CA_AUTO_ADDR_LIST=NO` so the demo never depends on a facility gateway.
- **EPICS mode:** the preset runs OSPREY's control system in `epics` mode against
  the soft IOC at `127.0.0.1:5064` (the bundled `simulation` gateway location).

## Troubleshooting

- *`phoebus_*` tools return `phoebus_unreachable`* — Phoebus (with the bridge) isn't
  running. Start `run_demo.sh`; confirm `curl -s http://127.0.0.1:7979/displays`.
- *`channel_read DEMO:*` times out* — the soft IOC isn't up, or CA isn't on loopback.
  Check `/tmp/osprey_demo_ioc.log` and the `EPICS_CA_ADDR_LIST` env.
- *Phoebus panel shows disconnected PVs (white/magenta), but `caget DEMO:Setpoint`
  works* — Phoebus's pure-Java CA ignores the `EPICS_CA_*` environment variables by
  default (it uses `auto_addr_list=true`, a LAN broadcast that never probes the
  loopback-bound IOC). `run_demo.sh` launches Phoebus with `-Djca.use_env=true` so it
  honors `EPICS_CA_ADDR_LIST=127.0.0.1`. If you launch Phoebus yourself, add that flag,
  or set the Phoebus preferences `org.phoebus.pv.ca/addr_list=127.0.0.1` and
  `org.phoebus.pv.ca/auto_addr_list=false` via a `-settings` file.
- *Phoebus jar not found* — build it (step 1) or set `PHOEBUS_REPO`/`PHOEBUS_JAR`.
