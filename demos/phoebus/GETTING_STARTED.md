# Getting Started — OSPREY ↔ Phoebus Standalone Walkthrough

This is a self-contained onboarding guide for running the **OSPREY ↔ Phoebus native
interaction demo** end-to-end on a single machine (your laptop or a lab workstation
with a desktop session). It drives a *live* Phoebus control panel from an OSPREY agent:
the agent perceives the widget tree, snapshots widgets, and drives controls — via
synthetic GUI events **or** the semantic PV path — while also reading/writing the
**same PVs** directly through its EPICS connector.

For the architecture and a per-tool tour, see [`README.md`](./README.md) in this folder.
This file is the "from nothing, get it running" recipe.

> **Status:** internal ALS work-in-progress. Both repos are private ALS GitLab forks.
> Do **not** push the Phoebus side to the upstream ControlSystemStudio/phoebus repo.

---

## What you need

A **logged-in desktop session** (Phoebus is a JavaFX GUI app — on a headless Linux box
you'll need `Xvfb`/VNC; on macOS or a normal Linux desktop it just works), plus:

| Tool | Notes |
|------|-------|
| **JDK 21** | macOS: `brew install openjdk@21`. The launcher defaults `JAVA_HOME=/opt/homebrew/opt/openjdk@21`. |
| **Maven** | to build the Phoebus product jar. |
| **`uv`** | https://docs.astral.sh/uv/ — runs OSPREY and the soft IOC. |
| **`git`, `curl`, `jq`** | `jq` only for the manual sanity check. |
| **An LLM API key** | e.g. `CBORG_API_KEY` (LBNL institutional) or `ANTHROPIC_API_KEY`. |

The two repos are assumed to be **siblings** in the same parent directory
(e.g. both under `~/LBL/ML/`).

---

## Step 0 — Clone both ALS GitLab repos (as siblings)

```bash
mkdir -p ~/LBL/ML && cd ~/LBL/ML

# OSPREY agent framework + this demo (IOC, panel, launcher, MCP server, preset, skill)
git clone https://git.als.lbl.gov/physics/tools/osprey.git
cd osprey && git checkout feature/phoebus-demo-profile && cd ..

# CS-Studio Phoebus fork carrying the in-JVM agent bridge
git clone https://git.als.lbl.gov/physics/tools/phoebus.git
cd phoebus && git checkout feature/phoebus-agent-bridge && cd ..
```

## Step 1 — Install the OSPREY CLI from the demo branch

The demo needs the **demo-branch** OSPREY (it carries the `phoebus` MCP server, the
`phoebus-standalone` preset, and the `phoebus-walkthrough` skill) — *not* the PyPI
release. Install it editable from your clone:

```bash
cd ~/LBL/ML/osprey
uv tool install --editable .      # puts `osprey` on your PATH
osprey --version                  # sanity check
```

## Step 2 — Build the Phoebus product jar (once)

```bash
cd ~/LBL/ML/phoebus
mvn -pl phoebus-product -am install -DskipTests
```

This produces `phoebus-product/target/product-*.jar` — the GUI app `run_demo.sh`
launches, with the agent bridge baked in. (If Maven complains about missing platform
artifacts, run the dependency bootstrap in `phoebus/dependencies/` first, then retry.)

## Step 3 — Build the OSPREY demo project from the preset

```bash
cd ~/LBL/ML/osprey
osprey build phoebus-demo --preset phoebus-standalone
```

Then add your API key to `phoebus-demo/.env`, e.g.:

```bash
echo 'CBORG_API_KEY=sk-...' >> phoebus-demo/.env   # or ANTHROPIC_API_KEY=...
```

## Step 4 — Launch the demo stack (leave it running)

```bash
cd ~/LBL/ML/osprey
bash demos/phoebus/run_demo.sh
```

Wait for **`✓ Stack is UP`**. A Phoebus window opens showing live `DEMO:*` values.
Sanity check from another shell:

```bash
curl -s "http://127.0.0.1:7979/perceive?display=active" \
  | jq '.widgets[] | select(.pv) | {name, value: .pv.value}'
```

> Headless Linux box (e.g. appsdev2)? Run the launcher under `xvfb-run -a bash
> demos/phoebus/run_demo.sh` (and use a VNC session if you want to *see* the panel).

## Step 5 — Start OSPREY and run the walkthrough

In a **separate terminal**:

```bash
cd ~/LBL/ML/osprey/phoebus-demo
osprey web            # add --port 8088 if 8087 is taken
```

Open the printed URL (default `http://127.0.0.1:8087`), pick the **Terminal** tab, and type:

```
run the phoebus walkthrough
```

The agent perceives the panel, snapshots a widget, clicks **"Set 42"** (synthetic GUI
event) and verifies `DEMO:Readback` both via the GUI *and* via OSPREY's EPICS read,
contrasts a semantic write, and shows a direct PV write surfacing on the panel's
`Status` widget.

> Prefer a terminal to the browser? Step 5 also works as
> `cd phoebus-demo && claude`, then `run the phoebus walkthrough`.

---

## If something's off

- **`phoebus_*` tools return `phoebus_unreachable`** — Phoebus/bridge isn't up. Start
  `run_demo.sh`; confirm `curl -s http://127.0.0.1:7979/displays`.
- **Panel shows disconnected (white/magenta) PVs but `caget DEMO:Setpoint` works** —
  Phoebus's pure-Java CA ignores `EPICS_CA_*` by default. `run_demo.sh` already launches
  it with `-Djca.use_env=true`; if you launch Phoebus yourself, add that flag.
- **`channel_read DEMO:*` times out** — soft IOC isn't up or CA isn't on loopback;
  check `/tmp/osprey_demo_ioc.log`.
- **Phoebus jar not found** — finish Step 2, or set `PHOEBUS_REPO`/`PHOEBUS_JAR`.

See [`README.md`](./README.md) → *Troubleshooting* for the full list, and the headless
test suite (`tests/e2e/test_phoebus_demo_e2e.py`) which exercises the OSPREY half with
no desktop required.
