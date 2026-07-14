# PyAT Virtual Accelerator — full IOC image

A single-container EPICS soft-IOC for OSPREY's Control Assistant Tutorial:
PyAT physics for the SR lattice, `pythonSoftIOC` for Channel Access, and the
same in-repo `SimulationEngine` the `mock` connector uses for everything
outside the lattice. Selected via `control_system.type: virtual_accelerator`
(`mock` stays the default; `epics` remains production-pointed and untouched).

The VA service itself (`manifest/`, `lattice/`, `ioc/`, `entrypoint.py`, and
the Phase-1 reachability probe `probe/`) lives at
`src/osprey/services/virtual_accelerator/` and ships as part of the `osprey`
package; only the `Containerfile` (this **full** image, serving the entire
1,228-channel namespace) stays here. `probe/` is a minimal toy-ring image
used only to prove the CA host↔container path works at all.

## Quick start

```bash
scripts/va/run_va.sh
```

Builds the image on first run (cached after that; `OSPREY_VA_REBUILD=1` to
force a rebuild after editing anything under
`src/osprey/services/virtual_accelerator/` or this `Containerfile`), then
serves CA on `localhost:5064` using the packaged control_assistant
preset's own `data/simulation/` as a zero-argument default. Point it at a
real project instead:

```bash
scripts/va/run_va.sh /path/to/your/project/data/simulation
```

Ctrl-C (or `docker stop`) shuts the IOC down cleanly.

## Run contract

- **Bind-mount `data/simulation/` — the directory, never a single file** —
  to `/data/simulation` in the container (`VA_DATA_DIR` env var overrides the
  mount point). `active_scenarios` lives inside it and is bind-mounted so an
  `osprey sim apply NAME` on the host (which atomic-renames a new
  `active_scenarios` into place) is visible to the container without a
  restart: mounting the *directory* means the rename's inode swap survives
  the mount; mounting a single file would not (the old inode would stay
  bind-mounted while the host swapped to a new one).
- **Port `5064/tcp`**, Channel Access name-server mode
  (`EPICS_CA_NAME_SERVERS=<host>:5064`, `EPICS_CA_AUTO_ADDR_LIST=NO` on the
  connecting client) — the one host↔container CA configuration proven to
  work across container runtimes (see
  `src/osprey/services/virtual_accelerator/probe/README.md`'s reachability
  matrix; UDP broadcast discovery is not published because it is not relied
  upon). Port 5064 matches the shipped **"Local Simulation"** gateway preset
  (`src/osprey/templates/data/facility_gateways.py`) exactly, so a project
  using it needs no config changes beyond selecting
  `control_system.type: virtual_accelerator`.
- The container reports readiness by printing `virtual accelerator IOC
  serving PVs: <N> channels ...` to stdout; `scripts/va/build_and_boot_check.sh`
  polls container logs for this line rather than guessing a fixed sleep.

## What it serves

The full namespace-union manifest
(`src/osprey/services/virtual_accelerator/manifest/channel_manifest.json`,
~1,228 addresses) — the served set is generated from the tutorial's
channel-finder databases, never hand-listed. Three physics-fidelity
partitions:

- **pyat-coupled** (SR magnet currents + BPM positions): a real PyAT lattice
  (`osprey.services.virtual_accelerator.lattice`) recomputes the closed
  orbit synchronously in the setpoint write handler
  (`ioc/physics_bridge.py`) — readback-after-write is deterministic, never
  dependent on a polling tick.
- **sp-echo** (BR/BTS magnets, SR RF/VAC setpoints): writing the setpoint
  echoes onto its readback immediately, with no physics — wired entirely
  inside `ioc/records.py`.
- **static-noisy** (everything else — GOLDEN references, status flags,
  temperatures, pressures): driven by the in-image `SimulationEngine`
  (`ioc/engine_source.py`) from the bind-mounted `machine.json`, polling
  `active_scenarios` once a second; channels the engine doesn't define fall
  back to the same generic PV-taxonomy synthesis the `mock` connector uses
  for unknown channels, so `mock` and this IOC never present different
  values for anything neither one has real data for.

## Image contents and why they're pinned this way

- **Base:** `python:3.11-slim`, built and run **native-arch** — no
  `--platform` pin, so the image matches the host architecture. On x86_64,
  `accelerator-toolbox` and `softioc` (with softioc's
  `epicscorelibs`/`pvxslibs` dependencies) install as prebuilt
  `manylinux2014_x86_64` wheels. On arm64 (Apple Silicon), no
  `manylinux_aarch64` wheels are published for these packages, so they build
  from source at image-build time — this pulls in a C toolchain and the
  EPICS base C library, making the arm64 build slower and heavier, but it
  runs natively with no emulation overhead.
- **`accelerator-toolbox==0.7.1`, `softioc==4.7.0`** — *not* the
  `accelerator-toolbox==0.6.1` / `softioc==4.5.0` pins from the Phase-1
  probe investigation. Those were fine for the probe (a standalone script
  with no `osprey` import), but **osprey's own `pyproject.toml` requires
  Python ≥3.11, while `softioc==4.5.0` publishes wheels no newer than
  cp310** — the two constraints are mutually exclusive in one interpreter.
  `softioc` first publishes a cp311 wheel at 4.6.1; 4.7.0 is the latest
  stable and is the same version already validated by
  `tests/va/test_record_factory.py`. `accelerator-toolbox` is bumped to
  0.7.1 to match what this repo's own `uv.lock` already resolves (and what
  `lattice/response.py` / `ioc/physics_bridge.py` were actually built and
  tested against), rather than carrying a stale pre-existing pin forward.
  **The probe's `3.10` / `0.6.1` / `4.5.0` pins remain correct and
  intentional for
  `src/osprey/services/virtual_accelerator/probe/Containerfile`** — that
  image never installs `osprey`, so it never hits the `>=3.11` conflict. The two images using
  different Python/PyAT/softioc versions is deliberate, not drift; please
  don't "fix" the probe to match this one, or vice versa.
- **`osprey` installed from the repo source**, not PyPI — the image always
  matches whatever checkout built it (this feature may not be released to
  PyPI yet). Only `SimulationEngine` and `pv_taxonomy` are used from the
  full package; the rest of osprey's dependency graph (FastAPI, Playwright,
  scikit-learn, ...) comes along regardless, per the plan's accepted scope
  ("osprey installed in the image, for SimulationEngine") — a materially
  heavier image than the toy probe. This is a known, accepted tradeoff for
  a tutorial container, not an oversight.

## Building manually

The build context **must** be a staging directory containing exactly
`pyproject.toml`, `README.md`, `src/`, and
`docker/virtual-accelerator/Containerfile` — never the repo root, which also
contains `.venv/`, `.git/`, and worktrees that would make every build re-tar
gigabytes of unrelated content for no benefit. `scripts/va/run_va.sh` and
`scripts/va/build_and_boot_check.sh` both stage this automatically; if
building by hand, reproduce the same staging step first.

`manifest/paths.py` locates the channel-finder database JSON files via the
installed `osprey.templates` package location
(`Path(osprey.templates.__file__).parent`), not a fixed-depth `__file__`
climb — so the VA modules under `src/osprey/services/virtual_accelerator/`
need no special copy step; they ship automatically with the `src/` copy the
`Containerfile`'s `pip install .` already installs.

## Validating

```bash
scripts/va/build_and_boot_check.sh [DATA_DIR]
```

Stages the build context, builds the image, boots a container (bind-mounting
`DATA_DIR`, defaulting to the packaged control_assistant preset's own
`data/simulation/`), waits up to 60s for the ready log line, then reads
`SR:DIAG:BPM:01:POSITION:X` over CA from the host — this PV is seeded by the
physics bridge's initial closed-orbit solve at boot, so a successful read
with no write exercises the full
manifest → records → physics-bridge → lattice chain. Exits 0 only if all of
that succeeds; tears the container down either way.
