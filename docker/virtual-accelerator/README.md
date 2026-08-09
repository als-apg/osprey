# PyAT Virtual Accelerator — full image

A single-container EPICS server for OSPREY's Control Assistant Tutorial: PyAT
physics for the SR lattice, the `lume-pva-apg` serving stack for the wire, and
the same in-repo `SimulationEngine` the `mock` connector uses for everything
outside the lattice. Selected via `control_system.type: virtual_accelerator`
(`mock` stays the default; `epics` remains production-pointed and untouched).

One process serves two transports. The facility's whole channel namespace is
co-hosted on **Channel Access** — the authoritative view of the machine, and
what `EPICSConnector` reads and writes. The physics model's own variables are
served natively on **PVAccess** alongside it. A write arriving on either
transport moves both views; readings reach Channel Access only. See
`serving/runner.py` for what is and is not synchronised between them.

The VA service itself (`manifest/`, `lattice/`, `ioc/`, `serving/`,
`entrypoint.py`) lives at `src/osprey/services/virtual_accelerator/` and ships
as part of the `osprey` package; only the `Containerfile` (this **full** image,
serving the entire manifest namespace) stays here. A separate, minimal
toy-ring reachability probe — used only to prove the CA host↔container path
works at all — lives under `scripts/va/probe_pcaspy/`.

## Quick start

```bash
scripts/va/run_va.sh
```

Builds the image on first run (cached after that; `OSPREY_VA_REBUILD=1` to
force a rebuild after editing anything under
`src/osprey/services/virtual_accelerator/`, this `Containerfile`, or the
`virtual-accelerator` extra in `pyproject.toml`), then serves CA on
`localhost:5064` using the packaged control_assistant preset's own
`data/simulation/` as a zero-argument default. Point it at a real project
instead:

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
  `scripts/va/probe_pcaspy/README.md`'s reachability
  matrix; UDP broadcast discovery is not published because it is not relied
  upon). Port 5064 matches the shipped **"Local Simulation"** gateway preset
  (`src/osprey/templates/data/facility_gateways.py`) exactly, so a project
  using it needs no config changes beyond selecting
  `control_system.type: virtual_accelerator`.
- **The published port and the port the server binds must be the same
  number.** A CA search reply carries the server's own port, so a remap like
  `-p 5164:5064` hands every client an address nothing listens on, with no
  useful error. Pass `EPICS_CA_SERVER_PORT` to move both together; the image
  derives `EPICS_CAS_SERVER_PORT` from it, which is the variable the CA
  *server* library actually reads (it does not fall back to the client-side
  one). The PVAccess server's port is not published — PVA is served inside the
  container only.
- The container reports readiness by printing `virtual accelerator IOC
  serving PVs: <N> channels` to stdout — the whole line, with nothing after
  the count; the `(<X> pyat-coupled, <Y> static-noisy)` breakdown is its own
  earlier line. `scripts/va/build_and_boot_check.sh` polls container logs for
  the readiness line rather than guessing a fixed sleep.

## What it serves

The full namespace-union manifest
(`src/osprey/services/virtual_accelerator/manifest/channel_manifest.json`) —
a few thousand addresses, with the authoritative count in that file's own
`_metadata.total_channels` rather than repeated here, since the served set is
generated from the tutorial's channel-finder databases and never hand-listed.
Three physics-fidelity partitions:

- **pyat-coupled** (SR magnet currents + BPM positions): a real PyAT lattice
  (`osprey.services.virtual_accelerator.lattice`) recomputes the closed
  orbit synchronously in the setpoint write handler
  (`ioc/physics_bridge.py`) — readback-after-write is deterministic, never
  dependent on a polling tick.
- **sp-echo** (BR/BTS magnets, SR RF/VAC setpoints): writing the setpoint
  echoes onto its readback immediately, with no physics — decided in
  `serving/write_path.py` against the database `serving/pvdb.py` builds.
- **static-noisy** (everything else — GOLDEN references, status flags,
  temperatures, pressures): driven by the in-image `SimulationEngine`
  (`ioc/engine_source.py`) from the bind-mounted `machine.json`, polling
  `active_scenarios` once a second; channels the engine doesn't define fall
  back to the same generic PV-taxonomy synthesis the `mock` connector uses
  for unknown channels, so `mock` and this IOC never present different
  values for anything neither one has real data for.

## Image contents and why they're pinned this way

- **Base:** `python:3.11-slim`, pinned to **`linux/amd64`** — deliberately
  single-arch. `pcaspy`, the Channel Access server underneath the serving
  stack, publishes no `linux/aarch64` wheel at any interpreter, so an arm64
  image would have to compile EPICS base and `epics-modules/pcas` from source
  before it could build `pcaspy` at all, on every cold build. amd64 is also
  what CI runs. There is no arm64 variant and no source-build path for one;
  on an Apple Silicon host this image runs emulated, which is the accepted
  cost of the pin. Everything installs from prebuilt `manylinux_x86_64`
  wheels, so the image carries no C toolchain.

  A build-time guard right after the `FROM` refuses any other architecture.
  It exists because the failure it prevents is silent: osprey's
  `virtual-accelerator` extra marks `pcaspy` with
  `sys_platform == 'linux' and platform_machine == 'x86_64'`, and an
  environment marker that does not match is not an error — pip installs
  nothing for it. Without the guard, an aarch64 build would succeed and
  produce an image with **no Channel Access server**, first visible as a
  runtime `ImportError` inside `serving/runner.py`.
- **`lume-pva-apg[ca,pva]`** — the serving stack. The `Containerfile` never
  declares it as an install target or constrains its version: it is an exact
  pin inside osprey's `virtual-accelerator` extra, so it arrives with
  `.[virtual-accelerator]` below and this image cannot drift from what
  `pyproject.toml` declares.
  `[ca]` brings `pcaspy` (Channel Access), `[pva]` brings `p4p` (PVAccess) —
  both are required, because the value layer is `p4p`-typed even on the CA
  side — and `lume-base` comes with the core, since the serving layer imports
  `lume` at module scope, so even a lattice-free boot needs it and gets
  `h5py`/`matplotlib`/`scipy` along with it. `pip` is told
  `--only-binary pcaspy` as a guard: a wheel always exists on this platform,
  so a build that reaches for the sdist should fail immediately rather than
  stall inside an EPICS compile.
- **`accelerator-toolbox==0.7.1`** — matching what this repo's own `uv.lock`
  resolves, and what `lattice/response.py` and `ioc/physics_bridge.py` were
  built and tested against. Installed before `osprey` so a resolver backtrack
  can never silently substitute a different PyAT than the lattice code
  expects.
- **`osprey` installed from the repo source**, not PyPI — the image always
  matches whatever checkout built it (this feature may not be released to
  PyPI yet). The whole dependency graph (FastAPI, Playwright, scikit-learn,
  ...) comes along regardless, per the plan's accepted scope — a materially
  heavier image than the toy probe. This is a known, accepted tradeoff for a
  tutorial container, not an oversight.

## Building manually

The build context **must** be a staging directory containing exactly
`pyproject.toml`, `README.md`, `src/`, and
`docker/virtual-accelerator/Containerfile` — never the repo root, which also
contains `.venv/`, `.git/`, and worktrees that would make every build re-tar
gigabytes of unrelated content for no benefit.
`scripts/va/run_va.sh` and `scripts/va/build_and_boot_check.sh` both stage this
automatically; if building by hand, reproduce the same staging step first.

That staging directory deliberately has no `.git`, and osprey's version comes
from the git tag (hatch-vcs), so the build would otherwise have no version to
report and would fail outright. The host resolves the version and passes it as
`--build-arg OSPREY_VERSION=...`; the build stamps it into
`src/osprey/_version.py`, which is what `osprey.__version__` reports inside the
container. A build that omits the arg still succeeds but honestly reports an
unknown version rather than a plausible wrong one.

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
`data/simulation/`), waits up to 60s for the ready log line, then reads a PV
over CA from the host. Exits 0 only if all of that succeeds; tears the
container down either way.

`OSPREY_VA_CA_PORT` overrides the port, for a host where something else
already holds 5064.

Worth knowing if you extend it: **reading a BPM position at boot proves
connectivity, not physics.** The tutorial lattice's closed orbit with no
correctors excited is exactly zero, so `SR:DIAG:BPM:01:POSITION:X` reads `0`
on a fully working IOC — indistinguishable from an unseeded PV. What
exercises the manifest → serving database → physics bridge → lattice chain is
writing a corrector and requiring the orbit to move: `SR:MAG:HCM:01:CURRENT:SP`
= 0.5 puts `SR:DIAG:BPM:01:POSITION:X` at ~4.5e-6.
