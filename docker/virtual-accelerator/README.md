# PyAT Virtual Accelerator — full image

A single-container EPICS server for OSPREY's Control Assistant Tutorial: the
simulator view a build writes, served through one composite (PyAT physics for
the lattice-backed models, the texture for every other channel) on the
`lume-pva-apg` serving stack. Selected via
`control_system.type: virtual_accelerator` (`mock` stays the default; `epics`
remains production-pointed and untouched).

How the service is put together — layers, transports, LUME pins and the model
seam — is documented once, on the *Virtual Accelerator* architecture page
(`docs/source/architecture/virtual-accelerator.rst`). This file covers running
and building the image.

The VA service itself (`serving/`, `entrypoint.py`) lives at
`src/osprey/services/virtual_accelerator/` and ships as part of the `osprey`
package; only the `Containerfile` (this **full** image) stays here. The image's
`CMD` runs `osprey.services.virtual_accelerator.entrypoint` by name. A
separate, minimal reachability probe — used only to prove the CA
host↔container path works at all — lives under `scripts/va/probe_pcaspy/`.

## Quick start

`osprey build` writes the simulator view into `build/data/simulator/`, and
`osprey up` starts the compose block that runs this image against it. To run
the image by hand against a built project:

```bash
docker run --rm -p 5064:5064/tcp \
    -v <project>/build/data:/data:ro \
    -v <project>/var/agent_data/simulation:/state/simulation:ro \
    -v <project>/var/simulator:/var/simulator \
    -e VA_INSTANCE=virtual_accelerator \
    -e VA_STATE_DIR=/state/simulation \
    osprey-va-full:latest
```

Ctrl-C (or `docker stop`) shuts the IOC down cleanly.

## Run contract

- **Bind-mount the render's data root `build/data/`** to `/data` in the
  container. The IOC serves the simulator view at `/data/simulator/`
  (`VA_DATA_DIR` names a different root): `served_models.json`,
  `addresses.json`, `variables.json`, `seeds.json`, `scenarios.json` and the
  decks of the lattice-backed models, re-rendered from the project's
  `data/facility/` on every build. Read-only; the IOC never writes it, and a
  missing view file refuses the boot by its path.
- **Set `VA_INSTANCE`** to `virtual_accelerator` or `live_standin`.
  **Required** — a missing or unknown value refuses the boot. The value is
  written into the composite's log records and the model RPC's `status`
  reply. The compose blocks `osprey build` renders set it on each instance.
- **Bind-mount the repo's `var/agent_data/simulation/`** to `/state/simulation`
  and point `VA_STATE_DIR` at it. It holds `active_scenarios`, which
  `osprey sim apply NAME` rewrites on the host while the system runs — hence a
  mount separate from the build-owned data root. **Mount the directory,
  never the single file:** `sim apply` atomic-renames a new `active_scenarios`
  into place, and a directory mount lets that inode swap through, so the
  composite sees the change on its next pass with no restart. With
  `VA_STATE_DIR` unset the IOC serves `nominal` alone.
- **Bind-mount the repo's `var/simulator/`** read-write to `/var/simulator`
  for the virtual accelerator, or `var/simulator/standin/` for the live
  stand-in. It is the one directory the IOC writes: the composite appends each
  physics model's log records to `<model>.log` there, so a reader tells the
  two machines apart by where a record sits.
- **`VA_POLL_INTERVAL_S`** is the period of the runner's own passes, in
  seconds, greater than zero (the compose block fills it from
  `simulation.tick_s`); unset, the default tick applies. **`VA_MODEL_WRITE_TOKEN`**
  arms model RPC writes; unset refuses them.
- **Port `5064/tcp`**, Channel Access name-server mode
  (`EPICS_CA_NAME_SERVERS=<host>:5064`, `EPICS_CA_AUTO_ADDR_LIST=NO` on the
  connecting client) — the one host↔container CA configuration proven to
  work across container runtimes (see
  `scripts/va/probe_pcaspy/README.md`'s reachability
  matrix; UDP broadcast discovery is not published because it is not relied
  upon). 5064 is the port the server binds and the image publishes; it is the
  CA default, so a project needs no config changes beyond selecting
  `control_system.type: virtual_accelerator`.
- **The published port and the port the server binds must be the same
  number.** A CA search reply carries the server's own port, so a remap like
  `-p 5164:5064` hands every client an address nothing listens on, with no
  useful error. Pass `EPICS_CA_SERVER_PORT` to move both together; the image
  derives `EPICS_CAS_SERVER_PORT` from it, which is the variable the CA
  *server* library actually reads (it does not fall back to the client-side
  one). The PVAccess server's port is not published by the image itself.
- The container reports readiness by printing `virtual accelerator IOC
  serving PVs: <N> channels` to stdout — the whole line, with nothing after
  the count, where `<N>` is the number of channels `addresses.json` lists.
  The line is printed once the first publishing pass has published every
  served channel. A first pass that fails exits the container non-zero
  without it.

## What it serves

Every channel of the simulator view's `addresses.json`, and one status channel
per served physics model, on Channel Access and PVAccess; the container has no
namespace of its own. Which models are served is the view's
`served_models.json`, from the profile's `simulation.models`.

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
- **`accelerator-toolbox==0.8.0`** — matching what this repo's own `uv.lock`
  resolves, and what the pyAT engine plug-in was built and tested against.
  Installed before `osprey` so a resolver backtrack can never silently
  substitute a different PyAT than the plug-in expects.
- **`osprey` installed from the repo source**, not PyPI — the image always
  matches whatever checkout built it (this feature may not be released to
  PyPI yet). The whole dependency graph (FastAPI, Playwright, scikit-learn,
  ...) comes along regardless, per the plan's accepted scope — a materially
  heavier image than the toy probe. This is a known, accepted tradeoff for a
  tutorial container, not an oversight.

## Building manually

The build context **must** be a staging directory containing exactly
`pyproject.toml`, `README.md`, `src/`, `packages/`, and
`docker/virtual-accelerator/Containerfile` — never the repo root, which also
contains `.venv/`, `.git/`, and worktrees that would make every build re-tar
gigabytes of unrelated content for no benefit.
`scripts/va/build_and_boot_check.sh` stages this
automatically; if building by hand, reproduce the same staging step first.

That staging directory deliberately has no `.git`, and osprey's version comes
from the git tag (hatch-vcs), so the build would otherwise have no version to
report and would fail outright. The host resolves the version and passes it as
`--build-arg OSPREY_VERSION=...`; the build stamps it into
`src/osprey/_version.py`, which is what `osprey.__version__` reports inside the
container. A build that omits the arg still succeeds but honestly reports an
unknown version rather than a plausible wrong one.

The VA modules under `src/osprey/services/virtual_accelerator/` ship with the
`src/` copy and the `osprey-connectors` workspace member with the `packages/`
copy, both of which the `Containerfile` installs, so no extra copy step exists.

## Validating

```bash
scripts/va/build_and_boot_check.sh [DATA_ROOT]
```

Stages the build context, builds the image and boots a container serving a
simulator view. `DATA_ROOT` is a render's data root, `<project>/build/data`;
the script refuses one without `simulator/served_models.json`. With no
`DATA_ROOT` it renders the demo view from the packaged example facility,
`src/osprey/templates/facilities/example`, the way `osprey build` does. Either
way it serves a copy of the view with the declared motion removed from the two
monitor readings it measures, so a served reading is the model's reading and
nothing else. Before asserting anything it runs an identity handshake, so the
steps below measure its own container and not another one holding the port.
Then it asserts eight steps:

1. The ready line carries the marker and a positive channel count.
2. A Channel Access read of a quiescent BPM answers; this is the baseline.
3. Exciting a corrector moves the BPMs off zero.
4. The runner's own control PV is absent, beside a known-good PV that answers.
5. The model RPC answers from the host over the published PVAccess port:
   `status` carries its six keys, `info` lists the served addresses and the
   models' own variables, a `set` without a token is refused, a `set` with the
   run's token is accepted, and `diff` then shows the written offset.
6. A PVAccess put above the drive limit lands clamped, on both views.
7. A refused put moves nothing.
8. Both served ports are live inside the container.

Exits 0 only if all of that holds; tears the container down either way.

| Variable | Default | What it does |
| --- | --- | --- |
| `OSPREY_VA_CA_PORT` | `5164` | Channel Access port to bind and publish. |
| `OSPREY_VA_PVA_PORT` | `5175` | PVAccess port to bind and publish. |
| `OSPREY_VA_RUNTIME` | auto-detected | Container runtime, podman or docker. |
| `OSPREY_VA_BOOT_TIMEOUT_SECS` | `240` | Seconds to wait for the ready line; covers a linux/amd64 boot under emulation on an arm64 host. |

Building the image needs BuildKit (`docker buildx`): the `Containerfile`'s
`FROM --platform=linux/amd64` is honoured by BuildKit only, and a legacy
builder pulls the host's own architecture and stops at the `Containerfile`'s
architecture guard.

Worth knowing if you extend it: **a quiescent BPM read proves connectivity,
not physics.** The demo deck's closed orbit with no corrector excited is
exactly zero, so a BPM reads `0` on a fully working IOC, indistinguishable
from an unseeded PV. The corrector write is what proves the view → composite →
physics model chain: only that chain can move a reading.
