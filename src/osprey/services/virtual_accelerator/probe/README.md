# Phase-1 probe: containerized PyAT + Channel Access reachability

This is the Phase-1 hard gate for the PyAT Virtual Accelerator backend: prove
that an OCI-containerized soft-IOC running a toy PyAT lattice is reachable
over real EPICS Channel Access from the macOS host, through an unmodified CA
client (the same protocol OSPREY's `EPICSConnector` uses). It is a plumbing
probe, not a physics deliverable -- the lattice is a single corrector, one
drift, and one BPM monitor with no bending magnets.

Run the gate: `scripts/va/probe_build_and_caget.sh` (builds the image, runs
the container, reads a PV from the host, tears the container down, exits 0
on success).

## What it serves

Three PVs, backed by a real (if trivial) PyAT `lattice_pass`:

- `PROBE:HCM:CURRENT:SP` -- corrector current setpoint (write)
- `PROBE:HCM:CURRENT:RB` -- corrector current readback
- `PROBE:BPM:POSITION:X` -- BPM horizontal position (mm), recomputed via
  `at.lattice_pass` on every `SP` write using a fixed toy scale factor
  (`1e-4 rad/A`) and a 2 m drift, so `caput SP 50` deterministically yields
  `BPM:POSITION:X = 10`. Verified end-to-end with `caput`/`caget`.

## Runtime and architecture used

**Docker Desktop** (`/usr/local/bin/docker`, engine `linux/arm64`), not
Podman. Podman was tried first (`podman-machine-default`, applehv, arm64) but
its storage layer was found to be broken independent of anything in this
task: **every** attempt to create a new container -- including a bare, freshly
pulled `alpine:latest` with a single layer -- failed with `creating overlay
mount ...: input/output error`, while an already-running, pre-existing
container on that same machine continued working fine. Disk space and inode
usage inside the VM were normal (25% used). This affects the whole podman
machine, not anything specific to this image; it was already broken before
this task touched it. `scripts/va/probe_build_and_caget.sh` auto-detects this
with a cheap health-check container run and falls back to docker, so it will
pick podman back up automatically once that machine is repaired.

**(Historical — Phase-1.)** Originally the probe was pinned to
**`linux/amd64`** explicitly — the `Containerfile` took a `TARGET_PLATFORM`
build arg defaulting to `linux/amd64`, rather than the host's native
`linux/arm64`. The reason at the time: **neither `accelerator-toolbox==0.6.1`
nor `softioc==4.5.0`** (nor softioc's `epicscorelibs`/`pvxslibs` dependencies)
publish `manylinux_aarch64` wheels on PyPI -- only macOS and
`manylinux2014_x86_64` wheels exist for those pinned versions. Building
`epicscorelibs`/`pvxslibs` from source on arm64 means building the full EPICS
base C library, which was judged out of scope for a toy probe, so the probe
ran under `linux/amd64` emulation (via Docker Desktop's Rosetta/QEMU backend
on Apple Silicon) to install both pinned packages as prebuilt wheels with no
compiler in the image. This cost real emulation overhead but the probe still
booted to serving PVs in about 2 seconds. `softioc==4.5.0` also caps at
Python 3.10 for its manylinux wheel (no cp311+ Linux wheel), so the probe's
base image is `python:3.10-slim`, not a newer Python.

That amd64 pin is no longer a constraint anywhere: the probe `Containerfile`
now builds native-arch too (no `TARGET_PLATFORM` arg), compiling these deps
from source on arm64 exactly as the full VA image does (see
`docker/virtual-accelerator/README.md`). Only the probe's *version* pins
(`accelerator-toolbox==0.6.1`, `softioc==4.5.0`, and the `python:3.10-slim`
base they require) are retained here as the historical Phase-1 record.

## Empirical CA host<->container reachability finding

**Every host-side CA configuration tested succeeded**, which is a more
permissive result than the working assumption going in (that only CA
name-server/TCP mode would work, and UDP broadcast would not cross the
macOS<->container boundary). Tested from the host (`darwin-aarch64` EPICS
base `caget`/`caput`), against the container with **both** `5064/tcp` and
`5064/udp` published to the same host port:

| Host CA config | `EPICS_CA_AUTO_ADDR_LIST` | `EPICS_CA_ADDR_LIST` / `EPICS_CA_NAME_SERVERS` | Result |
|---|---|---|---|
| Name-server (expected-working baseline) | `NO` | `EPICS_CA_NAME_SERVERS=localhost:5064` | **Works** |
| Default auto broadcast | `YES` (default) | unset | **Works** |
| Explicit non-loopback broadcast | `NO` | `EPICS_CA_ADDR_LIST=255.255.255.255:5064` | **Works** |
| Explicit loopback address | `NO` | `EPICS_CA_ADDR_LIST=127.0.0.1:5064` | **Works** |
| No search target at all | `NO` | both unset/empty | Fails (expected -- "Empty PV search address list", no target configured) |

Full round trip verified, not just a static read: `caput PROBE:HCM:CURRENT:SP
50` -> `PROBE:HCM:CURRENT:RB` reads back `50` and `PROBE:BPM:POSITION:X`
updates to `10` (the PyAT-computed value), all over CA through the
container boundary.

**Why broadcast also works here, most likely:** this was measured under
**Docker Desktop**, not Podman. Docker Desktop's port-publishing proxy binds
the published port directly on the macOS host (`0.0.0.0:5064`); when the
`caget` client -- itself running on the same macOS host -- sends a UDP
datagram to `255.255.255.255:5064`, the host's own network stack delivers a
copy to any local socket already bound to that port, independent of whether
the packet would actually be forwarded across a real subnet. That is a
property of broadcast delivery on the sending host itself, not evidence that
broadcast frames are being routed into the container's network namespace over
a real bridge. **This finding has not been reproduced on Podman** (its
storage fault above blocked testing there), and Podman's networking (a real
Linux VM with its own virtual NIC via gvproxy) may not exhibit the same
local-delivery shortcut -- so the original name-server-only assumption may
still hold for Podman specifically. Downstream tasks that pick a CA config
(gateway preset, VA config block) should not assume broadcast is reliable
across container runtimes; **name-server mode (`EPICS_CA_NAME_SERVERS`,
`EPICS_CA_AUTO_ADDR_LIST=NO`) is the one configuration confirmed to work in
both the expected-baseline case and every other case tested, and is the only
one that doesn't depend on runtime-specific broadcast-delivery quirks**. Treat
it as the supported configuration; treat broadcast-mode success under Docker
Desktop as an artifact of this specific runtime, not a portable guarantee.

## macOS specifics

- Host: Apple Silicon (arm64), macOS. Container runtime VM (`podman machine`)
  is applehv-backed arm64; Docker Desktop's engine is also `linux/arm64`.
- (Historical — Phase-1.) The probe's `linux/amd64` image ran under Docker
  Desktop's built-in cross-arch emulation (confirmed working here); no manual
  `qemu-user-static` binfmt setup was needed. The full VA image no longer
  pins amd64 and builds native-arch instead.
- Host CA client used: `~/EPICS/epics-base/bin/darwin-aarch64/caget` /
  `caput` (real EPICS base build, not a Python stand-in). The gate script
  falls back to a `pyepics` one-liner via the worktree `.venv` if that binary
  is absent on a given machine.

## Known non-goal

The podman-machine storage fault documented above is a pre-existing,
host-wide infrastructure issue unrelated to this feature; it was reported
separately and is not something this task's file ownership can or should fix.
