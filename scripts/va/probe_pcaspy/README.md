# pcaspy transport + behaviour probe

Hard gate for the LUME serving layer. The virtual accelerator serves EPICS
Channel Access to host clients over a **TCP-only name-server transport** —
clients set `EPICS_CA_NAME_SERVERS=localhost:<port>` and
`EPICS_CA_AUTO_ADDR_LIST=NO`, because UDP beacon and search traffic does not
cross the container boundary on macOS. That topology had only ever been proven
against pythonSoftIOC's `rsrv`. This probe establishes whether pcaspy's
portable CAS — a completely different server implementation — behaves the same
way, and it answers three questions the write path and telemetry depend on.

Run it:

```bash
bash scripts/va/probe_pcaspy/run_probe.sh
```

Exits 0 only if all three behaviours **and** the negative control pass, and
prints a `PASS`/`FAIL` line per behaviour so a partial result stays legible.

## What is proven

| # | Behaviour | What it gates |
|---|---|---|
| 1 | `caget`/`caput` reach the served PVs over name-server TCP | every client read and write |
| 2 | an `asyn: True` PV committed later via `callbackPV` blocks a `caput` with put-completion (`wait=True`) | a setpoint write must not complete until the physics solve has committed |
| 3 | a `camonitor` subscription receives events the server posts on its own | `:RB` echo and all telemetry delivery |

## Results

Host: Apple Silicon (arm64), macOS 24.6.0, podman. Client: `pyepics` 3.5.9 from
the worktree venv. Container publishes **TCP only** on `127.0.0.1:5164`, no UDP.
Client environment for every positive run:

```
EPICS_CA_NAME_SERVERS=localhost:5164
EPICS_CA_AUTO_ADDR_LIST=NO
EPICS_CA_ADDR_LIST=            (empty)
```

The client refuses to score itself unless that environment is genuinely
name-server-only — a probe that passes with the transport misconfigured proves
nothing.

| pcaspy | arch | transport | monitor-events | async-completion | gate |
|---|---|---|---|---|---|
| 0.8.0 | arm64 (wheel) | PASS | PASS | **FAIL** | FAILED |
| 0.8.0 | amd64 (wheel, emulated) | PASS | PASS | **FAIL** | FAILED |
| 0.8.1 | amd64 (wheel, emulated) | PASS | PASS | PASS | PASSED |
| 0.8.1 | arm64 (source build) | PASS | PASS | PASS | **PASSED** |

Negative control passed in every run: repointing the client at
`localhost:5165`, where nothing listens, made the PVs unreachable. The positive
results therefore really travelled the configured name server and did not fall
back to broadcast search.

Measured on the passing native-arm64 run: put-completion returned after
**2.068 s** against a server-side delay of 2.0 s, with the committed value
already readable the instant the client unblocked; the synchronous control
write on the same circuit returned in 0.002 s, so the blocking is attributable
to the asynchronous write and not to round-trip latency. Eight server-initiated
monitor events arrived in a 4.0 s window with no client write in flight. Gate
runtime with the image already built: **~28 s**.

**Verdict: pcaspy, at version 0.8.1 or newer.** The CA layer does not need to
switch to caproto.

## pcaspy 0.8.0 cannot serve an asynchronous write

This is the one hard finding, and it is a floor on the version, not on the
library. In 0.8.0 `SimplePV.writeNotify` calls
`self.startAsyncWrite(context)`, passing the `casClientInfo` where a `casCtx`
is required. SWIG rejects the argument, the exception escapes through the
director, and the **server process aborts**:

```
TypeError: in method 'PV_startAsyncWrite', argument 2 of type 'casCtx const &'
Internal failure - unexpected problem with client's input - forcing disconnect
terminate called after throwing an instance of 'Swig::DirectorMethodException'
```

The failure mode is worse than a rejection: the client's `caput` with
`wait=True` **returned success after 43 ms**, while the value never landed and
the server died underneath it. A write path built on 0.8.0 would report
committed setpoints that were never committed.

0.8.1 fixes it with `self.startAsyncWrite(context.ctx)`. That looks like a
one-line Python change, but `casClientInfo.ctx` is a **new binding in the
0.8.1 compiled extension** (`_cas.casClientInfo_ctx_get`) and does not exist in
0.8.0, so the 0.8.0 wheels cannot be patched from Python. 0.8.1 is a genuine
floor. Reproduce either half with `PCASPY_VERSION=0.8.0 bash run_probe.sh`.

## Packaging consequence: no aarch64 wheel for 0.8.1

pcaspy 0.8.0 publishes `manylinux2014_aarch64` wheels; **0.8.1 publishes
manylinux wheels for x86_64 only**. Since 0.8.1 is the floor, installing pcaspy
with `--only-binary pcaspy` fails outright on arm64.

pcaspy's sdist does not fall back to `epicscorelibs` — its `setup.py` requires
a real `EPICS_BASE` plus `EPICS_HOST_ARCH`, and since EPICS 7 removed PCAS from
base it also requires the separate `epics-modules/pcas` module and `swig`. The
`Containerfile` here does exactly that as a fallback when no wheel matches the
platform: EPICS base `R7.0.9`, then `pcas`, then pcaspy from sdist. It works —
that is how the passing native-arm64 row above was produced — and takes about
**nine minutes cold**, after which it is layer-cached.

Consequence for the image work: amd64 installs the wheel and skips all of it,
arm64 compiles. CI's amd64 lanes will therefore never exercise the source-build
path that local arm64 development depends on.

## Notes for the serving layer

- **Commit before signalling.** `callbackPV` ends the async write. Call
  `setParam` + `updatePVs` *first*, so a client unblocking on put-completion is
  guaranteed to read the committed value. `probe_server.py` does this in that
  order deliberately.
- **Completion can only mean success.** `callbackPV` calls
  `endAsyncWrite(S_casApp_success)` unconditionally — there is no path to
  signal failure to the client. A rejected write must be expressed by
  withholding the echo (and optionally `setParamStatus`), not by the completion
  status.
- **One async write per PV at a time.** `writeNotify` returns
  `S_casApp_postponeAsyncIO` while an async write is in flight on that PV, so
  the server library serialises concurrent writes to the same address itself.
- **Returning `False` from `Driver.write` on an asyn PV** ends the async write
  with `S_cas_success` and sets `WRITE_ALARM`/`INVALID_ALARM` — the client still
  sees a successful put.
- **Monitor posting** is `setParam` + `updatePVs`, the equivalent of softioc's
  `.set()`. `updateValue` is a no-op unless a client has subscribed, and
  `updatePV` skips any PV declared with `scan > 0`, so served PVs must not use
  pcaspy's `scan` field if the driver posts their values itself.
- **The container's CA server port must equal the published host port.** The
  search reply carries the server's own port number, so a
  `-p 5164:5064` style remap would hand clients an unreachable port. Set both
  `EPICS_CA_SERVER_PORT` and `EPICS_CAS_SERVER_PORT`.

## Files

- `probe_server.py` — the pcaspy server: one synchronous PV pair, one
  `asyn: True` PV pair with a delayed commit, one server-driven telemetry
  counter.
- `probe_client.py` — host CA client; scores the three behaviours, and with
  `--expect-unreachable` runs the negative control.
- `Containerfile` — wheel install with an EPICS-base source-build fallback.
- `run_probe.sh` — the gate.

Knobs: `PCASPY_VERSION`, `PROBE_PLATFORM` (e.g. `linux/amd64`), `PROBE_CA_PORT`
(default 5164 — **not** 5064, which belongs to the running virtual accelerator),
`PROBE_KEEP_CONTAINER=1` to leave the container up for the separate client
coexistence check, `OSPREY_VA_RUNTIME` to force docker or podman.
