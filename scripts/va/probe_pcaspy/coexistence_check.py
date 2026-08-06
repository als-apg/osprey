"""Does a pcaspy server poison a pyepics Channel Access client in the same process?

pythonSoftIOC does: importing its server into a process breaks Channel Access
client operation in that same process, which is why the softioc test suites had
to be split across subprocesses. pcaspy is a different server implementation, so
the answer has to be established rather than assumed -- it decides whether the
rewritten record-factory and fault tests can be plain in-process tests or need a
subprocess split.

Run it against the probe server left up by
``PROBE_KEEP_CONTAINER=1 bash run_probe.sh``::

    python coexistence_check.py

Prints one ``PHASE <name> PASS|FAIL -- <detail>`` line per phase and a final
``COEXISTS=yes|no``. Exits 0 whenever it reached a trustworthy verdict; exits
non-zero only if the harness itself could not be trusted (the control failed, so
a negative result would say nothing about pcaspy).

Phases, each in its own clean process so a crash in one cannot be misattributed:

- ``control``           no pcaspy anywhere: caput a sentinel, read it back.
- ``control-negative``  no pcaspy, wrong name-server port: must be unreachable.
- ``server-first``      import pcaspy and build the server, *then* caget. This is
                        the direction that threatens the tests.
- ``server-first-negative``  same, wrong port: must be unreachable.
- ``client-first``      caget, then build the server, then caget again -- a suite
                        may import in either order.

The local pvdb deliberately reuses the container's PV names with a sentinel-free
default, so a caget that returned the local (unserved) copy instead of the
container's would be caught by the value comparison, and the negative phases
catch any other discovery path.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import signal
import subprocess
import sys
import time

PREFIX = os.environ.get("PROBE_PV_PREFIX", "PCASPY:")
SP = PREFIX + "SYNC:SP"
CA_PORT = os.environ.get("PROBE_CA_PORT", "5164")
DEAD_PORT = os.environ.get("PROBE_DEAD_PORT", "5165")
CONNECT_TIMEOUT_S = float(os.environ.get("PROBE_CONNECT_TIMEOUT_S", "15"))
# Per-phase wall clock. Generous enough that exceeding it means a hang, not a
# slow connect.
PHASE_TIMEOUT_S = float(os.environ.get("PROBE_PHASE_TIMEOUT_S", "90"))
# A phase dumps every thread's stack and dies slightly before the orchestrator
# would give up on it, so a hang arrives as a diagnosable traceback rather than
# as a bare timeout. 4.3 has to know *where* a hang happens, not just that one does.
PHASE_SELF_DUMP_S = PHASE_TIMEOUT_S - 20

# Port the in-process pcaspy server binds. It must not be 5064: that belongs to
# the running virtual accelerator, and it must not be the probe container's port
# either. Nothing ever connects here -- the server is constructed, never served.
LOCAL_CAS_PORT = os.environ.get("PROBE_LOCAL_CAS_PORT", "5166")

# Value the local, unserved pvdb starts at. A caget that returns this instead of
# the sentinel came from the in-process server, not from the container.
LOCAL_SENTINEL = -999.0

# pcaspy publishes no cp313 macOS arm64 wheel, so the worktree venv (3.13) cannot
# host it. Fall through to an interpreter that can, loudly.
FALLBACK_PYTHON = os.environ.get(
    "OSPREY_COEXIST_PYTHON",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".venv-coexist", "bin", "python"),
)


def ca_env(port: str) -> None:
    """Point the client at a name server and remove every other discovery path."""
    os.environ["EPICS_CA_NAME_SERVERS"] = f"localhost:{port}"
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    os.environ["EPICS_CA_ADDR_LIST"] = ""
    # Keep the in-process server off 5064 and off the probe container's port.
    os.environ["EPICS_CAS_SERVER_PORT"] = LOCAL_CAS_PORT


LOCAL_PVDB = {
    "SYNC:SP": {"type": "float", "prec": 4, "value": LOCAL_SENTINEL},
    "SYNC:RB": {"type": "float", "prec": 4, "value": LOCAL_SENTINEL},
    "ASYNC:SP": {"type": "float", "prec": 4, "asyn": True, "value": LOCAL_SENTINEL},
    "TELEM:COUNTER": {"type": "float", "prec": 3, "value": LOCAL_SENTINEL},
}


def import_pcaspy():
    """Just the import -- no server object anywhere."""
    import pcaspy  # noqa: PLC0415 - the import under test

    print(f"  imported pcaspy {pcaspy.__version__} (no server object constructed)", flush=True)
    return pcaspy


def build_bare_server():
    """Construct a SimpleServer but register no PVs."""
    pcaspy = import_pcaspy()
    server = pcaspy.SimpleServer()
    print("  constructed a bare SimpleServer (no createPV, no Driver)", flush=True)
    return server


def build_unserved_server():
    """Import pcaspy and construct a SimpleServer + pvdb without serving it."""
    pcaspy = import_pcaspy()
    server = pcaspy.SimpleServer()
    server.createPV(PREFIX, LOCAL_PVDB)
    driver = pcaspy.Driver()
    print(
        f"  built a SimpleServer with {len(LOCAL_PVDB)} PVs "
        "(never served; server.process() is not called)",
        flush=True,
    )
    # Returned so they stay alive for the rest of the phase -- a server garbage
    # collected before the caget would not be testing anything.
    return server, driver


def do_caget(expect: float | None, name: str = SP) -> tuple[bool, str]:
    """caget a PV. ``expect`` None means it must be unreachable.

    ``expect`` NaN means "must be reachable, any value" -- used for the
    free-running telemetry PV, which has no predictable value.
    """
    import epics  # noqa: PLC0415 - imported per phase, after any pcaspy import

    started = time.monotonic()
    value = epics.caget(name, timeout=CONNECT_TIMEOUT_S)
    elapsed = time.monotonic() - started
    detail = f"caget {name} -> {value!r} in {elapsed:.3f}s"
    if expect is None:
        if value is None:
            return True, detail + " (unreachable, as required)"
        if abs(value - LOCAL_SENTINEL) < 1e-9:
            return False, detail + " -- the in-process pcaspy server answered its own client"
        return False, detail + " -- reachable despite a dead name-server address"
    if value is None:
        return False, detail + " -- PV unreachable (client never connected)"
    if abs(value - LOCAL_SENTINEL) < 1e-9:
        return False, detail + " -- read the in-process pvdb, not the container"
    if expect != expect:  # NaN: reachable is all that is asked
        return True, detail + " (reachable, value not predictable)"
    if abs(value - expect) > 1e-6:
        return False, detail + f" -- expected the control's sentinel {expect!r}"
    return True, detail + f" (matches the control's sentinel {expect!r})"


def phase_control(sentinel: float) -> int:
    """No pcaspy in this process at all. Establishes that the client env works."""
    import epics  # noqa: PLC0415

    if epics.caput(SP, sentinel, wait=True, timeout=20) != 1:
        print(f"PHASE control FAIL -- caput {SP}={sentinel} did not complete", flush=True)
        return 1
    ok, detail = do_caget(sentinel)
    print(f"PHASE control {'PASS' if ok else 'FAIL'} -- {detail}", flush=True)
    return 0 if ok else 1


def phase_control_negative(_sentinel: float) -> int:
    ok, detail = do_caget(None)
    print(f"PHASE control-negative {'PASS' if ok else 'FAIL'} -- {detail}", flush=True)
    return 0 if ok else 1


def phase_import_only(sentinel: float) -> int:
    """Does merely importing pcaspy poison the client, or does it take a server?"""
    module = import_pcaspy()
    ok, detail = do_caget(sentinel)
    print(f"PHASE import-only {'PASS' if ok else 'FAIL'} -- {detail}", flush=True)
    del module
    return 0 if ok else 1


def phase_construct_only(sentinel: float) -> int:
    """A SimpleServer with no PVs registered -- narrows import vs construction."""
    server = build_bare_server()
    ok, detail = do_caget(sentinel)
    print(f"PHASE construct-only {'PASS' if ok else 'FAIL'} -- {detail}", flush=True)
    del server
    return 0 if ok else 1


def phase_server_first(sentinel: float) -> int:
    server, driver = build_unserved_server()
    ok, detail = do_caget(sentinel)
    print(f"PHASE server-first {'PASS' if ok else 'FAIL'} -- {detail}", flush=True)
    del server, driver
    return 0 if ok else 1


def phase_server_first_negative(_sentinel: float) -> int:
    server, driver = build_unserved_server()
    ok, detail = do_caget(None)
    print(f"PHASE server-first-negative {'PASS' if ok else 'FAIL'} -- {detail}", flush=True)
    del server, driver
    return 0 if ok else 1


def phase_libca_init_first(sentinel: float) -> int:
    """Initialise libca (no connection), then import pcaspy, then caget.

    If a bare ``initialize_libca()`` is enough to survive the pcaspy import, the
    mitigation is trivial -- a conftest that touches pyepics before anything
    imports the server. If it takes a real connection, the fix is far uglier.
    """
    import epics  # noqa: PLC0415

    epics.ca.initialize_libca()
    print("  called epics.ca.initialize_libca() (no PV connected yet)", flush=True)
    server, driver = build_unserved_server()
    ok, detail = do_caget(sentinel)
    print(f"PHASE libca-init-first {'PASS' if ok else 'FAIL'} -- {detail}", flush=True)
    del server, driver
    return 0 if ok else 1


def phase_client_first(sentinel: float) -> int:
    """Client first, then the server. Also checks a *fresh* connection afterwards.

    Re-reading an already-connected PV only proves the existing circuit survived.
    A test suite connects new PVs after its imports, so the second read targets a
    PV this process has never connected to before.
    """
    before_ok, before_detail = do_caget(sentinel)
    if not before_ok:
        print(f"PHASE client-first FAIL -- before the import: {before_detail}", flush=True)
        return 1
    server, driver = build_unserved_server()
    cached_ok, cached_detail = do_caget(sentinel)
    fresh_ok, fresh_detail = do_caget(float("nan"), PREFIX + "TELEM:COUNTER")
    ok = cached_ok and fresh_ok
    print(
        f"PHASE client-first {'PASS' if ok else 'FAIL'} -- "
        f"before the server: {before_detail}; "
        f"already-connected PV after the server: {cached_detail}; "
        f"NEW PV connected after the server: {fresh_detail}",
        flush=True,
    )
    del server, driver
    return 0 if ok else 1


PHASES = {
    "control": (phase_control, CA_PORT),
    "control-negative": (phase_control_negative, DEAD_PORT),
    "import-only": (phase_import_only, CA_PORT),
    "construct-only": (phase_construct_only, CA_PORT),
    "server-first": (phase_server_first, CA_PORT),
    "server-first-negative": (phase_server_first_negative, DEAD_PORT),
    "libca-init-first": (phase_libca_init_first, CA_PORT),
    "client-first": (phase_client_first, CA_PORT),
}

# Phases that must all pass for the verdict to be "coexists". client-first is
# reported but does not gate: a suite can be told which order to import in.
VERDICT_PHASES = ("control", "control-negative", "server-first", "server-first-negative")


def run_phase(name: str, sentinel: float) -> int:
    func, port = PHASES[name]
    ca_env(port)
    print(
        f"--- phase {name}: EPICS_CA_NAME_SERVERS={os.environ['EPICS_CA_NAME_SERVERS']} "
        f"EPICS_CA_AUTO_ADDR_LIST=NO EPICS_CA_ADDR_LIST='' ---",
        flush=True,
    )
    return func(sentinel)


def signal_name(returncode: int) -> str:
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal {-returncode}"


def scored_line(stdout: str, name: str) -> str | None:
    """The phase's own verdict line, if it managed to print one."""
    for line in stdout.splitlines():
        if line.strip().startswith(f"PHASE {name} "):
            return line.strip()
    return None


def describe_failure(name: str, result: subprocess.CompletedProcess | None, timed_out: bool) -> str:
    """Name the failure mode precisely -- 4.3 needs to know what it defends against.

    A phase that printed its own verdict decided the question *before* the
    process wound down, so that verdict wins. Only a phase that never got to a
    verdict is described by how it died -- otherwise the teardown hang that
    pyepics exhibits after any failed connect (with or without pcaspy present)
    would masquerade as the coexistence failure.
    """
    own = scored_line(result.stdout, name) if result is not None else None
    if own is not None:
        return f"SILENT TIMEOUT -- {own.split('--', 1)[-1].strip()}"
    if timed_out:
        return f"HANG (no verdict and no exit within {PHASE_TIMEOUT_S:.0f}s)"
    assert result is not None
    if "Timeout (" in result.stderr:
        return (
            f"HANG (no verdict; no progress for {PHASE_SELF_DUMP_S:.0f}s, and the "
            "phase dumped its own thread stacks -- see stderr below)"
        )
    if result.returncode < 0:
        return f"CRASH before the phase reported ({signal_name(result.returncode)})"
    if "Traceback" in result.stderr:
        last = [line for line in result.stderr.strip().splitlines() if line.strip()]
        return f"EXCEPTION ({last[-1] if last else 'unknown'})"
    return f"SILENT (exit {result.returncode} with no verdict line)"


def orchestrate() -> int:
    sentinel = 7.25
    print(f"orchestrator interpreter: {sys.executable}", flush=True)
    print(f"probe server expected at localhost:{CA_PORT}; sentinel={sentinel}", flush=True)

    outcomes: dict[str, tuple[bool, str]] = {}
    for name in PHASES:
        env = dict(os.environ)
        env["PROBE_COEXIST_SENTINEL"] = str(sentinel)
        timed_out = False
        result = None
        try:
            result = subprocess.run(  # noqa: S603
                [sys.executable, os.path.abspath(__file__), "--phase", name],
                capture_output=True,
                text=True,
                timeout=PHASE_TIMEOUT_S,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as expired:
            timed_out = True
            # Keep whatever the phase managed to say before it wedged: how far it
            # got is the whole diagnosis.
            result = subprocess.CompletedProcess(
                args=[name],
                returncode=-1,
                stdout=(expired.stdout or b"").decode(errors="replace")
                if isinstance(expired.stdout, bytes)
                else (expired.stdout or ""),
                stderr=(expired.stderr or b"").decode(errors="replace")
                if isinstance(expired.stderr, bytes)
                else (expired.stderr or ""),
            )
        if result is not None:
            for line in result.stdout.splitlines():
                print("  " + line, flush=True)

        # The verdict comes from the phase's own printed line, not from the exit
        # status: the phases exit with unwinding on purpose so that a crash
        # during interpreter teardown stays visible, and a teardown crash is a
        # different fact from the caget failing. Scoring on the returncode alone
        # would conflate the two.
        scored_pass = result is not None and f"PHASE {name} PASS" in result.stdout
        if scored_pass:
            wedged_at_exit = result.returncode != 0 or timed_out
            detail = (
                "passed"
                if not wedged_at_exit
                else (
                    "passed, then wedged during interpreter teardown in pyepics' "
                    "finalize_libca -- the caget itself was unaffected"
                )
            )
            outcomes[name] = (True, detail)
            if wedged_at_exit:
                print(f"  NOTE {name}: {detail}", flush=True)
        else:
            mode = describe_failure(name, result, timed_out)
            outcomes[name] = (False, mode)
            print(f"  PHASE {name} FAILED -- failure mode: {mode}", flush=True)
        if result is not None and result.stderr.strip():
            for line in result.stderr.strip().splitlines()[-12:]:
                print("  stderr| " + line, flush=True)

    print("--- summary ---", flush=True)
    for name, (ok, detail) in outcomes.items():
        print(f"PHASE {name} {'PASS' if ok else 'FAIL'} -- {detail}", flush=True)

    # Without a working control and negative control the verdict is worthless,
    # so say so instead of reporting a coexistence answer.
    if not outcomes["control"][0]:
        print(
            "HARNESS UNTRUSTWORTHY -- the control (no pcaspy in the process) could not "
            f"reach {SP}; check that the probe container is up on {CA_PORT}. "
            "No coexistence verdict is possible.",
            flush=True,
        )
        return 2
    if not outcomes["control-negative"][0]:
        print(
            "HARNESS UNTRUSTWORTHY -- the client reached the PV through a wrong "
            "name-server address, so the transport under test is not the one claimed. "
            "No coexistence verdict is possible.",
            flush=True,
        )
        return 2

    coexists = all(outcomes[name][0] for name in VERDICT_PHASES)
    print(
        "import order: "
        + (
            "immaterial (client-first behaves the same)"
            if outcomes["client-first"][0] == outcomes["server-first"][0]
            else f"MATTERS -- server-first={outcomes['server-first'][0]}, "
            f"client-first={outcomes['client-first'][0]}"
        ),
        flush=True,
    )
    print(f"COEXISTS={'yes' if coexists else 'no'}", flush=True)
    return 0


def maybe_reexec() -> None:
    """Re-exec under an interpreter that can host pcaspy, if this one cannot."""
    if os.environ.get("PROBE_COEXIST_REEXECED"):
        return
    try:
        import pcaspy  # noqa: F401, PLC0415

        return
    except Exception:  # noqa: BLE001 - any import failure means "cannot host pcaspy"
        pass
    fallback = os.path.abspath(FALLBACK_PYTHON)
    if not os.path.exists(fallback) or os.path.samefile(fallback, sys.executable):
        print(
            f"FATAL: pcaspy is not importable under {sys.executable} and no usable "
            f"fallback interpreter was found at {fallback}. pcaspy publishes no "
            "cp313 macOS arm64 wheel; create one with "
            "'uv venv --python 3.12 .venv-coexist && uv pip install --python "
            ".venv-coexist/bin/python pcaspy==0.8.1 pyepics==3.5.9'.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)
    print(
        f"NOTE: pcaspy is not importable under {sys.executable} "
        "(pcaspy publishes no cp313 macOS arm64 wheel); re-executing the whole "
        f"check under {fallback}.",
        flush=True,
    )
    os.environ["PROBE_COEXIST_REEXECED"] = "1"
    os.execv(fallback, [fallback, os.path.abspath(__file__), *sys.argv[1:]])


def main() -> int:
    parser = argparse.ArgumentParser(description="pcaspy/pyepics in-process coexistence check")
    parser.add_argument("--phase", choices=sorted(PHASES), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.phase:
        faulthandler.enable()
        faulthandler.dump_traceback_later(PHASE_SELF_DUMP_S, exit=True)
        return run_phase(args.phase, float(os.environ.get("PROBE_COEXIST_SENTINEL", "7.25")))

    maybe_reexec()
    return orchestrate()


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    if "--phase" in sys.argv:
        # Phase subprocesses exit *with* unwinding, so a crash during teardown
        # (this repo has a standing foot-gun where GC-finalized pyepics PVs can
        # segfault libca) stays visible to the orchestrator. The orchestrator
        # reads the verdict from the printed PHASE line, so a teardown crash is
        # reported as its own fact instead of being mistaken for a failed caget.
        sys.exit(code)
    # The orchestrator itself holds no live PVs, but skip unwinding anyway so an
    # already-decided verdict cannot be corrupted on the way out.
    os._exit(code)
