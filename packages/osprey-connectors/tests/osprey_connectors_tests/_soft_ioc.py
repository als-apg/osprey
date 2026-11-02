"""A real EPICS soft IOC for connector tests, served on loopback on probed ports.

The IOC is ``python -m epicscorelibs.ioc -d <db>`` (the same EPICS base the
connectors link against), started in its own session so :meth:`SoftIOC.kill`
can take the whole process group down with ``os.killpg`` — the way an IOC host
dies — and :meth:`SoftIOC.restart` can bring it back on the *same* ports, which
is what client reconnect and stale-transport tests need.

Records are caller-supplied: either ready db text or a list of
:class:`RecordSpec` (see :func:`ai_record`, :func:`enum_record`). Nothing here
imports pytest or a CA/PVA client, so a probe script can use it too::

    from osprey_connectors_tests._soft_ioc import SoftIOC, ai_record

    with SoftIOC([ai_record(f"SIM:AI{i}") for i in range(300)]) as ioc:
        env = ioc.client_env()      # point a CA / PVA client at this IOC
        ...

Ports are probed free (TCP and UDP) and never one of :data:`OCCUPIED_PORTS`,
the ports long-lived stacks on the development host hold.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "OCCUPIED_PORTS",
    "IOCPorts",
    "RecordSpec",
    "SoftIOC",
    "SoftIOCError",
    "ai_record",
    "enum_record",
    "kill_group",
    "probe_free_port",
    "render_db",
]

#: Ports held by long-lived stacks on the host (CA/PVA defaults, the VA, gateways).
OCCUPIED_PORTS = frozenset({5064, 5065, 5075, 5076, 10090, 25064, 25075, 59070})

#: Seconds the IOC may take to print "IOC Running" before it counts as dead;
#: start() returns as soon as it does.
READY_TIMEOUT_S = 60.0

#: Seconds a healthy IOC takes to start, even with 300 records; asserted by the
#: fixture's own speed tests, never waited on.
STARTUP_BUDGET_S = 5.0

_READY_LINE = "IOC Running"


class SoftIOCError(RuntimeError):
    """The soft IOC did not start, or was used in a state it is not in."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordSpec:
    """One db record: ``record(<rtype>, "<name>") { field(<k>, "<v>") ... }``."""

    name: str
    rtype: str = "ai"
    fields: Mapping[str, str] = field(default_factory=dict)

    def render(self) -> str:
        body = "".join(f'    field({key}, "{value}")\n' for key, value in self.fields.items())
        return f'record({self.rtype}, "{self.name}") {{\n{body}}}\n'


def ai_record(
    name: str,
    value: float = 0.0,
    *,
    egu: str = "mm",
    prec: int = 3,
    limits: tuple[float, float] = (-100.0, 100.0),
    alarm: tuple[float, float] | None = None,
    warning: tuple[float, float] | None = None,
) -> RecordSpec:
    """An ``ai`` record holding ``value``, writable with a CA/PVA put.

    ``limits`` become the display and control range (LOPR/HOPR, DRVL/DRVH are
    ``ao``-only, so an ``ai`` carries the display range for both). ``alarm`` sets
    LOLO/HIHI *with* LLSV/HHSV = MAJOR; ``warning`` sets LOW/HIGH with LSV/HSV =
    MINOR. Leaving them ``None`` gives a record without alarm severities, whose
    limits a client must report as absent.
    """
    fields: dict[str, str] = {
        "VAL": repr(float(value)),
        "EGU": egu,
        "PREC": str(prec),
        "LOPR": repr(float(limits[0])),
        "HOPR": repr(float(limits[1])),
        # A soft ai with no INP never clears UDF; without this every read would
        # carry an INVALID/UDF alarm that no test asked for.
        "UDFS": "NO_ALARM",
        "PINI": "YES",  # processed once at init, so the value carries a real timestamp
    }
    if alarm is not None:
        fields.update(
            LOLO=repr(float(alarm[0])), HIHI=repr(float(alarm[1])), LLSV="MAJOR", HHSV="MAJOR"
        )
    if warning is not None:
        fields.update(
            LOW=repr(float(warning[0])), HIGH=repr(float(warning[1])), LSV="MINOR", HSV="MINOR"
        )
    return RecordSpec(name=name, rtype="ai", fields=fields)


_MBB_STATES = ("ZR", "ON", "TW", "TH", "FR", "FV", "SX", "SV", "EI", "NI", "TE", "EL", "TV", "TT")


def enum_record(name: str, states: Sequence[str], value: int = 0) -> RecordSpec:
    """An ``mbbi`` enum record with ``states`` as its labels, holding index ``value``."""
    if not 1 <= len(states) <= len(_MBB_STATES):
        raise ValueError(f"an mbbi carries 1..{len(_MBB_STATES)} states, got {len(states)}")
    if not 0 <= value < len(states):
        raise ValueError(f"value {value} is not a state index of {list(states)}")
    fields: dict[str, str] = {"VAL": str(value), "UDFS": "NO_ALARM", "PINI": "YES"}
    for index, label in enumerate(states):
        prefix = _MBB_STATES[index]
        fields[f"{prefix}ST"] = label
        fields[f"{prefix}VL"] = str(index)
    return RecordSpec(name=name, rtype="mbbi", fields=fields)


def render_db(records: str | Sequence[RecordSpec]) -> str:
    """Db text for ``records``: db text passes through, specs are rendered in order."""
    if isinstance(records, str):
        return records
    names = [spec.name for spec in records]
    if len(set(names)) != len(names):
        raise ValueError("record names must be unique")
    return "".join(spec.render() for spec in records)


# ---------------------------------------------------------------------------
# Ports and processes
# ---------------------------------------------------------------------------


def probe_free_port(exclude: frozenset[int] | set[int] = frozenset()) -> int:
    """A port free for both TCP and UDP, outside :data:`OCCUPIED_PORTS`, held for the process.

    A CA server listens on TCP *and* UDP on one port number, the PVA broadcast
    and CA repeater ports are UDP, so the port must bind for both protocols. The
    harness's :func:`~osprey_connectors_tests._harness._ports.hold_ports` checks
    exactly that and holds the port's reservation until the process exits, so no
    other test is handed it and :meth:`SoftIOC.restart` can come back on it.
    """
    from osprey_connectors_tests._harness._ports import hold_ports

    for _ in range(64):
        (port,) = hold_ports(1)
        if port not in OCCUPIED_PORTS and port not in exclude:
            return port
    raise SoftIOCError("no free TCP+UDP port found in 64 reservations")


@dataclass(frozen=True)
class IOCPorts:
    """The four ports one soft IOC owns, all distinct."""

    ca_server: int
    pva_server: int
    pva_broadcast: int
    ca_repeater: int

    @classmethod
    def probe(cls) -> IOCPorts:
        taken: set[int] = set()
        picked = []
        for _ in range(4):
            port = probe_free_port(taken)
            taken.add(port)
            picked.append(port)
        return cls(*picked)

    def as_set(self) -> set[int]:
        return {self.ca_server, self.pva_server, self.pva_broadcast, self.ca_repeater}


def kill_group(proc: subprocess.Popen, sig: int = signal.SIGKILL, timeout: float = 5.0) -> None:
    """Signal ``proc``'s whole process group (it must lead its own session) and reap it."""
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, sig)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=timeout)


def _scrubbed_env() -> dict[str, str]:
    """The caller's environment minus every EPICS_* setting, so host config cannot leak in."""
    return {key: value for key, value in os.environ.items() if not key.startswith("EPICS_")}


# ---------------------------------------------------------------------------
# The IOC
# ---------------------------------------------------------------------------


class SoftIOC:
    """One soft IOC process serving ``records`` on loopback.

    ``start()`` returns as soon as the IOC printed "IOC Running"; it raises
    :class:`SoftIOCError`, naming the elapsed time and the IOC's output, when the
    IOC exits first or ``ready_timeout`` (a liveness deadline) passes.
    ``kill()`` signals the process group; ``restart()`` is kill + start on the
    same ports and the same db; ``stop()`` (and leaving the ``with`` block) also
    removes the working directory when the IOC created it.
    """

    def __init__(
        self,
        records: str | Sequence[RecordSpec],
        *,
        workdir: str | os.PathLike[str] | None = None,
        ports: IOCPorts | None = None,
        ready_timeout: float = READY_TIMEOUT_S,
    ) -> None:
        self.db_text = render_db(records)
        self.ports = ports or IOCPorts.probe()
        self.ready_timeout = ready_timeout
        self._own_workdir = workdir is None
        self.workdir = Path(workdir) if workdir is not None else None
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._lines: list[str] = []
        self._ready = threading.Event()

    # -- addresses ---------------------------------------------------------

    @property
    def host(self) -> str:
        return "127.0.0.1"

    @property
    def ca_address(self) -> str:
        return f"{self.host}:{self.ports.ca_server}"

    @property
    def pva_address(self) -> str:
        return f"{self.host}:{self.ports.pva_server}"

    def server_env(self) -> dict[str, str]:
        """The EPICS settings the IOC process runs with (loopback only, no beacons out)."""
        return {
            "EPICS_CAS_SERVER_PORT": str(self.ports.ca_server),
            "EPICS_CAS_INTF_ADDR_LIST": self.host,
            "EPICS_CAS_AUTO_BEACON_ADDR_LIST": "NO",
            "EPICS_CAS_BEACON_ADDR_LIST": "",
            "EPICS_PVAS_SERVER_PORT": str(self.ports.pva_server),
            "EPICS_PVAS_BROADCAST_PORT": str(self.ports.pva_broadcast),
            "EPICS_PVAS_INTF_ADDR_LIST": self.host,
            "EPICS_PVAS_AUTO_BEACON_ADDR_LIST": "NO",
            "EPICS_PVAS_BEACON_ADDR_LIST": "",
            # The IOC is also a CA and a PVA *client* (db links); without these its
            # client sockets would bind the default ports (5065, 5076).
            "EPICS_CA_REPEATER_PORT": str(self.ports.ca_repeater),
            "EPICS_CA_SERVER_PORT": str(self.ports.ca_server),
            "EPICS_CA_AUTO_ADDR_LIST": "NO",
            "EPICS_CA_ADDR_LIST": "",
            "EPICS_PVA_SERVER_PORT": str(self.ports.pva_server),
            "EPICS_PVA_BROADCAST_PORT": str(self.ports.pva_broadcast),
            "EPICS_PVA_AUTO_ADDR_LIST": "NO",
            "EPICS_PVA_ADDR_LIST": "",
        }

    def client_env(self, *, name_server: bool = False) -> dict[str, str]:
        """EPICS settings that point a CA and a PVA client at this IOC only.

        By default CA searches go by UDP to the IOC's address; ``name_server=True``
        searches over TCP instead (``EPICS_CA_NAME_SERVERS``), the gateway shape.
        """
        env = {
            "EPICS_CA_AUTO_ADDR_LIST": "NO",
            "EPICS_CA_SERVER_PORT": str(self.ports.ca_server),
            "EPICS_CA_REPEATER_PORT": str(self.ports.ca_repeater),
            "EPICS_PVA_AUTO_ADDR_LIST": "NO",
            "EPICS_PVA_ADDR_LIST": self.host,
            "EPICS_PVA_SERVER_PORT": str(self.ports.pva_server),
            "EPICS_PVA_BROADCAST_PORT": str(self.ports.pva_broadcast),
        }
        if name_server:
            env["EPICS_CA_ADDR_LIST"] = ""
            env["EPICS_CA_NAME_SERVERS"] = self.ca_address
        else:
            env["EPICS_CA_ADDR_LIST"] = self.host
        return env

    # -- lifecycle ---------------------------------------------------------

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def output(self) -> str:
        """Everything the IOC wrote to stdout/stderr since the last start."""
        return "".join(self._lines)

    def start(self) -> SoftIOC:
        if self.running:
            raise SoftIOCError("soft IOC is already running")
        if self.workdir is None:
            self.workdir = Path(tempfile.mkdtemp(prefix="soft-ioc-"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        db_path = self.workdir / "soft_ioc.db"
        db_path.write_text(self.db_text)

        self._lines = []
        self._ready = threading.Event()
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "epicscorelibs.ioc", "-d", str(db_path)],
            stdin=subprocess.PIPE,  # code.interact reads it; open stdin keeps the IOC up
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # "IOC Running" is written to stderr
            text=True,
            bufsize=1,
            env={**_scrubbed_env(), **self.server_env()},
            cwd=self.workdir,
            start_new_session=True,
        )
        self._reader = threading.Thread(
            target=self._drain, args=(self._proc,), name="soft-ioc-output", daemon=True
        )
        self._reader.start()

        started = time.monotonic()
        deadline = started + self.ready_timeout
        while not self._ready.wait(0.05):
            exited = self._proc.poll() is not None
            now = time.monotonic()
            if exited or now >= deadline:
                elapsed = now - started
                self._discard()
                reason = (
                    f"exited before {_READY_LINE!r}"
                    if exited
                    else f"printed no {_READY_LINE!r} within {self.ready_timeout}s"
                )
                raise SoftIOCError(
                    f"soft IOC {reason} after {elapsed:.1f}s; output:\n{self.output}"
                )
        return self

    def _drain(self, proc: subprocess.Popen[str]) -> None:
        # Keeps reading until EOF so the pipe never fills and blocks the IOC.
        assert proc.stdout is not None
        for line in proc.stdout:
            self._lines.append(line)
            if _READY_LINE in line:
                self._ready.set()

    def kill(self, sig: int = signal.SIGKILL) -> None:
        """Take the IOC's whole process group down with ``sig`` (default: SIGKILL)."""
        if self._proc is None:
            return
        self._discard(sig)

    def restart(self) -> SoftIOC:
        """Kill (if running) and start again on the same ports with the same records."""
        self.kill()
        return self.start()

    def stop(self) -> None:
        self.kill(signal.SIGTERM)
        if self._own_workdir and self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir = None

    def _discard(self, sig: int = signal.SIGKILL) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        kill_group(proc, sig)
        if proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.close()
        if self._reader is not None:
            self._reader.join(timeout=5.0)
            self._reader = None
        if proc.stdout is not None:
            proc.stdout.close()

    def wait_ca_listening(self, timeout: float = 5.0) -> None:
        """Block until the CA server accepts TCP connections (it does by "IOC Running")."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                with socket.create_connection((self.host, self.ports.ca_server), timeout=0.5):
                    return
            except OSError:
                if time.monotonic() >= deadline:
                    raise SoftIOCError(
                        f"CA server not listening on {self.ca_address} after {timeout}s"
                    ) from None
                time.sleep(0.05)

    def __enter__(self) -> SoftIOC:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
