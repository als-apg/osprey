"""A soft IOC on its own reserved port, for tests that need real Channel Access.

The IOC is ``python -m epicscorelibs.ioc``: the minimal soft IOC that ships
with ``epicscorelibs``, so a test needs no container and no EPICS base build.
It runs in its own process, so the test process that starts it loads no
Channel Access client.
"""

import contextlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from tests._ports import reserved_port

IOC_READY_TIMEOUT_S = 30.0

#: Lines the IOC logs when it could not bind its configured port and fell back.
#: The port is reserved for it, so either means the reservation was broken.
_PORT_FALLBACK_LINES = ("Configured TCP port was unavailable", "No TCP server started")


class SoftIOC:
    """One ``epicscorelibs`` soft IOC on its own port, serving the test records.

    The port is reserved for the IOC's whole life: the reservation is entered
    here and closed in ``stop()`` once the process is reaped.
    """

    def __init__(self, directory: Path, name: str, prefix: str, seed: float) -> None:
        self._reservation = contextlib.ExitStack()
        self.port = self._reservation.enter_context(reserved_port())
        database = directory / f"{name}.db"
        database.write_text(
            f'record(ao, "{prefix}:SP") {{ field(VAL, "{seed}") field(PREC, "3") }}\n'
            f'record(ai, "{prefix}:RB") {{ field(VAL, "{seed}") field(PREC, "3") }}\n'
        )
        # Output to a file, never an unread pipe: an IOC that fills a pipe
        # nobody drains blocks, and the file is what a failure message quotes.
        self.log = directory / f"{name}.log"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("EPICS_CA", "EPICS_PVA"))}
        env.update(
            {
                "EPICS_CA_SERVER_PORT": str(self.port),
                "EPICS_CAS_INTF_ADDR_LIST": "127.0.0.1",
                "EPICS_CA_AUTO_ADDR_LIST": "NO",
            }
        )
        # stdin stays open: the IOC's interactive shell exits the IOC on EOF.
        try:
            with self.log.open("wb") as output:
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "epicscorelibs.ioc", "-d", str(database)],
                    stdin=subprocess.PIPE,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
        except BaseException:
            self._reservation.close()
            raise
        self._wait_until_serving()

    def _log_text(self) -> str:
        try:
            return self.log.read_text(errors="replace")
        except OSError:
            return ""

    def _wait_until_serving(self) -> None:
        deadline = time.monotonic() + IOC_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.stop()
                raise RuntimeError(
                    f"soft IOC exited with {self.process.returncode}: {self._log_text()}"
                )
            log = self._log_text()
            if any(line in log for line in _PORT_FALLBACK_LINES):
                self.stop()
                raise RuntimeError(
                    f"soft IOC could not bind its reserved port 127.0.0.1:{self.port}: {log}"
                )
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.1)
        self.stop()
        raise RuntimeError(f"soft IOC never listened on 127.0.0.1:{self.port}: {self._log_text()}")

    def stop(self) -> None:
        try:
            self.process.kill()
            self.process.wait(timeout=10)
        finally:
            self._reservation.close()
