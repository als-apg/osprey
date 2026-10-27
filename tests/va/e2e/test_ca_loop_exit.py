"""A process holding an open Channel Access monitor exits clean.

The process's Channel Access loop is stopped and closed by the threading
shutdown hook, before aioca's own ``atexit`` teardown runs. libca calls back
into that loop from its own threads for as long as a channel is open, so the
order is only sound if no callback reaches the closed loop in between. The
proof is a real client: a child process opens a monitor on the bench IOC,
takes its first update and returns from ``main`` without stopping anything, so
both exit hooks run in the product's own order. Anything a late callback
raised would land on the child's stderr.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator

import pytest

from tests.fixtures import bench_ioc as bench
from tests.va._collection import assert_module_collected
from tests.va.e2e import conftest as e2e_conftest

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 1

#: The child: one monitor through the process's CA loop, one update, then a
#: plain return from ``main`` so the interpreter's own shutdown does the rest.
CHILD = """\
import sys
import threading

from aioca import camonitor

from osprey_connectors.control_system._ca_loop import get_ca_loop


def main(channel: str) -> None:
    first = threading.Event()

    def on_update(value) -> None:
        if not first.is_set():
            first.set()
            sys.stdout.write(f"{value.name} {float(value)}\\n")
            sys.stdout.flush()

    async def open_monitor():
        return camonitor(channel, on_update)

    get_ca_loop().run_sync(open_monitor(), timeout=10)
    if not first.wait(10):
        raise SystemExit("no update within 10 s")


main(sys.argv[1])
"""


@pytest.fixture(scope="module")
def bench_endpoint() -> Iterator[bench.BenchIOC]:
    """One bench IOC, up and answering, for the life of this module.

    Spelled out rather than imported because a test that takes an imported
    fixture by name shadows the import, which this repo's lint refuses; the
    image precondition and the container lifecycle are the fixture module's.
    """
    bench.require_bench_image()
    with bench.bench_ioc() as ioc:
        yield ioc


def test_a_process_holding_an_open_monitor_exits_clean(bench_endpoint: bench.BenchIOC) -> None:
    child = subprocess.run(
        [sys.executable, "-c", CHILD, bench.PROBE_CHANNEL],
        capture_output=True,
        text=True,
        timeout=60,
        env=e2e_conftest.ca_client_environment(bench_endpoint.port),
    )

    assert child.stderr == ""
    assert child.returncode == 0
    (update,) = child.stdout.splitlines()
    assert update == f"{bench.PROBE_CHANNEL} {float(bench.PROBE_VALUE)}"


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    assert_module_collected(request, __file__, MIN_COLLECTED_TESTS)
