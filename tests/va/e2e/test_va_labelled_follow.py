"""A Channel Access put to a command reaches its labelled status by index, on a served container.

A two-state command and its status may label their states differently
(``Clear / Set`` into ``Off / On``), and a command that holds a plain number
may select the state of a labelled status. A readback's seed ``follows`` its
setpoint, and the texture model copies the state by index. A client over
Channel Access writes an index or a number, never a label, so the whole path is
proved only from the far side of the wire:

* a put of ``1`` to the two-state command holds its second label, and the
  status follows into its second label, read both as an index and as text;
* a put past the command's labels is refused, and both channels read as before;
* a put of a whole number to the numeric command holds it, and the status
  takes the label of that index;
* a put of a fraction, or of a number past the status's labels, is refused,
  and both channels read as before.

The module boots one ``osprey-va-full`` container of its own, over a data root
rendered from the demo facility with the two pairs added; the suite's demo data
root is not touched. Every Channel Access operation runs in a subprocess that
leaves through ``os._exit``, for the reasons ``conftest.py`` gives.

The whole directory is opt-in behind ``OSPREY_VA_E2E_ENABLE=1``; the skip is
applied by ``conftest.pytest_collection_modifyitems``.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey_connectors_tests._harness._ports import reserved_port
from tests.va._collection import assert_module_collected
from tests.va.e2e import conftest as e2e_conftest

#: Container-name prefix; the run's own reserved port is appended.
CONTAINER_PREFIX = "osprey-va-e2e-labelled-follow"

#: The instance the container serves as.
INSTANCE = "virtual_accelerator"

#: A local run on Apple Silicon is emulated.
BOOT_TIMEOUT_S = 180.0

#: The prefix of the one line a Channel Access child reports on.
REPORT_MARK = "REPORT "

#: Bound on one Channel Access subprocess, and on a search in it.
CA_CHILD_TIMEOUT_S = 60.0
SEARCH_TIMEOUT_S = 10.0

#: How long a status may take to show the state its command was put to: the
#: echo is published in the put's own pass, so this only absorbs a slow host.
ECHO_DEADLINE_S = 10.0

# -- the two pairs -------------------------------------------------------------

#: A two-state command and its status, whose states are labelled differently.
LATCH_COMMAND = "E2E:LATCH:CMD"
LATCH_STATUS = "E2E:LATCH:STS"
LATCH_COMMAND_LABELS = ["Clear", "Set"]
LATCH_STATUS_LABELS = ["Off", "On"]

#: A command that holds a plain number, and the three-state status it selects.
MODE_COMMAND = "E2E:MODE:CMD"
MODE_STATUS = "E2E:MODE:STS"
MODE_STATUS_LABELS = ["Off", "Standby", "On"]

#: The channel records the two pairs add to the demo facility.
PAIR_CHANNELS: list[dict[str, Any]] = [
    {
        "id": LATCH_COMMAND,
        "role": "setpoint",
        "pair": LATCH_STATUS,
        "value_type": "bool",
        "options": LATCH_COMMAND_LABELS,
        "description": "e2e-only synthetic two-state command",
    },
    {
        "id": LATCH_STATUS,
        "value_type": "bool",
        "options": LATCH_STATUS_LABELS,
        "description": "e2e-only synthetic two-state status of other labels",
    },
    {
        "id": MODE_COMMAND,
        "role": "setpoint",
        "pair": MODE_STATUS,
        "description": "e2e-only synthetic command that holds a plain number",
    },
    {
        "id": MODE_STATUS,
        "value_type": "enum",
        "options": MODE_STATUS_LABELS,
        "description": "e2e-only synthetic three-state status",
    },
]

#: The seeds: each status follows its command by index, from the first state.
PAIR_SEEDS: dict[str, Any] = {
    LATCH_STATUS: {"follows": LATCH_COMMAND},
    MODE_COMMAND: {"nominal": 0.0},
    MODE_STATUS: {"follows": MODE_COMMAND},
}

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 6


# ---------------------------------------------------------------------------
# The data root and the container
# ---------------------------------------------------------------------------


def _stage_data_root() -> Path:
    """Render the demo facility with the two pairs added; the data root a container mounts."""
    root = Path(tempfile.mkdtemp(prefix="osprey-va-labelled-follow-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    with tempfile.TemporaryDirectory(prefix="osprey-va-labelled-facility-") as scratch:
        facility = Path(scratch) / "facility"
        shutil.copytree(e2e_conftest.PRESET_FACILITY_DIR, facility)
        with (facility / "records" / "channels.yaml").open("a", encoding="utf-8") as records:
            records.write(yaml.safe_dump(PAIR_CHANNELS, sort_keys=False))
        with (facility / "seeds.yaml").open("a", encoding="utf-8") as seeds:
            seeds.write(yaml.safe_dump(PAIR_SEEDS))
        _doc, view = e2e_conftest._render_view(facility, root)
    pairs = {
        pair["readback"]: pair
        for pair in json.loads((root / "simulator" / "follows.json").read_text())["pairs"]
    }
    for status in (LATCH_STATUS, MODE_STATUS):
        assert (pairs[status]["follow"], pairs[status].get("copy")) == ("seed", "index"), pairs[
            status
        ]
    assert view is not None
    return root


def _ca_child(port: int, body: str) -> dict[str, Any]:
    """Run *body* as a Channel Access client of the container on *port*; its JSON report.

    *body* is the body of a coroutine that binds ``out``, a JSON-serialisable
    dict, which the child writes and flushes before leaving through ``os._exit``.
    """
    indented = "".join(f"    {line}\n" for line in body.splitlines())
    code = (
        "import asyncio, json, os, sys, aioca\n"
        "out = {}\n"
        "async def main():\n"
        f"{indented}"
        "asyncio.run(main())\n"
        f"sys.stdout.write('\\n' + {REPORT_MARK!r} + json.dumps(out) + '\\n')\n"
        "sys.stdout.flush()\n"
        "os._exit(0)\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=CA_CHILD_TIMEOUT_S,
        env=e2e_conftest.ca_client_environment(port),
    )
    lines = [line for line in run.stdout.splitlines() if line.startswith(REPORT_MARK)]
    assert lines, f"the Channel Access child reported nothing:\n{run.stdout}\n{run.stderr}"
    report: dict[str, Any] = json.loads(lines[-1][len(REPORT_MARK) :])
    return report


def _served(port: int) -> bool:
    """Whether the container on *port* answers both commands, asked out of process."""
    try:
        report = _ca_child(
            port,
            "found = await aioca.connect("
            f"[{LATCH_COMMAND!r}, {MODE_COMMAND!r}], timeout=2.0, throw=False)\n"
            "out['ok'] = all(f.ok for f in found)",
        )
    except (subprocess.TimeoutExpired, AssertionError):
        return False
    return bool(report.get("ok"))


@contextlib.contextmanager
def _serving(root: Path) -> Iterator[int]:
    with reserved_port() as port:
        with e2e_conftest.serving_container(
            f"{CONTAINER_PREFIX}-{port}",
            port,
            "-e",
            f"VA_INSTANCE={INSTANCE}",
            *e2e_conftest.data_root_run_args(root),
            boot_timeout=BOOT_TIMEOUT_S,
            served=_served,
        ):
            yield port


@pytest.fixture(scope="module")
def port() -> Iterator[int]:
    """The container, up and serving the two pairs, for the life of this module."""
    e2e_conftest.require_image()
    with _serving(_stage_data_root()) as served_port:
        yield served_port


# ---------------------------------------------------------------------------
# Reads and puts
# ---------------------------------------------------------------------------


def _read_body(names: tuple[str, ...], key: str) -> str:
    """Child code that reads each of *names* as a number and as text into ``out[key]``."""
    return (
        f"out[{key!r}] = {{}}\n"
        f"for name in {list(names)!r}:\n"
        "    number = await aioca.caget(name, timeout=5.0, throw=False)\n"
        "    text = await aioca.caget(name, datatype=aioca.DBR_STRING, timeout=5.0, "
        "throw=False)\n"
        f"    out[{key!r}][name] = [float(number) if number.ok else None, "
        "str(text) if text.ok else None]\n"
    )


def _read(port: int, *names: str) -> dict[str, list[Any]]:
    """Each of *names* read over Channel Access, as ``[number, text]``."""
    return _ca_child(port, _read_body(names, "read"))["read"]


def _put(port: int, command: str, value: Any, status: str) -> dict[str, Any]:
    """One Channel Access put to *command*, with both channels read before and after.

    The report holds ``before`` and ``after`` (each channel as ``[number,
    text]``) and ``put`` (whether the server completed the put).
    """
    return _ca_child(
        port,
        _read_body((command, status), "before")
        + f"put = await aioca.caput({command!r}, {value!r}, wait=True, timeout=10.0, "
        "throw=False)\n"
        "out['put'] = put.ok\n" + _read_body((command, status), "after"),
    )


def _until(port: int, status: str, wanted: list[Any]) -> list[Any]:
    """*status* read until it is *wanted* or the echo deadline passes; the last read."""
    deadline = time.monotonic() + ECHO_DEADLINE_S
    while True:
        seen = _read(port, status)[status]
        if seen == wanted or time.monotonic() > deadline:
            return seen
        time.sleep(0.5)


def _start(port: int, command: str, value: Any, status: str, wanted: list[Any]) -> None:
    """Put *command* to *value* and wait until *status* reads *wanted*: a known start."""
    _put(port, command, value, status)
    assert _until(port, status, wanted) == wanted


# ---------------------------------------------------------------------------
# The legs
# ---------------------------------------------------------------------------


def test_both_pairs_start_in_their_first_state(port: int) -> None:
    read = _read(port, LATCH_COMMAND, LATCH_STATUS, MODE_COMMAND, MODE_STATUS)

    assert read[LATCH_COMMAND] == [0.0, "Clear"]
    assert read[LATCH_STATUS] == [0.0, "Off"]
    assert read[MODE_COMMAND][0] == 0.0
    assert read[MODE_STATUS] == [0.0, "Off"]


def test_a_put_of_one_to_a_two_state_command_reaches_its_status_by_index(port: int) -> None:
    _start(port, LATCH_COMMAND, 0, LATCH_STATUS, [0.0, "Off"])

    report = _put(port, LATCH_COMMAND, 1, LATCH_STATUS)

    assert report["put"] is True, report
    assert report["after"][LATCH_COMMAND] == [1.0, "Set"], report
    assert _until(port, LATCH_STATUS, [1.0, "On"]) == [1.0, "On"]


def test_a_put_past_the_commands_labels_is_refused_and_moves_neither(port: int) -> None:
    _start(port, LATCH_COMMAND, 1, LATCH_STATUS, [1.0, "On"])

    report = _put(port, LATCH_COMMAND, 2, LATCH_STATUS)

    assert report["after"] == report["before"], report
    assert report["after"][LATCH_COMMAND] == [1.0, "Set"], report
    assert _read(port, LATCH_STATUS)[LATCH_STATUS] == [1.0, "On"]


def test_a_whole_number_put_to_a_numeric_command_selects_the_status_of_that_index(
    port: int,
) -> None:
    _start(port, MODE_COMMAND, 0.0, MODE_STATUS, [0.0, "Off"])

    report = _put(port, MODE_COMMAND, 2.0, MODE_STATUS)

    assert report["put"] is True, report
    assert report["after"][MODE_COMMAND][0] == 2.0, report
    assert _until(port, MODE_STATUS, [2.0, "On"]) == [2.0, "On"]


@pytest.mark.parametrize("value", [1.5, 3.0], ids=["a fraction", "past the labels"])
def test_a_number_that_is_no_index_of_the_status_is_refused_and_moves_neither(
    port: int, value: float
) -> None:
    _start(port, MODE_COMMAND, 1.0, MODE_STATUS, [1.0, "Standby"])

    report = _put(port, MODE_COMMAND, value, MODE_STATUS)

    assert report["after"][MODE_COMMAND][0] == 1.0, report
    assert report["after"][MODE_STATUS] == [1.0, "Standby"], report


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    assert_module_collected(request, __file__, MIN_COLLECTED_TESTS)
