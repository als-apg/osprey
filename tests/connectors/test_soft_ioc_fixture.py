"""The soft-IOC test fixture, and the test-only ``ca_raw_search`` against it.

The fixture (``tests/connectors/_soft_ioc.py``) is what the CA bulk-path and
stale-transport tests stand on, so its promises are pinned here: probed ports
that avoid the host's occupied list, an IOC in its own session that is ready
within its startup budget even with 300 records, ``kill`` that takes the process
group down and ``restart`` on the same ports, and records that behave as their
specs say over a real wire.
"""

from __future__ import annotations

import os
import re
import signal
import struct
import time

import pytest

from tests.connectors import _ca_search, _soft_ioc
from tests.connectors._ca_search import ca_raw_search
from tests.connectors._soft_ioc import (
    OCCUPIED_PORTS,
    STARTUP_BUDGET_S,
    IOCPorts,
    RecordSpec,
    SoftIOC,
    SoftIOCError,
    ai_record,
    enum_record,
    probe_free_port,
    render_db,
)

# xdist_group("soft-ioc"): every file in this group starts real soft IOCs. Their
# ports are probed free and then bound by a child process a moment later, so two
# workers probing at once can pick the same port; the IOCs' CA/PVA client sockets
# and repeater also share the host's UDP space. One worker keeps the
# probe-then-bind window free of a second prober.
pytestmark = pytest.mark.xdist_group("soft-ioc")

ALARMED = "SOFTIOC:TEST:ALARMED"
PLAIN = "SOFTIOC:TEST:PLAIN"
MODE = "SOFTIOC:TEST:MODE"
UNSERVED = "SOFTIOC:TEST:NOBODY"


def _records() -> list[RecordSpec]:
    return [
        ai_record(ALARMED, 1.5, alarm=(-10.0, 10.0), warning=(-5.0, 5.0)),
        ai_record(PLAIN, 2.5),
        enum_record(MODE, ["Off", "On", "Fault"], value=1),
    ]


@pytest.fixture
def ioc(tmp_path):
    with SoftIOC(_records(), workdir=tmp_path) as running:
        yield running


def _group_gone(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:  # a recycled pgid owned by someone else
        return True
    return False


# ---------------------------------------------------------------------------
# Records and ports (no process)
# ---------------------------------------------------------------------------


class TestRecords:
    def test_alarmed_ai_carries_major_severities_and_plain_ai_none(self):
        alarmed = ai_record(ALARMED, alarm=(-10, 10)).fields
        plain = ai_record(PLAIN).fields
        assert (alarmed["HHSV"], alarmed["LLSV"]) == ("MAJOR", "MAJOR")
        assert (alarmed["HIHI"], alarmed["LOLO"]) == ("10.0", "-10.0")
        assert not {"HHSV", "LLSV", "HSV", "LSV", "HIHI", "LOLO"} & plain.keys()

    def test_enum_record_labels_every_state(self):
        spec = enum_record(MODE, ["Off", "On"], value=1)
        assert spec.rtype == "mbbi"
        assert (spec.fields["ZRST"], spec.fields["ONST"], spec.fields["VAL"]) == ("Off", "On", "1")

    @pytest.mark.parametrize(
        ("states", "value"), [([], 0), (["s"] * 17, 0), (["Off", "On"], 2), (["Off"], -1)]
    )
    def test_enum_record_rejects_impossible_states(self, states, value):
        with pytest.raises(ValueError):
            enum_record(MODE, states, value=value)

    def test_render_db_passes_text_through_and_renders_specs(self):
        assert render_db('record(ai, "X") {}\n') == 'record(ai, "X") {}\n'
        text = render_db([RecordSpec("A", "ai", {"VAL": "1"})])
        assert text == 'record(ai, "A") {\n    field(VAL, "1")\n}\n'

    def test_render_db_rejects_duplicate_names(self):
        with pytest.raises(ValueError, match="unique"):
            render_db([ai_record("A"), ai_record("A")])


class TestPorts:
    def test_probed_ports_are_distinct_and_never_occupied(self):
        for _ in range(20):
            ports = IOCPorts.probe()
            assert len(ports.as_set()) == 4
            assert not ports.as_set() & OCCUPIED_PORTS

    def test_probe_honours_an_exclude_set(self):
        first = probe_free_port()
        assert probe_free_port({first}) != first


# ---------------------------------------------------------------------------
# The IOC process
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_ioc_leads_its_own_session_and_serves_ca(self, ioc):
        assert ioc.running
        assert os.getsid(ioc.pid) == ioc.pid != os.getsid(0)
        ioc.wait_ca_listening(timeout=1.0)

    def test_300_records_are_ready_within_the_budget(self, tmp_path):
        names = [f"SOFTIOC:SMOKE:AI{i:03d}" for i in range(300)]
        started = time.monotonic()
        with SoftIOC([ai_record(name) for name in names], workdir=tmp_path) as big:
            assert time.monotonic() - started < STARTUP_BUDGET_S
            assert ca_raw_search(big.host, big.ports.ca_server, names, timeout=3.0) == set(names)

    def test_kill_takes_the_whole_group_down(self, ioc):
        pid = ioc.pid
        ioc.kill()
        assert not ioc.running
        assert _group_gone(pid)
        assert ca_raw_search(ioc.host, ioc.ports.ca_server, [PLAIN]) == set()

    def test_restart_serves_again_on_the_same_ports(self, ioc):
        ports, first_pid = ioc.ports, ioc.pid
        ioc.restart()
        assert ioc.ports == ports
        assert ioc.pid != first_pid
        assert _group_gone(first_pid)
        assert ca_raw_search(ioc.host, ports.ca_server, [PLAIN, MODE]) == {PLAIN, MODE}

    def test_restart_after_kill(self, ioc):
        ioc.kill(signal.SIGTERM)
        ioc.restart()
        assert ca_raw_search(ioc.host, ioc.ports.ca_server, [ALARMED]) == {ALARMED}

    def test_start_twice_is_refused(self, ioc):
        with pytest.raises(SoftIOCError, match="already running"):
            ioc.start()

    def test_a_broken_db_fails_on_its_exit_with_the_output(self, tmp_path):
        """A broken database fails on the IOC's exit, never on the readiness deadline; the reason says which, so no clock is needed."""
        with pytest.raises(SoftIOCError, match="exited before 'IOC Running'") as raised:
            SoftIOC('record(nosuchtype, "X") {}\n', workdir=tmp_path).start()
        assert "IOC Starting" in str(raised.value)

    def test_an_ioc_that_never_reports_ready_fails_at_its_deadline(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_soft_ioc, "_READY_LINE", "a line the IOC never prints")
        soft = SoftIOC([ai_record(PLAIN)], workdir=tmp_path, ready_timeout=0.5)
        started = time.monotonic()
        with pytest.raises(SoftIOCError, match=r"within 0\.5s") as raised:
            soft.start()
        elapsed = time.monotonic() - started
        assert 0.5 <= elapsed < STARTUP_BUDGET_S
        message = str(raised.value)
        assert re.search(r"after \d+\.\ds", message)
        # At 0.5 s a loaded host may not have printed "IOC Starting" yet; the
        # message carries whatever the IOC wrote, and the broken-db test pins
        # that real output reaches it.
        assert message.endswith(f"output:\n{soft.output}")
        assert not soft.running

    def test_stop_removes_a_workdir_it_created(self):
        soft = SoftIOC([ai_record(PLAIN)]).start()
        workdir = soft.workdir
        assert workdir is not None and workdir.is_dir()
        soft.stop()
        assert not soft.running
        assert not workdir.exists()

    def test_client_env_points_ca_and_pva_at_this_ioc_only(self, ioc):
        udp = ioc.client_env()
        assert udp["EPICS_CA_ADDR_LIST"] == "127.0.0.1"
        assert udp["EPICS_CA_SERVER_PORT"] == str(ioc.ports.ca_server)
        assert udp["EPICS_CA_AUTO_ADDR_LIST"] == udp["EPICS_PVA_AUTO_ADDR_LIST"] == "NO"
        assert "EPICS_CA_NAME_SERVERS" not in udp
        tcp = ioc.client_env(name_server=True)
        assert tcp["EPICS_CA_NAME_SERVERS"] == ioc.ca_address
        assert tcp["EPICS_CA_ADDR_LIST"] == ""


class TestPvaWire:
    """The records behave as specified when read over a real PVA wire."""

    @pytest.fixture
    def ctxt(self, ioc):
        from p4p.client.thread import Context

        env = ioc.client_env()
        conf = {key: value for key, value in env.items() if key.startswith("EPICS_PVA_")}
        context = Context("pva", conf=conf, useenv=False)
        yield context
        context.close()

    def test_ai_value_and_display(self, ctxt):
        value = ctxt.get(PLAIN, timeout=5.0)
        assert float(value) == 2.5
        assert value.raw.display.units == "mm"
        assert value.timestamp > 0  # PINI processed it

    def test_enum_carries_its_labels(self, ctxt):
        value = ctxt.get(MODE, timeout=5.0)
        assert value.raw.value.index == 1
        assert list(value.raw.value.choices) == ["Off", "On", "Fault"]

    def test_only_the_record_with_hhsv_alarms_above_hihi(self, ctxt):
        ctxt.put(ALARMED, 20.0, timeout=5.0)
        ctxt.put(PLAIN, 20.0, timeout=5.0)
        assert ctxt.get(ALARMED, timeout=5.0).severity == 2  # MAJOR
        assert ctxt.get(PLAIN, timeout=5.0).severity == 0


# ---------------------------------------------------------------------------
# ca_raw_search
# ---------------------------------------------------------------------------


class TestCaRawSearch:
    def test_served_names_are_answered_and_unserved_are_not(self, ioc):
        found = ca_raw_search(ioc.host, ioc.ports.ca_server, [PLAIN, UNSERVED, MODE])
        assert found == {PLAIN, MODE}

    def test_returns_as_soon_as_every_name_is_answered(self, ioc):
        started = time.monotonic()
        assert ca_raw_search(ioc.host, ioc.ports.ca_server, [PLAIN], timeout=5.0) == {PLAIN}
        assert time.monotonic() - started < 2.0

    def test_unserved_only_waits_out_the_timeout_and_answers_nothing(self, ioc):
        started = time.monotonic()
        assert ca_raw_search(ioc.host, ioc.ports.ca_server, [UNSERVED], timeout=0.3) == set()
        assert 0.25 <= time.monotonic() - started < 2.0

    def test_duplicate_names_are_searched_once(self, ioc):
        assert ca_raw_search(ioc.host, ioc.ports.ca_server, [PLAIN, PLAIN]) == {PLAIN}

    def test_a_closed_port_is_an_empty_set(self):
        assert ca_raw_search("127.0.0.1", probe_free_port(), [PLAIN]) == set()

    def test_no_names_never_connects(self, monkeypatch):
        def refuse(*_args, **_kwargs):
            raise AssertionError("connected for an empty name list")

        monkeypatch.setattr(_ca_search.socket, "create_connection", refuse)
        assert ca_raw_search("127.0.0.1", 1, []) == set()


class TestFrameParsing:
    """Reply framing, independent of a server: split reads and extended headers."""

    HEADER = struct.Struct(">HHHHII")

    def _reply(self, search_id: int, *, command: int = _ca_search.CA_PROTO_SEARCH) -> bytes:
        return self.HEADER.pack(command, 8, 5064, 0, 0xFFFFFFFF, search_id) + b"\0\x0d" + b"\0" * 6

    def test_partial_frames_wait_for_the_rest(self):
        by_id, found = {1: "A", 2: "B"}, set()
        stream = self._reply(1) + self._reply(2)
        rest = _ca_search._consume_frames(stream[:20], by_id, found)
        assert found == set() and rest == stream[:20]
        rest = _ca_search._consume_frames(rest + stream[20:], by_id, found)
        assert found == {"A", "B"} and rest == b""

    def test_other_commands_and_unknown_ids_are_skipped(self):
        found: set[str] = set()
        stream = (
            self.HEADER.pack(_ca_search.CA_PROTO_VERSION, 0, 0, 13, 0, 0)
            + self._reply(1, command=1)
            + self._reply(99)
            + self._reply(1)
        )
        assert _ca_search._consume_frames(stream, {1: "A"}, found) == b""
        assert found == {"A"}

    def test_extended_header_payload_is_skipped(self):
        payload = b"\0" * 70000
        big = self.HEADER.pack(1, 0xFFFF, 0, 0, 0, 1) + struct.pack(">II", len(payload), 1)
        found: set[str] = set()
        rest = _ca_search._consume_frames(big + payload + self._reply(1), {1: "A"}, found)
        assert rest == b"" and found == {"A"}
