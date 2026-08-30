"""Every connector confirms a write the same way.

The write contract's claim is not that each connector *has* a confirm flow —
it is that the three of them report the **same word** for the same situation.
A caller that reads ``result.outcome`` must not have to know whether the
channel behind it is EPICS, DOOCS, or the simulator.

That claim only holds if it is written down once. So the five scenarios below
each name their expected outcome exactly once, in ``_SCENARIOS``, and every
connector is driven through the same row. A connector that drifts fails its
row against the shared expectation, not against a private copy of it.

Each connector reaches the scenarios through its own seam, and the seams are
deliberately the ones the connector authors left:

- **Mock** — ``_confirming_read`` (noise-free, and *not* ``read_channel``) and
  ``_put``. Patching ``read_channel`` no longer steers a write at all.
- **EPICS** — an injected fake ``_epics.PV``. The confirming read runs for
  real down to ``pv.get()``; patching the read would hide the very thing the
  confirming read does differently.
- **DOOCS** — a fake ``doocs4py`` module whose ``set``/``get`` are the put and
  the confirming read.

Writes are enabled through each file's existing config-patch idiom, and no
limits validator is installed anywhere: limits are a different contract, and
leaving them out keeps the subject of this file on confirmation alone.
"""

import sys
from dataclasses import dataclass
from datetime import UTC
from unittest.mock import MagicMock, patch

import pytest

from osprey.connectors.control_system.base import WriteOutcome
from osprey.connectors.control_system.epics_connector import EPICSConnector
from osprey.connectors.control_system.mock_connector import MockConnector

# The one value every scenario writes, and the value a channel that did not
# keep it holds instead.
VALUE_SENT = 5.0
VALUE_HELD_INSTEAD = 4.7

READ_ERROR = "confirming read exploded"
PUT_ERROR = "control system refused the put"

_LIMITS_PATCH = "osprey.connectors.control_system.doocs_connector.LimitsValidator.from_config"
_TZ_PATCH = "osprey.connectors.control_system.doocs_connector.get_facility_timezone"


# ---------------------------------------------------------------------------
# The scenario table — one canonical expectation per row
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One situation a write can land in, and the result every connector owes.

    ``observed_value_set`` and ``error_message_set`` pin the field rule that
    travels with the outcome word: a mismatch carries what the channel holds
    and *no* message (both numbers are already on the result), while the two
    outcomes that carry nothing to compare carry the reason instead.

    ``confirming_reads`` is how many times the connector went back to the
    channel: exactly once when it confirms, and never when it has nothing to
    confirm — the put was refused, or confirmation was declined.
    """

    name: str
    outcome: WriteOutcome
    observed_value_set: bool
    error_message_set: bool
    confirming_reads: int


VALUE_EQUAL = Scenario(
    name="put_succeeds_value_equal",
    outcome=WriteOutcome.CONFIRMED,
    observed_value_set=True,
    error_message_set=False,
    confirming_reads=1,
)
VALUE_DIFFERS = Scenario(
    name="put_succeeds_value_differs",
    outcome=WriteOutcome.MISMATCH,
    observed_value_set=True,
    error_message_set=False,
    confirming_reads=1,
)
READ_RAISES = Scenario(
    name="confirming_read_raises",
    outcome=WriteOutcome.UNCONFIRMED,
    observed_value_set=False,
    error_message_set=True,
    confirming_reads=1,
)
PUT_FAILS = Scenario(
    name="put_fails",
    outcome=WriteOutcome.FAILED,
    observed_value_set=False,
    error_message_set=True,
    confirming_reads=0,
)
CONFIRM_DECLINED = Scenario(
    name="confirm_false",
    outcome=WriteOutcome.UNREQUESTED,
    observed_value_set=False,
    error_message_set=False,
    confirming_reads=0,
)

_SCENARIOS = [VALUE_EQUAL, VALUE_DIFFERS, READ_RAISES, PUT_FAILS, CONFIRM_DECLINED]


@dataclass(frozen=True)
class WriteRun:
    """What one connector did with one scenario."""

    result: object
    confirming_reads: int


def _confirm_argument(scenario: Scenario) -> bool | None:
    """``confirm=False`` is the scenario; every other row leaves it unset."""
    return False if scenario is CONFIRM_DECLINED else None


# ---------------------------------------------------------------------------
# Mock — seams: _put and _confirming_read
# ---------------------------------------------------------------------------


def _writes_enabled(key, default=None):
    """Enable writes and answer every other config lookup with its default."""
    if key == "control_system.writes_enabled":
        return True
    return default


async def _run_mock(scenario: Scenario, monkeypatch) -> WriteRun:
    """Drive MockConnector through ``scenario``.

    The store is what the mock confirms against, so "value differs" is a ``_put``
    that keeps a different number (a clamped setpoint), and "put fails" is a
    value the store cannot hold. Both mirror the mock's own unit tests.
    """
    monkeypatch.setattr("osprey.utils.config.get_config_value", _writes_enabled)
    connector = MockConnector()
    await connector.connect({"response_delay_ms": 0, "noise_level": 0.0})

    reads: list[str] = []
    real_confirming_read = connector._confirming_read

    async def counted_read(channel_address):
        reads.append(channel_address)
        if scenario is READ_RAISES:
            raise RuntimeError(READ_ERROR)
        return await real_confirming_read(channel_address)

    monkeypatch.setattr(connector, "_confirming_read", counted_read)

    if scenario is VALUE_DIFFERS:

        def clamping_put(channel_address, value):
            connector._state[channel_address] = VALUE_HELD_INSTEAD

        monkeypatch.setattr(connector, "_put", clamping_put)

    value = "not-a-number" if scenario is PUT_FAILS else VALUE_SENT
    result = await connector.write_channel(
        "TEST:CHANNEL:SP", value, confirm=_confirm_argument(scenario)
    )
    await connector.disconnect()

    return WriteRun(result=result, confirming_reads=len(reads))


# ---------------------------------------------------------------------------
# EPICS — seam: an injected fake _epics.PV
# ---------------------------------------------------------------------------


def _fake_pv(value, *, pv_type="time_double", labels=None):
    """A fake pyepics PV standing in for the channel a write confirms against."""
    pv = MagicMock()
    pv.wait_for_connection.return_value = True
    pv.connected = True
    pv.get.return_value = value
    pv.timestamp = 1_750_000_000.0
    pv.units = "mA"
    pv.precision = 3
    pv.status = 0
    pv.severity = 0
    pv.type = pv_type
    if labels is not None:
        pv.enum_strs = labels
    return pv


def _epics_connector(monkeypatch, *, pv, caput=True):
    """A connected EPICS connector with no limits database and writes allowed.

    connect() is skipped by injecting the runtime state it would have built;
    the base class's writes gate is opened at the property so ``write_channel``
    reaches the confirm flow this file is about.
    """
    monkeypatch.setattr(EPICSConnector, "_writes_enabled", property(lambda self: True))
    epics = MagicMock()
    epics.caput.return_value = caput
    epics.PV.return_value = pv
    connector = EPICSConnector()
    connector._epics = epics
    connector._limits_validator = None
    connector._timeout = 5.0
    connector._connected = True
    connector._epics_configured = True
    return connector


async def _run_epics(scenario: Scenario, monkeypatch) -> WriteRun:
    """Drive EPICSConnector through ``scenario``.

    ``caput`` returning False is the put the control system did not take; the
    channel's own reading is what separates confirmed from mismatched.
    """
    observed = VALUE_HELD_INSTEAD if scenario is VALUE_DIFFERS else VALUE_SENT
    pv = _fake_pv(observed)
    if scenario is READ_RAISES:
        pv.get.side_effect = TimeoutError(READ_ERROR)

    connector = _epics_connector(monkeypatch, pv=pv, caput=scenario is not PUT_FAILS)
    result = await connector.write_channel("SR:CH", VALUE_SENT, confirm=_confirm_argument(scenario))

    return WriteRun(result=result, confirming_reads=pv.get.call_count)


# ---------------------------------------------------------------------------
# DOOCS — seam: a fake doocs4py module
# ---------------------------------------------------------------------------


def _eq_data(value):
    """A mock EqData object as returned by ``doocs4py.get()``."""
    ts = MagicMock()
    ts.get_seconds_and_microseconds_since_epoch.return_value = (1_700_000_000, 500_000)

    eq = MagicMock()
    eq.get_data.return_value = value
    eq.macropulse = 12345
    eq.timestamp = ts
    return eq


def _fake_doocs4py(observed):
    d = MagicMock()
    d.__version__ = "2.0.0"
    d.names.return_value = [("FACILITY", "XFEL")]
    d.get.return_value = _eq_data(observed)
    d.set.return_value = None
    return d


async def _run_doocs(scenario: Scenario, monkeypatch) -> WriteRun:
    """Drive DOOCSConnector through ``scenario`` against a fake doocs4py."""
    observed = VALUE_HELD_INSTEAD if scenario is VALUE_DIFFERS else VALUE_SENT
    mock_d4py = _fake_doocs4py(observed)
    if scenario is READ_RAISES:
        mock_d4py.get.side_effect = RuntimeError(READ_ERROR)
    if scenario is PUT_FAILS:
        mock_d4py.set.side_effect = RuntimeError(PUT_ERROR)

    with (
        patch.dict(sys.modules, {"doocs4py": mock_d4py}),
        patch(_LIMITS_PATCH, return_value=None),
        patch(_TZ_PATCH, return_value=UTC),
        patch("osprey.utils.config.get_config_value", side_effect=_writes_enabled),
    ):
        from osprey.connectors.control_system.doocs_connector import DOOCSConnector

        conn = DOOCSConnector()
        await conn.connect({})
        result = await conn.write_channel(
            "FAC/DEV/LOC/PROP", VALUE_SENT, confirm=_confirm_argument(scenario)
        )
        await conn.disconnect()

    return WriteRun(result=result, confirming_reads=mock_d4py.get.call_count)


_DRIVERS = {
    "mock": _run_mock,
    "epics": _run_epics,
    "doocs": _run_doocs,
}


# ---------------------------------------------------------------------------
# The parity matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("connector_name", list(_DRIVERS), ids=list(_DRIVERS))
@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s.name for s in _SCENARIOS])
class TestWriteOutcomeParity:
    """The same situation gets the same outcome word from all three connectors."""

    async def test_the_outcome_word_is_the_same_for_every_connector(
        self, connector_name, scenario, monkeypatch
    ):
        run = await _DRIVERS[connector_name](scenario, monkeypatch)

        assert run.result.outcome is scenario.outcome

    async def test_the_fields_that_travel_with_the_outcome_are_the_same(
        self, connector_name, scenario, monkeypatch
    ):
        """A mismatch names what the channel holds and says nothing else.

        The two outcomes that have nothing to compare carry the reason in
        ``error_message`` instead — and confirmation being declined is neither
        a failure nor a finding, so it carries nothing at all.
        """
        run = await _DRIVERS[connector_name](scenario, monkeypatch)

        assert (run.result.observed_value is not None) is scenario.observed_value_set
        assert (run.result.error_message is not None) is scenario.error_message_set
        # Whatever happened to the write, it was never a refusal.
        assert run.result.refusal_reason is None

    async def test_the_channel_is_re_read_exactly_when_there_is_something_to_confirm(
        self, connector_name, scenario, monkeypatch
    ):
        """``unrequested`` and ``failed`` return before any read, everywhere.

        Re-reading after a put the control system did not take would report a
        stale value as though it were this write's, and a declined confirmation
        must not quietly pay for one.
        """
        run = await _DRIVERS[connector_name](scenario, monkeypatch)

        assert run.confirming_reads == scenario.confirming_reads


class TestScenarioTable:
    """The table itself is the contract, so it is pinned like one."""

    def test_every_outcome_word_appears_exactly_once(self):
        """Five scenarios, five distinct outcomes — the whole closed set.

        A sixth outcome added to ``WriteOutcome`` without a row here would
        leave a word no connector is checked for, except ``refused``, which is
        the base class's guard and not part of the confirm flow.
        """
        covered = [scenario.outcome for scenario in _SCENARIOS]

        assert len(set(covered)) == len(covered)
        assert set(covered) == set(WriteOutcome) - {WriteOutcome.REFUSED}


# ---------------------------------------------------------------------------
# EPICS-only: the parts of confirmation no other connector has
# ---------------------------------------------------------------------------


class TestEpicsOnlyConfirmation:
    """EPICS confirms through Channel Access, which adds two things of its own."""

    async def test_an_enum_label_written_as_text_is_confirmed_by_its_index(self, monkeypatch):
        """An mbbo takes "ON" and reads back 1; that is the same state.

        EPICS is the only connector that reports an ``enum_label``, and without
        it the comparison would see ``"ON" != 1`` and call a write the machine
        took exactly as sent a mismatch. DOOCS, which has no labels, would
        report that same pairing as a mismatch — so this is deliberately not a
        parity row.
        """
        pv = _fake_pv(1, pv_type="time_enum", labels=("OFF", "ON"))
        connector = _epics_connector(monkeypatch, pv=pv)

        result = await connector.write_channel("SR:VALVE", "ON")

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == 1

    async def test_the_confirming_read_bypasses_the_monitor_cache(self, monkeypatch):
        """pyepics' auto-monitor cache can still hold the pre-write value.

        The put callback says the IOC processed the write, not that a cached
        subscription update has arrived — so the confirming read goes to the
        wire rather than comparing a stale reading against the setpoint.
        """
        pv = _fake_pv(VALUE_SENT)
        connector = _epics_connector(monkeypatch, pv=pv)

        await connector.write_channel("SR:CH", VALUE_SENT)

        assert pv.get.call_args.kwargs["use_monitor"] is False
