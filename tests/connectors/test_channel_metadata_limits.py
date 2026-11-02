"""``ChannelMetadata`` limits and ``writable``: one normalising rule, three sources.

The rule (applied on construction, so every connector and the IPC decode share
it): a missing, non-numeric or non-finite bound is ``None``; a pair whose low is
not below its high is ``None`` on both sides; ``writable`` is a ``bool`` or
``None``. The sources are the CA one-shot read (the CTRL-form reading's limits and
the client's access right), the PVA metadata reply (``display`` / ``control`` /
``valueAlarm``) and anything a test double hands over.
"""

import dataclasses
import math
import types
from unittest.mock import MagicMock

import aioca
import numpy as np
import pytest

from osprey_connectors.control_system._base import ChannelMetadata, normalise_limit_pair
from osprey_connectors.control_system._epics_connector import (
    _PVA_METADATA_REQUEST,
    EPICSConnector,
)
from osprey_connectors.ipc._frames import _decode_value, _encode_value
from osprey_connectors_tests._epics_fakes import ECA_TIMEOUT, FakeChannel, ca_connector, ca_reading

LIMIT_FIELDS = (
    "display_low",
    "display_high",
    "control_low",
    "control_high",
    "warning_low",
    "warning_high",
    "alarm_low",
    "alarm_high",
)


# ---------------------------------------------------------------------------
# The normalising rule
# ---------------------------------------------------------------------------


class TestNormaliseLimitPair:
    def test_a_valid_pair_comes_back_as_floats(self):
        assert normalise_limit_pair(0, 10) == (0.0, 10.0)
        assert all(isinstance(b, float) for b in normalise_limit_pair(0, 10))

    @pytest.mark.parametrize(
        ("low", "high"),
        [(0, 0), (5.0, 5.0), (10.0, -10.0)],
        ids=["unset-zero-zero", "equal", "inverted"],
    )
    def test_low_not_below_high_drops_both(self, low, high):
        assert normalise_limit_pair(low, high) == (None, None)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "5", True, MagicMock(), None])
    def test_a_bad_bound_is_none_and_the_other_survives(self, bad):
        assert normalise_limit_pair(bad, 3.0) == (None, 3.0)
        assert normalise_limit_pair(-3.0, bad) == (-3.0, None)

    def test_numpy_scalars_count_as_numbers(self):
        assert normalise_limit_pair(np.float64(-1.5), np.int32(2)) == (-1.5, 2.0)

    def test_it_is_idempotent(self):
        once = normalise_limit_pair(np.float32(1.0), 4)
        assert normalise_limit_pair(*once) == once


class TestChannelMetadataFields:
    def test_the_seven_new_fields_default_to_none(self):
        meta = ChannelMetadata()
        for name in ("control_low", "control_high", "warning_low", "warning_high"):
            assert getattr(meta, name) is None
        assert meta.alarm_low is None and meta.alarm_high is None
        assert meta.writable is None

    def test_every_pair_is_normalised_on_construction(self):
        meta = ChannelMetadata(
            display_low=0,
            display_high=0,  # EPICS "unset" range
            control_low=-5,
            control_high=math.nan,
            warning_low=1,
            warning_high=9,
            alarm_low=12,
            alarm_high=2,  # inverted
        )
        assert (meta.display_low, meta.display_high) == (None, None)
        assert (meta.control_low, meta.control_high) == (-5.0, None)
        assert (meta.warning_low, meta.warning_high) == (1.0, 9.0)
        assert (meta.alarm_low, meta.alarm_high) == (None, None)

    @pytest.mark.parametrize(
        ("given", "stored"),
        [
            (True, True),
            (False, False),
            (1, True),
            (0, False),
            (None, None),
            (MagicMock(), None),
            ("yes", None),
        ],
    )
    def test_writable_is_a_bool_or_none(self, given, stored):
        assert ChannelMetadata(writable=given).writable is stored

    def test_positional_construction_is_unchanged(self):
        """The new fields come after ``raw_metadata``: old positional calls still line up."""
        meta = ChannelMetadata("mA", 3)
        assert (meta.units, meta.precision) == ("mA", 3)

    def test_the_ipc_round_trip_preserves_every_field(self):
        meta = ChannelMetadata(
            units="A",
            display_low=-1,
            display_high=1,
            control_low=-2,
            control_high=2,
            warning_low=-3,
            warning_high=3,
            alarm_low=-4,
            alarm_high=4,
            writable=True,
        )
        blobs: list[bytes] = []
        decoded = _decode_value(_encode_value(meta, blobs), blobs)
        assert decoded == meta
        assert {f.name for f in dataclasses.fields(decoded)} >= set(LIMIT_FIELDS) | {"writable"}


# ---------------------------------------------------------------------------
# CA one-shot read: limits from CTRL, writable from the access right
# ---------------------------------------------------------------------------


def _ctrl_reading(**limits):
    """A CTRL-form reading of ``SR:CH`` carrying the given ``*_limit`` attributes."""
    ctrl = ca_reading(0.0, units="mA", precision=2)
    for name, bound in limits.items():
        setattr(ctrl, name, bound)
    return ctrl


async def _ca_read_metadata(ctrl, *, write=True):
    channel = FakeChannel(ca_reading(42.0), ctrl=ctrl, write=write)
    return await ca_connector(channels={"SR:CH": channel}, timeout=1.0).read_channel(
        "SR:CH", timeout=1.0
    )


class TestCaOneShotLimits:
    async def test_ctrl_limits_and_write_access_reach_the_metadata(self):
        ctrl = _ctrl_reading(
            lower_disp_limit=0.0,
            upper_disp_limit=500.0,
            lower_ctrl_limit=10.0,
            upper_ctrl_limit=400.0,
            lower_warning_limit=50.0,
            upper_warning_limit=450.0,
            lower_alarm_limit=20.0,
            upper_alarm_limit=480.0,
        )
        meta = (await _ca_read_metadata(ctrl, write=True)).metadata

        assert (meta.display_low, meta.display_high) == (0.0, 500.0)
        assert (meta.control_low, meta.control_high) == (10.0, 400.0)
        assert (meta.warning_low, meta.warning_high) == (50.0, 450.0)
        assert (meta.alarm_low, meta.alarm_high) == (20.0, 480.0)
        assert meta.writable is True

    async def test_unset_limits_and_read_only_access(self):
        """A record with no HHSV/LLSV limits reports 0/0 — which is no range at all."""
        ctrl = _ctrl_reading(
            lower_disp_limit=0.0,
            upper_disp_limit=0.0,
            lower_ctrl_limit=0.0,
            upper_ctrl_limit=0.0,
            lower_warning_limit=math.nan,
            upper_warning_limit=math.nan,
            lower_alarm_limit=0.0,
            upper_alarm_limit=0.0,
        )
        meta = (await _ca_read_metadata(ctrl, write=False)).metadata

        assert all(getattr(meta, name) is None for name in LIMIT_FIELDS)
        assert meta.writable is False

    async def test_a_pv_without_ctrl_attributes_reads_with_no_limits(self):
        meta = (await _ca_read_metadata(_ctrl_reading(), write=None)).metadata

        assert all(getattr(meta, name) is None for name in LIMIT_FIELDS)
        assert meta.writable is None
        assert meta.units == "mA"

    async def test_a_magicmock_pv_yields_none_not_a_crash(self):
        """A MagicMock control reading answers every attribute with a MagicMock."""
        meta = (await _ca_read_metadata(MagicMock(), write=MagicMock())).metadata

        assert all(getattr(meta, name) is None for name in LIMIT_FIELDS)
        assert meta.writable is None

    async def test_a_failing_ctrl_fetch_keeps_the_value(self):
        failed = aioca.CANothing("SR:CH", ECA_TIMEOUT)
        result = await _ca_read_metadata(failed, write=True)

        assert result.value == 42.0
        assert all(getattr(result.metadata, name) is None for name in LIMIT_FIELDS)
        assert result.metadata.writable is True


# ---------------------------------------------------------------------------
# PVA metadata: display / control / valueAlarm
# ---------------------------------------------------------------------------


class _Value:
    """A p4p Value stand-in: nested dicts read through ``get``."""

    def __init__(self, fields, type_id="epics:nt/NTScalar:1.0"):
        self._fields = {k: _Value(v, "") if isinstance(v, dict) else v for k, v in fields.items()}
        self._type_id = type_id

    def getID(self):
        return self._type_id

    def get(self, name, default=None):
        return self._fields.get(name, default)


class _Context:
    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def get(self, name, request=None, timeout=None):  # noqa: ARG002 - p4p Context.get signature
        self.requests.append(request)
        return self.reply


def _pva_connector(reply) -> EPICSConnector:
    connector = EPICSConnector()
    connector._pva_channel_globs = ["SR:PVA:*"]
    connector._p4p = types.SimpleNamespace(
        client=types.SimpleNamespace(thread=types.SimpleNamespace())
    )
    connector._pva_context = _Context(reply)
    connector._timeout = 1.0
    connector._connected = True
    return connector


class TestPvaLimits:
    def test_the_metadata_request_asks_for_control_and_value_alarm(self):
        assert _PVA_METADATA_REQUEST == "field(alarm,timeStamp,display,control,valueAlarm)"

    async def test_every_limit_pair_is_mapped(self):
        reply = _Value(
            {
                "alarm": {"severity": 0, "status": 0, "message": ""},
                "display": {"units": "A", "limitLow": -10.0, "limitHigh": 10.0},
                "control": {"limitLow": -8.0, "limitHigh": 8.0},
                "valueAlarm": {
                    "lowAlarmLimit": -9.0,
                    "lowWarningLimit": -7.0,
                    "highWarningLimit": 7.0,
                    "highAlarmLimit": 9.0,
                },
            }
        )
        connector = _pva_connector(reply)
        meta = await connector.get_metadata("SR:PVA:I")

        assert connector._pva_context.requests == [_PVA_METADATA_REQUEST]
        assert (meta.display_low, meta.display_high) == (-10.0, 10.0)
        assert (meta.control_low, meta.control_high) == (-8.0, 8.0)
        assert (meta.warning_low, meta.warning_high) == (-7.0, 7.0)
        assert (meta.alarm_low, meta.alarm_high) == (-9.0, 9.0)
        assert meta.writable is None  # PVA reports no access right here

    async def test_absent_control_and_value_alarm_mean_no_limits(self):
        reply = _Value({"display": {"units": "A", "limitLow": 0.0, "limitHigh": 5.0}})
        meta = await _pva_connector(reply).get_metadata("SR:PVA:I")

        assert (meta.display_low, meta.display_high) == (0.0, 5.0)
        for name in LIMIT_FIELDS[2:]:
            assert getattr(meta, name) is None

    async def test_non_numeric_and_unset_fields_yield_none(self):
        reply = _Value(
            {
                "display": {"limitLow": "low", "limitHigh": 5.0},
                "control": {"limitLow": 0.0, "limitHigh": 0.0},
                "valueAlarm": {"lowAlarmLimit": math.nan, "highAlarmLimit": 3.0},
            }
        )
        meta = await _pva_connector(reply).get_metadata("SR:PVA:I")

        assert (meta.display_low, meta.display_high) == (None, 5.0)
        assert (meta.control_low, meta.control_high) == (None, None)
        assert (meta.alarm_low, meta.alarm_high) == (None, 3.0)

    async def test_a_full_read_maps_the_same_limits(self):
        reply = _Value(
            {
                "value": 1.5,
                "control": {"limitLow": -1.0, "limitHigh": 2.0},
                "valueAlarm": {"lowWarningLimit": -0.5, "highWarningLimit": 1.8},
            }
        )
        result = await _pva_connector(reply).read_channel("SR:PVA:I")

        assert result.value == 1.5
        assert (result.metadata.control_low, result.metadata.control_high) == (-1.0, 2.0)
        assert (result.metadata.warning_low, result.metadata.warning_high) == (-0.5, 1.8)

    async def test_magicmock_fields_yield_none(self):
        """A MagicMock reply answers every field with a MagicMock — none is a limit."""
        meta = await _pva_connector(MagicMock()).get_metadata("SR:PVA:I")

        assert all(getattr(meta, name) is None for name in LIMIT_FIELDS)
