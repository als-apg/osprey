"""
Tests for automatic confirmation policy resolution in connectors.

The limits database is the single home of write policy: a connector asked to
write with ``confirm=None`` resolves the channel's own ``confirm`` entry, then
the fleet default of ``True``. An explicit ``confirm`` from the caller outranks
both, and a batch resolves per channel exactly as a sequence of single writes
would.
"""

import json

import pytest

from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.testing import config_of
from osprey_connectors_tests._config_trees import (
    config_tree,
    control_system_config,
    installed,
    serve_config,
)
from osprey_connectors_tests._write_fakes import (
    ARMED_SECTION,
    arm_switch,
    armed_context,
    serve_writes_enabled,
)
from tests.facility.served_tree import in_process_config, served_tree

#: Every address this module writes.
WRITTEN = (
    "BATCH:CH1",
    "BATCH:CH2",
    "BLIND:CHANNEL",
    "PICKY:CHANNEL",
    "PLAIN:CHANNEL",
    "QUIET:CHANNEL",
    "TEST:CHANNEL",
)


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch, tmp_path):
    """The login's switch on for the simulator target, connectors stamped with it.

    With :func:`serve_writes_enabled` serving a write tier that allows, that is
    everything the reference monitor's verdict asks before a write reaches the
    transport; a test that disarms serves a tier that does not.
    """
    return arm_switch(monkeypatch, tmp_path / "armed-data")


def _write_limits_db(tmp_path, limits_db):
    limits_file = tmp_path / "limits.json"
    limits_file.write_text(json.dumps(limits_db, indent=2))
    return limits_file


def _limits_config(limits_file, **extra):
    """A config that allows writes and checks limits against ``limits_file``.

    :data:`ARMED_SECTION` (a tier that allows writes) with a ``limits_checking``
    block; *extra* sets further dotted keys over it.
    """
    values = {
        "control_system.limits_checking.enabled": True,
        "control_system.limits_checking.database_path": str(limits_file),
        "control_system.limits_checking.mode": "exclusive",
        "control_system.limits_checking.on_violation": "skip",
        **extra,
    }
    return config_of(config_tree(values, base={"control_system": ARMED_SECTION}), environ={})


async def _connected_mock(tmp_path, limits_file=None, **extra):
    """A connected mock with writes enabled, serving a tree of :data:`WRITTEN`.

    The confirming read is noise-free, so the assertions are about the
    *outcome* rather than the mock's synthetic jitter.
    """
    if limits_file is None:
        serve_writes_enabled()
    else:
        serve_config(_limits_config(limits_file, **extra))

    connector = VAInProcessConnector()
    await connector.connect(
        armed_context(
            settings=in_process_config(
                served_tree(tmp_path / "served", WRITTEN), response_delay_ms=0
            )
        )
    )
    return connector


class TestConfirmResolution:
    """Pin the resolution order documented on ``write_channel``.

    1. the channel's own ``confirm`` entry, 2. the fleet default ``True``.
    """

    async def test_fleet_default_confirms_without_a_limits_database(self, tmp_path):
        """Layer 2: no database means no policy to read, so the write confirms."""
        connector = await _connected_mock(tmp_path)
        assert connector._limits_validator is None

        result = await connector.write_channel("TEST:CHANNEL", 100.0)

        assert result.outcome is WriteOutcome.CONFIRMED

        await connector.disconnect()

    async def test_fleet_default_confirms_when_the_database_is_silent(self, tmp_path):
        """Layer 2 again: an entry that states no ``confirm`` still confirms."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "QUIET:CHANNEL": {"writable": True, "min_value": 0.0, "max_value": 100.0},
            },
        )
        connector = await _connected_mock(tmp_path, limits_file)

        result = await connector.write_channel("QUIET:CHANNEL", 50.0)

        assert result.outcome is WriteOutcome.CONFIRMED

        await connector.disconnect()

    async def test_channel_entry_beats_the_fleet_default(self, tmp_path):
        """Layer 1: a channel whose entry opts out of confirmation writes blind."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "PLAIN:CHANNEL": {
                    "writable": True,
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "confirm": False,
                },
            },
        )
        connector = await _connected_mock(tmp_path, limits_file)

        result = await connector.write_channel("PLAIN:CHANNEL", 50.0)

        assert result.outcome is WriteOutcome.UNREQUESTED
        assert result.observed is None

        await connector.disconnect()

    async def test_each_channel_entry_decides_for_itself(self, tmp_path):
        """Layer 1: one channel's entry opts in beside one that declines."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "PICKY:CHANNEL": {
                    "writable": True,
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "confirm": True,
                },
                "PLAIN:CHANNEL": {
                    "writable": True,
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "confirm": False,
                },
            },
        )
        connector = await _connected_mock(tmp_path, limits_file)

        picky = await connector.write_channel("PICKY:CHANNEL", 50.0)
        plain = await connector.write_channel("PLAIN:CHANNEL", 50.0)

        assert picky.outcome is WriteOutcome.CONFIRMED
        assert plain.outcome is WriteOutcome.UNREQUESTED

        await connector.disconnect()

    async def test_channel_entry_can_opt_out_of_a_confirming_fleet(self, tmp_path):
        """Layer 1: one channel declines confirmation."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "BLIND:CHANNEL": {
                    "writable": True,
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "confirm": False,
                },
            },
        )
        connector = await _connected_mock(tmp_path, limits_file)

        result = await connector.write_channel("BLIND:CHANNEL", 50.0)

        assert result.outcome is WriteOutcome.UNREQUESTED

        await connector.disconnect()


class TestExplicitConfirmOverridesTheDatabase:
    """An explicit ``confirm`` is the caller's answer and outranks every layer."""

    async def test_explicit_false_overrides_a_confirming_channel(self, tmp_path):
        limits_file = _write_limits_db(
            tmp_path,
            {
                "PICKY:CHANNEL": {
                    "writable": True,
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "confirm": True,
                },
            },
        )
        connector = await _connected_mock(tmp_path, limits_file)

        assert (await connector.write_channel("PICKY:CHANNEL", 50.0)).outcome is (
            WriteOutcome.CONFIRMED
        )

        overridden = await connector.write_channel("PICKY:CHANNEL", 50.0, confirm=False)
        assert overridden.outcome is WriteOutcome.UNREQUESTED

        await connector.disconnect()

    async def test_explicit_true_overrides_a_declining_channel(self, tmp_path):
        limits_file = _write_limits_db(
            tmp_path,
            {
                "BLIND:CHANNEL": {
                    "writable": True,
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "confirm": False,
                },
            },
        )
        connector = await _connected_mock(tmp_path, limits_file)

        result = await connector.write_channel("BLIND:CHANNEL", 50.0, confirm=True)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed is not None

        await connector.disconnect()

    async def test_explicit_none_resolves_like_an_omitted_confirm(self, tmp_path):
        """``confirm=None`` is "no opinion", never ``False``.

        The batch path forwards nothing rather than ``False``, but a caller that
        does pass ``None`` explicitly must land in the same place as one that
        leaves the keyword off.
        """
        limits_file = _write_limits_db(
            tmp_path,
            {
                "PLAIN:CHANNEL": {
                    "writable": True,
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "confirm": False,
                },
            },
        )
        connector = await _connected_mock(tmp_path, limits_file)

        explicit_none = await connector.write_channel("PLAIN:CHANNEL", 50.0, confirm=None)
        omitted = await connector.write_channel("PLAIN:CHANNEL", 50.0)

        assert explicit_none.outcome is WriteOutcome.UNREQUESTED
        assert omitted.outcome is WriteOutcome.UNREQUESTED

        await connector.disconnect()


class TestBatchConfirmResolution:
    """``write_multiple_channels`` resolves per channel, like single writes."""

    async def test_batch_resolves_each_channel_independently(self, tmp_path):
        """An omitted ``confirm`` lets every channel in the batch decide its own."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "BATCH:CH1": {
                    "writable": True,
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "confirm": True,
                },
                "BATCH:CH2": {
                    "writable": True,
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "confirm": False,
                },
            },
        )
        connector = await _connected_mock(tmp_path, limits_file)

        results = await connector.write_multiple_channels(
            [("BATCH:CH1", 10.0), ("BATCH:CH2", 20.0)]
        )

        assert [r.address for r in results] == ["BATCH:CH1", "BATCH:CH2"]
        assert results[0].outcome is WriteOutcome.CONFIRMED
        assert results[1].outcome is WriteOutcome.UNREQUESTED

        await connector.disconnect()

    async def test_batch_forwards_an_explicit_confirm_to_every_channel(self, tmp_path):
        """One ``confirm`` for the batch overrides what each channel declares."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "BATCH:CH1": {
                    "writable": True,
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "confirm": True,
                },
                "BATCH:CH2": {"writable": True, "min_value": 0.0, "max_value": 100.0},
            },
        )
        connector = await _connected_mock(tmp_path, limits_file)

        declined = await connector.write_multiple_channels(
            [("BATCH:CH1", 10.0), ("BATCH:CH2", 20.0)], confirm=False
        )
        assert [r.outcome for r in declined] == [
            WriteOutcome.UNREQUESTED,
            WriteOutcome.UNREQUESTED,
        ]

        await connector.disconnect()

    async def test_batch_without_a_limits_database_confirms_every_channel(self, tmp_path):
        """No database, no policy: the fleet default applies to each channel."""
        connector = await _connected_mock(tmp_path)

        results = await connector.write_multiple_channels(
            [("BATCH:CH1", 10.0), ("BATCH:CH2", 20.0)]
        )

        assert [r.outcome for r in results] == [
            WriteOutcome.CONFIRMED,
            WriteOutcome.CONFIRMED,
        ]

        await connector.disconnect()


class TestWritesDisabledOutranksConfirmation:
    """The write verdict refuses before any policy is resolved."""

    async def test_a_disabled_write_is_refused_not_unrequested(self, tmp_path):
        view = served_tree(tmp_path, ["TEST:CHANNEL"])
        connector = VAInProcessConnector()
        with installed(control_system_config({})):
            await connector.connect(
                armed_context(settings=in_process_config(view, response_delay_ms=0))
            )

            result = await connector.write_channel("TEST:CHANNEL", 100.0, confirm=False)

            assert result.outcome is WriteOutcome.REFUSED
            assert result.refusal_reason == "WRITES_DISABLED"
            assert result.error_message is not None

            await connector.disconnect()
