"""
Tests for automatic confirmation policy resolution in connectors.

The limits database is the single home of write policy: a connector asked to
write with ``confirm=None`` resolves the channel's own ``confirm`` entry, then
the ``defaults`` block's, then the fleet default of ``True``. An explicit
``confirm`` from the caller outranks all three, and a batch resolves per
channel exactly as a sequence of single writes would.
"""

import json
from unittest.mock import patch

from osprey.connectors.control_system.base import WriteOutcome
from osprey.connectors.control_system.mock_connector import MockConnector


def _config_with_writes_enabled(key, default=None):
    """Config lookup that enables writes and leaves everything else defaulted."""
    if key == "control_system.writes_enabled":
        return True
    return default


def _write_limits_db(tmp_path, limits_db):
    limits_file = tmp_path / "limits.json"
    limits_file.write_text(json.dumps(limits_db, indent=2))
    return limits_file


def _limits_config(limits_file, **extra):
    """Config map that enables limits checking against ``limits_file``.

    The connector resolves its posture from the nested ``control_system``
    section while the database path is still read by its dotted key, so the map
    answers both spellings of one deployment. The section is derived from the
    dotted entries — including any an ``extra`` overrode — rather than written
    twice, so the two spellings cannot drift apart.
    """
    config_map = {
        "control_system.limits_checking.enabled": True,
        "control_system.limits_checking.database_path": str(limits_file),
        "control_system.limits_checking.allow_unlisted_channels": False,
        "control_system.limits_checking.on_violation": "skip",
        "control_system.writes_enabled": True,
    }
    config_map.update(extra)
    config_map["control_system"] = {
        "writes_enabled": config_map["control_system.writes_enabled"],
        "limits_checking": {
            key.split(".")[-1]: value
            for key, value in config_map.items()
            if key.startswith("control_system.limits_checking.")
        },
    }

    def get_config_value(key, default=None):
        return config_map.get(key, default)

    return get_config_value


async def _connected_mock(monkeypatch, limits_file=None, **extra):
    """A connected mock with writes enabled and noise switched off.

    Noise off is what makes the assertions about the *outcome* rather than the
    mock's synthetic jitter; the confirming read is noise-free either way.
    """
    config = (
        _config_with_writes_enabled if limits_file is None else _limits_config(limits_file, **extra)
    )
    monkeypatch.setattr("osprey.utils.config.get_config_value", config)

    connector = MockConnector()
    await connector.connect({"response_delay_ms": 0, "noise_level": 0.0})
    return connector


class TestConfirmResolution:
    """Pin the resolution order documented on ``write_channel``.

    1. the channel's own ``confirm`` entry, 2. the limits database's
    ``defaults.confirm``, 3. the fleet default ``True``.
    """

    async def test_fleet_default_confirms_without_a_limits_database(self, monkeypatch):
        """Layer 3: no database means no policy to read, so the write confirms."""
        connector = await _connected_mock(monkeypatch)
        assert connector._limits_validator is None

        result = await connector.write_channel("TEST:CHANNEL", 100.0)

        assert result.outcome is WriteOutcome.CONFIRMED

        await connector.disconnect()

    async def test_fleet_default_confirms_when_the_database_is_silent(self, tmp_path, monkeypatch):
        """Layer 3 again: a database that mentions no ``confirm`` still confirms."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "defaults": {"writable": True},
                "QUIET:CHANNEL": {"min_value": 0.0, "max_value": 100.0},
            },
        )
        connector = await _connected_mock(monkeypatch, limits_file)

        result = await connector.write_channel("QUIET:CHANNEL", 50.0)

        assert result.outcome is WriteOutcome.CONFIRMED

        await connector.disconnect()

    async def test_defaults_block_beats_the_fleet_default(self, tmp_path, monkeypatch):
        """Layer 2: a fleet that has opted out of confirmation writes blind."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "defaults": {"writable": True, "confirm": False},
                "PLAIN:CHANNEL": {"min_value": 0.0, "max_value": 100.0},
            },
        )
        connector = await _connected_mock(monkeypatch, limits_file)

        result = await connector.write_channel("PLAIN:CHANNEL", 50.0)

        assert result.outcome is WriteOutcome.UNREQUESTED
        assert result.observed_value is None

        await connector.disconnect()

    async def test_channel_entry_beats_the_defaults_block(self, tmp_path, monkeypatch):
        """Layer 1: one channel's entry overrides the block, in both directions."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "defaults": {"writable": True, "confirm": False},
                "PICKY:CHANNEL": {"min_value": 0.0, "max_value": 100.0, "confirm": True},
                "PLAIN:CHANNEL": {"min_value": 0.0, "max_value": 100.0},
            },
        )
        connector = await _connected_mock(monkeypatch, limits_file)

        picky = await connector.write_channel("PICKY:CHANNEL", 50.0)
        plain = await connector.write_channel("PLAIN:CHANNEL", 50.0)

        assert picky.outcome is WriteOutcome.CONFIRMED
        assert plain.outcome is WriteOutcome.UNREQUESTED

        await connector.disconnect()

    async def test_channel_entry_can_opt_out_of_a_confirming_fleet(self, tmp_path, monkeypatch):
        """The other direction of layer 1: one channel declines confirmation."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "defaults": {"writable": True, "confirm": True},
                "BLIND:CHANNEL": {"min_value": 0.0, "max_value": 100.0, "confirm": False},
            },
        )
        connector = await _connected_mock(monkeypatch, limits_file)

        result = await connector.write_channel("BLIND:CHANNEL", 50.0)

        assert result.outcome is WriteOutcome.UNREQUESTED

        await connector.disconnect()


class TestExplicitConfirmOverridesTheDatabase:
    """An explicit ``confirm`` is the caller's answer and outranks every layer."""

    async def test_explicit_false_overrides_a_confirming_channel(self, tmp_path, monkeypatch):
        limits_file = _write_limits_db(
            tmp_path,
            {
                "defaults": {"writable": True, "confirm": True},
                "PICKY:CHANNEL": {"min_value": 0.0, "max_value": 100.0, "confirm": True},
            },
        )
        connector = await _connected_mock(monkeypatch, limits_file)

        assert (await connector.write_channel("PICKY:CHANNEL", 50.0)).outcome is (
            WriteOutcome.CONFIRMED
        )

        overridden = await connector.write_channel("PICKY:CHANNEL", 50.0, confirm=False)
        assert overridden.outcome is WriteOutcome.UNREQUESTED

        await connector.disconnect()

    async def test_explicit_true_overrides_a_declining_channel(self, tmp_path, monkeypatch):
        limits_file = _write_limits_db(
            tmp_path,
            {
                "defaults": {"writable": True, "confirm": False},
                "BLIND:CHANNEL": {"min_value": 0.0, "max_value": 100.0},
            },
        )
        connector = await _connected_mock(monkeypatch, limits_file)

        result = await connector.write_channel("BLIND:CHANNEL", 50.0, confirm=True)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value is not None

        await connector.disconnect()

    async def test_explicit_none_resolves_like_an_omitted_confirm(self, tmp_path, monkeypatch):
        """``confirm=None`` is "no opinion", never ``False``.

        The batch path forwards nothing rather than ``False``, but a caller that
        does pass ``None`` explicitly must land in the same place as one that
        leaves the keyword off.
        """
        limits_file = _write_limits_db(
            tmp_path,
            {
                "defaults": {"writable": True, "confirm": False},
                "PLAIN:CHANNEL": {"min_value": 0.0, "max_value": 100.0},
            },
        )
        connector = await _connected_mock(monkeypatch, limits_file)

        explicit_none = await connector.write_channel("PLAIN:CHANNEL", 50.0, confirm=None)
        omitted = await connector.write_channel("PLAIN:CHANNEL", 50.0)

        assert explicit_none.outcome is WriteOutcome.UNREQUESTED
        assert omitted.outcome is WriteOutcome.UNREQUESTED

        await connector.disconnect()


class TestBatchConfirmResolution:
    """``write_multiple_channels`` resolves per channel, like single writes."""

    async def test_batch_resolves_each_channel_independently(self, tmp_path, monkeypatch):
        """An omitted ``confirm`` lets every channel in the batch decide its own."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "defaults": {"writable": True, "confirm": False},
                "BATCH:CH1": {"min_value": 0.0, "max_value": 100.0, "confirm": True},
                "BATCH:CH2": {"min_value": 0.0, "max_value": 100.0},
            },
        )
        connector = await _connected_mock(monkeypatch, limits_file)

        results = await connector.write_multiple_channels(
            [("BATCH:CH1", 10.0), ("BATCH:CH2", 20.0)]
        )

        assert [r.channel_address for r in results] == ["BATCH:CH1", "BATCH:CH2"]
        assert results[0].outcome is WriteOutcome.CONFIRMED
        assert results[1].outcome is WriteOutcome.UNREQUESTED

        await connector.disconnect()

    async def test_batch_forwards_an_explicit_confirm_to_every_channel(self, tmp_path, monkeypatch):
        """One ``confirm`` for the batch overrides what each channel declares."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "defaults": {"writable": True, "confirm": True},
                "BATCH:CH1": {"min_value": 0.0, "max_value": 100.0, "confirm": True},
                "BATCH:CH2": {"min_value": 0.0, "max_value": 100.0},
            },
        )
        connector = await _connected_mock(monkeypatch, limits_file)

        declined = await connector.write_multiple_channels(
            [("BATCH:CH1", 10.0), ("BATCH:CH2", 20.0)], confirm=False
        )
        assert [r.outcome for r in declined] == [
            WriteOutcome.UNREQUESTED,
            WriteOutcome.UNREQUESTED,
        ]

        await connector.disconnect()

    async def test_batch_without_a_limits_database_confirms_every_channel(self, monkeypatch):
        """No database, no policy: the fleet default applies to each channel."""
        connector = await _connected_mock(monkeypatch)

        results = await connector.write_multiple_channels(
            [("BATCH:CH1", 10.0), ("BATCH:CH2", 20.0)]
        )

        assert [r.outcome for r in results] == [
            WriteOutcome.CONFIRMED,
            WriteOutcome.CONFIRMED,
        ]

        await connector.disconnect()


class TestWritesDisabledOutranksConfirmation:
    """The ``writes_enabled`` gate refuses before any policy is resolved."""

    async def test_a_disabled_write_is_refused_not_unrequested(self):
        connector = MockConnector()
        with patch("osprey.utils.config.get_config_value", return_value=False):
            await connector.connect({"response_delay_ms": 0})

            result = await connector.write_channel("TEST:CHANNEL", 100.0, confirm=False)

            assert result.outcome is WriteOutcome.REFUSED
            assert result.refusal_reason == "WRITES_DISABLED"
            assert result.error_message is not None

            await connector.disconnect()
