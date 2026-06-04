"""Tests for TriggerRegistry."""

import pytest

from osprey.dispatch.registry import TriggerRegistry
from osprey.dispatch.trigger_config import TriggerConfig


def _make_trigger(name: str, source: str = "webhook") -> TriggerConfig:
    return TriggerConfig(
        name=name,
        source=source,
        action={"prompt": f"Do something for {name}"},
    )


@pytest.mark.asyncio
async def test_register_and_list():
    """register() adds trigger; list_triggers() returns it with correct fields."""
    reg = TriggerRegistry()
    t = _make_trigger("beam_loss", "epics")
    await reg.register(t)

    listing = await reg.list_triggers()
    assert len(listing) == 1
    entry = listing[0]
    assert entry["name"] == "beam_loss"
    assert entry["source"] == "epics"
    assert entry["status"] == "active"
    assert entry["last_fired"] is None


@pytest.mark.asyncio
async def test_get_status_known_trigger():
    """get_status returns correct dict for a registered trigger."""
    reg = TriggerRegistry()
    await reg.register(_make_trigger("fill_pattern"))
    status = await reg.get_status("fill_pattern")
    assert status["name"] == "fill_pattern"
    assert status["status"] == "active"
    assert status["last_fired"] is None


@pytest.mark.asyncio
async def test_get_status_unknown_trigger_raises():
    """get_status raises KeyError for unregistered trigger."""
    reg = TriggerRegistry()
    with pytest.raises(KeyError, match="not registered"):
        await reg.get_status("nonexistent")


@pytest.mark.asyncio
async def test_record_event_updates_last_fired_and_history():
    """record_event updates last_fired and appends to history."""
    reg = TriggerRegistry()
    await reg.register(_make_trigger("orbit_drift"))
    await reg.record_event("orbit_drift", {"bpm": 42}, "ok")

    status = await reg.get_status("orbit_drift")
    assert status["last_fired"] is not None

    history = await reg.get_history("orbit_drift")
    assert len(history) == 1
    entry = history[0]
    assert entry["event_data"] == {"bpm": 42}
    assert entry["result"] == "ok"
    assert "timestamp" in entry


@pytest.mark.asyncio
async def test_get_history_respects_limit():
    """get_history(limit=N) returns at most N most recent entries."""
    reg = TriggerRegistry()
    await reg.register(_make_trigger("heartbeat"))
    for i in range(10):
        await reg.record_event("heartbeat", {"i": i}, "ok")

    recent = await reg.get_history("heartbeat", limit=3)
    assert len(recent) == 3
    # Should be the last 3
    assert [e["event_data"]["i"] for e in recent] == [7, 8, 9]


@pytest.mark.asyncio
async def test_re_registration_resets_status_but_keeps_history():
    """Re-registering a trigger resets status fields but preserves existing history deque."""
    reg = TriggerRegistry()
    await reg.register(_make_trigger("bpm_spike"))
    await reg.record_event("bpm_spike", {"val": 1}, "ok")

    # Re-register
    await reg.register(_make_trigger("bpm_spike", source="new_source"))
    status = await reg.get_status("bpm_spike")
    assert status["source"] == "new_source"
    assert status["last_fired"] is None

    # History deque is preserved (not wiped)
    history = await reg.get_history("bpm_spike")
    assert len(history) == 1


@pytest.mark.asyncio
async def test_record_event_unknown_trigger_raises():
    """record_event raises KeyError for unregistered trigger."""
    reg = TriggerRegistry()
    with pytest.raises(KeyError, match="not registered"):
        await reg.record_event("ghost", {}, "ok")


@pytest.mark.asyncio
async def test_multiple_triggers_isolated():
    """Events for one trigger do not appear in another's history."""
    reg = TriggerRegistry()
    await reg.register(_make_trigger("alpha"))
    await reg.register(_make_trigger("beta"))

    await reg.record_event("alpha", {"x": 1}, "ok")
    await reg.record_event("alpha", {"x": 2}, "ok")
    await reg.record_event("beta", {"y": 99}, "ok")

    alpha_hist = await reg.get_history("alpha")
    beta_hist = await reg.get_history("beta")
    assert len(alpha_hist) == 2
    assert len(beta_hist) == 1


@pytest.mark.asyncio
async def test_record_event_truncates_oversized_payload():
    """record_event replaces a payload whose JSON exceeds the cap with a truncation marker."""
    from osprey.dispatch.registry import _MAX_EVENT_DATA_BYTES

    reg = TriggerRegistry()
    await reg.register(_make_trigger("big_payload"))

    # Build a payload whose JSON serialization exceeds the cap.
    big_value = "x" * (_MAX_EVENT_DATA_BYTES + 1000)
    await reg.record_event("big_payload", {"blob": big_value}, "ok")

    history = await reg.get_history("big_payload")
    assert len(history) == 1
    stored = history[0]["event_data"]
    assert stored["_truncated"] is True
    assert stored["_original_size"] > _MAX_EVENT_DATA_BYTES
    # Preview is capped at 4096 chars of the JSON string.
    assert len(stored["_preview"]) == 4096
    assert "blob" not in stored


@pytest.mark.asyncio
async def test_record_event_keeps_small_payload_untruncated():
    """A payload under the cap is stored verbatim (no truncation marker)."""
    reg = TriggerRegistry()
    await reg.register(_make_trigger("small_payload"))
    await reg.record_event("small_payload", {"k": "v"}, "ok")

    stored = (await reg.get_history("small_payload"))[0]["event_data"]
    assert stored == {"k": "v"}
    assert "_truncated" not in stored


@pytest.mark.asyncio
async def test_get_history_since_seconds_filters_old_events():
    """get_history(since_seconds=0) excludes already-recorded events; a wide window includes them."""
    reg = TriggerRegistry()
    await reg.register(_make_trigger("windowed"))
    await reg.record_event("windowed", {"n": 1}, "ok")

    # With a zero-second window, the cutoff is "now" and a slightly-earlier
    # recorded event falls before it, so it is filtered out.
    assert await reg.get_history("windowed", since_seconds=0) == []

    # With a generous window, the event is within range and returned.
    recent = await reg.get_history("windowed", since_seconds=3600)
    assert len(recent) == 1
    assert recent[0]["event_data"] == {"n": 1}


@pytest.mark.asyncio
async def test_get_history_since_seconds_none_returns_all():
    """since_seconds=None (default) applies no time filtering."""
    reg = TriggerRegistry()
    await reg.register(_make_trigger("nofilter"))
    await reg.record_event("nofilter", {"n": 1}, "ok")
    await reg.record_event("nofilter", {"n": 2}, "ok")

    history = await reg.get_history("nofilter", since_seconds=None)
    assert len(history) == 2


@pytest.mark.asyncio
async def test_get_history_limit_zero_returns_all():
    """limit<=0 returns all entries (no slicing) in newest-last order."""
    reg = TriggerRegistry()
    await reg.register(_make_trigger("allhist"))
    for i in range(5):
        await reg.record_event("allhist", {"i": i}, "ok")

    all_zero = await reg.get_history("allhist", limit=0)
    assert [e["event_data"]["i"] for e in all_zero] == [0, 1, 2, 3, 4]

    all_neg = await reg.get_history("allhist", limit=-1)
    assert [e["event_data"]["i"] for e in all_neg] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_get_history_newest_last_ordering():
    """get_history returns the most recent `limit` entries in newest-last order."""
    reg = TriggerRegistry()
    await reg.register(_make_trigger("ordered"))
    for i in range(6):
        await reg.record_event("ordered", {"i": i}, "ok")

    recent = await reg.get_history("ordered", limit=4)
    assert [e["event_data"]["i"] for e in recent] == [2, 3, 4, 5]


@pytest.mark.asyncio
async def test_get_history_unknown_trigger_raises():
    """get_history raises KeyError for an unregistered trigger."""
    reg = TriggerRegistry()
    with pytest.raises(KeyError, match="not registered"):
        await reg.get_history("ghost")


@pytest.mark.asyncio
async def test_set_status_valid_transitions():
    """set_status accepts 'disabled' and 'active' and updates the stored status."""
    reg = TriggerRegistry()
    await reg.register(_make_trigger("toggle"))

    assert await reg.set_status("toggle", "disabled") == "disabled"
    assert (await reg.get_status("toggle"))["status"] == "disabled"

    assert await reg.set_status("toggle", "active") == "active"
    assert (await reg.get_status("toggle"))["status"] == "active"


@pytest.mark.asyncio
async def test_set_status_invalid_status_raises_value_error():
    """set_status raises ValueError for a status outside the valid set."""
    reg = TriggerRegistry()
    await reg.register(_make_trigger("badstatus"))
    with pytest.raises(ValueError, match="Invalid status"):
        await reg.set_status("badstatus", "paused")


@pytest.mark.asyncio
async def test_set_status_unknown_trigger_raises_key_error():
    """set_status raises KeyError for an unregistered trigger (with a valid status)."""
    reg = TriggerRegistry()
    with pytest.raises(KeyError, match="not registered"):
        await reg.set_status("ghost", "disabled")


@pytest.mark.asyncio
async def test_history_deque_capped_at_history_max():
    """The per-trigger history deque is bounded by _HISTORY_MAX via its maxlen."""
    from osprey.dispatch.registry import _HISTORY_MAX

    reg = TriggerRegistry()
    await reg.register(_make_trigger("capped"))

    # Verify the cap is configured on the deque rather than overflowing it
    # (default _HISTORY_MAX is large).
    assert reg._history["capped"].maxlen == _HISTORY_MAX
