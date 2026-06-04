"""Tests for DispatchPool."""

import asyncio

import pytest

from osprey.dispatch.pool import DispatchPool, QueueFullError


@pytest.mark.asyncio
async def test_submit_runs_dispatch_fn_when_capacity_available():
    """Submit runs dispatch_fn immediately when pool has capacity."""
    pool = DispatchPool(max_concurrent=2, max_queue_depth=5)
    called = []

    async def fn():
        called.append(True)

    dispatch_id = await pool.submit("test_trigger", fn)
    await asyncio.sleep(0)  # yield to allow task to run
    await asyncio.sleep(0)

    assert isinstance(dispatch_id, str)
    assert len(dispatch_id) == 36  # UUID format
    assert called == [True]


@pytest.mark.asyncio
async def test_pool_respects_max_concurrent():
    """Pool never runs more than max_concurrent dispatches simultaneously."""
    max_concurrent = 2
    pool = DispatchPool(max_concurrent=max_concurrent, max_queue_depth=10)

    hold_event = asyncio.Event()
    running_count = []
    peak_concurrent = [0]

    async def fn():
        running_count.append(1)
        peak_concurrent[0] = max(peak_concurrent[0], sum(running_count))
        await hold_event.wait()
        running_count.pop()

    # Submit more tasks than max_concurrent
    for i in range(4):
        await pool.submit(f"trigger_{i}", fn)

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    status = pool.get_pool_status()
    assert status["running"] <= max_concurrent

    # Release all held dispatches
    hold_event.set()
    await asyncio.sleep(0.1)

    assert peak_concurrent[0] <= max_concurrent


@pytest.mark.asyncio
async def test_queue_full_error_raised_when_exceeded():
    """QueueFullError is raised when queue exceeds max_queue_depth."""
    pool = DispatchPool(max_concurrent=1, max_queue_depth=2)

    hold_event = asyncio.Event()

    async def blocking_fn():
        await hold_event.wait()

    # Fill the running slot
    await pool.submit("trigger_0", blocking_fn)
    await asyncio.sleep(0)

    # Fill the queue
    await pool.submit("trigger_1", blocking_fn)
    await pool.submit("trigger_2", blocking_fn)

    # Next submit should raise
    with pytest.raises(QueueFullError):
        await pool.submit("trigger_overflow", blocking_fn)

    hold_event.set()
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_get_pool_status_returns_correct_counts():
    """get_pool_status returns accurate running, max, and queued counts."""
    pool = DispatchPool(max_concurrent=2, max_queue_depth=10)

    hold_event = asyncio.Event()

    async def blocking_fn():
        await hold_event.wait()

    # Submit 2 (fills running slots) + 3 queued
    for i in range(5):
        await pool.submit(f"trigger_{i}", blocking_fn)

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    status = pool.get_pool_status()
    assert status["max"] == 2
    assert status["running"] == 2
    assert status["queued"] == 3

    hold_event.set()
    await asyncio.sleep(0.1)

    final = pool.get_pool_status()
    assert final["running"] == 0
    assert final["queued"] == 0


@pytest.mark.asyncio
async def test_fifo_ordering_is_maintained():
    """Queued dispatches are executed in FIFO order."""
    pool = DispatchPool(max_concurrent=1, max_queue_depth=10)

    hold_event = asyncio.Event()
    execution_order = []

    async def make_fn(label):
        async def fn():
            await hold_event.wait()
            execution_order.append(label)

        return fn

    # Submit 1 (fills running slot) then 3 queued
    await pool.submit("first", await make_fn("first"))
    await asyncio.sleep(0)

    for label in ["second", "third", "fourth"]:
        await pool.submit(label, await make_fn(label))

    hold_event.set()
    await asyncio.sleep(0.2)

    assert execution_order == ["first", "second", "third", "fourth"]


@pytest.mark.asyncio
async def test_error_path_records_error_status_and_releases_slot():
    """When dispatch_fn raises, result is recorded as error and the slot is freed."""
    pool = DispatchPool(max_concurrent=1, max_queue_depth=5)

    async def failing_fn():
        raise ValueError("boom")

    failing_id = await pool.submit("failing_trigger", failing_fn)
    await asyncio.sleep(0)  # yield so _run executes
    await asyncio.sleep(0)

    result = pool.get_result(failing_id)
    assert result["status"] == "error"
    assert result["error"] == "boom"
    assert result["trigger_name"] == "failing_trigger"

    # Slot must have been freed in the finally block: a new submit runs immediately.
    ran = []

    async def ok_fn():
        ran.append(True)

    next_id = await pool.submit("next_trigger", ok_fn)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert ran == [True]
    assert pool.get_result(next_id)["status"] == "completed"
    # Counters returned to zero.
    assert pool.get_pool_status()["running"] == 0


@pytest.mark.asyncio
async def test_slot_released_after_success():
    """After a fn completes successfully the running count returns to 0."""
    pool = DispatchPool(max_concurrent=1, max_queue_depth=5)

    async def ok_fn():
        return "value"

    dispatch_id = await pool.submit("ok_trigger", ok_fn)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert pool.get_result(dispatch_id)["status"] == "completed"
    assert pool.get_result(dispatch_id)["result"] == "value"
    assert pool.get_pool_status()["running"] == 0


@pytest.mark.asyncio
async def test_queue_full_does_not_leak_overflow_result():
    """A QueueFullError submit leaves no result entry for the rejected dispatch_id."""
    pool = DispatchPool(max_concurrent=1, max_queue_depth=1)

    hold_event = asyncio.Event()

    async def blocking_fn():
        await hold_event.wait()

    # Pin the single running slot deterministically.
    await pool.submit("running", blocking_fn)
    await asyncio.sleep(0)

    # Fill the queue to max_queue_depth.
    await pool.submit("queued", blocking_fn)

    # Capture the set of known result keys before the rejected submit.
    keys_before = set(pool._results.keys())

    with pytest.raises(QueueFullError):
        await pool.submit("overflow", blocking_fn)

    # No new result key was leaked by the rejected submit.
    assert set(pool._results.keys()) == keys_before

    hold_event.set()
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_queued_item_drains_after_running_slot_frees():
    """A queued dispatch runs once the running slot is released (FIFO drain)."""
    pool = DispatchPool(max_concurrent=1, max_queue_depth=5)

    first_event = asyncio.Event()

    async def first_fn():
        await first_event.wait()
        return "first-result"

    async def second_fn():
        return "second-result"

    first_id = await pool.submit("first", first_fn)
    await asyncio.sleep(0)  # let first claim the running slot

    second_id = await pool.submit("second", second_fn)
    await asyncio.sleep(0)

    # Second is still queued while first holds the slot.
    assert pool.get_pool_status()["queued"] == 1
    assert pool.get_result(second_id)["status"] == "pending"

    # Release the first; the queued second should drain and complete.
    first_event.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert pool.get_result(first_id)["status"] == "completed"
    assert pool.get_result(first_id)["result"] == "first-result"
    assert pool.get_result(second_id)["status"] == "completed"
    assert pool.get_result(second_id)["result"] == "second-result"
    assert pool.get_pool_status()["running"] == 0
    assert pool.get_pool_status()["queued"] == 0


@pytest.mark.asyncio
async def test_get_result_returns_none_for_unknown_id():
    """get_result returns None for a dispatch_id the pool has never seen."""
    pool = DispatchPool(max_concurrent=1, max_queue_depth=5)

    assert pool.get_result("nonexistent-dispatch-id") is None
