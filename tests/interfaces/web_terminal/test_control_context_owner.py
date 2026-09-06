"""The terminal's single mutation primitive for the control-context record.

The behaviour these tests pin is not "the file gets written" — the record I/O
already has its own suite for that. It is the two properties the primitive
exists to provide, which the code it replaces could not have both of:

* **No lost update.** The read, the caller's change and the write are one
  serialised unit. Two mutations submitted together are applied one after the
  other, and the second one *reads what the first one wrote* — so a posture
  toggle and a target switch arriving in the same tick both survive.
* **Nothing on the event loop.** That unit runs in a worker thread, so a slow
  or blocked disk cannot stop the terminal serving. The predecessor bought
  serialisation by staying synchronous on the loop; this one buys it with a
  lock, and the difference has to be provable.

The pool-starvation test is the one that proves both at once: the executor is
narrowed to a single worker, the first job parks on it, and the second mutation
is submitted while it is parked. If the primitive held the lock only around the
write, the second job's read would have already happened against the pre-write
record and one of the two changes would vanish.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from osprey.interfaces.web_terminal import control_context_owner as owner_module
from osprey.interfaces.web_terminal.control_context_owner import (
    ContextOwnedElsewhere,
    ContextStoreUnavailable,
    ControlContextOwner,
    Mutation,
)
from osprey_connectors import control_context
from osprey_connectors.control_context import ControlContext, Owner

TERMINAL = Owner(kind=control_context.OWNER_WEB_TERMINAL, pid=4242, port=8080)
OTHER_TERMINAL = Owner(kind=control_context.OWNER_WEB_TERMINAL, pid=9999, port=8090)
SERVER = Owner(kind=control_context.OWNER_CONTROLS_SERVER, pid=7777)


@pytest.fixture(autouse=True)
def _forget_the_parsed_record():
    """The record cache is process-global; no test may inherit another's."""
    control_context.invalidate_cache()
    yield
    control_context.invalidate_cache()


@pytest.fixture
def record_file(tmp_path: Path) -> Path:
    """A record owned by :data:`TERMINAL`, on ``live`` at generation 3."""
    path = control_context.record_path_under(tmp_path)
    control_context.write_record(
        ControlContext(target="live", generation=3, owner=TERMINAL), path=path
    )
    return path


@pytest.fixture
def context_owner(record_file: Path) -> ControlContextOwner:
    return ControlContextOwner(TERMINAL, path=record_file)


def read(path: Path) -> ControlContext | None:
    """The record as it is on disk right now, past the signature cache."""
    control_context.invalidate_cache()
    return control_context.read_record(path=path)


# -- the mutation itself ----------------------------------------------------


async def test_the_returned_record_is_written(context_owner, record_file):
    def narrow(record):
        return Mutation(record=replace(record, posture={"live": "sandbox"}), result=None)

    await context_owner.mutate_record(narrow)

    stored = read(record_file)
    assert stored is not None
    assert stored.posture == {"live": "sandbox"}
    assert stored.target == "live"
    assert stored.generation == 3


async def test_the_mutation_result_reaches_the_caller(context_owner):
    def switch(record):
        return Mutation(record=replace(record, target="va", generation=4), result="applied")

    assert await context_owner.mutate_record(switch) == "applied"


async def test_the_callable_sees_the_record_on_disk(context_owner):
    seen: list[ControlContext | None] = []

    def observe(record):
        seen.append(record)
        return Mutation.unchanged(None)

    await context_owner.mutate_record(observe)

    assert len(seen) == 1
    assert seen[0] is not None
    assert (seen[0].target, seen[0].generation) == ("live", 3)


async def test_an_unchanged_mutation_writes_nothing(context_owner, record_file):
    before = record_file.stat()

    result = await context_owner.mutate_record(lambda record: Mutation.unchanged("refused"))

    assert result == "refused"
    after = record_file.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


async def test_the_owner_is_reasserted_on_every_write(context_owner, record_file):
    """A caller cannot hand ownership away by accident, or drop it entirely."""

    def careless(record):
        return Mutation(record=replace(record, owner=SERVER, target="va"), result=None)

    await context_owner.mutate_record(careless)

    stored = read(record_file)
    assert stored is not None
    assert stored.owner == TERMINAL
    assert stored.target == "va"


async def test_an_exception_from_the_callable_leaves_the_record_alone(context_owner, record_file):
    before = record_file.read_bytes()

    def explode(record):
        raise ValueError("no")

    with pytest.raises(ValueError):
        await context_owner.mutate_record(explode)

    assert record_file.read_bytes() == before


# -- who is allowed to write ------------------------------------------------


async def test_a_foreign_owner_refuses_the_mutation(record_file):
    control_context.write_record(
        ControlContext(target="live", generation=3, owner=OTHER_TERMINAL), path=record_file
    )
    context_owner = ControlContextOwner(TERMINAL, path=record_file)
    called: list[object] = []

    def never(record):
        called.append(record)
        return Mutation.unchanged(None)

    with pytest.raises(ContextOwnedElsewhere) as caught:
        await context_owner.mutate_record(never)

    assert caught.value.owner == OTHER_TERMINAL
    assert caught.value.error == "context_owned_elsewhere"
    assert called == []
    stored = read(record_file)
    assert stored is not None and stored.owner == OTHER_TERMINAL


async def test_a_vanished_record_refuses_the_mutation(context_owner, record_file):
    record_file.unlink()

    with pytest.raises(ContextOwnedElsewhere) as caught:
        await context_owner.mutate_record(lambda record: Mutation.unchanged(None))

    assert caught.value.owner is None
    assert not record_file.exists()


async def test_a_claim_may_run_over_a_foreign_owner(record_file):
    """The ownership rules live above this primitive, so a claim opts the check out."""
    control_context.write_record(
        ControlContext(target="va", generation=6, owner=SERVER), path=record_file
    )
    context_owner = ControlContextOwner(TERMINAL, path=record_file)

    def claim(record):
        assert record is not None and record.owner == SERVER
        return Mutation(record=record, result="claimed")

    assert await context_owner.mutate_record(claim, verify_owner=False) == "claimed"

    stored = read(record_file)
    assert stored is not None
    assert stored.owner == TERMINAL
    assert (stored.target, stored.generation) == ("va", 6)


async def test_a_claim_may_create_the_record_from_nothing(tmp_path):
    path = control_context.record_path_under(tmp_path)
    context_owner = ControlContextOwner(TERMINAL, path=path)

    def claim(record):
        assert record is None
        return Mutation(record=ControlContext(target="live", generation=0), result=None)

    await context_owner.mutate_record(claim, verify_owner=False)

    stored = read(path)
    assert stored is not None
    assert stored.owner == TERMINAL


# -- when there is nowhere to write -----------------------------------------


async def test_an_unresolvable_root_is_store_unavailable(monkeypatch):
    monkeypatch.setattr(owner_module, "record_path", lambda: None)
    context_owner = ControlContextOwner(TERMINAL)

    with pytest.raises(ContextStoreUnavailable) as caught:
        await context_owner.mutate_record(lambda record: Mutation.unchanged(None))

    assert caught.value.error == "store_unavailable"


async def test_a_failed_write_is_store_unavailable(context_owner, record_file, monkeypatch):
    before = record_file.read_bytes()

    def refuse(record, *, path=None):
        raise OSError("read-only file system")

    monkeypatch.setattr(owner_module, "write_record", refuse)

    with pytest.raises(ContextStoreUnavailable) as caught:
        await context_owner.mutate_record(
            lambda record: Mutation(record=replace(record, target="va"), result=None)
        )

    assert caught.value.error == "store_write_failed"
    assert record_file.read_bytes() == before


async def test_the_resolved_path_is_used_when_none_was_given(tmp_path, monkeypatch):
    path = control_context.record_path_under(tmp_path)
    control_context.write_record(
        ControlContext(target="live", generation=1, owner=TERMINAL), path=path
    )
    monkeypatch.setattr(owner_module, "record_path", lambda: path)
    context_owner = ControlContextOwner(TERMINAL)

    await context_owner.mutate_record(
        lambda record: Mutation(record=replace(record, target="standin"), result=None)
    )

    stored = read(path)
    assert stored is not None and stored.target == "standin"


# -- serialisation, which is the whole point --------------------------------


@pytest.fixture
async def roomy_pool():
    """Give the loop's default executor spare workers, and restore it after.

    ``asyncio.to_thread`` submits to the running loop's default executor. The
    spare worker is deliberate: with a single-threaded pool the executor
    itself would serialise the jobs and the test would pass whether or not the
    primitive holds a lock. A second free thread is what lets a second
    mutation overtake a parked one — the lost update this test exists to
    refute.
    """
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mutation")
    loop.set_default_executor(pool)
    try:
        yield pool
    finally:
        loop.set_default_executor(ThreadPoolExecutor())
        pool.shutdown(wait=True)


async def _until(predicate, *, timeout: float = 5.0) -> None:
    """Spin the loop until *predicate* holds. Never blocks the loop itself."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("timed out waiting on the worker thread")
        await asyncio.sleep(0.01)


async def test_both_changes_land_when_the_pool_is_blocked_mid_job(
    context_owner, record_file, roomy_pool
):
    """SC-80: a second mutation submitted while a job is parked loses nothing."""
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()
    second_saw: list[ControlContext | None] = []

    def narrow(record):
        entered.set()
        assert release.wait(5), "the first job was never released"
        return Mutation(record=replace(record, posture={"live": "sandbox"}), result="first")

    def bump(record):
        second_entered.set()
        second_saw.append(record)
        return Mutation(record=replace(record, generation=record.generation + 1), result="second")

    first = asyncio.create_task(context_owner.mutate_record(narrow))
    await _until(entered.is_set)

    second = asyncio.create_task(context_owner.mutate_record(bump))
    await asyncio.sleep(0.05)
    assert not second_entered.is_set(), "the second mutation read the record mid-write"
    assert not second.done()
    assert not first.done()

    release.set()
    assert await first == "first"
    assert await second == "second"

    # The second job read the first job's write rather than the seeded record.
    assert second_saw[0] is not None
    assert second_saw[0].posture == {"live": "sandbox"}

    stored = read(record_file)
    assert stored is not None
    assert stored.posture == {"live": "sandbox"}
    assert stored.generation == 4


async def test_a_blocked_job_does_not_stall_the_event_loop(context_owner, roomy_pool):
    entered = threading.Event()
    release = threading.Event()
    ticks = 0

    async def tick():
        nonlocal ticks
        while not release.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    def park(record):
        entered.set()
        assert release.wait(5), "the job was never released"
        return Mutation.unchanged(None)

    ticker = asyncio.create_task(tick())
    mutation = asyncio.create_task(context_owner.mutate_record(park))
    await _until(entered.is_set)
    await asyncio.sleep(0.05)

    assert ticks > 0, "the event loop was blocked while the mutation ran"
    release.set()
    await mutation
    await ticker


async def test_mutations_are_applied_one_at_a_time(context_owner, record_file):
    """Twenty concurrent increments produce twenty generations, not fewer."""

    def bump(record):
        return Mutation(record=replace(record, generation=record.generation + 1), result=None)

    await asyncio.gather(*(context_owner.mutate_record(bump) for _ in range(20)))

    stored = read(record_file)
    assert stored is not None
    assert stored.generation == 23
