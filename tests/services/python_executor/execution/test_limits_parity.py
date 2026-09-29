"""The armed partition is held to the write-surface table.

A readonly run refuses every entry point in ``_CLIENT_WRITE_TARGETS``. A run
with writes armed answers each one differently — refuses the raw call because
the connector is the one route to the machine, limits-checks it because the
connector has no route for it yet, keeps an action's own split, or lets it
through because it writes no device. Those four answers are written down as
:data:`_ARMED_BLOCKED`, :data:`_ARMED_CHECKED`, :data:`_ARMED_RPC` and
:data:`_ARMED_PASSED`.

Written down, they can go stale: a row added to the table with no bucket would
have no stated fate in an armed run, and a bucket entry left behind by a
removed row would describe a write surface that no longer exists. Both
directions are pinned here, and every failure names the offending row.

The buckets are keyed by the table's own spelling, ``PyTango`` rows included:
each row a guard patches carries its own answer.
"""

from osprey.services.python_executor.write_surface import (
    _ARMED_BLOCKED,
    _ARMED_CHECKED,
    _ARMED_PASSED,
    _ARMED_RPC,
    _CLIENT_WRITE_TARGETS,
)

#: The buckets, by the name a failure should print.
_BUCKETS = {
    "_ARMED_BLOCKED": _ARMED_BLOCKED,
    "_ARMED_CHECKED": _ARMED_CHECKED,
    "_ARMED_RPC": _ARMED_RPC,
    "_ARMED_PASSED": _ARMED_PASSED,
}


def _table_rows():
    """Every ``(dotted, attribute)`` in the client half of the write surface."""
    return [(dotted, attr) for dotted, attrs in _CLIENT_WRITE_TARGETS for attr in attrs]


def test_every_client_row_is_in_exactly_one_bucket():
    """A row with no bucket, or with two, is the drift this partition exists to catch."""
    for row in _table_rows():
        holders = [name for name, bucket in _BUCKETS.items() if row in bucket]
        assert holders, (
            f"{row[0]}.{row[1]} is in the write surface but in no armed bucket — "
            "say whether an armed run blocks it, limits-checks it, keeps its rpc split, "
            "or passes it"
        )
        assert len(holders) == 1, (
            f"{row[0]}.{row[1]} is in more than one armed bucket: {', '.join(holders)}"
        )


def test_no_bucket_names_a_row_outside_the_table():
    """A bucket entry that outlives its row would describe a surface that is gone."""
    table = set(_table_rows())
    for name, bucket in _BUCKETS.items():
        for dotted, attr in bucket:
            assert (dotted, attr) in table, (
                f"{name} names {dotted}.{attr}, which is not in _CLIENT_WRITE_TARGETS"
            )


def test_the_buckets_partition_the_table_exactly():
    """Together the buckets are the table: same rows, none counted twice."""
    rows = _table_rows()
    assert len(rows) == len(set(rows)), "the table lists a row twice"
    assert set().union(*_BUCKETS.values()) == set(rows)
    assert sum(len(bucket) for bucket in _BUCKETS.values()) == len(rows)


def test_every_bucket_entry_carries_a_reason():
    """A bucket is a reason per row; an empty one answers nothing."""
    for name, bucket in _BUCKETS.items():
        for (dotted, attr), reason in bucket.items():
            assert isinstance(reason, str) and reason.strip(), (
                f"{name}[{dotted}.{attr}] carries no reason"
            )


def test_synchronous_group_put_is_in_the_table_and_blocked():
    """``epics.ca.sg_put`` writes a channel like ``put`` does, only deferred to a flush."""
    assert ("epics.ca", "sg_put") in _table_rows()
    assert ("epics.ca", "sg_put") in _ARMED_BLOCKED


def test_rpc_bucket_holds_the_actions_only():
    """The rpc split covers p4p rpc and the Tango command spellings, nothing else."""
    assert set(_ARMED_RPC) == {
        *((f"p4p.client.{f}.Context", "rpc") for f in ("raw", "thread", "asyncio", "cothread")),
        *(
            (cls, attr)
            for cls in ("tango.DeviceProxy", "tango.Connection", "tango.Group")
            for attr in ("command_inout", "command_inout_asynch")
        ),
        ("PyTango.DeviceProxy", "command_inout"),
    }


def test_passed_bucket_holds_the_non_device_writes_only():
    """Only server-side SharedPV calls and the Tango database write pass an armed run."""
    assert set(_ARMED_PASSED) == {
        *(
            (f"p4p.server.{f}.SharedPV", attr)
            for f in ("raw", "thread", "asyncio")
            for attr in ("post", "open")
        ),
        ("tango.DeviceProxy", "put_property"),
    }


def test_checked_bucket_holds_the_pvaccess_puts_only():
    """PVAccess puts are the one write the connector cannot carry yet, so they
    keep their limits check instead of being refused; nothing else may."""
    assert {dotted.split(".")[0] for dotted, _attr in _ARMED_CHECKED} == {"p4p", "pvaccess"}
    assert all(attr.startswith(("put", "asyncPut", "parsePut")) for _dotted, attr in _ARMED_CHECKED)
    assert not any(dotted.startswith("p4p.server") for dotted, _attr in _ARMED_CHECKED)
