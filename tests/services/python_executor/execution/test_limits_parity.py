"""The limits partition is held to the write-surface table.

A readonly run refuses every entry point in ``_CLIENT_WRITE_TARGETS``. A
readwrite run does something *different* with each one — checks the value
against the channel limits, refuses it outright, or cannot reach it at all.
Those three answers are written down as :data:`_LIMITS_WRAPPED`,
:data:`_LIMITS_REFUSED` and :data:`_LIMITS_UNWRAPPABLE`.

Written down, they can go stale: a row added to the table with no bucket would
quietly claim a limits check it never gets, and a bucket entry left behind by a
removed row would describe a write surface that no longer exists. Both
directions are pinned here, and every failure names the offending row.

The buckets are keyed by the canonical ``tango`` spelling. ``PyTango`` is the
legacy alias for the same package, so ``PyTango.DeviceProxy`` resolves to the
same class object and its fate is the ``tango.DeviceProxy`` row's fate; the
alias rows are canonicalised before the lookup rather than duplicated.
"""

import pytest

from osprey.services.python_executor.write_surface import (
    _CLIENT_WRITE_TARGETS,
    _LIMITS_REFUSED,
    _LIMITS_UNWRAPPABLE,
    _LIMITS_WRAPPED,
)

pytestmark = pytest.mark.unit


#: The buckets, by the name a failure should print.
_BUCKETS = {
    "_LIMITS_WRAPPED": _LIMITS_WRAPPED,
    "_LIMITS_REFUSED": _LIMITS_REFUSED,
    "_LIMITS_UNWRAPPABLE": _LIMITS_UNWRAPPABLE,
}

_ALIAS_PACKAGE = "PyTango"
_CANONICAL_PACKAGE = "tango"


def _canonical(dotted):
    """The spelling the buckets are keyed by.

    ``PyTango`` re-exports the ``tango`` package, so both dotted names resolve
    to one class object and a guard patching either patches both.
    """
    if dotted == _ALIAS_PACKAGE or dotted.startswith(_ALIAS_PACKAGE + "."):
        return _CANONICAL_PACKAGE + dotted[len(_ALIAS_PACKAGE) :]
    return dotted


def _table_rows():
    """Every ``(dotted, attribute)`` in the client half of the write surface."""
    return [(dotted, attr) for dotted, attrs in _CLIENT_WRITE_TARGETS for attr in attrs]


def test_every_client_row_is_in_exactly_one_bucket():
    """A row with no bucket, or with two, is the drift this partition exists to catch."""
    for dotted, attr in _table_rows():
        row = (_canonical(dotted), attr)
        holders = [name for name, bucket in _BUCKETS.items() if row in bucket]
        assert holders, (
            f"{dotted}.{attr} is in the write surface but in no limits bucket — "
            "say whether a limits-checked run wraps it, refuses it, or cannot "
            "reach it at all"
        )
        assert len(holders) == 1, (
            f"{dotted}.{attr} is in more than one limits bucket: {', '.join(holders)}"
        )


def test_no_bucket_names_a_row_outside_the_table():
    """A bucket entry that outlives its row would describe a surface that is gone."""
    table = {(_canonical(dotted), attr) for dotted, attr in _table_rows()}
    for name, bucket in _BUCKETS.items():
        for dotted, attr in bucket:
            assert (dotted, attr) in table, (
                f"{name} names {dotted}.{attr}, which is not in _CLIENT_WRITE_TARGETS"
            )


def test_every_bucket_entry_carries_a_reason():
    """A bucket is a reason per row; an empty one answers nothing."""
    for name, bucket in _BUCKETS.items():
        for (dotted, attr), reason in bucket.items():
            assert reason.strip(), f"{name}[{dotted}.{attr}] carries no reason"


def test_alias_rows_resolve_to_a_row_the_table_spells_canonically():
    """The legacy ``PyTango`` rows are covered by their ``tango`` twins, not on their own.

    That only holds while the canonical row exists. An alias row naming an
    attribute ``tango`` does not list would be bucketed by accident of the
    rewrite rather than by anyone having looked at it.
    """
    table = {(dotted, attr) for dotted, attr in _table_rows()}
    alias_rows = [row for row in table if row[0].startswith(_ALIAS_PACKAGE)]
    assert alias_rows, "the alias contract this test pins no longer has any rows"
    for dotted, attr in alias_rows:
        assert (_canonical(dotted), attr) in table, (
            f"{dotted}.{attr} has no {_CANONICAL_PACKAGE} row to inherit its limits bucket from"
        )
