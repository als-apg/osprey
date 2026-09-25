"""Which earlier runs a dispatched agent may read back."""

from __future__ import annotations

import uuid

import pytest

from osprey.mcp_server.dispatch_worker.prior_answers import (
    MAX_PRIOR_ANSWER_RUNS,
    format_run_ids,
    is_run_id,
    keep_run_ids,
    parse_run_ids,
)

_RUN = "3f2b6c1e-8a4d-4f0e-9b1a-2c3d4e5f6a7b"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (_RUN, True),
        (_RUN.upper(), False),
        ("{" + _RUN + "}", False),
        ("run-1", False),
        ("../x", False),
        ("", False),
        (None, False),
        (123, False),
    ],
)
def test_is_run_id_accepts_only_canonical_uuids(value, expected):
    assert is_run_id(value) is expected


def test_keep_run_ids_drops_malformed_and_duplicate_ids():
    other = str(uuid.uuid4())

    assert keep_run_ids([_RUN, "../etc", None, _RUN, other, 7]) == [_RUN, other]


def test_keep_run_ids_keeps_the_newest_when_over_the_cap():
    ids = [str(uuid.uuid4()) for _ in range(MAX_PRIOR_ANSWER_RUNS + 1)]

    assert keep_run_ids(ids) == ids[1:]


def test_format_and_parse_round_trip():
    ids = [str(uuid.uuid4()) for _ in range(3)]

    assert parse_run_ids(format_run_ids(ids)) == frozenset(ids)


@pytest.mark.parametrize("raw", [None, "", " , "])
def test_parse_run_ids_of_nothing_is_empty(raw):
    assert parse_run_ids(raw) == frozenset()
