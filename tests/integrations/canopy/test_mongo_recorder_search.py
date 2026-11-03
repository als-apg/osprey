"""The recorder adapter's ``search`` and ``metadata`` against a throwaway archive store.

``search`` validates its regular expression, matches whole names, sorts and
then applies the limit; neither call reports a bookkeeping field as a channel.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from canopy.core.errors import BadRequest
from canopy.mole.adapters.base import PVNotFoundInArchiver
from canopy.mole.config import ArchiverInstanceConfig
from canopy.mole.schemas.pv import PVMetadata
from tests.integrations.canopy import _recorder_store as store

from osprey.integrations.canopy.mongo_recorder import RECORDER_TYPE, OspreyMongoRecorderAdapter

NAME = "osprey"
BPM_01 = "BTS:DIAG:BPM:01:POSITION:X"
BPM_02 = "BTS:DIAG:BPM:02:POSITION:X"
BPM_10 = "BTS:DIAG:BPM:10:POSITION:X"
BPM_01_Y = "BTS:DIAG:BPM:01:POSITION:Y"
STATE = "BTS:DIAG:STATE"
CHANNELS = [BPM_01, BPM_02, BPM_10, BPM_01_Y, STATE]

#: A fixed whole-second epoch (2026-09-30T12:00:00Z).
BASE_S = 1_790_769_600

BAD_PATTERN = "pattern is a regular expression (use .* not *)"


@pytest.fixture
async def adapter(recorder_archive) -> AsyncIterator[OspreyMongoRecorderAdapter]:
    store.seed_samples(
        recorder_archive.writer,
        (
            (store.utc_slot(BASE_S + i), {pv: float(i + k) for k, pv in enumerate(CHANNELS)})
            for i in range(20)
        ),
    )
    store.write_manifest(recorder_archive.collection)
    cfg = ArchiverInstanceConfig(
        type=RECORDER_TYPE,
        url=recorder_archive.url,
        api_key_env=store.PASSWORD_ENV,
        timeout_s=10.0,
        extras={"collection": store.COLLECTION},
    )
    made = OspreyMongoRecorderAdapter.from_config(NAME, cfg)
    assert made.config_error is None, made.config_error
    yield made
    await made.close()


async def test_search_full_matches_and_sorts(adapter) -> None:
    assert await adapter.search("BTS:DIAG:BPM:.*:POSITION:X", limit=100) == [BPM_01, BPM_02, BPM_10]
    assert await adapter.search(".*", limit=100) == sorted(CHANNELS)


async def test_search_is_a_full_match_not_a_substring_search(adapter) -> None:
    assert await adapter.search("BPM", limit=100) == []
    assert await adapter.search("BTS:DIAG:BPM:01", limit=100) == []
    assert await adapter.search(STATE, limit=100) == [STATE]


async def test_search_supports_anchors_and_classes(adapter) -> None:
    assert await adapter.search("^BTS:DIAG:BPM:0[12]:POSITION:X$", limit=100) == [BPM_01, BPM_02]


async def test_search_applies_limit_after_sorting(adapter) -> None:
    assert await adapter.search(".*BPM.*", limit=2) == [BPM_01, BPM_01_Y]
    assert await adapter.search(".*", limit=1) == [sorted(CHANNELS)[0]]


async def test_search_excludes_bookkeeping_fields(adapter) -> None:
    found = await adapter.search(".*", limit=1000)
    assert not {"_id", "date", "expireAt", "fingerprint", "seeded_at"} & set(found)


@pytest.mark.parametrize(
    "pattern",
    [
        "*BPM*",  # a glob, not a regex: fails re.compile
        "BPM[",  # unterminated class
        "BTS:DIAG:BPM(01)",  # parentheses are outside the alphabet
        "BTS DIAG",  # space is outside the alphabet
        "a|b",  # alternation is outside the alphabet
        ".*a.*b.*c.*d",  # four stars
        "A" * 129,  # too long
    ],
)
async def test_search_rejects_bad_patterns(adapter, pattern: str) -> None:
    with pytest.raises(BadRequest) as exc:
        await adapter.search(pattern, limit=100)
    assert exc.value.detail == BAD_PATTERN


async def test_search_accepts_the_edges_of_the_limits(adapter) -> None:
    assert await adapter.search(".*a.*b.*c", limit=100) == []
    assert await adapter.search("A" * 128, limit=100) == []


async def test_metadata_names_the_pv_and_archiver(adapter) -> None:
    meta = await adapter.metadata(BPM_01)
    assert meta == PVMetadata(pv=BPM_01, archiver=NAME)
    assert meta.units is None and meta.precision is None


async def test_metadata_of_an_unknown_pv_raises(adapter) -> None:
    with pytest.raises(PVNotFoundInArchiver):
        await adapter.metadata("PV:UNKNOWN")
