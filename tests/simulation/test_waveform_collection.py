"""Where archived waveforms live, and what one stored trace looks like.

Traces do not ride in the per-timestamp snapshot documents: each one is its own
document in a sibling collection. Every writer (the seeder, the live recorder),
every reader and the reseed drop must agree on that collection's name, so it is
derived in exactly one place, :func:`waveform_collection`, and a source scan
pins that nothing else spells the suffix.

The stored document is equally shared: :func:`waveform_document` is the single
definition the seeder, the recorder and the reader are each asserted against.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.archiver.field_names import field_name
from osprey_connectors.simulation import archive
from osprey_connectors.simulation.archive import (
    DATE_FIELD,
    EXPIRE_FIELD,
    prepare_collection,
    waveform_collection,
    waveform_document,
)
from tests._container_support import is_docker_available
from tests._mongo_container import MONGO_AUTH_DB, started_mongo

REPO_ROOT = Path(__file__).resolve().parents[2]

T0 = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)
EXPIRES = T0 + timedelta(days=30)
CHANNEL = "SR:SEK:WAVEFORM"


# ---------------------------------------------------------------------------
# The name
# ---------------------------------------------------------------------------


def test_the_waveform_collection_is_the_scalar_name_plus_a_suffix():
    assert waveform_collection("x") == "x_waveforms"
    assert waveform_collection("pv_history") == "pv_history_waveforms"


def test_the_helper_is_part_of_the_public_surface():
    assert "waveform_collection" in archive.__all__
    assert "waveform_document" in archive.__all__


#: The suffix spelled as string content: the end of a literal or an f-string.
#: An identifier such as ``read_waveforms(`` does not match, which is the point —
#: only a *name being built* is a second definition of the collection.
_SUFFIX_SPELLING = re.compile(r"""_waveforms["']""")


def test_no_other_code_spells_the_suffix():
    """One derivation, so a renamed suffix can never leave a writer behind."""
    archive_py = Path(archive.__file__).resolve()
    hits: list[str] = []
    for root in (REPO_ROOT / "src", REPO_ROOT / "packages"):
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for number, line in enumerate(text.splitlines(), start=1):
                if _SUFFIX_SPELLING.search(line):
                    hits.append(f"{path.resolve()}:{number}")

    assert len(hits) == 1, hits
    assert hits[0].startswith(f"{archive_py}:")


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def test_a_waveform_document_holds_date_trace_and_expiry():
    values = [0.0, 0.5, 1.0, 0.75]

    document = waveform_document(CHANNEL, T0, values, expire_at=EXPIRES)

    assert document == {DATE_FIELD: T0, field_name(CHANNEL): values, EXPIRE_FIELD: EXPIRES}


def test_the_trace_is_a_plain_list_of_floats_of_the_channel_shape():
    """Plain doubles (BSON arrays), not numpy scalars pymongo cannot encode."""
    import numpy as np

    values = np.linspace(0.0, 1.0, 500, dtype=np.float32)

    trace = waveform_document(CHANNEL, T0, values, expire_at=EXPIRES)[field_name(CHANNEL)]

    assert isinstance(trace, list)
    assert len(trace) == 500
    assert all(type(value) is float for value in trace)
    assert trace[-1] == 1.0


def test_integer_samples_are_stored_as_floats():
    trace = waveform_document(CHANNEL, T0, [1, 2, 3], expire_at=EXPIRES)[field_name(CHANNEL)]

    assert trace == [1.0, 2.0, 3.0]
    assert all(type(value) is float for value in trace)


def test_the_field_is_the_channels_archive_field_name():
    """The same naming a scalar sample uses, so a reader needs no second rule."""
    address = "SR01C:SEK:Waveform.VAL"

    document = waveform_document(address, T0, [1.0], expire_at=EXPIRES)

    assert set(document) == {DATE_FIELD, field_name(address), EXPIRE_FIELD}


@pytest.mark.parametrize("values", [[], [[1.0, 2.0], [3.0, 4.0]]], ids=["empty", "2-d"])
def test_a_trace_must_be_a_non_empty_one_dimensional_array(values):
    with pytest.raises(ValueError):
        waveform_document(CHANNEL, T0, values, expire_at=EXPIRES)


# ---------------------------------------------------------------------------
# Preparation: the same indexes as the scalar collection
# ---------------------------------------------------------------------------


class _IndexLog:
    """A collection handle that records the indexes asked of it."""

    def __init__(self) -> None:
        self.indexes: list[tuple[Any, dict[str, Any]]] = []

    def create_index(self, keys: Any, **kwargs: Any) -> None:
        self.indexes.append((keys, kwargs))


def test_prepare_gives_the_waveform_collection_the_scalar_indexes():
    scalar, waveforms = _IndexLog(), _IndexLog()

    prepare_collection(scalar, waveforms=waveforms)

    assert waveforms.indexes == scalar.indexes
    assert (EXPIRE_FIELD, {"expireAfterSeconds": 0}) in waveforms.indexes
    assert (DATE_FIELD, {}) in waveforms.indexes


def test_prepare_without_a_waveform_handle_touches_only_the_scalar_collection():
    scalar = _IndexLog()

    prepare_collection(scalar)

    assert scalar.indexes


@pytest.fixture(scope="module")
def mongo_database():
    """A real store, so the index is checked as MongoDB records it."""
    if not is_docker_available():
        pytest.skip("Docker not available — needed to inspect real indexes.")

    from pymongo import MongoClient

    username, password = "waveuser", "wavepass123"
    with started_mongo("mongodb-waveform", username=username, password=password) as (
        host,
        port,
    ):
        client = MongoClient(
            host=host,
            port=port,
            username=username,
            password=password,
            authSource=MONGO_AUTH_DB,
        )
        try:
            yield client["waveform_test_db"]
        finally:
            client.close()


def _ttl_indexes(handle) -> dict[str, int]:
    return {
        info["key"][0][0]: info["expireAfterSeconds"]
        for info in handle.index_information().values()
        if "expireAfterSeconds" in info
    }


def test_the_prepared_waveform_collection_carries_the_expiry_index(mongo_database):
    scalar = mongo_database["pv_history"]
    waveforms = mongo_database[waveform_collection(scalar.name)]
    scalar.drop()
    waveforms.drop()
    try:
        prepare_collection(scalar, waveforms=waveforms)

        assert _ttl_indexes(waveforms) == _ttl_indexes(scalar) == {EXPIRE_FIELD: 0}
        assert any(
            info["key"] == [(DATE_FIELD, 1)] for info in waveforms.index_information().values()
        )
    finally:
        scalar.drop()
        waveforms.drop()
