"""The seed's value-transform hook: what it may change, and what it must not.

The archive belongs to the machine, and a model has no past. Where a deployment
serves a live stand-in, the machine whose present is recorded is the perturbed
one — so the seeded past has to carry the same systematic offsets, and
:func:`synthesize_documents` grew one seam for a caller to declare that with.

The seam is narrow on purpose, and these tests pin both halves of that:

* **It reaches the values and nothing else.** A transform changes what a
  channel's samples are, never how many documents there are, what order they
  come in, what instants they carry, or which channels they hold. A hook that
  could reshape the store would be a second implementation of the grid.

* **It cannot be applied invisibly.** A store seeded with offsets holds
  different numbers than one seeded without them, so the fingerprint has to say
  so — otherwise the next deploy reads MATCH and leaves a past belonging to
  another machine in place. The legacy case matters too: a manifest written
  before the field existed describes an untransformed store, and must keep
  comparing MATCH rather than reseeding for a knob nobody changed.

No container is needed anywhere here. The arithmetic, the document shape and
the fingerprint are pure, and the two calls a manifest comparison makes on a
store are stubbed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

from osprey.simulation.archiver_seed import (
    DATE_FIELD,
    MANIFEST_ID,
    SeedKnobs,
    SeedState,
    compare_fingerprint,
    seed_base,
    seed_fingerprint,
    synthesize_documents,
    write_manifest,
)

# A fixed anchor, so a failure is reproducible tomorrow.
T0 = datetime(2026, 3, 14, 9, 26, 53, tzinfo=UTC)

SMALL = SeedKnobs(retention_days=1, hot_span_hours=1, hot_cadence_sec=600, tail_cadence_sec=3600)

CHANNELS = [
    {"address": "SR:DIAG:BPM:03:POSITION:X", "record_type": "ai"},
    {"address": "SR:DIAG:BPM:03:POSITION:Y", "record_type": "ai"},
    {"address": "SR:VAC:IP07:PRESSURE", "record_type": "ai"},
    {"address": "SR:STATUS:VALID", "record_type": "bi"},
]
ADDRESSES = [str(channel["address"]) for channel in CHANNELS]

# The perturbed address and the offset on it, in the shape the stand-in ships:
# a static transverse displacement, subtracted from the truth to give a reading.
PERTURBED = "SR:DIAG:BPM:03:POSITION:X"
OFFSET = 1.5e-4

EPOCHS = np.asarray([T0.timestamp() + step * 10.0 for step in range(6)])


def _subtract_offset(address: str, values):
    """The host's stand-in transform, in miniature: one address, one offset."""
    if address != PERTURBED:
        return values
    return [float(value) - OFFSET for value in values]


class _StubCollection:
    """The calls :func:`seed_base` and :func:`compare_fingerprint` make, and no more."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] | None = None

    def create_index(self, *args: Any, **kwargs: Any) -> None:
        return None

    def insert_many(self, documents: list[dict[str, Any]], **kwargs: Any) -> None:
        self.documents.extend(documents)

    def replace_one(self, _filter: Any, document: dict[str, Any], **kwargs: Any) -> None:
        self.manifest = document

    def find_one(self, _filter: Any, **kwargs: Any) -> dict[str, Any] | None:
        return self.manifest


# ---------------------------------------------------------------------------
# What the transform reaches
# ---------------------------------------------------------------------------


def test_seed_transform_subtracts_the_offset_on_the_address_it_names() -> None:
    """Exactly ``value - offset``, not approximately.

    The stand-in's readout is ``(truth - offset)`` through a chain left at
    identity, so the seed reproduces it by subtraction of the value it already
    synthesized. Anything looser here would be the seed inventing physics.
    """
    plain = synthesize_documents(CHANNELS, EPOCHS)

    transformed = synthesize_documents(CHANNELS, EPOCHS, value_transform=_subtract_offset)

    assert [document[PERTURBED] for document in transformed] == [
        document[PERTURBED] - OFFSET for document in plain
    ]


def test_seed_transform_passes_every_other_address_through_untouched() -> None:
    """A channel the transform does not name keeps the value this module computed —
    including the flag channel, whose type must survive the pass."""
    plain = synthesize_documents(CHANNELS, EPOCHS)

    transformed = synthesize_documents(CHANNELS, EPOCHS, value_transform=_subtract_offset)

    for address in ADDRESSES:
        if address == PERTURBED:
            continue
        assert [d[address] for d in transformed] == [d[address] for d in plain]
    assert all(isinstance(document["SR:STATUS:VALID"], bool) for document in transformed)


def test_seed_transform_of_none_is_the_untransformed_seed() -> None:
    """The default path is byte-identical to not having the parameter at all."""
    assert synthesize_documents(CHANNELS, EPOCHS, value_transform=None) == synthesize_documents(
        CHANNELS, EPOCHS
    )


def test_seed_transform_does_not_change_the_document_shape_or_order() -> None:
    """One document per timestamp, ascending, each carrying every channel.

    The transform is a seam over *values*. A hook that could add, drop or
    reorder documents would be a second opinion about the grid, which is the
    one thing this module refuses to have.
    """
    documents = synthesize_documents(CHANNELS, EPOCHS, value_transform=_subtract_offset)

    assert len(documents) == len(EPOCHS)
    stamps = [document[DATE_FIELD] for document in documents]
    assert stamps == sorted(stamps)
    assert [stamp.timestamp() for stamp in stamps] == list(EPOCHS)
    for document in documents:
        assert set(document) == {DATE_FIELD, *ADDRESSES}


def test_seed_transform_returning_the_wrong_length_is_refused() -> None:
    """A short return would leave the tail of a channel's window unwritten, which
    reads back as a channel that simply stops rather than as a bug."""
    with pytest.raises(ValueError):
        synthesize_documents(CHANNELS, EPOCHS, value_transform=lambda address, values: values[:2])


# ---------------------------------------------------------------------------
# seed_base forwards it
# ---------------------------------------------------------------------------


def test_seed_base_applies_the_seed_transform_to_every_chunk() -> None:
    """Including across a chunk boundary: synthesis runs per chunk, so a hook
    threaded into only the first would leave most of the window unperturbed."""
    collection = _StubCollection()
    plain = _StubCollection()

    seed_base(
        collection,  # type: ignore[arg-type]
        CHANNELS,
        SMALL,
        t0=T0,
        chunk_size=2,
        value_transform=_subtract_offset,
        transform_fingerprint={"kind": "bpm_offsets", "offsets": {PERTURBED: OFFSET}},
    )
    seed_base(plain, CHANNELS, SMALL, t0=T0, chunk_size=2)  # type: ignore[arg-type]

    # Keyed on the sample instant, not on arrival: the inserts run on a pool and
    # are deliberately unordered, so chunk order is not part of the contract.
    seeded = {document[DATE_FIELD]: document[PERTURBED] for document in collection.documents}
    untransformed = {document[DATE_FIELD]: document[PERTURBED] for document in plain.documents}
    assert len(seeded) == len(untransformed) > 2
    assert seeded == {stamp: value - OFFSET for stamp, value in untransformed.items()}


def test_seed_base_records_the_seed_transform_in_the_manifest() -> None:
    """The manifest is what the next deploy compares against, so the description
    has to survive the write rather than only the call."""
    collection = _StubCollection()

    seed_base(
        collection,  # type: ignore[arg-type]
        CHANNELS,
        SMALL,
        t0=T0,
        chunk_size=64,
        value_transform=_subtract_offset,
        transform_fingerprint={"kind": "bpm_offsets", "offsets": {PERTURBED: OFFSET}},
    )

    assert collection.manifest is not None
    stored = collection.manifest["fingerprint"]["value_transform"]
    assert stored is not None
    assert PERTURBED in stored


# ---------------------------------------------------------------------------
# The fingerprint
# ---------------------------------------------------------------------------


def _fingerprint(transform_fingerprint=None) -> dict[str, Any]:
    return seed_fingerprint(
        SMALL, ADDRESSES, compression="zstd", transform_fingerprint=transform_fingerprint
    )


def test_a_seed_transform_moves_the_fingerprint() -> None:
    """Without this, a store seeded with the stand-in's offsets compares MATCH
    against one seeded without them — and the deploy that introduced the
    stand-in would leave another machine's past in place."""
    assert _fingerprint({"kind": "bpm_offsets", "offsets": {PERTURBED: OFFSET}}) != _fingerprint()


def test_a_changed_seed_transform_moves_the_fingerprint() -> None:
    """A different offset is a different past, not a different opinion about one."""
    one = _fingerprint({"kind": "bpm_offsets", "offsets": {PERTURBED: OFFSET}})

    other = _fingerprint({"kind": "bpm_offsets", "offsets": {PERTURBED: -OFFSET}})

    assert one["value_transform"] != other["value_transform"]


def test_the_seed_transform_description_is_folded_in_canonically() -> None:
    """Key order is not a knob. Two spellings of one description must not read as
    two stores, or a dict that happened to iterate differently would reseed."""
    one = _fingerprint({"kind": "bpm_offsets", "offsets": {"A": 1.0, "B": 2.0}})

    other = _fingerprint({"offsets": {"B": 2.0, "A": 1.0}, "kind": "bpm_offsets"})

    assert one == other


def test_no_seed_transform_leaves_the_fingerprint_as_it_was() -> None:
    """The field is present and null rather than absent: that is what lets a
    manifest predating it compare equal (see the legacy test below)."""
    fingerprint = _fingerprint()

    assert fingerprint["value_transform"] is None
    assert {key: value for key, value in fingerprint.items() if key != "value_transform"} == {
        "schema_version": 1,
        "retention_days": SMALL.retention_days,
        "hot_span_hours": SMALL.hot_span_hours,
        "hot_cadence_sec": SMALL.hot_cadence_sec,
        "tail_cadence_sec": SMALL.tail_cadence_sec,
        "compression": "zstd",
        "channel_count": len(ADDRESSES),
        "channel_set_sha256": _fingerprint()["channel_set_sha256"],
    }


def test_a_manifest_predating_the_seed_transform_field_still_matches() -> None:
    """An untransformed store seeded by an older OSPREY has no ``value_transform``
    in its manifest. Absent means no transform, which is what it was — reseeding
    it would throw away a correct multi-minute seed for a knob nobody moved."""
    collection = _StubCollection()
    legacy = _fingerprint()
    del legacy["value_transform"]
    write_manifest(collection, legacy, seeded_at=T0)  # type: ignore[arg-type]

    comparison = compare_fingerprint(collection, _fingerprint())  # type: ignore[arg-type]

    assert comparison.state is SeedState.MATCH


def test_a_manifest_predating_the_seed_transform_field_mismatches_a_transformed_seed() -> None:
    """The other half of the same rule: a stand-in added to an existing deployment
    does have to rebuild the base, and the diff says which knob moved."""
    collection = _StubCollection()
    legacy = _fingerprint()
    del legacy["value_transform"]
    write_manifest(collection, legacy, seeded_at=T0)  # type: ignore[arg-type]

    comparison = compare_fingerprint(
        collection,  # type: ignore[arg-type]
        _fingerprint({"kind": "bpm_offsets", "offsets": {PERTURBED: OFFSET}}),
    )

    assert comparison.state is SeedState.MISMATCH
    assert [key for key, _stored, _expected in comparison.differences] == ["value_transform"]
    assert collection.manifest is not None
    assert collection.manifest["_id"] == MANIFEST_ID
