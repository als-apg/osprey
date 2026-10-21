"""The archive's one field-name encoding: reversible, and the identity where it can be."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from datetime import UTC, datetime

import pytest

from osprey_connectors.archiver.field_names import (
    _RESERVED_FIELDS,
    channel_address,
    field_name,
)
from tests._builds import BuiltProject

ADDRESSES = [
    "SR:BPM1:X",
    "REC.RBV",
    "REC.HLS",
    "REC.LLS",
    "REC.MIP",
    "REC.VELO",
    "REC.CW",
    "REC.CCW",
    "REC.MOVN",
    "A.B.C",
    "A.B",
    "A%2EB",
    "A%252EB",
    "$LEAD",
    "MID$DLE",
    "PCT%25",
    "100%",
    "%2E",
    "NUL\x00X",
    "a{b}-c_d:e",
    "REC.",
    ".REC",
    "%64ate",
    "Date",
    "dates",
    "SR:date",
    "_ID",
]

# The archive's own document fields and the field a channel of that name is
# stored under: the first character escaped, the rest kept.
RESERVED = {
    "_id": "%5Fid",
    "date": "%64ate",
    "expireAt": "%65xpireAt",
    "osprey_densified": "%6Fsprey_densified",
    "fingerprint": "%66ingerprint",
    "seeded_at": "%73eeded_at",
    "touched_windows": "%74ouched_windows",
    "touched_anchor": "%74ouched_anchor",
    "coverage": "%63overage",
}

EVERY_ADDRESS = [*ADDRESSES, *RESERVED]


def test_the_table_is_the_reserved_list() -> None:
    assert set(RESERVED) == _RESERVED_FIELDS


@pytest.mark.parametrize(("name", "field"), RESERVED.items())
def test_a_reserved_name_is_stored_with_its_first_character_escaped(name: str, field: str) -> None:
    assert field_name(name) == field
    assert channel_address(field) == name


@pytest.mark.parametrize("address", EVERY_ADDRESS)
def test_every_address_round_trips(address: str) -> None:
    assert channel_address(field_name(address)) == address


@pytest.mark.parametrize("address", EVERY_ADDRESS)
def test_a_field_name_holds_nothing_mongodb_reads_as_syntax(address: str) -> None:
    field = field_name(address)
    assert "." not in field
    assert "\x00" not in field
    assert not field.startswith("$")


def test_distinct_addresses_never_share_a_field() -> None:
    fields = [field_name(address) for address in EVERY_ADDRESS]
    for (a, fa), (b, fb) in itertools.combinations(zip(EVERY_ADDRESS, fields, strict=True), 2):
        assert fa != fb, (a, b)


@pytest.mark.parametrize("address", EVERY_ADDRESS)
def test_no_field_is_a_reserved_name(address: str) -> None:
    assert field_name(address) not in _RESERVED_FIELDS


def test_an_address_storable_before_encodes_to_itself(
    built_control_assistant: BuiltProject,
) -> None:
    """An archive written before this encoding holds only such fields, so it reads back unchanged.

    Every address with no ``%``, no ``.``, no NUL and no leading ``$``, and that
    is not one of the archive's own document fields, is its own field name; the
    shipped demo facility's channels are all of that kind.
    """
    view = json.loads(built_control_assistant.outputs[0].files["data/simulator/addresses.json"])
    plain = [
        address
        for address in ADDRESSES
        if not any(c in address for c in "%.\x00")
        and not address.startswith("$")
        and address not in _RESERVED_FIELDS
    ]
    addresses = [*view["channels"], *plain]
    assert view["channels"]
    for address in addresses:
        assert field_name(address) == address


@pytest.mark.parametrize("field", ["A%2", "A%zz", "A%2e"])
def test_an_unescaped_percent_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match="not an archive field name"):
        channel_address(field)


def test_an_empty_address_is_refused() -> None:
    with pytest.raises(ValueError):
        field_name("")


def test_the_module_imports_only_the_standard_library() -> None:
    probe = (
        "import sys\n"
        "import osprey_connectors.archiver.field_names\n"
        "assert 'pymongo' not in sys.modules, 'pymongo was imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


class _Recording:
    """A collection stand-in that keeps every document, filter and update it is handed."""

    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.filters: list[dict] = []
        self.updates: list[dict] = []
        self.upserts: list[bool] = []

    def replace_one(self, filter: dict, document: dict, upsert: bool = False) -> None:
        self.filters.append(filter)
        self.documents.append(document)
        self.upserts.append(upsert)

    def update_one(self, filter: dict, update: dict, upsert: bool = False) -> None:
        self.filters.append(filter)
        self.updates.append(update)
        self.upserts.append(upsert)


class _Series:
    """An archive composite stand-in: every channel reads 1.0 at every moment."""

    def series(self, _pv: str, moments: list[float]) -> list[float]:
        return [1.0] * len(moments)


def test_the_reserved_list_is_every_field_the_archive_writes_besides_a_channel() -> None:
    """Each writer of the archive collection, asked what it writes beside a channel.

    The list in ``field_names`` is literal because that module imports nothing
    from the writers; this is what binds it to them, in both directions.
    """
    from osprey.services.archiver_recorder.config import RecorderSettings
    from osprey.services.archiver_recorder.store import ArchiveWriter
    from osprey.simulation import apply
    from osprey_connectors.simulation import archive

    instant = datetime(2026, 1, 1, tzinfo=UTC)
    # MongoDB's primary key: every document carries it whether or not a writer names it.
    fields = {"_id"}

    # The base seeder's sample documents.
    fields |= {archive.DATE_FIELD, archive.EXPIRE_FIELD}

    manifest = _Recording()
    archive.write_manifest(
        manifest,
        {"schema_version": 2},
        seeded_at=instant,
        report=archive.SeedReport(channels=1),
    )
    fields |= set(manifest.documents[0])

    ledger = _Recording()
    apply._write_ledger(ledger, {"A": (0.0, 1.0)}, 0.0)
    fields |= set(ledger.updates[0]["$set"])

    [dense] = apply._dense_documents(_Series(), {}, {0.0: ("A",)})
    fields |= set(dense) - {field_name("A")}
    fields.add(apply.EXPIRE_FIELD_NAME)

    recorded = _Recording()
    writer = ArchiveWriter(
        RecorderSettings(
            host="archiver-mongodb",
            port=27017,
            database="osprey_archiver",
            collection="pv_history",
            auth_source="admin",
            username="osprey",
            password_env="MONGO_ROOT_PASSWORD",
            timeout_s=5,
            cadence_sec=10,
            tail_cadence_sec=60,
            poll_sec=30,
            hot_span_hours=48,
            retention_days=30,
        ),
        "pw",
    )
    writer._collection = recorded
    writer.write_sample(instant, {"A": 1.0})
    fields |= set(recorded.filters[0]) | set(recorded.updates[0]["$setOnInsert"])

    assert fields == _RESERVED_FIELDS
