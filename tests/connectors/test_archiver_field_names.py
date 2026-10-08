"""The archive's one field-name encoding: reversible, and the identity where it can be."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys

import pytest

from osprey_connectors.archiver.field_names import channel_address, field_name
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
]


@pytest.mark.parametrize("address", ADDRESSES)
def test_every_address_round_trips(address: str) -> None:
    assert channel_address(field_name(address)) == address


@pytest.mark.parametrize("address", ADDRESSES)
def test_a_field_name_holds_nothing_mongodb_reads_as_syntax(address: str) -> None:
    field = field_name(address)
    assert "." not in field
    assert "\x00" not in field
    assert not field.startswith("$")


def test_distinct_addresses_never_share_a_field() -> None:
    fields = [field_name(address) for address in ADDRESSES]
    for (a, fa), (b, fb) in itertools.combinations(zip(ADDRESSES, fields, strict=True), 2):
        assert fa != fb, (a, b)


def test_an_address_storable_before_encodes_to_itself(
    built_control_assistant: BuiltProject,
) -> None:
    """An archive written before this encoding holds only such fields, so it reads back unchanged.

    Every address with no ``%``, no ``.``, no NUL and no leading ``$`` is its own
    field name; the shipped demo facility's channels are all of that kind.
    """
    view = json.loads(built_control_assistant.outputs[0].files["data/simulator/addresses.json"])
    plain = [
        address
        for address in ADDRESSES
        if not any(c in address for c in "%.\x00") and not address.startswith("$")
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
