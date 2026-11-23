"""The panels' vendored Screen-IR schema is the proven PyDMConverter's, byte for byte.

``osprey_panels`` ships a copy of ``pydmconverter/ir/data/screen-ir.schema.json``
and names the Screen-IR version it writes (``screen_ir.SCREEN_IR_VERSION``);
``proven.SCREEN_IR_VERSION`` names the version of the PyDMConverter this OSPREY
is proven against. Here both are held to the installed PyDMConverter, which the
contract runs at the pinned commit, so moving the pin without taking the schema
again fails. Importing PyDMConverter at module level carries the ``canopy`` mark.
"""

from __future__ import annotations

from importlib.resources import files

import pydmconverter.ir

from osprey.integrations.canopy import proven
from osprey_panels import screen_ir

#: The schema's file name, in the package and in PyDMConverter alike.
SCHEMA_FILE = "screen-ir.schema.json"


def vendored_schema() -> bytes:
    return files("osprey_panels").joinpath(SCHEMA_FILE).read_bytes()


def converter_schema() -> bytes:
    return files(pydmconverter.ir).joinpath("data", SCHEMA_FILE).read_bytes()


def same_schema(vendored: bytes, converter: bytes) -> bool:
    """Whether the vendored schema is the converter's copy: equal bytes, nothing looser."""
    return vendored == converter


def test_the_vendored_schema_is_the_converters_copy() -> None:
    assert same_schema(vendored_schema(), converter_schema())


def test_an_altered_schema_is_not_the_converters_copy() -> None:
    converter = converter_schema()
    vendored = converter.replace(b'"1.1"', b'"1.2"', 1)
    assert vendored != converter, "the alteration must change the bytes"
    assert not same_schema(vendored, converter)
    assert not same_schema(converter + b"\n", converter)


def test_the_proven_version_is_the_one_the_panels_write() -> None:
    assert proven.SCREEN_IR_VERSION == screen_ir.SCREEN_IR_VERSION


def test_the_proven_version_is_the_converters_schema_version() -> None:
    assert screen_ir.schema()["$id"] == f"urn:canopy:screen-ir:{proven.SCREEN_IR_VERSION}"
