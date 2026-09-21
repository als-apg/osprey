"""The variable catalog is derived from the tree the service is given.

Every source here -- the channel manifest, ``machine.json`` and
``channel_limits.json`` -- arrives from one served data tree, so the bands a
model refuses to boot against are the bands that facility shipped. The tests
below fix that: a nominal outside its band refuses, and the band it was
weighed against is the one in the served file rather than any tree the
package happens to carry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from lume.variables import ScalarVariable

from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_STATIC_NOISY,
    READBACK_SUBFIELD,
    SETPOINT_SUBFIELD,
)
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths
from osprey.services.virtual_accelerator.model import catalog as catalog_module
from osprey.services.virtual_accelerator.model.catalog import (
    _load_limit_bands,
    build_variable_catalog,
)

# Addresses carry no facility's vocabulary: the catalog reads a channel's
# declared partition and subfield, never its address text.
SETPOINT = "R1:PWR:FAM_A:01:CUR:SP"
READBACK = "R1:PWR:FAM_A:01:CUR:RB"
READING = "R1:DIA:MON_A:01:POS:X"
STATIC = "R1:VAC:GAU_A:01:PRES:RB"


def _channel(address: str, *, partition: str, subfield: str, **overrides: Any) -> dict:
    """One manifest channel in the full per-channel schema."""
    channel = {
        "address": address,
        "ring": "R1",
        "system": "PWR",
        "family": "FAM_A",
        "device": "01",
        "field": "CUR",
        "subfield": subfield,
        "partition": partition,
        "record_type": "ai",
        "noise": False,
    }
    channel.update(overrides)
    return channel


@pytest.fixture
def manifest() -> list[dict]:
    """A manifest holding one of each case the catalog decides between."""
    return [
        _channel(SETPOINT, partition=PARTITION_PYAT_COUPLED, subfield=SETPOINT_SUBFIELD),
        _channel(READBACK, partition=PARTITION_PYAT_COUPLED, subfield=READBACK_SUBFIELD),
        _channel(
            READING,
            partition=PARTITION_PYAT_COUPLED,
            subfield="X",
            system="DIA",
            family="MON_A",
            field="POS",
        ),
        _channel(
            STATIC,
            partition=PARTITION_STATIC_NOISY,
            subfield=READBACK_SUBFIELD,
            system="VAC",
            family="GAU_A",
            field="PRES",
        ),
    ]


def _write_tree(
    root: Path,
    *,
    machine: dict | None = None,
    limits: dict | None = None,
) -> ManifestPaths:
    """Lay out a served data tree and return its :class:`ManifestPaths`."""
    root.mkdir(parents=True, exist_ok=True)
    if machine is not None:
        (root / "simulation").mkdir(parents=True, exist_ok=True)
        (root / "simulation" / "machine.json").write_text(
            json.dumps({"name": "fixture", "channels": machine})
        )
    if limits is not None:
        (root / "channel_limits.json").write_text(json.dumps(limits))
    return ManifestPaths(data_root=root)


@pytest.fixture
def paths(tmp_path: Path) -> ManifestPaths:
    """A served tree whose nominal sits inside its band."""
    return _write_tree(
        tmp_path / "data",
        machine={
            SETPOINT: {"value": 10.0, "units": "A"},
            READING: {"value": 0.0, "units": "mm"},
            STATIC: {"value": 5e-8, "units": "Torr"},
        },
        limits={
            "_version": "1.0",
            "defaults": {"writable": True, "confirm": True},
            SETPOINT: {"min_value": 0.0, "max_value": 20.0},
            READING: {"min_value": -1.0, "max_value": 1.0, "writable": False},
        },
    )


class TestCatalogShape:
    """Which channels become variables, and what each one declares."""

    def test_holds_the_coupled_channels_the_model_serves(self, paths, manifest):
        """Setpoints and readings are variables; the setpoint echo and the
        channels no lattice backs are not."""
        catalog = build_variable_catalog(paths, manifest, {})

        assert set(catalog) == {SETPOINT, READING}

    def test_setpoint_declares_the_served_nominal_unit_and_band(self, paths, manifest):
        variable = build_variable_catalog(paths, manifest, {})[SETPOINT]

        assert variable.name == SETPOINT
        assert variable.read_only is False
        assert variable.default_value == 10.0
        assert variable.unit == "A"
        assert variable.value_range == (0.0, 20.0)

    def test_reading_is_read_only_and_declares_no_band(self, paths, manifest):
        """A band is a write limit: the model reads a monitor, so a range on
        it would declare an enforcement nothing performs."""
        variable = build_variable_catalog(paths, manifest, {})[READING]

        assert variable.read_only is True
        assert variable.value_range is None
        assert variable.unit == "mm"

    def test_channel_absent_from_the_served_machine_json_declares_no_nominal(
        self, tmp_path, manifest
    ):
        paths = _write_tree(
            tmp_path / "data",
            machine={},
            limits={SETPOINT: {"min_value": 0.0, "max_value": 20.0}},
        )

        variable = build_variable_catalog(paths, manifest, {})[SETPOINT]

        assert variable.default_value is None
        assert variable.unit is None


class TestServedBands:
    """The band a variable is weighed against comes from the served file."""

    def test_the_band_is_the_one_the_tree_ships(self, tmp_path, manifest):
        """Two trees, same manifest, different limits: each catalog declares
        its own tree's band."""
        machine = {SETPOINT: {"value": 10.0}}
        narrow = _write_tree(
            tmp_path / "narrow",
            machine=machine,
            limits={SETPOINT: {"min_value": 9.0, "max_value": 11.0}},
        )
        wide = _write_tree(
            tmp_path / "wide",
            machine=machine,
            limits={SETPOINT: {"min_value": 0.0, "max_value": 100.0}},
        )

        assert build_variable_catalog(narrow, manifest, {})[SETPOINT].value_range == (9.0, 11.0)
        assert build_variable_catalog(wide, manifest, {})[SETPOINT].value_range == (0.0, 100.0)

    def test_nominal_outside_the_served_band_refuses(self, tmp_path, manifest):
        """The boot refusal: a nominal the served band excludes fails
        construction rather than serving a value the facility's own limits
        say cannot be written."""
        paths = _write_tree(
            tmp_path / "data",
            machine={SETPOINT: {"value": 30.0}},
            limits={SETPOINT: {"min_value": 0.0, "max_value": 20.0}},
        )

        with pytest.raises(ValueError, match="out of valid range"):
            build_variable_catalog(paths, manifest, {})

    def test_missing_machine_json_names_the_served_file(self, tmp_path, manifest):
        """No packaged fallback: a tree without a scenario seed fails, rather
        than seeding a facility's addresses from another tree's nominals."""
        paths = _write_tree(
            tmp_path / "data", limits={SETPOINT: {"min_value": 0.0, "max_value": 20.0}}
        )

        with pytest.raises(FileNotFoundError) as excinfo:
            build_variable_catalog(paths, manifest, {})

        assert str(paths.machine_json) in str(excinfo.value)

    def test_missing_limits_file_names_the_served_file(self, tmp_path, manifest):
        """Likewise for the limits: the model declares the bands it was
        given, and invents none."""
        paths = _write_tree(tmp_path / "data", machine={SETPOINT: {"value": 10.0}})

        with pytest.raises(FileNotFoundError) as excinfo:
            build_variable_catalog(paths, manifest, {})

        assert str(paths.channel_limits) in str(excinfo.value)


class TestLimitBands:
    """Which entries of a limits file become bands."""

    def _bands(self, tmp_path: Path, limits: dict, setpoints=frozenset({SETPOINT})):
        path = tmp_path / "channel_limits.json"
        path.write_text(json.dumps(limits))
        return _load_limit_bands(path, setpoints=setpoints)

    def test_unknown_entry_keys_are_ignored(self, tmp_path):
        """An emitted file stamps each entry it owns; the stamp is metadata
        to this reader, so a stamped file loads unchanged."""
        bands = self._bands(
            tmp_path,
            {
                SETPOINT: {
                    "min_value": 0.0,
                    "max_value": 20.0,
                    "_provenance": "exported 2026-09-17",
                }
            },
        )

        assert bands == {SETPOINT: (0.0, 20.0)}

    def test_metadata_keys_are_not_channels(self, tmp_path):
        bands = self._bands(
            tmp_path,
            {
                "_comment": "underscore keys are metadata",
                "_version": "1.0",
                SETPOINT: {"min_value": 0.0, "max_value": 20.0},
            },
        )

        assert bands == {SETPOINT: (0.0, 20.0)}

    def test_defaults_are_merged_under_each_entry(self, tmp_path):
        """``defaults`` decides an entry that says nothing, and loses to one
        that does."""
        other = "R1:PWR:FAM_A:02:CUR:SP"
        bands = self._bands(
            tmp_path,
            {
                "defaults": {"writable": False},
                SETPOINT: {"min_value": 0.0, "max_value": 20.0},
                other: {"min_value": 0.0, "max_value": 20.0, "writable": True},
            },
            setpoints=frozenset({SETPOINT, other}),
        )

        assert bands == {other: (0.0, 20.0)}

    def test_entries_outside_the_setpoint_set_carry_no_band(self, tmp_path):
        """The writable half is named by the manifest, not by the address
        text -- a limits file holds an entry per address, readings included."""
        bands = self._bands(
            tmp_path,
            {
                SETPOINT: {"min_value": 0.0, "max_value": 20.0},
                READING: {"min_value": -1.0, "max_value": 1.0},
            },
        )

        assert bands == {SETPOINT: (0.0, 20.0)}

    def test_entry_missing_a_bound_carries_no_band(self, tmp_path):
        bands = self._bands(tmp_path, {SETPOINT: {"min_value": 0.0}})

        assert bands == {}

    def test_bounds_are_read_as_floats(self, tmp_path):
        bands = self._bands(tmp_path, {SETPOINT: {"min_value": 0, "max_value": 20}})

        assert bands == {SETPOINT: (0.0, 20.0)}
        assert all(isinstance(bound, float) for bound in bands[SETPOINT])


class TestActionVariables:
    """The seam that binds a variable to the lattice it drives."""

    def test_an_addresss_factory_builds_its_variable(self, paths, manifest):
        """A bound variable is the catalog's own fields handed to the
        factory for that address -- one derivation, not two."""
        built: dict[str, Any] = {}

        def record(channel, **scalar_kwargs):
            built["channel"] = channel
            built["kwargs"] = scalar_kwargs
            return ScalarVariable(**scalar_kwargs)

        catalog = build_variable_catalog(paths, manifest, {SETPOINT: record})

        assert built["channel"]["address"] == SETPOINT
        assert built["kwargs"]["value_range"] == (0.0, 20.0)
        assert built["kwargs"]["default_value"] == 10.0
        assert catalog[SETPOINT].name == SETPOINT

    def test_an_address_without_a_factory_is_declared_only(self, paths, manifest):
        """Every address the mapping does not name is a plain variable: the
        declared catalog, bound to nothing."""
        catalog = build_variable_catalog(
            paths, manifest, {SETPOINT: lambda channel, **kwargs: ScalarVariable(**kwargs)}
        )

        assert type(catalog[READING]) is ScalarVariable


class TestServingPath:
    """The catalog reaches no tree but the one it is given."""

    def test_no_packaged_source_is_named(self):
        source = Path(catalog_module.__file__).read_text()

        for forbidden in ("PACKAGE_PATHS", "build_manifest", "osprey.templates"):
            assert forbidden not in source
