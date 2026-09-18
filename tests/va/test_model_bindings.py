"""Every variable the model drives is built from the bindings document.

``build_action_variables`` is the whole of the facility knowledge the model
layer holds, and after this rewrite it holds none of its own: what element a
channel writes, which attribute and component it lands on, how many slices
share it and whether the value moves with the beam rigidity are all read off
``va_bindings.json``. The tests below pin exactly that -- that each kind
becomes the class that implements it, that the names and axes are the
document's rather than anything derived from a family or a subfield, and that
a binding is never quietly dropped.

They exercise the seam the catalog calls, so the factories are called the way
:func:`~osprey.services.virtual_accelerator.model.catalog.build_variable_catalog`
calls them -- ``factory(channel, **scalar_kwargs)`` -- with one test driving
the real catalog end to end over a served tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from lume.variables import ScalarVariable
from pydantic import ValidationError

from osprey.services.virtual_accelerator.bindings import Linear, Table, parse_bindings
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_STATIC_NOISY,
    READBACK_SUBFIELD,
    SETPOINT_SUBFIELD,
)
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths
from osprey.services.virtual_accelerator.model.bindings import (
    build_action_variables,
    couple_energy_knob,
)
from osprey.services.virtual_accelerator.model.catalog import build_variable_catalog
from osprey.services.virtual_accelerator.model.variables import (
    EnergyVariable,
    KickVariable,
    MonitorVariable,
    RFVariable,
    StrengthVariable,
)

SHA = "b" * 64

# The addresses the fixture document binds. They carry no facility's
# vocabulary, and -- deliberately -- no relationship to the element names
# below: the bindings document is the only thing that pairs the two.
QUAD_SP = "R1:PWR:QUAD_A:07:CUR:SP"
CORR_SP = "R1:PWR:CORR_A:03:CUR:SP"
CAVITY_SP = "R1:RF:CAV_A:01:FREQ:SP"
BPM_Y = "R1:DIA:MON_A:12:POS:Y"
BEND_SP = "R1:PWR:BEND_A:01:CUR:SP"
UNBOUND_SP = "R1:VAC:GAU_A:01:PRES:RB"


def _linear(gain: float = 2.0, offset: float = 0.5) -> dict:
    return {"kind": "linear", "gain": gain, "offset": offset}


def _table(grid: list | None = None, values: list | None = None) -> dict:
    return {
        "kind": "table",
        "grid": [0.0, 1.0, 2.0] if grid is None else grid,
        "values": [0.0, 1.5, 3.0] if values is None else values,
    }


def _strength(**overrides: Any) -> dict:
    body = {
        "kind": "strength",
        "family": "quad_a",
        "setpoint_address": QUAD_SP,
        "readback_address": "R1:PWR:QUAD_A:07:CUR:RB",
        "readback": "inverse",
        # Nothing like "QUAD_A07": the deck's own name for the element, which
        # is the only reason the document has to carry it.
        "element": "qf_sector3_a",
        "attribute": "PolynomB",
        "index": 1,
        "slices": [
            {"element": "qf_sector3_a", "weight": 1.0},
            {"element": "qf_sector3_b", "weight": 1.0},
        ],
        "owner": "quad_a",
        "calibration": _linear(),
        "monitor_inverse": _linear(gain=0.5, offset=-0.25),
        "nominal": 12.5,
        "energy_scaling": "brho",
        "energy_table": None,
    }
    body.update(overrides)
    return body


def _kick(**overrides: Any) -> dict:
    body = _strength(
        kind="kick",
        family="corr_a",
        setpoint_address=CORR_SP,
        readback_address="R1:PWR:CORR_A:03:CUR:RB",
        readback="identity",
        element="ch_sector1",
        attribute="KickAngle",
        index=0,
        slices=[
            {"element": "ch_sector1", "weight": 0.5},
            {"element": "ch_sector1_tail", "weight": 0.5},
        ],
        owner="corr_a",
        monitor_inverse=None,
        nominal=0.0,
    )
    body.update(overrides)
    return body


def _rf(**overrides: Any) -> dict:
    body = _strength(
        kind="rf",
        family="cav_a",
        setpoint_address=CAVITY_SP,
        readback_address="R1:RF:CAV_A:01:FREQ:RB",
        readback="inverse",
        element="rfcav_1",
        attribute="Frequency",
        index=None,
        slices=[
            {"element": "rfcav_1", "weight": 1.0},
            {"element": "rfcav_2", "weight": 1.0},
        ],
        owner="cav_a",
        calibration=_linear(gain=1.0, offset=0.0),
        monitor_inverse=_linear(gain=1.0, offset=0.0),
        nominal=499.64e6,
        energy_scaling="none",
    )
    body.update(overrides)
    return body


def _monitor(**overrides: Any) -> dict:
    body = _strength(
        kind="monitor",
        family="mon_a",
        setpoint_address=BPM_Y,
        readback_address=None,
        readback="inverse",
        element="bpm_sector9",
        # The vertical axis behind an address whose subfield the channel
        # spells differently -- see the axis test.
        attribute="y",
        index=None,
        slices=[{"element": "bpm_sector9", "weight": 1.0}],
        owner="mon_a",
        calibration=_linear(gain=1.0e-3, offset=0.0),
        monitor_inverse=_linear(gain=1.0e3, offset=0.0),
        nominal=None,
        energy_scaling="none",
    )
    body.update(overrides)
    return body


def _energy(**overrides: Any) -> dict:
    body = _strength(
        kind="energy",
        family="bend_a",
        setpoint_address=BEND_SP,
        readback_address=None,
        readback="same_as_setpoint",
        element=None,
        attribute=None,
        index=None,
        slices=[],
        owner=None,
        calibration=None,
        monitor_inverse=None,
        nominal=1.0,
        energy_scaling="none",
        energy_table=_table(),
    )
    body.update(overrides)
    return body


def _document(*bindings: dict, **overrides: Any) -> Any:
    """Parse a bindings document, so every fixture obeys the real schema."""
    body = {
        "system": "R1",
        "energy_gev": 2.5,
        "lattice_sha256": SHA,
        "bindings": list(bindings) or [_strength(), _kick(), _rf(), _monitor(), _energy()],
    }
    body.update(overrides)
    return parse_bindings(body)


def _channel(address: str, *, subfield: str = SETPOINT_SUBFIELD, **overrides: Any) -> dict:
    """One manifest channel, as the catalog hands it to a factory."""
    channel = {
        "address": address,
        "ring": "R1",
        "system": "PWR",
        "family": "FAM_A",
        "device": "99",
        "field": "CUR",
        "subfield": subfield,
        "partition": PARTITION_PYAT_COUPLED,
        "record_type": "ai",
        "noise": False,
    }
    channel.update(overrides)
    return channel


def _build(
    factories: dict,
    address: str,
    *,
    channel: dict | None = None,
    **overrides: Any,
) -> Any:
    """Call one factory exactly as ``build_variable_catalog`` calls it."""
    scalar_kwargs = {
        "name": address,
        "read_only": False,
        "default_validation_config": "none",
        "default_value": 1.0,
        "value_range": None,
        "unit": "A",
    }
    scalar_kwargs.update(overrides)
    return factories[address](channel or _channel(address), **scalar_kwargs)


@pytest.fixture
def factories() -> dict:
    """The factory mapping for a document holding one of every kind."""
    return build_action_variables(_document())


class TestTheFactoryMapping:
    """One factory per binding, keyed by the address the binding claims."""

    def test_every_binding_gets_a_factory_and_none_is_dropped(self, factories: dict) -> None:
        """A binding the model cannot build is a boot failure, not a gap.

        The mapping is keyed by address, so a kind silently skipped would show
        up as a missing key here and, downstream, as a plain unbound
        ``ScalarVariable`` in the catalog -- a channel that accepts writes and
        moves nothing.
        """
        assert set(factories) == {QUAD_SP, CORR_SP, CAVITY_SP, BPM_Y, BEND_SP}

    def test_a_monitor_is_keyed_by_the_address_it_publishes_on(self, factories: dict) -> None:
        """A read-only binding carries its address in ``setpoint_address``.

        It has no setpoint at all: the schema puts the monitor's own address
        there and leaves ``readback_address`` null, so one key rule covers
        both halves of the document.
        """
        document = _document(_monitor())
        assert set(build_action_variables(document)) == {BPM_Y}

    def test_the_factories_are_in_document_order(self) -> None:
        """Order matters: it is the order the energy knob adopts them in."""
        document = _document(_monitor(), _energy(), _strength())
        assert list(build_action_variables(document)) == [BPM_Y, BEND_SP, QUAD_SP]


class TestWhatEachKindBuilds:
    """The class per kind, and the fields it is handed from the binding."""

    def test_a_strength_replicates_its_setpoint_over_every_slice(self, factories: dict) -> None:
        variable = _build(factories, QUAD_SP)
        assert isinstance(variable, StrengthVariable)
        assert [(b.element_name, b.attribute, b.index, b.weight) for b in variable.bindings] == [
            ("qf_sector3_a", "PolynomB", 1, 1.0),
            ("qf_sector3_b", "PolynomB", 1, 1.0),
        ]
        assert variable.calibration == Linear(gain=2.0, offset=0.5)
        assert variable.monitor_inverse == Linear(gain=0.5, offset=-0.25)
        assert variable.energy_scaling == "brho"
        assert variable.deck_energy_gev == 2.5

    def test_a_kick_is_shared_over_the_slices_it_is_split_across(self, factories: dict) -> None:
        variable = _build(factories, CORR_SP)
        assert isinstance(variable, KickVariable)
        assert [(b.element_name, b.index, b.weight) for b in variable.bindings] == [
            ("ch_sector1", 0, 0.5),
            ("ch_sector1_tail", 0, 0.5),
        ]
        assert variable.bindings[0].attribute == "KickAngle"

    def test_an_identity_readback_carries_no_inverse(self, factories: dict) -> None:
        """``monitor_inverse=None`` is what makes the readback an echo.

        Dropping the keyword would default it to ``None`` too, which is why
        the fixture's kick uses ``identity`` and its strength does not: the
        two have to be distinguishable from the constructed variable.
        """
        assert _build(factories, CORR_SP).monitor_inverse is None
        assert _build(factories, QUAD_SP).monitor_inverse is not None

    def test_the_rf_frequency_reaches_every_cavity_in_full(self, factories: dict) -> None:
        variable = _build(factories, CAVITY_SP)
        assert isinstance(variable, RFVariable)
        assert [(b.element_name, b.attribute, b.index, b.weight) for b in variable.bindings] == [
            ("rfcav_1", "Frequency", None, 1.0),
            ("rfcav_2", "Frequency", None, 1.0),
        ]

    def test_a_monitor_reads_one_axis_of_one_element(self, factories: dict) -> None:
        variable = _build(factories, BPM_Y, read_only=True, unit="mm", default_value=0.0)
        assert isinstance(variable, MonitorVariable)
        assert variable.element_name == "bpm_sector9"
        assert variable.axis == "y"
        assert variable.monitor_inverse == Linear(gain=1.0e3, offset=0.0)
        assert variable.read_only is True

    def test_the_energy_knob_is_driven_by_its_own_table(self, factories: dict) -> None:
        variable = _build(factories, BEND_SP)
        assert isinstance(variable, EnergyVariable)
        assert variable.energy_table == Table(grid=(0.0, 1.0, 2.0), values=(0.0, 1.5, 3.0))
        assert variable.nominal == 1.0
        assert variable.deck_energy_gev == 2.5


class TestTheFacilityFactsComeFromTheDocument:
    """No element name, axis or attribute is derived from an address again."""

    def test_the_bound_element_is_the_documents_name(self, factories: dict) -> None:
        """Not ``f'{family}{device}'``: that concatenation is gone.

        The channel handed in says family ``FAM_A`` device ``99``, and the
        binding says ``qf_sector3_a``. Only the document knows which element
        a facility's channel drives, so only the document may say.
        """
        variable = _build(factories, QUAD_SP, channel=_channel(QUAD_SP))
        assert {b.element_name for b in variable.bindings} == {
            "qf_sector3_a",
            "qf_sector3_b",
        }

    def test_the_axis_is_the_bindings_attribute_not_the_address_subfield(
        self, factories: dict
    ) -> None:
        """The subfield-to-axis table is gone with the concatenation.

        A facility spelling its vertical subfield ``VERT`` -- or ``X`` on a
        channel the document binds to ``y`` -- used to read the wrong
        coordinate or none at all.
        """
        variable = _build(
            factories,
            BPM_Y,
            channel=_channel(BPM_Y, subfield="VERT"),
            read_only=True,
            default_value=0.0,
        )
        assert variable.axis == "y"

    def test_the_written_component_is_the_bindings_index(self) -> None:
        """A sextupole's ``PolynomB[2]`` is a document fact, not a family one."""
        document = _document(_strength(index=2))
        variable = _build(build_action_variables(document), QUAD_SP)
        assert [b.index for b in variable.bindings] == [2, 2]


class TestScalarFieldsStayTheCatalogs:
    """The binding adds a binding; it never restates the catalog's fields."""

    def test_the_catalogs_scalar_kwargs_reach_the_variable(self, factories: dict) -> None:
        variable = _build(
            factories,
            QUAD_SP,
            default_value=8.0,
            value_range=(0.0, 20.0),
            unit="A",
        )
        assert variable.name == QUAD_SP
        assert variable.default_value == 8.0
        assert variable.value_range == (0.0, 20.0)
        assert variable.unit == "A"
        assert variable.read_only is False

    def test_a_nominal_outside_its_band_is_still_a_refusal(self, factories: dict) -> None:
        """Construction-time band validation is the catalog's, and survives.

        The factory adds fields to the same ``ScalarVariable`` construction,
        so the band check the catalog relies on is not something a bound
        variable can slip past.
        """
        with pytest.raises(ValidationError):
            _build(factories, QUAD_SP, default_value=99.0, value_range=(0.0, 20.0))


class TestARefusalIsNeverASkip:
    """A binding the model cannot build stops the boot."""

    def test_a_slice_weight_the_kind_forbids_is_a_refusal(self) -> None:
        """A strength shared out instead of replicated: every slice weighs 1.

        The rule lives on the variable class, and the factory hands the
        weights straight to it rather than repairing or dropping them.
        """
        document = _document(
            _strength(
                slices=[
                    {"element": "qf_sector3_a", "weight": 0.5},
                    {"element": "qf_sector3_b", "weight": 0.5},
                ]
            )
        )
        with pytest.raises(ValidationError, match="each slice weighs 1.0"):
            _build(build_action_variables(document), QUAD_SP)

    def test_a_kick_that_does_not_share_evenly_is_a_refusal(self) -> None:
        document = _document(
            _kick(
                slices=[
                    {"element": "ch_sector1", "weight": 1.0},
                    {"element": "ch_sector1_tail", "weight": 1.0},
                ]
            )
        )
        with pytest.raises(ValidationError, match="each slice weighs 0.5"):
            _build(build_action_variables(document), CORR_SP)


class TestThroughTheCatalog:
    """The mapping is what the catalog consumes, so build one."""

    @pytest.fixture
    def manifest(self) -> list[dict]:
        return [
            _channel(QUAD_SP),
            _channel(QUAD_SP.replace(":SP", ":RB"), subfield=READBACK_SUBFIELD),
            _channel(BPM_Y, subfield="Y", system="DIA", family="MON_A", field="POS"),
            _channel(BEND_SP),
            _channel(
                UNBOUND_SP,
                subfield=READBACK_SUBFIELD,
                partition=PARTITION_STATIC_NOISY,
            ),
        ]

    @pytest.fixture
    def paths(self, tmp_path: Path) -> ManifestPaths:
        root = tmp_path / "data"
        (root / "simulation").mkdir(parents=True)
        (root / "simulation" / "machine.json").write_text(
            json.dumps(
                {
                    "name": "fixture",
                    "channels": {
                        QUAD_SP: {"value": 12.5, "units": "A"},
                        BPM_Y: {"value": 0.0, "units": "mm"},
                        BEND_SP: {"value": 1.0, "units": "A"},
                    },
                }
            )
        )
        (root / "channel_limits.json").write_text(
            json.dumps(
                {
                    "_version": "1.0",
                    "defaults": {"writable": True},
                    QUAD_SP: {"min_value": 0.0, "max_value": 20.0},
                    BEND_SP: {"min_value": 0.0, "max_value": 2.0},
                }
            )
        )
        return ManifestPaths(data_root=root)

    def test_the_catalog_binds_every_address_the_document_claims(
        self, paths: ManifestPaths, manifest: list[dict]
    ) -> None:
        catalog = build_variable_catalog(paths, manifest, build_action_variables(_document()))
        assert isinstance(catalog[QUAD_SP], StrengthVariable)
        assert isinstance(catalog[BPM_Y], MonitorVariable)
        assert isinstance(catalog[BEND_SP], EnergyVariable)
        # The setpoint echo is the serving layer's, never a model variable.
        assert QUAD_SP.replace(":SP", ":RB") not in catalog

    def test_a_coupled_address_with_no_binding_stays_unbound(
        self, paths: ManifestPaths, manifest: list[dict]
    ) -> None:
        """The catalog's own fallback, unchanged: a plain declared variable.

        Which is exactly why a dropped binding would be invisible without
        the count this module's other tests pin.
        """
        catalog = build_variable_catalog(
            paths, manifest, build_action_variables(_document(_monitor(), _energy()))
        )
        assert type(catalog[QUAD_SP]) is ScalarVariable


class TestTheEnergyKnobAdoptsWhatItRescales:
    """``couple`` is not optional: without it the knob moves nothing but eV."""

    @pytest.fixture
    def catalog(self, factories: dict) -> dict:
        return {
            QUAD_SP: _build(factories, QUAD_SP),
            CORR_SP: _build(factories, CORR_SP),
            CAVITY_SP: _build(factories, CAVITY_SP),
            BPM_Y: _build(factories, BPM_Y, read_only=True, default_value=0.0),
            BEND_SP: _build(factories, BEND_SP),
        }

    def test_every_rigidity_scaled_writable_is_adopted(self, catalog: dict) -> None:
        """The brho ones and only those: the rf frequency does not scale."""
        assert couple_energy_knob(catalog) == (QUAD_SP, CORR_SP)

    def test_the_knob_snapshots_what_the_rescale_touches(self, catalog: dict) -> None:
        """Adoption is what puts the rescaled fields under the rollback.

        The knob reports its adopted variables' own snapshot targets, so a
        failed batch restores the strengths it rescaled as well as the
        energy. Asserting through the variables rather than a simulator keeps
        this a test of the coupling.
        """
        couple_energy_knob(catalog)
        knob = catalog[BEND_SP]
        assert [variable.name for variable in knob._scaled] == [QUAD_SP, CORR_SP]

    def test_a_document_with_no_energy_knob_couples_nothing(self, factories: dict) -> None:
        """A facility whose bends are not a control channel is not a failure."""
        assert couple_energy_knob({QUAD_SP: _build(factories, QUAD_SP)}) == ()


class TestThePackageReExportsTheSeam:
    def test_the_model_package_offers_the_bindings_seam(self) -> None:
        """Both halves: the mapping, and the coupling its caller must run."""
        from osprey.services.virtual_accelerator import model

        assert model.build_action_variables is build_action_variables
        assert model.couple_energy_knob is couple_energy_knob
