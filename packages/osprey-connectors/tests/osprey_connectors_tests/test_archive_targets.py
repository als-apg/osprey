"""Which control targets a deployment's one ``archiver`` block serves.

The ``archive`` row of :func:`service_targets` is the single answer every reader
of the archive shares: a store is offered to a session only on a target whose
history it holds, and a target with no archive is told so rather than handed
another machine's. These tests pin one case per class — the facility's archive
(the appliance, MYA, DOOCS or an unknown type), which the live machine and the
stand-in that rehearses it share; the mongodb store, which is the virtual
accelerator's on every deployment that built one, whatever machine its sessions
start on; and every shape that names no machine at all — in both spellings of
the settings block.

The table replaced ``archive_targets`` and ``archive_withheld``. Both are kept
below, verbatim, as the oracle: every case reads the table through
:func:`archive_targets` and :func:`archive_withheld`, which assert
``archive_withheld(c, t) == (table.archive[t] is None)`` on every target and
``archive_targets(c) == table.history_targets()`` before they answer.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from types import MappingProxyType

import pytest

from osprey_connectors.archiver_selection import resolve_archiver_settings, resolve_archiver_type
from osprey_connectors.connector_types import (
    CONTROL_TARGETS,
    DOOCS_ARCHIVER,
    EPICS_ARCHIVER,
    MOCK_ARCHIVER,
    MONGODB_ARCHIVER,
    MYA_ARCHIVER,
    TARGET_LIVE,
    TARGET_STANDIN,
    TARGET_VA,
    ControlTarget,
)
from osprey_connectors.deployment.simulator import virtual_accelerator_port
from osprey_connectors.service_targets import (
    _NO_TARGETS,
    _STORE_ARCHIVERS,
    _STORE_TARGETS,
    APPLIANCE_TARGETS,
    service_targets,
)

# -- the old answers, verbatim, as the oracle ----------------------------------

#: The top-level config section selecting and configuring the archiver.
_OLD_ARCHIVER_SECTION_KEY = "archiver"

#: The archiver types that are the deployment's own store. Every other type
#: except the mock is the facility's archive.
_OLD_STORE_ARCHIVERS: frozenset[str] = frozenset({MONGODB_ARCHIVER})

#: The setting under which a facility archive names where it is, per type.
#: ``None``: the connector finds its server without one (MYA through the client
#: library's configured ``myquery_server``, DOOCS through the facility's name
#: service), so its block always names an address. A type not listed names it
#: under :data:`_OLD_DEFAULT_ADDRESS_SETTING`.
_OLD_ADDRESS_SETTINGS: Mapping[str, str | None] = MappingProxyType(
    {EPICS_ARCHIVER: "url", MYA_ARCHIVER: None, DOOCS_ARCHIVER: None}
)

#: The address setting of a facility archive :data:`_OLD_ADDRESS_SETTINGS` does not list.
_OLD_DEFAULT_ADDRESS_SETTING = "url"

#: The targets the facility's archive serves: the real machine and the
#: stand-in that rehearses it.
_OLD_APPLIANCE_TARGETS: frozenset[ControlTarget] = frozenset({TARGET_LIVE, TARGET_STANDIN})

#: The targets the deployment's own store serves: the virtual accelerator.
_OLD_STORE_TARGETS: frozenset[ControlTarget] = frozenset({TARGET_VA})

#: No archive for any target.
_OLD_NO_TARGETS: frozenset[ControlTarget] = frozenset()


def _old_archive_targets(config: object) -> frozenset[ControlTarget]:
    """The control targets whose history this deployment's ``archiver`` block holds.

    Args:
        config: The full config mapping, as loaded from ``config.yml`` — not the
            ``archiver:`` section, because the answer for the mongodb store
            turns on whether the deployment built a simulator, which the
            section alone does not carry. Anything that is not a mapping states
            no archiver, which is the mock's.

    Returns:
        The targets per the module table — :data:`_OLD_APPLIANCE_TARGETS`,
        :data:`_OLD_STORE_TARGETS` or :data:`_OLD_NO_TARGETS`. A target absent from the
        set has no archive: a caller holding it has nothing to offer, never a
        default to fall back on.
    """
    if not isinstance(config, Mapping):
        # No section is the mock's, which has no targets.
        return _OLD_NO_TARGETS
    section = config.get(_OLD_ARCHIVER_SECTION_KEY)
    archiver_type = resolve_archiver_type(section)
    if archiver_type == MOCK_ARCHIVER:
        return _OLD_NO_TARGETS

    try:
        settings = resolve_archiver_settings(section)
    except ValueError:
        # The factory refuses this block with the same error, so no connector
        # is ever built from it and there is no store to attribute.
        return _OLD_NO_TARGETS

    if archiver_type in _OLD_STORE_ARCHIVERS:
        return (
            _OLD_STORE_TARGETS if virtual_accelerator_port(config) is not None else _OLD_NO_TARGETS
        )
    return (
        _OLD_APPLIANCE_TARGETS
        if _old_names_an_address(archiver_type, settings)
        else _OLD_NO_TARGETS
    )


def _old_archive_withheld(config: object, target: str) -> bool:
    """Whether a session on *target* must be told it has no archive.

    The agent's reader of the one ``archiver:`` block asks this before it
    opens the connector, and the answer is the table's: an attributed block
    whose targets do not include *target* is another machine's past, and is
    withheld. The mock is not withheld here; whether it is an honest pairing is
    the archiver pairing's question, asked at build and at the target switch.

    Args:
        config: The full config mapping, as loaded from ``config.yml``; anything
            that is not a mapping states the mock.
        target: The control target the session is on.

    Returns:
        ``True`` when the block is not the mock and does not serve *target*.
    """
    section = config.get(_OLD_ARCHIVER_SECTION_KEY) if isinstance(config, Mapping) else None
    if resolve_archiver_type(section) == MOCK_ARCHIVER:
        return False
    return target not in _old_archive_targets(config)


def _old_names_an_address(archiver_type: str, settings: Mapping[str, object]) -> bool:
    """Whether a facility archive's *settings* state where it is.

    A non-blank string under the type's :data:`_OLD_ADDRESS_SETTINGS` entry, or
    always for a type whose connector finds its server without one. The
    connector's own reader decides whether the address is well formed; this
    only establishes that the deployment said one.
    """
    key = _OLD_ADDRESS_SETTINGS.get(archiver_type, _OLD_DEFAULT_ADDRESS_SETTING)
    if key is None:
        return True
    address = settings.get(key)
    return isinstance(address, str) and bool(address.strip())


# -- the table's answers, each pinned to the oracle ----------------------------


def archive_targets(config: object) -> frozenset[ControlTarget]:
    """The table's history targets, after both equivalences are asserted for *config*."""
    table = service_targets(config)
    for target in CONTROL_TARGETS:
        assert (table.archive[target] is None) == _old_archive_withheld(config, target), target
    held = table.history_targets()
    assert held == _old_archive_targets(config)
    return held


def archive_withheld(config: object, target: ControlTarget) -> bool:
    """Whether the table holds no archive for *target*, pinned to the oracle."""
    archive_targets(config)
    return service_targets(config).archive[target] is None


STANDIN_PORT = 5074
VA_PORT = 5064
APPLIANCE_URL = "https://archiver.example.org:8443"
MONGO_SETTINGS = {"uri": "mongodb://localhost:27017", "database": "archive"}


def va_config(archiver: object, *, baseline: str = "virtual_accelerator") -> dict:
    """A deployment that built a simulator and starts its sessions on *baseline*."""
    return {
        "control_system": {"type": baseline},
        "services": {"virtual_accelerator": {"port": VA_PORT}},
        "archiver": archiver,
    }


def standin_config(
    archiver: object, *, recorded: bool = True, baseline: str = "virtual_accelerator"
) -> dict:
    """A deployment that built a simulator and a stand-in beside it."""
    config = va_config(archiver, baseline=baseline)
    config["services"]["live_standin"] = {"port": STANDIN_PORT}
    if recorded:
        config["deployed_services"] = ["virtual_accelerator", "mongodb", "archiver_recorder"]
    return config


def facility_config(archiver: object, *, control_type: str = "epics") -> dict:
    """A deployment that built no simulator, on the facility's machine or *control_type*."""
    return {"control_system": {"type": control_type}, "archiver": archiver}


def block(archiver_type: str, settings: dict, *, legacy: bool = False) -> dict:
    """An ``archiver:`` section in the current spelling, or the own-name one."""
    return {"type": archiver_type, (archiver_type if legacy else "settings"): settings}


#: Both spellings of the settings block, for the rows that read one.
SPELLINGS = pytest.mark.parametrize("legacy", [False, True], ids=["settings", "own-name"])


class TestTheTargetSets:
    """The three answers, spelled once and never overlapping."""

    def test_the_appliance_serves_the_real_machine_and_its_rehearsal(self) -> None:
        assert APPLIANCE_TARGETS == {TARGET_LIVE, TARGET_STANDIN}

    def test_the_store_serves_the_virtual_accelerator(self) -> None:
        assert _STORE_TARGETS == {TARGET_VA}

    def test_no_target_is_served_by_both(self) -> None:
        assert not APPLIANCE_TARGETS & _STORE_TARGETS
        assert _NO_TARGETS == frozenset()


class TestTheAppliance:
    """``epics_archiver`` with a url is the facility's past: live and standin."""

    @SPELLINGS
    def test_on_the_facilitys_own_deployment(self, legacy: bool) -> None:
        archiver = block(EPICS_ARCHIVER, {"url": APPLIANCE_URL}, legacy=legacy)
        assert archive_targets(facility_config(archiver)) == APPLIANCE_TARGETS

    @SPELLINGS
    def test_on_any_deployment_whatever_it_stood_up(self, legacy: bool) -> None:
        """An appliance is never the simulation's; the control system does not matter."""
        archiver = block(EPICS_ARCHIVER, {"url": APPLIANCE_URL}, legacy=legacy)
        assert archive_targets(standin_config(archiver)) == APPLIANCE_TARGETS
        assert archive_targets(va_config(archiver)) == APPLIANCE_TARGETS
        assert archive_targets(facility_config(archiver, control_type="mock")) == (
            APPLIANCE_TARGETS
        )

    def test_the_stand_in_is_shown_the_real_machines_past(self) -> None:
        """The whole point of the pairing: a rehearsal sees what the machine saw."""
        archiver = block(EPICS_ARCHIVER, {"url": APPLIANCE_URL})
        targets = archive_targets(standin_config(archiver))
        assert TARGET_STANDIN in targets
        assert TARGET_LIVE in targets
        assert TARGET_VA not in targets

    @pytest.mark.parametrize("settings", [{}, {"url": ""}, {"url": "   "}, {"url": None}])
    def test_without_a_url_it_is_no_archive(self, settings: dict) -> None:
        """A block that cannot be dialled is not a deployment saying it has an archive."""
        assert archive_targets(facility_config(block(EPICS_ARCHIVER, settings))) == _NO_TARGETS

    def test_with_no_settings_block_at_all(self) -> None:
        assert archive_targets(facility_config({"type": EPICS_ARCHIVER})) == _NO_TARGETS

    def test_a_url_that_is_not_a_string_names_nothing(self) -> None:
        archiver = block(EPICS_ARCHIVER, {"url": ["https://archiver.example.org"]})
        assert archive_targets(facility_config(archiver)) == _NO_TARGETS


class TestTheFacilitysOtherArchives:
    """Every archiver that is neither the store nor the mock is the facility's."""

    @pytest.mark.parametrize("archiver_type", [MYA_ARCHIVER, DOOCS_ARCHIVER])
    @SPELLINGS
    def test_mya_and_doocs_serve_the_real_machine_and_its_rehearsal(
        self, archiver_type: str, legacy: bool
    ) -> None:
        """Their connectors find the server without a setting (the client
        library's ``myquery_server``, the name service), so the block names one."""
        archiver = block(archiver_type, {}, legacy=legacy)
        assert archive_targets(facility_config(archiver)) == APPLIANCE_TARGETS
        assert archive_targets(standin_config(archiver)) == APPLIANCE_TARGETS

    @pytest.mark.parametrize("archiver_type", [MYA_ARCHIVER, DOOCS_ARCHIVER])
    def test_beside_a_simulator_they_are_withheld_from_it(self, archiver_type: str) -> None:
        """A JLab or DESY deployment with a stand-in no longer shows the real
        machine's past on ``va``."""
        config = standin_config(block(archiver_type, {}))
        assert archive_withheld(config, TARGET_VA)
        assert not archive_withheld(config, TARGET_LIVE)
        assert not archive_withheld(config, TARGET_STANDIN)

    def test_mya_with_its_own_server_setting(self) -> None:
        archiver = block(MYA_ARCHIVER, {"myquery_server": "epicsweb.example.org"})
        assert archive_targets(va_config(archiver)) == APPLIANCE_TARGETS

    def test_an_unknown_type_naming_a_url(self) -> None:
        archiver = block("sql_archiver", {"url": APPLIANCE_URL})
        assert archive_targets(va_config(archiver)) == APPLIANCE_TARGETS

    def test_an_unknown_type_naming_no_url(self) -> None:
        assert archive_targets(va_config(block("sql_archiver", {}))) == _NO_TARGETS


class TestTheStore:
    """``mongodb_archiver`` is the virtual accelerator's, on every deployment that built one."""

    @SPELLINGS
    def test_on_a_virtual_accelerator(self, legacy: bool) -> None:
        archiver = block(MONGODB_ARCHIVER, MONGO_SETTINGS, legacy=legacy)
        assert archive_targets(va_config(archiver)) == _STORE_TARGETS

    @SPELLINGS
    def test_beside_a_stand_in_it_is_still_the_virtual_accelerators(self, legacy: bool) -> None:
        """The recorder samples the simulator whether or not a stand-in runs
        beside it, and the seed never carried the stand-in's offsets. So the
        store is ``va``'s, and the stand-in — which rehearses the real machine
        — gets nothing from it."""
        archiver = block(MONGODB_ARCHIVER, MONGO_SETTINGS, legacy=legacy)
        assert archive_targets(standin_config(archiver)) == _STORE_TARGETS
        assert archive_targets(standin_config(archiver, recorded=False)) == _STORE_TARGETS

    @pytest.mark.parametrize("baseline", ["live_standin", "epics", "mock"])
    def test_whatever_machine_the_sessions_start_on(self, baseline: str) -> None:
        """``control_system.type`` names the session baseline, not the store's
        owner: a deployment opening on its stand-in — or on the facility's
        machine — still built the simulator, still seeded its past and still
        records it, so a session that switches to ``va`` is shown that past."""
        archiver = block(MONGODB_ARCHIVER, MONGO_SETTINGS)
        assert archive_targets(standin_config(archiver, baseline=baseline)) == _STORE_TARGETS
        assert archive_targets(va_config(archiver, baseline=baseline)) == _STORE_TARGETS

    def test_the_recorder_listing_is_not_read(self) -> None:
        """Whose past the store holds follows from the simulator the deployment
        built, not from which services a render lists: a tier render lists
        none, and is told the simulator's port by the reach contract."""
        config = va_config(block(MONGODB_ARCHIVER, MONGO_SETTINGS))
        config["deployed_services"] = []
        assert archive_targets(config) == _STORE_TARGETS

    def test_with_no_settings_block_at_all(self) -> None:
        """The factory's own default settings: still the store, still ``va``."""
        assert archive_targets(va_config({"type": MONGODB_ARCHIVER})) == _STORE_TARGETS

    @pytest.mark.parametrize(
        "control_type", ["epics", "mock", "live_standin", "virtual_accelerator"]
    )
    def test_a_deployment_that_built_no_simulator_offers_it_to_nobody(
        self, control_type: str
    ) -> None:
        """No ``services.virtual_accelerator.port``, no simulator of its own —
        including a deployment whose ``va`` block dials a simulator some other
        host runs. Nothing this deployment seeded describes that machine, so the
        store is offered to no target."""
        archiver = block(MONGODB_ARCHIVER, MONGO_SETTINGS)
        assert archive_targets(facility_config(archiver, control_type=control_type)) == _NO_TARGETS

    @pytest.mark.parametrize("value", ["", None, True, {"port": 5064}])
    def test_a_services_block_naming_no_port_built_nothing(self, value: object) -> None:
        config = va_config(block(MONGODB_ARCHIVER, MONGO_SETTINGS))
        config["services"]["virtual_accelerator"]["port"] = value
        assert archive_targets(config) == _NO_TARGETS

    def test_with_no_services_section(self) -> None:
        assert archive_targets({"archiver": block(MONGODB_ARCHIVER, MONGO_SETTINGS)}) == _NO_TARGETS


class TestNoArchive:
    """Every shape that names no machine's past."""

    def test_the_mock(self) -> None:
        archiver = block(MOCK_ARCHIVER, {"url": APPLIANCE_URL})
        assert archive_targets(facility_config(archiver)) == _NO_TARGETS
        assert archive_targets(va_config(archiver)) == _NO_TARGETS

    @pytest.mark.parametrize("section", [None, {}, "mongodb_archiver", ["mongodb_archiver"]])
    def test_a_block_that_is_not_a_mapping_or_names_no_type(self, section: object) -> None:
        assert archive_targets(va_config(section)) == _NO_TARGETS

    def test_no_block_and_no_config(self) -> None:
        assert archive_targets({}) == _NO_TARGETS
        assert archive_targets({"control_system": {"type": "virtual_accelerator"}}) == _NO_TARGETS

    def test_both_settings_spellings_at_once(self) -> None:
        """The factory refuses this block, so no connector is built from it."""
        archiver = {
            "type": MONGODB_ARCHIVER,
            "settings": MONGO_SETTINGS,
            MONGODB_ARCHIVER: MONGO_SETTINGS,
        }
        assert archive_targets(va_config(archiver)) == _NO_TARGETS

    def test_a_settings_value_that_is_not_a_mapping(self) -> None:
        assert archive_targets(va_config({"type": EPICS_ARCHIVER, "settings": "url"})) == (
            _NO_TARGETS
        )


class TestWithheld:
    """What the agent's reader asks before opening the one ``archiver:`` block."""

    def test_the_store_is_the_one_class_the_table_names(self) -> None:
        assert _STORE_ARCHIVERS == {MONGODB_ARCHIVER}

    def test_the_store_is_withheld_from_the_real_machine_and_its_rehearsal(self) -> None:
        config = standin_config(block(MONGODB_ARCHIVER, MONGO_SETTINGS), baseline="live_standin")
        assert archive_withheld(config, TARGET_LIVE)
        assert archive_withheld(config, TARGET_STANDIN)
        assert not archive_withheld(config, TARGET_VA)

    def test_the_appliance_is_withheld_from_the_simulator(self) -> None:
        config = standin_config(block(EPICS_ARCHIVER, {"url": APPLIANCE_URL}))
        assert archive_withheld(config, TARGET_VA)
        assert not archive_withheld(config, TARGET_LIVE)
        assert not archive_withheld(config, TARGET_STANDIN)

    def test_an_attributed_block_that_serves_nobody_is_withheld_from_everybody(self) -> None:
        config = facility_config(block(MONGODB_ARCHIVER, MONGO_SETTINGS))
        assert all(archive_withheld(config, t) for t in (TARGET_LIVE, TARGET_STANDIN, TARGET_VA))

    def test_the_mock_is_the_pairings_question_not_this_ones(self) -> None:
        """The mock serves a mock deployment's sessions today and is refused
        beside a simulated machine by the archiver pairing; this table has no
        opinion on it and withholds nothing."""
        config = va_config(block(MOCK_ARCHIVER, {"url": APPLIANCE_URL}))
        assert not any(
            archive_withheld(config, t) for t in (TARGET_LIVE, TARGET_STANDIN, TARGET_VA)
        )

    def test_no_block_withholds_nothing(self) -> None:
        assert not archive_withheld({}, TARGET_VA)


class TestPurity:
    def test_the_config_is_not_modified(self) -> None:
        config = standin_config(block(EPICS_ARCHIVER, {"url": APPLIANCE_URL}))
        before = copy.deepcopy(config)
        archive_targets(config)
        assert config == before

    def test_the_answer_is_a_function_of_the_mapping_alone(self) -> None:
        config = va_config(block(MONGODB_ARCHIVER, MONGO_SETTINGS))
        assert archive_targets(config) == archive_targets(copy.deepcopy(config)) == _STORE_TARGETS


class TestTheArchiveRow:
    """The spec each target opens: the type, the settings and the rehearsal mark."""

    def test_the_appliance_spec_carries_the_block_and_marks_the_rehearsal(self) -> None:
        settings = {"url": APPLIANCE_URL}
        table = service_targets(facility_config(block(EPICS_ARCHIVER, settings)))
        live, standin = table.archive[TARGET_LIVE], table.archive[TARGET_STANDIN]
        assert live is not None and standin is not None
        assert (live.kind, live.type, live.target, live.rehearsal) == (
            "archive",
            EPICS_ARCHIVER,
            TARGET_LIVE,
            False,
        )
        assert dict(live.settings) == settings
        assert standin.rehearsal
        assert table.archive[TARGET_VA] is None

    def test_the_mock_is_opened_on_every_target_and_holds_no_history(self) -> None:
        table = service_targets(va_config(block(MOCK_ARCHIVER, {"sample_rate_hz": 2})))
        specs = [table.archive[target] for target in CONTROL_TARGETS]
        assert all(spec is not None and spec.type == MOCK_ARCHIVER for spec in specs)
        assert table.history_targets() == _NO_TARGETS

    def test_no_block_is_the_mock_with_no_settings(self) -> None:
        spec = service_targets({}).archive[TARGET_VA]
        assert spec is not None
        assert (spec.type, dict(spec.settings)) == (MOCK_ARCHIVER, {})

    def test_the_row_is_the_archive_field(self) -> None:
        table = service_targets(va_config(block(MONGODB_ARCHIVER, MONGO_SETTINGS)))
        assert table.row("archive") == table.archive

    def test_a_spec_never_aliases_the_config(self) -> None:
        config = facility_config(block(EPICS_ARCHIVER, {"url": APPLIANCE_URL}))
        spec = service_targets(config).archive[TARGET_LIVE]
        assert spec is not None
        config["archiver"]["settings"]["url"] = "https://elsewhere.example.org"
        assert spec.settings["url"] == APPLIANCE_URL
