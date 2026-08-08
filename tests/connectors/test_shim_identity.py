"""Module-identity guarantees for the osprey-connectors extraction.

Main osprey re-exports the moved modules under their historical paths via
sys.modules-aliasing shims; these tests pin the contract that both names
resolve to the SAME module object (patching/isinstance safe).
"""


def test_osprey_connectors_is_installed():
    import osprey_connectors

    assert osprey_connectors.__version__ == "0.1.0"


def test_errors_shim_preserves_module_identity():
    import osprey_connectors.errors

    import osprey.errors

    assert osprey.errors is osprey_connectors.errors


def test_utils_shims_preserve_module_identity():
    import osprey_connectors.config
    import osprey_connectors.logger
    import osprey_connectors.relative_time

    import osprey.utils.config
    import osprey.utils.logger
    import osprey.utils.relative_time

    assert osprey.utils.config is osprey_connectors.config
    assert osprey.utils.logger is osprey_connectors.logger
    assert osprey.utils.relative_time is osprey_connectors.relative_time


def test_patching_through_shim_reaches_real_module(monkeypatch):
    import osprey_connectors.config as real_config

    monkeypatch.setattr("osprey.utils.config.get_config_value", lambda *a, **k: "patched")
    assert real_config.get_config_value("anything") == "patched"


def test_simulation_core_shims_preserve_module_identity():
    import osprey_connectors.simulation.engine

    import osprey.simulation
    import osprey.simulation.engine

    assert osprey.simulation.engine is osprey_connectors.simulation.engine
    assert osprey.simulation.SimulationEngine is osprey_connectors.simulation.SimulationEngine


def test_connector_shims_preserve_module_identity():
    import osprey_connectors.archiver.base
    import osprey_connectors.control_system.epics_connector
    import osprey_connectors.factory

    import osprey.connectors.archiver.base
    import osprey.connectors.control_system.epics_connector
    import osprey.connectors.factory

    assert (
        osprey.connectors.control_system.epics_connector
        is osprey_connectors.control_system.epics_connector
    )
    assert osprey.connectors.archiver.base is osprey_connectors.archiver.base
    assert osprey.connectors.factory is osprey_connectors.factory


def test_exception_identity_across_namespaces():
    from osprey_connectors.errors import ChannelWriteBlockedError as new

    from osprey.errors import ChannelWriteBlockedError as old

    assert old is new
