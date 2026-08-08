"""Module-identity guarantees for the osprey-connectors extraction.

Main osprey re-exports the moved modules under their historical paths via
sys.modules-aliasing shims; these tests pin the contract that both names
resolve to the SAME module object (patching/isinstance safe).
"""


def test_osprey_connectors_is_installed():
    import osprey_connectors

    assert osprey_connectors.__version__ == "0.1.0"


def test_errors_shim_preserves_module_identity():
    import osprey.errors
    import osprey_connectors.errors

    assert osprey.errors is osprey_connectors.errors


def test_utils_shims_preserve_module_identity():
    import osprey.utils.config
    import osprey.utils.logger
    import osprey.utils.relative_time
    import osprey_connectors.config
    import osprey_connectors.logger
    import osprey_connectors.relative_time

    assert osprey.utils.config is osprey_connectors.config
    assert osprey.utils.logger is osprey_connectors.logger
    assert osprey.utils.relative_time is osprey_connectors.relative_time


def test_patching_through_shim_reaches_real_module(monkeypatch):
    import osprey_connectors.config as real_config

    monkeypatch.setattr("osprey.utils.config.get_config_value", lambda *a, **k: "patched")
    assert real_config.get_config_value("anything") == "patched"
