"""Pytest configuration for the ``osprey-connectors`` suite.

This suite runs where only the connectors distribution is installed, so it
cannot lean on the framework's ``tests/conftest.py``. Its guards are the
framework-free ones that conftest shares, imported by name from
``tests/_connectors_harness.py`` (see that module for why by name and not as a
plugin). A guard added there for both suites belongs in this list too.

It sits above the ``osprey_connectors_tests`` package rather than inside it: the
session-level hooks it carries (``pytest_xdist_make_scheduler`` above all) are
read only from the conftests pytest loads before collection, which are the ones
beside the paths it was given.
"""

from tests._connectors_harness import (  # noqa: F401 - fixtures and hooks, used by name
    FileOrGroupScheduling,
    _clear_dotenv_timezone,
    _no_host_timezone_leak,
    agent_data_never_the_checkout,
    control_context_root,
    ioc_prefix,
    no_agent_data_in_the_repo,
    no_audit_ledger_in_the_repo,
    pytest_collection_finish,
    pytest_configure,
    pytest_runtest_logfinish,
    pytest_runtest_logstart,
    pytest_runtest_setup,
    pytest_runtest_teardown,
    pytest_unconfigure,
    pytest_xdist_make_scheduler,
    reset_connector_config_cache,
    reset_second_real_block_warning,
    restore_cwd,
    restore_environ,
    restore_environ_per_module,
    restore_root_logging,
    session_posture_leak_guard,
    write_control_context,
    write_server_report,
)
