"""Pytest configuration for the ``osprey-connectors`` suite.

The guards are the framework-free ones ``tests/conftest.py`` also runs,
imported by name from ``_harness._guards`` (see that module for why not as a
plugin); a guard added there belongs in both lists. This file sits above the
``osprey_connectors_tests`` package so its session hooks load before
collection and so pytest puts this directory on ``sys.path``.
"""

from osprey_connectors_tests._harness._guards import (  # noqa: F401 - fixtures and hooks, used by name
    _clear_dotenv_timezone,
    _no_host_timezone_leak,
    agent_data_never_the_checkout,
    control_context_root,
    ioc_prefix,
    long_lived_session_mark_per_test,
    no_agent_data_in_the_repo,
    no_spool_in_the_repo,
    pytest_collection_finish,
    pytest_configure,
    pytest_runtest_setup,
    pytest_runtest_teardown,
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
