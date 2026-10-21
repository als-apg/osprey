"""Pytest configuration for the ``osprey-connectors`` suite.

This suite runs where only the connectors distribution is installed, so it
cannot lean on the framework's ``tests/conftest.py``. Its guards are the
framework-free ones that conftest shares, imported by name from
``osprey_connectors_tests._harness._guards`` (see that module for why by name
and not as a plugin). A guard added there for both suites belongs in this list
too.

It sits above the ``osprey_connectors_tests`` package rather than inside it: the
session-level hooks it carries (``pytest_xdist_make_scheduler`` above all) are
read only from the conftests pytest loads before collection, which are the ones
beside the paths it was given. Being there also puts this directory on
``sys.path`` (pytest's ``prepend`` import mode does so for a conftest outside any
package), which is what makes ``osprey_connectors_tests`` importable here
without a ``pythonpath`` entry.

The dependency runs one way. The framework's tests may import
``osprey_connectors_tests`` and its ``_harness`` (the repository root's
``pyproject.toml`` puts this directory on their path for that), so a helper
both suites need lives here once. Nothing under this directory may import the
framework's ``tests`` package or ``osprey``: this suite is the one that has to
pass with neither installed, and
``osprey_connectors_tests/test_suite_is_self_contained.py`` holds that line.
"""

from osprey_connectors_tests._harness._guards import (  # noqa: F401 - fixtures and hooks, used by name
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
