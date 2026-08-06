"""Tests for MCP server lifecycle helpers (``osprey.mcp_server.startup``).

Covers the startup-timing context manager, config-builder priming
(present/absent/failing OSPREY_CONFIG), workspace singleton init, and the shared
``run_mcp_server`` entry point wiring. Logging-mechanism behavior itself lives in
``tests/utils/test_configure_logging.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from osprey.mcp_server import startup

# ---------------------------------------------------------------------------
# startup_timer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_startup_timer_emits_timing_line(capsys, monkeypatch):
    monkeypatch.setattr(startup, "_server_label", "workspace")
    with startup.startup_timer("phase_x"):
        pass
    err = capsys.readouterr().err
    assert "[STARTUP-TIMING] workspace | phase_x:" in err
    assert "ms" in err


@pytest.mark.unit
def test_startup_timer_emits_on_exception(capsys, monkeypatch):
    """The timing line is printed even when the wrapped block raises."""
    monkeypatch.setattr(startup, "_server_label", "svc")
    with pytest.raises(ValueError):
        with startup.startup_timer("boom"):
            raise ValueError("x")
    assert "[STARTUP-TIMING] svc | boom:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# prime_config_builder
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prime_config_builder_noop_without_env(monkeypatch):
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    with patch("osprey.utils.config.get_config_builder") as gcb:
        startup.prime_config_builder()
    gcb.assert_not_called()


@pytest.mark.unit
def test_prime_config_builder_primes_and_loads_categories(monkeypatch):
    monkeypatch.setenv("OSPREY_CONFIG", "/tmp/does-not-matter/config.yml")
    with (
        patch("osprey.utils.config.get_config_builder") as gcb,
        patch(
            "osprey.stores.type_registry.load_categories_from_config", return_value=2
        ) as load_cat,
    ):
        startup.prime_config_builder()
    gcb.assert_called_once()
    assert gcb.call_args.kwargs["config_path"] == "/tmp/does-not-matter/config.yml"
    assert gcb.call_args.kwargs["set_as_default"] is True
    load_cat.assert_called_once()


@pytest.mark.unit
def test_prime_config_builder_expands_vars(monkeypatch):
    monkeypatch.setenv("MYROOT", "/opt/osprey")
    monkeypatch.setenv("OSPREY_CONFIG", "$MYROOT/config.yml")
    with (
        patch("osprey.utils.config.get_config_builder") as gcb,
        patch("osprey.stores.type_registry.load_categories_from_config", return_value=0),
    ):
        startup.prime_config_builder()
    assert gcb.call_args.kwargs["config_path"] == "/opt/osprey/config.yml"


@pytest.mark.unit
def test_prime_config_builder_swallows_priming_failure(monkeypatch):
    """A failure to prime is non-fatal (logged, not raised)."""
    # Assert on the module logger, not caplog: full-suite logging reconfiguration
    # can cut propagation to the root logger, making caplog order-dependent.
    monkeypatch.setenv("OSPREY_CONFIG", "/tmp/config.yml")
    with (
        patch(
            "osprey.utils.config.get_config_builder",
            side_effect=RuntimeError("bad config"),
        ),
        patch.object(startup, "logger") as mock_logger,
    ):
        startup.prime_config_builder()  # must not raise
    logged = [str(c) for c in mock_logger.warning.call_args_list]
    assert any("priming failed" in msg.lower() for msg in logged)


@pytest.mark.unit
def test_prime_config_builder_survives_category_load_failure(monkeypatch):
    """Category loading is best-effort; its failure doesn't abort priming."""
    monkeypatch.setenv("OSPREY_CONFIG", "/tmp/config.yml")
    with (
        patch("osprey.utils.config.get_config_builder"),
        patch(
            "osprey.stores.type_registry.load_categories_from_config",
            side_effect=RuntimeError("registry down"),
        ),
        patch.object(startup, "logger") as mock_logger,
    ):
        startup.prime_config_builder()  # must not raise
    logged = [str(c) for c in mock_logger.warning.call_args_list]
    assert any("category loading failed" in msg.lower() for msg in logged)


# ---------------------------------------------------------------------------
# initialize_workspace_singletons
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_initialize_workspace_singletons(tmp_path):
    """The artifact store is rooted at the SHARED data root, never a
    session-relocated path — session isolation lives in the index."""
    with (
        patch("osprey.stores.artifact_store.initialize_artifact_store") as init,
        patch("osprey.utils.workspace.resolve_shared_data_root", return_value=tmp_path) as resolve,
    ):
        startup.initialize_workspace_singletons()
    resolve.assert_called_once_with()
    init.assert_called_once_with(workspace_root=tmp_path)


# ---------------------------------------------------------------------------
# run_mcp_server
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_mcp_server_wires_startup_sequence(monkeypatch):
    """Derives the label from the module path and drives dotenv->import->create->run."""
    # run_mcp_server reassigns the module-level label; re-set it via monkeypatch
    # so teardown restores the original value (serial lane, no global residue).
    monkeypatch.setattr(startup, "_server_label", startup._server_label)
    server = MagicMock()
    mod = MagicMock()
    mod.create_server.return_value = server

    order = MagicMock()

    with (
        patch("osprey.mcp_env.load_dotenv_from_project") as load_dotenv,
        patch("osprey.utils.logger.configure_logging") as configure,
        patch("importlib.import_module", return_value=mod) as import_module,
    ):
        order.attach_mock(configure, "configure_logging")
        order.attach_mock(import_module, "import_module")
        startup.run_mcp_server("osprey.mcp_server.workspace.server")

    load_dotenv.assert_called_once()
    configure.assert_called_once()
    import_module.assert_called_once_with("osprey.mcp_server.workspace.server")
    mod.create_server.assert_called_once()
    server.run.assert_called_once()
    # Logging must be configured BEFORE the server module is imported: anything
    # it logs at import time would otherwise be dropped, or worse, land on the
    # stdout the JSON-RPC transport owns.
    assert [name for name, _, _ in order.mock_calls] == ["configure_logging", "import_module"]
    # Label is the second-to-last dotted segment.
    assert startup._server_label == "workspace"
