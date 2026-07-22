"""Tests for MCP server lifecycle helpers (``osprey.mcp_server.startup``).

Covers the startup-timing context manager, the stderr logging redirect and its
idempotence guard, config-builder priming (present/absent/failing OSPREY_CONFIG),
workspace singleton init, and the shared ``run_mcp_server`` entry point wiring.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from rich.logging import RichHandler

from osprey.mcp_server import startup


@pytest.fixture
def restore_root_logging():
    """Snapshot and restore the root logger so redirect tests don't leak handlers."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_lib_levels = {
        lib: logging.getLogger(lib).level
        for lib in ["httpx", "httpcore", "requests", "urllib3", "LiteLLM"]
    }
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    for lib, lvl in saved_lib_levels.items():
        logging.getLogger(lib).setLevel(lvl)


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
# redirect_logging_to_stderr
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_redirect_installs_stderr_rich_handler(restore_root_logging):
    root = logging.getLogger()
    root.handlers[:] = []  # start clean for a deterministic assertion
    startup.redirect_logging_to_stderr()
    rich_handlers = [h for h in root.handlers if isinstance(h, RichHandler)]
    assert len(rich_handlers) == 1
    assert rich_handlers[0].console.stderr is True
    # Noisy third-party loggers are pinned to WARNING.
    assert logging.getLogger("httpx").level == logging.WARNING


@pytest.mark.unit
def test_redirect_is_idempotent(restore_root_logging):
    """A second call must not add a second RichHandler (stdout stays clean)."""
    root = logging.getLogger()
    root.handlers[:] = []
    startup.redirect_logging_to_stderr()
    startup.redirect_logging_to_stderr()
    rich_handlers = [h for h in root.handlers if isinstance(h, RichHandler)]
    assert len(rich_handlers) == 1


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
    with patch("osprey.stores.artifact_store.initialize_artifact_store") as init:
        startup.initialize_workspace_singletons(tmp_path)
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

    with (
        patch("osprey.mcp_env.load_dotenv_from_project") as load_dotenv,
        patch.object(startup, "redirect_logging_to_stderr") as redirect,
        patch("importlib.import_module", return_value=mod) as import_module,
    ):
        startup.run_mcp_server("osprey.mcp_server.workspace.server")

    load_dotenv.assert_called_once()
    redirect.assert_called_once()
    import_module.assert_called_once_with("osprey.mcp_server.workspace.server")
    mod.create_server.assert_called_once()
    server.run.assert_called_once()
    # Label is the second-to-last dotted segment.
    assert startup._server_label == "workspace"
