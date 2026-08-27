"""Tests for MCP server lifecycle helpers (``osprey.mcp_server.startup``).

Covers the startup-timing context manager, config-builder priming
(present/absent/failing OSPREY_CONFIG), workspace singleton init, and the shared
``run_mcp_server`` entry point wiring. Logging-mechanism behavior itself lives in
``tests/utils/test_configure_logging.py``.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import fastmcp
import pytest

from osprey.mcp_server import channel_finder_common, startup

# Imported HERE, at test-module scope, on purpose: several tests patch
# ``importlib.import_module`` globally, and ``install_audit_middleware``'s lazy
# import of this module would otherwise run its whole import chain through the
# mock. Pre-importing puts it in ``sys.modules``, so the lazy import resolves
# from there. The import-closure tests below use subprocesses precisely so this
# process's own imports cannot mask a regression.
from osprey.mcp_server.audit_middleware import AuditMiddleware  # noqa: E402

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


# ---------------------------------------------------------------------------
# Audit middleware install wiring
#
# The install site is deliberately INSIDE ``run_mcp_server``, after
# ``load_dotenv_from_project()``: ``fastmcp.settings`` snapshots the environment
# at fastmcp-import time, so importing the middleware (which imports fastmcp) at
# ``startup.py`` module scope would freeze the transport before a project
# ``.env`` had been loaded, and the skip predicate would then disagree with the
# transport the server actually speaks.
# ---------------------------------------------------------------------------


def _run(server_module: str, *, transport: str, monkeypatch, logger_mock=None):
    """Drive ``run_mcp_server`` with a mock server and a pinned fastmcp transport.

    Returns the mock server object so callers can assert on ``add_middleware``.
    """
    monkeypatch.setattr(startup, "_server_label", startup._server_label)
    monkeypatch.setattr(fastmcp.settings, "transport", transport)

    server = MagicMock()
    mod = MagicMock()
    mod.create_server.return_value = server

    stack = [
        patch("osprey.mcp_env.load_dotenv_from_project"),
        patch("osprey.utils.logger.configure_logging"),
        patch("importlib.import_module", return_value=mod),
    ]
    if logger_mock is not None:
        stack.append(patch.object(startup, "logger", logger_mock))
    with ExitStack() as es:
        for ctx in stack:
            es.enter_context(ctx)
        startup.run_mcp_server(server_module)
    return server


@pytest.mark.unit
def test_audit_middleware_is_installed_on_the_stdio_path(monkeypatch):
    """Every stdio framework server gets exactly one AuditMiddleware."""
    server = _run("osprey.mcp_server.workspace.server", transport="stdio", monkeypatch=monkeypatch)

    server.add_middleware.assert_called_once()
    (installed,) = server.add_middleware.call_args.args
    assert isinstance(installed, AuditMiddleware)
    # Installed BEFORE the server starts serving, never after.
    names = [c[0] for c in server.mock_calls]
    assert names.index("add_middleware") < names.index("run")


@pytest.mark.unit
def test_audit_middleware_is_skipped_off_stdio_with_one_named_warning(monkeypatch):
    """The event dispatcher (FASTMCP_TRANSPORT=http) is excluded by the predicate.

    The skip is never silent: one WARNING names the server that did not get it.
    """
    mock_logger = MagicMock()
    server = _run(
        "osprey.dispatch.server", transport="http", monkeypatch=monkeypatch, logger_mock=mock_logger
    )

    server.add_middleware.assert_not_called()
    server.run.assert_called_once()
    skipped = [
        str(call)
        for call in mock_logger.warning.call_args_list
        if "audit middleware NOT installed" in call.args[0]
    ]
    assert len(skipped) == 1, mock_logger.warning.call_args_list
    said = skipped[0]
    assert "dispatch" in said
    assert "http" in said


@pytest.mark.unit
def test_the_skip_predicate_reads_fastmcp_settings_not_the_environment(monkeypatch):
    """An env var fastmcp never saw must not decide the install.

    ``FASTMCP_TRANSPORT`` set after fastmcp imported has no effect on the
    transport the server actually speaks, so it must have none on the predicate
    either -- otherwise a late env write silently drops the audit layer while
    the server keeps talking stdio.
    """
    monkeypatch.setenv("FASTMCP_TRANSPORT", "http")
    server = _run("osprey.mcp_server.workspace.server", transport="stdio", monkeypatch=monkeypatch)
    server.add_middleware.assert_called_once()


@pytest.mark.unit
def test_the_skip_predicate_follows_settings_with_no_env_var_at_all(monkeypatch):
    """The mirror case: settings say http, the environment says nothing."""
    monkeypatch.delenv("FASTMCP_TRANSPORT", raising=False)
    server = _run("osprey.dispatch.server", transport="http", monkeypatch=monkeypatch)
    server.add_middleware.assert_not_called()


@pytest.mark.unit
def test_fastmcp_transport_is_the_single_predicate_seam(monkeypatch):
    """``fastmcp_transport()`` is the one place the transport is read.

    The fastmcp contract test (a project ``.env`` ``FASTMCP_TRANSPORT`` honored
    by both the predicate and the actual transport) hangs off this seam.
    """
    import fastmcp

    monkeypatch.setattr(fastmcp.settings, "transport", "http")
    assert startup.fastmcp_transport() == "http"
    monkeypatch.setattr(fastmcp.settings, "transport", "stdio")
    assert startup.fastmcp_transport() == startup.STDIO_TRANSPORT == "stdio"


# ---------------------------------------------------------------------------
# Import-order contract
# ---------------------------------------------------------------------------


def _module_scope_imports(module) -> set[str]:
    """Dotted names imported at MODULE SCOPE (``if TYPE_CHECKING:`` excluded)."""
    tree = ast.parse(Path(inspect.getsourcefile(module)).read_text())
    names: set[str] = set()
    for node in tree.body:  # top level only: a nested/guarded import is not module scope
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.unit
def test_startup_does_not_import_fastmcp_or_the_middleware_at_module_scope():
    imported = _module_scope_imports(startup)
    assert not [n for n in imported if n == "fastmcp" or n.startswith("fastmcp.")]
    assert "osprey.mcp_server.audit_middleware" not in imported


@pytest.mark.unit
def test_channel_finder_common_does_not_import_fastmcp_at_module_scope():
    """The fastmcp-before-dotenv wrinkle: ``run_cf_main``'s own module used to
    import fastmcp at module scope, so every ``python -m
    osprey.mcp_server.channel_finder_<variant>`` froze ``fastmcp.settings``
    before the project ``.env`` was loaded."""
    imported = _module_scope_imports(channel_finder_common)
    assert not [n for n in imported if n == "fastmcp" or n.startswith("fastmcp.")]


@pytest.mark.parametrize(
    "module",
    ["osprey.mcp_server.startup", "osprey.mcp_server.channel_finder_common"],
)
@pytest.mark.unit
def test_importing_an_entry_point_module_does_not_pull_in_fastmcp(module):
    """The real closure, not just the direct imports: a transitive import of
    fastmcp would freeze its settings just as effectively."""
    code = (
        "import sys, importlib\n"
        f"importlib.import_module({module!r})\n"
        "print('fastmcp' in sys.modules,"
        " 'osprey.mcp_server.audit_middleware' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "False False", f"{module} import closure: {out}"


# ---------------------------------------------------------------------------
# run_cf_main
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_cf_main_delegates_to_run_mcp_server(monkeypatch):
    """Folded in, not re-implemented: one startup sequence, one install site."""
    seen: list[str] = []
    monkeypatch.setattr(startup, "run_mcp_server", seen.append)
    channel_finder_common.run_cf_main("osprey.mcp_server.channel_finder_graph.server")
    assert seen == ["osprey.mcp_server.channel_finder_graph.server"]


@pytest.mark.unit
def test_run_cf_main_installs_the_audit_middleware_too(monkeypatch):
    """Wired identically: a channel-finder variant is audited like any other."""
    monkeypatch.setattr(startup, "_server_label", startup._server_label)
    monkeypatch.setattr(fastmcp.settings, "transport", "stdio")

    server = MagicMock()
    mod = MagicMock()
    mod.create_server.return_value = server

    with (
        patch("osprey.mcp_env.load_dotenv_from_project"),
        patch("osprey.utils.logger.configure_logging"),
        patch("importlib.import_module", return_value=mod),
    ):
        channel_finder_common.run_cf_main("osprey.mcp_server.channel_finder_graph.server")

    server.add_middleware.assert_called_once()
    assert isinstance(server.add_middleware.call_args.args[0], AuditMiddleware)
    assert startup._server_label == "channel_finder_graph"
