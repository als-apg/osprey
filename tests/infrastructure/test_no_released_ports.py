"""Ratchet: no test picks a port by binding port 0 and releasing it.

A port taken from ``bind(0)`` and closed again lies in the kernel's ephemeral
range and is free for any ``connect()`` or ``bind(0)`` on the host until the
server the test starts binds it. A test takes its port from ``tests._ports``
instead: a server it controls binds port 0 itself and reports the number, a
free port that is held is a ``reserved_port()`` / ``reserved_ports()``, and an
occupied port is a ``bound_socket()``.

A site is a ``.bind((<host>, 0))`` call, whatever expression the host is, or a
test's use of a product helper that picks a port the same way: an import of
``osprey.interfaces._serving.free_port`` or a call of
``osprey.infrastructure.proxy.lifecycle.find_free_port``. ``REMAINING_SITES``
lists the files that still hold one. A file outside it fails the ratchet, and
so does a listed file that no longer holds one, so the list only shrinks.
"""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]

#: Files under ``tests/`` the ratchet never reads: the reservation's own probes
#: bind explicit ports.
EXEMPT = frozenset({"_ports.py"})

#: Product helpers that pick a port by binding port 0 and releasing it, by
#: ``(module, name)``.
PORT_PICKERS = frozenset(
    {
        ("osprey.interfaces._serving", "free_port"),
        ("osprey.infrastructure.proxy.lifecycle", "find_free_port"),
    }
)

#: Files under ``tests/`` that still pick a released port, sorted.
REMAINING_SITES: tuple[str, ...] = (
    "cli/test_als_lane.py",
    "cli/test_web_bind.py",
    "cli/test_web_cmd.py",
    "cli/test_web_session_lifetime.py",
    "connectors/test_epics_ca_softioc.py",
    "deployment/test_host_ports.py",
    "deployment/test_qmd_entrypoint_loopback_probe.py",
    "deployment/web_terminals/test_auth_serving.py",
    "e2e/_ca_client_name_sniffer.py",
    "e2e/_orm_stack.py",
    "e2e/claude_code/test_channel_finder_mcp_benchmarks.py",
    "e2e/fixtures/gchat_pubsub_emulator.py",
    "e2e/test_dispatch_tutorial.py",
    "e2e/test_gchat_bridge_e2e.py",
    "e2e/test_nextcloud_talk_bridge_e2e.py",
    "e2e/test_teams_bridge_e2e.py",
    "e2e/test_va_substrate_equivalence.py",
    "e2e/web_terminals/test_loopback_chokepoint.py",
    "e2e/web_terminals/test_terminal_auth_e2e.py",
    "fixtures/bench_ioc.py",
    "health/core/test_ariel.py",
    "hooks/test_approval_queue_enrichment.py",
    "hooks/test_notebook_update_hook.py",
    "hooks/test_osprey_panels_context.py",
    "hooks/test_panels_context_hook.py",
    "infrastructure/proxy/test_proxy_lifecycle.py",
    "infrastructure/test_server_launcher.py",
    "integration/test_tiled_export.py",
    "interfaces/_panel_launch.py",
    "interfaces/channel_finder/test_server_launcher.py",
    "interfaces/conftest.py",
    "interfaces/web_terminal/test_jupyter_sidecar.py",
    "mcp_server/bluesky/test_draft_tools_emit.py",
    "mcp_server/test_artifact_store.py",
    "mcp_server/test_backend_emit_sites.py",
    "mcp_server/test_endpoint_prober.py",
    "mcp_server/test_notify_agent_activity.py",
    "pty/conftest.py",
    "services/ariel_search/integration/test_default_deployment_degrades.py",
    "services/ariel_search/integration/test_enhance_image_modules.py",
    "services/ariel_search/test_attachment_fetch.py",
    "services/ariel_search/test_hybrid_search.py",
    "services/ariel_search/test_image_caption.py",
    "services/ariel_search/test_image_embedding.py",
    "services/ariel_search/test_image_lane.py",
    "services/ariel_search/test_llama_stub.py",
    "services/ariel_search/test_provider_resolver.py",
    "services/ariel_search/test_vision_errors.py",
    "services/bluesky_bridge/test_data_route.py",
    "services/bluesky_bridge/test_draft.py",
    "services/bluesky_bridge/test_read_bounded.py",
    "stores/test_gallery_launch_on_save.py",
    "test_graphdb_container.py",
    "va/e2e/conftest.py",
    "va/e2e/test_bluesky_lanes.py",
    "va/e2e/test_live_standin.py",
    "va/e2e/test_model_status_va.py",
    "va/e2e/test_per_target_posture.py",
    "va/e2e/test_planted_journal.py",
    "va/e2e/test_substrate_parity.py",
    "va/e2e/test_target_switch.py",
    "va/e2e/test_va_put_surface.py",
    "va/test_apply_fault.py",
)

_TOKEN = re.compile(r"\.bind\b|\bfree_port\b|\bfind_free_port\b")


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        receiver = _dotted(node.value)
        return None if receiver is None else f"{receiver}.{node.attr}"
    return None


def _binds_port_zero(node: ast.Call) -> bool:
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "bind"):
        return False
    if len(node.args) != 1 or not isinstance(node.args[0], ast.Tuple):
        return False
    elts = node.args[0].elts
    if len(elts) != 2:
        return False
    port = elts[1]
    return isinstance(port, ast.Constant) and port.value == 0


def has_released_port(source: str) -> bool:
    """Whether a module picks a port by binding port 0 and releasing it."""
    if not _TOKEN.search(source):
        return False
    tree = ast.parse(source)
    pickers = {name for _, name in PORT_PICKERS}
    picker_modules = {module for module, _ in PORT_PICKERS}
    # Names a picker's module is bound to in this file, mapped to the module.
    module_names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in picker_modules:
                    module_names[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if (node.module, alias.name) in PORT_PICKERS:
                    return True
                full = f"{node.module}.{alias.name}"
                if full in picker_modules:
                    module_names[alias.asname or alias.name] = full
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _binds_port_zero(node):
            return True
        if isinstance(node, ast.Attribute) and node.attr in pickers:
            module = module_names.get(_dotted(node.value) or "")
            if module is not None and (module, node.attr) in PORT_PICKERS:
                return True
    return False


def released_port_sites(root: Path) -> set[str]:
    """Files under *root*, relative and POSIX, that pick a released port."""
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
        and path.relative_to(root).as_posix() not in EXEMPT
        and has_released_port(path.read_text(encoding="utf-8"))
    }


def ratchet_failures(found: set[str], remaining: tuple[str, ...]) -> list[str]:
    """What the found sites break of the ratchet: new files and stale entries."""
    listed = set(remaining)
    failures = [
        f"{path}: picks a port by binding 0 and releasing it; take the port from "
        "tests._ports, which names each remedy"
        for path in sorted(found - listed)
    ]
    failures += [
        f"{path}: no released port left; delete it from REMAINING_SITES"
        for path in sorted(listed - found)
    ]
    if list(remaining) != sorted(listed) or len(remaining) != len(listed):
        failures.append("REMAINING_SITES is not a sorted list of distinct paths")
    return failures


def test_no_new_released_port_sites():
    assert ratchet_failures(released_port_sites(TESTS_ROOT), REMAINING_SITES) == []


@pytest.mark.parametrize(
    "source",
    [
        'import socket\nwith socket.socket() as s:\n    s.bind(("127.0.0.1", 0))\n',
        "def pick(sock, host):\n    sock.bind((host, 0))\n",
        "def pick(self):\n    self.sock.bind((self.host, 0))\n",
        "def pick(sock, hosts):\n    sock.bind((hosts[0], 0))\n",
        "def pick(sock, host):\n    sock.bind((host(), 0))\n",
        "from osprey.interfaces._serving import free_port as _free_port\n",
        "def port():\n    from osprey.interfaces._serving import free_port\n    return free_port()\n",
        "import osprey.infrastructure.proxy.lifecycle as lifecycle\n"
        "port = lifecycle.find_free_port()\n",
        "import osprey.infrastructure.proxy.lifecycle\n"
        "port = osprey.infrastructure.proxy.lifecycle.find_free_port()\n",
        "from osprey.infrastructure.proxy import lifecycle\nport = lifecycle.find_free_port()\n",
        "from osprey.infrastructure.proxy.lifecycle import find_free_port\n",
    ],
    ids=[
        "str-host",
        "name-host",
        "attribute-host",
        "subscript-host",
        "call-host",
        "serving-import-aliased",
        "serving-import-in-function",
        "lifecycle-module-aliased",
        "lifecycle-module-plain",
        "lifecycle-from-package",
        "lifecycle-name",
    ],
)
def test_every_site_shape_is_found(source):
    assert has_released_port(source)


@pytest.mark.parametrize(
    "source",
    [
        'def serve(sock, port):\n    sock.bind(("127.0.0.1", port))\n',
        'def serve(sock):\n    sock.bind(("127.0.0.1", 8080))\n',
        "def free_port():\n    return 20000\n",
        "import osprey.infrastructure.proxy.lifecycle as lifecycle\nlifecycle.start()\n",
        "SOURCE = 's.bind((\"127.0.0.1\", 0))'\n",
    ],
    ids=["port-variable", "explicit-port", "local-free-port", "other-lifecycle-call", "string"],
)
def test_a_held_or_explicit_port_is_no_site(source):
    assert not has_released_port(source)


def _copy_tree(tmp_path: Path) -> Path:
    root = tmp_path / "tests"
    for relative in (*REMAINING_SITES, *EXEMPT):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TESTS_ROOT / relative, target)
    return root


def test_a_planted_site_fails_the_ratchet(tmp_path):
    root = _copy_tree(tmp_path)
    planted = root / "planted" / "test_planted.py"
    planted.parent.mkdir()
    planted.write_text('import socket\nsocket.socket().bind(("127.0.0.1", 0))\n')
    failures = ratchet_failures(released_port_sites(root), REMAINING_SITES)
    assert len(failures) == 1
    assert failures[0].startswith("planted/test_planted.py:")


def test_a_listed_file_without_its_site_fails_the_ratchet(tmp_path):
    root = _copy_tree(tmp_path)
    converted = REMAINING_SITES[0]
    (root / converted).write_text('"""Converted."""\n')
    failures = ratchet_failures(released_port_sites(root), REMAINING_SITES)
    assert failures == [f"{converted}: no released port left; delete it from REMAINING_SITES"]


def test_the_exempt_reservation_module_is_never_a_site(tmp_path):
    root = tmp_path / "tests"
    root.mkdir()
    (root / "_ports.py").write_text('import socket\nsocket.socket().bind(("127.0.0.1", 0))\n')
    assert released_port_sites(root) == set()
