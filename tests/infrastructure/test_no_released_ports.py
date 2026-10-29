"""Ratchet: no new code picks a port by binding port 0 and releasing it.

A port taken from ``bind(0)`` and closed again lies in the kernel's ephemeral
range and is free for any ``connect()`` or ``bind(0)`` on the host until the
server it was picked for binds it. A server binds port 0 itself and reports
the port it was given instead (``run_app_server`` yields the URL of an app on
a socket it owns); where the number must be known before the bind, a held
``reserved_port()`` from ``osprey_connectors_tests._harness._ports`` supplies
it.

A site is either a use of ``free_port`` or ``find_free_port``, or a
``.bind((<host>, 0))`` on a socket whose ``getsockname()`` the same scope reads
and which that scope never ``listen()``s on. A socket that listens is a server
holding the port it reports, which is the remedy, not a site.

The walk covers ``src/`` and ``tests/``. ``REMAINING_SITES`` maps each file
that still holds sites, relative to the repository root, to its count. A file
outside it fails the ratchet, and so does a file with more sites than listed or
with fewer, so the counts only shrink.
"""

from __future__ import annotations

import ast
import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The trees walked, relative to the repository root.
TREES = ("src", "tests")

#: Helpers that return a port number after releasing it.
RELEASING_HELPERS = frozenset({"free_port", "find_free_port"})

#: Files that still pick a released port, relative to the repository root,
#: sorted, each with its number of sites.
REMAINING_SITES: dict[str, int] = {
    "src/osprey/cli/theme_lab_cmd.py": 1,
    "src/osprey/cli/web_cmd.py": 1,
    "src/osprey/infrastructure/proxy/lifecycle.py": 2,
    "src/osprey/interfaces/_serving.py": 1,
    "tests/cli/test_als_lane.py": 1,
    "tests/cli/test_web_bind.py": 1,
    "tests/cli/test_web_cmd.py": 43,
    "tests/cli/test_web_session_lifetime.py": 9,
    "tests/deployment/test_qmd_entrypoint_loopback_probe.py": 1,
    "tests/deployment/web_terminals/test_auth_serving.py": 2,
    "tests/e2e/_dispatch_harness.py": 2,
    "tests/e2e/_orm_stack.py": 1,
    "tests/e2e/claude_code/test_channel_finder_mcp_benchmarks.py": 1,
    "tests/e2e/fixtures/gchat_pubsub_emulator.py": 1,
    "tests/e2e/test_dispatch_allowlist_parity.py": 1,
    "tests/e2e/test_target_switch_agentic.py": 1,
    "tests/e2e/test_va_substrate_equivalence.py": 1,
    "tests/e2e/web_terminals/test_loopback_chokepoint.py": 1,
    "tests/e2e/web_terminals/test_session_restart_e2e.py": 1,
    "tests/e2e/web_terminals/test_terminal_auth_e2e.py": 1,
    "tests/fixtures/bench_ioc.py": 1,
    "tests/health/core/test_ariel.py": 1,
    "tests/hooks/test_approval_queue_enrichment.py": 3,
    "tests/hooks/test_notebook_update_hook.py": 1,
    "tests/hooks/test_osprey_panels_context.py": 1,
    "tests/hooks/test_panels_context_hook.py": 3,
    "tests/infrastructure/proxy/test_proxy_lifecycle.py": 1,
    "tests/infrastructure/test_server_launcher.py": 6,
    "tests/integration/test_tiled_export.py": 1,
    "tests/interfaces/_panel_launch.py": 1,
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py": 1,
    "tests/interfaces/web_terminal/test_opaque_panel_browser.py": 1,
    "tests/interfaces/web_terminal/test_panels_browser.py": 5,
    "tests/interfaces/web_terminal/test_panels_collab_browser.py": 1,
    "tests/mcp_server/bluesky/test_draft_tools_emit.py": 1,
    "tests/mcp_server/test_artifact_store.py": 2,
    "tests/mcp_server/test_backend_emit_sites.py": 2,
    "tests/mcp_server/test_endpoint_prober.py": 1,
    "tests/pty/conftest.py": 1,
    "tests/services/ariel_search/integration/test_default_deployment_degrades.py": 1,
    "tests/services/ariel_search/integration/test_enhance_image_modules.py": 1,
    "tests/services/ariel_search/test_attachment_fetch.py": 1,
    "tests/services/ariel_search/test_hybrid_search.py": 1,
    "tests/services/ariel_search/test_image_caption.py": 4,
    "tests/services/ariel_search/test_image_embedding.py": 1,
    "tests/services/ariel_search/test_image_lane.py": 1,
    "tests/services/ariel_search/test_llama_stub.py": 1,
    "tests/services/ariel_search/test_provider_resolver.py": 8,
    "tests/services/ariel_search/test_semantic_search.py": 1,
    "tests/services/ariel_search/test_vision_errors.py": 1,
    "tests/services/bluesky_bridge/test_data_route.py": 1,
    "tests/stores/test_gallery_launch_on_save.py": 1,
    "tests/test_graphdb_container.py": 1,
    "tests/va/e2e/conftest.py": 2,
    "tests/va/e2e/test_bench_ioc_live.py": 1,
    "tests/va/e2e/test_bluesky_lanes.py": 1,
    "tests/va/e2e/test_live_standin.py": 2,
    "tests/va/e2e/test_model_status_va.py": 2,
    "tests/va/e2e/test_per_target_posture.py": 1,
    "tests/va/e2e/test_planted_journal.py": 1,
    "tests/va/e2e/test_substrate_parity.py": 1,
    "tests/va/e2e/test_target_switch.py": 2,
    "tests/va/e2e/test_va_put_surface.py": 2,
    "tests/va/test_apply_fault.py": 4,
    "tests/va/test_lume_pva_seam.py": 4,
}

_TOKEN = re.compile(r"\.bind\b|free_port\b")

_SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node of *scope* outside the functions nested in it."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def _method_receiver(node: ast.AST, method: str) -> str | None:
    """The dumped receiver of a ``<receiver>.<method>(...)`` call, else None."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
    ):
        return ast.dump(node.func.value)
    return None


def _binds_port_zero(node: ast.AST) -> bool:
    if _method_receiver(node, "bind") is None:
        return False
    assert isinstance(node, ast.Call)
    if len(node.args) != 1 or not isinstance(node.args[0], ast.Tuple):
        return False
    elts = node.args[0].elts
    if len(elts) != 2:
        return False
    port = elts[1]
    return isinstance(port, ast.Constant) and port.value == 0


def _released_binds(scope: ast.AST) -> int:
    nodes = list(_scope_nodes(scope))
    read = {r for n in nodes if (r := _method_receiver(n, "getsockname")) is not None}
    listening = {r for n in nodes if (r := _method_receiver(n, "listen")) is not None}
    return sum(
        1
        for n in nodes
        if _binds_port_zero(n)
        and (receiver := _method_receiver(n, "bind")) in read
        and receiver not in listening
    )


def _helper_uses(tree: ast.AST) -> int:
    return sum(
        1
        for node in ast.walk(tree)
        if (isinstance(node, ast.Name) and node.id in RELEASING_HELPERS)
        or (isinstance(node, ast.Attribute) and node.attr in RELEASING_HELPERS)
    )


def released_port_count(source: str) -> int:
    """How many places in a module pick a port by binding 0 and releasing it."""
    if not _TOKEN.search(source):
        return 0
    tree = ast.parse(source)
    scopes = [node for node in ast.walk(tree) if isinstance(node, _SCOPES)]
    return _helper_uses(tree) + sum(_released_binds(scope) for scope in scopes)


def released_port_sites(anchor: Path) -> dict[str, int]:
    """Files of the trees under *anchor*, relative to it and POSIX, with their site counts."""
    found = {}
    for tree in TREES:
        for path in (anchor / tree).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            count = released_port_count(path.read_text(encoding="utf-8"))
            if count:
                found[path.relative_to(anchor).as_posix()] = count
    return found


def ratchet_failures(found: dict[str, int], remaining: dict[str, int]) -> list[str]:
    """What the found sites break of the ratchet: new or grown sites, and stale counts."""
    remedy = (
        "bind port 0 in the server and read the bound port back from its socket "
        "(run_app_server does this for an app); where the number is needed before "
        "the bind, hold a reserved_port() from osprey_connectors_tests._harness._ports"
    )
    failures = []
    for path in sorted(found.keys() | remaining.keys()):
        now, listed = found.get(path, 0), remaining.get(path, 0)
        if now > listed:
            failures.append(
                f"{path}: {now} pick-and-release port site(s), {listed} allowed; {remedy}"
            )
        elif now == 0:
            failures.append(f"{path}: no released port left; delete it from REMAINING_SITES")
        elif now < listed:
            failures.append(
                f"{path}: {now} site(s) left, {listed} listed; lower it to {now} in REMAINING_SITES"
            )
    if list(remaining) != sorted(remaining):
        failures.append("REMAINING_SITES is not sorted by path")
    return failures


def test_no_new_released_port_sites():
    assert ratchet_failures(released_port_sites(REPO_ROOT), REMAINING_SITES) == []


@pytest.mark.parametrize(
    "source",
    [
        'import socket\nwith socket.socket() as s:\n    s.bind(("127.0.0.1", 0))\n'
        "    port = s.getsockname()[1]\n",
        "def pick(sock, host):\n    sock.bind((host, 0))\n    return sock.getsockname()[1]\n",
        "def pick(self):\n    self.sock.bind((self.host, 0))\n    return self.sock.getsockname()\n",
        "from osprey.interfaces._serving import free_port\nport = free_port()\n",
        "from osprey.interfaces import _serving\nport = _serving.free_port()\n",
        "def start(factory):\n    return factory(port_factory=find_free_port)\n",
    ],
    ids=[
        "with-block",
        "name-host",
        "attribute-socket",
        "free-port-call",
        "free-port-attribute",
        "find-free-port-reference",
    ],
)
def test_every_site_shape_is_found(source):
    assert released_port_count(source) == 1


@pytest.mark.parametrize(
    "source",
    [
        'def serve(sock):\n    sock.bind(("127.0.0.1", 0))\n    sock.listen(1)\n'
        "    return sock.getsockname()[1]\n",
        'def serve(sock, port):\n    sock.bind(("127.0.0.1", port))\n'
        "    return sock.getsockname()[1]\n",
        'def probe(sock):\n    sock.bind(("::1", 0))\n',
        "SOURCE = 's.bind((\"127.0.0.1\", 0)); free_port()'\n",
    ],
    ids=["listening-server", "explicit-port", "no-port-read", "string"],
)
def test_a_held_or_explicit_port_is_no_site(source):
    assert released_port_count(source) == 0


def _copy_tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for tree in TREES:
        (root / tree).mkdir(parents=True)
    for relative in REMAINING_SITES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
    return root


def test_a_planted_site_fails_the_ratchet(tmp_path):
    root = _copy_tree(tmp_path)
    planted = root / "tests/planted/test_planted.py"
    planted.parent.mkdir()
    planted.write_text("from osprey.interfaces._serving import free_port\nPORT = free_port()\n")
    failures = ratchet_failures(released_port_sites(root), REMAINING_SITES)
    assert len(failures) == 1
    assert failures[0].startswith("tests/planted/test_planted.py: 1 pick-and-release")
    assert "bind port 0 in the server" in failures[0]


def test_a_grown_listed_file_fails_the_ratchet(tmp_path):
    root = _copy_tree(tmp_path)
    grown = next(iter(REMAINING_SITES))
    with (root / grown).open("a", encoding="utf-8") as handle:
        handle.write("\n\ndef _planted():\n    return free_port()\n")
    failures = ratchet_failures(released_port_sites(root), REMAINING_SITES)
    assert len(failures) == 1
    assert failures[0].startswith(f"{grown}: {REMAINING_SITES[grown] + 1} pick-and-release")


def test_a_converted_file_fails_the_ratchet_until_it_is_delisted(tmp_path):
    root = _copy_tree(tmp_path)
    converted = next(iter(REMAINING_SITES))
    (root / converted).write_text('"""Converted."""\n')
    failures = ratchet_failures(released_port_sites(root), REMAINING_SITES)
    assert failures == [f"{converted}: no released port left; delete it from REMAINING_SITES"]


def test_a_count_above_the_tree_must_be_lowered():
    failures = ratchet_failures({"tests/a.py": 2}, {"tests/a.py": 3})
    assert failures == ["tests/a.py: 2 site(s) left, 3 listed; lower it to 2 in REMAINING_SITES"]
