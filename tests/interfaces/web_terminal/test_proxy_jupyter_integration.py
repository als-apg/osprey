"""The notebook panel end to end: a real sidecar behind the terminal's panel proxy.

The sidecar tests pin the launch contract against the server directly; the
panel-proxy tests pin the header boundary against a stubbed backend. This module
puts the two together — a real ``jupyter_server`` on loopback, reached only
through ``/panel/jupyter/*`` — so what is asserted here is what a browser at the
terminal's own origin actually gets: the lab page, a kernel on the one
kernelspec, a live channels socket, and neither the sidecar's token nor its
cookies.

Every test shares one sidecar and one kernel (module-scoped fixtures). Starting
either is the expensive part, and nothing here mutates state another test reads.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.parse import urljoin

import httpx
import pytest
import websockets
from fastapi.testclient import TestClient

from osprey.interfaces.common_middleware import compute_url_prefix
from osprey.interfaces.web_terminal.app import UNIVERSAL_PANELS, create_app
from osprey.interfaces.web_terminal.jupyter_sidecar import (
    KERNELSPEC_NAME,
    STARTER_NOTEBOOK_NAME,
    JupyterSidecar,
)

#: Both halves of every test here spawn a process: the sidecar, then a kernel.
pytestmark = pytest.mark.slow

READY_TIMEOUT = 90.0

#: How long a kernel gets to answer one request. Generous: the first message on
#: a fresh socket waits for the interpreter to import the whole kernel stack.
KERNEL_TIMEOUT = 180.0

#: The panel's path under the terminal's origin — the only way in, here.
PANEL = "/panel/jupyter"

#: An identity the deployment stamps on the terminal. ``scrub_sandbox_child_env``
#: keeps it, so it must reach a cell; a name in the ``OSPREY_TERMINAL_`` family
#: is dropped, so its absence in a cell must be the scrub's doing and not the
#: test environment simply never having set one.
AUDIT_IDENTITY = "notebook-integration"
TERMINAL_FAMILY_PROBE = "OSPREY_TERMINAL_PROBE"

#: What a cell reports about its own environment.
PROBE_CODE = (
    "import os, json; "
    'print(json.dumps({"tok": "JUPYTER_TOKEN" in os.environ, '
    '"term": [k for k in os.environ if k.startswith("OSPREY_TERMINAL_")], '
    '"cfg": "OSPREY_CONFIG" in os.environ, '
    '"audit": "OSPREY_AUDIT_IDENTITY" in os.environ}))'
)

#: The Jupyter messaging-protocol version a hand-built message declares.
PROTOCOL_VERSION = "5.3"

#: The websocket subprotocol JupyterLab offers on a kernel's channels socket.
KERNEL_WS_PROTOCOL_V1 = "v1.kernel.websocket.jupyter.org"


# ---------------------------------------------------------------------------
# The sidecar, the terminal, and one kernel
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def notebook_env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """The environment a sidecar launch reads, and a root to keep it all under."""
    root = tmp_path_factory.mktemp("notebook-panel")
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("OSPREY_CONFIG", str(root / "does-not-exist.yml"))
        # Keep every per-launch tempdir under the test's own tree.
        environment.setenv("TMPDIR", str(root))
        environment.setenv("OSPREY_AUDIT_IDENTITY", AUDIT_IDENTITY)
        environment.setenv(TERMINAL_FAMILY_PROBE, "reachable-from-the-terminal")
        yield root


@pytest.fixture(scope="module")
def sidecar(notebook_env: Path) -> Iterator[JupyterSidecar]:
    """A real ``jupyter_server`` on an OS-assigned loopback port."""
    running = JupyterSidecar(notebook_env / "shared", compute_url_prefix(), None)
    try:
        running.preflight()
        running.spawn()
        running.wait_ready(READY_TIMEOUT)
        yield running
    finally:
        running.stop()


@pytest.fixture(scope="module")
def proxied(sidecar: JupyterSidecar, notebook_env: Path) -> Iterator[TestClient]:
    """A web terminal whose ``jupyter`` panel resolves to the running sidecar.

    The launcher's two publications are made by hand rather than by enabling the
    panel, so a failed launch shows up as this fixture erroring with the
    sidecar's own message instead of as a 404 from a lifespan that swallowed it.
    """
    workspace = notebook_env / "_agent_data"
    workspace.mkdir(exist_ok=True)
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(set(UNIVERSAL_PANELS), [], None),
        ),
        # Entering the TestClient runs the app's lifespan, which auto-launches a
        # companion server for every enabled panel — and ``UNIVERSAL_PANELS`` is
        # ``{"artifacts"}``, so the artifact server starts for real: a daemon
        # thread running uvicorn on a configured port, whose app factory resolves
        # the shared data root. Two consequences, both observed: unstamped, that
        # root is the repository, so the thread creates ``var/agent_data/artifacts``
        # and trips the repo-leak guard; and the port is fixed rather than
        # OS-assigned, so under xdist a second worker's launch loses the bind and
        # the thread dies in uvicorn's startup path. This module exercises
        # ``/panel/jupyter`` and nothing else, and publishes that URL by hand
        # below, so it needs no companion server at all.
        patch(
            "osprey.infrastructure.server_launcher.ensure_web_server",
            lambda key: None,
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as client:
            app.state.jupyter_server_url = sidecar.url
            app.state.panel_auth_headers = {"jupyter": dict(sidecar.auth_headers)}
            yield client


@pytest.fixture(scope="module")
def started_session(proxied: TestClient) -> Iterator[httpx.Response]:
    """One notebook session, started through the proxy and deleted afterwards."""
    response = proxied.post(
        f"{PANEL}/api/sessions",
        json={
            "path": STARTER_NOTEBOOK_NAME,
            "name": STARTER_NOTEBOOK_NAME,
            "type": "notebook",
            "kernel": {"name": KERNELSPEC_NAME},
        },
    )
    try:
        yield response
    finally:
        if response.status_code == 201:
            proxied.delete(f"{PANEL}/api/sessions/{response.json()['id']}")


@pytest.fixture(scope="module")
def kernel_id(started_session: httpx.Response) -> str:
    return str(started_session.json()["kernel"]["id"])


# ---------------------------------------------------------------------------
# Talking to a kernel over the proxied channels socket
# ---------------------------------------------------------------------------


def _message(msg_type: str, content: dict[str, Any], session_id: str) -> dict[str, Any]:
    """One Jupyter message in the JSON form the default subprotocol carries."""
    return {
        "header": {
            "msg_id": uuid.uuid4().hex,
            "session": session_id,
            "username": "osprey",
            "date": datetime.now(UTC).isoformat(),
            "msg_type": msg_type,
            "version": PROTOCOL_VERSION,
        },
        "parent_header": {},
        "metadata": {},
        "content": content,
        "channel": "shell",
        "buffers": [],
    }


def _reply_to(socket: Any, request: dict[str, Any], msg_type: str) -> dict[str, Any]:
    """Read until *request*'s reply of *msg_type* arrives, or fail on the deadline."""
    deadline = time.monotonic() + KERNEL_TIMEOUT
    while time.monotonic() < deadline:
        message = socket.receive_json()
        header = message.get("header") or {}
        parent = message.get("parent_header") or {}
        if (
            header.get("msg_type") == msg_type
            and parent.get("msg_id") == request["header"]["msg_id"]
        ):
            return message
    raise AssertionError(f"no {msg_type} within {KERNEL_TIMEOUT:.0f} s")


def _run_cell(socket: Any, code: str, session_id: str) -> str:
    """Execute *code* on the kernel and return everything it wrote to stdout.

    Reads to the ``idle`` status parented by the request rather than to the
    reply: the reply rides the shell channel and the output rides iopub, and
    only ``idle`` says the kernel is done with both.
    """
    request = _message(
        "execute_request",
        {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        session_id,
    )
    socket.send_json(request)

    written: list[str] = []
    deadline = time.monotonic() + KERNEL_TIMEOUT
    while time.monotonic() < deadline:
        message = socket.receive_json()
        header = message.get("header") or {}
        parent = message.get("parent_header") or {}
        if parent.get("msg_id") != request["header"]["msg_id"]:
            continue
        content = message.get("content") or {}
        if header.get("msg_type") == "stream" and content.get("name") == "stdout":
            written.append(str(content.get("text", "")))
        elif header.get("msg_type") == "error":
            raise AssertionError("\n".join(content.get("traceback", [])))
        elif header.get("msg_type") == "status" and content.get("execution_state") == "idle":
            return "".join(written)
    raise AssertionError(f"the cell did not finish within {KERNEL_TIMEOUT:.0f} s")


class _RecordingConnect:
    """``websockets.connect``, wrapped to record the handshake it then performs."""

    def __init__(self) -> None:
        self._connect = websockets.connect
        self.target: str | None = None
        self.headers: dict[str, str] = {}

    def __call__(self, target: str, **kwargs: Any) -> Any:
        self.target = target
        self.headers = dict(kwargs.get("additional_headers") or {})
        return self._connect(target, **kwargs)


# ---------------------------------------------------------------------------
# Talking to the sidecar without the proxy, to show what the proxy supplies
# ---------------------------------------------------------------------------


def _direct(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str]]:
    """Request *url* off the proxy's path; answer its status and headers."""
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return int(response.status), {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as exc:
        exc.read()
        return int(exc.code), {k.lower(): v for k, v in exc.headers.items()}


# ---------------------------------------------------------------------------
# The panel over HTTP
# ---------------------------------------------------------------------------


def test_the_panel_answers_through_the_proxy(proxied: TestClient) -> None:
    lab = proxied.get(f"{PANEL}/lab")
    status = proxied.get(f"{PANEL}/api/status")
    kernelspecs = proxied.get(f"{PANEL}/api/kernelspecs")

    assert lab.status_code == 200
    assert status.status_code == 200
    assert kernelspecs.status_code == 200
    assert sorted(kernelspecs.json()["kernelspecs"]) == [KERNELSPEC_NAME]


def test_the_backend_refuses_what_the_proxy_does_not_sign(
    proxied: TestClient, sidecar: JupyterSidecar
) -> None:
    """The credential the proxy injects is the one the sidecar is gating on."""
    unsigned, _ = _direct(f"{sidecar.url}/api/status")

    assert unsigned == 403
    assert proxied.get(f"{PANEL}/api/status").status_code == 200


def test_no_backend_cookie_reaches_the_browser(
    proxied: TestClient, sidecar: JupyterSidecar
) -> None:
    """The sidecar sets a login cookie; it must not be stored for the terminal."""
    _status, upstream_headers = _direct(f"{sidecar.url}/lab", sidecar.auth_headers)

    through_proxy = proxied.get(f"{PANEL}/lab")

    assert "set-cookie" in upstream_headers
    assert "set-cookie" not in {name.lower() for name in through_proxy.headers}


def test_a_session_starts_on_the_osprey_kernelspec(started_session: httpx.Response) -> None:
    assert started_session.status_code == 201
    assert started_session.json()["kernel"]["name"] == KERNELSPEC_NAME


# ---------------------------------------------------------------------------
# The channels socket
# ---------------------------------------------------------------------------


@pytest.mark.timeout(KERNEL_TIMEOUT)
def test_a_kernel_answers_over_the_proxied_socket(
    proxied: TestClient, sidecar: JupyterSidecar, kernel_id: str
) -> None:
    """A kernel_info round trip, and the upstream handshake that carried it."""
    session_id = uuid.uuid4().hex
    recorded = _RecordingConnect()

    with patch("websockets.connect", recorded):
        with proxied.websocket_connect(
            f"{PANEL}/api/kernels/{kernel_id}/channels?session_id={session_id}"
        ) as socket:
            request = _message("kernel_info_request", {}, session_id)
            socket.send_json(request)
            reply = _reply_to(socket, request, "kernel_info_reply")

    assert reply["content"]["status"] == "ok"
    assert recorded.target is not None
    assert recorded.target.endswith(
        f"{PANEL}/api/kernels/{kernel_id}/channels?session_id={session_id}"
    )
    handshake = {name.lower(): value for name, value in recorded.headers.items()}
    assert handshake["authorization"] == f"Bearer {sidecar.token}"


@pytest.mark.timeout(KERNEL_TIMEOUT)
def test_a_cell_sees_no_server_token_and_no_terminal_family(
    proxied: TestClient, kernel_id: str
) -> None:
    """What a cell inherits: the config and the audit identity, and nothing else.

    ``JUPYTER_TOKEN`` is the sidecar's own credential — the kernelspec carries it
    empty and the launcher pops the name outright, so a cell holds neither. The
    ``OSPREY_TERMINAL_`` family is dropped by ``scrub_sandbox_child_env``, which
    is only visible because the terminal's environment set one.
    """
    session_id = uuid.uuid4().hex

    with proxied.websocket_connect(
        f"{PANEL}/api/kernels/{kernel_id}/channels?session_id={session_id}"
    ) as socket:
        reported = json.loads(_run_cell(socket, PROBE_CODE, session_id))

    assert reported == {"tok": False, "term": [], "cfg": True, "audit": True}


@pytest.mark.timeout(KERNEL_TIMEOUT)
def test_the_kernel_protocol_the_browser_offers_is_negotiated_through_the_proxy(
    proxied: TestClient, kernel_id: str
) -> None:
    """JupyterLab offers the v1 kernel protocol; the proxy must answer with it.

    Under v1 every message is one binary frame, so the round trip here is
    framed with the server's own v1 helpers rather than as JSON.
    """
    from jupyter_server.services.kernels.connection.base import (
        deserialize_msg_from_ws_v1,
        serialize_msg_to_ws_v1,
    )

    session_id = uuid.uuid4().hex
    request = _message("kernel_info_request", {}, session_id)

    with proxied.websocket_connect(
        f"{PANEL}/api/kernels/{kernel_id}/channels?session_id={session_id}",
        subprotocols=[KERNEL_WS_PROTOCOL_V1],
    ) as socket:
        assert socket.accepted_subprotocol == KERNEL_WS_PROTOCOL_V1

        socket.send_bytes(
            serialize_msg_to_ws_v1(request, "shell", pack=lambda obj: json.dumps(obj).encode())
        )
        deadline = time.monotonic() + KERNEL_TIMEOUT
        while time.monotonic() < deadline:
            _channel, parts = deserialize_msg_from_ws_v1(socket.receive_bytes())
            header, parent = json.loads(parts[0]), json.loads(parts[1])
            if (
                header.get("msg_type") == "kernel_info_reply"
                and parent.get("msg_id") == request["header"]["msg_id"]
            ):
                assert json.loads(parts[3])["status"] == "ok"
                break
        else:
            raise AssertionError(f"no kernel_info_reply within {KERNEL_TIMEOUT:.0f} s")


# ---------------------------------------------------------------------------
# The control-target bar the proxy injects into the Lab page
# ---------------------------------------------------------------------------

#: The bar's entry module, and the directory the whole closure is served from.
LAB_BAR_MODULE = "control-target-lab-bar.js"
STATIC_JS = Path(__file__).resolve().parents[3] / "src/osprey/interfaces/web_terminal/static/js"

#: A relative ES-module specifier, in the two forms these files use. Comments
#: are stripped before this runs: the modules discuss their own imports in
#: prose, and a walker that read prose as an edge would demand files the
#: browser never fetches.
_IMPORT_RE = re.compile(r"\bfrom\s+'(\.[^']+)'|^\s*import\s+'(\.[^']+)'", re.MULTILINE)
_COMMENT_RE = re.compile(r"/\*.*?\*/|(?<!:)//[^\n]*", re.DOTALL)

#: ``new URL('<specifier>', import.meta.url)`` — how the bar addresses its two
#: stylesheets. Resolved against the module's own served URL below, which is
#: what a browser does with it.
_ASSET_URL_RE = re.compile(r"new URL\('([^']+)',\s*import\.meta\.url\)")


def _relative_imports(source: str) -> set[str]:
    stripped = _COMMENT_RE.sub("", source)
    return {(a or b) for a, b in _IMPORT_RE.findall(stripped)}


def _module_closure(entry: str) -> set[str]:
    """Every module reachable from *entry* by relative import, transitively.

    The same walk the bar's vitest suite does, from the other side. That suite
    pins WHICH modules the closure contains; this one pins that every one of
    them is reachable through the panel-scoped static route — the two failures
    are different, and only the second is a 404 in an operator's browser.
    """
    seen = {entry}
    queue = [entry]
    while queue:
        current = queue.pop()
        for specifier in _relative_imports((STATIC_JS / current).read_text(encoding="utf-8")):
            name = (STATIC_JS / current).parent.joinpath(specifier).resolve().name
            if name not in seen:
                seen.add(name)
                queue.append(name)
    return seen


def _lab_head(proxied: TestClient) -> str:
    """The ``<head>`` of the proxied Lab page."""
    body = proxied.get(f"{PANEL}/lab").text
    return body[: body.index("</head>")]


def test_the_lab_page_carries_the_control_target_bar(
    proxied: TestClient, sidecar: JupyterSidecar
) -> None:
    """The three tags, in the head, addressed at the hub rather than the panel.

    ``base_url`` is the sidecar's own — every path the Lab page emits is
    relative to it — so the module's ``src`` being exactly that plus the static
    route's own segment is the whole of "this resolves back onto the hub's
    files from inside the panel's namespace".
    """
    head = _lab_head(proxied)
    expected_src = f"{sidecar.base_url}terminal-static/js/{LAB_BAR_MODULE}"

    assert f'<script type="module" src="{expected_src}"></script>' in head
    assert head.count(LAB_BAR_MODULE) == 1

    # The prefix global is the OUTER prefix. A prefix of ``/panel/jupyter``
    # would aim every read the chip makes at JupyterLab, which answers none.
    assert f'window.__OSPREY_PREFIX__ = "{compute_url_prefix()}";' in head
    assert 'window.__OSPREY_PREFIX__ = "/panel' not in head


def test_the_injected_import_map_is_scoped_to_the_bars_own_modules(
    proxied: TestClient, sidecar: JupyterSidecar
) -> None:
    """One root-absolute specifier is retargeted, and only for the bar's files.

    ``confirm-skip.js`` imports ``/design-system/js/storage-scope.js`` — the
    ninth file the bar's graph fetches, and the only one that is not a relative
    import. The hub's own page retargets that specifier with a document-wide
    import map; here the mapping lives in a ``scopes`` entry keyed on the bar's
    directory, so nothing JupyterLab itself imports can be redirected by it.

    The target is the HUB's own mount, byte-for-byte what the hub's index.html
    maps the same specifier to — so the module resolves to the same file on
    both surfaces, and no rewrite pass runs over its body on the way.
    """
    head = _lab_head(proxied)
    raw = head.split('<script type="importmap">')[1].split("</script>")[0]
    import_map = json.loads(raw)

    scope = f"{sidecar.base_url}terminal-static/js/"
    assert list(import_map) == ["scopes"]
    assert list(import_map["scopes"]) == [scope]
    assert import_map["scopes"][scope] == {
        "/design-system/": f"{compute_url_prefix()}/design-system/"
    }

    # And the mapping resolves: the specifier confirm-skip.js writes, put
    # through this map, is a module the terminal answers with.
    mapped = import_map["scopes"][scope]["/design-system/"] + "js/storage-scope.js"
    resolved = proxied.get(mapped)

    assert resolved.status_code == 200
    assert resolved.headers["content-type"].startswith("text/javascript")


def test_the_injection_leaves_the_pages_own_encoding_declaration_readable(
    proxied: TestClient,
) -> None:
    """The charset meta stays inside the 1024 bytes the sniff may read.

    The tags go in at the TOP of ``<head>`` — the earliest point the prefix
    global can be set, and therefore the only point certainly before the module
    graph it configures — which puts them ahead of JupyterLab's own
    ``<meta charset>``.
    """
    body = proxied.get(f"{PANEL}/lab").text

    assert 0 < body.lower().index("<meta charset") < 1024


def test_the_relayed_lab_csp_says_nothing_about_scripts(proxied: TestClient) -> None:
    """No CSP is rewritten anywhere, because none has to be.

    jupyter-server's own policy governs framing (``frame-ancestors``) and says
    nothing about script sources, so the injected tags run. Should a backend
    ever grow a ``script-src`` — or a ``default-src``, which script sources
    fall back to — the bar would silently stop loading rather than break the
    page, and the answer would be a deliberate decision about that header. It
    would NOT be a rewrite hidden in the proxy: relaxing a backend's own CSP on
    its behalf is not something this proxy does.
    """
    csp = proxied.get(f"{PANEL}/lab").headers.get("content-security-policy", "").lower()

    assert "script-src" not in csp
    assert "default-src" not in csp


def test_the_panels_own_json_is_not_injected(proxied: TestClient) -> None:
    """The gate reads the body's type, and the sidecar's API is not a document."""
    status = proxied.get(f"{PANEL}/api/status")

    assert status.status_code == 200
    assert LAB_BAR_MODULE not in status.text


def test_a_users_own_html_file_is_served_untouched(proxied: TestClient, notebook_env: Path) -> None:
    """The finding a content-type gate could not see.

    jupyter-server serves a user's ``.html`` file as ``text/html`` with a
    ``<head>`` — indistinguishable from the Lab page to anything that reads
    only headers. The bar in there would be a second chip inside the file
    viewer, a second event stream against the browser's connection cap, and the
    hub's stylesheet cascading into a document somebody else wrote.

    Asserted against the real sidecar because the premise is the sidecar's
    behaviour, not ours: the content type is checked here too, so this test
    fails if jupyter-server ever stops serving user HTML as HTML and the page
    gate silently stops being load-bearing.
    """
    report = notebook_env / "shared" / "notebooks" / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "<html><head><title>a user's own report</title></head><body>hi</body></html>",
        encoding="utf-8",
    )

    served = proxied.get(f"{PANEL}/files/report.html")

    assert served.status_code == 200
    assert served.headers["content-type"].startswith("text/html")
    assert LAB_BAR_MODULE not in served.text
    assert "__OSPREY_PREFIX__" not in served.text


@pytest.mark.parametrize("module", sorted(_module_closure(LAB_BAR_MODULE)))
def test_every_module_the_bar_imports_is_served_through_the_proxy(
    proxied: TestClient, module: str
) -> None:
    """The whole import graph answers, as JavaScript, from inside the panel path.

    The first unreachable module breaks the graph, and the page reports that as
    nothing at all: no bar, and a 404 in a console nobody is watching.
    """
    response = proxied.get(f"{PANEL}/terminal-static/js/{module}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert response.content == (STATIC_JS / module).read_bytes()


def test_the_stylesheets_the_bar_addresses_answer_on_the_panels_paths(
    proxied: TestClient, sidecar: JupyterSidecar
) -> None:
    """Both ``new URL(…, import.meta.url)`` hops land on the hub's own files.

    The module derives its two stylesheets from its own URL rather than from an
    injected literal, which is what makes the per-user mount free — and what
    makes the hops themselves unassertable anywhere but here, against the real
    served address.
    """
    module_url = f"{sidecar.base_url}terminal-static/js/{LAB_BAR_MODULE}"
    specifiers = _ASSET_URL_RE.findall((STATIC_JS / LAB_BAR_MODULE).read_text(encoding="utf-8"))

    assert len(specifiers) == 2

    resolved = {urljoin(module_url, specifier) for specifier in specifiers}
    assert resolved == {
        f"{sidecar.base_url}terminal-static/css/terminal.css",
        f"{sidecar.base_url}design-system/css/tokens.css",
    }

    for url in sorted(resolved):
        response = proxied.get(url)
        assert response.status_code == 200, url
        assert response.headers["content-type"].startswith("text/css"), url
