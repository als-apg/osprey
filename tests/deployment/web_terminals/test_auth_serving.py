"""The authenticated perimeter, served for real: nginx + the auth sidecar.

Every other test of this feature looks at one side of the perimeter. The render
tests string-match Jinja output; ``test_nginx_validate.py`` proves the fragment
is a config nginx *accepts*; the sidecar's own unit tests drive its routes in
process. None of them can see what the two halves do to a request *together*,
and that is where this feature's sharpest properties live — a header nginx
emits from a decoded variable, an identity fixed by which location a normalised
path lands in, a cookie nginx must strip before the upstream sees it. Those are
observable only by putting a real request through a real nginx into a real
sidecar and looking at what came out the other end.

So this module composes the stack from the *rendered* artifacts and serves it:

* the **stub upstream** container stands in for the per-user terminal apps. It
  is a stdlib echo server that answers on each user's rendered loopback port
  with a JSON transcript of what it received, which is how assertions about
  what reached the app (the stripped ``Cookie``, the POST body, the stripped
  ``/u/<user>`` prefix) are made against the *upstream's* view rather than the
  client's guess. It also owns the network namespace and publishes the port.
* **nginx** and the **sidecar** join that namespace with
  ``--network container:<stub>``. This is what makes the rendered config work
  unmodified: it dials ``127.0.0.1:<port>`` for every upstream (the per-user
  apps and ``/verify`` alike), which is only true if all three processes share
  one loopback. Publishing from the namespace owner keeps the entry point
  reachable from the host on macOS too, where ``--network host`` shares the
  Docker VM's namespace and not the Mac's.

Nothing here is hand-written that the deployment produces for itself. The nginx
config, its mount paths, the sidecar's argv, its non-secret environment and the
external origin are all read out of ``render_web_terminals()``'s output;
``.env.auth`` is written by the production credential functions
(``ensure_auth_session_secrets`` and ``set_auth_password``, what the deploy
preflight and ``osprey users passwd`` call) and handed to the container through
``--env-file``, matching the rendered ``env_file:``. Only the passwords
themselves are chosen here, because minting deliberately never returns one.
The one thing this file must do that compose would otherwise have done is
*interpolate*: a bare ``docker run`` resolves no ``${...}``, so
:func:`_resolved_env_flags` replays that substitution against the deployment's
env chain before either service is started.

PASSWORD MODE ONLY, deliberately. The OIDC handshake is proven in process
against a real signing provider in ``tests/services/auth_sidecar/test_oidc_flow.py``;
an IdP container would have to live inside this shared namespace to be
reachable at all, and would re-prove nothing.

WHAT THE SIDECAR IMAGE HERE IS, AND IS NOT
------------------------------------------
The deployed sidecar image (``templates/modules/web_terminals/auth_sidecar/Dockerfile``)
installs the whole ``osprey-framework`` distribution — scipy, playwright,
accelerator-toolbox and the rest — which is a multi-minute, multi-hundred-MB
build this file cannot afford: it runs in the ordinary unit lane. The harness
image below installs only the part of that dependency set the sidecar's import
closure actually needs, plus the repository's own ``src/`` bind-mounted.

The one property that distinction endangers is ``python-multipart``: Starlette
requires it for EVERY form body, urlencoded included, so a sidecar image
missing it serves a login page whose form can never be submitted, and no unit
test would notice. The harness therefore installs nothing of its own choosing —
:func:`_declared_specs` takes each package's requirement *specifier from the
distribution's own metadata*, the same declaration ``pip install
osprey-framework`` resolves for the real image, and fails loudly if one is not
declared. Drop ``python-multipart`` from ``pyproject.toml`` and this harness
loses it exactly as the deployed image would, and
:func:`test_login_post_parses_its_form_body_in_the_container` goes red. What
remains unproven here is the *built image* itself: no job builds
``templates/modules/web_terminals/auth_sidecar/Dockerfile``, so a break confined
to that Dockerfile would reach a deployment without a test noticing.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import socket
import subprocess
import textwrap
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import requires as distribution_requires
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from packaging.requirements import Requirement

from osprey.deployment.web_terminals.artifacts import web_artifacts_dir
from osprey.deployment.web_terminals.auth_credentials import (
    AUTH_ENV_FILENAME,
    ensure_auth_session_secrets,
    ensure_terminal_secrets,
    purge_auth_credentials,
    set_auth_password,
    terminal_secret_var,
)
from osprey.deployment.web_terminals.personas import env_var_suffix
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.interfaces._serving import free_port
from osprey.services.auth_sidecar.passwords import generation_tag, hash_password
from osprey.services.auth_sidecar.sessions import SESSION_COOKIE_NAME, SessionCodec
from osprey.utils.dotenv import (
    ENV_LOCAL_FILENAME,
    ENV_SHARED_FILENAME,
    merge_chain,
    parse_dotenv_file,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
# The osprey.* tree reached via /src is shim-backed: config/logger/connectors
# live in the osprey-connectors workspace member, so its source joins the
# mount and the PYTHONPATH — the distribution is not yet installable from
# PyPI inside the harness image.
_CONNECTORS_SRC_DIR = _REPO_ROOT / "packages" / "osprey-connectors" / "src"

_USERS = ("alice", "bob", "carol")
"""The roster. ``carol`` exists so the throttle test can burn a user's attempt
window without locking out the user every other test logs in as — the throttle
is per-user and process-local, and the stack is shared across this module."""

_PASSWORDS = {"alice": "alice-pw-correct", "bob": "bob-pw-correct", "carol": "carol-pw-correct"}

_SHARED_PROXY = "http://proxy.invalid:3128"
"""A site-wide egress proxy, written into ``.env.shared`` by the stack below.

``.env.shared`` is the committed half of the env chain — the file ``osprey
init`` writes the commented-out ``HTTP_PROXY``/``HTTPS_PROXY``/``NO_PROXY``
block into, because a proxy is a setting the whole site shares rather than one
host's secret. The sidecar reaches it through ``${HTTPS_PROXY:-}``
passthrough, which is worth nothing unless a value living in *that* file
survives the trip into the container; :func:`_resolved_env_flags` is what
carries it, and
:func:`test_sidecar_environment_carries_the_proxy_resolved_from_the_env_chain`
is what proves it arrived. Unreachable by construction (``.invalid`` is
reserved): password-mode serving makes no outbound request, so the value is
only ever read back, never dialled."""

_PACKAGES = (
    "fastapi",
    "uvicorn",
    "authlib",
    "itsdangerous",
    "python-multipart",
    "httpx",
    "jinja2",
    "ruamel.yaml",
    "pyyaml",
    "click",
    "rich",
)
"""The sidecar's import closure, by distribution name.

Found by running ``import osprey.services.auth_sidecar.app`` in a bare
``python:3.11-slim`` and adding what it asked for: the app itself needs the
first five, and ``osprey.deployment.web_terminals.personas`` drags the render
and registry packages behind it. The *versions* are never named here — see
:func:`_declared_specs`.
"""


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.dockerbuild,
    pytest.mark.skipif(not _docker_available(), reason="docker not available"),
]


# ---------------------------------------------------------------------------
# Harness image
# ---------------------------------------------------------------------------


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_specs() -> tuple[str, ...]:
    """The requirement specifiers ``osprey-framework`` declares for :data:`_PACKAGES`.

    Read from the *installed distribution's* metadata rather than written out
    here, so the harness image can only contain what a deployment's
    ``pip install osprey-framework`` would also install. That is what keeps
    :func:`test_login_post_parses_its_form_body_in_the_container` honest about
    ``python-multipart``: the harness has it because the distribution declares
    it, not because this file asked for it.

    Raises:
        AssertionError: If the distribution does not declare one of them. That
            is a real defect rather than a harness problem — the sidecar imports
            the package, so the deployed image would be missing it too.
    """
    declared: dict[str, str] = {}
    for raw in distribution_requires("osprey-framework") or ():
        requirement = Requirement(raw)
        # Extras (``; extra == "dev"``) are not in a default install, so they
        # are not in the deployed image either.
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        declared[_canonical(requirement.name)] = str(requirement)

    missing = [name for name in _PACKAGES if _canonical(name) not in declared]
    assert not missing, (
        f"osprey-framework does not declare {missing}, which the auth sidecar imports. "
        "The deployed sidecar image installs the distribution's declared dependencies "
        "and would be missing them too."
    )
    return tuple(declared[_canonical(name)] for name in _PACKAGES)


def _harness_dockerfile() -> str:
    specs = " ".join(f'"{spec}"' for spec in _declared_specs())
    return textwrap.dedent(
        f"""\
        FROM python:3.11-slim
        RUN pip install --no-cache-dir {specs}
        ENV PYTHONPATH=/src:/connectors-src PYTHONDONTWRITEBYTECODE=1
        """
    )


def _build_harness_image(tmp_path: Path) -> str:
    """Build (or reuse) the image the sidecar and the stub upstream run from.

    The tag carries a digest of the Dockerfile, so a dependency-declaration
    change produces a new tag instead of silently reusing a stale layer — and
    two sessions running concurrently share a tag only when they would have
    built identical content.
    """
    dockerfile = _harness_dockerfile()
    tag = f"osprey-auth-serving-harness:{hashlib.sha256(dockerfile.encode()).hexdigest()[:12]}"

    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0:
        return tag

    context = tmp_path / "harness"
    context.mkdir()
    (context / "Dockerfile").write_text(dockerfile)
    result = subprocess.run(
        ["docker", "build", "-t", tag, str(context)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, f"harness image build failed:\n{result.stdout}\n{result.stderr}"
    return tag


TERMINAL_STAND_IN_MARKER = "osprey-terminal-stand-in"
"""Element id the stub serves at the app root, for a browser to look for.

The real per-user terminal app cannot run here — it needs the ``claude``
binary and a provider — so the browser suite's "you arrived at the terminal"
assertion is against this marker. What that proves is the perimeter half (the
navigation reached the user's own upstream through nginx with a session); the
terminal UI's own rendering is covered by the in-process suites under
``tests/interfaces/web_terminal/``.
"""

_STUB_UPSTREAM = f'''\
"""Stand-in for the per-user terminal apps.

Serves an HTML page at the app root, the way the terminal would to a
navigating browser, and echoes a JSON transcript of the request everywhere
else, which is how the serving tests assert on what actually reached the app.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = (
    "<!doctype html><title>Terminal</title>"
    \'<div id="{TERMINAL_STAND_IN_MARKER}">terminal for port %d</div>\'
)


class Echo(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _respond(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        port = self.server.server_address[1]
        if self.command == "GET" and self.path == "/":
            payload = (PAGE % port).encode()
            content_type = "text/html; charset=utf-8"
        else:
            payload = json.dumps(
                {{
                    "port": port,
                    "method": self.command,
                    "path": self.path,
                    "headers": {{key.lower(): value for key, value in self.headers.items()}},
                    "body": body.decode("utf-8", "replace"),
                }}
            ).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = _respond
    do_POST = _respond

    def log_message(self, *args):
        pass


servers = [ThreadingHTTPServer(("127.0.0.1", int(port)), Echo) for port in sys.argv[1:]]
for server in servers[1:]:
    threading.Thread(target=server.serve_forever, daemon=True).start()
servers[0].serve_forever()
'''


# ---------------------------------------------------------------------------
# The composed stack
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stack:
    """Everything a test needs to talk to the running perimeter."""

    port: int
    origin: str
    session_secret: str
    stored_hashes: dict[str, str]
    upstream_ports: dict[str, int]
    deployment_root: Path
    auth_container: str
    auth_run_argv: list[str]

    @property
    def env_auth_path(self) -> Path:
        return self.deployment_root / AUTH_ENV_FILENAME

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def recreate_auth(self) -> None:
        """Replace the sidecar container, the way a credential change obliges.

        compose bakes ``env_file`` content into a container at *creation* time,
        so a deployment that rewrites ``.env.auth`` must force-recreate the
        sidecar rather than restart it. This reproduces that half: the container
        is removed and started again from the same argv, which re-reads the file
        as it now stands on disk. The signing secrets live in that same file and
        are untouched, so cookies minted before the recreate still verify — which
        is what makes "this session is refused *now*" mean something.
        """
        subprocess.run(["docker", "rm", "-f", self.auth_container], capture_output=True, timeout=60)
        started = subprocess.run(self.auth_run_argv, capture_output=True, text=True, timeout=180)
        assert started.returncode == 0, started.stderr
        _wait_for(
            f"{self.base_url}/auth/login?user=alice",
            names=[self.auth_container],
            what="the recreated auth sidecar",
        )

    def codec(self) -> SessionCodec:
        """A codec on the sidecar's own signing secret.

        Forging a *validly signed* cookie is the only way to reach the expiry
        and credential-generation checks from outside: both are properties of a
        cookie the sidecar already trusts the signature of, and neither can be
        produced by waiting (the rendered lifetime is twelve hours) or by
        rotating a password (which needs a container recreate).
        """
        return SessionCodec(self.session_secret)

    def client(self, **kwargs: Any) -> httpx.Client:
        """A client with a cookie jar, which is the only correct way to drive this.

        The sidecar sets two cookies on different paths (the OIDC state cookie
        on ``/auth``, the session cookie on ``/``); a hand-assembled ``Cookie``
        header sends the wrong one and reports a confident false green.
        """
        return httpx.Client(base_url=self.base_url, follow_redirects=False, timeout=15, **kwargs)


def _config(nginx_port: int) -> dict[str, Any]:
    """The deployment this module serves.

    ``deploy.fqdn`` is the loopback address on purpose: the rendered
    ``OSPREY_AUTH_EXTERNAL_ORIGIN`` is derived from it and the nginx port, and
    it is the origin the sidecar's login POST requires a submission to declare.
    Pointing it at where the test actually reaches the stack means that check
    is exercised against a *rendered* value rather than an overridden one.

    ``allow_insecure_http`` is the shape a facility behind its own TLS
    terminator deploys — one server block, no certificate fixture, and the same
    gated surface.
    """
    return {
        "facility": {"name": "Demo Light Source", "prefix": "dls", "timezone": "UTC"},
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "127.0.0.1"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "nginx_port": nginx_port,
                "users": list(_USERS),
                "auth": {"method": "password", "allow_insecure_http": True},
            }
        },
    }


_UPSTREAM_RE = re.compile(
    r"location /u/(?P<user>[^/ ]+)/ \{.*?proxy_pass http://127\.0\.0\.1:(?P<port>\d+)/;",
    re.DOTALL,
)


def _rendered_upstream_ports(nginx_conf: str) -> dict[str, int]:
    """The loopback port nginx will really dial for each user.

    Read back out of the config rather than recomputed from the port bases, so
    the stub upstream listens where nginx looks even if allocation changes.
    """
    return {m.group("user"): int(m.group("port")) for m in _UPSTREAM_RE.finditer(nginx_conf)}


def _docker_logs(name: str) -> str:
    result = subprocess.run(
        ["docker", "logs", "--tail", "40", name], capture_output=True, text=True
    )
    return f"--- {name} ---\n{result.stdout}\n{result.stderr}"


def _wait_for(url: str, *, names: list[str], what: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last = "no attempt made"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=5)
        except httpx.HTTPError as exc:  # not up yet
            last = repr(exc)
        else:
            if response.status_code == 200:
                return
            last = f"HTTP {response.status_code}"
        time.sleep(0.5)
    logs = "\n".join(_docker_logs(name) for name in names)
    raise AssertionError(f"{what} never became ready ({last})\n{logs}")


_COMPOSE_INTERPOLATION = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
)
"""The ``${VAR}`` / ``${VAR:-default}`` references compose resolves for itself."""


def _resolved_env_flags(service: dict[str, Any], chain_env: Mapping[str, str]) -> list[str]:
    """A rendered service's ``environment:`` block as ``docker run -e`` flags.

    A **docker-run replay of compose interpolation**, and the one place in this
    file that performs one. Compose resolves every ``${...}`` in a service
    definition at container-create time, from the env chain its lifecycle verbs
    pass as ``--env-file``; a bare ``docker run`` resolves nothing, so an
    unreplayed entry reaches the container as the literal characters
    ``${VAR:-}`` and every assertion about that variable is then about a
    placeholder rather than about the value a deployment would deliver.

    Both interpolating services come through here. nginx carries one
    ``${OSPREY_TERMINAL_SECRET_<user>:-}`` per roster user; the auth sidecar
    carries the OIDC egress passthrough (``HTTP_PROXY``, ``HTTPS_PROXY``,
    ``NO_PROXY``). One resolver for the two, because a second implementation
    would be a second thing to keep true — and the sidecar's variables were the
    ones that had no replay at all until this helper covered both.

    Resolution order, matching what compose does with the chain:

    1. the **merged env chain** — ``.env.shared`` under ``.env``, the later file
       winning (:func:`osprey_connectors.dotenv.merge_chain`, reached here
       through the ``osprey.utils.dotenv`` re-export shim). Resolving against
       ``.env`` alone would silently drop every value an operator put in the
       committed half of the chain, which is exactly where a site-wide proxy
       lives;
    2. the entry's own ``:-`` default, when the chain has no non-empty value —
       compose's ``:-`` fires on unset *and* on empty;
    3. the empty string, for a reference that declares no default.

    One deliberate divergence: compose would consult the shell environment ahead
    of the chain, and this replay does not, so the resolved value is a property
    of the deployment's files alone and a proxy exported on the CI host cannot
    change what the assertions see.

    :param service: A rendered compose service definition.
    :param chain_env: The deployment's merged env chain.
    :return: Alternating ``-e`` / ``NAME=value`` docker-run arguments.
    :raises AssertionError: If a resolved value still contains a ``$`` — an
        interpolation shape this replay does not implement (a bare ``$VAR``, a
        ``$$`` escape, or a nested ``${...}``). ``$`` is the whole class rather
        than ``${`` alone because production refuses any chain value carrying
        one (``compose_unsafe_vars_in_chain``), so no legitimate resolved value
        has one to lose. Failing here is the point: the alternative is a
        container that reads the placeholder as a literal value and a green test
        that means nothing.
    """

    def _substitute(match: re.Match[str]) -> str:
        return chain_env.get(match["name"], "") or (match["default"] or "")

    flags: list[str] = []
    for entry in service.get("environment") or []:
        name, _, raw = entry.partition("=")
        value = _COMPOSE_INTERPOLATION.sub(_substitute, raw)
        assert "$" not in value, (
            f"{name} still carries an unresolved interpolation after the replay: {value!r}"
        )
        flags += ["-e", f"{name}={value}"]
    return flags


@contextmanager
def serving_stack(tmp_path: Path) -> Iterator[Stack]:
    """Bring the rendered perimeter up, and take it down again.

    A context manager rather than a fixture so the browser suite in
    ``tests/interfaces/web_terminal/test_auth_login_browser.py`` can drive the
    same stack: it needs exactly this perimeter (real nginx, real sidecar, real
    rendered config) with a Chromium page pointed at it instead of httpx, and a
    second implementation of it would be a second thing to keep true.
    """
    image = _build_harness_image(tmp_path)

    # `.env.auth` is written by the production credential functions, not
    # hand-rolled: `ensure_auth_session_secrets` is what a deploy preflight
    # calls, and `set_auth_password` is what `osprey users passwd` calls. The
    # sidecar then reads the real file through `--env-file`, matching the
    # rendered `env_file: .env.auth`. Known passwords rather than the minted
    # ones because `ensure_auth_credentials` deliberately never returns them.
    #
    # It lives outside the per-attempt render directory so a port retry cannot
    # re-mint the secrets underneath a container that already read them.
    deployment_root = tmp_path / "deployment"
    deployment_root.mkdir()
    ensure_auth_session_secrets(deployment_root)
    for user in _USERS:
        set_auth_password(user, _PASSWORDS[user], deployment_root)

    # The per-user operator secret nginx stamps onto every proxied request, minted
    # into the deploy `.env` by the same preflight function `osprey up` calls — not
    # hand-rolled, for the same reason the auth secrets above are not. nginx.conf
    # `include`s one envsubst'd snippet per authenticated location, and nginx
    # treats a missing include as a hard startup failure, so the container never
    # becomes ready until these exist and are handed to it (below).
    ensure_terminal_secrets(deployment_root, _USERS)
    env_local = parse_dotenv_file(deployment_root / ENV_LOCAL_FILENAME)
    terminal_secrets = {user: env_local[terminal_secret_var(user)] for user in _USERS}

    # The committed half of the chain, which no credential function writes: a
    # site-wide proxy is the operator-authored shape `osprey init` documents
    # there, and it is the value the sidecar's `${HTTPS_PROXY:-}` passthrough
    # has to survive the trip with. Written before the merge below, because
    # that merge is what every `${...}` in the rendered stack resolves against.
    (deployment_root / ENV_SHARED_FILENAME).write_text(f"HTTPS_PROXY={_SHARED_PROXY}\n")

    # This deployment's env chain as compose would read it: `.env.shared` under
    # `.env`, the local file winning on any key both set.
    chain_env = merge_chain(deployment_root)

    env_auth = parse_dotenv_file(deployment_root / AUTH_ENV_FILENAME)
    session_secret = env_auth["OSPREY_AUTH_SESSION_SECRET"]
    stored_hashes = {
        user: env_auth[f"OSPREY_AUTH_PW_HASH_{env_var_suffix(user)}"] for user in _USERS
    }

    token = uuid.uuid4().hex[:8]
    names = [
        f"osprey-authserving-nginx-{token}",
        f"osprey-authserving-auth-{token}",
        f"osprey-authserving-stub-{token}",
    ]
    nginx_name, auth_name, stub_name = names

    try:
        # A port free a moment ago can be taken before docker binds it (the
        # window is real: rendering and three container starts sit inside it),
        # and the daemon reports that as an opaque failure. Re-render on a
        # fresh port rather than turning a lost race into a red test.
        for attempt in range(3):
            nginx_port = free_port()
            artifacts = render_web_terminals(_config(nginx_port), terminal_secrets=terminal_secrets)
            # Lay the rendered artifacts out the way the real writer does —
            # under the repo's build/ zone — while `project` below plays the
            # pinned compose project directory the mount sources resolve
            # against. The two must agree or nginx starts with no config.
            project = tmp_path / f"project-{attempt}"
            for relative_path, content in artifacts.items():
                target = web_artifacts_dir(project) / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            (project / "stub_upstream.py").write_text(_STUB_UPSTREAM)

            upstream_ports = _rendered_upstream_ports(artifacts["nginx/nginx.conf"])
            assert set(upstream_ports) == set(_USERS), upstream_ports

            # The stub owns the namespace and the published port; nginx and the
            # sidecar join it, which is what makes the rendered loopback
            # `proxy_pass` targets resolve.
            started = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    stub_name,
                    "-p",
                    f"127.0.0.1:{nginx_port}:{nginx_port}",
                    "-v",
                    f"{project / 'stub_upstream.py'}:/stub_upstream.py:ro",
                    image,
                    "python",
                    "/stub_upstream.py",
                    *[str(upstream_ports[user]) for user in _USERS],
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if started.returncode == 0:
                break
            subprocess.run(["docker", "rm", "-f", stub_name], capture_output=True, timeout=60)
        else:
            raise AssertionError(f"the stub upstream never started:\n{started.stderr}")

        compose = yaml.safe_load(artifacts["docker-compose.web.yml"])
        nginx_service = compose["services"]["nginx"]
        auth_service = compose["services"]["auth"]

        external_origin = next(
            value.split("=", 1)[1]
            for value in auth_service["environment"]
            if value.startswith("OSPREY_AUTH_EXTERNAL_ORIGIN=")
        )
        assert external_origin == f"http://127.0.0.1:{nginx_port}", external_origin

        # The sidecar's environment is mostly literal, but the egress
        # passthrough (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`) is `${VAR:-}`
        # that compose would resolve from the deploy env chain. Handing those
        # through verbatim would set the variables to the literal placeholder
        # inside the container — a proxy URL of `${HTTPS_PROXY:-}`, which is
        # worse than unset. Same replay as nginx's, same helper.
        auth_env = _resolved_env_flags(auth_service, chain_env)
        # Kept as an argv the Stack can replay: a credential change obliges a
        # force-recreate, and the recreated container has to be the same one.
        auth_run_argv = [
            "docker",
            "run",
            "-d",
            "--name",
            auth_name,
            "--network",
            f"container:{stub_name}",
            "-v",
            f"{_SRC_DIR}:/src:ro",
            "-v",
            f"{_CONNECTORS_SRC_DIR}:/connectors-src:ro",
            "--env-file",
            str(deployment_root / AUTH_ENV_FILENAME),
            *auth_env,
            image,
            *auth_service["command"],
        ]
        auth_started = subprocess.run(auth_run_argv, capture_output=True, text=True, timeout=180)
        assert auth_started.returncode == 0, auth_started.stderr

        # Bind mounts come off the service's `volumes:` as "src:dst[:opts]"
        # strings; the envsubst output directory comes off its own `tmpfs:` key
        # as a "<path>[:opts]" string. They become different docker-run flags, so
        # they are read separately — both from the rendered service rather than
        # hardcoded, so a stack that stopped declaring either would fail here
        # exactly as a real `osprey up` would. The tmpfs entry's option string is
        # replayed verbatim: `mode=0700` is what compose hands the runtime, and
        # what the runtime hands the kernel.
        nginx_mounts: list[str] = []
        for volume in nginx_service["volumes"]:
            host_side, container_side = volume.split(":", 1)
            nginx_mounts += ["-v", f"{(project / host_side).resolve()}:{container_side}"]
        nginx_tmpfs: list[str] = []
        for entry in nginx_service.get("tmpfs") or []:
            nginx_tmpfs += ["--tmpfs", entry]

        # The rendered compose declares the templates mount and the envsubst
        # environment on the nginx service, but its per-user secret values are
        # `${OSPREY_TERMINAL_SECRET_<user>:-}` — resolved by compose from the
        # deploy env chain, which a bare `docker run` does not do. The replay
        # does it here (the minted secrets are in that chain), so the
        # entrypoint envsubsts each snippet with the real secret.
        nginx_env = _resolved_env_flags(nginx_service, chain_env)

        # The envsubst output directory (NGINX_ENVSUBST_OUTPUT_DIR=/etc/nginx/osprey)
        # must exist and be writable before the base image's entrypoint runs, or
        # its own guard silently skips the substitution and nginx then dies on the
        # missing include. Its restrictive mode is carried through from the
        # rendered `tmpfs:` entry: the substituted snippets hold every roster
        # user's operator secret, and this container is the real one.
        assert nginx_tmpfs, "the rendered nginx service declares no envsubst tmpfs"

        nginx_started = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                nginx_name,
                "--network",
                f"container:{stub_name}",
                *nginx_mounts,
                *nginx_tmpfs,
                *nginx_env,
                nginx_service["image"],
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert nginx_started.returncode == 0, nginx_started.stderr

        base_url = f"http://127.0.0.1:{nginx_port}"
        # The landing page is nginx's own healthcheck target and needs no
        # session; the login page is the first thing that requires the sidecar.
        _wait_for(f"{base_url}/", names=names, what="nginx")
        _wait_for(f"{base_url}/auth/login?user=alice", names=names, what="the auth sidecar")

        yield Stack(
            port=nginx_port,
            origin=external_origin,
            session_secret=session_secret,
            stored_hashes=stored_hashes,
            upstream_ports=upstream_ports,
            deployment_root=deployment_root,
            auth_container=auth_name,
            auth_run_argv=auth_run_argv,
        )
    finally:
        # Unconditionally, and by name: a `docker run` that fails to set up
        # networking still leaves the container behind, and the next attempt
        # would collide with its name.
        for name in names:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    """The rendered perimeter, running.

    Module-scoped: three containers and an image build are far too expensive
    per test, and every test below either logs in with its own client (so the
    cookie jars do not interact) or acts on a user nothing else touches.
    """
    with serving_stack(tmp_path_factory.mktemp("auth-serving")) as running:
        yield running


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_request(port: int, target: str, headers: dict[str, str]) -> tuple[bytes, bytes]:
    """Send a request line verbatim over a socket and return (head, body).

    curl and httpx both normalise a request target before it leaves the client,
    which silently rewrites away the very inputs the injection probes below are
    about. Nothing between this and nginx touches the bytes.
    """
    lines = [f"GET {target} HTTP/1.1", *(f"{key}: {value}" for key, value in headers.items())]
    payload = ("\r\n".join([*lines, "", ""])).encode()

    chunks: list[bytes] = []
    with socket.create_connection(("127.0.0.1", port), timeout=15) as sock:
        sock.sendall(payload)
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    head, _, body = b"".join(chunks).partition(b"\r\n\r\n")
    return head, body


def _header_lines(head: bytes, name: str) -> list[bytes]:
    """Every header line in ``head`` with the given field name.

    Split on CRLF only, so a bare LF smuggled into a header *value* stays
    inside the line it corrupted instead of being read as a line break — which
    is exactly the difference the probes below assert on.
    """
    prefix = f"{name.lower()}:".encode()
    return [line for line in head.split(b"\r\n") if line.lower().startswith(prefix)]


def _login(
    stack: Stack, client: httpx.Client, user: str, password: str | None = None
) -> httpx.Response:
    """Submit the login form the way the login page's own form would."""
    return client.post(
        "/auth/login",
        data={"user": user, "password": _PASSWORDS[user] if password is None else password},
        # Browsers send this on every POST and page script cannot set it; the
        # sidecar refuses a submission that declares no origin at all.
        headers={"Origin": stack.origin, "Content-Type": "application/x-www-form-urlencoded"},
    )


def _navigation() -> dict[str, str]:
    """The ``Accept`` a browser sends when a person navigates to a URL."""
    return {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


# ---------------------------------------------------------------------------
# 1-2. What the login redirect may carry
# ---------------------------------------------------------------------------


def test_trailing_newline_in_path_cannot_reach_the_location_header(stack: Stack) -> None:
    """A path ending in ``%0a`` must not put a newline in ``Location``.

    nginx percent-*decodes* ``$uri``, so by the time the allowlist map reads it
    the escape is a real LF and ``return 302`` would copy it into the header
    verbatim. PCRE's ``$`` also matches immediately before a trailing newline,
    so an allowlist anchored with ``$`` instead of ``\\z`` matches this path,
    emits it, and appends whatever follows the newline as response headers of
    the attacker's choosing. This exact request is the one case that tells the
    two anchors apart, and it only survives the trip over a raw socket.
    """
    # Act
    head, _ = _raw_request(
        stack.port,
        "/u/alice/x%0a",
        {"Host": f"127.0.0.1:{stack.port}", "Connection": "close", **_navigation()},
    )

    # Assert
    assert head.startswith(b"HTTP/1.1 302"), head
    locations = _header_lines(head, "Location")
    assert len(locations) == 1, head
    location = locations[0]
    assert b"\n" not in location, location
    assert location.endswith(b"next="), location


def test_crlf_in_path_cannot_inject_a_response_header(stack: Stack) -> None:
    """``%0d%0a`` in the path must not become a ``Set-Cookie`` on this origin.

    A cookie planted here would be planted on the very origin the terminals
    live on, by a link most effective against a logged-out victim — which is
    precisely who follows a login link.
    """
    # Act
    head, _ = _raw_request(
        stack.port,
        "/u/alice/x%0d%0aSet-Cookie:%20evil=1",
        {"Host": f"127.0.0.1:{stack.port}", "Connection": "close", **_navigation()},
    )

    # Assert
    assert b"set-cookie" not in head.lower(), head
    assert b"evil=1" not in head, head


# ---------------------------------------------------------------------------
# 3. Redirects stay relative
# ---------------------------------------------------------------------------


def test_bookmark_redirect_is_relative_and_drops_no_port(stack: Stack) -> None:
    """``/u/alice`` must redirect to exactly ``/u/alice/``.

    Left at nginx's default, ``absolute_redirect`` rebuilds a relative target
    as ``$host`` plus nginx's *own* listening port — which walks a browser that
    arrived through a facility TLS terminator onto cleartext on an internal
    port. The ``Host`` here carries a port precisely because that is the input
    the default reads.
    """
    # Act
    with stack.client() as client:
        response = client.get("/u/alice", headers={"Host": f"127.0.0.1:{stack.port}"})

    # Assert
    assert response.status_code == 301
    assert response.headers["location"] == "/u/alice/"


# ---------------------------------------------------------------------------
# 4. What a denied request looks like
# ---------------------------------------------------------------------------


def test_navigation_without_a_session_is_sent_to_the_login_page(stack: Stack) -> None:
    """A person typing a URL should land on the prompt, naming their own user."""
    # Act
    with stack.client() as client:
        response = client.get("/u/bob/", headers=_navigation())

    # Assert
    assert response.status_code == 302
    assert response.headers["location"] == "/auth/login?user=bob&next=/u/bob/"


@pytest.mark.parametrize(
    ("label", "headers"),
    [
        ("fetch/json", {"Accept": "application/json"}),
        ("eventsource", {"Accept": "text/event-stream"}),
        (
            # A WebSocket handshake carries `Accept: text/html` from the page
            # that opened it; the Upgrade token in front of it is what has to
            # push the match off the anchor.
            "websocket",
            {
                "Accept": "text/html,application/xhtml+xml",
                "Upgrade": "websocket",
                "Connection": "Upgrade",
                "Sec-WebSocket-Version": "13",
                # RFC 6455's own example nonce, not a secret.
                "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",  # gitleaks:allow
            },
        ),
    ],
)
def test_programs_get_a_bare_401_not_a_login_page(
    stack: Stack, label: str, headers: dict[str, str]
) -> None:
    """Anything a program issued must get the status, never a page of HTML.

    A JSON caller handed login HTML fails as a parse error far from the cause,
    and an EventSource reconnects forever against a page that will never become
    the stream it asked for.

    What is asserted is that this is a refusal and not a hand-off: no
    ``Location``, and nothing of the login form in the body. nginx's own 401
    body is a stock HTML page, which is fine — the caller reads the status, and
    a program that followed a redirect into a login page would not.
    """
    # Act
    with stack.client() as client:
        response = client.get("/u/alice/", headers=headers)

    # Assert
    assert response.status_code == 401, label
    assert "location" not in response.headers, label
    assert "<form" not in response.text.lower(), label
    assert "/auth/login" not in response.text, label


# ---------------------------------------------------------------------------
# 5. One user's session unlocks one user
# ---------------------------------------------------------------------------


def test_a_session_unlocks_its_own_user_and_nobody_else(stack: Stack) -> None:
    """alice's cookie opens alice's terminal and is refused at bob's."""
    # Arrange
    with stack.client() as client:
        landed = _login(stack, client, "alice")
        assert landed.status_code == 303, landed.text
        assert landed.headers["location"] == "/u/alice/"

        # Act
        allowed = client.get("/u/alice/", headers=_navigation())
        refused = client.get("/u/bob/", headers=_navigation())

    # Assert
    assert allowed.status_code == 200
    assert refused.status_code == 302
    assert refused.headers["location"].startswith("/auth/login?user=bob")


def test_login_post_from_a_foreign_origin_is_refused_before_anything_else(stack: Stack) -> None:
    """A form posted from somewhere else is not a login attempt at all.

    Without this, an attacker's page could post its own credentials into a
    victim's browser and log that browser into the *attacker's* roster user —
    every later action in that tab then belongs to the attacker's terminal.

    The refusal has to land before the form is read and before the throttle, so
    two things are asserted beyond the 400: no cookie was set, and the window
    was not consumed — the immediately following correct login still succeeds
    rather than coming back 429. ``bob`` is used because nothing else in this
    module ever fails his password, so a 429 here could only have come from
    this test.
    """
    # Act
    with stack.client() as client:
        foreign = client.post(
            "/auth/login",
            data={"user": "bob", "password": _PASSWORDS["bob"]},
            headers={
                "Origin": "https://operator-portal.evil.example.org",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        legitimate = _login(stack, client, "bob")

    # Assert
    assert foreign.status_code == 400
    assert "set-cookie" not in foreign.headers
    assert legitimate.status_code == 303, legitimate.text


def test_two_browsers_on_one_user_get_independent_sessions(stack: Stack) -> None:
    """Two clients logging in as the same user both work, and do not collide.

    The perimeter keys authorization off the cookie a request carries, not off
    any per-user server-side slot — so a second browser must neither be refused
    nor take the first one's session over. An operator with a desk machine and a
    laptop is the ordinary case, not an edge one.
    """
    # Act
    with stack.client() as first, stack.client() as second:
        _login(stack, first, "alice")
        _login(stack, second, "alice")

        first_view = first.get("/u/alice/", headers=_navigation())
        second_view = second.get("/u/alice/", headers=_navigation())

        # Assert
        assert first.cookies[SESSION_COOKIE_NAME] != second.cookies[SESSION_COOKIE_NAME]
        assert first_view.status_code == 200
        assert second_view.status_code == 200


def test_login_cookie_is_httponly_lax_and_root_scoped(stack: Stack) -> None:
    """Asserted on the raw header: ``response.cookies`` normalises the
    attributes away, and each of them is load-bearing.

    ``Secure`` must be absent under ``allow_insecure_http`` — a Secure cookie
    would never be sent back over the cleartext hop this posture exists for.
    """
    # Act
    with stack.client() as client:
        response = _login(stack, client, "alice")

    # Assert
    raw = response.headers["set-cookie"]
    assert raw.startswith(f"{SESSION_COOKIE_NAME}=")
    assert "HttpOnly" in raw
    assert "SameSite=lax" in raw
    assert "Path=/" in raw
    assert "Secure" not in raw


def test_dot_segment_path_confusion_is_authorized_as_the_user_it_reaches(stack: Stack) -> None:
    """``/u/alice/%2e%2e/bob/`` must be decided as *bob*, and refused.

    nginx decodes and collapses the dot segments before it picks a location, so
    this request routes into bob's block and takes bob's ``auth_request``. The
    failure this rules out is the opposite: a stack that matched alice's
    location (and so alice's authorization) while proxying to bob's upstream.
    Sent raw because httpx would resolve the dot segments client-side and never
    put this target on the wire.
    """
    # Arrange
    with stack.client() as client:
        _login(stack, client, "alice")
        cookie = client.cookies[SESSION_COOKIE_NAME]

    # Act — one hand-built Cookie header, deliberately: this probe cannot go
    # through the jar, and the session cookie is the only one scoped to "/".
    head, _ = _raw_request(
        stack.port,
        "/u/alice/%2e%2e/bob/",
        {
            "Host": f"127.0.0.1:{stack.port}",
            "Cookie": f"{SESSION_COOKIE_NAME}={cookie}",
            "Connection": "close",
            **_navigation(),
        },
    )

    # Assert
    assert head.startswith(b"HTTP/1.1 302"), head
    locations = _header_lines(head, "Location")
    assert locations == [b"Location: /auth/login?user=bob&next=/u/bob/"], head


# ---------------------------------------------------------------------------
# 6-7. What reaches the upstream
# ---------------------------------------------------------------------------


def test_authenticated_post_carries_its_body_to_the_upstream(stack: Stack) -> None:
    """An authenticated POST completes, and its body arrives unaltered.

    A POST is the one request shape that has to survive *two* trips through
    nginx — the ``auth_request`` subrequest and then the proxied request
    itself — and half a megabyte is well past the socket buffers either could
    hide a stall behind. If the subrequest and the proxied request ever
    disagreed about the body, this is where it would show as a hang.

    HONEST LIMIT: this does not discriminate ``proxy_pass_request_body off``.
    That directive protects against an auth backend that waits for the body it
    was told to expect; uvicorn answers the subrequest without reading one, so
    removing the directive leaves this test green. It is defence for the
    upstreams that do wait, and this harness cannot stand in for them.
    """
    # Arrange
    payload = '{"content": "' + "x" * (512 * 1024) + '"}'
    with stack.client() as client:
        _login(stack, client, "alice")

        # Act
        response = client.post(
            "/u/alice/api/chat",
            content=payload.encode(),
            headers={"Content-Type": "application/json"},
        )

    # Assert
    assert response.status_code == 200, response.text
    echoed = response.json()
    assert echoed["method"] == "POST"
    assert echoed["body"] == payload
    assert echoed["port"] == stack.upstream_ports["alice"]
    # The trailing slashes on the location and the proxy_pass target strip the
    # prefix, so the app never sees /u/alice.
    assert echoed["path"] == "/api/chat"


def test_upstream_receives_no_cookie_header(stack: Stack) -> None:
    """The browser's cookie jar stops at nginx.

    One jar serves this whole origin, so an operator with several users
    unlocked would otherwise hand alice's container a cookie naming all of
    them — inside a container that runs agent-generated code, which makes
    anything readable there a credential that has left the perimeter. Asserted
    on what the upstream *received*, which is the only place the difference
    exists.
    """
    # Arrange
    with stack.client() as client:
        _login(stack, client, "alice")
        _login(stack, client, "bob")
        assert SESSION_COOKIE_NAME in client.cookies

        # Act
        response = client.get("/u/alice/panel", headers={"X-Probe": "cookie-stripping"})

    # Assert
    assert response.status_code == 200
    received = response.json()["headers"]
    assert "cookie" not in received, received
    # A guard against a vacuous green: the request really did carry headers
    # through, so the missing Cookie is a strip rather than a lost request.
    assert received["x-probe"] == "cookie-stripping"


# ---------------------------------------------------------------------------
# 8. Sessions end
# ---------------------------------------------------------------------------


def test_expired_entry_no_longer_authorizes(stack: Stack) -> None:
    """A signed cookie whose entry has lapsed is refused.

    Forged rather than waited for: the rendered lifetime is twelve hours, and
    the property under test is what ``verify`` does with a cookie whose
    signature it *accepts*.
    """
    # Arrange
    codec = stack.codec()
    now = codec.now()
    state = codec.new_state().with_user(
        "alice",
        expires_at=now - 1,
        generation_tag=generation_tag(stack.stored_hashes["alice"]),
    )

    # Act
    with stack.client(cookies={SESSION_COOKIE_NAME: codec.encode(state)}) as client:
        response = client.get("/u/alice/", headers={"Accept": "application/json"})

    # Assert
    assert response.status_code == 401


def test_rotated_password_retires_sessions_minted_against_the_old_one(stack: Stack) -> None:
    """The credential-generation tag is what makes ``osprey users passwd`` bite.

    Rotation changes the stored hash, so every session carrying the old hash's
    tag stops verifying — with no server-side session state, and surviving the
    container recreate that rotation performs. The control case in the same
    test is what proves the refusal is the tag and not the forgery.
    """
    # Arrange
    codec = stack.codec()
    fresh = codec.now() + 3600
    stale = codec.new_state().with_user(
        "alice", expires_at=fresh, generation_tag=generation_tag(hash_password("a-rotated-secret"))
    )
    current = codec.new_state().with_user(
        "alice", expires_at=fresh, generation_tag=generation_tag(stack.stored_hashes["alice"])
    )

    # Act
    with stack.client(cookies={SESSION_COOKIE_NAME: codec.encode(stale)}) as client:
        rotated_out = client.get("/u/alice/", headers={"Accept": "application/json"})
    with stack.client(cookies={SESSION_COOKIE_NAME: codec.encode(current)}) as client:
        control = client.get("/u/alice/", headers={"Accept": "application/json"})

    # Assert
    assert rotated_out.status_code == 401
    assert control.status_code == 200


def test_logout_locks_one_user_and_leaves_the_others_unlocked(stack: Stack) -> None:
    """Logging alice out of a browser that also has bob unlocked keeps bob in."""
    # Arrange
    with stack.client() as client:
        _login(stack, client, "alice")
        _login(stack, client, "bob")

        # Act
        logout = client.get("/auth/logout?user=alice")
        alice_after = client.get("/u/alice/", headers={"Accept": "application/json"})
        bob_after = client.get("/u/bob/", headers={"Accept": "application/json"})

    # Assert
    assert logout.status_code == 303
    assert alice_after.status_code == 401
    assert bob_after.status_code == 200


def test_decommissioning_a_user_kills_their_live_session(stack: Stack) -> None:
    """Removing a user's credential must refuse the session they already hold.

    This is the claim behind decommissioning, and it is only observable in
    containers: the mocked lifecycle tests can assert the argv of a
    force-recreate, but not that the sidecar comes back having re-read
    ``.env.auth`` and now denies a cookie it accepted a moment ago. The purge is
    done by the production function, so a change that stopped removing the hash
    would surface here rather than being re-implemented by the test.

    The signing secrets survive in the same file, so the cookie is still
    *validly signed* across the recreate — the refusal is the missing
    credential, not a broken signature. bob is checked alongside to show the
    recreate did not simply lock everyone out.
    """
    # Arrange — the exact bytes, restored verbatim afterwards: re-running
    # `set_auth_password` would mint a NEW hash, and every other test's view of
    # alice's credential generation would silently go stale.
    original = stack.env_auth_path.read_bytes()
    with stack.client() as alice, stack.client() as bob:
        _login(stack, alice, "alice")
        _login(stack, bob, "bob")
        assert alice.get("/u/alice/", headers=_navigation()).status_code == 200

        try:
            # Act
            assert purge_auth_credentials("alice", stack.deployment_root) is True
            stack.recreate_auth()

            # Assert
            after = alice.get("/u/alice/", headers={"Accept": "application/json"})
            survivor = bob.get("/u/bob/", headers={"Accept": "application/json"})
            assert after.status_code == 401
            assert survivor.status_code == 200
        finally:
            stack.env_auth_path.write_bytes(original)
            stack.recreate_auth()


def test_repeated_failures_are_throttled_before_the_credential_is_evaluated(
    stack: Stack,
) -> None:
    """A second attempt inside the window is refused 429 — even a correct one.

    That last part is the property: the throttle is consulted *before* the
    credential is looked at, so a flood of guesses costs a dict lookup rather
    than a key derivation. ``carol`` exists only for this test, because the
    window it opens is per-user and outlives the request.
    """
    # Arrange / Act
    with stack.client() as client:
        first = _login(stack, client, "carol", password="wrong-1")
        second = _login(stack, client, "carol", password="wrong-2")
        # The right password, inside the window the wrong ones opened.
        third = _login(stack, client, "carol")

    # Assert
    assert first.status_code == 401
    assert second.status_code == 429
    assert int(second.headers["retry-after"]) >= 1
    assert third.status_code == 429
    assert "set-cookie" not in third.headers


# ---------------------------------------------------------------------------
# 9. The image can actually parse the form it serves
# ---------------------------------------------------------------------------


def test_login_post_parses_its_form_body_in_the_container(stack: Stack) -> None:
    """A real urlencoded login POST, parsed by the sidecar's own container.

    Starlette asserts on ``python-multipart`` for *every* form body, urlencoded
    included, so an image built from a narrower dependency set than the dev venv
    serves a login page whose form 500s on submission with a fully green unit
    suite behind it. The harness image installs only what the distribution
    declares (see :func:`_declared_specs`), so this passes because the
    declaration is there — not because the test asked for the package.
    """
    # Act
    with stack.client() as client:
        response = _login(stack, client, "alice")

        # Assert
        assert response.status_code == 303, response.text
        assert response.headers["location"] == "/u/alice/"
        assert client.get("/u/alice/", headers=_navigation()).status_code == 200


# ---------------------------------------------------------------------------
# 10. The docker-run replay of compose interpolation
# ---------------------------------------------------------------------------


def test_sidecar_environment_carries_the_proxy_resolved_from_the_env_chain(stack: Stack) -> None:
    """A site-wide `HTTPS_PROXY` in `.env.shared` reaches the running sidecar.

    The sidecar's only outbound traffic is the OIDC discovery/token/userinfo
    fetches, and on a site behind a corporate proxy they fail unless the host's
    proxy settings are in the container's environment. The compose definition
    carries them as `${HTTPS_PROXY:-}` — a reference, not a value — so what has
    to be proven is the whole path: a value an operator wrote in the *committed*
    half of the env chain, merged the way compose merges it, resolved by
    :func:`_resolved_env_flags`, and read back out of the process environment of
    the container that is actually serving. Reading it back with `os.environ`
    rather than `docker inspect` is deliberate: inspect shows what was
    requested, `os.environ` is what httpx will read.

    The empty pair matters as much as the set one. `${HTTP_PROXY:-}` with
    nothing in the chain must arrive as a variable that is SET and empty, which
    is how a proxy-less scheme is spelled — not as the literal `${HTTP_PROXY:-}`,
    which httpx would read as a proxy URL and fail every fetch against.
    """
    # Arrange
    probe = (
        "import json, os; "
        "print(json.dumps({name: os.environ.get(name) "
        "for name in ('HTTPS_PROXY', 'HTTP_PROXY', 'NO_PROXY')}))"
    )

    # Act
    result = subprocess.run(
        ["docker", "exec", stack.auth_container, "python", "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Assert
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    assert seen["HTTPS_PROXY"] == _SHARED_PROXY, seen
    assert seen["HTTP_PROXY"] == "", seen
    assert seen["NO_PROXY"] == "", seen
    # And the resolution happened in the replay rather than by accident: the
    # argv the container was created from carries the value, not the reference.
    assert f"HTTPS_PROXY={_SHARED_PROXY}" in stack.auth_run_argv, stack.auth_run_argv


def test_no_argument_handed_to_docker_run_survives_as_an_interpolation(stack: Stack) -> None:
    """Nothing in the sidecar's argv still reads `${...}`.

    :func:`_resolved_env_flags` refuses one value at a time; this is the same
    property over the whole assembled command line, which also covers the
    arguments that do not come from `environment:` — the mounts, the
    `--env-file` path and the rendered `command`. A placeholder that reaches
    `docker run` is not an error there: it becomes a literal value in the
    container, and every test above it goes green while asserting on a string
    no deployment would ever have produced. nginx is held to the same rule by
    the helper itself, on the way to its own `docker run`.
    """
    # Assert
    unresolved = [argument for argument in stack.auth_run_argv if "${" in argument]
    assert not unresolved, unresolved
