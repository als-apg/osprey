"""The bluesky-web sidecar's roster grant, proven at its HTTP gate.

In a multi-user deployment each per-user web terminal proxies its BLUESKY tab
into the shared ``bluesky_web`` sidecar with the operator secret ITS container
holds — ``OSPREY_TERMINAL_SECRET_<USER>`` from the deploy ``.env``, presented
under the fixed ``OSPREY_TERMINAL_SECRET`` name. The sidecar's compose file
must therefore list ``OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS: "1"`` plus one
``OSPREY_TERMINAL_SECRET_<USER>: ${...:-}`` per entitled user, and its web gate
(``osprey.interfaces.web_auth`` harvesting the roster, ``WebAuthMiddleware``
checking ``X-Osprey-Terminal-Secret``) must accept exactly those. ``osprey
build`` once rendered the sidecar's compose BEFORE the personas whose rendered
``config.yml`` decide entitlement — and read them from a build zone the build
in flight had not published yet — so the grant came out empty and every user's
panel answered ``401 invalid credential``. The one test on that seam asserted
on a re-render no verb performs, and shipped it green.

This module is the proof that would have caught it on its own: a REAL
``osprey init --preset control-assistant`` + ``osprey build`` render, the
deploy ``.env`` written by the SAME writers ``osprey up`` calls, the sidecar
started by ``docker compose`` from the compose files the build wrote (the
as-built list, resolved the way the start verbs resolve it), and four requests
at the gate that pin the whole chain: alice's own roster secret admits her,
bob's is refused because his persona shows no BLUESKY tab, no credential is
refused, the deployment-wide secret admits.

Kept OUT of ``tests/e2e/`` deliberately: files there carrying ``dockerbuild``
need their own CI job plus an ``--ignore`` in the shared e2e lane (see
``test_ci_workflow_wiring.py``). The precedent for real-docker tests in the
unit lane is ``test_auth_serving.py`` and ``test_env_digest_recreate_proof.py``
in this same directory — module-level skip when docker is unavailable,
exact-named teardown.

WHAT THE SIDECAR IMAGE HERE IS, AND IS NOT
------------------------------------------
The rendered compose builds the sidecar from
``templates/services/bluesky_web/Dockerfile``, which installs the whole
``osprey-framework`` distribution from PyPI (a multi-minute, multi-GB build)
and, under ``osprey build --dev``, overlays a wheel of this checkout. Neither is
affordable here, and the Dockerfile is not what this module tests: the gate
lives in this checkout's ``src/``. So the sidecar runs from a small harness
image — ``python:3.11-slim`` plus the part of the distribution's declared
dependency set the sidecar's import closure needs (taken from the installed
distribution's own metadata, exactly as ``test_auth_serving.py`` does, so the
harness can only contain what a deployment's ``pip install osprey-framework``
would also install) — with this repository's ``src/`` bind-mounted. Everything
else about the container is the rendered compose's own: the uvicorn argv, the
environment (the roster grant included), the config and audit mounts, the
network. A compose OVERRIDE file swaps in the harness image and the source
mounts, gives the container a per-run name, and replaces the rendered fixed
host port with an ephemeral loopback binding, because the deployment's port
layout (``10071`` by default) may be in use by a real deployment on the host
running this test.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import textwrap
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import requires as distribution_requires
from pathlib import Path

import httpx
import pytest
import yaml
from packaging.requirements import Requirement

from osprey.deployment.compose_generator import compose_base_cmd
from osprey.deployment.container_lifecycle import (
    _QSERVER_ZMQ_PRIVATE_KEY_SUFFIX,
    _QSERVER_ZMQ_PUBLIC_KEY_SUFFIX,
    _SERVICE_TOKEN_VARS,
    _ensure_bluesky_control_plane_keys,
    _ensure_service_tokens,
    as_built_compose_files,
)
from osprey.deployment.web_terminals.auth_credentials import terminal_secret_var
from osprey.deployment.web_terminals.personas import (
    config_declares_bluesky_panel,
    resolve_personas,
)
from osprey.deployment.web_terminals.provision import _provision_terminal_secrets
from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
from osprey.interfaces.web_auth import OPERATOR_SECRET_ENV, ROSTER_SECRET_ENV_PREFIX
from osprey.utils.dotenv import ENV_LOCAL_FILENAME, parse_dotenv_file
from tests.cli.test_persona_presets import _build_persona_stack


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

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
# The osprey.* tree reached via /src is shim-backed: config/logger/connectors
# live in the osprey-connectors workspace member, so its source joins the
# mount and the PYTHONPATH — the distribution is not yet installable from
# PyPI inside the harness image.
_CONNECTORS_SRC_DIR = _REPO_ROOT / "packages" / "osprey-connectors" / "src"

#: The compose service key the rendered sidecar file declares.
_SERVICE = "bluesky-web"

#: The panel URL a per-user terminal proxies its BLUESKY tab to.
_PANEL_PATH = "/bluesky/?embedded=true"

#: The two roster users the assertions name. ``alice`` is the ``readwrite``
#: persona (declares the BLUESKY tab); ``bob`` is ``readonly`` (does not).
#: The fixture re-checks both entitlements off the persona renders, so a
#: preset change fails there with the reason rather than as a bare 401.
_ENTITLED_USER = "alice"
_UNENTITLED_USER = "bob"

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

``import osprey.interfaces.bluesky_web.app`` in a bare ``python:3.11-slim``
needs the web stack (the first six) and the config/registry packages
``osprey.interfaces._app_setup`` drags in behind the auth middleware. The
*versions* are never named here — see :func:`_declared_specs`.
"""


# ---------------------------------------------------------------------------
# Harness image
# ---------------------------------------------------------------------------


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_specs() -> tuple[str, ...]:
    """The requirement specifiers ``osprey-framework`` declares for :data:`_PACKAGES`.

    Read from the *installed distribution's* metadata rather than written out
    here, so the harness image can only contain what a deployment's
    ``pip install osprey-framework`` would also install.

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
        f"osprey-framework does not declare {missing}, which the bluesky-web sidecar "
        "imports. The deployed sidecar image installs the distribution's declared "
        "dependencies and would be missing them too."
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


def _build_harness_image(tmp_path: Path) -> tuple[str, bool]:
    """Build (or reuse) the image the sidecar runs from.

    The tag carries a digest of the Dockerfile, so a dependency-declaration
    change produces a new tag instead of silently reusing a stale layer. The
    second value says whether THIS call built it: the teardown removes the
    image only then, leaving a tag another session (or the auth-serving
    harness, whose Dockerfile may be byte-identical) already had on the host.
    """
    dockerfile = _harness_dockerfile()
    tag = f"osprey-bluesky-roster-harness:{hashlib.sha256(dockerfile.encode()).hexdigest()[:12]}"

    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0:
        return tag, False

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
    return tag, True


# ---------------------------------------------------------------------------
# The composed sidecar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sidecar:
    """Everything a test needs to talk to the running sidecar."""

    base_url: str
    #: The deploy ``.env`` as compose interpolated it, secret values included.
    env: dict[str, str]

    def get(self, secret: str | None) -> httpx.Response:
        """One ``GET`` of the panel, carrying ``secret`` the way a terminal does.

        The per-user terminal's proxy presents the secret as the operator
        header; no cookie, no ``Origin``. A test that assembled a cookie or a
        bearer token would be exercising a different rung of the gate's
        credential ladder.
        """
        headers = {OPERATOR_SECRET_HEADER: secret} if secret is not None else {}
        return httpx.get(self.base_url + _PANEL_PATH, headers=headers, timeout=15)


def _scrubbed_env() -> dict[str, str]:
    """The process environment with every value that could pre-empt the render.

    Two consumers see it. The mint treats a process-env value as the operator's
    ("process env wins over .env", so a shell that exports
    ``OSPREY_TERMINAL_SECRET`` suppresses the mint), and compose interpolates
    from the process environment ahead of ``--env-file`` for the same names.
    Either way the container would run on a value this test never wrote to
    the file, and the interpolation contract under test would go unexercised.

    The RE manager's CURVE pair is scrubbed by suffix, one name per lane: a
    public half left in the process by an earlier test's shell-only mint would
    make this fixture's mint refuse to fabricate a private one, and compose
    would then stop the whole project at the bridge's ``:?`` guard.
    """
    minted = {var for names in _SERVICE_TOKEN_VARS.values() for var in names}
    steering = {"OSPREY_BLUESKY_WEB_IMAGE", "COMPOSE_PROJECT_NAME", "COMPOSE_FILE"}
    curve_pair = (_QSERVER_ZMQ_PRIVATE_KEY_SUFFIX, _QSERVER_ZMQ_PUBLIC_KEY_SUFFIX)
    return {
        name: value
        for name, value in os.environ.items()
        if name not in minted
        and name not in steering
        and not name.startswith(ROSTER_SECRET_ENV_PREFIX)
        and not name.endswith(curve_pair)
        and name != OPERATOR_SECRET_ENV
    }


def _write_deploy_env(repo: Path, host_config: dict) -> dict[str, str]:
    """Mint the deploy ``.env`` exactly as ``osprey up``'s preflight does.

    Three writers, all the production ones. ``_provision_terminal_secrets`` is
    the roster mint (one ``OSPREY_TERMINAL_SECRET_<USER>`` per user — the WHOLE
    roster, entitled or not, which is what makes bob's refusal below a refusal
    of a real secret rather than of a blank). ``_ensure_service_tokens`` is
    the per-service token mint that establishes the sidecar's own
    ``OSPREY_TERMINAL_SECRET`` and the launch token its compose interpolates
    without a default. ``_ensure_bluesky_control_plane_keys`` mints the RE
    manager's CURVE keypair, which is not the sidecar's at all but is
    ``:?``-required by the bridge's compose file — and compose refuses the
    whole project on a required variable, started service or not. Nothing is
    hand-written into the file, so what compose reads back through
    ``${OSPREY_TERMINAL_SECRET_ALICE:-}`` is a line the real mint wrote, in
    the real mint's spelling.
    """
    env_path = repo / ENV_LOCAL_FILENAME
    with pytest.MonkeyPatch.context() as patch:
        for name in set(os.environ) - set(_scrubbed_env()):
            patch.delenv(name)
        _provision_terminal_secrets(host_config["modules"]["web_terminals"], str(repo))
        _ensure_service_tokens(host_config, expose_network=False, env_path=env_path)
        _ensure_bluesky_control_plane_keys(host_config, env_path)
    return parse_dotenv_file(env_path)


def _rendered_container_port(repo: Path) -> int:
    """The port the rendered sidecar publishes, read off the build's own file.

    Read rather than recomputed from the port layout, so the override below
    binds the port uvicorn is actually told to listen on even if allocation
    changes. ``ports:`` entries are ``<bind>:<host>:<container>``.
    """
    rendered = yaml.safe_load(
        (repo / "build" / "services" / "bluesky_web" / "docker-compose.yml").read_text(
            encoding="utf-8"
        )
    )
    published = rendered["services"][_SERVICE]["ports"]
    assert len(published) == 1, published
    return int(str(published[0]).rsplit(":", 1)[1])


def _write_override(path: Path, *, image: str, container_name: str, container_port: int) -> None:
    """The one compose file this test adds to the as-built list.

    ``!override`` replaces the rendered ``ports:`` list rather than merging
    into it — compose unions sequence keys by default, which would keep the
    fixed host port published beside the ephemeral one and collide with a
    deployment already on it. ``pull_policy: never`` plus ``--no-build`` on
    the ``up`` means a missing harness tag fails loudly instead of triggering
    the rendered ``build:`` — the multi-GB real image — as a fallback.
    """
    document = {
        "services": {
            _SERVICE: {
                "image": image,
                "pull_policy": "never",
                "container_name": container_name,
                "volumes": [
                    f"{_SRC_DIR}:/src:ro",
                    f"{_CONNECTORS_SRC_DIR}:/connectors-src:ro",
                ],
            }
        }
    }
    text = yaml.safe_dump(document, sort_keys=False)
    # PyYAML has no way to emit a custom tag on a plain sequence, and the tag
    # is the whole point of the entry, so it is spliced in as text.
    text += f'    ports: !override\n      - "127.0.0.1::{container_port}"\n'
    path.write_text(text, encoding="utf-8")


def _entitlements(repo: Path, host_config: dict) -> dict[str, bool]:
    """Which roster users' persona renders declare the BLUESKY tab.

    The same question the build's grant answers, asked of the same files, so
    the assertions below rest on what the preset actually renders rather than
    on this module's memory of it.
    """
    entries = resolve_personas(
        host_config["modules"]["web_terminals"],
        host_config.get("registry") or {},
        (host_config.get("facility") or {}).get("prefix") or "",
        strict=True,
    )
    return {
        entry["name"]: config_declares_bluesky_panel(
            yaml.safe_load(
                (repo / "build" / entry["project"] / "config.yml").read_text(encoding="utf-8")
            )
        )
        for entry in entries
    }


def _docker_logs(name: str) -> str:
    result = subprocess.run(
        ["docker", "logs", "--tail", "40", name], capture_output=True, text=True, timeout=60
    )
    return f"--- {name} ---\n{result.stdout}{result.stderr}"


def _wait_for_health(base_url: str, *, container_name: str, timeout: float = 90.0) -> None:
    """Poll the sidecar's unauthenticated ``/health`` until it answers ``200``.

    Bounded rather than a fixed sleep: uvicorn takes a few seconds to import
    the app from the bind-mounted source, less on a warm host, and a fixed
    delay is either wasted or too short.
    """
    deadline = time.monotonic() + timeout
    last = "no attempt made"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=5)
        except httpx.HTTPError as exc:  # not up yet
            last = repr(exc)
        else:
            if response.status_code == 200:
                return
            last = f"HTTP {response.status_code}"
        time.sleep(0.5)
    raise AssertionError(
        f"the bluesky-web sidecar never became ready ({last})\n{_docker_logs(container_name)}"
    )


@pytest.fixture(scope="module")
def sidecar(tmp_path_factory) -> Iterator[Sidecar]:
    """The sidecar of a real control-assistant build, up on an ephemeral port."""
    tmp_path = tmp_path_factory.mktemp("bluesky-roster")
    repo = _build_persona_stack(tmp_path / "my-facility")
    host_config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    assert "bluesky_web" in host_config["deployed_services"]

    entitlements = _entitlements(repo, host_config)
    assert entitlements.get(_ENTITLED_USER) is True, entitlements
    assert entitlements.get(_UNENTITLED_USER) is False, entitlements

    env = _write_deploy_env(repo, host_config)
    for var in (OPERATOR_SECRET_ENV, terminal_secret_var(_ENTITLED_USER)):
        assert env.get(var, "").strip(), f"{var} was not minted into {ENV_LOCAL_FILENAME}"
    assert env.get(terminal_secret_var(_UNENTITLED_USER), "").strip(), (
        f"{_UNENTITLED_USER}'s secret was not minted: his refusal below must be of a real value"
    )

    image, built_here = _build_harness_image(tmp_path)

    # One compose project per run: the container name is a host-global docker
    # identifier and the network is namespaced by the project, so a unique
    # name is what keeps two concurrent runs — and the host's own deployments —
    # out of each other's way.
    project = f"ospreywf-bluesky-roster-{uuid.uuid4().hex[:8]}"
    container_name = f"{project}-{_SERVICE}"
    override = tmp_path / "bluesky-web.override.yml"
    _write_override(
        override,
        image=image,
        container_name=container_name,
        container_port=_rendered_container_port(repo),
    )

    # The as-built list — what `osprey up` starts — plus the override, pinned
    # to the repo root the way every lifecycle verb pins its invocation, so
    # the rendered relative mounts (`./build/services/bluesky_web/config.yml`,
    # `./var/audit/bluesky-web`) resolve where the build put them. Every file
    # is passed, not just the sidecar's: its rendered `depends_on` names the
    # bridge service, which only the bridge's own file defines; `--no-deps`
    # then starts the sidecar alone.
    compose_files = as_built_compose_files(host_config, repo)
    base = compose_base_cmd(["docker", "compose"], [*compose_files, str(override)], repo)
    base += ["-p", project]
    subprocess_env = _scrubbed_env()

    def _compose(*args: str, timeout: float = 180.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*base, *args], capture_output=True, text=True, timeout=timeout, env=subprocess_env
        )

    try:
        started = _compose("up", "-d", "--no-deps", "--no-build", _SERVICE)
        assert started.returncode == 0, f"{started.stdout}\n{started.stderr}"

        bound = _compose("port", _SERVICE, str(_rendered_container_port(repo)))
        assert bound.returncode == 0, f"{bound.stdout}\n{bound.stderr}"
        host, _, port = bound.stdout.strip().rpartition(":")
        assert host and port.isdigit(), bound.stdout
        base_url = f"http://{host}:{port}"

        _wait_for_health(base_url, container_name=container_name)
        yield Sidecar(base_url=base_url, env=env)
    finally:
        _compose("down", "-v", "--remove-orphans", "--timeout", "5")
        if built_here:
            subprocess.run(["docker", "image", "rm", "-f", image], capture_output=True, timeout=120)


# ---------------------------------------------------------------------------
# The four requests
# ---------------------------------------------------------------------------


def test_an_entitled_users_own_roster_secret_opens_the_panel(sidecar: Sidecar):
    """Alice's terminal proxies the BLUESKY tab with the secret its container
    holds — hers. The build listed her variable, compose interpolated it from
    the deploy ``.env``, and the gate accepted it beside the deployment-wide
    secret: the panel renders. This is the request that answered ``401`` before
    the build rendered the grant."""
    response = sidecar.get(sidecar.env[terminal_secret_var(_ENTITLED_USER)])
    assert response.status_code == 200, response.text
    assert response.headers.get("content-type", "").startswith("text/html"), response.headers


def test_an_unentitled_users_secret_is_refused(sidecar: Sidecar):
    """Bob's persona shows no BLUESKY tab, so the build hands the sidecar no
    key of his. His secret is a real value the roster mint wrote (every user
    gets one for their own terminal's front door) and it is refused here as
    a wrong credential, not as a missing one — the sidecar never learned it."""
    response = sidecar.get(sidecar.env[terminal_secret_var(_UNENTITLED_USER)])
    assert response.status_code == 401, response.text
    assert response.json()["detail"] == "invalid credential"


def test_no_credential_is_refused(sidecar: Sidecar):
    """The gate is closed by default: the panel is not an open route."""
    response = sidecar.get(None)
    assert response.status_code == 401, response.text


def test_the_deployment_wide_secret_opens_the_panel(sidecar: Sidecar):
    """The sidecar's own operator secret — the one ``osprey up`` mints under
    ``OSPREY_TERMINAL_SECRET`` — still admits. The roster grant widens the
    gate; it does not replace the credential a single-user operator holds."""
    response = sidecar.get(sidecar.env[OPERATOR_SECRET_ENV])
    assert response.status_code == 200, response.text
