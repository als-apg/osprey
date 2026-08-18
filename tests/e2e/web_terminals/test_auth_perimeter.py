"""Acceptance proof that ``osprey up`` stands up a WORKING login perimeter
from the **real** auth-sidecar image.

WHAT THIS FILE IS FOR, AND WHAT IT DELIBERATELY DOES NOT DUPLICATE
-----------------------------------------------------------------
``tests/deployment/web_terminals/test_auth_serving.py`` already covers the
perimeter's *semantics* exhaustively — header injection, cookie scope, cross-user
isolation, expiry, rotation, logout — against a composed nginx + sidecar stack.
It builds a deliberately NARROW harness image, though: only the sidecar's import
closure, with each package's specifier read from the distribution's own metadata.
That design is what makes a missing *declaration* fail loudly (drop
``python-multipart`` from ``pyproject.toml`` and its login-POST test goes red),
and it is the right trade for a test that runs in the ordinary unit lane.

What it cannot catch is a break confined to the DEPLOYED Dockerfile
(``templates/modules/web_terminals/auth_sidecar/Dockerfile``): a bad base image,
a wrong ``COPY``, a system library that stopped being installed. Nothing else in
the suite builds that file, so such a break would reach a facility untested.

This file is the one job that builds it. It is therefore deliberately NARROW in
the other direction — it asks only the questions that need the real image and a
real deploy, and leaves every semantic question to the unit lane:

  T1  ``up`` BUILDS the real sidecar image, labeled with this deployment's
      project (what ``osprey reset`` verifies per tag before removing one).
  T2  The sidecar container comes up and its ``/health`` reports ``configured``
      — i.e. the image can actually import and run the app, and it found the
      credentials the same deploy provisioned.
  T3  An unauthenticated request for a user's terminal is REFUSED at nginx.
  T4  A real login unlocks that user, and the authorized request reaches THAT
      USER'S upstream container — proving the whole chain (login → cookie →
      ``auth_request`` → prefix-stripped proxy) works on the deployed artifacts,
      not just on a harness.

THE DEPLOYMENT THIS BUILDS
--------------------------
A real deployment repo, through the surface an operator uses: ``osprey init``
writes the source zone from ``hello-world`` plus one ``-O`` overlay carrying
this lane's web-terminal stanza, and ``osprey build`` renders ``build/``. The
repo's DIRECTORY NAME is the deployment name, so it is also the compose project
name and the ``com.osprey.project`` label T1 asserts.

The persona catalog is added to ``profile.yml`` between those two commands,
which is the documented way to hand-write one: materialization requires every
catalog entry to name a preset that extends the host preset, and this lane
deliberately supplies a hand-written stub project instead of a rendered persona
(see below). A start accepts that — it verifies each referenced persona HAS a
usable project (``config.yml`` + ``Dockerfile``) and renders none itself.

WHY THE PERSONA STUB SERVES HTTP HERE
-------------------------------------
The other lifecycle fixtures use an alpine stub that only idles (``tail -f
/dev/null``), which is enough to inspect and exec into. That is NOT enough here:
with a dead upstream, an authorized request returns 502 and a REFUSED one
returns 401, so the test would be asserting on the difference between two
failures. The stub below serves one byte-identifiable page over busybox
``httpd``, so T4 can assert it received the response ALICE'S OWN container
produced — the positive control that distinguishes "auth let it through" from
"auth let it through and the proxy pointed somewhere else".

It is a stub for the same reason the sidecar is not: a real persona image is a
full framework install per persona, and this lane's subject is the perimeter in
front of the terminal, not the terminal.

COST, AND WHY THIS IS ITS OWN LANE
----------------------------------
Building the real sidecar means ``pip install osprey-framework`` with a C
toolchain — minutes, cold, on every run. The multi-user lifecycle lane is
all-stub and fast, and folding this into it would have made every lifecycle run
pay that. It lives in its own CI lane so the cost is visible on its own line.

``--dev`` is REQUIRED, not incidental: the Dockerfile pins
``osprey-framework==$OSPREY_VERSION``, and a source checkout's version is not on
PyPI. ``--dev`` sets ``OSPREY_DEV=1`` (relaxing the pin miss to an unpinned
prime) and stages the locally-built wheel the image then overlays, so what runs
here is this branch's code rather than a release.

CONTAINER-OPS SAFETY: every runtime-mutating call below names an EXACT resource
this test created — the ``<prefix>-nginx`` / ``<prefix>-auth`` / ``<prefix>-web-
<user>`` containers, this project's volumes, and the ``:local`` image tags — or
is the project-scoped ``compose down`` the deploy lifecycle itself uses. Nothing
here ever runs a prune, an ``-a``/``--all`` sweep, or a wildcard removal.
"""

from __future__ import annotations

import http.cookiejar
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.e2e._volumes import remove_project_volumes

# ``dockerbuild`` is load-bearing, not descriptive: a guard in
# tests/deployment/test_ci_workflow_wiring.py requires every file carrying it to
# be --ignore'd in the shared e2e-tests lane AND given its own job. Without the
# marker this file would be swept into that lane's glob over tests/e2e/ and the
# expensive sidecar build would run twice per PR.
pytestmark = [pytest.mark.e2e, pytest.mark.dockerbuild]

_SUPPORTED_RUNTIMES = ("docker", "podman")
RUNTIME = os.environ.get("OSPREY_E2E_RUNTIME", "docker")
if RUNTIME not in _SUPPORTED_RUNTIMES:
    raise RuntimeError(
        f"OSPREY_E2E_RUNTIME={RUNTIME!r} is not supported; expected one of {_SUPPORTED_RUNTIMES}"
    )

# The repo DIRECTORY name, which is the deployment name: the rendered
# config.yml's project_name, the compose project, and the com.osprey.project
# label on every image this deploy builds.
PROJECT_NAME = "osprey-e2e-auth-perimeter"
PREFIX = "authe2e"
PERSONA = "operator"
PERSONA_PROJECT = f"{PREFIX}-operator"
USERS = ("alice", "bob")
#: The preset the repo is materialized from — the smallest one that builds.
PRESET = "hello-world"

# Ports well clear of every other stack a developer may have up (the tutorial
# stacks hold 5064/5080/5432, and the lifecycle fixtures the 9000s/19081).
NGINX_PORT = 19180
AUTH_PORT = 19170
BASE_PORTS = {
    "web": 19191,
    "artifact": 19291,
    "ariel": 19391,
    "lattice": 19491,
    "channel_finder": 19591,
}

#: Served by each user's stub upstream, so an authorized response can be traced
#: back to the container that produced it.
UPSTREAM_MARKER = "osprey-e2e-auth-perimeter upstream"

AUTH_IMAGE_TAG = f"{PREFIX}-assistant-auth:local"
# `<catalog project>-<persona>:local`, exactly as resolve_personas derives a
# local-mode persona tag. Teardown-only, but spelled the way the render spells
# it so an `rmi` here removes the tag this deploy actually built.
PERSONA_IMAGE_TAG = f"{PERSONA_PROJECT}-{PERSONA}:local"

NGINX_C = f"{PREFIX}-nginx"
AUTH_C = f"{PREFIX}-auth"

# The sidecar build is a real framework install; everything else here is alpine.
DEPLOY_UP_TIMEOUT_SEC = 1800
VERB_TIMEOUT_SEC = 120
# `init` + `build` render the whole repo (no dependency install, no lifecycle
# hooks — see the flags below), which is filesystem work rather than network.
RENDER_TIMEOUT_SEC = 600
READY_TIMEOUT_SEC = 120.0

#: Seeded per user in the repo ``.env`` as ``OSPREY_AUTH_PW_<USER>``, the
#: documented way to set a password you already chose (provisioning step 2).
#: Deterministic on purpose: the alternative — scraping the password ``osprey
#: up`` prints for a generated credential — reads a rich-rendered line that may
#: wrap, which would split the secret at an arbitrary column. What that print
#: does is covered where it belongs, in the credential unit tests.
_PASSWORDS = {"alice": "alice-e2e-pw-correct", "bob": "bob-e2e-pw-correct"}

_ENV_CONTENT = "ANTHROPIC_API_KEY=fake-llm-key-value\n" + "".join(
    f"OSPREY_AUTH_PW_{user.upper()}={password}\n" for user, password in _PASSWORDS.items()
)


def _web_container(user: str) -> str:
    return f"{PREFIX}-web-{user}"


def _runtime_cli(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run([RUNTIME, *args], capture_output=True, text=True, timeout=timeout)


def _fmt(label: str, result: subprocess.CompletedProcess) -> str:
    return (
        f"{label} failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def _find_osprey_console_script() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _run_osprey(
    osprey_bin: Path, args: list[str], cwd: Path, timeout: int = VERB_TIMEOUT_SEC
) -> subprocess.CompletedProcess:
    """Run an ``osprey`` verb with the deploy-side runtime pinned to RUNTIME.

    Deliberately does NOT set ``OSPREY_PIP_SPEC``: the lifecycle fixtures poison
    it so a stub Dockerfile that ever started consuming it would fail loudly, but
    the sidecar image here is a REAL framework install and must resolve normally.
    """
    return subprocess.run(
        [str(osprey_bin), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": "", "CONTAINER_RUNTIME": RUNTIME},
    )


def _container_id(name: str) -> str | None:
    result = _runtime_cli("inspect", "--type", "container", "-f", "{{.Id}}", name, timeout=15)
    return result.stdout.strip() if result.returncode == 0 else None


def _image_exists(tag: str) -> bool:
    return _runtime_cli("inspect", "--type", "image", tag, timeout=15).returncode == 0


def _image_label(tag: str, key: str) -> str | None:
    result = _runtime_cli(
        "inspect",
        "--type",
        "image",
        "-f",
        f'{{{{ index .Config.Labels "{key}" }}}}',
        tag,
        timeout=15,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _logs(name: str) -> str:
    result = _runtime_cli("logs", "--tail", "80", name, timeout=20)
    return f"--- {name} stdout ---\n{result.stdout}\n--- {name} stderr ---\n{result.stderr}"


def _write_persona_project(root: Path) -> Path:
    """A minimal local-mode persona project whose stub actually SERVES.

    Beyond what the idling lifecycle stub provides (a ``dispatch`` user and the
    two mount-point directories the per-user compose service declares volumes
    onto), this one runs busybox ``httpd`` on the port compose hands it, so an
    authorized request through nginx has something real to reach. Binding
    ``127.0.0.1`` mirrors the production app: nginx shares the host network
    namespace and is the only thing meant to reach a per-user port.

    ``config.yml`` and ``Dockerfile`` are the two files a start requires of a
    persona's rendered project (``_check_existing_render``) — in BOTH copies a
    real build produces: the flat host render at the project path, and the
    container copy at the ``.image/<name>`` sibling whose ``build/Dockerfile``
    is what the persona image is actually built from
    (``_persona_image_context``). The stub writes the same two files into each,
    which is why something this small is accepted where a rendered persona
    would otherwise be.

    ``busybox-extras`` is REQUIRED and easy to lose: alpine's default busybox
    does NOT include the ``httpd`` applet, so without this package the CMD dies
    with "httpd: not found", the container crashloops, and the failure surfaces
    far from its cause — as ``osprey up`` aborting because it could not seed a
    container that never stayed up.
    """
    root.mkdir(parents=True)
    config_text = f"project_name: {PERSONA_PROJECT}\n"
    (root / "config.yml").write_text(config_text, encoding="utf-8")
    container_dir = f"/app/{PERSONA_PROJECT}"
    (root / "Dockerfile").write_text(
        "FROM alpine:3.20\n"
        # busybox-extras carries the httpd applet; the base busybox does not.
        "RUN apk add --no-cache busybox-extras \\\n"
        "    && adduser -D dispatch \\\n"
        "    && mkdir -p /data/claude-config \\\n"
        # var/agent_data, not a literal of this test's own choosing: the render
        # derives the per-user agent-data mount target from config.yml's
        # `agent_data.base_dir` (default `var/agent_data`), exactly as the
        # shipped project Dockerfile pre-creates it. A directory the image never
        # created is created root-owned at first start, and the non-root user
        # this container runs as cannot then write to its own memory.
        f"    && mkdir -p {container_dir}/var/agent_data \\\n"
        f"    && mkdir -p {container_dir}/.claude/skills \\\n"
        f"    && chown -R dispatch:dispatch /data/claude-config {container_dir} \\\n"
        # /srv belongs to dispatch too: the CMD writes index.html at start-up,
        # and a root-owned directory would make that write the next crashloop.
        "    && mkdir -p /srv \\\n"
        "    && chown -R dispatch:dispatch /srv\n"
        # Written at START-UP, not build time: OSPREY_TERMINAL_USER and
        # OSPREY_TERMINAL_WEB_PORT are per-user values compose supplies to the
        # container, so one image serves every user and each page names the
        # container that produced it. That marker is what makes T4 a proof of
        # ROUTING rather than merely of authorization.
        'CMD ["sh", "-c", "echo '
        f"'{UPSTREAM_MARKER} for '"
        '\\"$OSPREY_TERMINAL_USER\\" > /srv/index.html; '
        'exec httpd -f -p 127.0.0.1:\\"$OSPREY_TERMINAL_WEB_PORT\\" -h /srv"]\n',
        encoding="utf-8",
    )
    # The container copy the image is actually built from: `docker build
    # -f <context>/build/Dockerfile <context>` with the context at the
    # `.image/<name>` sibling. The stub's recipe holds no host paths, so the
    # container copy IS the host copy.
    image_build = root.parent / ".image" / root.name / "build"
    image_build.mkdir(parents=True)
    (image_build / "config.yml").write_text(config_text, encoding="utf-8")
    shutil.copy2(root / "Dockerfile", image_build / "Dockerfile")
    return root


def _override_text() -> str:
    """The ``-O`` overlay carrying this lane's whole web-terminal stanza.

    Dotted leaf keys under ``config:``, the one spelling a profile's config
    block accepts — and ``modules.web_terminals`` deliberately as ONE dotted key
    with a nested value, so it sets that subtree without replacing the rendered
    ``modules:`` mapping around it.

    ``allow_insecure_http`` is what lets auth render without TLS. That is the
    documented posture for a deployment behind a TLS terminator, and here it
    keeps the test off certificate management — the TLS mount has its own
    render-level coverage in ``tests/deployment/web_terminals/test_auth_tls_seams.py``.

    ``deployed_services: []`` drops everything the preset would otherwise
    deploy: this lane deploys the web tier and nothing else, and a backend
    service would only add containers and bound ports to a proof about nginx.
    """
    return yaml.safe_dump(
        {
            "config": {
                "container_runtime": RUNTIME,
                "facility.name": "E2E Auth Perimeter Fixture",
                "facility.prefix": PREFIX,
                "facility.timezone": "UTC",
                "deploy.fqdn": "127.0.0.1",
                "deployed_services": [],
                "modules.web_terminals": {
                    "enabled": True,
                    "image_source": "local",
                    "default_persona": PERSONA,
                    "nginx_port": NGINX_PORT,
                    "web_base_port": BASE_PORTS["web"],
                    "artifact_base_port": BASE_PORTS["artifact"],
                    "ariel_base_port": BASE_PORTS["ariel"],
                    "lattice_base_port": BASE_PORTS["lattice"],
                    "channel_finder_base_port": BASE_PORTS["channel_finder"],
                    "users": list(USERS),
                    "auth": {
                        "method": "password",
                        "port": AUTH_PORT,
                        "allow_insecure_http": True,
                    },
                },
            }
        },
        sort_keys=False,
    )


def _add_persona_catalog(repo: Path, persona_path: Path) -> None:
    """Point the profile's persona catalog at the hand-written stub project.

    After materialization, deliberately. ``osprey init`` renders one delta per
    catalog entry and requires each entry to name a preset that EXTENDS the host
    preset — the shape a rendered persona has. This lane's persona is a stub
    project instead (see the module docstring), so the entry is added to
    ``profile.yml`` once the repo exists and before ``osprey build`` reads it,
    which is the remedy materialization itself names for a hand-written persona.

    A YAML round-trip rather than a text splice: the emitted profile is plain
    YAML, and the comments a dump drops are documentation for an operator, not
    input to the build.
    """
    profile_path = repo / "profile.yml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["config"]["modules.web_terminals"]["personas"] = {
        PERSONA: {"project": PERSONA_PROJECT, "project_path": str(persona_path)}
    }
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")


def _make_repo(tmp_path: Path, osprey_bin: Path) -> Path:
    """``init`` + persona catalog + ``build`` — the repo this lane deploys.

    ``--skip-deps``/``--skip-lifecycle`` keep the render off the network and
    quick: nothing here runs the deployment's own venv, only its containers.
    """
    persona_path = _write_persona_project(tmp_path / "persona")
    repo = tmp_path / PROJECT_NAME
    override_path = tmp_path / "override.yml"
    override_path.write_text(_override_text(), encoding="utf-8")

    init = _run_osprey(
        osprey_bin,
        [
            "init",
            str(repo),
            "--preset",
            PRESET,
            "--no-git",
            "--override",
            str(override_path),
        ],
        tmp_path,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert init.returncode == 0, _fmt("osprey init (auth perimeter)", init)

    _add_persona_catalog(repo, persona_path)

    build = _run_osprey(
        osprey_bin,
        ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle", "--dev"],
        tmp_path,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert build.returncode == 0, _fmt("osprey build (auth perimeter)", build)

    # The repo root's .env is this deployment's whole secret store: the auth
    # provisioning reads each user's chosen OSPREY_AUTH_PW_<USER> from it, the
    # provider-secret gate reads the key, and `osprey up` refuses to start
    # without the file at all.
    env_path = repo / ".env"
    env_path.write_text(_ENV_CONTENT, encoding="utf-8")
    os.chmod(env_path, 0o600)
    return repo


def _teardown() -> None:
    """Exact-named sweep; failures swallowed (a safety net, never an assertion).

    The volume sweep is what keeps reruns honest: ``compose down`` (like
    ``osprey down``) keeps named volumes, and a rerun inheriting a previous
    attempt's per-user volumes would start from that attempt's state.
    """
    for user in USERS:
        _runtime_cli("rm", "-f", _web_container(user))
    _runtime_cli("rm", "-f", AUTH_C)
    _runtime_cli("rm", "-f", NGINX_C)
    _runtime_cli("compose", "-p", PROJECT_NAME, "down", timeout=60)
    remove_project_volumes(PROJECT_NAME, runtime=RUNTIME)
    for tag in (AUTH_IMAGE_TAG, PERSONA_IMAGE_TAG):
        _runtime_cli("rmi", "-f", tag, timeout=60)


def _browser() -> tuple[urllib.request.OpenerDirector, http.cookiejar.CookieJar]:
    """A client that keeps cookies the way a browser does.

    Login answers ``303 See Other`` and sets the session cookie on THAT response,
    so a plain ``urlopen`` — which follows the redirect itself — returns the
    followed response's headers and the cookie is gone by the time the caller
    sees anything. A cookie jar captures it mid-redirect and replays it on every
    later request through the same opener, which is both what a browser does and
    what makes the assertions here about the session rather than about headers.
    """
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar)), jar


def _request(
    target: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    port: int = NGINX_PORT,
    opener: urllib.request.OpenerDirector | None = None,
) -> tuple[int, dict[str, str], str]:
    """One HTTP request, returning (status, headers, body) and never raising on 4xx/5xx.

    Pass ``opener`` (from :func:`_browser`) to carry a session across requests;
    omit it for the unauthenticated probes, which must carry nothing.
    """
    req = urllib.request.Request(  # noqa: S310 - loopback only
        f"http://127.0.0.1:{port}{target}", method=method, data=data
    )
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    open_it = opener.open if opener is not None else urllib.request.urlopen
    try:
        with open_it(req, timeout=15) as resp:  # noqa: S310 - loopback only
            return resp.status, dict(resp.headers), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read().decode("utf-8", "replace")


def _wait_for_health(timeout: float) -> dict[str, Any]:
    """Poll the sidecar's own /health until it answers, or fail with its logs.

    A dead sidecar is the failure this lane exists to catch, so the message on
    timeout carries the container's logs — an image that cannot import the app
    says so there and nowhere else.
    """
    deadline = time.monotonic() + timeout
    last = "(no attempt yet)"
    while time.monotonic() < deadline:
        try:
            status, _, body = _request("/health", port=AUTH_PORT)
            if status == 200:
                return json.loads(body)
            last = f"HTTP {status}: {body[:200]}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last = str(exc)
        time.sleep(2.0)
    raise AssertionError(
        f"auth sidecar /health not ready after {timeout:.0f}s (last: {last})\n{_logs(AUTH_C)}"
    )


def _navigation() -> dict[str, str]:
    """The ``Accept`` a browser sends when a person navigates to a URL."""
    return {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


@pytest.fixture(scope="module")
def deployment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """One ``osprey up --dev`` for the whole module — the expensive part runs once."""
    if shutil.which(RUNTIME) is None:
        pytest.skip(f"{RUNTIME} not available")
    if _runtime_cli("ps", timeout=10).returncode != 0:
        pytest.skip(f"{RUNTIME} daemon not responding")

    tmp_path = tmp_path_factory.mktemp("auth-perimeter")
    osprey_bin = _find_osprey_console_script()
    repo = _make_repo(tmp_path, osprey_bin)

    _teardown()  # clear anything a previous crashed run stranded under these names
    try:
        # Run from inside the repo: every lifecycle verb walks up to the nearest
        # profile.yml, so standing in the deployment is what selects it.
        up = _run_osprey(osprey_bin, ["up", "--dev"], repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up.returncode == 0, _fmt("osprey up --dev (auth perimeter)", up)
        health = _wait_for_health(READY_TIMEOUT_SEC)
        yield {"repo": repo, "health": health, "up": up}
    finally:
        # The shipped teardown first — it stops the web stack the way the
        # lifecycle does, in the order that leaves nothing behind — then the
        # exact-named sweep as a safety net for anything it could not reach.
        _run_osprey(osprey_bin, ["down"], repo)
        _teardown()


def test_up_builds_the_real_sidecar_image(deployment: dict[str, Any]) -> None:
    """T1: the DEPLOYED Dockerfile builds, and the tag carries this project's label.

    The image is what every other assertion in this file runs against, and it is
    the artifact no other job in the suite produces. The ownership label matters
    on its own: an image tag is host-global and shared between two checkouts of
    one deployment, so it is the one class ``osprey reset`` cannot scope by
    checkout identity — it verifies ``com.osprey.project`` on the tag itself
    before removing it, and an unlabeled image survives a factory reset.
    """
    assert _image_exists(AUTH_IMAGE_TAG), f"{AUTH_IMAGE_TAG} was not built by 'osprey up'"
    assert _image_label(AUTH_IMAGE_TAG, "com.osprey.project") == PROJECT_NAME


def test_the_sidecar_runs_and_reports_itself_configured(deployment: dict[str, Any]) -> None:
    """T2: the container starts, the app imports, and it found this deploy's credentials.

    ``configured: false`` is the sidecar's fail-closed posture when it has no
    credentials — it still answers ``/health`` and refuses every other path. A
    ``true`` here therefore proves two separate things landed: the image can run
    the app at all, and the ``.env.auth`` the SAME deploy provisioned reached it.
    """
    assert _container_id(AUTH_C) is not None, f"{AUTH_C} was not created"
    assert deployment["health"].get("configured") is True, (
        f"sidecar came up unconfigured: {deployment['health']}\n{_logs(AUTH_C)}"
    )


def test_a_terminal_is_refused_without_a_session(deployment: dict[str, Any]) -> None:
    """T3: nginx refuses an unauthenticated request for a user's terminal.

    Asked with a *program's* ``Accept`` so the answer is a bare 401 rather than
    the login-page redirect a browser navigation receives — the redirect variant
    is covered in the unit lane; what matters here is that the deployed
    perimeter refuses at all.
    """
    status, _, _ = _request("/u/alice/", headers={"Accept": "application/json"})
    assert status == 401, f"unauthenticated request was not refused (got {status})"


def test_a_real_login_reaches_that_users_own_upstream(deployment: dict[str, Any]) -> None:
    """T4: login → cookie → auth_request → prefix-stripped proxy, end to end.

    The response body must be the one ALICE'S container serves. A 200 alone
    would only prove the request got past authentication; the per-user marker is
    what proves nginx then routed it to the right upstream with the ``/u/alice``
    prefix stripped — the two halves of the contract that can fail
    independently.
    """
    client, jar = _browser()
    body = urllib.parse.urlencode({"user": "alice", "password": _PASSWORDS["alice"]}).encode()
    status, _, _ = _request(
        "/auth/login",
        method="POST",
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            # The sidecar refuses a submission declaring no origin at all.
            "Origin": f"http://127.0.0.1:{NGINX_PORT}",
            **_navigation(),
        },
        opener=client,
    )
    assert status == 200, f"login was not accepted (got {status})\n{_logs(AUTH_C)}"
    assert [c.name for c in jar], f"login issued no session cookie\n{_logs(AUTH_C)}"

    status, _, page = _request(
        "/u/alice/",
        headers=_navigation(),
        opener=client,
    )
    assert status == 200, (
        f"authorized request did not reach the upstream (got {status})\n"
        f"{_logs(NGINX_C)}\n{_logs(_web_container('alice'))}"
    )
    assert f"{UPSTREAM_MARKER} for alice" in page, (
        f"authorized request reached the wrong upstream:\n{page[:400]}"
    )
