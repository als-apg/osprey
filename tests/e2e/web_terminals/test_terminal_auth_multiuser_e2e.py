"""Acceptance proof that the DEFAULT multi-user posture — ``auth.method: none``
— still closes the chain in front of every terminal, because the terminal app
owns the gate.

WHY THIS LANE EXISTS ALONGSIDE ``test_auth_perimeter.py``
--------------------------------------------------------
``test_auth_perimeter.py`` proves the *authenticated* perimeter: with
``auth.method: password`` nginx runs an ``auth_request`` against the sidecar,
injects the per-user operator secret as a header on the way through, and strips
the browser's cookie. There the sidecar is the front door and nginx vouches for
every request that reaches a container.

This lane proves the case that has NO sidecar and NO header injection at all —
the one a fresh ``osprey init`` gives you, since ``modules.web_terminals.auth``
defaults to ``method: none``. With auth off, nginx runs no ``auth_request``,
injects no ``X-Osprey-Terminal-Secret`` header, and passes the browser's cookie
straight THROUGH (all three effects hang off nginx.conf.j2's
``auth_method != "none" and not login_exempt`` predicate). So nothing in front
of the container authenticates anything: the per-user terminal app's own
``WebAuthMiddleware`` is the only thing standing between one user's browser and
another user's terminal.

That is the more dangerous half of the design, not the less: an operator who
never turns auth on is relying entirely on the app-owned gate. ``osprey`` mints
a per-user operator secret for every deployment regardless of auth method
(``ensure_terminal_secrets`` runs even when ``auth.method: none``), precisely so
that this gate has a credential to enforce. This test is the end-to-end proof
that it does.

WHAT THIS FILE ASSERTS, ON A REAL STACK
---------------------------------------
Against a real ``osprey up`` deployment — nginx plus a per-user terminal
container, reached over the published ``nginx_port`` with real HTTP clients:

  T1  A mutating request to a user's terminal API carrying NO credential — no
      cookie, and (because auth is off) no header nginx could have injected —
      is refused with ``401`` by the app itself, even though nginx passed it
      through untouched.
  T2  The per-user ``?token=`` login URL, whose token is that user's minted
      operator secret, is exchanged by a browser for a session cookie; the SAME
      mutating request carrying that cookie (which nginx forwards, since auth is
      off) is then accepted with ``200``. The gate the browser had to clear is
      the app's, reached only because the deployment minted the secret the login
      URL carries.

THE TERMINAL BEHIND NGINX IS REAL, NOT A STUB
---------------------------------------------
``test_auth_perimeter.py`` puts a one-page busybox stub behind nginx, because
its subject is the perimeter and a real terminal image would only add minutes to
a test about the sidecar. Here the terminal IS the subject: the gate under test
lives in the terminal app, so the container behind nginx must run the real
``osprey web`` — the actual ``WebAuthMiddleware``, the actual ``?token=`` →
cookie exchange handler, the actual ``PATCH /api/config`` operator route. A stub
would prove nothing about the gate.

The persona image is therefore a REAL framework install (``pip install
osprey-framework`` with a C toolchain plus the Claude Code CLI — minutes, cold),
which is why this file carries the ``dockerbuild`` marker and runs in its own CI
lane rather than the shared fast e2e glob. Its recipe is not hand-rolled: the
fixture renders a throwaway ``hello-world`` deployment with the shipped ``osprey
init``/``osprey build`` and reuses that render's own container repo — the
production project ``Dockerfile`` that runs ``osprey web``, and a real rendered
``config.yml`` the app can serve — as the persona's project. So the image built
here is the one a real deployment builds, not an approximation of it.

``--dev`` is REQUIRED for the same reason it is in the perimeter lane: the
project Dockerfile pins ``osprey-framework`` to this deployment's version, which
a source checkout does not publish to PyPI. ``--dev`` sets ``OSPREY_DEV=1``
(relaxing the pin miss to an unpinned prime) and stages the locally-built wheel
the image then overlays, so the terminal that runs here is this branch's code.

CONTAINER-OPS SAFETY: every runtime-mutating call below names an EXACT resource
this test created — the ``<prefix>-nginx`` / ``<prefix>-web-<user>`` containers,
this project's volumes, and the ``:local`` image tag — or is the project-scoped
``compose down`` the deploy lifecycle itself uses. Nothing here ever runs a
prune, an ``-a``/``--all`` sweep, or a wildcard removal.
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

from osprey.deployment.web_terminals.auth_credentials import terminal_secret_var
from osprey.utils.dotenv import parse_dotenv_file
from tests.e2e._volumes import remove_project_volumes

# ``dockerbuild`` is load-bearing, not descriptive: a guard in
# tests/deployment/test_ci_workflow_wiring.py requires every file carrying it to
# be --ignore'd in the shared e2e-tests lane AND given its own job. Without the
# marker this file would be swept into that lane's glob over tests/e2e/ and the
# expensive framework build would run in the wrong lane.
pytestmark = [pytest.mark.e2e, pytest.mark.dockerbuild]

_SUPPORTED_RUNTIMES = ("docker", "podman")
RUNTIME = os.environ.get("OSPREY_E2E_RUNTIME", "docker")
if RUNTIME not in _SUPPORTED_RUNTIMES:
    raise RuntimeError(
        f"OSPREY_E2E_RUNTIME={RUNTIME!r} is not supported; expected one of {_SUPPORTED_RUNTIMES}"
    )

# The repo DIRECTORY name, which is the deployment name: the compose project and
# the com.osprey.project label on every image this deploy builds.
PROJECT_NAME = "osprey-e2e-authoff-multiuser"
PREFIX = "authoff"
PERSONA = "operator"
PERSONA_PROJECT = f"{PREFIX}-operator"
#: One user is enough for the default-posture proof: the gate under test is the
#: app's own, and a second roster user would only add a second heavy container
#: to a proof about one terminal's front door.
USER = "alice"
USERS = (USER,)
#: The preset the throwaway persona render (and the host repo) are built from —
#: the smallest one that stands up a serving ``osprey web``.
PRESET = "hello-world"

# Ports well clear of every other stack a developer may have up (the tutorial
# stacks hold 5064/5080/5432; the lifecycle fixtures the 9000s/19081; the auth
# perimeter lane the 191xx/192xx band). This lane takes a distinct band.
NGINX_PORT = 19680
BASE_PORTS = {
    "web": 19671,
    "artifact": 19771,
    "ariel": 19871,
    "lattice": 19971,
    "channel_finder": 19981,
}

PERSONA_IMAGE_TAG = f"{PERSONA_PROJECT}-{PERSONA}:local"
NGINX_C = f"{PREFIX}-nginx"

# The persona build is a real framework install; give it room.
DEPLOY_UP_TIMEOUT_SEC = 1800
VERB_TIMEOUT_SEC = 120
# `init` + `build` render the whole repo (no dependency install, no lifecycle
# hooks — see the flags below), which is filesystem work rather than network.
RENDER_TIMEOUT_SEC = 600
# `osprey web` boot inside the container (import the framework, start the app)
# is slower than a static stub, so readiness is polled generously.
READY_TIMEOUT_SEC = 240.0

#: A benign, valid mutating update for the operator-only ``PATCH /api/config``
#: route: a single dot-notation field that ``config_update_fields`` can write to
#: the served ``config.yml``. The point is the STATUS the gate returns, not the
#: value written — a display label mutates nothing the deployment relies on.
_CONFIG_PATCH_BODY = json.dumps({"updates": {"web.app_name": "authoff-e2e"}}).encode()

# ZO_INGEST_SA_TOKEN is present but inert: hello-world renders an OpenObserve
# telemetry block whose password is `${ZO_INGEST_SA_TOKEN}`, and the deploy
# preflight statically refuses to generate `.env.users` while that reference is
# unresolvable — even though the block is disabled and the store is not deployed
# here. Supplying a value satisfies the resolver and is the documented remedy for
# a deployment behind an external store; with telemetry off the app never emits,
# so the value is never used.
_ENV_CONTENT = "ANTHROPIC_API_KEY=fake-llm-key-value\nZO_INGEST_SA_TOKEN=fake-telemetry-token\n"


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
    the persona image here is a REAL framework install and must resolve normally.
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


def _logs(name: str) -> str:
    result = _runtime_cli("logs", "--tail", "80", name, timeout=20)
    return f"--- {name} stdout ---\n{result.stdout}\n--- {name} stderr ---\n{result.stderr}"


def _render_reference_project(tmp_path: Path, osprey_bin: Path) -> Path:
    """Render a throwaway ``hello-world`` deployment and return its ``build/`` dir.

    The persona this lane deploys must run the REAL ``osprey web``, which needs a
    real container repo: the production project ``Dockerfile`` (the one that
    installs the framework and launches ``osprey web``) and a rendered
    ``config.yml`` the app can actually serve. Rather than hand-roll either — a
    hand-written recipe drifts from the shipped one, and a hand-written config
    misses keys the app reads — this renders a genuine deployment with the same
    ``osprey init``/``osprey build`` surface an operator uses, and
    :func:`_write_persona_project` reuses that render's own artifacts.

    ``--skip-deps``/``--skip-lifecycle`` keep the render off the network and
    quick: nothing here installs a venv or builds an image, it only writes files.

    Rendered under the persona's OWN project name (:data:`PERSONA_PROJECT`) on
    purpose: the production project Dockerfile bakes ``COPY . /app/<project>/``
    and ``WORKDIR /app/<project>`` from that name, and ``resolve_personas``
    derives the per-user ``container_project_dir`` as ``/app/<catalog project>``.
    Naming the reference render for the catalog project keeps the two equal, so
    the compose service's mount targets land on the directory the app runs in.
    """
    ref_repo = tmp_path / PERSONA_PROJECT
    # Telemetry OFF in the persona render too, for the same reason the host
    # override disables it: hello-world ships it enabled against an OpenObserve
    # store this lane does not deploy, so its rendered config.yml would carry
    # `openobserve.password: ${ZO_INGEST_SA_TOKEN}` — a credential only the store
    # can issue — and `osprey up` refuses to serve a terminal whose telemetry
    # names an unresolvable secret. The perimeter lane avoids this with a
    # config-less busybox stub; a real `osprey web` persona must render a config
    # that actually stands up.
    ref_override = tmp_path / "persona-ref-override.yml"
    ref_override.write_text(
        yaml.safe_dump({"config": {"claude_code.telemetry.enabled": False}}, sort_keys=False),
        encoding="utf-8",
    )
    init = _run_osprey(
        osprey_bin,
        ["init", str(ref_repo), "--preset", PRESET, "--no-git", "--override", str(ref_override)],
        tmp_path,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert init.returncode == 0, _fmt("osprey init (persona reference)", init)
    build = _run_osprey(
        osprey_bin,
        ["build", "--repo", str(ref_repo), "--skip-deps", "--skip-lifecycle"],
        tmp_path,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert build.returncode == 0, _fmt("osprey build (persona reference)", build)
    ref_build = ref_repo / "build"
    assert (ref_build / "config.yml").is_file() and (ref_build / "Dockerfile").is_file(), (
        f"reference render is incomplete at {ref_build}"
    )
    return ref_build


def _write_persona_project(root: Path, ref_build: Path) -> Path:
    """A local-mode persona project whose container runs the REAL ``osprey web``.

    The persona is spelled the way a start requires — ``config.yml`` + a
    ``Dockerfile`` in the flat render (``_check_existing_render`` reads these,
    and ``verify_persona_renders`` also parses the flat ``config.yml`` for its
    provider/model), and the same two under the container image context at the
    ``.image/<name>`` sibling whose ``build/Dockerfile`` the image is actually
    built from (``_persona_image_context``).

    Both copies come from the throwaway render (:func:`_render_reference_project`)
    rather than being written by hand:

    * the flat render's ``config.yml``/``Dockerfile`` are the reference render's
      own, so ``verify_persona_renders`` reads a real provider spec;
    * the image context is the reference render's WHOLE container repo
      (``build/.image/persona-ref/`` — ``profile.yml`` at its root, the render in
      ``build/`` below it), copied verbatim. Its ``Dockerfile`` is the production
      project image: ``COPY . /app/<name>/``, then ``osprey web`` from that
      directory, which walks up to the ``profile.yml`` this copy provides and
      serves its ``build/config.yml``. That is what makes the terminal behind
      nginx a real app rather than a stub.

    Under ``osprey up --dev`` the persona build stages the locally-built wheel
    into this same image context and passes ``OSPREY_PIP_SPEC``/``OSPREY_DEV``,
    which the copied production Dockerfile's deps and wheel layers consume — so
    the running terminal is this branch's code.
    """
    root.mkdir(parents=True)
    reference_context = ref_build / ".image" / PERSONA_PROJECT
    assert (reference_context / "profile.yml").is_file(), (
        f"reference container repo missing profile.yml at {reference_context}"
    )

    # The container image context, copied whole from the reference render. Named
    # for this persona's flat render so `_persona_image_context` (parent/.image/
    # <render name>) resolves to exactly this directory.
    image_context = root.parent / ".image" / root.name
    shutil.copytree(reference_context, image_context)

    # The flat render's two required files, from the same reference build.
    shutil.copy2(ref_build / "config.yml", root / "config.yml")
    shutil.copy2(ref_build / "Dockerfile", root / "Dockerfile")
    return root


def _override_text() -> str:
    """The ``-O`` overlay carrying this lane's whole web-terminal stanza.

    Dotted leaf keys under ``config:``, the one spelling a profile's config
    block accepts — and ``modules.web_terminals`` deliberately as ONE dotted key
    with a nested value, so it sets that subtree without replacing the rendered
    ``modules:`` mapping around it.

    ``auth.method: none`` is the whole point: it is the default multi-user
    posture, and it is what makes the app-owned gate the ONLY gate — nginx runs
    no ``auth_request``, injects no operator-secret header, and forwards the
    browser's cookie. With auth off there is no TLS obligation either (the render
    only requires TLS or ``allow_insecure_http`` when a real auth method would
    otherwise send session cookies across the network), so no certificate
    management enters this lane.

    ``deployed_services: []`` drops everything the preset would otherwise deploy:
    this lane deploys the web tier and nothing else, and a backend service would
    only add containers and bound ports to a proof about nginx. Telemetry goes
    off with them, and for the same reason the perimeter lane disables it — the
    preset ships it aimed at a store this deploy no longer runs.
    """
    return yaml.safe_dump(
        {
            "config": {
                "container_runtime": RUNTIME,
                "facility.name": "E2E Auth-Off Multiuser Fixture",
                "facility.prefix": PREFIX,
                "facility.timezone": "UTC",
                "deploy.fqdn": "127.0.0.1",
                "deployed_services": [],
                "claude_code.telemetry.enabled": False,
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
                    "auth": {"method": "none"},
                },
            }
        },
        sort_keys=False,
    )


def _add_persona_catalog(repo: Path, persona_path: Path) -> None:
    """Point the profile's persona catalog at the hand-written stub project.

    After materialization, deliberately. ``osprey init`` renders one delta per
    catalog entry and requires each entry to name a preset that EXTENDS the host
    preset — the shape a rendered persona has. This lane's persona is a project
    reused from a separate render instead (see :func:`_write_persona_project`),
    so the entry is added to ``profile.yml`` once the repo exists and before
    ``osprey build`` reads it, which is the remedy materialization itself names
    for a hand-written persona.
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
    ref_build = _render_reference_project(tmp_path, osprey_bin)
    persona_path = _write_persona_project(tmp_path / "persona", ref_build)
    repo = tmp_path / PROJECT_NAME
    override_path = tmp_path / "override.yml"
    override_path.write_text(_override_text(), encoding="utf-8")

    init = _run_osprey(
        osprey_bin,
        ["init", str(repo), "--preset", PRESET, "--no-git", "--override", str(override_path)],
        tmp_path,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert init.returncode == 0, _fmt("osprey init (auth-off multiuser)", init)

    _add_persona_catalog(repo, persona_path)

    build = _run_osprey(
        osprey_bin,
        ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle", "--dev"],
        tmp_path,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert build.returncode == 0, _fmt("osprey build (auth-off multiuser)", build)

    # The repo root's .env is this deployment's secret store: `osprey up` refuses
    # to start without it, the provider-secret gate reads the key, and
    # `ensure_terminal_secrets` mints each user's OSPREY_TERMINAL_SECRET_<USER>
    # into it — the value this test reads back to build the login URL.
    env_path = repo / ".env"
    env_path.write_text(_ENV_CONTENT, encoding="utf-8")
    os.chmod(env_path, 0o600)
    return repo


def _read_user_secret(repo: Path, user: str) -> str:
    """The user's minted operator secret, from the deploy ``.env``.

    ``ensure_terminal_secrets`` writes ``OSPREY_TERMINAL_SECRET_<USER>`` into the
    repo-root ``.env`` for EVERY deployment, auth method included — the value the
    per-user ``?token=`` login URL carries and the terminal app authenticates
    against. Read straight from the file (the one place both nginx and the app
    interpolate it from) rather than reconstructed, so the test uses the exact
    value the running container holds.
    """
    var = terminal_secret_var(user)
    parsed = parse_dotenv_file(repo / ".env")
    secret = (parsed.get(var) or "").strip()
    assert secret, f"{var} was not provisioned into the deploy .env for {user!r}"
    return secret


def _teardown() -> None:
    """Exact-named sweep; failures swallowed (a safety net, never an assertion).

    The volume sweep is what keeps reruns honest: ``compose down`` (like
    ``osprey down``) keeps named volumes, and a rerun inheriting a previous
    attempt's per-user volumes would start from that attempt's state.
    """
    for user in USERS:
        _runtime_cli("rm", "-f", _web_container(user))
    _runtime_cli("rm", "-f", NGINX_C)
    _runtime_cli("compose", "-p", PROJECT_NAME, "down", timeout=60)
    remove_project_volumes(PROJECT_NAME, runtime=RUNTIME)
    _runtime_cli("rmi", "-f", PERSONA_IMAGE_TAG, timeout=60)


def _browser() -> tuple[urllib.request.OpenerDirector, http.cookiejar.CookieJar]:
    """A client that keeps cookies the way a browser does.

    The token exchange answers ``303 See Other`` and sets the session cookie on
    THAT response, so a plain ``urlopen`` — which follows the redirect itself —
    would return the followed response and the cookie would be gone by the time
    the caller sees anything. A cookie jar captures it mid-redirect and replays
    it on every later request through the same opener, which is both what a
    browser does and what makes the assertions here about the session.
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
        with open_it(req, timeout=20) as resp:  # noqa: S310 - loopback only
            return resp.status, dict(resp.headers), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read().decode("utf-8", "replace")


def _navigation() -> dict[str, str]:
    """The ``Accept`` a browser sends when a person navigates to a URL."""
    return {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


def _wait_for_gate(user: str, timeout: float) -> None:
    """Poll the user's terminal through nginx until its app-owned gate answers.

    Readiness here is not "nginx is up" but "the terminal app behind it is up and
    refusing" — an unauthenticated probe that comes back ``401`` proves the real
    ``WebAuthMiddleware`` is running and gating, which is the precondition for
    both assertions. A ``502`` means nginx is up but the container is still
    booting (`osprey web` is a real framework import), so it keeps polling; a
    refused connection means nginx itself is not listening yet. On timeout the
    message carries both containers' logs, since a persona that never came up is
    the most likely cause and says so only there.
    """
    deadline = time.monotonic() + timeout
    last = "(no attempt yet)"
    while time.monotonic() < deadline:
        try:
            status, _, body = _request(
                f"/u/{user}/api/config",
                method="PATCH",
                data=_CONFIG_PATCH_BODY,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            if status == 401:
                return
            last = f"HTTP {status}: {body[:200]}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last = str(exc)
        time.sleep(3.0)
    raise AssertionError(
        f"{user}'s terminal gate not ready after {timeout:.0f}s (last: {last})\n"
        f"{_logs(NGINX_C)}\n{_logs(_web_container(user))}"
    )


@pytest.fixture(scope="module")
def deployment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """One ``osprey up --dev`` for the whole module — the expensive part runs once."""
    if shutil.which(RUNTIME) is None:
        pytest.skip(f"{RUNTIME} not available")
    if _runtime_cli("ps", timeout=10).returncode != 0:
        pytest.skip(f"{RUNTIME} daemon not responding")

    tmp_path = tmp_path_factory.mktemp("authoff-multiuser")
    osprey_bin = _find_osprey_console_script()
    repo = _make_repo(tmp_path, osprey_bin)

    _teardown()  # clear anything a previous crashed run stranded under these names
    try:
        # Run from inside the repo: every lifecycle verb walks up to the nearest
        # profile.yml, so standing in the deployment is what selects it.
        up = _run_osprey(osprey_bin, ["up", "--dev"], repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up.returncode == 0, _fmt("osprey up --dev (auth-off multiuser)", up)
        _wait_for_gate(USER, READY_TIMEOUT_SEC)
        yield {"repo": repo, "secret": _read_user_secret(repo, USER)}
    finally:
        # The shipped teardown first — it stops the web stack the way the
        # lifecycle does — then the exact-named sweep as a safety net.
        _run_osprey(osprey_bin, ["down"], repo)
        _teardown()


def test_up_builds_the_real_persona_image(deployment: dict[str, Any]) -> None:
    """The DEPLOYED persona image built, and the per-user container is running.

    Everything below runs against a real terminal app; this pins that the app is
    the real one — the persona ``:local`` tag was built and ``alice``'s container
    exists — rather than a stub, so a green auth assertion cannot be a green stub.
    """
    assert _image_exists(PERSONA_IMAGE_TAG), f"{PERSONA_IMAGE_TAG} was not built by 'osprey up'"
    assert _container_id(_web_container(USER)) is not None, (
        f"{_web_container(USER)} was not created"
    )


def test_uncredentialed_mutation_is_refused_by_the_app(deployment: dict[str, Any]) -> None:
    """T1: with auth off, the app itself refuses an uncredentialed mutating request.

    nginx passed this request straight through — it ran no ``auth_request`` and
    injected no operator-secret header, because ``auth.method: none`` — so the
    ``401`` can only be the terminal app's own ``WebAuthMiddleware``. That is the
    whole claim of the default posture: the chain is closed by the app-owned
    gate even when nothing in front of it authenticates anything.
    """
    status, _, body = _request(
        f"/u/{USER}/api/config",
        method="PATCH",
        data=_CONFIG_PATCH_BODY,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    assert status == 401, (
        f"uncredentialed mutation was not refused by the app (got {status})\n"
        f"{body[:300]}\n{_logs(_web_container(USER))}"
    )


def test_token_login_unlocks_the_mutation(deployment: dict[str, Any]) -> None:
    """T2: the per-user ``?token=`` login URL exchanges for a cookie that unlocks it.

    The token is ``alice``'s minted operator secret. A cookie-less browser
    following ``/u/alice/?token=<secret>`` clears the app gate on that one GET
    (the exchange path), and the handler answers ``303`` while setting the
    session cookie. nginx forwards that cookie on the way back — auth is off, so
    it strips nothing — and the SAME ``PATCH`` that was refused above, now
    carrying the cookie and this deployment's own ``Origin``, is accepted with
    ``200``. The gate the browser had to pass is the app's, and the credential
    that opened it is the secret the deployment minted.
    """
    client, jar = _browser()

    token = urllib.parse.quote(deployment["secret"], safe="")
    # Follow the login URL. The exchange sets the session cookie on the 303; the
    # jar captures it even though the opener then follows the redirect on to a
    # clean URL. What matters is the cookie, not where the redirect lands.
    _request(f"/u/{USER}/?token={token}", headers=_navigation(), opener=client)
    assert [c.name for c in jar], (
        f"token exchange issued no session cookie\n{_logs(_web_container(USER))}"
    )

    status, _, body = _request(
        f"/u/{USER}/api/config",
        method="PATCH",
        data=_CONFIG_PATCH_BODY,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            # A mutating cookie-authenticated request is Origin-checked by the
            # app; the deployment's external origin is its published nginx port.
            "Origin": f"http://127.0.0.1:{NGINX_PORT}",
        },
        opener=client,
    )
    assert status == 200, (
        f"cookie-authenticated mutation was not accepted (got {status})\n"
        f"{body[:300]}\n{_logs(_web_container(USER))}"
    )
