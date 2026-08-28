"""Acceptance proof for the two multi-user postures that run NO auth sidecar:
``auth.method: token`` (the default) and ``auth.method: none`` (open).

WHY THESE LANES EXIST ALONGSIDE ``test_auth_perimeter.py``
----------------------------------------------------------
``test_auth_perimeter.py`` proves the *walled* perimeter: with
``auth.method: password`` nginx runs an ``auth_request`` against the sidecar,
injects the per-user operator secret as a header on the way through, and strips
the browser's cookie. There the sidecar is the front door and nginx vouches for
every request that reaches a container — but only for a request that already
cleared a login.

The two postures proved here have no sidecar and no login wall at all, and they
differ in exactly one thing: whether nginx vouches for the caller.

  ``token``  nginx runs no ``auth_request``, injects NO operator-secret header,
             and passes the browser's cookie straight through. Nothing in front
             of the container authenticates anything, so the per-user terminal
             app's own ``WebAuthMiddleware`` is the only thing standing between
             one user's browser and another user's terminal. A browser gets in
             by presenting that user's minted secret once, as the ``?token=``
             magic link, which the app exchanges for a session cookie.

  ``none``   OPEN, navigation-only. nginx injects each user's operator secret on
             every non-exempt ``/u/<user>/`` location, so anyone who can reach
             nginx is served that user's terminal with no credential of their
             own. Navigation IS the authentication. A roster entry that declared
             ``login: false`` is deliberately left OUT of that injection: it is
             proxied ungated, and the app's own gate is then the only one in
             front of it.

``token`` is what an absent ``auth:`` block renders, and it carries what
``auth.method: none`` used to mean before the four-method scheme. That the two
spellings render the same bytes is pinned line by line in
``tests/deployment/web_terminals/test_auth_off_baseline_pin.py``; this module
spells the method explicitly and does not re-prove the equality.

WHAT THIS FILE ASSERTS, ON REAL STACKS
--------------------------------------
Two real ``osprey up --dev`` deployments — nginx plus one per-user terminal
container per roster entry, reached over each deployment's published
``nginx_port`` with real HTTP clients.

Against the ``token`` deployment (roster: ``alice``):

  T1  A mutating request to ``alice``'s terminal API carrying NO credential —
      no cookie, and (because ``token`` injects nothing) no header nginx could
      have supplied — is refused with ``401`` by the app itself, even though
      nginx passed it through untouched.
  T2  The per-user ``?token=`` login URL, whose token is that user's minted
      operator secret, is exchanged by a browser for a session cookie; the SAME
      mutating request carrying that cookie is then accepted with ``200``.

Against the ``open`` deployment (roster: ``alice``; ``kiosk`` with
``login: false``):

  O1  A cookie-less, token-less client both NAVIGATES to ``/u/alice/`` and
      performs the mutating ``PATCH`` that T1 was refused — and is served
      ``200`` for both. The credential it never presented was injected by
      nginx; navigation alone reached a credentialed terminal.
  O2  The exempt ``kiosk`` entry is proxied by that same nginx but carries no
      injected secret, so the identical requests are refused ``401`` — and
      refused by the terminal APP (a JSON refusal body), not by the perimeter.
      One deployment, one image, two locations: the difference is the injection.
  O3  Every per-user container carries the perimeter stamp
      (``OSPREY_WEB_PERIMETER=open`` plus this deployment's own web ports), and
      code executed through the shipped ``ExecutionWrapper`` INSIDE one of those
      containers is refused a connection to the deployment's nginx port.

WHAT O3 PROVES, AND WHAT IT DOES NOT
------------------------------------
It proves the whole chain that is specific to a deployment: the render stamped
the right ports onto the container, the executor's own parent-side reader parses
that stamp back inside the real image, and a wrapped child built from it refuses
the deployment's live nginx port. It runs an unarmed control child in the same
container first, which DOES connect — so the refusal is a refusal and not a port
that nothing was listening on.

It does NOT prove containment. The guard is emitted source in the child's own
namespace (``services/python_executor/execution/net_guard.py`` says so itself:
defense in depth, not a security boundary), so code that knows it is there takes
it down, and a shell or a browser tool never goes through it at all. That hole
is closed at deploy time instead, by the gate that refuses an open deployment
whose personas do not deny ``Bash``/``WebFetch``/``WebSearch``/Playwright — which
is why this lane's persona ships the rendered deny list and why an open
``osprey up`` here would otherwise be REFUSED rather than merely unsafe. Nor does
it drive the MCP server: that the executor calls the reader on every execution is
pinned by unit tests, not here.

THE TERMINAL BEHIND NGINX IS REAL, NOT A STUB
---------------------------------------------
``test_auth_perimeter.py`` puts a one-page busybox stub behind nginx, because
its subject is the perimeter and a real terminal image would only add minutes to
a test about the sidecar. Here the terminal IS the subject: the gate under test
lives in the terminal app, so the container behind nginx must run the real
``osprey web`` — the actual ``WebAuthMiddleware``, the actual ``?token=`` →
cookie exchange handler, the actual ``PATCH /api/config`` operator route, and
(for O3) the actual installed framework. A stub would prove nothing.

The persona image is therefore a REAL framework install (``pip install
osprey-framework`` with a C toolchain plus the Claude Code CLI — minutes, cold),
which is why this file carries the ``dockerbuild`` marker and runs in its own CI
lane rather than the shared fast e2e glob. Its recipe is not hand-rolled: the
fixture renders a throwaway ``hello-world`` deployment with the shipped ``osprey
init``/``osprey build`` and reuses that render's own artifacts — the production
project ``Dockerfile`` that runs ``osprey web``, a real rendered ``config.yml``
the app can serve, and the rendered ``.claude/settings.json`` whose deny list the
open-mode deploy gate reads.

ONE PERSONA, TWO DEPLOYMENTS. The two lanes share a single persona project and
therefore a single image tag, so the expensive build happens once and the second
``osprey up`` re-runs it against a byte-identical context (a runtime cache hit).
They differ only in facility prefix, compose project and port band, so both
stacks can be up at once without colliding.

``--dev`` is REQUIRED for the same reason it is in the perimeter lane: the
project Dockerfile pins ``osprey-framework`` to this deployment's version, which
a source checkout does not publish to PyPI. ``--dev`` sets ``OSPREY_DEV=1``
(relaxing the pin miss to an unpinned prime) and stages the locally-built wheel
the image then overlays, so the terminal that runs here is this branch's code.

CONTAINER-OPS SAFETY: every runtime-mutating call below names an EXACT resource
this test created — the ``<prefix>-nginx`` / ``<prefix>-web-<user>`` containers,
each lane's own volumes, and the shared ``:local`` image tag — or is the
project-scoped ``compose down`` the deploy lifecycle itself uses. Nothing here
ever runs a prune, an ``-a``/``--all`` sweep, or a wildcard removal.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment.web_terminals.auth_credentials import terminal_secret_var
from osprey.mcp_server.sandbox_env import PERIMETER_DENY_PORTS_ENV, PERIMETER_MARKER_ENV
from osprey.services.python_executor.execution.net_guard import NET_GUARD_REFUSAL_PREFIX
from osprey.utils.dotenv import parse_dotenv_file
from tests.e2e._volumes import remove_project_volumes

# ``dockerbuild`` is load-bearing, not descriptive: a guard in
# tests/deployment/test_ci_workflow_wiring.py requires every file carrying it to
# be --ignore'd in the shared e2e-tests lane AND given its own job. Without the
# marker this file would be swept into that lane's glob over tests/e2e/ and the
# expensive framework build would run in the wrong lane. It is also why both
# postures live in ONE module: a second file would need its own ignore entry and
# its own job, and would pay for the framework build a second time.
pytestmark = [pytest.mark.e2e, pytest.mark.dockerbuild]

_SUPPORTED_RUNTIMES = ("docker", "podman")
RUNTIME = os.environ.get("OSPREY_E2E_RUNTIME", "docker")
if RUNTIME not in _SUPPORTED_RUNTIMES:
    raise RuntimeError(
        f"OSPREY_E2E_RUNTIME={RUNTIME!r} is not supported; expected one of {_SUPPORTED_RUNTIMES}"
    )

#: The persona both lanes run, and the render its image is tagged for. Named
#: independently of either lane's facility prefix precisely because it is
#: SHARED: one render, one ``<project>:local`` image, built once and reused by
#: the second deployment. It must also differ from either deployment's own name
#: — lint refuses a persona project that shadows the deployment's worker tag.
PERSONA = "operator"
PERSONA_PROJECT = "authmulti-operator"
PERSONA_IMAGE_TAG = f"{PERSONA_PROJECT}:local"
#: The preset the throwaway persona render (and both host repos) are built from
#: — the smallest one that stands up a serving ``osprey web``.
PRESET = "hello-world"


@dataclass(frozen=True)
class Lane:
    """One deployment under test: a posture, a roster, and a port band.

    Everything that must differ between two stacks standing up at the same time
    is here rather than in module constants, so a helper cannot silently act on
    the wrong deployment: the facility prefix decides every container name, the
    project name decides the compose project and its volumes, and the port band
    decides what nginx and each terminal bind.

    ``users`` holds roster entries as they are authored in the profile. The
    ``open`` lane must use OBJECT entries: a bare-string entry runs no persona,
    so the open-mode deploy gate reads the DEPLOY project's settings.json for it
    (the zero-migration sentinel) instead of the persona's, and refuses a
    deployment whose persona is in fact clean.
    """

    posture: str
    project_name: str
    prefix: str
    nginx_port: int
    base_ports: dict[str, int]
    users: tuple[dict[str, Any] | str, ...]

    @property
    def usernames(self) -> tuple[str, ...]:
        return tuple(user if isinstance(user, str) else str(user["name"]) for user in self.users)

    @property
    def external_origin(self) -> str:
        """What ``deployment_external_origin`` derives for this lane.

        ``deploy.fqdn`` is ``127.0.0.1`` and TLS is off in both lanes, so the
        origin a browser reaches is the published nginx port — which is what the
        app checks a mutating request's ``Origin`` against.
        """
        return f"http://127.0.0.1:{self.nginx_port}"

    @property
    def web_ports(self) -> tuple[int, ...]:
        """Each roster user's terminal port, allocated as the render allocates it.

        ``allocate_ports`` adds the entry's INDEX to the family base, so the
        index is read off an object entry rather than taken from its position —
        the two agree here, and would stop agreeing the moment a roster declared
        them apart.
        """
        return tuple(
            self.base_ports["web"] + (position if isinstance(user, str) else int(user["index"]))
            for position, user in enumerate(self.users)
        )

    def container(self, user: str) -> str:
        return f"{self.prefix}-web-{user}"

    @property
    def nginx_container(self) -> str:
        return f"{self.prefix}-nginx"


# Ports well clear of every other stack a developer may have up (the tutorial
# stacks hold 5064/5080/5432; the lifecycle fixtures the 9000s/19081; the auth
# perimeter lane the 191xx/192xx band). Each lane takes a distinct slice of the
# 196xx/197xx-199xx band, because both stacks are up simultaneously: pytest
# tears a module-scoped fixture down at the end of the module, not at the end of
# the tests that used it.
TOKEN_LANE = Lane(
    posture="token",
    project_name="osprey-e2e-token-multiuser",
    prefix="authtok",
    nginx_port=19680,
    base_ports={
        "web": 19671,
        "artifact": 19771,
        "ariel": 19871,
        "lattice": 19971,
        "channel_finder": 19981,
    },
    #: One user is enough for the token proof: the gate under test is the app's
    #: own, and a second roster user would only add a second heavy container to
    #: a proof about one terminal's front door.
    users=("alice",),
)

OPEN_LANE = Lane(
    posture="none",
    project_name="osprey-e2e-open-multiuser",
    prefix="authopen",
    nginx_port=19690,
    base_ports={
        "web": 19691,
        "artifact": 19791,
        "ariel": 19891,
        "lattice": 19941,
        "channel_finder": 19951,
    },
    #: TWO entries, because the open posture's claim is a CONTRAST: `alice` is
    #: injected into, `kiosk` declared `login: false` and is not. One container
    #: each, from the same already-built image.
    users=(
        # `index` is spelled out because an OBJECT entry must carry one — profile
        # validation refuses an object roster entry with no index, since the
        # index (not the position) is what allocates every port family.
        {"name": "alice", "index": 0, "persona": PERSONA},
        {"name": "kiosk", "index": 1, "persona": PERSONA, "login": False},
    ),
)

# The persona build is a real framework install; give it room.
DEPLOY_UP_TIMEOUT_SEC = 1800
VERB_TIMEOUT_SEC = 120
# `init` + `build` render the whole repo (no dependency install, no lifecycle
# hooks — see the flags below), which is filesystem work rather than network.
RENDER_TIMEOUT_SEC = 600
# `osprey web` boot inside the container (import the framework, start the app)
# is slower than a static stub, so readiness is polled generously.
READY_TIMEOUT_SEC = 240.0
# The in-container guard probe imports the framework twice (the probe itself,
# then the wrapped child it spawns) on a container that is already serving.
GUARD_PROBE_TIMEOUT_SEC = 300

#: A benign, valid mutating update for the operator-only ``PATCH /api/config``
#: route: a single dot-notation field that ``config_update_fields`` can write to
#: the served ``config.yml``. The point is the STATUS the gate returns, not the
#: value written — a display label mutates nothing the deployment relies on.
_CONFIG_PATCH_BODY = json.dumps({"updates": {"web.app_name": "multiuser-e2e"}}).encode()

# ZO_INGEST_SA_TOKEN is present but inert: hello-world renders an OpenObserve
# telemetry block whose password is `${ZO_INGEST_SA_TOKEN}`, and the deploy
# preflight statically refuses to generate `.env.users` while that reference is
# unresolvable — even though the block is disabled and the store is not deployed
# here. Supplying a value satisfies the resolver and is the documented remedy for
# a deployment behind an external store; with telemetry off the app never emits,
# so the value is never used.
_ENV_CONTENT = "ANTHROPIC_API_KEY=fake-llm-key-value\nZO_INGEST_SA_TOKEN=fake-telemetry-token\n"

#: The O3 probe, run by the container's own interpreter inside a per-user
#: terminal container. Two children, one difference:
#:
#: * ARMED — an ``ExecutionWrapper`` built from the ports the executor's OWN
#:   parent-side reader parsed out of this container's environment. Imported by
#:   its real name on purpose: the private helper is the seam the executor calls
#:   on every execution, and a rename that left the stamp unread should break
#:   this probe rather than pass it.
#: * CONTROL — the same wrapper with no denied ports, which must CONNECT. It is
#:   what makes the armed refusal mean something: without it, a refusal is
#:   indistinguishable from nothing listening on that port.
#:
#: Answers on one ``OSPREY_PROBE`` line of JSON so the assertions read structured
#: data rather than scraping two children's interleaved output.
_GUARD_PROBE = """
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from osprey.mcp_server.python_executor.executor import _perimeter_denied_ports
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

PROBE_CODE = (
    "import socket\\n"
    "try:\\n"
    "    socket.create_connection(('127.0.0.1', __NGINX_PORT__), timeout=5).close()\\n"
    "    print('PROBE-CONNECTED')\\n"
    "except Exception as exc:\\n"
    "    print('PROBE-REFUSED', exc)\\n"
)


def run(denied_ports):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        script = folder / "wrapped_script.py"
        wrapper = ExecutionWrapper(perimeter_denied_ports=denied_ports)
        script.write_text(wrapper.create_wrapper(PROBE_CODE, folder), encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=120
        )
    return {"rc": done.returncode, "out": done.stdout, "err": done.stderr}


ports = _perimeter_denied_ports(os.environ)
print(
    "OSPREY_PROBE "
    + json.dumps({"ports": list(ports), "armed": run(ports), "control": run(())})
)
"""


def _runtime_cli(
    *args: str, timeout: int = 30, input_text: str | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [RUNTIME, *args], capture_output=True, text=True, timeout=timeout, input=input_text
    )


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


def _container_env(name: str) -> dict[str, str]:
    """The environment a process started in *name* inherits.

    Read from the running container rather than from the rendered compose file:
    the claim under test is that the stamp reached the process, and a compose
    line proves only that it was written down.
    """
    result = _runtime_cli("exec", name, "env", timeout=30)
    assert result.returncode == 0, _fmt(f"{RUNTIME} exec {name} env", result)
    env: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        if key:
            env[key] = value
    return env


def _render_reference_project(tmp_path: Path, osprey_bin: Path) -> Path:
    """Render a throwaway ``hello-world`` deployment and return its ``build/`` dir.

    The persona both lanes deploy must run the REAL ``osprey web``, which needs a
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
    # store these lanes do not deploy, so its rendered config.yml would carry
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
      (``build/.image/<persona project>/`` — ``profile.yml`` at its root, the
      render in ``build/`` below it), copied verbatim. Its ``Dockerfile`` is the
      production project image: ``COPY . /app/<name>/``, then ``osprey web`` from
      that directory, which walks up to the ``profile.yml`` this copy provides
      and serves its ``build/config.yml``. That is what makes the terminal behind
      nginx a real app rather than a stub.

    The flat render's ``.claude/`` tree comes along for a reason the open lane
    depends on: ``check_open_mode_requirements`` reads this directory's
    ``.claude/settings.json`` and REFUSES ``osprey up`` unless it denies
    ``Bash``, ``WebFetch``, ``WebSearch`` and the Playwright wildcard. Copying
    the rendered artifact rather than writing a deny list here is what keeps the
    lane honest: it proves the SHIPPED render satisfies the gate, so a future
    change to ``DENY_DEFAULTS`` that broke every open deployment would fail this
    lane instead of being papered over by a fixture's hand-made JSON.

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

    # The flat render's required files, from the same reference build.
    shutil.copy2(ref_build / "config.yml", root / "config.yml")
    shutil.copy2(ref_build / "Dockerfile", root / "Dockerfile")
    shutil.copytree(ref_build / ".claude", root / ".claude")
    return root


def _override_text(lane: Lane) -> str:
    """The ``-O`` overlay carrying one lane's whole web-terminal stanza.

    Dotted leaf keys under ``config:``, the one spelling a profile's config
    block accepts — and ``modules.web_terminals`` deliberately as ONE dotted key
    with a nested value, so it sets that subtree without replacing the rendered
    ``modules:`` mapping around it.

    ``auth.method`` is spelled explicitly in both lanes, ``token`` included even
    though an absent ``auth:`` block renders it: a reader of this file should not
    have to know the default to know which posture is under test. The two
    spellings' byte equality is pinned in
    ``tests/deployment/web_terminals/test_auth_off_baseline_pin.py``.

    Neither posture carries a TLS obligation — the render requires TLS or
    ``allow_insecure_http`` only when a sidecar would send session cookies across
    the network — so no certificate management enters either lane.

    ``deployed_services: []`` drops everything the preset would otherwise deploy:
    these lanes deploy the web tier and nothing else, and a backend service would
    only add containers and bound ports to a proof about nginx. Telemetry goes
    off with them, and for the same reason the perimeter lane disables it — the
    preset ships it aimed at a store this deploy no longer runs.
    """
    return yaml.safe_dump(
        {
            "config": {
                "container_runtime": RUNTIME,
                "facility.name": f"E2E Multiuser Fixture ({lane.posture})",
                "facility.prefix": lane.prefix,
                "facility.timezone": "UTC",
                "deploy.fqdn": "127.0.0.1",
                "deployed_services": [],
                "claude_code.telemetry.enabled": False,
                "modules.web_terminals": {
                    "enabled": True,
                    "image_source": "local",
                    "default_persona": PERSONA,
                    "nginx_port": lane.nginx_port,
                    "web_base_port": lane.base_ports["web"],
                    "artifact_base_port": lane.base_ports["artifact"],
                    "ariel_base_port": lane.base_ports["ariel"],
                    "lattice_base_port": lane.base_ports["lattice"],
                    "channel_finder_base_port": lane.base_ports["channel_finder"],
                    "users": [
                        dict(user) if isinstance(user, dict) else user for user in lane.users
                    ],
                    "auth": {"method": lane.posture},
                },
            }
        },
        sort_keys=False,
    )


def _add_persona_catalog(repo: Path, persona_path: Path) -> None:
    """Point the profile's persona catalog at the reused persona project.

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


def _make_repo(lane: Lane, tmp_path: Path, osprey_bin: Path, persona_path: Path) -> Path:
    """``init`` + persona catalog + ``build`` — the repo one lane deploys.

    ``--skip-deps``/``--skip-lifecycle`` keep the render off the network and
    quick: nothing here runs the deployment's own venv, only its containers.
    """
    repo = tmp_path / lane.project_name
    override_path = tmp_path / f"{lane.prefix}-override.yml"
    override_path.write_text(_override_text(lane), encoding="utf-8")

    init = _run_osprey(
        osprey_bin,
        ["init", str(repo), "--preset", PRESET, "--no-git", "--override", str(override_path)],
        tmp_path,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert init.returncode == 0, _fmt(f"osprey init ({lane.posture} multiuser)", init)

    _add_persona_catalog(repo, persona_path)

    build = _run_osprey(
        osprey_bin,
        ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle", "--dev"],
        tmp_path,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert build.returncode == 0, _fmt(f"osprey build ({lane.posture} multiuser)", build)

    # The repo root's .env is this deployment's secret store: `osprey up` refuses
    # to start without it, the provider-secret gate reads the key, and
    # `ensure_terminal_secrets` mints each user's OSPREY_TERMINAL_SECRET_<USER>
    # into it — the value the token lane reads back to build the login URL.
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


def _teardown(lane: Lane) -> None:
    """Exact-named sweep for one lane; failures swallowed (never an assertion).

    The volume sweep is what keeps reruns honest: ``compose down`` (like
    ``osprey down``) keeps named volumes, and a rerun inheriting a previous
    attempt's per-user volumes would start from that attempt's state.

    The persona image is NOT removed here: both lanes share one tag, so removing
    it with the other stack still running would untag an image in use. That
    removal belongs to the fixture that owns the build (:func:`persona_project`),
    which is set up first and therefore torn down last.
    """
    for user in lane.usernames:
        _runtime_cli("rm", "-f", lane.container(user))
    _runtime_cli("rm", "-f", lane.nginx_container)
    _runtime_cli("compose", "-p", lane.project_name, "down", timeout=60)
    remove_project_volumes(lane.project_name, runtime=RUNTIME)


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
    lane: Lane,
    target: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    opener: urllib.request.OpenerDirector | None = None,
) -> tuple[int, dict[str, str], str]:
    """One HTTP request, returning (status, headers, body) and never raising on 4xx/5xx.

    Pass ``opener`` (from :func:`_browser`) to carry a session across requests;
    omit it for the unauthenticated probes, which must carry nothing — and which
    is every probe in the open lane, where carrying nothing is the point.
    """
    req = urllib.request.Request(  # noqa: S310 - loopback only
        f"http://127.0.0.1:{lane.nginx_port}{target}", method=method, data=data
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


def _patch_config(
    lane: Lane, user: str, *, opener: urllib.request.OpenerDirector | None = None
) -> tuple[int, dict[str, str], str]:
    """The operator-only mutating request every posture is judged on.

    One call shape across both lanes, so the postures differ in their ANSWER and
    in nothing else. ``Origin`` is always this deployment's own — it is what a
    browser at the published nginx port would send, and the app checks it on a
    mutating request under both the cookie credential (strictly) and the
    nginx-injected header (leniently).
    """
    return _request(
        lane,
        f"/u/{user}/api/config",
        method="PATCH",
        data=_CONFIG_PATCH_BODY,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Origin": lane.external_origin,
        },
        opener=opener,
    )


def _wait_for_gate(lane: Lane, user: str, expected: int, timeout: float) -> None:
    """Poll one user's terminal through nginx until its gate answers *expected*.

    Readiness here is not "nginx is up" but "the terminal app behind it is up and
    deciding" — the mutating probe coming back with the status this posture owes
    proves the real ``WebAuthMiddleware`` is running (``401`` where nothing
    vouches for the caller, ``200`` where nginx injected the secret), which is
    the precondition for every assertion below. A ``502`` means nginx is up but
    the container is still booting (`osprey web` is a real framework import), so
    it keeps polling; a refused connection means nginx itself is not listening
    yet. On timeout the message carries both containers' logs, since a persona
    that never came up is the most likely cause and says so only there.
    """
    deadline = time.monotonic() + timeout
    last = "(no attempt yet)"
    while time.monotonic() < deadline:
        try:
            status, _, body = _patch_config(lane, user)
            if status == expected:
                return
            last = f"HTTP {status}: {body[:200]}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last = str(exc)
        time.sleep(3.0)
    raise AssertionError(
        f"{user}'s terminal gate did not answer {expected} within {timeout:.0f}s "
        f"(last: {last})\n{_logs(lane.nginx_container)}\n{_logs(lane.container(user))}"
    )


def _deploy(lane: Lane, tmp_path: Path, persona_path: Path) -> Iterator[dict[str, Any]]:
    """Stand one lane up, hand it to its tests, and take it down again.

    The body both lane fixtures share, so the two deployments cannot come to
    differ in how they are started, waited for, or cleaned up — only in the
    :class:`Lane` they are handed and in the readiness each posture owes.
    """
    osprey_bin = _find_osprey_console_script()
    repo = _make_repo(lane, tmp_path, osprey_bin, persona_path)

    _teardown(lane)  # clear anything a previous crashed run stranded under these names
    try:
        # Run from inside the repo: every lifecycle verb walks up to the nearest
        # profile.yml, so standing in the deployment is what selects it.
        up = _run_osprey(osprey_bin, ["up", "--dev"], repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up.returncode == 0, _fmt(f"osprey up --dev ({lane.posture} multiuser)", up)
        for user in lane.usernames:
            # Under `open` the injected location answers 200 and the exempt one
            # still 401s; under `token` nothing is injected anywhere.
            expected = 200 if lane.posture == "none" and _is_injected(lane, user) else 401
            _wait_for_gate(lane, user, expected, READY_TIMEOUT_SEC)
        yield {
            "lane": lane,
            "repo": repo,
            "secrets": {user: _read_user_secret(repo, user) for user in lane.usernames},
        }
    finally:
        # The shipped teardown first — it stops the web stack the way the
        # lifecycle does — then the exact-named sweep as a safety net.
        _run_osprey(osprey_bin, ["down"], repo)
        _teardown(lane)


def _is_injected(lane: Lane, user: str) -> bool:
    """Whether nginx injects *user*'s operator secret on their location.

    The roster's own ``login: false`` opt-out, read back off the authored entry:
    an exempt entry is proxied but never vouched for, whatever the method.
    """
    for entry in lane.users:
        if isinstance(entry, dict) and str(entry["name"]) == user:
            return entry.get("login") is not False
    return True


@pytest.fixture(scope="module")
def persona_project(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """The one persona project — and therefore the one image — both lanes deploy.

    Set up before either deployment and torn down after both, which is what lets
    it own the shared ``:local`` tag: a lane removing that image at its own
    teardown would be untagging an image the other lane's containers still run.
    """
    if shutil.which(RUNTIME) is None:
        pytest.skip(f"{RUNTIME} not available")
    if _runtime_cli("ps", timeout=10).returncode != 0:
        pytest.skip(f"{RUNTIME} daemon not responding")

    tmp_path = tmp_path_factory.mktemp("multiuser-persona")
    osprey_bin = _find_osprey_console_script()
    ref_build = _render_reference_project(tmp_path, osprey_bin)
    try:
        yield _write_persona_project(tmp_path / "persona", ref_build)
    finally:
        _runtime_cli("rmi", "-f", PERSONA_IMAGE_TAG, timeout=60)


@pytest.fixture(scope="module")
def token_deployment(
    tmp_path_factory: pytest.TempPathFactory, persona_project: Path
) -> Iterator[dict[str, Any]]:
    """One ``osprey up --dev`` under ``auth.method: token`` for the whole module."""
    yield from _deploy(TOKEN_LANE, tmp_path_factory.mktemp("token-multiuser"), persona_project)


@pytest.fixture(scope="module")
def open_deployment(
    tmp_path_factory: pytest.TempPathFactory, persona_project: Path
) -> Iterator[dict[str, Any]]:
    """One ``osprey up --dev`` under ``auth.method: none`` (open) for the whole module.

    That this fixture reaches ``yield`` at all is itself an assertion: the open
    deploy gate refuses ``osprey up`` unless every referenced persona's rendered
    ``.claude/settings.json`` denies the whole egress set, so a persona render
    that stopped denying them would fail here rather than deploy.
    """
    yield from _deploy(OPEN_LANE, tmp_path_factory.mktemp("open-multiuser"), persona_project)


# --------------------------------------------------------------------------
# token — the default posture: no sidecar, no injection, the app's own gate
# --------------------------------------------------------------------------


def test_up_builds_the_real_persona_image(token_deployment: dict[str, Any]) -> None:
    """The DEPLOYED persona image built, and the per-user container is running.

    Everything below runs against a real terminal app; this pins that the app is
    the real one — the persona ``:local`` tag was built and ``alice``'s container
    exists — rather than a stub, so a green auth assertion cannot be a green stub.
    """
    assert _image_exists(PERSONA_IMAGE_TAG), f"{PERSONA_IMAGE_TAG} was not built by 'osprey up'"
    assert _container_id(TOKEN_LANE.container("alice")) is not None, (
        f"{TOKEN_LANE.container('alice')} was not created"
    )


def test_uncredentialed_mutation_is_refused_by_the_app(token_deployment: dict[str, Any]) -> None:
    """T1: under ``token``, the app itself refuses an uncredentialed mutating request.

    nginx passed this request straight through — it ran no ``auth_request`` and
    injected no operator-secret header, because ``token`` injects nothing — so
    the ``401`` can only be the terminal app's own ``WebAuthMiddleware``. That is
    the whole claim of the default posture: the chain is closed by the app-owned
    gate even when nothing in front of it authenticates anything.
    """
    status, _, body = _patch_config(TOKEN_LANE, "alice")
    assert status == 401, (
        f"uncredentialed mutation was not refused by the app (got {status})\n"
        f"{body[:300]}\n{_logs(TOKEN_LANE.container('alice'))}"
    )


def test_token_login_unlocks_the_mutation(token_deployment: dict[str, Any]) -> None:
    """T2: the per-user ``?token=`` login URL exchanges for a cookie that unlocks it.

    The token is ``alice``'s minted operator secret. A cookie-less browser
    following ``/u/alice/?token=<secret>`` clears the app gate on that one GET
    (the exchange path), and the handler answers ``303`` while setting the
    session cookie. nginx forwards that cookie on the way back — ``token`` cuts
    nothing — and the SAME ``PATCH`` that was refused above, now carrying the
    cookie and this deployment's own ``Origin``, is accepted with ``200``. The
    gate the browser had to pass is the app's, and the credential that opened it
    is the secret the deployment minted.
    """
    client, jar = _browser()

    token = urllib.parse.quote(token_deployment["secrets"]["alice"], safe="")
    # Follow the login URL. The exchange sets the session cookie on the 303; the
    # jar captures it even though the opener then follows the redirect on to a
    # clean URL. What matters is the cookie, not where the redirect lands.
    _request(TOKEN_LANE, f"/u/alice/?token={token}", headers=_navigation(), opener=client)
    assert [c.name for c in jar], (
        f"token exchange issued no session cookie\n{_logs(TOKEN_LANE.container('alice'))}"
    )

    status, _, body = _patch_config(TOKEN_LANE, "alice", opener=client)
    assert status == 200, (
        f"cookie-authenticated mutation was not accepted (got {status})\n"
        f"{body[:300]}\n{_logs(TOKEN_LANE.container('alice'))}"
    )


# --------------------------------------------------------------------------
# none (open) — navigation-only: nginx vouches, the roster's exempt entry does not
# --------------------------------------------------------------------------


def test_open_navigation_reaches_a_credentialed_terminal(open_deployment: dict[str, Any]) -> None:
    """O1: no token, no cookie, no header of our own — and the terminal is ours.

    Both halves matter. The navigation ``GET`` shows a person reaching the page
    with nothing but a URL, which is the posture's promise. The ``PATCH`` — the
    very request the ``token`` lane above is refused — shows the session that
    navigation landed in is genuinely CREDENTIALED rather than merely a public
    page: an operator-only mutating route answered ``200``. The credential was
    never ours; nginx injected ``alice``'s operator secret on her location, which
    is the whole of what ``auth.method: none`` means.
    """
    status, _, body = _request(OPEN_LANE, "/u/alice/", headers=_navigation())
    assert status == 200, (
        f"navigating to alice's terminal was not served (got {status})\n"
        f"{body[:300]}\n{_logs(OPEN_LANE.container('alice'))}"
    )

    status, _, body = _patch_config(OPEN_LANE, "alice")
    assert status == 200, (
        f"the injected secret did not authenticate a mutating request (got {status})\n"
        f"{body[:300]}\n{_logs(OPEN_LANE.container('alice'))}"
    )


def test_open_exempt_entry_is_proxied_but_not_credentialed(
    open_deployment: dict[str, Any],
) -> None:
    """O2: the ``login: false`` entry is served by the same nginx and vouched for by nobody.

    ``kiosk`` declared ``login: false``, so its location is proxied without the
    secret injection its neighbour gets — the opt-out means "left out of whatever
    front door the rest of the deployment has", and under ``none`` that front
    door is the injection itself. What stands in front of ``kiosk`` afterwards is
    the app's own gate, which refuses both requests ``alice`` was served.

    That the refusal comes from the APP and not from the perimeter is asserted
    rather than assumed: nginx refuses with its own HTML error page, while the
    terminal app answers a JSON request with a JSON ``{"detail": ...}`` body. A
    ``401`` shaped that way was produced behind the proxy, which is what makes
    this a statement about the exempt entry rather than about nginx.
    """
    status, _, body = _patch_config(OPEN_LANE, "kiosk")
    assert status == 401, (
        f"the exempt entry's mutating route was not refused (got {status})\n"
        f"{body[:300]}\n{_logs(OPEN_LANE.container('kiosk'))}"
    )
    assert json.loads(body).get("detail"), (
        f"the 401 did not carry the terminal app's own refusal body: {body[:300]}"
    )

    status, _, body = _request(OPEN_LANE, "/u/kiosk/", headers=_navigation())
    assert status == 401, (
        f"navigating to the exempt terminal was credentialed after all (got {status})\n"
        f"{body[:300]}\n{_logs(OPEN_LANE.container('kiosk'))}"
    )


def test_open_stamps_the_perimeter_onto_every_user_container(
    open_deployment: dict[str, Any],
) -> None:
    """O3a: every per-user container knows it is open, and which ports that covers.

    Read out of the RUNNING containers, because the claim is that the stamp
    reaches the process the executor runs in. The deny-list is this deployment's
    own front door plus every roster user's terminal — the exact set the injected
    secret opens — and it is asserted whole rather than by membership: a stamp
    that named only the container's own port would leave a neighbour's terminal
    reachable, and would still pass an ``in`` check.

    The exempt entry is stamped too. Its location is not injected into, but its
    container shares the host network namespace with the ones that are, so code
    running there reaches ``alice``'s terminal exactly as easily.
    """
    expected = ",".join(str(port) for port in sorted({OPEN_LANE.nginx_port, *OPEN_LANE.web_ports}))
    for user in OPEN_LANE.usernames:
        env = _container_env(OPEN_LANE.container(user))
        assert env.get(PERIMETER_MARKER_ENV) == "open", (
            f"{user}'s container carries no open-perimeter marker: "
            f"{env.get(PERIMETER_MARKER_ENV)!r}"
        )
        assert env.get(PERIMETER_DENY_PORTS_ENV) == expected, (
            f"{user}'s container was stamped {env.get(PERIMETER_DENY_PORTS_ENV)!r}, "
            f"expected the deployment's own web ports {expected!r}"
        )


def test_open_refuses_executed_code_the_deployments_own_web_port(
    open_deployment: dict[str, Any],
) -> None:
    """O3b: inside the real image, a wrapped child cannot open the deployment's nginx port.

    The probe runs in ``alice``'s container, on the container's own interpreter
    and against the framework the image installed. It reads the deny-list back
    with the executor's own parent-side reader, builds the shipped
    ``ExecutionWrapper`` from it, and runs the wrapped script as a real child —
    the same sequence ``_execute_via_local`` performs for every execution.

    The CONTROL child is what makes the armed refusal a fact rather than a
    coincidence: built with no denied ports, in the same container, at the same
    moment, it connects to that port successfully. So the armed child's refusal
    is the guard refusing a reachable port, not a report about an empty socket.

    What is NOT claimed: containment. See the module docstring — the guard is a
    monkeypatch in the child's own namespace, and the deploy-time egress gate is
    what answers for a shell or a browser tool.
    """
    probe = _GUARD_PROBE.replace("__NGINX_PORT__", str(OPEN_LANE.nginx_port))
    result = _runtime_cli(
        "exec",
        "-i",
        OPEN_LANE.container("alice"),
        "python",
        "-",
        timeout=GUARD_PROBE_TIMEOUT_SEC,
        input_text=probe,
    )
    assert result.returncode == 0, _fmt("in-container perimeter guard probe", result)

    verdicts = [line for line in result.stdout.splitlines() if line.startswith("OSPREY_PROBE ")]
    assert verdicts, f"probe emitted no verdict line:\n{result.stdout}\n{result.stderr}"
    verdict = json.loads(verdicts[-1].removeprefix("OSPREY_PROBE "))

    assert OPEN_LANE.nginx_port in verdict["ports"], (
        f"the executor's own reader did not parse this deployment's nginx port "
        f"out of the container's stamp: {verdict['ports']}"
    )

    control = verdict["control"]["out"] + verdict["control"]["err"]
    assert "PROBE-CONNECTED" in control, (
        f"the control child could not reach the nginx port at all, so the armed "
        f"refusal below would prove nothing:\n{control[-800:]}"
    )

    armed = verdict["armed"]["out"] + verdict["armed"]["err"]
    assert NET_GUARD_REFUSAL_PREFIX in armed, (
        f"executed code was not refused the deployment's own web port:\n{armed[-800:]}"
    )
    assert "PROBE-CONNECTED" not in armed, (
        f"the armed child connected despite the guard:\n{armed[-800:]}"
    )
