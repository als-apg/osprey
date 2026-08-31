"""Acceptance proof for the WHOLE identity chain on real artifacts: a password
login at the deployed sidecar, an ``authorization:`` role that decides which
persona a user's container runs, the four identity headers nginx forwards
from the sidecar's answer, and the audit records those decisions leave behind in
each identity's OWN ledger subdirectory.

This lane is the automated acceptance for SC2, SC4 and SC5. Every other test of
these mechanisms renders a template, composes an app, or stubs a boundary. Here
the sidecar is the real image, the roles come out of a real ``osprey build``,
nginx is the rendered config, and the ledgers are files on the host that a real
container wrote through a real bind mount.

WHY THIS LANE EXISTS ALONGSIDE THE TWO IT REUSES
------------------------------------------------
``tests/e2e/web_terminals/test_auth_perimeter.py`` proves the *perimeter* on the
real sidecar image: login → cookie → ``auth_request`` → prefix-stripped proxy.
It keeps its deliberately stubby busybox upstream, because its subject is what
happens in FRONT of a terminal.

``tests/e2e/web_terminals/test_terminal_auth_multiuser_e2e.py`` proves the
DEFAULT posture (``auth.method: none``) with a real ``osprey web`` behind nginx,
because there the app-owned gate is the only gate.

Neither of them can answer the question this file exists for: with auth ON, does
the *identity* the sidecar established actually arrive at the container it
authorized, does the ROLE it resolved decide which persona that container runs,
and does every decision on the way land in the ledger of the identity that made
it — and only there. That question needs both halves real at once, which is why
this is the one job in the repo that builds the deployed sidecar Dockerfile AND
a real framework persona image in the same deployment.

THE DEPLOYMENT THIS BUILDS, AND WHY EACH ROSTER USER IS THERE
-------------------------------------------------------------
One ``osprey up --dev`` of a ``hello-world`` render with ``auth.method:
password``, an ``authorization:`` stanza binding two roles to two personas, and
a five-entry roster. Every entry earns its place by being the only shape that
can prove one thing:

=========  =============  ==========  =========================================
user       role           persona     what only this entry can prove
=========  =============  ==========  =========================================
``alice``  ``operator``   REAL app    A password login reaches HER container,
                                      and an in-container protected-key refusal
                                      lands in ``var/audit/alice/``.
``bob``    ``observer``   probe stub  Account, subject, role AND role source
                                      arrive at the upstream, exactly once
                                      each, carrying the sidecar's values —
                                      and client-forged headers do not.
``carol``  (none)         probe stub  A session with no role forwards the
                                      account and subject but NEITHER a role
                                      nor a role-source header at all —
                                      absent, never present-and-blank.
``kiosk``  (none)         probe stub  ``login: false``: the ungated branch
                                      forwards NONE of the four.
``dave``   (none)         probe stub  A role the boundary cannot carry refuses
                                      the login 403 instead of poisoning a
                                      header (see FAULT INJECTION below).
=========  =============  ==========  =========================================

WHAT THE ACCOUNT HEADER ASSERTIONS HERE DO AND DO NOT PROVE
-----------------------------------------------------------
This lane is password-only, and in a password deployment the account IS the
subject: both headers carry the roster username, so every assertion below
compares the two against the same value. That makes these assertions a proof
of nginx TRANSPORT — that the sidecar's ``X-Osprey-Auth-Account`` is captured
from the ``/verify`` answer, forwarded exactly once, replaces a client-forged
copy rather than joining it, and is cleared on the ungated arm — and NOT a
proof of OIDC semantics, where an account and a subject actually diverge. The
OIDC half is unit-covered instead, because the repo has no OIDC e2e harness to
drive a real IdP login through this chain.

**The persona each user runs is decided by the ROLE, never by a ``persona:``
pin.** That is not incidental to the fixture — it is the FR6 mechanism under
test. ``authorization.roles`` maps ``operator`` → the real framework persona and
``observer`` → the probe stub, ``default_persona`` catches the role-less
entries, and no roster entry here carries a ``persona:`` key at all (an entry
carrying both is refused by ``effective_persona``, deliberately). So
``test_alices_container_runs_the_persona_her_role_named`` fails the moment role
resolution stops reaching the render.

ONE REAL FRAMEWORK PERSONA, FOUR CHEAP PROBES — AND WHY THE SPLIT IS HONEST
--------------------------------------------------------------------------
The four non-alice entries run an alpine + python3 stub that answers every
request with the request's OWN headers as JSON. That is not a cost dodge that
weakens the proof; it is the only way to make the header assertions exact. The
real ``osprey web`` does not echo what it received, so a lane with only real
personas could assert that a request SUCCEEDED but never that the upstream saw
exactly one ``X-Osprey-Auth-Subject`` with the sidecar's value and no forged
one.

What actually drops a forged value is the FORWARD. A gated ``/u/<user>/``
declares ``proxy_set_header X-Osprey-Auth-Subject $osprey_auth_subject;``, and
naming a header in a location's proxy-header table replaces whatever the client
sent under that name. The template forbids a second directive for the same name
in the same location, so the COUNT below is a duplicate-forward regression
guard, not a claim that nginx skipped an empty clear. The unconditional
``proxy_set_header ... "";`` clears live in the UNGATED locations instead — the
exempt ``/u/kiosk/`` and each ``/_osprey_auth/<user>`` subrequest — where there
is no sidecar answer to forward and the clear is the only thing standing
between a client's header and the upstream. That is what
``test_the_exempt_branch_forwards_no_identity_header`` proves.

WHAT THE EXEMPT BRANCH DOES NOT PROVE HERE: it runs against the alpine stub, so
this lane shows only that nginx forwards none of the identity headers on the
ungated arm. It does not show that a ``login: false`` terminal is actually
usable by an anonymous person on a REAL image, where the same arm also
clears the operator secret and forwards only the app's own session cookie.
``tests/e2e/web_terminals/test_terminal_auth_multiuser_e2e.py`` drives that
ungated arm against a real ``osprey web``, so the pair covers it.

Alice's persona is the real thing for the complementary reason: the ledger
assertions need the real audit writer, the real ``/api/config`` protected-set
gate and the real per-user mount, none of which a stub has.

FAULT INJECTION (``dave``), STATED PLAINLY
------------------------------------------
Nothing a render can emit produces a role the identity boundary cannot carry —
lint holds role names to the username charset precisely so that cannot happen.
So the "wrong-role login is refused" arm is driven the way an operator actually
could break it: a hand-written ``OSPREY_AUTH_ROSTER_ROLE_DAVE`` in the
deployment's own ``.env.auth``, which is the sidecar's ``env_file`` and an
operator-maintained credential store (the documented home of the OIDC client
secret). ``dave`` carries no roster ``role:``, so the render emits nothing for
him and the injected value is the only source. The property under test is real
and load-bearing: the deployed sidecar refuses the LOGIN rather than minting a
session whose role it would then have to omit — "unsafe values are never
carried, and never quietly dropped either".

CI HONESTY
----------
``dockerbuild`` is load-bearing, not descriptive: the guard in
``tests/deployment/test_ci_workflow_wiring.py`` requires every file carrying it
to be ``--ignore``'d in the shared ``e2e-tests`` lane AND given its own job,
which is then registered in both halves of ``all-checks-passed`` (``needs:`` so
the roll-up waits, ``check_pr_lane`` so it cares). One more thing the lane must
do to be honest, and does: FAIL on any skipped test. pytest exits 0 on an
all-skipped run, so a runner without a container runtime — the module's only
skip path — would otherwise post a green check over a stack it never built; the
job reads its own junit report and refuses that. The job's ``if:`` carries the
epic dialect — same-repo PRs whose base is ``main`` OR ``epic/*`` — because this
lane certifies work that lands on epic phase branches first, and a safety lane
that only runs after the merge is one that can go red inside a green gate.

COST, AND THE BUDGET IT IS HELD TO
----------------------------------
Two real ``pip install osprey-framework`` builds with a C toolchain — the auth
sidecar's deployed Dockerfile and the production project image (plus the Claude
Code CLI) — in one job, cold, every run. Neighbouring lanes measure a single
such build at roughly 7–11 minutes on a runner; the deploy, the five container
starts and the assertions add a handful more, so the expected wall time is
~30–35 minutes — but the worst legal path is roughly twice that, which is why
the lane's step ceiling is 75 minutes rather than the 55 ``auth-perimeter-e2e``
and ``terminal-auth-multiuser-e2e`` carry: they build one image each, this one
builds both. ``test_ci_workflow_wiring.py`` pins that RELATION (strictly more
budget than either sibling) rather than the numbers. The stack is
built ONCE for the module and every assertion below runs against it.

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

from osprey.audit.protected import SURFACE_HTTP_CONFIG
from osprey.port_layout import DEFAULT_PORT_BASE, PORT_BASE_CONFIG_KEY, default_port
from osprey.services.auth_sidecar.identity_headers import (
    ACCOUNT_HEADER,
    ROLE_HEADER,
    ROLE_SOURCE_HEADER,
    SUBJECT_HEADER,
)
from osprey.services.auth_sidecar.sessions import SESSION_COOKIE_NAME
from tests.e2e._volumes import remove_project_volumes

# See "CI HONESTY" in the module docstring: this marker is what keeps the build
# out of the shared fast e2e glob and what the wiring guard scans for.
#: The port the project image serves on INSIDE the container: the ``web`` slot
#: of the layout, which ``Dockerfile.j2`` renders into its ``EXPOSE`` line. The
#: image is built from a config that never moves ``deployment.port_base``, so
#: the layout's default base is the right one to derive it at.
#:
#: Still true after the DEPLOYMENT moved to :data:`PORT_BASE` below, for two
#: independent reasons, and it is worth knowing both: the persona image comes
#: from :func:`_render_reference_project`, whose own override sets no base at
#: all, so its ``EXPOSE`` really is at the default base; and nothing binds this
#: number anyway, because the per-user compose service sets
#: ``OSPREY_TERMINAL_WEB_PORT``/``OSPREY_WEB_PORT`` explicitly to that user's
#: port off the DEPLOYMENT's base (``docker-compose.web.yml.j2``, which is also
#: what the healthcheck probes). This is the stub's fallback for the case that
#: never happens here, not the port it serves on.
_WEB_SLOT = default_port("web")

pytestmark = [pytest.mark.e2e, pytest.mark.dockerbuild]

_SUPPORTED_RUNTIMES = ("docker", "podman")
RUNTIME = os.environ.get("OSPREY_E2E_RUNTIME", "docker")
if RUNTIME not in _SUPPORTED_RUNTIMES:
    raise RuntimeError(
        f"OSPREY_E2E_RUNTIME={RUNTIME!r} is not supported; expected one of {_SUPPORTED_RUNTIMES}"
    )

# The repo DIRECTORY name, which is the deployment name: the compose project and
# the com.osprey.project label on every image this deploy builds.
PROJECT_NAME = "osprey-e2e-full-chain-auth"
PREFIX = "fullchain"
#: The preset the host repo and both persona projects are rendered from — the
#: smallest one that stands up a serving ``osprey web``.
PRESET = "hello-world"

#: The two catalog personas. ``operator`` is the REAL framework image; ``probe``
#: is the header-echoing stub. Their catalog *projects* differ because a
#: persona's image tag is ``<catalog project>:local`` — the render names the
#: image, so two personas sharing a project name would share a tag.
TERMINAL_PERSONA = "operator"
TERMINAL_PROJECT = f"{PREFIX}-operator"
PROBE_PERSONA = "probe"
PROBE_PROJECT = f"{PREFIX}-probe"

#: Role names, as ``authorization.roles`` declares them. ``operator`` binds the
#: real persona, ``observer`` the probe — so which image a container runs is
#: itself an assertion about role resolution.
ROLE_OPERATOR = "operator"
ROLE_OBSERVER = "observer"

#: The provenance value a password roster produces, spelled as it crosses the
#: wire rather than imported: this lane's subject is what the upstream actually
#: receives, so a rename of the sidecar's constant must fail here rather than
#: follow along. ``claim`` is the other member of the vocabulary and cannot
#: appear in this deployment, which has no IdP.
ROLE_SOURCE_ROSTER = "roster"

REAL_USER = "alice"
HEADER_USER = "bob"
NO_ROLE_USER = "carol"
EXEMPT_USER = "kiosk"
UNSAFE_ROLE_USER = "dave"
#: Declaration order is roster order, and roster order is index order.
USERS = (REAL_USER, HEADER_USER, NO_ROLE_USER, EXEMPT_USER, UNSAFE_ROLE_USER)
#: Every entry that authenticates. ``kiosk`` is exempt and gets no credential —
#: provisioning mints none for a ``login: false`` entry.
LOGIN_USERS = (REAL_USER, HEADER_USER, NO_ROLE_USER, UNSAFE_ROLE_USER)

#: This module's own thousand-port block (see test_dispatch_deploy.py's 20700
#: note): everything not pinned explicitly follows it instead of landing on a
#: real deployment's default 10000 block. Set once at render, as
#: ``deployment.port_base`` in this lane's ``-O`` overlay, and read back here to
#: derive every host port the assertions reach for — so a slot that moves in the
#: layout moves here without an edit, and no number below is a hand-assigned
#: literal that could drift from what the render actually binds.
#:
#: One knob replaces the old hand-pinned 200xx list, and the family spacing
#: changes with it: those families were spaced by TEN (20001/20011/20021…),
#: which held only because this roster is five users long. The layout spaces
#: per-user families by ONE HUNDRED — ``INDEX_MAX`` is 99 — so the five roster
#: users occupy the first five ports of each family's band and the bands cannot
#: run into each other however long a roster gets.
PORT_BASE = 23000

#: The two gateway ports this lane drives from the host, derived the same way
#: the render derives them: nginx at the base and the auth sidecar one above it.
#: The per-user families (``web``, ``artifact``, ``ariel``, ``lattice``,
#: ``channel_finder``) are NOT listed — nothing here reaches them directly, and
#: the render places them off the same base.
HOST_PORTS = {slot: default_port(slot, base=PORT_BASE) for slot in ("nginx", "auth")}

AUTH_IMAGE_TAG = f"{PREFIX}-assistant-auth:local"
# `<catalog project>:local`, exactly as resolve_personas derives a local-mode
# persona tag for a catalog entry that names its own project.
TERMINAL_IMAGE_TAG = f"{TERMINAL_PROJECT}:local"
PROBE_IMAGE_TAG = f"{PROBE_PROJECT}:local"

NGINX_C = f"{PREFIX}-nginx"
AUTH_C = f"{PREFIX}-auth"

# Two real framework installs; give them room. The step budget in CI is 55
# minutes (see the module docstring); this is the in-test ceiling below it.
DEPLOY_UP_TIMEOUT_SEC = 2400
VERB_TIMEOUT_SEC = 120
# `init` + `build` render the whole repo (no dependency install, no lifecycle
# hooks — see the flags below), which is filesystem work rather than network.
RENDER_TIMEOUT_SEC = 600
# The sidecar answers /health as soon as its app imports; the real persona's
# `osprey web` boot (import the framework, start the app) is slower.
AUTH_READY_TIMEOUT_SEC = 180.0
TERMINAL_READY_TIMEOUT_SEC = 300.0
PROBE_READY_TIMEOUT_SEC = 120.0

#: Deterministic per-user passwords, seeded into the repo ``.env`` as
#: ``OSPREY_AUTH_PW_<USER>`` — the documented way to set a password you already
#: chose. Deterministic on purpose: scraping the password ``osprey up`` prints
#: for a generated credential reads a rich-rendered line that may wrap, which
#: would split the secret at an arbitrary column.
_PASSWORDS = {user: f"{user}-full-chain-pw" for user in LOGIN_USERS}

#: The role ``dave``'s sidecar environment is poisoned with. Non-ASCII, so
#: ``is_header_safe`` refuses it; ``$``-free, because compose interpolates
#: ``${...}`` inside ``env_file`` values too. See FAULT INJECTION above.
_UNCARRIABLE_ROLE = "operatör"

#: ``.env.auth`` is APPEND-only on the provisioning path (``_append_entries``
#: opens it ``O_APPEND`` and only adds entries it cannot already find), so a
#: line written here before the first ``osprey up`` survives it and every later
#: one.
_ENV_AUTH_SEED = f"OSPREY_AUTH_ROSTER_ROLE_{UNSAFE_ROLE_USER.upper()}={_UNCARRIABLE_ROLE}\n"

# ZO_INGEST_SA_TOKEN is present but inert: hello-world renders an OpenObserve
# telemetry block whose password is `${ZO_INGEST_SA_TOKEN}`, and the deploy
# preflight statically refuses to generate `.env.users` while that reference is
# unresolvable — even though the block is disabled and the store is not deployed
# here. Supplying a value satisfies the resolver and is the documented remedy.
_ENV_CONTENT = (
    "ANTHROPIC_API_KEY=fake-llm-key-value\n"
    "ZO_INGEST_SA_TOKEN=fake-telemetry-token\n"
    + "".join(
        f"OSPREY_AUTH_PW_{user.upper()}={password}\n" for user, password in _PASSWORDS.items()
    )
)

#: Served by the probe upstream so a response can be traced to the container
#: that produced it — the positive control that separates "auth let it through"
#: from "auth let it through and the proxy pointed somewhere else".
PROBE_MARKER = "osprey-e2e-full-chain probe"

#: A benign, valid mutating update for the operator-only ``PATCH /api/config``
#: route: one dot-notation field ``config_update_fields`` may write. The point
#: is the STATUS the gate returns, not the value — a display label mutates
#: nothing the deployment relies on.
_BENIGN_PATCH_BODY = json.dumps({"updates": {"web.app_name": "full-chain-e2e"}}).encode()

#: The protected key the refusal arm aims at. ``approval.*`` is in
#: ``PROTECTED_CONFIG_KEYS`` because setting it changes what the agent may do,
#: which is exactly the class of edit the panel must refuse.
_PROTECTED_KEY = "approval.enabled"
#: The value the refused write would have set. Asserted ABSENT from every
#: record: an audit envelope carries identifiers, never values (SC2).
_PROTECTED_VALUE = "full-chain-must-never-be-recorded"
_PROTECTED_PATCH_BODY = json.dumps({"updates": {_PROTECTED_KEY: _PROTECTED_VALUE}}).encode()

#: How the running stack files that refusal — the envelope fields that identify
#: WHICH decision it was, observed against the real deployment. Pinned because
#: ``decision == "refused"`` alone would go green on a refusal from any other
#: surface in the same container (a hook, the MCP middleware, a later emitter),
#: which is not what this arm drives. The ``surface`` is imported rather than
#: spelled, so the record has to have been filed by the config route's own
#: constant and not by a lookalike; the ``subject`` is the refused KEY, which is
#: what makes this the protected-set answer and not some other refusal on the
#: same surface. If the layer that files this moves again, re-observe on a live
#: stack and re-pin — never loosen back to a bare ``refused``.
_REFUSAL_SUBJECT = _PROTECTED_KEY
_REFUSAL_REASON = "protected_key"

#: The audit zone, host side, relative to the deployment repo. Spelled once.
AUDIT_RELPATH = Path("var") / "audit"
#: The sidecar's fixed audit identity — one service, so its records name their
#: SUBJECTS rather than being filed under a username.
SIDECAR_IDENTITY = "sidecar"

#: The same zone as a TERMINAL CONTAINER sees it. The host only ever sees what
#: the render bound out, so a container's own view of ``var/audit`` is the only
#: place two things are observable at all: the MOUNT itself (what is and is not
#: under the zone from inside), and any ledger a writer puts at the audit ROOT
#: rather than under an identity subdirectory — that file lives in the
#: container's writable layer, outside every per-user bind, and no host-side
#: read can reach it.
CONTAINER_AUDIT_DIR = f"/app/{TERMINAL_PROJECT}/var/audit"


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
    it so a stub Dockerfile that ever started consuming it would fail loudly,
    but two of the images here are REAL framework installs and must resolve
    normally.
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


def _container_image(name: str) -> str | None:
    result = _runtime_cli(
        "inspect", "--type", "container", "-f", "{{.Config.Image}}", name, timeout=15
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _image_exists(tag: str) -> bool:
    return _runtime_cli("inspect", "--type", "image", tag, timeout=15).returncode == 0


def _logs(name: str) -> str:
    result = _runtime_cli("logs", "--tail", "80", name, timeout=20)
    return f"--- {name} stdout ---\n{result.stdout}\n--- {name} stderr ---\n{result.stderr}"


def _exec(container: str, *argv: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run one read-only command INSIDE an exact container this test created.

    Names a ``<prefix>-web-<user>`` container built by this deployment, never a
    pattern — see CONTAINER-OPS SAFETY in the module docstring.
    """
    return _runtime_cli("exec", container, *argv, timeout=timeout)


# ---------------------------------------------------------------------------
# The two persona projects
# ---------------------------------------------------------------------------


def _render_reference_project(root: Path, osprey_bin: Path) -> Path:
    """Render a throwaway ``hello-world`` deployment and return its ``build/``.

    The persona ``alice``'s role resolves to must run the REAL ``osprey web``,
    which needs a real container repo: the production project ``Dockerfile``
    (the one that installs the framework and launches ``osprey web``) and a
    rendered ``config.yml`` the app can actually serve. Rather than hand-roll
    either — a hand-written recipe drifts from the shipped one, and a
    hand-written config misses keys the app reads — this renders a genuine
    deployment through the same ``osprey init``/``osprey build`` surface an
    operator uses, and :func:`_write_terminal_persona_project` reuses that
    render's own artifacts.

    ``--skip-deps``/``--skip-lifecycle`` keep the render off the network and
    quick: nothing here installs a venv or builds an image, it only writes
    files.

    Rendered under the persona's OWN catalog project name on purpose: the
    production project Dockerfile bakes ``COPY . /app/<project>/`` and
    ``WORKDIR /app/<project>`` from that name, and ``resolve_personas`` derives
    the per-user ``container_project_dir`` as ``/app/<catalog project>``. Naming
    the reference render for the catalog project keeps the two equal, so the
    compose service's mount targets — the per-user AUDIT bind above all — land
    on the directory the app actually runs in.
    """
    root.mkdir(parents=True, exist_ok=True)
    ref_repo = root / TERMINAL_PROJECT
    # Telemetry OFF in the persona render too, for the same reason the host
    # override disables it: hello-world ships it enabled against an OpenObserve
    # store this lane does not deploy, so its rendered config.yml would carry
    # `openobserve.password: ${ZO_INGEST_SA_TOKEN}` — a credential only the
    # store can issue — and `osprey up` refuses to serve a terminal whose
    # telemetry names an unresolvable secret.
    ref_override = root / "persona-ref-override.yml"
    ref_override.write_text(
        yaml.safe_dump({"config": {"claude_code.telemetry.enabled": False}}, sort_keys=False),
        encoding="utf-8",
    )
    init = _run_osprey(
        osprey_bin,
        ["init", str(ref_repo), "--preset", PRESET, "--no-git", "--override", str(ref_override)],
        root,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert init.returncode == 0, _fmt("osprey init (terminal persona reference)", init)
    build = _run_osprey(
        osprey_bin,
        ["build", "--repo", str(ref_repo), "--skip-deps", "--skip-lifecycle"],
        root,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert build.returncode == 0, _fmt("osprey build (terminal persona reference)", build)
    ref_build = ref_repo / "build"
    assert (ref_build / "config.yml").is_file() and (ref_build / "Dockerfile").is_file(), (
        f"reference render is incomplete at {ref_build}"
    )
    return ref_build


def _write_terminal_persona_project(root: Path, ref_build: Path) -> Path:
    """A local-mode persona project whose container runs the REAL ``osprey web``.

    The persona is spelled the way a start requires — ``config.yml`` +
    ``Dockerfile`` in the flat render (``_check_existing_render`` reads these,
    and ``verify_persona_renders`` also parses the flat ``config.yml``), and the
    same two under the container image context at the ``.image/<name>`` sibling
    whose ``build/Dockerfile`` the image is actually built from
    (``_persona_image_context``).

    Both copies come from the throwaway render rather than being written by
    hand: the image context is the reference render's WHOLE container repo,
    copied verbatim, whose ``Dockerfile`` is the production project image
    (``COPY . /app/<name>/``, then ``osprey web`` from that directory). That is
    what makes alice's terminal a real app — with the real audit writer and the
    real ``/api/config`` protected-set gate — rather than a stub.

    Under ``osprey up --dev`` the persona build stages the locally-built wheel
    into this same image context, so the running terminal is this branch's code.
    """
    root.mkdir(parents=True, exist_ok=True)
    reference_context = ref_build / ".image" / TERMINAL_PROJECT
    assert (reference_context / "profile.yml").is_file(), (
        f"reference container repo missing profile.yml at {reference_context}"
    )

    image_context = root.parent / ".image" / root.name
    shutil.copytree(reference_context, image_context)

    shutil.copy2(ref_build / "config.yml", root / "config.yml")
    shutil.copy2(ref_build / "Dockerfile", root / "Dockerfile")
    return root


#: The probe upstream: an HTTP server whose whole job is to report the request
#: it received. ``get_all`` is the load-bearing call — it returns EVERY
#: occurrence of a header name, which is what lets the assertions below count
#: rather than merely look up, and a duplicated forward is a real failure mode
#: of a location that emits both a clear and a forward for one name.
_PROBE_SOURCE = f'''\
"""Header-reporting upstream for the full-chain auth e2e lane."""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

MARKER = {PROBE_MARKER!r}
USER = os.environ.get("OSPREY_TERMINAL_USER", "")
HOST = os.environ.get("OSPREY_TERMINAL_BIND_HOST", "127.0.0.1")
PORT = int(os.environ.get("OSPREY_TERMINAL_WEB_PORT", "{_WEB_SLOT}"))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
        if self.path == "/health":
            self._send(200, b"ok", "text/plain")
            return
        names = sorted({{name.lower() for name in self.headers.keys()}})
        payload = {{
            "marker": MARKER,
            "user": USER,
            "path": self.path,
            "headers": {{name: self.headers.get_all(name) or [] for name in names}},
        }}
        self._send(200, json.dumps(payload).encode("utf-8"), "application/json")

    def log_message(self, *args):  # noqa: A003 - silence the access log
        return


HTTPServer((HOST, PORT), Handler).serve_forever()
'''


#: The setup tool the deny entry has to name for a persona to read as unable to
#: edit its own deployment. Spelled here rather than imported so the stub's
#: config is a literal an operator could have written; the import-side spelling
#: is ``profile_conventions.SETUP_PATCH_TOOL``.
_SETUP_PATCH_TOOL = "mcp__osprey_workspace__setup_patch"


def _probe_config_yml() -> dict[str, Any]:
    """The probe persona's rendered ``config.yml`` — floors and a project name.

    The two floors are asserted TWICE by the build, at two altitudes, and the
    delta beside the profile answers only the first:

    * at the PROFILE altitude ``osprey build`` reads the catalog's
      ``build_profile`` delta (:data:`_PROBE_DELTA`);
    * at the RENDERED altitude the deploy path reads this file, the persona's
      own project config, and a persona whose project never mentions the keys
      reads as holding both surfaces.

    Both have to say the same thing, which is exactly the property the two
    checks exist to enforce — a delta that floors a tier while the render it
    produced does not is the drift they catch. Written in the NESTED spelling
    because this is a rendered ``config.yml``, not a profile ``config:`` block.
    """
    return {
        "project_name": PROBE_PROJECT,
        "web": {"config_panel": {"enabled": False}},
        "claude_code": {"permissions": {"deny": [_SETUP_PATCH_TOOL]}},
    }


def _write_probe_persona_project(root: Path) -> Path:
    """A minimal local-mode persona project whose stub REPORTS its request.

    Beyond what the idling lifecycle stub provides (a ``dispatch`` user and the
    mount-point directories the per-user compose service declares volumes onto),
    this one runs a small ``http.server`` that answers every request with the
    request's own headers as JSON. Binding ``127.0.0.1`` mirrors the production
    app: nginx shares the host network namespace and is the only thing meant to
    reach a per-user port.

    ``config.yml`` and ``Dockerfile`` are the two files a start requires of a
    persona's rendered project, in BOTH copies a real build produces: the flat
    host render at the project path, and the container copy at the
    ``.image/<name>`` sibling whose ``build/Dockerfile`` the image is built
    from, with the build CONTEXT at that sibling. ``probe.py`` therefore sits
    beside the Dockerfile and is copied as ``build/probe.py``.

    ``curl`` is installed for the per-user service's healthcheck (``curl -fsS
    http://<bind>:<port>/health``); without it the container runs but reports
    unhealthy, which reads in a failure log like a broken deployment.
    """
    root.mkdir(parents=True, exist_ok=True)
    config_text = yaml.safe_dump(_probe_config_yml(), sort_keys=False)
    (root / "config.yml").write_text(config_text, encoding="utf-8")
    container_dir = f"/app/{PROBE_PROJECT}"
    dockerfile_text = (
        "FROM alpine:3.20\n"
        "RUN apk add --no-cache python3 curl \\\n"
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
        f"    && chown -R dispatch:dispatch /data/claude-config {container_dir}\n"
        "COPY build/probe.py /srv/probe.py\n"
        'CMD ["python3", "/srv/probe.py"]\n'
    )
    (root / "Dockerfile").write_text(dockerfile_text, encoding="utf-8")

    # The container copy the image is actually built from: `docker build
    # -f <context>/build/Dockerfile <context>` with the context at the
    # `.image/<name>` sibling. The stub's recipe holds no host paths, so the
    # container copy IS the host copy.
    image_build = root.parent / ".image" / root.name / "build"
    image_build.mkdir(parents=True)
    (image_build / "config.yml").write_text(config_text, encoding="utf-8")
    (image_build / "Dockerfile").write_text(dockerfile_text, encoding="utf-8")
    (image_build / "probe.py").write_text(_PROBE_SOURCE, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# The deployment
# ---------------------------------------------------------------------------


def _roster() -> list[dict[str, Any]]:
    """The five roster entries, as ``modules.web_terminals.users``.

    Dict entries carry an explicit ``index``: ``normalize_users`` accepts a bare
    string (index = position) or a dict, and a dict without an ``index`` int is
    dropped entirely. Declaration order and index agree here, which is what the
    per-user port assignment reads.

    No entry carries a ``persona:`` key. ``effective_persona`` refuses an entry
    naming both a persona and a role — a role binds the persona, and a pin is
    the pre-authorization mechanism for the same slot — so binding by role is
    not a stylistic choice here, it is the only spelling this roster may use.
    """
    entries: list[dict[str, Any]] = []
    for index, user in enumerate(USERS):
        entry: dict[str, Any] = {"name": user, "index": index}
        if user == REAL_USER:
            entry["role"] = ROLE_OPERATOR
        elif user == HEADER_USER:
            entry["role"] = ROLE_OBSERVER
        if user == EXEMPT_USER:
            # Only the literal boolean False exempts an entry; absent or True
            # both mean "login required".
            entry["login"] = False
        entries.append(entry)
    return entries


def _override_text() -> str:
    """The ``-O`` overlay carrying this lane's whole web-terminal stanza.

    Dotted leaf keys under ``config:``, the one spelling a profile's config
    block accepts — and ``modules.web_terminals`` deliberately as ONE dotted key
    with a nested value, so it sets that subtree without replacing the rendered
    ``modules:`` mapping around it.

    ``deployment.port_base`` is the ONE port knob: nginx, the auth sidecar and
    every per-user family follow it, so this lane's whole stack moves into
    :data:`PORT_BASE`'s block with nothing else renumbered. Nothing here pins an
    individual port — a pinned port is a second source of truth for a number the
    render already derives, and the pair silently disagrees the moment the
    layout moves.

    ``allow_insecure_http`` is what lets ``auth.method: password`` render
    without TLS. That is the documented posture for a deployment behind a TLS
    terminator, and here it keeps the lane off certificate management — the TLS
    mount has its own render-level coverage.

    ``deployed_services: []`` drops everything the preset would otherwise
    deploy: this lane deploys the web tier and nothing else, and a backend
    service would only add containers and bound ports. Telemetry goes off with
    them, and for the same reason — the preset ships it aimed at a store this
    deploy no longer runs, and preflight refuses to generate ``.env.users`` for
    a telemetry block naming a credential no deploy on this config can issue.
    """
    return yaml.safe_dump(
        {
            "config": {
                "container_runtime": RUNTIME,
                "facility.name": "E2E Full-Chain Auth Fixture",
                "facility.prefix": PREFIX,
                "facility.timezone": "UTC",
                "deploy.fqdn": "127.0.0.1",
                "deployed_services": [],
                "claude_code.telemetry.enabled": False,
                PORT_BASE_CONFIG_KEY: PORT_BASE,
                "modules.web_terminals": {
                    "enabled": True,
                    "image_source": "local",
                    # The role-less entries land here, which is why the cheap
                    # persona is the default and the expensive one is reached
                    # only through a role.
                    "default_persona": PROBE_PERSONA,
                    "users": _roster(),
                    "auth": {
                        "method": "password",
                        "allow_insecure_http": True,
                    },
                    # The static half of the authorization stanza. No `claims:`
                    # half: that one is the OIDC binding, and a password login
                    # presents no ID token — the roster is the authority here.
                    "authorization": {
                        "roles": {
                            ROLE_OPERATOR: {"persona": TERMINAL_PERSONA},
                            ROLE_OBSERVER: {"persona": PROBE_PERSONA},
                        }
                    },
                },
            }
        },
        sort_keys=False,
    )


#: The probe persona's delta, ``build_profile: personas/probe.yml``. Two keys,
#: both required and both TRUE of the stub: it serves no Config panel and runs
#: no agent, so the persona holds neither deployment-editing surface.
#:
#: This is not fixture ceremony. ``kiosk`` carries ``login: false``, and the
#: build refuses an open terminal whose persona it cannot READ — "a persona
#: nobody can read cannot be shown to hold anything less than both surfaces".
#: A hand-written project has no readable delta by default, which is why the
#: two lanes this one reuses have no exempt user at all. Declaring the delta is
#: therefore the shape a facility would have to use for a public kiosk too:
#: the exempt entry runs an explicitly unprivileged tier, and the guard is
#: satisfied by the persona being harmless rather than by the check being
#: skipped.
_PROBE_DELTA = {
    "config": {
        # `web.config_panel.enabled` — the panel is ON unless a layer turns it
        # off, so the key has to be written rather than merely unmentioned.
        "web.config_panel.enabled": False,
        # `claude_code.permissions.deny` carrying the setup tool: the deny is
        # what removes the capability, and an unwritten one leaves it in the
        # agent's hands.
        "claude_code.permissions.deny": [_SETUP_PATCH_TOOL],
    }
}
_PROBE_BUILD_PROFILE = f"personas/{PROBE_PERSONA}.yml"


def _add_persona_catalog(repo: Path, terminal_path: Path, probe_path: Path) -> None:
    """Point the profile's persona catalog at the two hand-written projects.

    After materialization, deliberately. ``osprey init`` renders one delta per
    catalog entry and requires each entry to name a preset that EXTENDS the host
    preset — the shape a rendered persona has. Both personas here are projects
    supplied from elsewhere instead, so the entries are added to ``profile.yml``
    once the repo exists and before ``osprey build`` reads it, which is the
    remedy materialization itself names for a hand-written persona.

    The probe entry additionally names a ``build_profile`` and the delta is
    written beside the profile (see :data:`_PROBE_DELTA` for why the exempt
    entry forces this). One side effect is worth knowing about rather than
    being surprised by: ``osprey build`` materializes a project from that delta
    under ``build/<project name>-probe/``. Nothing uses it — the catalog's
    ``project_path`` is what every reader of a persona's project consults, and
    the image context is that path's ``.image`` sibling — and it costs a file
    render rather than an image build, so it is left alone rather than
    suppressed with a flag this lane would then be the only caller of.

    A YAML round-trip rather than a text splice: the emitted profile is plain
    YAML, and the comments a dump drops are documentation for an operator, not
    input to the build.
    """
    delta_path = repo / _PROBE_BUILD_PROFILE
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    delta_path.write_text(yaml.safe_dump(_PROBE_DELTA, sort_keys=False), encoding="utf-8")

    profile_path = repo / "profile.yml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["config"]["modules.web_terminals"]["personas"] = {
        TERMINAL_PERSONA: {"project": TERMINAL_PROJECT, "project_path": str(terminal_path)},
        PROBE_PERSONA: {
            "project": PROBE_PROJECT,
            "project_path": str(probe_path),
            "build_profile": _PROBE_BUILD_PROFILE,
        },
    }
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")


def _make_repo(tmp_path: Path, osprey_bin: Path) -> Path:
    """``init`` + persona catalog + ``build`` — the repo this lane deploys.

    ``--skip-deps``/``--skip-lifecycle`` keep the render off the network and
    quick: nothing here runs the deployment's own venv, only its containers.
    """
    ref_build = _render_reference_project(tmp_path / "reference", osprey_bin)
    # Persona project directories are named for their CATALOG PROJECT, not for
    # their role in this fixture: lint refuses a `project_path` whose basename
    # differs from the entry's `project`, because the render `osprey build`
    # writes is found by project and the two must agree. The reference render
    # gets its own parent directory so it does not collide with the terminal
    # persona's project directory of the same name.
    personas_root = tmp_path / "personas"
    terminal_path = _write_terminal_persona_project(personas_root / TERMINAL_PROJECT, ref_build)
    probe_path = _write_probe_persona_project(personas_root / PROBE_PROJECT)

    repo = tmp_path / PROJECT_NAME
    override_path = tmp_path / "override.yml"
    override_path.write_text(_override_text(), encoding="utf-8")

    init = _run_osprey(
        osprey_bin,
        ["init", str(repo), "--preset", PRESET, "--no-git", "--override", str(override_path)],
        tmp_path,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert init.returncode == 0, _fmt("osprey init (full-chain auth)", init)

    _add_persona_catalog(repo, terminal_path, probe_path)

    build = _run_osprey(
        osprey_bin,
        ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle", "--dev"],
        tmp_path,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert build.returncode == 0, _fmt("osprey build (full-chain auth)", build)

    # The one knob, read back off the render before anything binds. Every host
    # port this lane reaches is `default_port(slot, base=PORT_BASE)`, and that
    # derivation is only true if the render actually resolved the same base: an
    # overlay key that failed to land leaves the deploy on the default 10000
    # block, where it collides with a real deployment and every assertion below
    # then fails as a connection error thirty minutes into a container build.
    # Cheaper to fail here, naming the cause.
    rendered = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    resolved_base = (rendered.get("deployment") or {}).get("port_base")
    # Stricter than the shared `_orm_stack.assert_off_default_block` floor (which
    # only refuses the default block): this lane knows its own band, so it says
    # so. The two messages are written to read the same way on purpose.
    on_default_block = (
        " — they would be in the framework DEFAULT block, shared with every real "
        "deployment on this host"
        if resolved_base in (None, DEFAULT_PORT_BASE)
        else ""
    )
    assert resolved_base == PORT_BASE, (
        f"{PROJECT_NAME} (built by {__name__}) resolved "
        f"{PORT_BASE_CONFIG_KEY}={resolved_base!r}, not {PORT_BASE}: this lane's ports would "
        f"not be in its own block{on_default_block}.\n"
        f"Fix in {__name__}: `_override_text` sets `{PORT_BASE_CONFIG_KEY}=<band>` in this "
        f"lane's --override config block and PORT_BASE is the band it books — that key "
        f"failing to land is what this reads back."
    )

    # The repo root's .env is this deployment's secret store: the auth
    # provisioning reads each user's chosen OSPREY_AUTH_PW_<USER> from it, the
    # provider-secret gate reads the key, and `osprey up` refuses to start
    # without the file at all.
    env_path = repo / ".env"
    env_path.write_text(_ENV_CONTENT, encoding="utf-8")
    os.chmod(env_path, 0o600)

    # The fault injection, written BEFORE the first `osprey up`: provisioning
    # appends the hashes and the session key to whatever is already here.
    env_auth_path = repo / ".env.auth"
    env_auth_path.write_text(_ENV_AUTH_SEED, encoding="utf-8")
    os.chmod(env_auth_path, 0o600)
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
    _runtime_cli("compose", "-p", PROJECT_NAME, "down", timeout=90)
    remove_project_volumes(PROJECT_NAME, runtime=RUNTIME)
    for tag in (AUTH_IMAGE_TAG, TERMINAL_IMAGE_TAG, PROBE_IMAGE_TAG):
        _runtime_cli("rmi", "-f", tag, timeout=60)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect handler that refuses to follow, so the caller sees the 3xx.

    Load-bearing for the login, and the fixture's first real bug taught why: a
    successful login answers ``303 See Other`` pointing at the user's terminal,
    and an opener that follows it reports whatever the TERMINAL said. Alice's
    is the real ``osprey web``, minutes from serving at that moment, so a
    perfectly good login came back as the upstream's ``502`` — the assertion
    conflated "the sidecar accepted this credential" with "the container behind
    nginx has finished booting", which are two facts with two different waits.

    Returning ``None`` makes urllib raise ``HTTPError`` for the 3xx, which
    :func:`_request` already turns into a plain status. The cookie is captured
    either way: ``HTTPCookieProcessor`` runs as a response processor before
    ``HTTPErrorProcessor`` dispatches the error.
    """

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102 - see class
        return None


def _browser() -> tuple[urllib.request.OpenerDirector, http.cookiejar.CookieJar]:
    """A client that keeps cookies the way a browser does — and stops at a 3xx.

    Login sets the session cookie on the ``303`` itself, so the jar is what
    carries the session into every later request through the same opener. The
    redirect is deliberately NOT followed (see :class:`_NoRedirect`); nothing
    here needs to land on the target, and following it would make the login's
    status a statement about the upstream instead of about the login.
    """
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar), _NoRedirect())
    return opener, jar


def _request(
    target: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    port: int = HOST_PORTS["nginx"],
    opener: urllib.request.OpenerDirector | None = None,
) -> tuple[int, dict[str, str], str]:
    """One HTTP request, returning (status, headers, body); never raises on 4xx/5xx.

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
        with open_it(req, timeout=30) as resp:  # noqa: S310 - loopback only
            return resp.status, dict(resp.headers), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read().decode("utf-8", "replace")


def _navigation() -> dict[str, str]:
    """The ``Accept`` a browser sends when a person navigates to a URL."""
    return {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


#: What the sidecar answers a login it accepted. The browser is sent on to the
#: user's terminal; this lane stops here on purpose (see :class:`_NoRedirect`).
LOGIN_ACCEPTED = 303


def _login(user: str) -> tuple[urllib.request.OpenerDirector, http.cookiejar.CookieJar, int, str]:
    """Log ``user`` in at the deployed sidecar; return the opener, jar and outcome.

    The status is the SIDECAR's own answer — :data:`LOGIN_ACCEPTED` for a
    credential it took, ``403`` for a login the identity matrix refused, ``401``
    for a credential it rejected — and never the upstream's, which is a
    different question asked by a different wait.
    """
    client, jar = _browser()
    body = urllib.parse.urlencode({"user": user, "password": _PASSWORDS[user]}).encode()
    status, _, page = _request(
        "/auth/login",
        method="POST",
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            # The sidecar refuses a submission declaring no origin at all.
            "Origin": f"http://127.0.0.1:{HOST_PORTS['nginx']}",
            **_navigation(),
        },
        opener=client,
    )
    return client, jar, status, page


def _session_for(user: str) -> urllib.request.OpenerDirector:
    """A logged-in browser for ``user``, or a failure naming the sidecar's log.

    Both halves are asserted: the sidecar ACCEPTED the credential, and it issued
    a session cookie. A 303 with an empty jar would mean the browser was sent to
    a terminal it cannot open, which reads as a working login right up until the
    first authorized request.
    """
    client, jar, status, page = _login(user)
    assert status == LOGIN_ACCEPTED, (
        f"login for {user!r} was not accepted (got {status})\n{page[:300]}\n{_logs(AUTH_C)}"
    )
    assert SESSION_COOKIE_NAME in {cookie.name for cookie in jar}, (
        f"login for {user!r} issued no session cookie "
        f"(jar: {sorted(cookie.name for cookie in jar)})\n{_logs(AUTH_C)}"
    )
    return client


def _probe_report(
    user: str,
    *,
    opener: urllib.request.OpenerDirector | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """GET ``/u/<user>/`` through nginx and parse the probe's report.

    Fails with the response body rather than a JSON traceback when the upstream
    was not the probe — reaching the wrong container is the failure this helper
    is most likely to meet, and the body says which one answered.
    """
    status, _, body = _request(f"/u/{user}/", headers=headers or _navigation(), opener=opener)
    assert status == 200, (
        f"request for {user!r} did not reach the probe upstream (got {status})\n"
        f"{body[:400]}\n{_logs(NGINX_C)}\n{_logs(_web_container(user))}"
    )
    try:
        report = json.loads(body)
    except json.JSONDecodeError:  # pragma: no cover - diagnostic path
        raise AssertionError(f"upstream for {user!r} is not the probe:\n{body[:400]}") from None
    assert report.get("marker") == PROBE_MARKER, f"unexpected upstream report: {report}"
    assert report.get("user") == user, (
        f"request for {user!r} was proxied to {report.get('user')!r}'s container"
    )
    return report


def _forwarded(report: dict[str, Any], header: str) -> list[str]:
    """Every occurrence of ``header`` the upstream received, in arrival order.

    The list, not a lookup, because the COUNT is a regression guard. The gated
    location claims each identity header name with exactly one
    ``proxy_set_header <name> $var;`` forward — one directive per name per
    location is what the template requires — and that forward is what replaces
    a client's own value. A second directive for the same name would be the
    render bug this catches, and ``headers.get(name)`` would report it as a
    clean single value.
    """
    return list(report.get("headers", {}).get(header.lower(), []))


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def _wait_for_health(timeout: float) -> dict[str, Any]:
    """Poll the sidecar's own /health until it answers, or fail with its logs.

    A dead sidecar is one of the two failures this lane exists to catch, so the
    message on timeout carries the container's logs — an image that cannot
    import the app says so there and nowhere else.
    """
    deadline = time.monotonic() + timeout
    last = "(no attempt yet)"
    while time.monotonic() < deadline:
        try:
            status, _, body = _request("/health", port=HOST_PORTS["auth"])
            if status == 200:
                return json.loads(body)
            last = f"HTTP {status}: {body[:200]}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last = str(exc)
        time.sleep(2.0)
    raise AssertionError(
        f"auth sidecar /health not ready after {timeout:.0f}s (last: {last})\n{_logs(AUTH_C)}"
    )


def _wait_for_gate(user: str, timeout: float) -> None:
    """Poll a gated user's terminal until nginx REFUSES it.

    Readiness here is not "nginx is up" but "the perimeter is up and refusing":
    an unauthenticated, program-``Accept`` request that comes back ``401`` means
    nginx ran its ``auth_request`` and the sidecar answered. A ``502`` means
    nginx is up but something upstream is still booting, so it keeps polling; a
    refused connection means nginx itself is not listening yet.
    """
    deadline = time.monotonic() + timeout
    last = "(no attempt yet)"
    while time.monotonic() < deadline:
        try:
            status, _, body = _request(f"/u/{user}/", headers={"Accept": "application/json"})
            if status == 401:
                return
            last = f"HTTP {status}: {body[:200]}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last = str(exc)
        time.sleep(2.0)
    raise AssertionError(
        f"{user}'s perimeter not ready after {timeout:.0f}s (last: {last})\n"
        f"{_logs(NGINX_C)}\n{_logs(AUTH_C)}"
    )


def _wait_for_upstream(user: str, opener: urllib.request.OpenerDirector, timeout: float) -> None:
    """Poll an AUTHORIZED request for ``user`` until the upstream answers 200.

    Separate from :func:`_wait_for_gate` on purpose: the perimeter comes up with
    nginx and the sidecar, while the container behind it boots on its own clock
    — the real ``osprey web`` far more slowly than the probe. Polling the
    authorized path is the only way to tell "the app is still importing" (502)
    from "the app is up and answering".
    """
    deadline = time.monotonic() + timeout
    last = "(no attempt yet)"
    while time.monotonic() < deadline:
        try:
            status, _, body = _request(f"/u/{user}/", headers=_navigation(), opener=opener)
            if status == 200:
                return
            last = f"HTTP {status}: {body[:200]}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last = str(exc)
        time.sleep(3.0)
    raise AssertionError(
        f"{user}'s upstream never answered after {timeout:.0f}s (last: {last})\n"
        f"{_logs(NGINX_C)}\n{_logs(_web_container(user))}"
    )


# ---------------------------------------------------------------------------
# The ledgers, host side
# ---------------------------------------------------------------------------


def _ledger_records(repo: Path, identity: str) -> list[dict[str, Any]]:
    """Every record under ``var/audit/<identity>/``, tagged with its file name.

    Read from the HOST, which is the whole point: these files were written
    inside a container by a process that dropped to uid 1000, and the deploy
    provisioned the directory setgid so the host account that ran the deploy
    can read them back. A ``PermissionError`` here is a real finding about the
    mount's ownership, so it is allowed to surface rather than being swallowed
    into "no records".
    """
    directory = repo / AUDIT_RELPATH / identity
    if not directory.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:  # pragma: no cover - diagnostic path
                raise AssertionError(
                    f"{path} holds a line that is not JSON: {line[:200]}"
                ) from None
            record["_ledger"] = path.name
            records.append(record)
    return records


def _subdir_listing(repo: Path, identity: str) -> str:
    """What IS under an identity's audit subdirectory, for a failure message.

    A missing record has two very different causes — the mount never carried
    the directory, or it carried it and nothing was written — and the remedy
    differs, so the message says which one it met.
    """
    directory = repo / AUDIT_RELPATH / identity
    if not directory.is_dir():
        return "the subdirectory itself is missing"
    present = sorted(entry.name for entry in directory.glob("*"))
    return f"present: {present}" if present else "the subdirectory is empty"


def _all_records(repo: Path) -> list[dict[str, Any]]:
    """Every record under every identity's subdirectory, tagged with its owner."""
    root = repo / AUDIT_RELPATH
    if not root.is_dir():
        return []
    collected: list[dict[str, Any]] = []
    for identity_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for record in _ledger_records(repo, identity_dir.name):
            record["_identity_dir"] = identity_dir.name
            collected.append(record)
    return collected


ENVELOPE_KEYS = (
    "ts",
    "surface",
    "actor",
    "posture",
    "posture_source",
    "session",
    "subject",
    "decision",
    "reason",
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def deployment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """One ``osprey up --dev`` for the whole module — the expensive part runs once.

    Two real framework image builds live behind this fixture, so everything the
    module asserts is arranged to run against a single stack: the three sessions
    that must SUCCEED are established here and handed to the tests, rather than
    each test paying a login round trip. ``dave``'s login is deliberately not
    among them — his refusal is the observation, so it belongs in the test that
    asserts it.
    """
    if shutil.which(RUNTIME) is None:
        pytest.skip(f"{RUNTIME} not available")
    if _runtime_cli("ps", timeout=10).returncode != 0:
        pytest.skip(f"{RUNTIME} daemon not responding")

    tmp_path = tmp_path_factory.mktemp("full-chain-auth")
    osprey_bin = _find_osprey_console_script()
    repo = _make_repo(tmp_path, osprey_bin)

    _teardown()  # clear anything a previous crashed run stranded under these names
    try:
        # Run from inside the repo: every lifecycle verb walks up to the nearest
        # profile.yml, so standing in the deployment is what selects it.
        up = _run_osprey(osprey_bin, ["up", "--dev"], repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up.returncode == 0, _fmt("osprey up --dev (full-chain auth)", up)

        health = _wait_for_health(AUTH_READY_TIMEOUT_SEC)
        _wait_for_gate(REAL_USER, AUTH_READY_TIMEOUT_SEC)

        sessions = {user: _session_for(user) for user in (REAL_USER, HEADER_USER, NO_ROLE_USER)}
        _wait_for_upstream(HEADER_USER, sessions[HEADER_USER], PROBE_READY_TIMEOUT_SEC)
        _wait_for_upstream(REAL_USER, sessions[REAL_USER], TERMINAL_READY_TIMEOUT_SEC)

        yield {"repo": repo, "health": health, "sessions": sessions, "up": up}
    finally:
        # The shipped teardown first — it stops the web stack the way the
        # lifecycle does, in the order that leaves nothing behind — then the
        # exact-named sweep as a safety net for anything it could not reach.
        _run_osprey(osprey_bin, ["down"], repo)
        _teardown()


# ---------------------------------------------------------------------------
# T1 — the artifacts under test are the real ones
# ---------------------------------------------------------------------------


def test_up_builds_the_real_sidecar_and_both_persona_images(deployment: dict[str, Any]) -> None:
    """Both framework builds and the probe image landed, and the sidecar is configured.

    Every assertion below runs against these three images; this one pins that
    they are the artifacts a deployment actually builds, so a green auth result
    cannot be a green stub. ``configured: true`` is the sidecar's own report
    that the ``.env.auth`` this deploy provisioned reached it — its fail-closed
    posture with no credentials is ``configured: false`` plus a refusal on
    every other path.
    """
    assert _image_exists(AUTH_IMAGE_TAG), f"{AUTH_IMAGE_TAG} was not built by 'osprey up'"
    assert _image_exists(TERMINAL_IMAGE_TAG), f"{TERMINAL_IMAGE_TAG} was not built by 'osprey up'"
    assert _image_exists(PROBE_IMAGE_TAG), f"{PROBE_IMAGE_TAG} was not built by 'osprey up'"
    assert _container_id(AUTH_C) is not None, f"{AUTH_C} was not created"
    assert deployment["health"].get("configured") is True, (
        f"sidecar came up unconfigured: {deployment['health']}\n{_logs(AUTH_C)}"
    )


def test_alices_container_runs_the_persona_her_role_named(deployment: dict[str, Any]) -> None:
    """FR6, end to end: the ROLE decided which image each container runs.

    No roster entry here carries a ``persona:`` key — alice reaches the real
    framework persona only because ``authorization.roles`` maps her ``operator``
    role onto it, and the role-less entries fall to ``default_persona``. Break
    role resolution anywhere between the parser and ``resolve_personas`` and
    alice's container comes up on the probe image, which this compares by tag.
    """
    assert _container_image(_web_container(REAL_USER)) == TERMINAL_IMAGE_TAG, (
        f"{REAL_USER}'s role did not resolve to the {TERMINAL_PERSONA!r} persona"
    )
    for user in (HEADER_USER, NO_ROLE_USER, EXEMPT_USER, UNSAFE_ROLE_USER):
        assert _container_image(_web_container(user)) == PROBE_IMAGE_TAG, (
            f"{user} did not resolve to the {PROBE_PERSONA!r} persona"
        )


# ---------------------------------------------------------------------------
# T2–T6 — what the chain forwards, observed at the upstream
# ---------------------------------------------------------------------------


def test_a_password_login_reaches_that_users_own_terminal(deployment: dict[str, Any]) -> None:
    """SC4's positive half on the REAL app: login → cookie → auth_request → proxy.

    A 200 from a route only the app can serve proves the whole chain ran and
    landed on alice's own container: nginx authorized the request against the
    sidecar's session and proxied it with the ``/u/alice`` prefix stripped, and
    the operator secret nginx injects is what let it past the terminal app's own
    ``WebAuthMiddleware``. The two halves fail independently, which is why this
    asserts a real route rather than any 200.
    """
    status, _, body = _request(
        f"/u/{REAL_USER}/api/config",
        method="PATCH",
        data=_BENIGN_PATCH_BODY,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Origin": f"http://127.0.0.1:{HOST_PORTS['nginx']}",
        },
        opener=deployment["sessions"][REAL_USER],
    )
    assert status == 200, (
        f"authorized mutation did not reach {REAL_USER}'s terminal (got {status})\n"
        f"{body[:400]}\n{_logs(NGINX_C)}\n{_logs(_web_container(REAL_USER))}"
    )


def test_the_upstream_receives_exactly_one_of_each_identity_header(
    deployment: dict[str, Any],
) -> None:
    """SC4: the sidecar's identity arrives at the container, once, unduplicated.

    The gated location emits exactly ONE ``proxy_set_header`` per identity name
    — the FORWARD — and naming a header in a location's proxy-header table
    replaces whatever the client sent under it. The unconditional clears live in
    the UNGATED locations instead (module docstring above), and a second
    directive for the same name here is what
    ``tests/deployment/web_terminals/test_nginx_auth_surface.py::
    test_the_gated_location_forwards_instead_of_also_clearing`` forbids. So the
    count below is a duplicate-forward regression guard: the single-directive
    rule was asserted only by reading the rendered config until this lane
    existed, and a duplicated forward would leave the terminal reading whichever
    copy its header parser happened to prefer.

    The role's VALUE is the roster role ``bob``'s entry declares, which is what
    makes this an assertion about the whole binding — roster ``role:`` →
    sidecar session → verify's response headers → ``auth_request_set`` →
    forwarded header — rather than about nginx alone. The role-source header
    rides that same binding one step further: ``roster`` is the only provenance
    a password deployment can produce, so its arrival proves the third header
    travels the whole chain and is not merely rendered into the config.

    The account header is asserted against the same value as the subject
    because this lane is password-only, where the account IS the subject. So
    what its arrival proves is nginx TRANSPORT — captured from ``/verify``,
    forwarded once, uncorrupted — and not OIDC semantics, where the two
    diverge; that half is unit-covered, there being no OIDC e2e harness.
    """
    report = _probe_report(HEADER_USER, opener=deployment["sessions"][HEADER_USER])
    assert _forwarded(report, ACCOUNT_HEADER) == [HEADER_USER], (
        f"{ACCOUNT_HEADER} did not arrive exactly once carrying the roster account: {report}"
    )
    assert _forwarded(report, SUBJECT_HEADER) == [HEADER_USER], (
        f"{SUBJECT_HEADER} did not arrive exactly once carrying the roster username: {report}"
    )
    assert _forwarded(report, ROLE_HEADER) == [ROLE_OBSERVER], (
        f"{ROLE_HEADER} did not arrive exactly once carrying the roster role: {report}"
    )
    assert _forwarded(report, ROLE_SOURCE_HEADER) == [ROLE_SOURCE_ROSTER], (
        f"{ROLE_SOURCE_HEADER} did not arrive exactly once carrying the role's"
        f" roster provenance: {report}"
    )


def test_a_client_forged_identity_header_never_reaches_the_upstream(
    deployment: dict[str, Any],
) -> None:
    """SC4: no proxying location accepts a client-supplied value for any of them.

    Sent by an AUTHORIZED client, deliberately: an unauthenticated forgery is
    stopped by the perimeter and would prove nothing about the header handling.
    This request clears the gate and still must not smuggle an identity.

    The mechanism is the FORWARD, not a clear. ``/u/<user>/`` declares one
    ``proxy_set_header`` per identity header name against the
    ``auth_request_set`` variable, and claiming a name in a location's
    proxy-header table replaces whatever the client sent under it. Delete those
    forwards and the forged values arrive verbatim, which is what makes this
    assert honest; the unconditional empty clears this lane also relies on are
    in the ungated locations and are proved by
    ``test_the_exempt_branch_forwards_no_identity_header``. The forged
    role-source is the sharpest of the four: a client that could name its own
    provenance would be able to dress a roster role up as an IdP claim, so the
    gated forward has to win over the forgery there too.

    The forged account is the one a downstream authorization check would read
    as "who is this", so the forward has to REPLACE it rather than append a
    second copy — an upstream reading the first of two would be reading the
    client's. Password-only lane, so the honest value here equals the subject.
    """
    report = _probe_report(
        HEADER_USER,
        opener=deployment["sessions"][HEADER_USER],
        headers={
            **_navigation(),
            ACCOUNT_HEADER: "root",
            SUBJECT_HEADER: "root",
            ROLE_HEADER: "admin",
            ROLE_SOURCE_HEADER: "claim",
        },
    )
    assert _forwarded(report, ACCOUNT_HEADER) == [HEADER_USER], (
        f"a forged {ACCOUNT_HEADER} survived the gated location — replaced, never"
        f" appended to: {report}"
    )
    assert _forwarded(report, SUBJECT_HEADER) == [HEADER_USER], (
        f"a forged {SUBJECT_HEADER} survived the gated location: {report}"
    )
    assert _forwarded(report, ROLE_HEADER) == [ROLE_OBSERVER], (
        f"a forged {ROLE_HEADER} survived the gated location: {report}"
    )
    assert _forwarded(report, ROLE_SOURCE_HEADER) == [ROLE_SOURCE_ROSTER], (
        f"a forged {ROLE_SOURCE_HEADER} survived the gated location: {report}"
    )


def test_a_session_with_no_role_forwards_an_account_and_subject_and_no_role_header(
    deployment: dict[str, Any],
) -> None:
    """The deny-safe contract: an absent role is ABSENT, never present-and-blank.

    ``carol`` is a roster entry with no ``role:``, so her session carries the
    empty role — and every consumer must read a missing ``X-Osprey-Auth-Role``
    as "no privileges". A present-but-empty header would be read by a consumer
    checking presence as a role that exists, which is the accidental-grant this
    pins shut. Only a runtime observation can tell the two apart. The
    role-source header is absent for the same reason and by the same rule: it
    describes where a role came from, so with no role there is nothing for it
    to describe and an empty ``roster`` would be a provenance claim about a
    privilege nobody holds. The account rides the opposite rule and is asserted
    here for it: it names WHO the request is, not what they may do, so it
    arrives for every authorized session whether or not a role was resolved.
    """
    report = _probe_report(NO_ROLE_USER, opener=deployment["sessions"][NO_ROLE_USER])
    assert _forwarded(report, ACCOUNT_HEADER) == [NO_ROLE_USER], (
        f"{ACCOUNT_HEADER} did not arrive for a role-less session: {report}"
    )
    assert _forwarded(report, SUBJECT_HEADER) == [NO_ROLE_USER], (
        f"{SUBJECT_HEADER} did not arrive for a role-less session: {report}"
    )
    assert _forwarded(report, ROLE_HEADER) == [], (
        f"{ROLE_HEADER} arrived for a session that holds no role: {report}"
    )
    assert _forwarded(report, ROLE_SOURCE_HEADER) == [], (
        f"{ROLE_SOURCE_HEADER} arrived beside no role at all: {report}"
    )


def test_the_exempt_branch_forwards_no_identity_header(deployment: dict[str, Any]) -> None:
    """SC4: a ``login: false`` location establishes no identity, so it forwards none.

    ``kiosk`` sits on the ungated branch: nginx runs no ``auth_request`` for it,
    so there is no sidecar answer to forward — and the unconditional clears mean
    the terminal receives none of the four headers rather than empty ones.
    Reached with no credential at all, which is what ``login: false`` means.

    Forged headers are sent on the same request: the ungated branch is the one
    where a client could most plausibly hope to supply its own identity, and the
    clears are the only thing stopping it. All four identity headers are
    forged and all four have to be absent — the role-source clear included,
    since a provenance arriving where no identity was established would be a
    claim about a role the branch never resolved, and the account clear most of
    all, since an account arriving on the ungated arm is an unauthenticated
    client naming itself to the upstream.
    """
    report = _probe_report(
        EXEMPT_USER,
        headers={
            **_navigation(),
            ACCOUNT_HEADER: "root",
            SUBJECT_HEADER: "root",
            ROLE_HEADER: "admin",
            ROLE_SOURCE_HEADER: "claim",
        },
    )
    assert _forwarded(report, ACCOUNT_HEADER) == [], (
        f"the exempt branch forwarded {ACCOUNT_HEADER}: {report}"
    )
    assert _forwarded(report, SUBJECT_HEADER) == [], (
        f"the exempt branch forwarded {SUBJECT_HEADER}: {report}"
    )
    assert _forwarded(report, ROLE_HEADER) == [], (
        f"the exempt branch forwarded {ROLE_HEADER}: {report}"
    )
    assert _forwarded(report, ROLE_SOURCE_HEADER) == [], (
        f"the exempt branch forwarded {ROLE_SOURCE_HEADER}: {report}"
    )


def test_a_session_for_one_user_cannot_reach_anothers_terminal(
    deployment: dict[str, Any],
) -> None:
    """The perimeter is per-user, not per-deployment.

    ``alice`` holds a valid session; nginx still runs ``bob``'s own
    ``auth_request`` for ``/u/bob/``, and the sidecar answers on the identity
    the session names rather than on "is anybody logged in". Without that, one
    login would open every terminal in the roster.
    """
    status, _, body = _request(
        f"/u/{HEADER_USER}/",
        headers={"Accept": "application/json"},
        opener=deployment["sessions"][REAL_USER],
    )
    assert status == 401, (
        f"{REAL_USER}'s session reached {HEADER_USER}'s terminal (got {status})\n{body[:300]}"
    )


# ---------------------------------------------------------------------------
# T7 — a role the boundary cannot carry refuses the login
# ---------------------------------------------------------------------------


def test_a_login_whose_role_cannot_be_carried_is_refused(deployment: dict[str, Any]) -> None:
    """The identity matrix fails CLOSED on a role the header boundary cannot carry.

    ``dave``'s password is correct — the credential verifies — and the login is
    still refused ``403``, because the role his sidecar environment holds is not
    ASCII and could not cross the ``auth_request`` boundary intact. The
    alternative behaviours are both wrong and both are what this pins shut:
    minting the session and silently omitting the header (a terminal running
    with less identity than the deployment believes it forwarded), or emitting
    it and letting a proxy mangle it.

    See FAULT INJECTION in the module docstring for why this value is
    hand-written into ``.env.auth`` rather than rendered.
    """
    _client, _jar, status, page = _login(UNSAFE_ROLE_USER)
    assert status == 403, (
        f"a login carrying an uncarriable role was not refused with 403 (got {status})\n"
        f"{page[:300]}\n{_logs(AUTH_C)}"
    )


def test_the_refused_login_minted_no_session(deployment: dict[str, Any]) -> None:
    """And the refusal left the browser holding nothing.

    The matrix is asked before anything is minted precisely so a refusal cannot
    leave a usable cookie behind; this is the observable half of that ordering.
    A cookie that survived would open ``dave``'s terminal on the next request.
    """
    client, jar, status, _ = _login(UNSAFE_ROLE_USER)
    assert status == 403
    # The SESSION cookie specifically, not "the jar is empty": the login page is
    # allowed its own non-session cookies, and asserting on their absence would
    # make this test fail for a reason that has nothing to do with the matrix.
    assert SESSION_COOKIE_NAME not in {cookie.name for cookie in jar}, (
        f"a refused login left {UNSAFE_ROLE_USER} holding a session cookie"
    )
    reached, _, _ = _request(
        f"/u/{UNSAFE_ROLE_USER}/", headers={"Accept": "application/json"}, opener=client
    )
    assert reached == 401, (
        f"a refused login still reached {UNSAFE_ROLE_USER}'s terminal (got {reached})"
    )


# ---------------------------------------------------------------------------
# T8–T11 — SC2: what the run left in the ledgers
# ---------------------------------------------------------------------------


def test_the_login_events_land_in_the_sidecars_own_ledger(deployment: dict[str, Any]) -> None:
    """SC2, first half: the sidecar files its logins under ITS identity, naming the user.

    ``var/audit/sidecar/auth_sidecar.jsonl`` — the sidecar's own mount-isolated
    subdirectory, with the roster user as the record's SUBJECT and ``sidecar``
    as its ACTOR. That split is the point: a login record filed under the
    username would read as that user having written it.

    The refusal is asserted beside the success so the ledger is shown to record
    both outcomes by category, which is what makes it answerable after the fact.
    It is the refusal ``test_a_login_whose_role_cannot_be_carried_is_refused``
    drove: this test reads the trail one stack left, so it has to run after the
    tests that act, and pytest's definition order is what puts it there. Keep it
    below them.
    """
    records = _ledger_records(deployment["repo"], SIDECAR_IDENTITY)
    assert records, (
        f"the sidecar wrote no records under var/audit/{SIDECAR_IDENTITY}/\n{_logs(AUTH_C)}"
    )
    successes = [
        r for r in records if r.get("subject") == REAL_USER and r.get("decision") == "allowed"
    ]
    assert successes, f"no allowed login recorded for {REAL_USER}: {records}"
    for record in successes:
        # The SURFACE is asserted per record rather than by pinning the
        # directory's file list: "logins are filed as auth_sidecar" is the
        # claim, and a second sidecar surface appearing one day is not a
        # regression this test should invent.
        assert record["_ledger"] == "auth_sidecar.jsonl", (
            f"a login was filed under an unexpected surface: {record}"
        )
        assert record.get("actor") == SIDECAR_IDENTITY, (
            f"a login record named {record.get('actor')!r} as its writer: {record}"
        )
        for key in ENVELOPE_KEYS:
            assert key in record, f"login record is missing {key!r}: {record}"

    refusals = [
        r
        for r in records
        if r.get("subject") == UNSAFE_ROLE_USER and r.get("decision") == "refused"
    ]
    assert refusals, f"the refused login for {UNSAFE_ROLE_USER} was not recorded: {records}"


def test_the_protected_key_refusal_lands_in_that_users_own_ledger(
    deployment: dict[str, Any],
) -> None:
    """SC2, second half: an in-container refusal is host-visible under alice's subdir.

    The write is a genuine one an operator could attempt — ``PATCH /api/config``
    naming a key in the protected set — refused inside alice's container by the
    real gate. Three mechanisms have to hold for the record to be readable here,
    and removing any one of them fails this test rather than degrading quietly:
    the per-user ``var/audit/alice`` bind (without it the record accumulates in
    the container's writable layer), ``OSPREY_AUDIT_IDENTITY``/the writer's
    identity resolution (without it the record files under a different name),
    and the emitter itself.

    The benign PATCH in ``test_a_password_login_reaches_that_users_own_terminal``
    is the control: the panel IS reachable for alice, so this 403 is the
    protected set answering and not the panel gate.

    The record is matched on its identifying fields, not on ``decision ==
    "refused"`` alone: a bare refusal filter would pass on a record from any
    other surface in that container — a hook, the MCP middleware, an emitter a
    later task adds — and hide the loss of this path entirely. The three that
    identify it are the ``surface`` (imported from the config route's own
    constant), the refused KEY as the ``subject``, and the ``protected_key``
    reason.
    """
    status, _, body = _request(
        f"/u/{REAL_USER}/api/config",
        method="PATCH",
        data=_PROTECTED_PATCH_BODY,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Origin": f"http://127.0.0.1:{HOST_PORTS['nginx']}",
        },
        opener=deployment["sessions"][REAL_USER],
    )
    assert status == 403, (
        f"a protected-key write was not refused (got {status})\n"
        f"{body[:400]}\n{_logs(_web_container(REAL_USER))}"
    )

    # The container writes through its own append; give the bind a moment to
    # show it rather than racing the response.
    deadline = time.monotonic() + 30.0
    refusals: list[dict[str, Any]] = []
    seen: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        seen = _ledger_records(deployment["repo"], REAL_USER)
        refusals = [
            r
            for r in seen
            if r.get("decision") == "refused"
            and r.get("surface") == SURFACE_HTTP_CONFIG
            and r.get("subject") == _REFUSAL_SUBJECT
            and r.get("reason") == _REFUSAL_REASON
        ]
        if refusals:
            break
        time.sleep(1.0)

    assert refusals, (
        f"no {SURFACE_HTTP_CONFIG}/{_REFUSAL_REASON} refusal naming {_REFUSAL_SUBJECT!r} under "
        f"var/audit/{REAL_USER}/ ({_subdir_listing(deployment['repo'], REAL_USER)}; "
        f"records read: {seen})\n"
        f"{_logs(_web_container(REAL_USER))}"
    )
    for record in refusals:
        assert record.get("actor") == REAL_USER, (
            f"a record in {REAL_USER}'s ledger names {record.get('actor')!r} as its actor: {record}"
        )
        for key in ENVELOPE_KEYS:
            assert key in record, f"refusal record is missing {key!r}: {record}"


def test_one_users_records_never_appear_in_anothers_ledger(deployment: dict[str, Any]) -> None:
    """SC2's isolation half: a terminal's own view of the zone holds only ITS subdir.

    Isolation here is by MOUNT, not by permission. The render binds
    ``var/audit/<user>`` — one directory, never the audit ROOT — into each
    user's container, so from inside alice's container the zone contains her
    subdirectory and no other roster identity's. Widening that bind to
    ``var/audit`` (the mistake the compose template's own comment warns about)
    hands every user the whole trail, and this is the assert that trips.

    It has to be read from INSIDE. On the host every subdirectory is visible to
    the account that ran the deploy by construction, so no host-side read can
    tell a per-user bind from a root one.

    The actor == owning-directory sweep below is kept, but as a SECOND and
    different guard: it pins the WRITER's routing rule — the directory component
    is re-resolved from the process identity and never taken from the envelope's
    ``actor`` field. It cannot stand in for the mount check, because a container
    handed the whole root would still write under its own identity and still
    satisfy it.
    """
    container = _web_container(REAL_USER)
    listing = _exec(container, "ls", "-1", CONTAINER_AUDIT_DIR)
    assert listing.returncode == 0, _fmt(f"ls {CONTAINER_AUDIT_DIR} in {container}", listing)
    entries = {name.strip() for name in listing.stdout.splitlines() if name.strip()}
    assert REAL_USER in entries, (
        f"{REAL_USER}'s own audit subdirectory is missing from her container's view of "
        f"{CONTAINER_AUDIT_DIR}: {sorted(entries)}"
    )
    foreign = (entries & (set(USERS) | {SIDECAR_IDENTITY})) - {REAL_USER}
    assert not foreign, (
        f"{REAL_USER}'s container can see {sorted(foreign)} under {CONTAINER_AUDIT_DIR}: the "
        f"render bound the audit ROOT instead of var/audit/{REAL_USER} "
        f"(full listing: {sorted(entries)})"
    )

    for record in _all_records(deployment["repo"]):
        owner = record["_identity_dir"]
        if owner == SIDECAR_IDENTITY:
            continue
        assert record.get("actor") == owner, (
            f"var/audit/{owner}/ holds a record whose actor is {record.get('actor')!r}: {record}"
        )


def test_no_record_carries_a_value_only_an_identifier(deployment: dict[str, Any]) -> None:
    """SC2: envelopes hold identifiers and keys, never the content of a write.

    The refused PATCH carried a distinctive value and a password was checked on
    every login in this run; neither may appear anywhere in the trail. This is
    the property that makes the audit zone safe to hand to an operator who is
    not cleared for the deployment's secrets.

    Reads the trail the tests above left, so it belongs last in the file —
    pytest's definition order is what guarantees the value was actually
    submitted before this looks for it.

    Sweeping the WHOLE zone takes two reads, not one. The host sees every
    per-user subdirectory the render bound out, and nothing else: a ledger
    written at the audit ROOT lands in the container's writable layer, outside
    every per-user bind, is invisible from the host and dies with the container.
    A host-only sweep is therefore structurally blind to exactly the file a
    value would be least likely to be noticed in. The second read closes that:
    an in-container ``cat`` of every ``*.jsonl`` under the zone, whatever
    ledgers happen to exist there.

    Today every refusal this lane drives files under an identity subdirectory,
    so the two reads see the same lines — the in-container half is deliberately
    written against the zone rather than against a named file so that a
    root-level ledger reappearing does not silently drop out of "the whole
    trail".

    Both halves are guarded non-empty: ``json.dumps([])`` contains no value and
    no password, so a trail that was never written would pass this vacuously,
    and so would a sweep that read no lines.
    """
    records = _all_records(deployment["repo"])
    assert records, (
        "the host-side audit trail is empty, so this sweep would pass vacuously "
        f"({_subdir_listing(deployment['repo'], REAL_USER)})"
    )

    container = _web_container(REAL_USER)
    swept = _exec(
        container,
        "find",
        CONTAINER_AUDIT_DIR,
        "-name",
        "*.jsonl",
        "-exec",
        "cat",
        "{}",
        "+",
        timeout=60,
    )
    assert swept.returncode == 0, _fmt(f"sweeping {CONTAINER_AUDIT_DIR} in {container}", swept)
    assert swept.stdout.strip(), (
        f"the in-container sweep of {CONTAINER_AUDIT_DIR} read no ledger lines, so its half "
        f"of this assertion would pass vacuously\n{_logs(container)}"
    )

    haystack = json.dumps(records) + "\n" + swept.stdout
    assert _PROTECTED_VALUE not in haystack, "a refused write's VALUE was recorded"
    for password in _PASSWORDS.values():
        assert password not in haystack, "a password reached the audit trail"


# ---------------------------------------------------------------------------
# The shared default-block guard — no deploy, no Docker
# ---------------------------------------------------------------------------
#
# This lane pins its OWN base by asserting `deployment.port_base == PORT_BASE`
# in `_make_repo` (stricter than the shared guard, which only refuses the
# default block). The shared guard covers the lanes that render through
# `tests/e2e/_orm_stack.build_project_subprocess` instead, and its unit
# coverage lives here because this is the module whose render-time check it is
# modelled on — the two failure messages are deliberately written to read the
# same way, and keeping them in one file is what stops one from drifting.
#
# Nothing below builds, renders or starts anything: the guard reads one file,
# so a synthetic `build/config.yml` in `tmp_path` is the whole fixture.


def _synthetic_render(tmp_path: Path, port_base: int | None) -> Path:
    """A repo whose ``build/config.yml`` names ``port_base`` — the one file the
    shared guard reads back.

    ``None`` writes a ``deployment:`` block that sets no base at all, which is
    exactly how a dropped overlay or a forgotten ``port_base=`` argument looks
    on disk: the key is simply absent and the deployment resolves the
    framework default.
    """
    repo = tmp_path / "synthetic-render"
    build = repo / "build"
    build.mkdir(parents=True)
    deployment: dict[str, Any] = {} if port_base is None else {"port_base": port_base}
    build.joinpath("config.yml").write_text(
        yaml.safe_dump({"deployment": deployment}), encoding="utf-8"
    )
    return repo


@pytest.mark.parametrize(
    "port_base",
    [
        pytest.param(DEFAULT_PORT_BASE, id="explicit-default"),
        pytest.param(None, id="unset"),
    ],
)
def test_the_shared_guard_refuses_a_render_on_the_default_block(
    tmp_path: Path, port_base: int | None
) -> None:
    """A stack rendered on the default block is refused before anything binds.

    Both spellings are the same fault and must both fail: a lane that never
    passed a base, and a lane that passed the default one. Whichever it is, the
    render would bind 10000-10999 — the block a real deployment on the host
    already claims — and the first symptom without this guard is a connection
    error deep inside a container build.
    """
    from tests.e2e import _orm_stack

    repo = _synthetic_render(tmp_path, port_base)

    with pytest.raises(AssertionError) as excinfo:
        _orm_stack.assert_off_default_block(repo, "some-e2e-lane")

    message = str(excinfo.value)
    # The three things the message must carry to be actionable: WHICH stack,
    # WHO built it, and the one-line fix.
    assert "some-e2e-lane" in message, message
    assert "test_full_chain_auth" in message, message
    assert f"{PORT_BASE_CONFIG_KEY}=<band>" in message, message
    assert str(DEFAULT_PORT_BASE) in message, message


def test_the_shared_guard_passes_a_render_in_its_own_band(tmp_path: Path) -> None:
    """A render that moved off the default block is accepted.

    The negative half: without it the guard could refuse everything and the
    test above would still pass.
    """
    from tests.e2e import _orm_stack

    _orm_stack.assert_off_default_block(_synthetic_render(tmp_path, PORT_BASE), "some-e2e-lane")


def test_the_shared_guard_refuses_a_render_that_never_happened(tmp_path: Path) -> None:
    """No ``build/config.yml`` at all fails as a missing render, not as a
    crash — a build that silently produced nothing is its own fault, and
    reading the base off a file that is not there must not surface as a
    ``FileNotFoundError`` from inside a helper."""
    from tests.e2e import _orm_stack

    repo = tmp_path / "never-rendered"
    repo.mkdir()

    with pytest.raises(AssertionError, match="rendered no config"):
        _orm_stack.assert_off_default_block(repo, "some-e2e-lane")
