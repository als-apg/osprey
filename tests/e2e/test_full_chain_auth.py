"""Acceptance proof for the WHOLE identity chain on real artifacts: a password
login at the deployed sidecar, an ``authorization:`` role that decides which
render a user's container runs, the four identity headers nginx forwards
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
authorized, does the ROLE it resolved decide which render that container runs,
and does every decision on the way land in the ledger of the identity that made
it — and only there. That question needs both halves real at once, which is why
this is the one job in the repo that builds the deployed sidecar Dockerfile AND
a real framework render image in the same deployment.

THE DEPLOYMENT THIS BUILDS, AND WHY EACH ROSTER USER IS THERE
-------------------------------------------------------------
One ``osprey up --dev`` of a ``hello-world`` render with ``auth.method:
password``, an ``authorization:`` stanza binding two roles to two renders, and
a six-entry roster. Every entry earns its place by being the only shape that
can prove one thing:

=========  =============  ==========  =========================================
user       role           render     what only this entry can prove
=========  =============  ==========  =========================================
``alice``  ``operator``   REAL app    A password login reaches HER container,
                                      and an in-container protected-key refusal
                                      is recorded under ``alice``.
``bob``    ``observer``   probe stub  Account, subject, role AND role source
                                      arrive at the upstream, exactly once
                                      each, carrying the sidecar's values —
                                      and client-forged headers do not.
``carol``  (none)         probe stub  A session with no role forwards the
                                      account and subject but NEITHER a role
                                      nor a role-source header at all —
                                      absent, never present-and-blank.
``dave``   (none)         probe stub  A role the boundary cannot carry refuses
                                      the login 403 instead of poisoning a
                                      header (see FAULT INJECTION below).
``kiosk``  (none)         probe stub  A shared card (``access: [self,
                                      roster]``): its sessions are absent from
                                      an opener's own history view, and a
                                      sign-in as the card lists nothing.
``erin``   (none)         probe stub  ``history: all``: the history page shows
                                      every person's sessions.
=========  =============  ==========  =========================================

``carol`` is also a reader (``agent_record.readers``), so the history page shows her the reader's scope of
the agent record until a second ``osprey up`` withdraws her from the readers
and maps her back to her own view.

WHAT THE ACCOUNT HEADER ASSERTIONS HERE DO AND DO NOT PROVE
-----------------------------------------------------------
This lane is password-only, and in a password deployment the account IS the
subject: both headers carry the roster username, so every assertion below
compares the two against the same value. That makes these assertions a proof
of nginx TRANSPORT — that the sidecar's ``X-Osprey-Auth-Account`` is captured
from the ``/verify`` answer, forwarded exactly once, replaces a client-forged
copy rather than joining it, and is cleared on the ungated arm — and NOT a
proof of OIDC semantics, where an account and a subject actually diverge. That
half is ``tests/deployment/web_terminals/test_auth_serving_oidc.py``: one OIDC
login through the rendered nginx and the real sidecar, against a stub provider
in the serving harness's network namespace.

**The render each user runs is decided by the ROLE, never by a ``tier:``
pin.** That is not incidental to the fixture — it is the FR6 mechanism under
test. ``authorization.roles`` maps ``operator`` → the real framework render and
``observer`` → the probe stub, ``default_tier`` catches the role-less
entries, and no roster entry here carries a ``tier:`` key at all (an entry
carrying both is refused by ``effective_tier``, deliberately). So
``test_alices_container_runs_the_render_her_role_named`` fails the moment role
resolution stops reaching the render.

ONE REAL FRAMEWORK RENDER, FOUR CHEAP PROBES — AND WHY THE SPLIT IS HONEST
--------------------------------------------------------------------------
The four non-alice entries run an alpine + python3 stub that answers every
request with the request's OWN headers as JSON. That is not a cost dodge that
weakens the proof; it is the only way to make the header assertions exact. The
real ``osprey web`` does not echo what it received, so a lane with only real
renders could assert that a request SUCCEEDED but never that the upstream saw
exactly one ``X-Osprey-Auth-Subject`` with the sidecar's value and no forged
one.

What actually drops a forged value is the FORWARD. A gated ``/u/<user>/``
declares ``proxy_set_header X-Osprey-Auth-Subject $osprey_auth_subject;``, and
naming a header in a location's proxy-header table replaces whatever the client
sent under that name. The template forbids a second directive for the same name
in the same location, so the COUNT below is a duplicate-forward regression
guard, not a claim that nginx skipped an empty clear. The unconditional
``proxy_set_header ... "";`` clears live in the UNGATED locations instead —
each ``/_osprey_auth/<user>`` subrequest — where there is no sidecar answer to
forward and the clear is the only thing standing between a client's header and
the upstream.

Alice's render is the real thing for the complementary reason: the ledger
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
this test created — the ``<project>-nginx`` / ``<project>-auth`` / ``<project>-web-
<user>`` / ``<project>-history`` containers, this project's volumes, and the ``:local`` image tags — or
is the project-scoped ``compose down`` the deploy lifecycle itself uses. Nothing
here ever runs a prune, an ``-a``/``--all`` sweep, or a wildcard removal.
"""

from __future__ import annotations

import copy
import http.cookiejar
import json
import os
import re
import shutil
import subprocess
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
from osprey.deployment.compose_generator import resolve_project_name
from osprey.deployment.web_terminals.provision import force_recreate_auth_sidecar
from osprey.deployment.web_terminals.render_naming import render_project
from osprey.port_layout import DEFAULT_PORT_BASE, PORT_BASE_CONFIG_KEY, default_port
from osprey.services.auth_sidecar.identity_headers import (
    ACCOUNT_HEADER,
    ROLE_HEADER,
    ROLE_SOURCE_HEADER,
    SUBJECT_HEADER,
)
from osprey.services.auth_sidecar.revocation import REVOCATION_FILE_NAME
from osprey.services.auth_sidecar.sessions import SESSION_COOKIE_NAME
from osprey.utils.workspace import TIERS_DIRNAME
from osprey_connectors.config import ConfigBuilder
from tests.e2e._deploy_support import (
    RUNTIME,
    container_id,
    find_osprey_console_script,
    format_failure,
    image_exists,
    run_osprey,
    runtime_cli,
    runtime_logs,
)
from tests.e2e._render_stub import stub_render_image
from tests.e2e._volumes import remove_project_volumes
from tests.e2e.profile_edits import set_pairs

# See "CI HONESTY" in the module docstring: this marker is what keeps the build
# out of the shared fast e2e glob and what the wiring guard scans for.
#: The port the project image serves on INSIDE the container: the ``web`` slot
#: of the layout, which ``Dockerfile.j2`` renders into its ``EXPOSE`` line. The
#: image is built from a config that never moves ``deployment.port_base``, so
#: the layout's default base is the right one to derive it at.
#:
#: Still true after the DEPLOYMENT moved to :data:`PORT_BASE` below, because
#: nothing binds this number: the per-user compose service sets
#: ``OSPREY_TERMINAL_WEB_PORT``/``OSPREY_WEB_PORT`` explicitly to that user's
#: port off the DEPLOYMENT's base (``docker-compose.web.yml.j2``, which is also
#: what the healthcheck probes). This is the stub's fallback for the case that
#: never happens here, not the port it serves on.
_WEB_SLOT = default_port("web")

pytestmark = [pytest.mark.e2e, pytest.mark.dockerbuild]

# The repo DIRECTORY name, which is the deployment name: the compose project and
# the com.osprey.project label on every image this deploy builds.
PROJECT_NAME = "osprey-e2e-full-chain-auth"
#: The compose project name the web tier's containers and the sidecar image
#: are named by.
PROJECT = resolve_project_name({"project_name": PROJECT_NAME})
#: The preset the repo is materialized from — the smallest one that stands up a
#: serving ``osprey web``.
PRESET = "hello-world"
#: The tier ``osprey init`` is handed for every tier this lane names: init
#: writes the tiers OSPREY ships, and this lane's two are its own files, which
#: replace it once the repo exists (:func:`_author_tiers`).
BUNDLED_STAND_IN = "readonly"

#: This deployment's own two tiers, each a file in ``tiers/``. ``operator`` runs
#: the REAL framework image; ``probe`` is the header-echoing stub. A tier's image
#: is tagged by its render project, ``<project_name>-<tier>``.
TERMINAL_RENDER = "operator"
TERMINAL_PROJECT, _ = render_project(PROJECT_NAME, TERMINAL_RENDER)
PROBE_RENDER = "probe"
PROBE_PROJECT, _ = render_project(PROJECT_NAME, PROBE_RENDER)

#: Role names, as ``authorization.roles`` declares them. ``operator`` binds the
#: real render, ``observer`` the probe — so which image a container runs is
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
UNSAFE_ROLE_USER = "dave"
#: A shared card: its ``access`` admits every roster person as an opener, and
#: its sessions are the card's, not the opener's.
SHARED_CARD = "kiosk"
#: The one person whose roster entry carries ``history: all``.
ALL_USER = "erin"
#: Declaration order is roster order, and roster order is index order.
USERS = (REAL_USER, HEADER_USER, NO_ROLE_USER, UNSAFE_ROLE_USER, SHARED_CARD, ALL_USER)
#: Every entry that authenticates — the whole roster, behind a password wall.
LOGIN_USERS = USERS

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

AUTH_IMAGE_TAG = f"{PROJECT}-auth:local"
# `<render project>:local`, exactly as resolve_cards derives a local-mode
# tier's tag.
TERMINAL_IMAGE_TAG = f"{TERMINAL_PROJECT}:local"
PROBE_IMAGE_TAG = f"{PROBE_PROJECT}:local"

NGINX_C = f"{PROJECT}-nginx"
AUTH_C = f"{PROJECT}-auth"
#: The project image ``osprey up`` builds from the project root, which the
#: history service runs (the render names it; a history case pins that).
PROJECT_IMAGE_TAG = f"{PROJECT}:local"

HISTORY_C = f"{PROJECT}-history"

# Two real framework installs; give them room. The step budget in CI is 55
# minutes (see the module docstring); this is the in-test ceiling below it.
DEPLOY_UP_TIMEOUT_SEC = 2400
# `init` + `build` render the whole repo (no dependency install, no lifecycle
# hooks — see the flags below), which is filesystem work rather than network.
RENDER_TIMEOUT_SEC = 600
# The sidecar answers /health as soon as its app imports; the real render's
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

_ENV_CONTENT = "ANTHROPIC_API_KEY=fake-llm-key-value\n" + "".join(
    f"OSPREY_AUTH_PW_{user.upper()}={password}\n" for user, password in _PASSWORDS.items()
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

#: The state zone, host side, relative to the deployment repo: where the
#: sidecar keeps its revoked-session file.
STATE_RELPATH = Path("var") / "state"
#: The sidecar's fixed audit identity — one service, so its records name their
#: SUBJECTS rather than being filed under a username.
SIDECAR_IDENTITY = "sidecar"

#: The spool root as a TERMINAL CONTAINER sees it. The render binds each
#: container's own ``var/spool/<identity>`` and nothing else, so the container's
#: view of the root is the only place the MOUNT itself is observable: no
#: host-side read can tell a per-user bind from a bind of the whole root.
CONTAINER_SPOOL_ROOT = f"/app/{TERMINAL_PROJECT}/var/spool"

#: Seconds a test waits for an audit entry to reach the record: the writer's
#: segment seals, the spool consumer ingests it, and a read answers it.
AUDIT_SETTLE_S = 60.0


def _web_container(user: str) -> str:
    return f"{PROJECT}-web-{user}"


def _container_image(name: str) -> str | None:
    result = runtime_cli(
        "inspect", "--type", "container", "-f", "{{.Config.Image}}", name, timeout=15
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _exec(container: str, *argv: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run one read-only command INSIDE an exact container this test created.

    Names a ``<project>-web-<user>`` container built by this deployment, never a
    pattern — see CONTAINER-OPS SAFETY in the module docstring.
    """
    return runtime_cli("exec", container, *argv, timeout=timeout)


# ---------------------------------------------------------------------------
# The two tiers
# ---------------------------------------------------------------------------


#: The probe upstream: an HTTP server whose whole job is to report the request
#: it received. ``get_all`` is the load-bearing call — it returns EVERY
#: occurrence of a header name, which is what lets the assertions below count
#: rather than merely look up, and a duplicated forward is a real failure mode
#: of a location that emits both a clear and a forward for one name.
#: An agent-authored HTML artifact, as the probe serves it under ``/u/<user>/``:
#: the history cases open it to prove that a page on the terminals' origin can
#: neither read a ``/history/`` answer nor make the page run a query. The script
#: is its own file so no policy on inline script decides the outcome.
_ARTIFACT_PAGE = (
    "<!doctype html><html><head><meta charset=utf-8><title>artifact</title></head>"
    '<body><script src="artifact.js"></script></body></html>'
)
#: ``?mode=read`` fetches the target with the browser's cookies and records
#: whether the answer could be read; ``?mode=embed`` points an iframe and an
#: image at it.
_ARTIFACT_SCRIPT = """\
(function () {
  var query = new URLSearchParams(location.search);
  var target = query.get("target");
  window.__osprey_artifact = "pending";
  if (query.get("mode") === "read") {
    fetch(target, {credentials: "include"})
      .then(function (r) { return r.text(); })
      .then(function (b) { window.__osprey_artifact = "read:" + b.length; })
      .catch(function (e) { window.__osprey_artifact = "blocked:" + e.name; });
    return;
  }
  var frame = document.createElement("iframe");
  frame.src = target;
  document.body.appendChild(frame);
  var image = document.createElement("img");
  image.src = target;
  document.body.appendChild(image);
  window.__osprey_artifact = "embedded";
})();
"""

_PROBE_SOURCE = f'''\
"""Header-reporting upstream for the full-chain auth e2e lane."""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

MARKER = {PROBE_MARKER!r}
USER = os.environ.get("OSPREY_TERMINAL_USER", "")
HOST = os.environ.get("OSPREY_TERMINAL_BIND_HOST", "127.0.0.1")
PORT = int(os.environ.get("OSPREY_TERMINAL_WEB_PORT", "{_WEB_SLOT}"))
ARTIFACT_PAGE = {_ARTIFACT_PAGE!r}
ARTIFACT_SCRIPT = {_ARTIFACT_SCRIPT!r}


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
        path = self.path.split("?", 1)[0]
        if path.endswith("/artifact.html"):
            self._send(200, ARTIFACT_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path.endswith("/artifact.js"):
            self._send(200, ARTIFACT_SCRIPT.encode("utf-8"), "text/javascript")
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


#: The setup tool the deny entry has to name for a render to read as unable to
#: edit its own deployment. Spelled here rather than imported so the stub's
#: config is a literal an operator could have written; the import-side spelling
#: is ``profile_conventions.SETUP_PATCH_TOOL``.
_SETUP_PATCH_TOOL = "mcp__osprey_workspace__setup_patch"


def _tier_files() -> dict[str, str]:
    """This lane's two tier files, by name, as they are written into ``tiers/``.

    ``operator`` states no differences: it renders the deployment's own posture,
    and its image is the production project image, so alice's terminal is the
    real app. ``probe`` states the three deployment-editing floors — the Config
    panel off, the scaffold gallery read-only and the setup tool denied. It is the default tier and the tier
    of the shared card, and the build refuses either unless its render holds
    neither surface; the guard is satisfied by the render being harmless, never
    by the check being skipped.
    """
    return {
        TERMINAL_RENDER: f"name: {PROJECT_NAME} ({TERMINAL_RENDER})\n",
        PROBE_RENDER: yaml.safe_dump(
            {
                "name": f"{PROJECT_NAME} ({PROBE_RENDER})",
                "config": {
                    "web.config_panel.enabled": False,
                    "web.scaffold_gallery.write_enabled": False,
                    "claude_code.permissions.deny": [_SETUP_PATCH_TOOL],
                },
            },
            sort_keys=False,
        ),
    }


def _probe_dockerfile() -> str:
    """The probe tier's image recipe: a stub that REPORTS its request.

    Beyond what the idling lifecycle stub provides (a ``dispatch`` user and the
    mount-point directories the per-user compose service declares volumes onto),
    this one runs a small ``http.server`` that answers every request with the
    request's own headers as JSON. Binding ``127.0.0.1`` mirrors the production
    app: nginx shares the host network namespace and is the only thing meant to
    reach a per-user port.

    It replaces the recipe ``osprey build`` rendered for the tier
    (:func:`tests.e2e._render_stub.stub_render_image`), with ``probe.py`` beside
    it in the image context. The tier's rendered ``config.yml`` stays, so a
    start checks it exactly as it would a real one.

    ``curl`` is installed for the per-user service's healthcheck (``curl -fsS
    http://<bind>:<port>/health``); without it the container runs but reports
    unhealthy, which reads in a failure log like a broken deployment.
    """
    container_dir = f"/app/{PROBE_PROJECT}"
    return (
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


# ---------------------------------------------------------------------------
# The deployment
# ---------------------------------------------------------------------------


def _roster() -> list[dict[str, Any]]:
    """The five roster entries, as ``modules.web_terminals.users``.

    Dict entries carry an explicit ``index``: ``normalize_users`` accepts a bare
    string (index = position) or a dict, and a dict without an ``index`` int is
    dropped entirely. Declaration order and index agree here, which is what the
    per-user port assignment reads.

    No entry carries a ``tier:`` key. ``effective_tier`` refuses an entry
    naming both a render and a role — a role binds the render, and a pin is
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
        elif user == SHARED_CARD:
            # `roster` opens the card to whoever names themselves as its
            # opener; `self` keeps the card's own credential working too.
            entry["access"] = ["self", "roster"]
        elif user == ALL_USER:
            entry["history"] = "all"
        entries.append(entry)
    return entries


def _record_readers() -> list[dict[str, Any]]:
    """The agent record's readers: ``carol``, so her reader viewer reads prompts.

    The sidecar names ``reader:carol`` for her sign-in while she is one of
    these readers; the record answers that viewer with her kinds.
    """
    return [{"name": NO_ROLE_USER, "kinds": ["prompt", "answer"]}]


def _profile_edits() -> dict[str, Any]:
    """The edits carrying this lane's whole web-terminal stanza.

    Dotted leaf keys under ``config:``, the one spelling a profile's config
    block accepts — and ``modules.web_terminals`` deliberately as ONE dotted key
    with a nested value, so it states that subtree's leaves without replacing
    the rendered ``modules:`` mapping around it.

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

    Sign-on brings the history page and with it the history page's origin.
    """
    return {
        "config": {
            "container_runtime": RUNTIME,
            "system.timezone": "UTC",
            "deploy.fqdn": "127.0.0.1",
            PORT_BASE_CONFIG_KEY: PORT_BASE,
            "modules.web_terminals": {
                "enabled": True,
                "image_source": "local",
                # The role-less entries land here, which is why the cheap
                # render is the default and the expensive one is reached
                # only through a role.
                "default_tier": PROBE_RENDER,
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
                        ROLE_OPERATOR: {"tier": TERMINAL_RENDER},
                        ROLE_OBSERVER: {"tier": PROBE_RENDER},
                    }
                },
            },
        },
        "agent_record": {"readers": _record_readers()},
    }


#: The profile key every web-terminal edit of this lane sits under.
_WEB_TERMINALS = "modules.web_terminals"


def _bundled_stand_in(edits: dict[str, Any]) -> dict[str, Any]:
    """*edits* with every tier they name replaced by :data:`BUNDLED_STAND_IN`.

    ``osprey init`` writes a file only for a tier OSPREY ships and refuses a
    roster naming any other: a facility's own tier exists as a file in the
    repo's ``tiers/``, which the repo must exist to hold. So the repo is made
    with a bundled tier wherever the lane names one, and
    :func:`_author_tiers` puts the lane's own tiers in their place.

    The shared card's ``access`` waits for the lane's roster too. Under this
    preset the stand-in renders the deployment's own editing surfaces, which
    init rightly refuses to share; the probe tier the card really runs holds
    none of them, and the build judges the card again against it.
    """
    stand_in = copy.deepcopy(edits)
    web_terminals = stand_in["config"]["modules.web_terminals"]
    web_terminals["default_tier"] = BUNDLED_STAND_IN
    roles = (web_terminals.get("authorization") or {}).get("roles") or {}
    for binding in roles.values():
        binding["tier"] = BUNDLED_STAND_IN
    for entry in web_terminals["users"]:
        entry.pop("access", None)
    return stand_in


def _author_tiers(repo: Path, edits: dict[str, Any]) -> None:
    """Write this lane's own tiers into ``tiers/`` and point the roster at them.

    The way a facility adds a tier: a file in ``tiers/`` (:func:`_tier_files`),
    and the profile's ``default_tier``, roster and role bindings naming it. The stand-in
    tier ``osprey init`` wrote is removed, so the repo holds exactly the tiers
    it deploys.

    A YAML round-trip rather than a text splice: the emitted profile is plain
    YAML, and the comments a dump drops are documentation for an operator, not
    input to the build.
    """
    tiers = repo / TIERS_DIRNAME
    (tiers / f"{BUNDLED_STAND_IN}.yml").unlink()
    for name, text in _tier_files().items():
        (tiers / f"{name}.yml").write_text(text, encoding="utf-8")

    wanted = edits["config"]["modules.web_terminals"]
    profile_path = repo / "profile.yml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    # The profile spells its config as dotted leaves, so a role binding is the
    # leaf ``...authorization.roles.<role>.tier``.
    config = profile["config"]
    config[f"{_WEB_TERMINALS}.default_tier"] = wanted["default_tier"]
    config[f"{_WEB_TERMINALS}.users"] = copy.deepcopy(wanted["users"])
    roles = (wanted.get("authorization") or {}).get("roles") or {}
    for role, binding in roles.items():
        config[f"{_WEB_TERMINALS}.authorization.roles.{role}.tier"] = binding["tier"]
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")


def _shared_card_credential() -> str:
    """The ``.env.auth`` line that lets the shared card sign in as itself.

    A password deployment mints no credential for a shared card — its own
    sign-in is an identity-provider mapping, and no lane here runs a provider —
    so the one history case about a sign-in AS a shared card writes the card's
    hash by hand, the way an operator maintains this file (and the way
    ``dave``'s role is injected). Provisioning keeps an entry it finds.
    """
    from osprey.services.auth_sidecar.passwords import hash_password

    return f"OSPREY_AUTH_PW_HASH_{SHARED_CARD.upper()}={hash_password(_PASSWORDS[SHARED_CARD])}\n"


def _make_repo(tmp_path: Path, osprey_bin: Path, edits: dict[str, Any] | None = None) -> Path:
    """``init`` + this lane's tiers + ``build`` + the probe recipe — the repo this lane deploys.

    *edits* replaces :func:`_profile_edits` for a render this lane only reads.

    ``--skip-deps``/``--skip-lifecycle`` keep the render off the network and
    quick: nothing here runs the deployment's own venv, only its containers.
    """
    edits = edits or _profile_edits()
    repo = tmp_path / PROJECT_NAME

    init = run_osprey(
        osprey_bin,
        [
            "init",
            str(repo),
            "--preset",
            PRESET,
            "--no-git",
            *set_pairs(_bundled_stand_in(edits)),
        ],
        tmp_path,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert init.returncode == 0, format_failure("osprey init (full-chain auth)", init)

    _author_tiers(repo, edits)

    build = run_osprey(
        osprey_bin,
        ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle", "--dev"],
        tmp_path,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert build.returncode == 0, format_failure("osprey build (full-chain auth)", build)
    stub_render_image(repo, PROBE_PROJECT, _probe_dockerfile(), {"probe.py": _PROBE_SOURCE})

    # The one knob, read back off the render before anything binds. Every host
    # port this lane reaches is `default_port(slot, base=PORT_BASE)`, and that
    # derivation is only true if the render actually resolved the same base: an
    # key that failed to land leaves the deploy on the default 10000
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
        f"Fix in {__name__}: `_profile_edits` sets `{PORT_BASE_CONFIG_KEY}=<band>` in this "
        f"lane's config edits and PORT_BASE is the band it books — that key "
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
    env_auth_path.write_text(_ENV_AUTH_SEED + _shared_card_credential(), encoding="utf-8")
    os.chmod(env_auth_path, 0o600)
    return repo


def _teardown() -> None:
    """Exact-named sweep; failures swallowed (a safety net, never an assertion).

    The volume sweep is what keeps reruns honest: ``compose down`` (like
    ``osprey down``) keeps named volumes, and a rerun inheriting a previous
    attempt's per-user volumes would start from that attempt's state.
    """
    for user in USERS:
        runtime_cli("rm", "-f", _web_container(user))
    runtime_cli("rm", "-f", AUTH_C)
    runtime_cli("rm", "-f", NGINX_C)
    runtime_cli("rm", "-f", HISTORY_C)
    runtime_cli("compose", "-p", PROJECT_NAME, "down", timeout=90)
    remove_project_volumes(PROJECT_NAME, runtime=RUNTIME)
    for tag in (AUTH_IMAGE_TAG, TERMINAL_IMAGE_TAG, PROBE_IMAGE_TAG, PROJECT_IMAGE_TAG):
        runtime_cli("rmi", "-f", tag, timeout=60)


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

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # see class
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
    req = urllib.request.Request(  # loopback only
        f"http://127.0.0.1:{port}{target}", method=method, data=data
    )
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    open_it = opener.open if opener is not None else urllib.request.urlopen
    try:
        with open_it(req, timeout=30) as resp:  # loopback only
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
        f"login for {user!r} was not accepted (got {status})\n{page[:300]}\n{runtime_logs(AUTH_C)}"
    )
    assert SESSION_COOKIE_NAME in {cookie.name for cookie in jar}, (
        f"login for {user!r} issued no session cookie "
        f"(jar: {sorted(cookie.name for cookie in jar)})\n{runtime_logs(AUTH_C)}"
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
        f"{body[:400]}\n{runtime_logs(NGINX_C)}\n{runtime_logs(_web_container(user))}"
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
        f"auth sidecar /health not ready after {timeout:.0f}s (last: {last})\n{runtime_logs(AUTH_C)}"
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
        f"{runtime_logs(NGINX_C)}\n{runtime_logs(AUTH_C)}"
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
        f"{runtime_logs(NGINX_C)}\n{runtime_logs(_web_container(user))}"
    )


# ---------------------------------------------------------------------------
# The audit trail, read from the agent record
# ---------------------------------------------------------------------------


def _audit_records(repo: Path, identity: str | None = None) -> list[dict[str, Any]]:
    """Every audit entry the record serves the history login, as envelope bodies.

    Each body is tagged ``_identity`` with the identity that wrote it — the
    trusted identity of the spool directory the entry arrived in, never a
    field the writer chose — and filtered to *identity* when one is given.
    Read with the history login's ``all`` viewer, the one view that spans
    every identity.
    """
    answer = _record_client(repo, "all").recent_audit()
    assert answer.available, f"the agent record is unavailable: {answer.unavailable}"
    # The sweeps over this list claim to cover every audit entry, so a page the
    # record cut short (incomplete, or capped with more behind it) is refused
    # rather than swept as if it were whole.
    assert answer.complete, f"the record's audit answer is incomplete: {answer.reason}"
    assert not answer.more, "the record holds more audit entries than one page serves"
    records: list[dict[str, Any]] = []
    for item in answer.data or []:
        if not isinstance(item, dict) or not isinstance(item.get("body"), dict):
            continue
        if identity is not None and item.get("identity") != identity:
            continue
        records.append({**item["body"], "_identity": item.get("identity")})
    return records


def _await_audit(
    repo: Path, identity: str | None, matches: Any, *, timeout: float = AUDIT_SETTLE_S
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Poll the record until an entry of *identity* satisfies *matches*.

    Returns the matching records and everything read on the last poll, so a
    failure message can show what WAS there.
    """
    deadline = time.monotonic() + timeout
    seen: list[dict[str, Any]] = []
    while True:
        seen = _audit_records(repo, identity)
        found = [record for record in seen if matches(record)]
        if found or time.monotonic() >= deadline:
            return found, seen
        time.sleep(1.0)


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
    if runtime_cli("ps", timeout=10).returncode != 0:
        pytest.skip(f"{RUNTIME} daemon not responding")

    tmp_path = tmp_path_factory.mktemp("full-chain-auth")
    osprey_bin = find_osprey_console_script()
    repo = _make_repo(tmp_path, osprey_bin)

    _teardown()  # clear anything a previous crashed run stranded under these names
    try:
        # Run from inside the repo: every lifecycle verb walks up to the nearest
        # profile.yml, so standing in the deployment is what selects it.
        up = run_osprey(osprey_bin, ["up", "--dev"], repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up.returncode == 0, format_failure("osprey up --dev (full-chain auth)", up)

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
        run_osprey(osprey_bin, ["down"], repo)
        _teardown()


# ---------------------------------------------------------------------------
# T1 — the artifacts under test are the real ones
# ---------------------------------------------------------------------------


def test_up_builds_the_real_sidecar_and_both_render_images(deployment: dict[str, Any]) -> None:
    """Both framework builds and the probe image landed, and the sidecar is configured.

    Every assertion below runs against these three images; this one pins that
    they are the artifacts a deployment actually builds, so a green auth result
    cannot be a green stub. ``configured: true`` is the sidecar's own report
    that the ``.env.auth`` this deploy provisioned reached it — its fail-closed
    posture with no credentials is ``configured: false`` plus a refusal on
    every other path.
    """
    assert image_exists(AUTH_IMAGE_TAG), f"{AUTH_IMAGE_TAG} was not built by 'osprey up'"
    assert image_exists(TERMINAL_IMAGE_TAG), f"{TERMINAL_IMAGE_TAG} was not built by 'osprey up'"
    assert image_exists(PROBE_IMAGE_TAG), f"{PROBE_IMAGE_TAG} was not built by 'osprey up'"
    assert container_id(AUTH_C) is not None, f"{AUTH_C} was not created"
    assert deployment["health"].get("configured") is True, (
        f"sidecar came up unconfigured: {deployment['health']}\n{runtime_logs(AUTH_C)}"
    )


@pytest.mark.usefixtures("deployment")
def test_alices_container_runs_the_render_her_role_named() -> None:
    """FR6, end to end: the ROLE decided which image each container runs.

    No roster entry here carries a ``tier:`` key — alice reaches the real
    framework render only because ``authorization.roles`` maps her ``operator``
    role onto it, and the role-less entries fall to ``default_tier``. Break
    role resolution anywhere between the parser and ``resolve_cards`` and
    alice's container comes up on the probe image, which this compares by tag.
    """
    assert _container_image(_web_container(REAL_USER)) == TERMINAL_IMAGE_TAG, (
        f"{REAL_USER}'s role did not resolve to the {TERMINAL_RENDER!r} render"
    )
    for user in (HEADER_USER, NO_ROLE_USER, UNSAFE_ROLE_USER):
        assert _container_image(_web_container(user)) == PROBE_IMAGE_TAG, (
            f"{user} did not resolve to the {PROBE_RENDER!r} render"
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
        f"{body[:400]}\n{runtime_logs(NGINX_C)}\n{runtime_logs(_web_container(REAL_USER))}"
    )


@pytest.mark.usefixtures("deployment")
def test_a_sign_in_without_a_card_reaches_that_users_own_terminal() -> None:
    """The card-less entry through nginx: the form, then the cards it opened.

    This roster shares one card with every roster person (``kiosk``), so a
    verified credential opens the person's own terminal AND the shared one, and
    the answer is the list of both — own card first — with a session that
    already reaches the person's own terminal through nginx. The one-card
    redirect is the sidecar's own unit case.
    """
    status, _, page = _request("/auth/enter", headers=_navigation())
    assert status == 200, f"sign-in page not served (got {status})\n{page[:300]}"
    assert 'name="username"' in page

    client, jar = _browser()
    body = urllib.parse.urlencode(
        {"username": REAL_USER, "password": _PASSWORDS[REAL_USER]}
    ).encode()
    status, _, page = _request(
        "/auth/enter",
        method="POST",
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": f"http://127.0.0.1:{HOST_PORTS['nginx']}",
            **_navigation(),
        },
        opener=client,
    )
    assert status == 200, (
        f"sign-in for {REAL_USER!r} did not list its cards (got {status})\n"
        f"{page[:300]}\n{runtime_logs(AUTH_C)}"
    )
    own, shared = f'href="/u/{REAL_USER}/"', f'href="/u/{SHARED_CARD}/"'
    assert own in page and shared in page, page[:1500]
    assert page.index(own) < page.index(shared), "the person's own card is not listed first"
    assert SESSION_COOKIE_NAME in {cookie.name for cookie in jar}
    status, _, page = _request(f"/u/{REAL_USER}/", headers=_navigation(), opener=client)
    assert status == 200, f"the card-less session does not reach /u/{REAL_USER}/ ({status})"


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
    diverge; that half is ``tests/deployment/web_terminals/test_auth_serving_oidc.py``.
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
    in the ungated ``/_osprey_auth/<user>`` subrequest locations. The forged
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


@pytest.mark.usefixtures("deployment")
def test_a_login_whose_role_cannot_be_carried_is_refused() -> None:
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
        f"{page[:300]}\n{runtime_logs(AUTH_C)}"
    )


@pytest.mark.usefixtures("deployment")
def test_the_refused_login_minted_no_session() -> None:
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

    Recorded under the ``sidecar`` identity on the ``auth_sidecar`` surface, with
    the roster user as the record's SUBJECT and ``sidecar`` as its ACTOR. That split is the point: a login record filed under the
    username would read as that user having written it.

    The refusal is asserted beside the success so the ledger is shown to record
    both outcomes by category, which is what makes it answerable after the fact.
    It is the refusal ``test_a_login_whose_role_cannot_be_carried_is_refused``
    drove: this test reads the trail one stack left, so it has to run after the
    tests that act, and pytest's definition order is what puts it there. Keep it
    below them.
    """
    successes, records = _await_audit(
        deployment["repo"],
        SIDECAR_IDENTITY,
        lambda r: r.get("subject") == REAL_USER and r.get("decision") == "allowed",
    )
    assert records, f"the sidecar recorded no audit entries\n{runtime_logs(AUTH_C)}"
    assert successes, f"no allowed login recorded for {REAL_USER}: {records}"
    for record in successes:
        # The SURFACE is asserted per record rather than by pinning the
        # directory's file list: "logins are filed as auth_sidecar" is the
        # claim, and a second sidecar surface appearing one day is not a
        # regression this test should invent.
        assert record["surface"] == "auth_sidecar", (
            f"a login was filed under an unexpected surface: {record}"
        )
        assert record.get("actor") == SIDECAR_IDENTITY, (
            f"a login record named {record.get('actor')!r} as its writer: {record}"
        )
        for key in ENVELOPE_KEYS:
            assert key in record, f"login record is missing {key!r}: {record}"

    refusals, records = _await_audit(
        deployment["repo"],
        SIDECAR_IDENTITY,
        lambda r: r.get("subject") == UNSAFE_ROLE_USER and r.get("decision") == "refused",
    )
    assert refusals, f"the refused login for {UNSAFE_ROLE_USER} was not recorded: {records}"


def test_the_protected_key_refusal_is_recorded_under_that_user(
    deployment: dict[str, Any],
) -> None:
    """SC2, second half: an in-container refusal is recorded under alice's identity.

    The write is a genuine one an operator could attempt — ``PATCH /api/config``
    naming a key in the protected set — refused inside alice's container by the
    real gate. Three mechanisms have to hold for the record to be readable here,
    and removing any one of them fails this test rather than degrading quietly:
    the per-user ``var/spool/alice`` bind (without it the line accumulates in
    the container's writable layer and never reaches the record), the spool
    directory's identity (without it the entry is recorded under a different
    name), and the emitter itself.

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
        f"{body[:400]}\n{runtime_logs(_web_container(REAL_USER))}"
    )

    # The entry reaches the record through the spool consumer; wait for it
    # rather than racing the response.
    refusals, seen = _await_audit(
        deployment["repo"],
        REAL_USER,
        lambda r: (
            r.get("decision") == "refused"
            and r.get("surface") == SURFACE_HTTP_CONFIG
            and r.get("subject") == _REFUSAL_SUBJECT
            and r.get("reason") == _REFUSAL_REASON
        ),
    )

    assert refusals, (
        f"no {SURFACE_HTTP_CONFIG}/{_REFUSAL_REASON} refusal naming {_REFUSAL_SUBJECT!r} "
        f"recorded under {REAL_USER!r} (records read: {seen})\n"
        f"{runtime_logs(_web_container(REAL_USER))}"
    )
    for record in refusals:
        assert record.get("actor") == REAL_USER, (
            f"an entry recorded under {REAL_USER} names {record.get('actor')!r} as its actor: "
            f"{record}"
        )
        for key in ENVELOPE_KEYS:
            assert key in record, f"refusal record is missing {key!r}: {record}"


def test_one_users_records_never_appear_under_anothers_identity(
    deployment: dict[str, Any],
) -> None:
    """SC2's isolation half: a terminal's own view of the spool holds only ITS directory.

    Isolation here is by MOUNT, not by permission. The render binds
    ``var/spool/<user>`` — one directory, never the spool ROOT — into each
    user's container, so from inside alice's container the root contains her
    directory and no other roster identity's. Widening that bind to the root
    would let one user's process write lines the record attributes to another,
    and this is the assert that trips.

    It has to be read from INSIDE. On the host every directory is visible to
    the account that ran the deploy by construction, so no host-side read can
    tell a per-user bind from a root one.

    The actor == recorded-identity sweep below is a SECOND and different guard:
    it pins that the record takes an entry's identity from the spool directory
    it arrived in, which for a terminal is the user the envelope names. It
    cannot stand in for the mount check, because a container handed the whole
    root would still write under its own identity and still satisfy it.
    """
    container = _web_container(REAL_USER)
    listing = _exec(container, "ls", "-1", CONTAINER_SPOOL_ROOT)
    assert listing.returncode == 0, format_failure(
        f"ls {CONTAINER_SPOOL_ROOT} in {container}", listing
    )
    entries = {name.strip() for name in listing.stdout.splitlines() if name.strip()}
    assert REAL_USER in entries, (
        f"{REAL_USER}'s own spool directory is missing from her container's view of "
        f"{CONTAINER_SPOOL_ROOT}: {sorted(entries)}"
    )
    foreign = (entries & (set(USERS) | {SIDECAR_IDENTITY})) - {REAL_USER}
    assert not foreign, (
        f"{REAL_USER}'s container can see {sorted(foreign)} under {CONTAINER_SPOOL_ROOT}: the "
        f"render bound the spool ROOT instead of var/spool/{REAL_USER} "
        f"(full listing: {sorted(entries)})"
    )

    records = _audit_records(deployment["repo"])
    assert any(record["_identity"] == REAL_USER for record in records), (
        f"no audit entry recorded under {REAL_USER!r}, so the sweep below would pass vacuously"
    )
    for record in records:
        owner = record["_identity"]
        if owner not in USERS:
            continue
        assert record.get("actor") == owner, (
            f"an entry recorded under {owner!r} names {record.get('actor')!r} as its actor: "
            f"{record}"
        )


def test_no_record_carries_a_value_only_an_identifier(deployment: dict[str, Any]) -> None:
    """SC2: envelopes hold identifiers and keys, never the content of a write.

    The refused PATCH carried a distinctive value and a password was checked on
    every login in this run; neither may appear anywhere in the trail. This is
    the property that makes the audit trail safe to hand to an operator who is
    not cleared for the deployment's secrets.

    Reads the trail the tests above left, so it belongs last in the file —
    pytest's definition order is what guarantees the value was actually
    submitted before this looks for it. The sweep covers every audit entry the
    record holds, whichever identity wrote it.

    Guarded non-empty: ``json.dumps([])`` contains no value and no password, so
    a trail that was never written would pass this vacuously.
    """
    records = _audit_records(deployment["repo"])
    assert records, "the audit trail in the record is empty, so this sweep would pass vacuously"

    haystack = json.dumps(records)
    assert _PROTECTED_VALUE not in haystack, "a refused write's VALUE was recorded"
    for password in _PASSWORDS.values():
        assert password not in haystack, "a password reached the audit trail"


def _replay(user: str, cookie: str) -> int:
    """The status a bare request carrying only ``cookie`` gets at ``user``'s terminal."""
    status, _, _ = _request(
        f"/u/{user}/",
        headers={"Cookie": f"{SESSION_COOKIE_NAME}={cookie}", "Accept": "application/json"},
    )
    return status


def test_a_logged_out_cookie_stays_refused_after_the_login_service_is_recreated(
    deployment: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cookie copied before a logout is refused by a recreated sidecar.

    Recreates the sidecar, so it belongs after every test that reads the stack
    as the fixture left it — pytest's definition order is what guarantees they
    ran first. A recreate, not a restart: a restart keeps the container's
    writable layer, so a revocation file written to the wrong place would still
    pass. The container id changing is what proves the refusal came from the
    file on the host's ``var/state/sidecar`` bind.

    The positive control keeps the refusal from being vacuous: a fresh login
    for the same user after the recreate reaches her upstream.
    """
    repo = deployment["repo"]
    client, jar, status, page = _login(NO_ROLE_USER)
    assert status == LOGIN_ACCEPTED, (
        f"login for {NO_ROLE_USER!r} was not accepted (got {status})\n{page[:300]}"
    )
    copied = next((c.value for c in jar if c.name == SESSION_COOKIE_NAME), None)
    assert copied, f"login for {NO_ROLE_USER!r} issued no session cookie\n{runtime_logs(AUTH_C)}"

    status, _, body = _request(
        f"/auth/logout?user={urllib.parse.quote(NO_ROLE_USER)}", opener=client
    )
    assert status == 303, f"logout answered {status}\n{body[:300]}\n{runtime_logs(AUTH_C)}"
    assert _replay(NO_ROLE_USER, copied) == 401, "the logged-out cookie was not refused"

    before = container_id(AUTH_C)
    monkeypatch.setenv("CONTAINER_RUNTIME", RUNTIME)
    config = ConfigBuilder(str(repo / "build" / "config.yml")).raw_config
    force_recreate_auth_sidecar(config, repo_root=repo)
    after = container_id(AUTH_C)
    assert before and after and before != after, (
        f"the sidecar was not recreated (id {before!r} -> {after!r})\n{runtime_logs(AUTH_C)}"
    )

    _wait_for_health(AUTH_READY_TIMEOUT_SEC)
    _wait_for_gate(NO_ROLE_USER, AUTH_READY_TIMEOUT_SEC)
    assert _replay(NO_ROLE_USER, copied) == 401, (
        f"the recreated sidecar accepted a cookie logged out before it\n{runtime_logs(AUTH_C)}"
    )
    revocations = repo / STATE_RELPATH / SIDECAR_IDENTITY
    assert (revocations / REVOCATION_FILE_NAME).exists(), (
        f"no revoked-session file under var/state/{SIDECAR_IDENTITY}/ "
        f"(present: {sorted(p.name for p in revocations.glob('*'))})"
    )

    _probe_report(NO_ROLE_USER, opener=_session_for(NO_ROLE_USER))


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


def _rendered_config(repo: Path) -> dict[str, Any]:
    return yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The history page: `/history/` on the history page's origin, behind real nginx
# ---------------------------------------------------------------------------
#
# The sidecar maps one sign-in to one agent-record viewer (`all`,
# `own:<person>` or `reader:<name>`), nginx forwards that answer with the
# page's gate secret, and the page reads the agent record with its `history`
# login, naming that viewer on every read. These cases mean something only
# with all three real. Every case name carries `history`; the withdrawal
# case runs last because it changes the deployment with a second `osprey up`.

HISTORY_PATH = "/history/"
#: The history page's agent-record login: the one login that may name a viewer.
HISTORY_LOGIN = "history"
#: The ledger directory the page's own audit records land in.
HISTORY_IDENTITY = "history"
#: How long a just-written audit record may take to become readable (the
#: ledger file is appended by another process).
HISTORY_SETTLE_SEC = 15.0
#: How long a just-spooled entry may take to reach the record's reads.
RECORD_INGEST_SEC = 90.0
_PERSON_RE = re.compile(r'<strong class="person">([^<]*)</strong>')


def _auth_none_edits() -> dict[str, Any]:
    """This lane's edits with sign-on off: no `auth.method`, no roles."""
    edits = copy.deepcopy(_profile_edits())
    web_terminals = edits["config"]["modules.web_terminals"]
    web_terminals["auth"] = {"method": "none"}
    web_terminals.pop("authorization")
    for entry in web_terminals["users"]:
        entry.pop("role", None)
    return edits


def _rendered_services(repo: Path) -> dict[str, dict[str, Any]]:
    """Every compose service the render wrote under ``build/``, and the web slice's, by name.

    The web slice (nginx, the terminals, the history page) is rendered by
    ``osprey up`` from ``build/config.yml``, through the function read here.
    """
    from osprey.deployment.web_terminals.render import render_web_terminals

    services: dict[str, dict[str, Any]] = {}
    for path in sorted((repo / "build").rglob("*.yml")):
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if isinstance(document, dict) and isinstance(document.get("services"), dict):
            services.update(document["services"])
    web = yaml.safe_load(render_web_terminals(_rendered_config(repo))["docker-compose.web.yml"])
    services.update(web.get("services") or {})
    return services


def _history_server_block(repo: Path) -> str:
    """The history page's ``server`` block of the nginx config *repo*'s render produces.

    Rendered from ``build/config.yml`` through the function ``osprey up`` writes
    the deployed file with, so it reads exactly what a deploy of that render
    serves.
    """
    from osprey.agent_record.config import history_origin_port
    from osprey.deployment.web_terminals.render import render_web_terminals

    config = _rendered_config(repo)
    port = history_origin_port(config)
    text = render_web_terminals(config)["nginx/nginx.conf"]
    for start in (m.start() for m in re.finditer(r"(?m)^server \{", text)):
        depth, end = 0, start
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        block = text[start:end]
        if re.search(rf"(?m)^\s*listen {port}\b", block):
            return block
    raise AssertionError(f"no rendered server block listens on the history port {port}")


def _record_client(repo: Path, viewer: str) -> Any:
    """A client of the deployed agent record with the page's own login and *viewer*."""
    from osprey.agent_record.client import RecordClient
    from osprey.agent_record.credentials import login_file, read_login
    from osprey.port_layout import resolve_port_base

    config = _rendered_config(repo)
    block = config.get("agent_record") or {}
    port = int(block.get("port") or default_port("agent_record", base=resolve_port_base(config)))
    token = read_login(repo / login_file(HISTORY_LOGIN), HISTORY_LOGIN)
    assert token, f"no {HISTORY_LOGIN!r} login at the repo root"
    return RecordClient(f"http://127.0.0.1:{port}", HISTORY_LOGIN, token, viewer=viewer)


def _record_sessions(repo: Path, viewer: str) -> set[str]:
    """The session ids the agent record lists for *viewer* over the last day."""
    import datetime

    since = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)
    answer = _record_client(repo, viewer).list_sessions(since=since)
    assert answer.available, f"the record is unavailable to {viewer!r}: {answer.unavailable}"
    return {str(row.get("session_id")) for row in answer.data or [] if isinstance(row, dict)}


def _record_ingest_state(repo: Path) -> str:
    """The record's per-source ingest health and the spool's ownership, for a failure message."""
    import subprocess

    import httpx

    from osprey.agent_record.contract import HEALTH_SOURCES_PATH
    from osprey.agent_record.credentials import login_file, read_login
    from osprey.port_layout import resolve_port_base
    from osprey.utils.workspace import SPOOL_DIR_RELPATH

    config = _rendered_config(repo)
    block = config.get("agent_record") or {}
    port = int(block.get("port") or default_port("agent_record", base=resolve_port_base(config)))
    token = read_login(repo / login_file("operator"), "operator")
    try:
        health = httpx.get(
            f"http://127.0.0.1:{port}{HEALTH_SOURCES_PATH}",
            auth=("operator", token or ""),
            timeout=10,
        ).text[:4000]
    except httpx.HTTPError as exc:
        health = f"unreadable: {exc}"
    listing = subprocess.run(
        ["ls", "-lnaR", str(repo / SPOOL_DIR_RELPATH)], capture_output=True, text=True
    ).stdout[:4000]
    return f"--- record health ---\n{health}\n--- spool ---\n{listing}"


#: Where a terminal container's Claude Code transcripts live: its
#: ``<user>-claude-projects`` volume, the transcript root the record tails.
TERMINAL_PROJECTS_DIR = "/data/claude-config/projects"


def _transcript_prompt(identity: str, session: str) -> None:
    """One prompt in *identity*'s terminal transcript, appended the way Claude Code writes it.

    A prompt is a transcript fact: the record takes it from the terminal's
    transcripts volume and from nowhere else. Appended inside that user's own
    container (exact name, see CONTAINER-OPS SAFETY), into its mounted volume.
    """
    import uuid as _uuid
    from datetime import UTC, datetime

    line = {
        "parentUuid": None,
        "isSidechain": False,
        "userType": "external",
        "cwd": "/data/workspace",
        "sessionId": session,
        "version": "2.1.300",
        "type": "user",
        "message": {"role": "user", "content": f"a prompt from {identity}"},
        "uuid": str(_uuid.uuid4()),
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }
    directory = f"{TERMINAL_PROJECTS_DIR}/-e2e-history"
    result = subprocess.run(
        [
            RUNTIME,
            "exec",
            "-i",
            "-u",
            "0",
            _web_container(identity),
            "sh",
            "-c",
            f'mkdir -p "{directory}" && cat >> "{directory}/{session}.jsonl"',
        ],
        input=json.dumps(line) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, format_failure(f"append {identity}'s transcript", result)


@pytest.fixture(scope="module")
def history_page(deployment: dict[str, Any]) -> dict[str, Any]:
    """The two origins and one prompt session per person, in each person's transcript."""
    from osprey.agent_record.config import history_app_port, history_origin_port

    repo: Path = deployment["repo"]
    config = _rendered_config(repo)
    nonce = f"{os.getpid()}-{int(time.time())}"
    sessions = {
        user: f"hist-{nonce}-{user}" for user in (REAL_USER, HEADER_USER, NO_ROLE_USER, SHARED_CARD)
    }
    for user, session in sessions.items():
        _transcript_prompt(user, session)
    listed = _settled(
        lambda: _record_sessions(repo, "all"),
        lambda seen: set(sessions.values()) <= seen,
        timeout=RECORD_INGEST_SEC,
    )
    assert set(sessions.values()) <= listed, (
        f"the record never listed {sessions}: {listed}\n{_record_ingest_state(repo)}"
    )
    return {
        "repo": repo,
        "reader": f"http://127.0.0.1:{history_origin_port(config)}",
        "history_origin_port": history_origin_port(config),
        "history_port": history_app_port(config),
        "terminal": f"http://127.0.0.1:{HOST_PORTS['nginx']}",
        "sessions": sessions,
    }


def _history_sign_in(
    page_info: dict[str, Any],
    user: str,
    client: urllib.request.OpenerDirector | None = None,
) -> urllib.request.OpenerDirector:
    """Sign *user* in on the terminals' entry page, returning to ``/history/``.

    The browser a signed-out visitor is sent to: the card-less password form,
    with the history page as ``next``. Pass *client* to add a second person to
    the same browser.
    """
    from osprey.services.auth_sidecar.routes.entry import ENTRY_PATH

    if client is None:
        client, _ = _browser()
    target = page_info["reader"] + HISTORY_PATH
    body = urllib.parse.urlencode(
        {"username": user, "password": _PASSWORDS[user], "next": target}
    ).encode()
    status, headers, page = _request(
        ENTRY_PATH,
        method="POST",
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": page_info["terminal"],
            **_navigation(),
        },
        opener=client,
    )
    location = {name.lower(): value for name, value in headers.items()}.get("location")
    assert status == LOGIN_ACCEPTED and location == target, (
        f"sign-in for {user!r} answered {status} -> {location!r}\n{page[:300]}\n{runtime_logs(AUTH_C)}"
    )
    return client


def _history_get(
    page_info: dict[str, Any], client: urllib.request.OpenerDirector | None, target: str
) -> tuple[int, str]:
    """A navigation to *target* on the history page's origin with *client*'s cookies."""
    status, _, body = _request(
        target, headers=_navigation(), port=page_info["history_origin_port"], opener=client
    )
    return status, body


def _viewing_as(body: str) -> str | None:
    match = _PERSON_RE.search(body)
    return match.group(1) if match else None


def _settled(read: Any, ready: Any, timeout: float = HISTORY_SETTLE_SEC) -> Any:
    """Re-read until *ready* holds for the result or *timeout* passes; return the last read."""
    deadline = time.monotonic() + timeout
    value = read()
    while not ready(value) and time.monotonic() < deadline:
        time.sleep(0.5)
        value = read()
    return value


def test_history_is_not_served_under_auth_method_none_by_render(tmp_path: Path) -> None:
    """Without sign-on there is no history page, and no server on its origin's port.

    A render, not a second deployment (owner ruling F2): the compose set carries
    no ``history`` service, and nginx renders no server for the history page's
    origin, since the page is all that origin answers.
    """
    from osprey.agent_record.config import history_origin_port
    from osprey.deployment.web_terminals.render import render_web_terminals

    repo = _make_repo(tmp_path, find_osprey_console_script(), edits=_auth_none_edits())

    services = _rendered_services(repo)
    assert "nginx" in services, sorted(services)
    assert "history" not in services, sorted(services)
    config = _rendered_config(repo)
    nginx = render_web_terminals(config)["nginx/nginx.conf"]
    port = history_origin_port(config)
    assert not re.search(rf"(?m)^\s*listen {port}\b", nginx), nginx
    assert "verify-history" not in nginx


def test_history_service_runs_the_project_image_the_render_names(
    deployment: dict[str, Any],
) -> None:
    """The page runs the project image, the tag the lane's cleanup removes."""
    repo: Path = deployment["repo"]
    image = str(_rendered_services(repo)["history"]["image"])
    match = re.fullmatch(r"\$\{OSPREY_WORKER_IMAGE:-(.+)\}", image)
    tag = match.group(1) if match else image
    assert tag == PROJECT_IMAGE_TAG, image
    assert _container_image(HISTORY_C) == tag, runtime_logs(HISTORY_C)
    # The location the auth-none case finds absent is present here.
    assert "location /history/" in _history_server_block(repo)


def test_history_a_forged_header_call_from_a_web_container_is_refused_and_recorded(
    history_page: dict[str, Any],
) -> None:
    """A terminal calling the page past nginx gets 401, filed under `history`."""
    repo: Path = history_page["repo"]

    def refusals() -> list[dict[str, Any]]:
        return [
            record
            for record in _audit_records(repo, HISTORY_IDENTITY)
            if record.get("surface") == "web_auth" and record.get("decision") == "refused"
        ]

    before = len(refusals())
    forged = {
        "X-Osprey-History-Person": REAL_USER,
        "X-Osprey-History-Scope": "all",
        "X-Osprey-History-View": "all",
        "X-Osprey-History-Entry": REAL_USER,
        "X-Osprey-History-Choices": f"{REAL_USER}={REAL_USER}",
        "X-Osprey-Terminal-Secret": "forged-history-gate-secret",
        SUBJECT_HEADER: REAL_USER,
    }
    script = (
        "import urllib.error, urllib.request\n"
        f"request = urllib.request.Request('http://127.0.0.1:{history_page['history_port']}/', "
        f"headers={forged!r})\n"
        "try:\n"
        "    print(urllib.request.urlopen(request, timeout=10).status)\n"
        "except urllib.error.HTTPError as error:\n"
        "    print(error.code)\n"
    )
    result = _exec(_web_container(HEADER_USER), "python3", "-c", script)
    assert result.returncode == 0, format_failure("forged history call", result)
    assert result.stdout.strip() == "401", format_failure("forged history call", result)

    seen = _settled(refusals, lambda records: len(records) > before)
    assert len(seen) > before, (
        f"no web_auth refusal recorded under {HISTORY_IDENTITY!r}\n{runtime_logs(HISTORY_C)}"
    )
    assert all(record.get("actor") == HISTORY_IDENTITY for record in seen[before:]), seen[before:]


def test_history_an_unauthenticated_visit_signs_in_on_the_terminals_and_returns_to_the_page(
    history_page: dict[str, Any],
) -> None:
    from playwright.sync_api import sync_playwright

    from osprey.services.auth_sidecar.routes.entry import ENTRY_PATH

    target = f"{history_page['reader']}/history/session/{history_page['sessions'][REAL_USER]}"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(target, wait_until="load")
            landed = urllib.parse.urlsplit(page.url)
            assert f"{landed.scheme}://{landed.netloc}" == history_page["terminal"], page.url
            assert landed.path == ENTRY_PATH, page.url

            page.fill("#username", REAL_USER)
            page.fill("#password", _PASSWORDS[REAL_USER])
            with page.expect_navigation(url=target):
                page.click("button.login-submit")
            assert page.url == target
            assert page.locator("strong.person").inner_text() == REAL_USER
            assert page.locator("section.transcript").count() == 1, page.content()[:2000]
        finally:
            browser.close()


def test_history_a_two_person_console_shows_each_person_their_own_scope(
    history_page: dict[str, Any],
) -> None:
    """Both people in one browser: each selector shows that person's own sessions,
    and with no selector the last sign-in is shown, again after a re-login."""
    sessions = history_page["sessions"]
    client = _history_sign_in(history_page, REAL_USER)
    _history_sign_in(history_page, HEADER_USER, client)

    for person, other in ((REAL_USER, HEADER_USER), (HEADER_USER, REAL_USER)):
        status, body = _history_get(history_page, client, f"{HISTORY_PATH}?as={person}")
        assert status == 200, body[:400]
        assert _viewing_as(body) == person, body[:400]
        assert "scope-own" in body
        assert sessions[person] in body, f"{person}'s own session is not listed"
        assert sessions[other] not in body, f"{other}'s session is listed for {person}"

    status, body = _history_get(history_page, client, HISTORY_PATH)
    assert status == 200 and _viewing_as(body) == HEADER_USER, body[:400]

    _history_sign_in(history_page, REAL_USER, client)
    status, body = _history_get(history_page, client, HISTORY_PATH)
    assert status == 200 and _viewing_as(body) == REAL_USER, body[:400]
    assert sessions[HEADER_USER] not in body


def _browser_sign_in(page: Any, history_page: dict[str, Any], user: str) -> None:
    """Sign *user* in in a real browser the way a signed-out visitor is: via the page."""
    target = history_page["reader"] + HISTORY_PATH
    page.goto(target, wait_until="load")
    page.fill("#username", user)
    page.fill("#password", _PASSWORDS[user])
    with page.expect_navigation(url=target):
        page.click("button.login-submit")


def _artifact_url(history_page: dict[str, Any], user: str, mode: str) -> str:
    query = urllib.parse.urlencode({"mode": mode, "target": history_page["reader"] + HISTORY_PATH})
    return f"{history_page['terminal']}/u/{user}/artifact.html?{query}"


def test_history_an_artifact_under_u_cannot_read_a_history_response(
    history_page: dict[str, Any],
) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            _browser_sign_in(page, history_page, HEADER_USER)
            page.goto(_artifact_url(history_page, HEADER_USER, "read"), wait_until="load")
            page.wait_for_function(
                "window.__osprey_artifact && window.__osprey_artifact !== 'pending'",
                timeout=15_000,
            )
            outcome = page.evaluate("window.__osprey_artifact")
            assert outcome.startswith("blocked:"), f"the artifact read the history page: {outcome}"
        finally:
            browser.close()


def _history_page_statuses() -> list[int]:
    """The status of every request for ``/history/`` the history page's nginx server logged.

    A refusal is logged under the 403 page it was answered with, the internal
    redirect ``error_page`` makes, so both paths are the page's requests.
    """
    result = runtime_cli("logs", NGINX_C, timeout=30)
    return [
        int(match.group(1))
        for match in re.finditer(
            r'"GET /history(?:/|-403\.html) HTTP/[0-9.]+" (\d{3}) ',
            result.stdout + result.stderr,
        )
    ]


def test_history_an_iframe_and_an_image_from_an_artifact_run_no_read(
    history_page: dict[str, Any],
) -> None:
    """nginx refuses both embeds with 403 before the sign-in is asked, so the
    page never runs and reads nothing of the record; the same browser opening
    the page is answered by it."""
    from playwright.sync_api import sync_playwright

    reader_history = history_page["reader"] + HISTORY_PATH
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            _browser_sign_in(page, history_page, HEADER_USER)
            # Every request the browser sent for the page, by kind, and the
            # status each was answered with (an image the browser refuses to
            # decode still has nginx's response).
            sent: list[str] = []
            answered: list[tuple[str, int]] = []
            page.on(
                "request",
                lambda request: (
                    sent.append(request.resource_type) if request.url == reader_history else None
                ),
            )
            page.on(
                "response",
                lambda response: (
                    answered.append((response.request.resource_type, response.status))
                    if response.url == reader_history
                    else None
                ),
            )
            failed: list[tuple[str, str | None]] = []
            page.on(
                "requestfailed",
                lambda request: (
                    failed.append((request.resource_type, request.failure))
                    if request.url == reader_history
                    else None
                ),
            )
            logged = len(_history_page_statuses())
            page.goto(_artifact_url(history_page, HEADER_USER, "embed"), wait_until="load")
            _settled(lambda: sorted(sent), lambda seen: len(seen) >= 2)
            assert sorted(sent) == ["document", "image"], (
                f"the artifact's iframe and image never reached /history/: sent {sent}"
            )
            # What nginx answered each embed, from its own log: the 403 page's
            # resource policy keeps the browser from handing a refused image's
            # response to the page at all, so the browser cannot report it.
            statuses = _settled(
                lambda: _history_page_statuses()[logged:], lambda seen: len(seen) >= 2
            )
            assert statuses == [403, 403], (
                f"an embed of /history/ was not refused before the page: nginx answered "
                f"{statuses}; the browser saw {answered} and failed {failed}"
            )

            # The positive control: the same browser opening the page is
            # answered by it, as the person, from the record.
            response = page.goto(reader_history, wait_until="load")
            assert response is not None and response.status == 200, page.content()[:400]
            assert page.locator("strong.person").inner_text() == HEADER_USER
            assert history_page["sessions"][HEADER_USER] in page.content()
        finally:
            browser.close()


def test_history_a_shared_card_session_is_absent_from_the_openers_own_view_and_present_under_all(
    history_page: dict[str, Any],
) -> None:
    sessions = history_page["sessions"]
    opener = _history_sign_in(history_page, REAL_USER)
    for entry in (REAL_USER, SHARED_CARD):
        status, body = _history_get(history_page, opener, f"{HISTORY_PATH}?as={entry}")
        assert status == 200, body[:400]
        assert _viewing_as(body) == REAL_USER, f"as={entry}: {body[:400]}"
        assert sessions[REAL_USER] in body
        assert sessions[SHARED_CARD] not in body, f"the shared card's session shows as={entry}"

    viewer = _history_sign_in(history_page, ALL_USER)
    status, body = _history_get(history_page, viewer, HISTORY_PATH)
    assert status == 200, body[:400]
    assert _viewing_as(body) == ALL_USER and "scope-all" in body, body[:400]
    assert sessions[SHARED_CARD] in body and sessions[REAL_USER] in body


def test_history_a_sign_in_as_a_shared_card_without_all_lists_nothing(
    history_page: dict[str, Any],
) -> None:
    client = _history_sign_in(history_page, SHARED_CARD)
    status, body = _history_get(history_page, client, HISTORY_PATH)
    assert status == 403, f"{status}: {body[:400]}"
    listed = [session for session in history_page["sessions"].values() if session in body]
    assert listed == [], listed


@pytest.fixture(scope="module")
def history_withdrawal(deployment: dict[str, Any], history_page: dict[str, Any]) -> dict[str, Any]:
    """``carol`` leaves ``readers[]``, then a real ``osprey up``.

    ``carol``'s browser is signed in before and kept open across it. What held
    before is asserted here, so the cases after it compare against a state
    that was real.
    """
    repo: Path = deployment["repo"]
    sessions = history_page["sessions"]

    carol = _history_sign_in(history_page, NO_ROLE_USER)
    status, body = _history_get(history_page, carol, HISTORY_PATH)
    assert status == 200 and "scope-reader" in body, body[:400]
    assert sessions[REAL_USER] in body, "the reader's scope does not show alice's session"
    # The page read as the reader the sidecar named: the record lists alice's
    # session to that viewer and not to carol's own.
    assert sessions[REAL_USER] in _record_sessions(repo, f"reader:{NO_ROLE_USER}")
    assert sessions[REAL_USER] not in _record_sessions(repo, f"own:{NO_ROLE_USER}")

    profile_path = repo / "profile.yml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    record = profile["agent_record"]
    record["readers"] = [r for r in record["readers"] if r["name"] != NO_ROLE_USER]
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    osprey_bin = find_osprey_console_script()
    build = run_osprey(
        osprey_bin,
        ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle", "--dev"],
        repo,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert build.returncode == 0, format_failure("osprey build (history withdrawal)", build)
    up = run_osprey(osprey_bin, ["up", "--dev"], repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
    assert up.returncode == 0, format_failure("osprey up --dev (history withdrawal)", up)
    _wait_for_health(AUTH_READY_TIMEOUT_SEC)
    return {"carol": carol}


def test_history_a_reader_withdrawn_from_readers_is_back_in_their_own_scope_after_up(
    history_page: dict[str, Any], history_withdrawal: dict[str, Any]
) -> None:
    sessions = history_page["sessions"]
    carol = history_withdrawal["carol"]

    def answer() -> tuple[int, str]:
        return _history_get(history_page, carol, HISTORY_PATH)

    status, body = _settled(answer, lambda reply: reply[0] == 200)
    assert status == 200, body[:400]
    assert _viewing_as(body) == NO_ROLE_USER and "scope-own" in body, body[:400]
    assert sessions[NO_ROLE_USER] in body
    assert sessions[REAL_USER] not in body, "the withdrawn reader still sees alice's session"
    # The record no longer configures the reader: a read naming it is refused,
    # and carol's own viewer still lists her session alone.
    withdrawn = _record_client(history_page["repo"], f"reader:{NO_ROLE_USER}").list_sessions()
    assert not withdrawn.available, withdrawn
    own = _record_sessions(history_page["repo"], f"own:{NO_ROLE_USER}")
    assert sessions[NO_ROLE_USER] in own and sessions[REAL_USER] not in own, own
