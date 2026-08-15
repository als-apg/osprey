"""Real-container e2e for the durability of the secrets a deploy mints.

``tests/deployment/test_service_tokens_writeback.py`` pins the minting rules
against hand-built directory trees and a directly-called
``_ensure_service_tokens``. What it cannot show is the property the feature
exists for: that a facility can **wipe and rebuild its deployment** and the
stack comes back up on the secrets its *already-initialized docker volumes*
were created with. That claim only holds if the whole chain holds at once —
``osprey init`` writing the repo, ``osprey build`` rendering ``build/`` from
it, ``osprey up`` minting into the repo's ``.env``, the containers adopting
those values, a rebuild that replaces ``build/`` whole leaving those values
alone, and the redeployed containers still authenticating against the *same*
volumes. Every link is a different module, and a break in any one of them is
invisible to a unit test that owns only its own link.

The layout is what makes that survivable, and it is the thing under test: a
deployment repo keeps ``.env`` at its root, beside ``profile.yml``, while
``build/`` is derived in full and disposable. A minted secret therefore lives
outside everything a build replaces — and if it did not, every rebuild would
lock the facility out of its own volumes.

So this file drives the real CLI against a real container runtime:

* **rebuild** (:func:`test_minted_secrets_survive_a_rebuild_and_a_redeploy`)
  — ``osprey up`` mints ``ZO_ROOT_USER_PASSWORD`` and ``ARIEL_DB_PASSWORD``
  into ``<repo>/.env`` under their own minted section, and OpenObserve and
  Postgres both actually authenticate with them. ``build/`` is then deleted
  outright and re-rendered, the containers are dropped while their volumes are
  kept, and the redeploy has to bring *fresh* containers up on the *identical*
  secrets against the *same* volumes — proven by a row written into Postgres
  before the rebuild still being readable after it. These are exactly the two
  keys that would otherwise degrade the deployment silently: both compose
  templates carry an insecure ``${VAR:-default}``, and both stores read their
  password only when *initializing* a fresh volume, so a re-minted secret does
  not fail loudly — it locks the facility out of a store that is still happily
  running on the old one.

* **contradicted by the shell**
  (:func:`test_divergent_shell_export_never_overwrites_the_pinned_secret`)
  — the same deployment, with a secret exported into the shell that disagrees
  with the one in ``.env``. Minting is append-only and never rewrites a value
  that is already there, so the file stays byte-identical: the value the
  running volumes were initialized with remains recoverable, and neither value
  is ever printed.

* **persona stack**
  (:func:`test_personas_are_rendered_by_the_build_from_this_repos_own_deltas`)
  — the same repo layout seen from the persona side. ``osprey build`` renders
  each persona project from a delta in this repo's own ``personas/``, into
  ``build/`` where the rest of the render lives, so a persona is as disposable
  as everything else a build produces and a start has one to verify. This case
  needs no container: the render is the subject.

----------------------------------------------------------------------------
CONTAINER-OPS SAFETY (every runtime-mutating call in this file honors this)
----------------------------------------------------------------------------
Every container and volume here is exact-named off a deployment name this file
owns (``osprey-e2e-wb-*`` — the repo DIRECTORY name, which is what the render
records as ``project_name`` and what compose scopes its resources by), so
nothing it removes can belong to another stack:

  * containers: ``osprey-e2e-wb-<case>-openobserve`` /
    ``osprey-e2e-wb-<case>-ariel-postgres``
  * volumes:    ``osprey-e2e-wb-<case>_openobserve_data`` /
    ``osprey-e2e-wb-<case>_ariel_postgres_data``

Nothing here ever runs ``system prune``, ``volume prune``, ``container
prune``, ``-a``/``--all`` on a removal, or a wildcard container-name match.
Every teardown call names one exact resource or uses ``compose -p <project>
down`` (project-label-scoped, containers/networks only, never ``-v``), and runs
from a ``finally`` so a failed assertion mid-sequence still cleans up.

Only public upstream images are pulled (``pgvector/pgvector:pg16``,
``public.ecr.aws/zinclabs/openobserve``) — no image is built here, which is
what keeps a full run to minutes rather than the tens of minutes a persona or
project image would cost.

Gating: needs a container runtime (daemon actually running, not just the CLI
installed). Runtime is ``docker`` by default; set ``OSPREY_E2E_RUNTIME=podman``
to run against podman instead (any other value fails at collection time with a
clear error).
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

from osprey.deployment.reset import MINTED_ENV_BANNERS
from osprey.utils.dotenv import parse_dotenv_text

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.dockerbuild]

# ---------------------------------------------------------------------------
# Container runtime selection — same contract as tests/e2e/test_deploy_lifecycle.py:
# ``docker`` by default, ``podman`` opt-in via OSPREY_E2E_RUNTIME (the CI podman
# lane sets it). Any other value fails clearly at collection time rather than
# silently falling back to docker.
# ---------------------------------------------------------------------------
_SUPPORTED_RUNTIMES = ("docker", "podman")
RUNTIME = os.environ.get("OSPREY_E2E_RUNTIME", "docker")
if RUNTIME not in _SUPPORTED_RUNTIMES:
    raise RuntimeError(
        f"OSPREY_E2E_RUNTIME={RUNTIME!r} is not supported; expected one of {_SUPPORTED_RUNTIMES}"
    )

# ---------------------------------------------------------------------------
# Identity. One deployment name per case — the repo DIRECTORY name, which the
# render records as `project_name` and compose scopes its resources by — so the
# tests can run in any order, or concurrently with anything else on the host,
# without sharing a container name, a volume namespace, or a host port.
#
# Ports sit in a band disjoint from every other e2e file's (test_deploy_lifecycle
# spans 19081-20601, test_dispatch_deploy publishes 8020) AND from the ports a
# developer's own demo stack conventionally holds (5064/5080/5432).
# ---------------------------------------------------------------------------
PROJECT_REBUILD = "osprey-e2e-wb-rebuild"
PROJECT_DIVERGENT = "osprey-e2e-wb-divergent"
PROJECT_PERSONA = "osprey-e2e-wb-persona"

PORTS_REBUILD = {"postgres": 21432, "openobserve": 21482}
PORTS_DIVERGENT = {"postgres": 21433, "openobserve": 21483}
PORTS_PERSONA = {"postgres": 21435, "openobserve": 21485}
# Per-user web-terminal port families for the persona case. Never bound: the
# persona deploy is driven only as far as its preflight (see that test), so
# these exist to keep the host-port preflight itself conflict-free.
PORTS_PERSONA_WEB = {
    "nginx": 21600,
    "web": 21610,
    "artifact": 21620,
    "ariel": 21630,
    "lattice": 21640,
    "channel_finder": 21650,
}

# The two secrets under test. Both are *volume-pinned*: OpenObserve and Postgres
# read their password only when initializing a fresh data volume, so a re-minted
# value locks the operator out of a store that keeps running on the old one —
# which is precisely why `.env` has to sit outside everything a build replaces.
ZO_PASSWORD_VAR = "ZO_ROOT_USER_PASSWORD"
DB_PASSWORD_VAR = "ARIEL_DB_PASSWORD"
OPENOBSERVE_USER = "root@example.com"  # the compose template's non-secret default

# Vars an operator's shell may legitimately export, which ``_effective_value``
# would then treat as authoritative — stripped from every subprocess so this
# file's results never depend on the environment it is run from. The divergent-
# export case re-adds one deliberately.
_AMBIENT_SECRET_VARS = (
    ZO_PASSWORD_VAR,
    DB_PASSWORD_VAR,
    "EVENT_DISPATCHER_TOKEN",
    "DISPATCH_WORKER_TOKEN",
    "BLUESKY_LAUNCH_TOKEN",
    "BLUESKY_TILED_API_KEY",
    "ARIEL_DSN",
)

BUILD_TIMEOUT_SEC = 600
DEPLOY_TIMEOUT_SEC = 600
SERVICE_READY_TIMEOUT_SEC = 120.0

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


# ---------------------------------------------------------------------------
# Subprocess + output helpers
# ---------------------------------------------------------------------------


def _find_osprey_console_script() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment every ``osprey`` subprocess in this file runs under.

    ``CONTAINER_RUNTIME`` structurally couples the deploy-side runtime to the
    assert-side :func:`_runtime_cli` runtime. ``OSPREY_PIP_SPEC`` is the
    documented operator escape from the unreleased-version pin refusal; no image
    is built here, so its value is inert — but it stays a loud failure so that a
    future change which *does* reach an image build cannot silently install a
    real release. Every var in ``_AMBIENT_SECRET_VARS`` is dropped: they are
    read from the ambient environment by the very code under test.
    """
    env = {
        **os.environ,
        "CLAUDECODE": "",
        "CONTAINER_RUNTIME": RUNTIME,
        "OSPREY_PIP_SPEC": "osprey-framework==0.0.0+e2e-stub",
        # Rich hard-wraps its log lines to the reported terminal width; a wide
        # value keeps failure output readable. Assertions never rely on it —
        # see _squash.
        "COLUMNS": "200",
    }
    for var in _AMBIENT_SECRET_VARS:
        env.pop(var, None)
    env.update(extra or {})
    return env


def _run_osprey(
    args: list[str],
    cwd: Path,
    timeout: int = DEPLOY_TIMEOUT_SEC,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(_find_osprey_console_script()), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_child_env(extra_env),
    )


def _runtime_cli(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run([RUNTIME, *args], capture_output=True, text=True, timeout=timeout)


def _fmt(label: str, result: subprocess.CompletedProcess) -> str:
    return (
        f"{label} failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def _squash(result: subprocess.CompletedProcess | str) -> str:
    """Combined output with ANSI escapes and ALL whitespace removed.

    The CLI logs through rich, which colorizes and hard-wraps: a path or a
    sentence in the output is routinely split across lines mid-token. Removing
    whitespace entirely makes any contiguous fragment searchable regardless of
    where the wrap landed, so assertions pin *what was said* rather than the
    terminal width it was said at. Compare against :func:`_needle`.
    """
    text = result if isinstance(result, str) else (result.stdout or "") + (result.stderr or "")
    return "".join(_ANSI_RE.sub("", text).split())


def _needle(text: str) -> str:
    """A literal put through the same squashing as :func:`_squash`'s haystack."""
    return "".join(text.split())


# ---------------------------------------------------------------------------
# Deployment-repo construction
# ---------------------------------------------------------------------------


def _services_override(ports: dict[str, int]) -> str:
    """A ``-O`` overlay adding a Postgres service beside the preset's OpenObserve.

    Two services, deliberately: they are the two ``_SERVICE_TOKEN_VARS`` entries
    whose secret initializes a docker volume, and covering both proves the mint
    is not special-casing one recipe (``ARIEL_DB_PASSWORD`` is hex,
    ``ZO_ROOT_USER_PASSWORD`` is the four-character-class OpenObserve policy).

    Dotted leaf keys under ``config:``, the one spelling the profile's config
    block accepts — a nested mapping would wholesale-replace the rendered
    subtree instead of setting the addressed leaf.
    """
    return (
        "config:\n"
        "  services.postgresql.path: ./services/postgresql\n"
        "  services.postgresql.database_name: ariel\n"
        "  services.postgresql.username: ariel\n"
        f"  services.postgresql.port_host: {ports['postgres']}\n"
        f"  services.openobserve.port: {ports['openobserve']}\n"
        "  deployed_services:\n"
        "    - openobserve\n"
        "    - postgresql\n"
    )


def _init_and_build(base_dir: Path, name: str, preset: str, override_text: str) -> Path:
    """``osprey init <dir> --preset P -O F`` then ``osprey build --repo <dir>``.

    Two commands, because the surface has two: ``init`` writes the repo's source
    zone from the preset, ``build`` renders ``build/`` from it. The repo
    DIRECTORY name is the deployment name, so ``name`` is what every container,
    volume and image tag below is derived from.

    ``--skip-deps``/``--skip-lifecycle`` keep the render network-free and quick:
    nothing here runs the deployment's own venv, only its compose stack.
    ``--no-git`` because no test reads the history.

    Returns the repo root.
    """
    override_path = base_dir / f"{name}-override.yml"
    override_path.write_text(override_text, encoding="utf-8")
    repo = base_dir / name

    init = _run_osprey(
        [
            "init",
            str(repo),
            "--preset",
            preset,
            "--no-git",
            "--override",
            str(override_path),
        ],
        cwd=base_dir,
        timeout=BUILD_TIMEOUT_SEC,
    )
    assert init.returncode == 0, _fmt(f"osprey init {name}", init)

    build = _run_osprey(
        ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"],
        cwd=base_dir,
        timeout=BUILD_TIMEOUT_SEC,
    )
    assert build.returncode == 0, _fmt(f"osprey build {name}", build)
    assert (repo / "build" / "config.yml").is_file(), f"osprey build rendered nothing in {repo}"

    _seed_repo_env(repo)
    return repo


def _seed_repo_env(repo: Path) -> None:
    """Give the repo the ``.env`` ``osprey up`` refuses to start without.

    The repo root's ``.env`` is the deployment's whole secret store and the file
    every compose invocation is pointed at, so ``up`` aborts when it is absent.
    ``osprey init`` writes one only when the shell exports a key for the
    profile's provider, and these cases deliberately run with every relevant
    variable stripped (see :func:`_child_env`) — this is the ``cp .env.example
    .env`` the CLI itself recommends, done for the operator.
    """
    env_path = repo / ".env"
    if not env_path.exists():
        shutil.copy(repo / ".env.example", env_path)


def _env_of(path: Path) -> dict[str, str]:
    return parse_dotenv_text(path.read_text(encoding="utf-8"))


def _persona_catalog(repo: Path) -> dict[str, dict]:
    """``modules.web_terminals.personas`` as rendered into ``build/config.yml``."""
    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    return web_terminals.get("personas") or {}


# ---------------------------------------------------------------------------
# Container-side assertions
# ---------------------------------------------------------------------------


def _pg_container(project_name: str) -> str:
    return f"{project_name}-ariel-postgres"


def _openobserve_container(project_name: str) -> str:
    return f"{project_name}-openobserve"


def _volume_names(project_name: str) -> tuple[str, str]:
    """The two named volumes compose creates for this project's stores."""
    return (f"{project_name}_openobserve_data", f"{project_name}_ariel_postgres_data")


def _volume_exists(name: str) -> bool:
    return _runtime_cli("volume", "inspect", name, timeout=15).returncode == 0


def _container_exists(name: str) -> bool:
    return _runtime_cli("inspect", "--type", "container", name, timeout=15).returncode == 0


def _psql(project_name: str, password: str, sql: str) -> subprocess.CompletedProcess:
    """Run one statement against the Postgres container, authenticating with ``password``.

    The connection is made to ``ariel-postgres`` — the network alias the compose
    template pins so in-network consumers can resolve the store — and NOT to
    ``127.0.0.1``. That distinction is the whole reason this helper is a real
    credential check: the upstream image's ``pg_hba.conf`` trusts loopback
    connections unconditionally, so a ``-h 127.0.0.1`` connection succeeds with
    any password at all (or none), which would make both the positive and the
    negative assertion below vacuous. Reaching the container by its network
    address instead lands on the ``scram-sha-256`` rule, where ``PGPASSWORD`` is
    genuinely checked against what the volume was initialized with.
    """
    return _runtime_cli(
        "exec",
        "-e",
        f"PGPASSWORD={password}",
        _pg_container(project_name),
        "psql",
        "-h",
        "ariel-postgres",
        "-U",
        "ariel",
        "-d",
        "ariel",
        "-tAc",
        sql,
        timeout=30,
    )


def _wait_for_postgres(project_name: str, password: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = "(no attempt yet)"
    while time.monotonic() < deadline:
        result = _psql(project_name, password, "select 1")
        if result.returncode == 0 and result.stdout.strip() == "1":
            return
        last = (result.stdout + result.stderr).strip() or f"rc={result.returncode}"
        time.sleep(2.0)
    raise AssertionError(
        f"postgres in {_pg_container(project_name)} never accepted the deploy's "
        f"ARIEL_DB_PASSWORD within {timeout:.0f}s (last: {last})"
    )


def _openobserve_status(port: int, password: str, path: str = "/api/default/streams") -> int:
    """HTTP status for a Basic-auth request to OpenObserve; ``0`` if unreachable.

    ``/api/default/streams`` is an authenticated read: it answers 200 for the
    root credentials and 401 for anything else, which makes it a real
    credential check rather than a liveness probe (``/healthz`` answers 200
    unauthenticated).
    """
    token = base64.b64encode(f"{OPENOBSERVE_USER}:{password}".encode()).decode()
    request = urllib.request.Request(  # noqa: S310 - localhost only
        f"http://127.0.0.1:{port}{path}",
        method="GET",
        headers={"Authorization": f"Basic {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:  # noqa: S310
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, ConnectionError, OSError):
        return 0


def _wait_for_openobserve(port: int, password: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = 0
    while time.monotonic() < deadline:
        last = _openobserve_status(port, password)
        if last == 200:
            return
        time.sleep(2.0)
    raise AssertionError(
        f"openobserve on :{port} never authenticated the deploy's {ZO_PASSWORD_VAR} "
        f"within {timeout:.0f}s (last HTTP status: {last or 'unreachable'})"
    )


def _assert_stores_authenticate(
    project_name: str, ports: dict[str, int], env: dict[str, str]
) -> None:
    """Both stores accept the deploy's secrets — and reject a wrong one.

    The negative half matters as much as the positive: OpenObserve's compose
    template carries a publicly-known ``${ZO_ROOT_USER_PASSWORD:-Complexpass#123}``
    fallback, so a store that answered 200 to *anything* would pass a
    positive-only check while being wide open.
    """
    _wait_for_postgres(project_name, env[DB_PASSWORD_VAR], SERVICE_READY_TIMEOUT_SEC)
    _wait_for_openobserve(ports["openobserve"], env[ZO_PASSWORD_VAR], SERVICE_READY_TIMEOUT_SEC)

    wrong_pg = _psql(project_name, "not-the-minted-password", "select 1")
    assert wrong_pg.returncode != 0, (
        "postgres accepted a password the deploy never minted — the volume is not "
        "actually protected by ARIEL_DB_PASSWORD"
    )
    # Pin WHY it failed: a rejected connection and an unresolvable host both
    # exit non-zero, and only one of them is evidence about the password.
    assert "password authentication failed" in wrong_pg.stderr, (
        f"postgres refused the wrong password for the wrong reason:\n{wrong_pg.stderr}"
    )
    assert _openobserve_status(ports["openobserve"], "not-the-minted-password") == 401, (
        "openobserve did not reject a wrong root password — it is likely still "
        "running on the compose template's public default"
    )


def _teardown_project(project_name: str) -> None:
    """Exact-named sweep of everything this file could have created for a project.

    Best-effort: failures (resource never created, already gone) are swallowed —
    this is a safety net, not an assertion. Every call names one exact container,
    one exact volume, or is the project-scoped ``compose down`` the guardrail
    allows (label-based, containers/networks only, never ``-v``).
    """
    _runtime_cli("rm", "-f", _openobserve_container(project_name))
    _runtime_cli("rm", "-f", _pg_container(project_name))
    _runtime_cli("compose", "-p", project_name, "down", timeout=120)
    for volume in _volume_names(project_name):
        _runtime_cli("volume", "rm", volume)


@pytest.fixture(autouse=True)
def _require_container_runtime() -> None:
    """Skip the module gracefully off a runtime host (CLI absent or daemon down)."""
    if shutil.which(RUNTIME) is None:
        pytest.skip(f"{RUNTIME} not available")
    if _runtime_cli("ps", timeout=15).returncode != 0:
        pytest.skip(f"{RUNTIME} daemon not responding")


# =============================================================================
# (a) A rebuild is not a way to lose a secret
# =============================================================================


def test_minted_secrets_survive_a_rebuild_and_a_redeploy(tmp_path: Path) -> None:
    """Mint, wipe build/, rebuild, redeploy on the same volumes.

    The sequence, and what each step is there to prove:

    1. ``osprey init`` + ``osprey build`` produce the deployment. Its ``.env``
       carries no service secret yet — the deploy is what brings these into
       existence, so a repo that already carried them would prove nothing.
    2. ``osprey up`` mints both secrets into ``<repo>/.env`` under their own
       minted section, and both containers authenticate with them (and reject a
       wrong password).
    3. A row is written into Postgres — the witness that step 6 is talking to
       the same volume, not a fresh one that happens to accept the password.
    4. ``build/`` is deleted outright and ``osprey build`` re-renders it. That
       is the whole claim of the layout: the render zone is disposable, the
       secrets zone is not, and the overlay is deliberately NOT passed again —
       everything it configured lives in ``profile.yml`` now, so a deployment
       that comes back with the same ports is the source zone doing its job.
    5. ``osprey down`` removes the containers and keeps both volumes, so what
       follows cannot be the step-2 processes still running on the secret they
       were started with.
    6. ``osprey up`` again: FRESH containers, the same volumes, the same
       secrets, and the row from step 3 still readable.
    """
    project_name = PROJECT_REBUILD
    ports = PORTS_REBUILD

    try:
        repo = _init_and_build(tmp_path, project_name, "hello-world", _services_override(ports))
        env_path = repo / ".env"
        pre_deploy = _env_of(env_path)
        for var in (ZO_PASSWORD_VAR, DB_PASSWORD_VAR):
            assert not pre_deploy.get(var), (
                f"{var} is already in {env_path} before any deploy — this test cannot "
                "then show that the deploy is what put it there"
            )

        # -- 2. first deploy: mint -> .env -> containers ----------------------
        up1 = _run_osprey(["up", "-d"], repo)
        assert up1.returncode == 0, _fmt("osprey up (first)", up1)

        assert env_path.stat().st_mode & 0o777 == 0o600, (
            "the repo .env holds this deployment's secrets and must be kept private"
        )
        env_text = env_path.read_text(encoding="utf-8")
        assert MINTED_ENV_BANNERS[0] in env_text, (
            "minted secrets landed in .env without the banner that marks them as "
            f"deploy-minted, which is what `osprey reset` strips them by:\n{env_text}"
        )

        minted = _env_of(env_path)
        for var in (ZO_PASSWORD_VAR, DB_PASSWORD_VAR):
            assert minted.get(var), f"{var} was not minted into {env_path}"

        _assert_stores_authenticate(project_name, ports, minted)

        # -- 3. a witness row, so step 6 proves SAME VOLUME, not just same password
        seeded = _psql(
            project_name,
            minted[DB_PASSWORD_VAR],
            "create table wb_witness (note text); insert into wb_witness values ('pre-rebuild')",
        )
        assert seeded.returncode == 0, _fmt("seed the witness row", seeded)

        volumes = _volume_names(project_name)
        assert all(_volume_exists(volume) for volume in volumes), (
            f"expected both named volumes to exist after the first deploy: {volumes}"
        )

        # -- 4. delete the whole render, then rebuild it ----------------------
        # `rm -rf build/` is the operator gesture the layout promises costs
        # nothing. Deleting it rather than rebuilding over it is what makes the
        # assertion below about the ZONE rather than about this build happening
        # to leave the file alone.
        shutil.rmtree(repo / "build")
        rebuild = _run_osprey(
            ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"],
            cwd=tmp_path,
            timeout=BUILD_TIMEOUT_SEC,
        )
        assert rebuild.returncode == 0, _fmt("osprey build (re-render)", rebuild)
        assert (repo / "build" / "config.yml").is_file(), "the rebuild rendered nothing"

        rebuilt_env = _env_of(env_path)
        for var in (ZO_PASSWORD_VAR, DB_PASSWORD_VAR):
            assert rebuilt_env.get(var) == minted[var], (
                f"{var} did not survive the rebuild: .env came back with a different "
                "value, so the redeployed stack would be locked out of the volume it "
                "initialized"
            )

        # -- 5. drop the containers, KEEP the volumes -------------------------
        # Without this the redeploy below can be a no-op: compose recreates a
        # container only when its definition changed, and after a rebuild that
        # re-rendered the same values nothing has. The step-2 processes would
        # still be running, still holding the secret they were STARTED with,
        # and the authentication assertion in step 6 would re-confirm them
        # rather than prove that a fresh container adopted the re-read value.
        # `osprey down` is compose `down` with no ``-v``, so it removes this
        # deployment's containers and network and leaves both named volumes —
        # exactly the "same volumes, new containers" state the claim needs.
        down = _run_osprey(["down"], repo)
        assert down.returncode == 0, _fmt("osprey down (between rebuild and redeploy)", down)
        assert all(_volume_exists(volume) for volume in volumes), (
            "`osprey down` removed a named volume — it must tear down containers "
            "only, or the redeploy below would meet a fresh store and prove nothing "
            "about the secret surviving"
        )
        for container in (_pg_container(project_name), _openobserve_container(project_name)):
            assert not _container_exists(container), (
                f"{container} survived `osprey down`, so the redeploy below would "
                "reuse the process started in step 2 and the authentication assertion "
                "would say nothing about the re-read secret"
            )

        # -- 6. redeploy: same volumes, new containers, same secrets ----------
        up2 = _run_osprey(["up", "-d"], repo)
        assert up2.returncode == 0, _fmt("osprey up (after rebuild)", up2)
        assert all(_volume_exists(volume) for volume in volumes), (
            "the redeploy did not run against the volumes the first deploy created"
        )

        _assert_stores_authenticate(project_name, ports, minted)

        witness = _psql(project_name, minted[DB_PASSWORD_VAR], "select note from wb_witness")
        assert witness.returncode == 0, _fmt("read the witness row after the rebuild", witness)
        assert witness.stdout.strip() == "pre-rebuild", (
            "the witness row is gone — the redeploy came up against a fresh volume, so "
            "'the secret survived' would have been vacuously true"
        )

    finally:
        _teardown_project(project_name)


def test_divergent_shell_export_never_overwrites_the_pinned_secret(tmp_path: Path) -> None:
    """An exported secret that disagrees with ``.env`` is warned about, not written in.

    A minted secret is pinned by the volume that was initialized with it and by
    every container already trusting it, so the copy in ``.env`` is the one that
    can still open those volumes. Minting is append-only and skips any variable
    that already has a value, whether the value came from the file or from the
    shell — so an operator who exports a different value gets a deployment whose
    secret store is left exactly as it was, and the pinned value stays
    recoverable rather than being silently replaced by one no volume accepts.

    But compose gives the *shell* precedence over ``--env-file`` when it
    substitutes the compose document, so the export is not inert: it reaches the
    containers. The stack then runs on a credential its volumes never adopted —
    the store still answers to the value in ``.env`` and to nothing else, which
    is asserted below rather than assumed. That is the divergence the deploy has
    to *say* out loud, by variable name, before the operator meets it as an
    unexplained authentication failure days later.

    Neither value may reach the output either. A deploy prints where secrets
    went, which variables it wrote, and which ones the shell is shadowing —
    never what any of them holds.
    """
    project_name = PROJECT_DIVERGENT  # own identity; see the module docstring
    ports = PORTS_DIVERGENT
    divergent = "d1vergentexportvaluenotfromthisrepo"

    try:
        repo = _init_and_build(tmp_path, project_name, "hello-world", _services_override(ports))
        env_path = repo / ".env"

        up1 = _run_osprey(["up", "-d"], repo)
        assert up1.returncode == 0, _fmt("osprey up (baseline)", up1)
        minted = _env_of(env_path)
        pinned = minted[DB_PASSWORD_VAR]
        assert pinned != divergent
        env_before = env_path.read_text(encoding="utf-8")

        # The volumes exist and answer to the pinned value BEFORE the divergent
        # export, so everything asserted after it is about the export rather
        # than about a store that was never initialized in the first place.
        _assert_stores_authenticate(project_name, ports, minted)

        up2 = _run_osprey(["up", "-d"], repo, extra_env={DB_PASSWORD_VAR: divergent})
        assert up2.returncode == 0, _fmt("osprey up (divergent export)", up2)

        output = _squash(up2)

        # ANTI-VACUITY, and the point of the whole test. The two `.env`
        # assertions below are true by construction if the export never got far
        # enough to matter: minting skips a variable that already has a value,
        # so "the file did not change" is what a completely ignored export looks
        # like too. The preflight warning is the evidence that the deploy SAW
        # the divergence — it fires only when a compose file interpolates the
        # name, the store pins it, and the shell disagrees, which is precisely
        # the state this test set up. Without it, a regression that dropped the
        # export on the floor would leave every other assertion here green.
        assert _needle(DB_PASSWORD_VAR) in output, (
            f"the deploy never named {DB_PASSWORD_VAR} as shadowed by the shell — either "
            "the divergence preflight is gone, or the export did not reach the deploy "
            "at all, in which case the .env assertions below prove nothing:\n"
            f"{up2.stdout}\n{up2.stderr}"
        )
        # The CONSEQUENCE, not the warning's headline: which side wins is now
        # provider-specific (podman-compose reads its env file first and ignores
        # the export, inverting Docker's answer), so the sentence that names the
        # winner is the one an operator needs and the one worth pinning. This
        # lane is Docker.
        assert _needle("substitutestheEXPORTEDvalue") in output, (
            "the deploy named the variable but not what is happening to it; an "
            "operator has to be told the shell's value is the one compose will "
            f"substitute:\n{up2.stdout}\n{up2.stderr}"
        )

        assert env_path.read_text(encoding="utf-8") == env_before, (
            "the divergent export rewrote the deployment's .env — the value the "
            "running stack's volume was initialized with is now unrecoverable"
        )
        assert _env_of(env_path)[DB_PASSWORD_VAR] == pinned, (
            f"{DB_PASSWORD_VAR} in {env_path} is no longer the value the volumes were "
            "initialized with, so the next deploy from this repo would be locked out"
        )

        # The store is still the pinned value's, and the exported one opens
        # nothing: the warning above describes a real hazard rather than a
        # theoretical one. `_assert_stores_authenticate` re-waits for postgres,
        # which is what covers the container compose recreated on the changed
        # interpolation.
        _assert_stores_authenticate(project_name, ports, minted)
        rejected = _psql(project_name, divergent, "select 1")
        assert rejected.returncode != 0, (
            "postgres accepted the exported value — the divergence would be harmless, "
            "and this test would not be about anything"
        )
        assert "password authentication failed" in rejected.stderr, (
            f"postgres refused the exported value for the wrong reason:\n{rejected.stderr}"
        )

        assert divergent not in output, (
            "the deploy printed the exported secret VALUE; it may name a variable, "
            "never what either side holds"
        )
        assert pinned not in output, "the deploy printed the value in its own .env"

    finally:
        _teardown_project(project_name)


# =============================================================================
# (c) Persona stack
# =============================================================================


def test_personas_are_rendered_by_the_build_from_this_repos_own_deltas(tmp_path: Path) -> None:
    """Personas are rendered by the BUILD, from the repo's own ``personas/``.

    ``control-assistant`` is the shipped two-persona stack. ``osprey init``
    writes a delta per persona into ``<repo>/personas/`` and rewrites the
    catalog's ``build_profile`` to point at those files; ``osprey build`` then
    renders each persona project into ``build/`` beside the deployment's own
    render. That anchoring is the whole feature: a delta merges over the profile
    the deployment was built from, which is what gives every persona the host's
    data tree and its convention artifacts.

    Rendering at build time is what makes ``build/`` a complete account of what
    a start will run — a start renders nothing and refuses when a referenced
    persona has no project — so this drives ``init`` and ``build`` only. No
    container is created and no persona IMAGE is built: those take tens of
    minutes and prove nothing about profile resolution.

    A persona's ``.env`` is deliberately NOT asserted against the deployment's
    minted secrets. Minting happens at ``up``, after the render, so a persona's
    copy could only ever hold what existed when the build ran; the secrets that
    reach a persona container come from the repo ``.env`` the deploy mounts,
    which the rebuild case above is the proof of.
    """
    project_name = PROJECT_PERSONA
    web = PORTS_PERSONA_WEB
    override = (
        "config:\n"
        f"  services.postgresql.port_host: {PORTS_PERSONA['postgres']}\n"
        f"  services.openobserve.port: {PORTS_PERSONA['openobserve']}\n"
        f"  modules.web_terminals.nginx_port: {web['nginx']}\n"
        f"  modules.web_terminals.web_base_port: {web['web']}\n"
        f"  modules.web_terminals.artifact_base_port: {web['artifact']}\n"
        f"  modules.web_terminals.ariel_base_port: {web['ariel']}\n"
        f"  modules.web_terminals.lattice_base_port: {web['lattice']}\n"
        f"  modules.web_terminals.channel_finder_base_port: {web['channel_finder']}\n"
    )

    try:
        # The repo root IS the profile root: profile.yml, personas/ and the
        # deltas the catalog points at all live here.
        repo = _init_and_build(tmp_path, project_name, "control-assistant", override)

        catalog = _persona_catalog(repo)
        assert catalog, "the control-assistant build produced no persona catalog"

        for persona, entry in catalog.items():
            delta = repo / "personas" / f"{persona}.yml"
            assert entry.get("build_profile") == f"personas/{persona}.yml", (
                f"persona {persona!r} does not point at a delta in this repo: "
                f"{entry.get('build_profile')!r}"
            )
            assert delta.is_file(), f"osprey init did not write {delta}"

            # The catalog's project_path is repo-relative and lands inside the
            # render, which is what makes a persona disposable with the rest of
            # build/ and what a start verifies before it builds any image.
            persona_dir = (repo / entry["project_path"]).resolve()
            assert persona_dir.is_relative_to(repo / "build"), (
                f"persona {persona!r} was rendered at {persona_dir}, outside build/ — "
                "a persona project is build output and must not outlive a rebuild"
            )
            assert (persona_dir / "config.yml").is_file(), (
                f"persona {persona!r} was not rendered at {persona_dir}"
            )
            assert (persona_dir / "Dockerfile").is_file(), (
                f"persona {persona!r} render has no Dockerfile, so a start would "
                f"refuse it as partial: {persona_dir}"
            )

            manifest = json.loads(
                (persona_dir / ".osprey-manifest.json").read_text(encoding="utf-8")
            )
            rendered_from = Path(manifest["build_args"]["profile_path_abs"]).resolve()
            assert rendered_from == delta.resolve(), (
                f"persona {persona!r} was rendered from {rendered_from}, not from the "
                "delta in this deployment's own repo — it would carry a different "
                "facility's data, conventions and secrets"
            )

            # No persona materializes a source zone of its own: there is exactly
            # one profile in a deployment repo, and it is the one at the root.
            assert not (persona_dir / "profile.yml").exists(), (
                f"persona {persona!r} materialized its own profile at {persona_dir}"
            )

    finally:
        # Nothing is started here — the exact-named sweep runs anyway, in case a
        # future change to the build ever reaches the runtime.
        _teardown_project(project_name)
