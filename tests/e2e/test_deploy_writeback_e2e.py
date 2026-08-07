"""Real-container e2e for deploy-time secret write-back into the owning profile.

``tests/deployment/test_service_tokens_writeback.py`` pins the write-back rules
against hand-built directory trees and a directly-called
``_ensure_service_tokens``. What it cannot show is the property the feature
exists for: that a facility can **wipe and rebuild its project** and the stack
comes back up on the secrets its *already-initialized docker volumes* were
created with. That claim only holds if the whole chain holds at once — ``osprey
build`` materializing a profile, ``osprey deploy up`` minting into that
profile's ``.env``, the containers adopting those values, ``osprey build
--force`` re-deriving the project ``.env`` from the profile, and the redeployed
containers still authenticating against the *same* volumes. Every link is a
different module, and a break in any one of them is invisible to a unit test
that owns only its own link.

So this file drives the real CLI against a real container runtime, over the
topologies the feature has to survive:

* **profile present** (:func:`test_profile_topology_survives_a_forced_rebuild`)
  — the intended shape. A ``--preset`` build materializes
  ``<project>-profile/``; ``deploy up`` mints ``ZO_ROOT_USER_PASSWORD`` and
  ``ARIEL_DB_PASSWORD`` into that profile's ``.env`` under its own section, the
  project derives them, and OpenObserve and Postgres both actually
  authenticate with them. The project ``.env`` is then deleted outright (an
  operator wipe, or a fresh clone of a project whose ``.env`` was never
  committed), ``osprey build --force`` re-renders, the containers are dropped
  while their volumes are kept, and the redeploy has to bring *fresh*
  containers up on the *identical* secrets against the *same* volumes — proven
  by a row written into Postgres before the rebuild still being readable after
  it. These are exactly the two keys that would otherwise degrade the topology
  silently: both compose templates carry an insecure ``${VAR:-default}``, and
  both stores read their password only when *initializing* a fresh volume, so a
  re-minted secret does not fail loudly — it locks the facility out of a store
  that is still happily running on the old one.

* **profile present, contradicted**
  (:func:`test_divergent_shell_export_warns_and_never_overwrites_the_profile`)
  — the same topology with an exported secret that disagrees with the profile.
  Append-only means the profile's copy wins: the disagreement is reported by
  variable name (never by value), the profile file is left byte-identical, the
  project ``.env`` comes back on the profile's value, and the manifest flag
  drops to ``false``.

* **profile absent** (:func:`test_degraded_topology_deploys_and_flags_itself`)
  — a project tree carried somewhere its profile did not follow. The deploy
  must still work: mint into the project ``.env``, warn while naming the
  profile path it could not write, and stamp ``secrets_synced_to_profile:
  false`` so a later ``build --force`` knows the project holds the only copy.

* **persona stack** (:func:`test_persona_autorender_resolves_from_the_profile`)
  — ``deploy up`` auto-renders each persona from a delta in the *parent
  profile's* ``personas/`` directory, which is what gives every persona the
  host's data tree and the host's secrets. The proof is that the freshly
  minted deploy secrets appear in each rendered persona's own ``.env``, and
  that no persona materialized a profile directory of its own.

----------------------------------------------------------------------------
CONTAINER-OPS SAFETY (every runtime-mutating call in this file honors this)
----------------------------------------------------------------------------
Every container and volume here is exact-named off a ``project_name`` this file
owns (``osprey-e2e-wb-*``), so nothing it removes can belong to another stack:

  * containers: ``osprey-e2e-wb-<case>-openobserve`` /
    ``osprey-e2e-wb-<case>-ariel-postgres``
  * volumes: ``osprey-e2e-wb-<case>_openobserve_data`` /
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

from osprey.cli.templates.manifest import read_secrets_synced_to_profile
from osprey.utils.dotenv import DEPLOY_MINTED_BANNER, parse_dotenv_text

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
# Identity. One project_name per case so the three tests can run in any order —
# or concurrently with anything else on the host — without sharing a container
# name, a volume namespace, or a host port.
#
# Ports sit in a band disjoint from every other e2e file's (test_deploy_lifecycle
# spans 19081-20601, test_dispatch_deploy publishes 8020) AND from the ports a
# developer's own demo stack conventionally holds (5064/5080/5432).
# ---------------------------------------------------------------------------
PROJECT_PROFILE = "osprey-e2e-wb-profile"
PROJECT_DIVERGENT = "osprey-e2e-wb-divergent"
PROJECT_ORPHAN = "osprey-e2e-wb-orphan"
PROJECT_PERSONA = "osprey-e2e-wb-persona"

PORTS_PROFILE = {"postgres": 21432, "openobserve": 21482}
PORTS_DIVERGENT = {"postgres": 21433, "openobserve": 21483}
PORTS_ORPHAN = {"postgres": 21434, "openobserve": 21484}
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
# which is precisely why the profile has to carry them across a rebuild.
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
# Project construction
# ---------------------------------------------------------------------------


def _services_override(ports: dict[str, int]) -> str:
    """A ``-O`` overlay adding a Postgres service beside the preset's OpenObserve.

    Two services, deliberately: they are the two ``_SERVICE_TOKEN_VARS`` entries
    whose secret initializes a docker volume, and covering both proves the
    write-back is not special-casing one recipe (``ARIEL_DB_PASSWORD`` is hex,
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


def _build_from_preset(
    output_dir: Path, project_name: str, preset: str, override_text: str
) -> Path:
    """``osprey build <name> --preset <preset> -O <overlay>``; returns the project dir.

    ``--skip-deps``/``--skip-lifecycle`` keep the render network-free and quick:
    nothing here runs the project's own venv, only its compose stack.
    """
    override_path = output_dir / f"{project_name}-override.yml"
    override_path.write_text(override_text, encoding="utf-8")
    result = _run_osprey(
        [
            "build",
            project_name,
            "--preset",
            preset,
            "--override",
            str(override_path),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(output_dir),
        ],
        cwd=output_dir,
        timeout=BUILD_TIMEOUT_SEC,
    )
    assert result.returncode == 0, _fmt(f"osprey build {project_name}", result)
    project_dir = output_dir / project_name
    assert project_dir.is_dir(), f"osprey build did not create {project_dir}"
    return project_dir


def _profile_dir(output_dir: Path, project_name: str) -> Path:
    """Where a ``--preset`` build materializes the profile it then builds from."""
    return output_dir / f"{project_name}-profile"


def _env_of(path: Path) -> dict[str, str]:
    return parse_dotenv_text(path.read_text(encoding="utf-8"))


def _persona_catalog(project_dir: Path) -> dict[str, dict]:
    """``modules.web_terminals.personas`` as rendered into the project config."""
    config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
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
# (a) Profile-present topology
# =============================================================================


def test_profile_topology_survives_a_forced_rebuild(tmp_path: Path) -> None:
    """Mint into the profile, wipe the project, rebuild, redeploy on the same volumes.

    The sequence, and what each step is there to prove:

    1. ``osprey build --preset`` materializes ``<project>-profile/``. Its
       ``.env`` does not exist yet — the deploy is what brings these secrets
       into existence, so a profile that already carried them would prove
       nothing.
    2. ``osprey deploy up`` mints both secrets into the *profile* ``.env``
       under the deploy section, the project ``.env`` carries the same values,
       the manifest records the sync, and both containers authenticate with
       them (and reject a wrong password).
    3. A row is written into Postgres — the witness that step 5 is talking to
       the same volume, not a fresh one that happens to accept the password.
    4. The project ``.env`` is deleted and ``osprey build --force`` re-renders.
       The overlay is deliberately NOT passed again: everything it configured
       lives in the profile now, so a project that comes back with the same
       ports and the same secrets is the profile being the source of truth.
    5. ``osprey deploy down`` removes the containers and keeps both volumes, so
       what follows cannot be the step-2 processes still running on the secret
       they were started with.
    6. ``osprey deploy up`` again: FRESH containers, the same volumes, the same
       secrets, and the row from step 3 still readable.
    """
    project_name = PROJECT_PROFILE
    ports = PORTS_PROFILE
    profile_dir = _profile_dir(tmp_path, project_name)
    profile_env = profile_dir / ".env"

    try:
        project_dir = _build_from_preset(
            tmp_path, project_name, "hello-world", _services_override(ports)
        )
        assert (profile_dir / "profile.yml").is_file(), (
            f"--preset build did not materialize a profile at {profile_dir}"
        )
        assert not profile_env.exists(), (
            "the profile already carries a .env before any deploy — this test cannot "
            "then show that the deploy is what put the secrets there"
        )

        # -- 2. first deploy: mint -> profile -> project -> containers --------
        up1 = _run_osprey(["deploy", "up", "-d"], project_dir)
        assert up1.returncode == 0, _fmt("osprey deploy up (first)", up1)

        assert profile_env.is_file(), (
            f"deploy did not create the profile .env at {profile_env}:\n{up1.stdout}"
        )
        assert profile_env.stat().st_mode & 0o777 == 0o600, (
            "the profile .env holds facility secrets and must be created private"
        )
        profile_text = profile_env.read_text(encoding="utf-8")
        assert DEPLOY_MINTED_BANNER in profile_text, (
            f"minted secrets landed in the profile .env without their own section:\n{profile_text}"
        )

        minted = _env_of(profile_env)
        for var in (ZO_PASSWORD_VAR, DB_PASSWORD_VAR):
            assert minted.get(var), f"{var} was not persisted to the profile .env"
        project_env = _env_of(project_dir / ".env")
        assert {var: project_env.get(var) for var in (ZO_PASSWORD_VAR, DB_PASSWORD_VAR)} == {
            var: minted[var] for var in (ZO_PASSWORD_VAR, DB_PASSWORD_VAR)
        }, "the project .env and the profile .env disagree on the freshly minted secrets"
        assert read_secrets_synced_to_profile(project_dir) is True

        _assert_stores_authenticate(project_name, ports, minted)

        # -- 3. a witness row, so step 5 proves SAME VOLUME, not just same password
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

        # -- 4. wipe the project .env, then rebuild from the profile alone ----
        (project_dir / ".env").unlink()
        rebuild = _run_osprey(
            [
                "build",
                project_name,
                "--preset",
                "hello-world",
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
                "--force",
            ],
            cwd=tmp_path,
            timeout=BUILD_TIMEOUT_SEC,
        )
        assert rebuild.returncode == 0, _fmt("osprey build --force", rebuild)

        rebuilt_env = _env_of(project_dir / ".env")
        for var in (ZO_PASSWORD_VAR, DB_PASSWORD_VAR):
            assert rebuilt_env.get(var) == minted[var], (
                f"{var} did not survive the rebuild: the project .env came back with a "
                "different value, so the redeployed stack would be locked out of the "
                "volume it initialized"
            )

        # -- 5. drop the containers, KEEP the volumes -------------------------
        # Without this the redeploy below can be a no-op: compose recreates a
        # container only when its definition changed, and after a rebuild that
        # re-derived the same values nothing has. The step-2 processes would
        # still be running, still holding the secret they were STARTED with,
        # and the authentication assertion in step 6 would re-confirm them
        # rather than prove that a fresh container adopted the re-derived
        # value. `osprey deploy down` is compose `down` with no ``-v``, so it
        # removes this project's containers and network and leaves both named
        # volumes — exactly the "same volumes, new containers" state the claim
        # needs.
        down = _run_osprey(["deploy", "down"], project_dir)
        assert down.returncode == 0, _fmt("osprey deploy down (between rebuild and redeploy)", down)
        assert all(_volume_exists(volume) for volume in volumes), (
            "`osprey deploy down` removed a named volume — it must tear down "
            "containers only, or the redeploy below would meet a fresh store and "
            "prove nothing about the secret surviving"
        )
        for container in (_pg_container(project_name), _openobserve_container(project_name)):
            assert not _container_exists(container), (
                f"{container} survived `osprey deploy down`, so the redeploy below "
                "would reuse the process started in step 2 and the authentication "
                "assertion would say nothing about the re-derived secret"
            )

        # -- 6. redeploy: same volumes, new containers, same secrets ----------
        up2 = _run_osprey(["deploy", "up", "-d"], project_dir)
        assert up2.returncode == 0, _fmt("osprey deploy up (after rebuild)", up2)
        assert all(_volume_exists(volume) for volume in volumes), (
            "the redeploy did not run against the volumes the first deploy created"
        )
        assert read_secrets_synced_to_profile(project_dir) is True

        _assert_stores_authenticate(project_name, ports, minted)

        witness = _psql(project_name, minted[DB_PASSWORD_VAR], "select note from wb_witness")
        assert witness.returncode == 0, _fmt("read the witness row after the rebuild", witness)
        assert witness.stdout.strip() == "pre-rebuild", (
            "the witness row is gone — the redeploy came up against a fresh volume, so "
            "'the secret survived' would have been vacuously true"
        )

    finally:
        _teardown_project(project_name)


def test_divergent_shell_export_warns_and_never_overwrites_the_profile(tmp_path: Path) -> None:
    """An exported secret that disagrees with the profile is reported, not written.

    A minted secret is pinned by the volume that was initialized with it and by
    every container already trusting it, so the profile's copy is authoritative
    and the write-back is append-only. What a disagreement earns is a warning
    naming the *variable* — never either value — and
    ``secrets_synced_to_profile: false``, because the deploy is now running on
    something the profile does not account for.

    The export only becomes the effective value once the key is absent from the
    project ``.env``: the config loader reads that file over ``os.environ``
    (``load_dotenv(..., override=True)``), so with the key present the export
    never reaches the write-back at all. Removing it first is what makes this
    the divergence the test is named for rather than a no-op.
    """
    project_name = PROJECT_DIVERGENT  # own identity; see the module docstring
    ports = PORTS_DIVERGENT
    profile_env = _profile_dir(tmp_path, project_name) / ".env"
    divergent = "d1vergentexportvaluenotfromthisprofile"

    try:
        project_dir = _build_from_preset(
            tmp_path, project_name, "hello-world", _services_override(ports)
        )
        up1 = _run_osprey(["deploy", "up", "-d"], project_dir)
        assert up1.returncode == 0, _fmt("osprey deploy up (baseline)", up1)
        pinned = _env_of(profile_env)[DB_PASSWORD_VAR]
        assert pinned != divergent

        project_env_path = project_dir / ".env"
        kept = [
            line
            for line in project_env_path.read_text(encoding="utf-8").splitlines(keepends=True)
            if not line.startswith(f"{DB_PASSWORD_VAR}=")
        ]
        project_env_path.write_text("".join(kept), encoding="utf-8")

        profile_before = profile_env.read_text(encoding="utf-8")
        up2 = _run_osprey(
            ["deploy", "up", "-d"], project_dir, extra_env={DB_PASSWORD_VAR: divergent}
        )
        assert up2.returncode == 0, _fmt("osprey deploy up (divergent export)", up2)

        assert profile_env.read_text(encoding="utf-8") == profile_before, (
            "the divergent export overwrote the profile .env — the value the running "
            "stack's volume was initialized with is now unrecoverable"
        )
        output = _squash(up2)
        assert _needle(f"{DB_PASSWORD_VAR} differs") in output, (
            f"the conflict was not reported by variable name:\n{up2.stdout}\n{up2.stderr}"
        )
        assert divergent not in output, (
            "the deploy printed the conflicting secret VALUE; a warning names the "
            "variable, never what either side holds"
        )
        assert pinned not in output, "the deploy printed the profile's secret value"
        assert read_secrets_synced_to_profile(project_dir) is False, (
            "the project is running on a secret the profile does not carry, so the "
            "manifest must not claim the two are in sync"
        )

        # "The profile was not overwritten" is only half the claim: the profile
        # also has to WIN. The deploy-mode derivation runs after the conflict is
        # reported, and the key it re-derives here is the one the export tried to
        # displace — so the project .env must come back holding the profile's
        # value, not the exported one. Without this a future change that quietly
        # let the export through would still pass every assertion above.
        assert _env_of(project_dir / ".env")[DB_PASSWORD_VAR] == pinned, (
            "the project .env did not come back on the profile's value — the "
            "divergent export won the round trip, so the next rebuild would carry "
            "a secret the profile cannot reproduce"
        )

    finally:
        _teardown_project(project_name)


# =============================================================================
# (b) Profile-absent topology
# =============================================================================


def test_degraded_topology_deploys_and_flags_itself(tmp_path: Path) -> None:
    """A project tree carried away from its profile still deploys — loudly.

    The project is built in one directory and then copied to another, and the
    original (profile included) is removed entirely: the manifest's
    ``profile_path_abs`` now names a path that does not exist, which is exactly
    what an operator sees after moving a project to another host, or cloning a
    repo whose profile lives elsewhere. The deploy must not fail over it — but
    it must also not pretend: the warning names the profile ``.env`` it could
    not write, and the manifest flag records that the project ``.env`` is the
    only copy of these secrets.
    """
    project_name = PROJECT_ORPHAN
    ports = PORTS_ORPHAN
    origin = tmp_path / "origin"
    elsewhere = tmp_path / "elsewhere"
    origin.mkdir()
    elsewhere.mkdir()

    try:
        built = _build_from_preset(origin, project_name, "hello-world", _services_override(ports))
        orphaned_profile_env = _profile_dir(origin, project_name) / ".env"

        project_dir = elsewhere / project_name
        shutil.copytree(built, project_dir)
        shutil.rmtree(origin)
        assert not orphaned_profile_env.parent.exists()

        up = _run_osprey(["deploy", "up", "-d"], project_dir)
        assert up.returncode == 0, _fmt("osprey deploy up (no reachable profile)", up)

        output = _squash(up)
        assert _needle(str(orphaned_profile_env)) in output, (
            "the degraded deploy did not name the profile .env it could not write, so "
            f"an operator cannot tell where the secrets were supposed to go:\n{up.stdout}"
        )
        assert read_secrets_synced_to_profile(project_dir) is False, (
            "the project .env now holds the only copy of these secrets; the manifest "
            "flag is what warns `build --force` before it wipes them"
        )

        minted = _env_of(project_dir / ".env")
        for var in (ZO_PASSWORD_VAR, DB_PASSWORD_VAR):
            assert minted.get(var), f"{var} was not minted into the project .env"
        _assert_stores_authenticate(project_name, ports, minted)

    finally:
        _teardown_project(project_name)


# =============================================================================
# (c) Persona stack
# =============================================================================


def test_persona_autorender_resolves_from_the_profile(tmp_path: Path) -> None:
    """Personas are rendered from the parent profile's ``personas/`` — and only there.

    ``multi-user-demo`` is the shipped two-persona stack. Materializing its
    profile writes a delta per persona into ``<profile>/personas/`` and rewrites
    the catalog's ``build_profile`` to point at those files; ``deploy up`` then
    auto-renders each persona project from its delta. That anchoring is the
    whole feature: a delta merges over the profile the deployed project was
    built from, which is what gives every persona the host's data tree, its
    convention artifacts, and its secrets.

    The deploy is driven only as far as its web-terminal preflight. Personas
    auto-render there, and the fail-closed credential gate immediately after
    stops the run — before the persona image builds, which take tens of minutes
    and prove nothing about profile resolution. The stop point is asserted
    explicitly, so a future reordering that moved the render behind the gate
    fails this test instead of silently emptying it.
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
    profile_dir = _profile_dir(tmp_path, project_name)

    try:
        project_dir = _build_from_preset(tmp_path, project_name, "multi-user-demo", override)

        catalog = _persona_catalog(project_dir)
        assert catalog, "the multi-user-demo build produced no persona catalog"
        for persona, entry in catalog.items():
            delta = profile_dir / "personas" / f"{persona}.yml"
            assert entry.get("build_profile") == f"personas/{persona}.yml", (
                f"persona {persona!r} does not point at a delta in the profile: "
                f"{entry.get('build_profile')!r}"
            )
            assert delta.is_file(), f"profile materialization did not write {delta}"

        # The deploy mints the stack's secrets into the profile .env first, so
        # every persona rendered a moment later derives the SAME values — the
        # observable consequence of "a delta anchors at the profile root".
        up = _run_osprey(["deploy", "up", "-d"], project_dir)
        assert up.returncode != 0, (
            "the persona deploy was expected to stop at its fail-closed credential "
            "gate (no provider key is set anywhere), but it went on to build persona "
            f"images:\n{up.stdout}\n{up.stderr}"
        )
        output = _squash(up)
        assert _needle("would leave web terminals unauthenticated") in output, (
            "the deploy did not stop where this test expects (the credential gate "
            f"immediately after the persona auto-render):\n{up.stdout}\n{up.stderr}"
        )

        minted = _env_of(profile_dir / ".env")
        for var in (ZO_PASSWORD_VAR, DB_PASSWORD_VAR):
            assert minted.get(var), f"{var} was not minted into the profile .env"

        for persona, entry in catalog.items():
            persona_dir = (project_dir / entry["project_path"]).resolve()
            assert (persona_dir / "config.yml").is_file(), (
                f"persona {persona!r} was not auto-rendered at {persona_dir}"
            )

            manifest = json.loads(
                (persona_dir / ".osprey-manifest.json").read_text(encoding="utf-8")
            )
            rendered_from = Path(manifest["build_args"]["profile_path_abs"]).resolve()
            assert rendered_from == (profile_dir / "personas" / f"{persona}.yml").resolve(), (
                f"persona {persona!r} was rendered from {rendered_from}, not from the "
                "delta in the deployed project's own profile — it would carry a "
                "different facility's data, conventions and secrets"
            )

            persona_env = _env_of(persona_dir / ".env")
            for var in (ZO_PASSWORD_VAR, DB_PASSWORD_VAR):
                assert persona_env.get(var) == minted[var], (
                    f"persona {persona!r} did not inherit {var} from the parent "
                    "profile, so its containers would not trust the host's stores"
                )

            orphan_profile = persona_dir.parent / f"{persona_dir.name}-profile"
            assert not orphan_profile.exists(), (
                f"auto-rendering persona {persona!r} materialized a profile of its own "
                f"at {orphan_profile}; a persona is a delta over its host's profile and "
                "must never own one"
            )

    finally:
        # The preflight aborts before any compose invocation, so nothing should
        # exist — the exact-named sweep runs anyway, in case it ever does.
        _teardown_project(project_name)
