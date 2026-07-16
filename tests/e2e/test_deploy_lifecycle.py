"""Real-container e2e for the multi-user web-terminal deploy lifecycle.

The shipped ``hello-world`` deploy-e2e fixture (``tests/e2e/test_dispatch_deploy.py``
covers the dispatch stack; the CI ``deploy-e2e`` job covers a services-only
``hello-world`` project) declares NO ``modules.web_terminals.users`` and so can
never reach ``osprey deploy``'s per-user verbs (``decommission``/``prune``/
``nuke``/``seed``) or the idempotent web-terminal ``up`` reconcile. This test
drives all of them against a REAL Docker daemon, using
``tests/e2e/fixtures/multi_user_config.yml`` — a facility config whose
``project_name``/``facility.prefix`` make every resource it produces exact-named
and obviously throwaway.

It proves LIFECYCLE MECHANICS (containers/volumes get created, reconciled,
seeded, retained/purged/archived, pruned, and torn down correctly) — not the
real terminal application. The per-user image
(``tests/e2e/fixtures/Dockerfile.web_terminal_stub``) is a tiny alpine stand-in
that just stays running (``tail -f /dev/null``) and has what seeding.py's
container-side reconcile scripts need: a ``dispatch`` user and the two mount-point
directories the compose template declares volumes onto. It is pushed into a
throwaway local registry container this test starts, so ``docker compose pull``
(which ``deploy up``'s web-terminal reconcile always runs — see
``container_lifecycle._deploy_up_web_terminals``) genuinely succeeds against a
locally-resolvable image instead of failing with "pull access denied" against a
registry that doesn't exist.

----------------------------------------------------------------------------
CONTAINER-OPS SAFETY (every runtime-mutating call in this file honors this)
----------------------------------------------------------------------------
Every container/volume this test creates is exact-named off the fixture's
``project_name: osprey-e2e-mus-p3`` / ``facility.prefix: e2e``:

  * containers: ``e2e-web-<alice|bob|carol|dave|erin>``, ``e2e-nginx``,
    plus the test's own ``osprey-e2e-mus-p3-registry`` helper container
    (not part of the deployed stack — a throwaway local image registry).
  * volumes: ``osprey-e2e-mus-p3_<user>-claude-config`` /
    ``osprey-e2e-mus-p3_<user>-agent-data``.

Nothing here ever runs ``system prune``, ``volume prune``, ``container prune``,
``-a``/``--all`` on a removal, or a wildcard/glob container-name match — every
teardown call (both inline, as the lifecycle verbs under test are exercised, and
in the module-level/test-level ``finally`` safety nets) names an exact resource,
or uses ``docker compose -p osprey-e2e-mus-p3 down`` (project-label-scoped,
containers/networks only, never ``-v``) — the same pattern
``osprey.deployment.web_terminals.lifecycle`` uses internally. Teardown runs from
a ``finally`` block so a failed assertion mid-sequence still cleans up.

Gating: needs Docker (daemon actually running, not just the CLI installed).
Skipped entirely if unavailable — see the ``stub_image`` fixture.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from importlib.resources import as_file, files
from pathlib import Path

import pytest
import yaml

from osprey.deployment.compose_generator import resolve_user_volume_names
from osprey.deployment.web_terminals.ports import normalize_users
from osprey.utils import config_writer

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.dockerbuild]

# ---------------------------------------------------------------------------
# Fixture identity — kept in sync with tests/e2e/fixtures/multi_user_config.yml
# and Dockerfile.web_terminal_stub. See that YAML's header comment for the
# exact-naming rationale.
# ---------------------------------------------------------------------------
PROJECT_NAME = "osprey-e2e-mus-p3"
FACILITY_PREFIX = "e2e"

# Users beyond the fixture's own committed [alice, bob] — extended onto the
# roster at test setup (via config_writer.config_replace_list, the same
# helper decommission_user itself uses) so one run can exercise every
# decommission disposal (retain/purge/archive) plus a prune orphan without
# needing multiple deploy cycles.
USERS = ("alice", "bob", "carol", "dave", "erin")

REGISTRY_CONTAINER = "osprey-e2e-mus-p3-registry"
REGISTRY_PORT = 19081
REGISTRY_HOST = f"localhost:{REGISTRY_PORT}"
STUB_IMAGE = f"{REGISTRY_HOST}/web-terminal:latest"
LOCAL_BUILD_TAG = "osprey-e2e-mus-p3-stub:build"

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CONFIG_FIXTURE = FIXTURES_DIR / "multi_user_config.yml"
STUB_DOCKERFILE = FIXTURES_DIR / "Dockerfile.web_terminal_stub"

_BASE_MD_MARKER = "osprey-e2e-mus-p3 fixture CLAUDE.md base"
_BASE_MD_CONTENT = f"# {_BASE_MD_MARKER}\n\nSeeded by tests/e2e/test_deploy_lifecycle.py.\n"

DEPLOY_UP_TIMEOUT_SEC = 180
DEPLOY_VERB_TIMEOUT_SEC = 60
REGISTRY_READY_TIMEOUT_SEC = 30.0


def _web_container(user: str) -> str:
    return f"{FACILITY_PREFIX}-web-{user}"


def _nginx_container() -> str:
    return f"{FACILITY_PREFIX}-nginx"


def _volume_names(user: str) -> tuple[str, str]:
    return resolve_user_volume_names({"project_name": PROJECT_NAME}, user)


# ---------------------------------------------------------------------------
# Low-level docker/CLI helpers
# ---------------------------------------------------------------------------


def _find_osprey_console_script() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _docker(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def _run_osprey(
    osprey_bin: Path, args: list[str], cwd: Path, timeout: int = DEPLOY_VERB_TIMEOUT_SEC
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(osprey_bin), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": ""},
    )


def _fmt(label: str, result: subprocess.CompletedProcess) -> str:
    return (
        f"{label} failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def _container_id(name: str) -> str | None:
    result = _docker("inspect", "--type", "container", "-f", "{{.Id}}", name, timeout=15)
    return result.stdout.strip() if result.returncode == 0 else None


def _volume_exists(name: str) -> bool:
    return _docker("volume", "inspect", name, timeout=15).returncode == 0


def _container_env_var(name: str, var: str) -> str | None:
    result = _docker("exec", name, "sh", "-c", f"printenv {var}", timeout=15)
    return result.stdout.strip() if result.returncode == 0 else None


def _compose_project_containers(project: str) -> list[str]:
    """Read-only: containers still labeled as belonging to *project* (any state)."""
    result = _docker(
        "ps",
        "-a",
        "--filter",
        f"label=com.docker.compose.project={project}",
        "--format",
        "{{.Names}}",
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _roster(config_path: Path) -> list[dict]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    users_raw = data.get("modules", {}).get("web_terminals", {}).get("users")
    return normalize_users(users_raw)


def _roster_names(config_path: Path) -> set[str]:
    return {entry["name"] for entry in _roster(config_path)}


def _roster_indices(config_path: Path, names: tuple[str, ...]) -> tuple[int | None, ...]:
    by_name = {entry["name"]: entry["index"] for entry in _roster(config_path)}
    return tuple(by_name.get(name) for name in names)


def _teardown_all_project_resources() -> None:
    """Best-effort, exact-named sweep of everything this test could have created.

    Runs from a ``finally`` in the main test so a failed assertion mid-sequence
    still leaves nothing stranded. Every call below names one exact container,
    one exact volume, or is the project-scoped ``compose down`` the guardrail
    explicitly allows (label-based, containers/networks only, never ``-v``) —
    never a prune, ``-a``/``--all``, or wildcard. Failures (resource already
    gone, never created, etc.) are swallowed: this is a safety net, not an
    assertion.
    """
    for user in USERS:
        _docker("rm", "-f", _web_container(user))
    _docker("rm", "-f", _nginx_container())
    _docker("compose", "-p", PROJECT_NAME, "down", timeout=60)
    for user in USERS:
        for volume in _volume_names(user):
            _docker("volume", "rm", volume)


def _wait_for_registry(port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_err = "(no attempt yet)"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(  # noqa: S310 - localhost only
                f"http://localhost:{port}/v2/", timeout=3.0
            ) as resp:
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = str(exc)
        time.sleep(1.0)
    raise AssertionError(
        f"local registry on :{port} not ready after {timeout:.0f}s (last: {last_err})"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stub_image() -> Iterator[str]:
    """Build the throwaway stub image and push it into a throwaway local registry.

    Yields the pushed image ref (``localhost:19081/web-terminal:latest``, matching
    ``registry.url`` in multi_user_config.yml). Skips the whole module if Docker
    isn't installed or the daemon isn't actually responding — the graceful,
    non-runtime-environment path the task requires.

    Teardown removes the registry container (exact name) and the local image
    tags this fixture created (exact tags — never ``rmi -a``/a sweep).
    """
    if shutil.which("docker") is None:
        pytest.skip("docker not available")
    if _docker("ps", timeout=10).returncode != 0:
        pytest.skip("docker daemon not responding")

    build = _docker(
        "build", "-f", str(STUB_DOCKERFILE), "-t", LOCAL_BUILD_TAG, str(FIXTURES_DIR), timeout=120
    )
    if build.returncode != 0:
        pytest.fail(_fmt("stub image build", build))

    # Exact-named: remove any stale registry container a previous crashed run
    # left behind before starting a fresh one under the same name.
    _docker("rm", "-f", REGISTRY_CONTAINER)
    run_registry = _docker(
        "run",
        "-d",
        "--name",
        REGISTRY_CONTAINER,
        "-p",
        f"{REGISTRY_PORT}:5000",
        "registry:2",
        timeout=60,
    )
    if run_registry.returncode != 0:
        pytest.fail(_fmt("local registry container start", run_registry))

    try:
        _wait_for_registry(REGISTRY_PORT, REGISTRY_READY_TIMEOUT_SEC)

        tag = _docker("tag", LOCAL_BUILD_TAG, STUB_IMAGE, timeout=15)
        if tag.returncode != 0:
            pytest.fail(_fmt("stub image tag", tag))
        push = _docker("push", STUB_IMAGE, timeout=60)
        if push.returncode != 0:
            pytest.fail(_fmt("stub image push", push))

        # Untag the local convenience copy so `docker compose pull` (which
        # `deploy up`'s web-terminal reconcile always runs) genuinely pulls
        # from the registry rather than short-circuiting on an already-present
        # local tag — this is the exact behavior that makes the fixture prove
        # "locally-resolvable image", not "image docker already happens to have".
        _docker("rmi", STUB_IMAGE)

        yield STUB_IMAGE
    finally:
        _docker("rm", "-f", REGISTRY_CONTAINER)
        _docker("rmi", "-f", STUB_IMAGE)
        _docker("rmi", "-f", LOCAL_BUILD_TAG)


@pytest.fixture
def project_dir(tmp_path: Path, stub_image: str) -> Path:
    """A throwaway project directory seeded from multi_user_config.yml."""
    dest = tmp_path / "project"
    dest.mkdir()
    shutil.copy(CONFIG_FIXTURE, dest / "config.yml")

    context_dir = dest / "docker" / "web-terminal-context"
    context_dir.mkdir(parents=True)
    (context_dir / "base.md").write_text(_BASE_MD_CONTENT, encoding="utf-8")

    # docker-compose.web.yml.j2 mounts this unconditionally
    # (`env_file: .env.production`) — compose fails to even parse the file
    # without it. Empty is fine: this fixture needs no real provider secrets.
    (dest / ".env.production").write_text("", encoding="utf-8")

    # prepare_compose_files() always renders the top-level
    # services/docker-compose.yml.j2 (an empty osprey-network declaration —
    # harmless even for a web-terminals-only deploy) via a bare
    # FileSystemLoader(".") — i.e. relative to the project's own CWD, not a
    # package resource. A real `osprey build` scaffolds this file into every
    # project; this hand-built fixture project has to copy it in itself.
    services_template = files("osprey").joinpath("templates/services/docker-compose.yml.j2")
    with as_file(services_template) as template_path:
        services_dir = dest / "services"
        services_dir.mkdir(parents=True)
        shutil.copy(template_path, services_dir / "docker-compose.yml.j2")

    return dest


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def test_deploy_lifecycle_full_sequence(project_dir: Path) -> None:
    """Drive up -> idempotent up -> seed -> decommission(x3 dispositions) ->
    prune -> nuke against a real Docker daemon, asserting each lifecycle
    guarantee the verbs document.
    """
    osprey_bin = _find_osprey_console_script()
    config_path = project_dir / "config.yml"

    # Extend the committed [alice, bob] roster to the five users this scenario
    # needs, via the same config-editing helper decommission_user itself uses.
    config_writer.config_replace_list(
        config_path, ["modules", "web_terminals", "users"], list(USERS)
    )

    try:
        # --------------------------------------------------------------
        # up: brings up every per-user container + nginx
        # --------------------------------------------------------------
        up1 = _run_osprey(osprey_bin, ["deploy", "up"], project_dir, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up1.returncode == 0, _fmt("deploy up (1st)", up1)

        all_containers = [_web_container(u) for u in USERS] + [_nginx_container()]
        ids_after_first_up = {name: _container_id(name) for name in all_containers}
        missing = [name for name, cid in ids_after_first_up.items() if cid is None]
        assert not missing, f"container(s) not created by 'deploy up': {missing}"

        # --------------------------------------------------------------
        # C2 — idempotency: a 2nd `up` must recreate ZERO containers
        # --------------------------------------------------------------
        up2 = _run_osprey(osprey_bin, ["deploy", "up"], project_dir, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up2.returncode == 0, _fmt("deploy up (2nd)", up2)
        ids_after_second_up = {name: _container_id(name) for name in all_containers}
        assert ids_after_second_up == ids_after_first_up, (
            "idempotent 'deploy up' recreated container(s): "
            f"{[n for n in all_containers if ids_after_second_up[n] != ids_after_first_up[n]]}"
        )

        # --------------------------------------------------------------
        # C3 — seed: CLAUDE.md present; a user-installed (non-managed)
        # skill dir survives a reseed.
        # --------------------------------------------------------------
        claude_md = _docker("exec", _web_container("alice"), "cat", "/data/claude-config/CLAUDE.md")
        assert claude_md.returncode == 0, _fmt("cat alice's CLAUDE.md", claude_md)
        assert _BASE_MD_MARKER in claude_md.stdout

        skill_dir = f"/app/{FACILITY_PREFIX}-assistant/.claude/skills/user-installed"
        make_skill = _docker(
            "exec",
            "-u",
            "0",
            _web_container("alice"),
            "sh",
            "-c",
            f"mkdir -p {skill_dir} && touch {skill_dir}/marker.txt",
        )
        assert make_skill.returncode == 0, _fmt("create user-installed skill dir", make_skill)

        reseed = _run_osprey(osprey_bin, ["deploy", "seed", "alice"], project_dir)
        assert reseed.returncode == 0, _fmt("deploy seed alice", reseed)

        survived = _docker("exec", _web_container("alice"), "test", "-f", f"{skill_dir}/marker.txt")
        assert survived.returncode == 0, "user-installed skill dir did not survive reseed"

        # --------------------------------------------------------------
        # Baseline for the survivor-ports check, captured before touching bob.
        # --------------------------------------------------------------
        carol_port_before = _container_env_var(_web_container("carol"), "OSPREY_WEB_PORT")
        dave_port_before = _container_env_var(_web_container("dave"), "OSPREY_WEB_PORT")
        assert carol_port_before and dave_port_before
        indices_before = _roster_indices(config_path, ("carol", "dave"))

        # --------------------------------------------------------------
        # C5a — decommission (retain, default): container + roster entry
        # gone, BOTH volumes still exist.
        # --------------------------------------------------------------
        bob_claude_vol, bob_agent_vol = _volume_names("bob")
        decomm_bob = _run_osprey(
            osprey_bin, ["deploy", "decommission", "bob", "--yes"], project_dir
        )
        assert decomm_bob.returncode == 0, _fmt("deploy decommission bob", decomm_bob)
        assert _container_id(_web_container("bob")) is None
        assert "bob" not in _roster_names(config_path)
        assert _volume_exists(bob_claude_vol), "retain must not remove the claude-config volume"
        assert _volume_exists(bob_agent_vol), "retain must not remove the agent-data volume"

        # --------------------------------------------------------------
        # C6 — survivor ports: decommissioning bob (mid-list) must not
        # shift carol/dave's frozen indices (and therefore ports).
        # --------------------------------------------------------------
        indices_after = _roster_indices(config_path, ("carol", "dave"))
        assert indices_after == indices_before, (
            f"decommissioning a mid-list user shifted survivor indices: "
            f"before={indices_before} after={indices_after}"
        )
        assert _container_env_var(_web_container("carol"), "OSPREY_WEB_PORT") == carol_port_before
        assert _container_env_var(_web_container("dave"), "OSPREY_WEB_PORT") == dave_port_before

        # --------------------------------------------------------------
        # C5b — decommission --purge: both volumes actually removed.
        # --------------------------------------------------------------
        carol_claude_vol, carol_agent_vol = _volume_names("carol")
        decomm_carol = _run_osprey(
            osprey_bin, ["deploy", "decommission", "carol", "--purge", "--yes"], project_dir
        )
        assert decomm_carol.returncode == 0, _fmt("deploy decommission carol --purge", decomm_carol)
        assert not _volume_exists(carol_claude_vol), "purge must remove the claude-config volume"
        assert not _volume_exists(carol_agent_vol), "purge must remove the agent-data volume"

        # --------------------------------------------------------------
        # C5c — decommission --archive: a readable tarball is written
        # before the volume is removed.
        # --------------------------------------------------------------
        dave_claude_vol, dave_agent_vol = _volume_names("dave")
        decomm_dave = _run_osprey(
            osprey_bin, ["deploy", "decommission", "dave", "--archive", "--yes"], project_dir
        )
        assert decomm_dave.returncode == 0, _fmt("deploy decommission dave --archive", decomm_dave)

        archive_dir = project_dir / "web_terminal_archives"
        claude_tarball = archive_dir / f"{dave_claude_vol}.tar.gz"
        agent_tarball = archive_dir / f"{dave_agent_vol}.tar.gz"
        assert claude_tarball.is_file(), f"expected archive tarball at {claude_tarball}"
        assert agent_tarball.is_file(), f"expected archive tarball at {agent_tarball}"
        assert tarfile.is_tarfile(claude_tarball)
        with tarfile.open(claude_tarball) as tf:
            names = {Path(n).name for n in tf.getnames()}
        assert "CLAUDE.md" in names, f"archived claude-config tarball missing CLAUDE.md: {names}"

        assert not _volume_exists(dave_claude_vol), "archive must remove the volume after archiving"
        assert not _volume_exists(dave_agent_vol), "archive must remove the volume after archiving"

        # --------------------------------------------------------------
        # C7 — prune: simulate config drift (erin hand-removed from the
        # roster WITHOUT running decommission — exactly the scenario
        # prune_users exists for), then prune off-roster resources.
        #
        # bob is ALSO an orphan at this point: his container is already gone
        # (removed by the retain-mode decommission above), but his two
        # volumes are still there by design (retain keeps them) and he is
        # off-roster — prune_users discovers orphans by volume existence
        # independently of container existence, so `--purge` sweeps bob's
        # leftover volumes here too, not just erin's live container+volumes.
        # --------------------------------------------------------------
        remaining = [entry for entry in _roster(config_path) if entry["name"] != "erin"]
        config_writer.config_replace_list(
            config_path, ["modules", "web_terminals", "users"], remaining
        )
        assert "erin" not in _roster_names(config_path)
        # erin's container/volumes still exist in the runtime — she is now an
        # orphan the roster no longer accounts for.
        assert _container_id(_web_container("erin")) is not None
        assert _volume_exists(bob_claude_vol) and _volume_exists(bob_agent_vol), (
            "bob's retained volumes should still be around going into prune"
        )

        erin_claude_vol, erin_agent_vol = _volume_names("erin")
        alice_claude_vol, alice_agent_vol = _volume_names("alice")

        prune = _run_osprey(osprey_bin, ["deploy", "prune", "--purge", "--yes"], project_dir)
        assert prune.returncode == 0, _fmt("deploy prune --purge", prune)

        assert _container_id(_web_container("erin")) is None, (
            "prune must remove the orphan's container"
        )
        assert not _volume_exists(erin_claude_vol), "prune --purge must remove the orphan's volumes"
        assert not _volume_exists(erin_agent_vol), "prune --purge must remove the orphan's volumes"
        assert not _volume_exists(bob_claude_vol), (
            "prune --purge must also sweep bob's previously-retained orphan volumes"
        )
        assert not _volume_exists(bob_agent_vol), (
            "prune --purge must also sweep bob's previously-retained orphan volumes"
        )

        assert _container_id(_web_container("alice")) is not None, (
            "prune must not touch on-roster users"
        )
        assert _volume_exists(alice_claude_vol), "prune must not touch on-roster volumes"
        assert _volume_exists(alice_agent_vol), "prune must not touch on-roster volumes"

        # --------------------------------------------------------------
        # nuke: full project teardown.
        # --------------------------------------------------------------
        nuke = _run_osprey(osprey_bin, ["deploy", "nuke", "--yes"], project_dir)
        assert nuke.returncode == 0, _fmt("deploy nuke", nuke)

        assert _container_id(_web_container("alice")) is None
        assert _container_id(_nginx_container()) is None
        assert not _volume_exists(alice_claude_vol), "nuke must remove every remaining volume"
        assert not _volume_exists(alice_agent_vol), "nuke must remove every remaining volume"

        leftover = _compose_project_containers(PROJECT_NAME)
        assert leftover == [], f"nuke left project-labeled containers behind: {leftover}"

    finally:
        _teardown_all_project_resources()
