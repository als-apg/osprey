"""Real-container e2e for the multi-user web-terminal deploy lifecycle.

The shipped ``hello-world`` deploy-e2e fixture (``tests/e2e/test_dispatch_deploy.py``
covers the dispatch stack; the CI ``deploy-e2e`` job covers a services-only
``hello-world`` project) declares NO ``modules.web_terminals.users`` and so can
never reach the roster verbs (``osprey users remove``/``prune``/``seed``),
``osprey reset``, or the idempotent web-terminal ``up`` reconcile. This test
drives all of them against a REAL Docker daemon, using
``tests/e2e/fixtures/multi_user_config.yml`` — a facility config whose
``project_name``/``facility.prefix`` make every resource it produces exact-named
and obviously throwaway.

Each fixture below is a three-zone deployment repo: ``profile.yml`` at the root
marks it, ``.env`` beside it is its secret store, ``var/`` holds durable state,
and ``build/`` holds the render the start verbs read. That render is the
committed fixture config rather than a build's output, stamped with the
provenance the drift gate expects — see :func:`_seed_repo`.

It proves LIFECYCLE MECHANICS (containers/volumes get created, reconciled,
seeded, retained/purged/archived, pruned, and torn down correctly) — not the
real terminal application. The per-user image
(``tests/e2e/fixtures/Dockerfile.web_terminal_stub``) is a tiny alpine stand-in
that just stays running (``tail -f /dev/null``) and has what seeding.py's
container-side reconcile scripts need: a ``dispatch`` user and the two mount-point
directories the compose template declares volumes onto. It is pushed into a
throwaway local registry container this test starts, so ``docker compose pull``
(which ``osprey up``'s web-terminal reconcile always runs — see
``web_terminals.provision.deploy_up_web_terminals``) genuinely succeeds against a
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

Gating: needs a container runtime (daemon actually running, not just the CLI
installed). Runtime is ``docker`` by default; set ``OSPREY_E2E_RUNTIME=podman``
to run this file's real-runtime e2e against podman instead (any other value
fails at collection time with a clear error). Skipped entirely if the chosen
runtime's CLI/daemon is unavailable — see the ``stub_image`` fixture.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from osprey.deployment.compose_generator import (
    QMD_CORPUS_ROOT,
    QMD_OKF_COLLECTION,
    resolve_user_volume_names,
)
from osprey.deployment.qmd_service import QMDServiceConfig
from osprey.deployment.staleness import BUILD_DIRNAME
from osprey.deployment.web_terminals.personas import normalize_users
from osprey.services.qmd.client import QMDClient
from osprey.utils import config_writer

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.dockerbuild]

# ---------------------------------------------------------------------------
# Container runtime selection — ``docker`` by default, ``podman`` opt-in via
# OSPREY_E2E_RUNTIME (the CI podman lane sets this). Any other value fails
# clearly at collection time rather than silently falling back to docker.
# ---------------------------------------------------------------------------
_SUPPORTED_RUNTIMES = ("docker", "podman")
RUNTIME = os.environ.get("OSPREY_E2E_RUNTIME", "docker")
if RUNTIME not in _SUPPORTED_RUNTIMES:
    raise RuntimeError(
        f"OSPREY_E2E_RUNTIME={RUNTIME!r} is not supported; expected one of {_SUPPORTED_RUNTIMES}"
    )

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


def _runtime_cli(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run([RUNTIME, *args], capture_output=True, text=True, timeout=timeout)


def _run_osprey(
    osprey_bin: Path, args: list[str], cwd: Path, timeout: int = DEPLOY_VERB_TIMEOUT_SEC
) -> subprocess.CompletedProcess:
    # CONTAINER_RUNTIME takes priority over the fixture's committed
    # container_runtime: docker (see runtime_helper.get_runtime_command) —
    # forcing it to RUNTIME here structurally couples the deploy-side runtime
    # to the assert-side _runtime_cli runtime, regardless of what the ambient
    # environment (or CI job) does or doesn't set.
    #
    # OSPREY_PIP_SPEC is the documented operator escape from the unreleased-
    # version pin refusal (a start from a dev checkout cannot honestly pin a
    # PyPI release). Every image these tests build is a stub Dockerfile that
    # never declares the build arg, so the spec is inert — but its VALUE must
    # stay a loud failure: if a future fixture ever consumes it, pip must
    # refuse rather than silently install a real release.
    return subprocess.run(
        [str(osprey_bin), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={
            **os.environ,
            "CLAUDECODE": "",
            "CONTAINER_RUNTIME": RUNTIME,
            "OSPREY_PIP_SPEC": "osprey-framework==0.0.0+e2e-stub",
        },
    )


def _fmt(label: str, result: subprocess.CompletedProcess) -> str:
    return (
        f"{label} failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def _container_id(name: str) -> str | None:
    result = _runtime_cli("inspect", "--type", "container", "-f", "{{.Id}}", name, timeout=15)
    return result.stdout.strip() if result.returncode == 0 else None


def _volume_exists(name: str) -> bool:
    return _runtime_cli("volume", "inspect", name, timeout=15).returncode == 0


def _container_env_var(name: str, var: str) -> str | None:
    result = _runtime_cli("exec", name, "sh", "-c", f"printenv {var}", timeout=15)
    return result.stdout.strip() if result.returncode == 0 else None


def _compose_project_containers(project: str) -> list[str]:
    """Read-only: containers still labeled as belonging to *project* (any state)."""
    result = _runtime_cli(
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
        _runtime_cli("rm", "-f", _web_container(user))
    _runtime_cli("rm", "-f", _nginx_container())
    _runtime_cli("compose", "-p", PROJECT_NAME, "down", timeout=60)
    for user in USERS:
        for volume in _volume_names(user):
            _runtime_cli("volume", "rm", volume)


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
    ``registry.url`` in multi_user_config.yml). Skips the whole module if the
    selected runtime (``docker`` by default, or ``podman`` via
    ``OSPREY_E2E_RUNTIME``) isn't installed or its daemon isn't actually
    responding — the graceful, non-runtime-environment path the task requires.

    Teardown removes the registry container (exact name) and the local image
    tags this fixture created (exact tags — never ``rmi -a``/a sweep).
    """
    if shutil.which(RUNTIME) is None:
        pytest.skip(f"{RUNTIME} not available")
    if _runtime_cli("ps", timeout=10).returncode != 0:
        pytest.skip(f"{RUNTIME} daemon not responding")

    build = _runtime_cli(
        "build", "-f", str(STUB_DOCKERFILE), "-t", LOCAL_BUILD_TAG, str(FIXTURES_DIR), timeout=120
    )
    if build.returncode != 0:
        pytest.fail(_fmt("stub image build", build))

    # Exact-named: remove any stale registry container a previous crashed run
    # left behind before starting a fresh one under the same name.
    _runtime_cli("rm", "-f", REGISTRY_CONTAINER)
    run_registry = _runtime_cli(
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

        tag = _runtime_cli("tag", LOCAL_BUILD_TAG, STUB_IMAGE, timeout=15)
        if tag.returncode != 0:
            pytest.fail(_fmt("stub image tag", tag))
        push = _runtime_cli("push", STUB_IMAGE, timeout=60)
        if push.returncode != 0:
            pytest.fail(_fmt("stub image push", push))

        # Untag the local convenience copy so `docker compose pull` (which
        # `osprey up`'s web-terminal reconcile always runs) genuinely pulls
        # from the registry rather than short-circuiting on an already-present
        # local tag — this is the exact behavior that makes the fixture prove
        # "locally-resolvable image", not "image docker already happens to have".
        _runtime_cli("rmi", STUB_IMAGE)

        yield STUB_IMAGE
    finally:
        _runtime_cli("rm", "-f", REGISTRY_CONTAINER)
        _runtime_cli("rmi", "-f", STUB_IMAGE)
        _runtime_cli("rmi", "-f", LOCAL_BUILD_TAG)


def _stamp_build_manifest(repo: Path) -> None:
    """Stamp ``build/`` with the provenance a start verb's drift gate reads.

    ``osprey up`` refuses a repo whose ``build/`` carries no manifest, and
    refuses again when the fingerprint the manifest records does not match the
    one it recomputes from ``profile.yml``. These fixtures assemble ``build/``
    by hand — the facility config under test is a committed fixture, not
    something a profile renders — so the fingerprint a real build would have
    stamped is written here through the framework's own helpers, which is what
    makes the gate read CLEAN and the verbs below run as an operator's would.
    """
    from osprey.cli.templates.manifest import get_framework_release_version
    from osprey.deployment.staleness import build_manifest_path, profile_fingerprint

    fingerprint = profile_fingerprint(repo / "profile.yml")
    assert fingerprint is not None, f"fixture profile.yml at {repo} did not resolve"
    build_manifest_path(repo).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "creation": {
                    "osprey_version": get_framework_release_version(),
                    "preset_hash": fingerprint.profile_hash,
                    "profile_key_digests": fingerprint.key_digests,
                },
            }
        ),
        encoding="utf-8",
    )


def _seed_repo(
    dest: Path,
    config_text: str,
    *,
    name: str,
    env_text: str = "",
    write_users_env: bool = True,
) -> Path:
    """A throwaway three-zone deployment repo whose ``build/`` is the fixture config.

    The repo IS the deployment: ``profile.yml`` at its root is what every
    repo-scoped verb finds it by, ``.env`` at that root is its whole secret
    store, ``var/`` holds durable state, and ``build/`` holds the render the
    start verbs read. Only the last of those has fixture content — these tests
    drive lifecycle mechanics against a committed facility config, so the config
    is placed where a build would have rendered it rather than derived from a
    profile.

    ``profile.yml`` is deliberately minimal, and the roster deliberately lives
    only in the rendered config: these tests edit the roster where the roster
    verbs' engine reads it, and a profile that also spelled it would make every
    roster edit drift the build fingerprint.

    Args:
        dest: Directory to create as the repo.
        config_text: The rendered ``build/config.yml``.
        name: The profile name, for ``profile.yml``.
        env_text: Contents of the repo's ``.env``. Empty is fine — ``up``
            requires the file to exist, not to hold anything.
        write_users_env: Write an empty ``.env.users`` beside ``.env``.
            The web compose file names it unconditionally, so compose fails to
            parse without one; the local-mode persona test turns this off
            because generating that file from ``.env`` is its subject.
    """
    build_dir = dest / BUILD_DIRNAME
    build_dir.mkdir(parents=True)
    (build_dir / "config.yml").write_text(config_text, encoding="utf-8")

    # The per-user overlay tree seeding.py reads is build output, so it sits
    # under build/ (seeding._context_dir).
    context_dir = build_dir / "docker" / "web-terminal-context"
    context_dir.mkdir(parents=True)
    (context_dir / "base.md").write_text(_BASE_MD_CONTENT, encoding="utf-8")

    (dest / "profile.yml").write_text(f"name: {name}\n", encoding="utf-8")

    # `osprey up` refuses to start a repo with no .env — it is the file every
    # compose invocation is pointed at with --env-file. The web tier's
    # .env.users sits beside it: the rendered web compose file lives in
    # build/, but compose resolves its env_file entries against the project
    # directory, which every OSPREY compose invocation pins to the repo root.
    (dest / ".env").write_text(env_text, encoding="utf-8")
    if write_users_env:
        (dest / ".env.users").write_text("", encoding="utf-8")

    for state_dir in ("var/agent_data", "var/audit"):
        (dest / state_dir).mkdir(parents=True, exist_ok=True)

    _stamp_build_manifest(dest)
    return dest


@pytest.fixture
def repo(tmp_path: Path, stub_image: str) -> Path:
    """A throwaway deployment repo whose build/ is multi_user_config.yml."""
    return _seed_repo(
        tmp_path / "project",
        CONFIG_FIXTURE.read_text(encoding="utf-8"),
        name=PROJECT_NAME,
    )


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def test_deploy_lifecycle_full_sequence(repo: Path) -> None:
    """Drive up -> idempotent up -> users seed -> users remove (x3 dispositions)
    -> users prune -> reset against a real Docker daemon, asserting each
    lifecycle guarantee the verbs document.
    """
    osprey_bin = _find_osprey_console_script()
    config_path = repo / BUILD_DIRNAME / "config.yml"

    # Extend the committed [alice, bob] roster to the five users this scenario
    # needs, via the same config-editing helper decommission_user itself uses.
    config_writer.config_replace_list(
        config_path, ["modules", "web_terminals", "users"], list(USERS)
    )

    try:
        # --------------------------------------------------------------
        # up: brings up every per-user container + nginx
        # --------------------------------------------------------------
        up1 = _run_osprey(osprey_bin, ["up"], repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up1.returncode == 0, _fmt("osprey up (1st)", up1)

        all_containers = [_web_container(u) for u in USERS] + [_nginx_container()]
        ids_after_first_up = {name: _container_id(name) for name in all_containers}
        missing = [name for name, cid in ids_after_first_up.items() if cid is None]
        assert not missing, f"container(s) not created by 'osprey up': {missing}"

        # --------------------------------------------------------------
        # C2 — idempotency: a 2nd `up` must recreate ZERO containers
        # --------------------------------------------------------------
        up2 = _run_osprey(osprey_bin, ["up"], repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up2.returncode == 0, _fmt("osprey up (2nd)", up2)
        ids_after_second_up = {name: _container_id(name) for name in all_containers}
        assert ids_after_second_up == ids_after_first_up, (
            "idempotent 'osprey up' recreated container(s): "
            f"{[n for n in all_containers if ids_after_second_up[n] != ids_after_first_up[n]]}"
        )

        # --------------------------------------------------------------
        # C3 — seed: CLAUDE.md present; a user-installed (non-managed)
        # skill dir survives a reseed.
        # --------------------------------------------------------------
        claude_md = _runtime_cli(
            "exec", _web_container("alice"), "cat", "/data/claude-config/CLAUDE.md"
        )
        assert claude_md.returncode == 0, _fmt("cat alice's CLAUDE.md", claude_md)
        assert _BASE_MD_MARKER in claude_md.stdout

        skill_dir = f"/app/{FACILITY_PREFIX}-assistant/.claude/skills/user-installed"
        make_skill = _runtime_cli(
            "exec",
            "-u",
            "0",
            _web_container("alice"),
            "sh",
            "-c",
            f"mkdir -p {skill_dir} && touch {skill_dir}/marker.txt",
        )
        assert make_skill.returncode == 0, _fmt("create user-installed skill dir", make_skill)

        reseed = _run_osprey(osprey_bin, ["users", "seed", "alice"], repo)
        assert reseed.returncode == 0, _fmt("users seed alice", reseed)

        survived = _runtime_cli(
            "exec", _web_container("alice"), "test", "-f", f"{skill_dir}/marker.txt"
        )
        assert survived.returncode == 0, "user-installed skill dir did not survive reseed"

        # --------------------------------------------------------------
        # Baseline for the survivor-ports check, captured before touching bob.
        # --------------------------------------------------------------
        carol_port_before = _container_env_var(_web_container("carol"), "OSPREY_WEB_PORT")
        dave_port_before = _container_env_var(_web_container("dave"), "OSPREY_WEB_PORT")
        assert carol_port_before and dave_port_before
        indices_before = _roster_indices(config_path, ("carol", "dave"))

        # --------------------------------------------------------------
        # C5a — users remove (retain, default): container + roster entry
        # gone, BOTH volumes still exist.
        # --------------------------------------------------------------
        bob_claude_vol, bob_agent_vol = _volume_names("bob")
        decomm_bob = _run_osprey(osprey_bin, ["users", "remove", "bob", "--yes"], repo)
        assert decomm_bob.returncode == 0, _fmt("users remove bob", decomm_bob)
        assert _container_id(_web_container("bob")) is None
        assert "bob" not in _roster_names(config_path)
        assert _volume_exists(bob_claude_vol), "retain must not remove the claude-config volume"
        assert _volume_exists(bob_agent_vol), "retain must not remove the agent-data volume"

        # --------------------------------------------------------------
        # C6 — survivor ports: removing bob (mid-list) must not shift
        # carol/dave's frozen indices (and therefore ports).
        # --------------------------------------------------------------
        indices_after = _roster_indices(config_path, ("carol", "dave"))
        assert indices_after == indices_before, (
            f"decommissioning a mid-list user shifted survivor indices: "
            f"before={indices_before} after={indices_after}"
        )
        assert _container_env_var(_web_container("carol"), "OSPREY_WEB_PORT") == carol_port_before
        assert _container_env_var(_web_container("dave"), "OSPREY_WEB_PORT") == dave_port_before

        # --------------------------------------------------------------
        # C5b — users remove --purge: both volumes actually removed.
        # --------------------------------------------------------------
        carol_claude_vol, carol_agent_vol = _volume_names("carol")
        decomm_carol = _run_osprey(
            osprey_bin, ["users", "remove", "carol", "--purge", "--yes"], repo
        )
        assert decomm_carol.returncode == 0, _fmt("users remove carol --purge", decomm_carol)
        assert not _volume_exists(carol_claude_vol), "purge must remove the claude-config volume"
        assert not _volume_exists(carol_agent_vol), "purge must remove the agent-data volume"

        # --------------------------------------------------------------
        # C5c — users remove --archive: a readable tarball is written
        # before the volume is removed.
        # --------------------------------------------------------------
        dave_claude_vol, dave_agent_vol = _volume_names("dave")
        decomm_dave = _run_osprey(
            osprey_bin, ["users", "remove", "dave", "--archive", "--yes"], repo
        )
        assert decomm_dave.returncode == 0, _fmt("users remove dave --archive", decomm_dave)

        # Archived workspaces are durable state, so they land in the var/ zone.
        archive_dir = repo / "var" / "web_terminal_archives"
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
        # roster WITHOUT running `users remove` — exactly the scenario
        # prune_users exists for), then prune off-roster resources.
        #
        # bob is ALSO an orphan at this point: his container is already gone
        # (removed by the retain-mode removal above), but his two
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

        prune = _run_osprey(osprey_bin, ["users", "prune", "--purge", "--yes"], repo)
        assert prune.returncode == 0, _fmt("users prune --purge", prune)

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
        # reset: full teardown of this checkout's deployment. Every container
        # and volume it removes carries this repo's identity label as well as
        # the project name, and build/ goes with them.
        # --------------------------------------------------------------
        reset = _run_osprey(osprey_bin, ["reset", "--yes"], repo)
        assert reset.returncode == 0, _fmt("osprey reset", reset)

        assert _container_id(_web_container("alice")) is None
        assert _container_id(_nginx_container()) is None
        assert not _volume_exists(alice_claude_vol), "reset must remove every remaining volume"
        assert not _volume_exists(alice_agent_vol), "reset must remove every remaining volume"

        leftover = _compose_project_containers(PROJECT_NAME)
        assert leftover == [], f"reset left project-labeled containers behind: {leftover}"

        # The source zone survives a reset; the disposable render does not.
        assert (repo / "profile.yml").is_file(), "reset must never touch the source zone"
        assert not (repo / BUILD_DIRNAME).exists(), "reset must delete build/"

    finally:
        _teardown_all_project_resources()


# =============================================================================
# Two-project isolation: A's destructive verbs must never touch B's resources
# =============================================================================
#
# The single-project test above proves lifecycle MECHANICS (up/idempotency/
# seed/remove/prune/reset) in isolation. It cannot prove ISOLATION itself
# — that two independent OSPREY deployments on the same host never
# cross-contaminate — because it only ever has one compose project running.
# This is the corrected, persona-agnostic acceptance shape for the phase-3
# label-discovery caveat (see lifecycle.py's module docstring's volume- and
# image-scoping boundaries): it stands up two real compose projects (A, B)
# from two distinct temp deployment repos and asserts A's prune/remove/reset
# never name or touch B's containers, volumes, or images.
#
# The image-isolation case is the subtle one: container/volume isolation is
# enforced by the compose-assigned ``com.docker.compose.project`` label, which
# two projects can never share. Image *tags*, however, are host-global — two
# independent deployments that happen to choose the same persona catalog
# ``project``/persona-name pair (plausible: "assistant"/"default" is an
# obvious convention two facilities might both reach for) would produce the
# identical ``<project>-<persona>:local`` tag string. This test manufactures
# exactly that collision — a tag project A's own roster resolves to, but
# which is (independently) labeled with B's ``com.osprey.project`` value —
# and asserts reset's label verification skips it rather than trusting the
# name match.
#
# Same container-ops safety guardrail as the rest of this file: every
# removal argv below names one exact container, one exact volume, one exact
# image tag, or is a project-scoped ``compose -p <project> down`` (never
# ``-v``, never a prune/wildcard/glob).

PROJECT_NAME_A = "osprey-e2e-two-proj-a"
PROJECT_NAME_B = "osprey-e2e-two-proj-b"
PREFIX_A = "e2ea"
PREFIX_B = "e2eb"
USERS_A = ("keeper", "keeper2", "second", "orphan")
USERS_B = ("main", "orphan")

PORTS_A = {"nginx": 19180, "web": 19500, "artifact": 19600, "ariel": 19700, "lattice": 19800}
PORTS_B = {"nginx": 19181, "web": 19900, "artifact": 20000, "ariel": 20100, "lattice": 20200}

# Persona catalog "project" values chosen so that A's own roster reference
# ("mainp") produces a tag distinct from B, while the "borrowed" persona
# models a same-shaped tag that happens to collide with what a (hypothetical)
# sibling deployment already built and labeled as its own.
A_PERSONA_PROJECT = "e2e-two-proj-a-persona"
COLLISION_PERSONA_PROJECT = "e2e-two-proj-collision"
TAG_A_OWN = f"{A_PERSONA_PROJECT}-mainp:local"
TAG_COLLISION = f"{COLLISION_PERSONA_PROJECT}-borrowed:local"


def _make_isolation_repo(
    dest: Path,
    *,
    project_name: str,
    prefix: str,
    ports: dict[str, int],
    users: tuple[str, ...],
) -> Path:
    """A throwaway deployment repo like the ``repo`` fixture, parametrized.

    Distinct ``project_name``/``facility.prefix``/ports/roster so that two
    instances (A, B) can be deployed and run concurrently on the same host
    without colliding — the two-project-isolation test needs both fixtures at
    once, which a single scoped pytest fixture can't provide. Two DIRECTORIES
    also mean two repo identities, which is the label ``reset`` scopes on.
    """
    _seed_repo(dest, CONFIG_FIXTURE.read_text(encoding="utf-8"), name=project_name)

    config_path = dest / BUILD_DIRNAME / "config.yml"
    config_writer.config_update_fields(
        config_path,
        {
            "project_name": project_name,
            "facility.prefix": prefix,
            "modules.web_terminals.nginx_port": ports["nginx"],
            "modules.web_terminals.web_base_port": ports["web"],
            "modules.web_terminals.artifact_base_port": ports["artifact"],
            "modules.web_terminals.ariel_base_port": ports["ariel"],
            "modules.web_terminals.lattice_base_port": ports["lattice"],
        },
    )
    config_writer.config_replace_list(
        config_path, ["modules", "web_terminals", "users"], list(users)
    )
    return dest


def test_deploy_lifecycle_two_project_isolation(tmp_path: Path, stub_image: str) -> None:
    """Two independent compose projects (A, B) on one host: A's users prune/
    users remove/reset must never name or touch B's resources.

    Drives:
      * ``osprey up`` for both A and B (concurrently running stacks).
      * A hand-edited orphan on BOTH sides (config drift, same scenario the
        single-project test's prune case exercises) — so a label-filter
        regression would have something of B's to leak into A's output.
      * A's ``users prune --dry-run``: must list only A's own orphan.
      * A's ``users remove``: must never name a B container/project.
      * A's ``reset``: must remove exactly A's own label-verified persona
        image, leave a same-shaped-but-differently-labeled ("collision") tag
        alone, and leave every one of B's containers/volumes running.
    """
    osprey_bin = _find_osprey_console_script()

    repo_a = _make_isolation_repo(
        tmp_path / "project-a",
        project_name=PROJECT_NAME_A,
        prefix=PREFIX_A,
        ports=PORTS_A,
        users=USERS_A,
    )
    repo_b = _make_isolation_repo(
        tmp_path / "project-b",
        project_name=PROJECT_NAME_B,
        prefix=PREFIX_B,
        ports=PORTS_B,
        users=USERS_B,
    )
    config_path_a = repo_a / BUILD_DIRNAME / "config.yml"
    config_path_b = repo_b / BUILD_DIRNAME / "config.yml"

    image_tags_built: list[str] = []

    try:
        # --------------------------------------------------------------
        # up: both projects, running concurrently.
        # --------------------------------------------------------------
        up_a = _run_osprey(osprey_bin, ["up"], repo_a, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up_a.returncode == 0, _fmt("osprey up (project A)", up_a)
        up_b = _run_osprey(osprey_bin, ["up"], repo_b, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up_b.returncode == 0, _fmt("osprey up (project B)", up_b)

        for user in USERS_A:
            assert _container_id(f"{PREFIX_A}-web-{user}") is not None, (
                f"project A user {user!r} container not created by 'osprey up'"
            )
        for user in USERS_B:
            assert _container_id(f"{PREFIX_B}-web-{user}") is not None, (
                f"project B user {user!r} container not created by 'osprey up'"
            )

        # Simulate config drift on BOTH sides — hand-remove one user from
        # each roster without running `users remove` — so each project has a
        # real orphan container+volumes for prune to discover.
        remaining_a = [entry for entry in _roster(config_path_a) if entry["name"] != "orphan"]
        config_writer.config_replace_list(
            config_path_a, ["modules", "web_terminals", "users"], remaining_a
        )
        remaining_b = [entry for entry in _roster(config_path_b) if entry["name"] != "orphan"]
        config_writer.config_replace_list(
            config_path_b, ["modules", "web_terminals", "users"], remaining_b
        )

        # --------------------------------------------------------------
        # Isolation 1 — A's prune --dry-run must list only A's own orphan,
        # never anything belonging to B.
        # --------------------------------------------------------------
        prune_dry = _run_osprey(osprey_bin, ["users", "prune", "--dry-run"], repo_a)
        assert prune_dry.returncode == 0, _fmt("users prune --dry-run (A)", prune_dry)
        plan = prune_dry.stdout
        assert "orphan" in plan, f"A's own orphan not mentioned in the dry-run plan:\n{plan}"
        assert PROJECT_NAME_B not in plan, f"A's users prune --dry-run named B's project:\n{plan}"
        assert f"{PREFIX_B}-web-orphan" not in plan, (
            f"A's users prune --dry-run named B's orphan container:\n{plan}"
        )
        b_orphan_volumes = resolve_user_volume_names({"project_name": PROJECT_NAME_B}, "orphan")
        assert all(volume not in plan for volume in b_orphan_volumes), (
            f"A's users prune --dry-run named one of B's volumes:\n{plan}"
        )
        # Dry-run is a true no-op: both sides' orphans are untouched.
        assert _container_id(f"{PREFIX_A}-web-orphan") is not None
        assert _container_id(f"{PREFIX_B}-web-orphan") is not None

        # --------------------------------------------------------------
        # Isolation 2 — A's users remove must never name B's resources.
        # --------------------------------------------------------------
        decomm = _run_osprey(osprey_bin, ["users", "remove", "second", "--yes"], repo_a)
        assert decomm.returncode == 0, _fmt("users remove second (A)", decomm)
        assert PROJECT_NAME_B not in decomm.stdout, (
            f"A's users remove named B's project:\n{decomm.stdout}"
        )
        assert f"{PREFIX_B}-web-" not in decomm.stdout, (
            f"A's users remove named a B container:\n{decomm.stdout}"
        )
        assert _container_id(f"{PREFIX_A}-web-second") is None

        # --------------------------------------------------------------
        # Isolation 3 — A's reset removes exactly A's own label-verified
        # persona image; a same-shaped tag labeled as belonging to B
        # survives; B's entire stack (containers + volumes) is untouched.
        # --------------------------------------------------------------
        keeper_idx, keeper2_idx = _roster_indices(config_path_a, ("keeper", "keeper2"))
        config_writer.config_replace_list(
            config_path_a,
            ["modules", "web_terminals", "users"],
            [
                {"name": "keeper", "index": keeper_idx},
                {"name": "keeper2", "index": keeper2_idx, "persona": "borrowed"},
            ],
        )
        config_writer.config_update_fields(
            config_path_a,
            {
                "modules.web_terminals.image_source": "local",
                "modules.web_terminals.default_persona": "mainp",
                "modules.web_terminals.personas": {
                    "mainp": {"project": A_PERSONA_PROJECT},
                    "borrowed": {"project": COLLISION_PERSONA_PROJECT},
                },
            },
        )

        # A trivial local image — content is irrelevant, only the tag and the
        # com.osprey.project label matter to reset's verification.
        build_ctx = tmp_path / "persona-build-ctx"
        build_ctx.mkdir()
        (build_ctx / "Dockerfile").write_text('FROM alpine:3.20\nCMD ["true"]\n', encoding="utf-8")

        build_own = _runtime_cli(
            "build",
            "-t",
            TAG_A_OWN,
            "--label",
            f"com.osprey.project={PROJECT_NAME_A}",
            str(build_ctx),
            timeout=60,
        )
        assert build_own.returncode == 0, _fmt("build A's own persona image", build_own)
        image_tags_built.append(TAG_A_OWN)

        # Same tag SHAPE (<project>-<persona>:local) that A's own roster
        # resolves to via the "borrowed" persona, but labeled as belonging to
        # B — simulating an independent deployment that happens to have
        # built/owned this exact tag string.
        build_collision = _runtime_cli(
            "build",
            "-t",
            TAG_COLLISION,
            "--label",
            f"com.osprey.project={PROJECT_NAME_B}",
            str(build_ctx),
            timeout=60,
        )
        assert build_collision.returncode == 0, _fmt(
            "build B-labeled collision image", build_collision
        )
        image_tags_built.append(TAG_COLLISION)

        reset_a = _run_osprey(osprey_bin, ["reset", "--yes"], repo_a)
        assert reset_a.returncode == 0, _fmt("osprey reset (A)", reset_a)

        assert _runtime_cli("image", "inspect", TAG_A_OWN, timeout=15).returncode != 0, (
            "reset did not remove A's own label-verified persona image"
        )
        assert _runtime_cli("image", "inspect", TAG_COLLISION, timeout=15).returncode == 0, (
            "reset removed a same-shaped tag labeled as belonging to a different "
            "project — the com.osprey.project label verification did not hold"
        )

        assert _container_id(f"{PREFIX_A}-web-keeper") is None
        assert _container_id(f"{PREFIX_A}-nginx") is None

        assert _container_id(f"{PREFIX_B}-web-main") is not None, (
            "A's reset tore down a container belonging to project B"
        )
        assert _container_id(f"{PREFIX_B}-nginx") is not None, (
            "A's reset tore down project B's nginx container"
        )
        b_main_volumes = resolve_user_volume_names({"project_name": PROJECT_NAME_B}, "main")
        assert all(_volume_exists(volume) for volume in b_main_volumes), (
            "A's reset removed a volume belonging to project B"
        )

    finally:
        # Exact-named sweep of every resource either project could have
        # created — mirrors _teardown_all_project_resources's shape, just
        # over two rosters/projects/image tags instead of one.
        for prefix, users in ((PREFIX_A, USERS_A), (PREFIX_B, USERS_B)):
            for user in users:
                _runtime_cli("rm", "-f", f"{prefix}-web-{user}")
            _runtime_cli("rm", "-f", f"{prefix}-nginx")
        for project in (PROJECT_NAME_A, PROJECT_NAME_B):
            _runtime_cli("compose", "-p", project, "down", timeout=60)
        for project, users in ((PROJECT_NAME_A, USERS_A), (PROJECT_NAME_B, USERS_B)):
            for user in users:
                for volume in resolve_user_volume_names({"project_name": project}, user):
                    _runtime_cli("volume", "rm", volume)
        for tag in image_tags_built:
            _runtime_cli("rmi", "-f", tag)


# =============================================================================
# Heterogeneous personas: local-mode build-from-checkout, and registry-mode
# default-persona-only pull
# =============================================================================
#
# The two tests above prove lifecycle mechanics and cross-project isolation for
# the phase-3 homogeneous (every user = one CI-baked image) shape. This section
# is the phase-4 acceptance: a facility config whose roster spans more than one
# persona (its own image + its own project identity), reconciled by `deploy
# up` under BOTH `image_source` modes.
#
# LOCAL MODE (the core case): a temp checkout that ships two fully
# self-contained persona project directories (each just a Dockerfile +
# config.yml — real render output isn't needed, only what
# resolve_personas()/build_persona_images() actually read) alongside a
# facility config referencing them, and ONLY a `.env` file — no
# `.env.users`, no registry, no CI. `osprey up` must derive
# `.env.users` from `.env`, build both persona images locally (never
# pulling — a purely local, never-pushed tag would make `compose pull`
# hard-fail, so a green `osprey up` here is itself proof the pull-guard held),
# and bring up a heterogeneous roster (two users sharing the default persona,
# one on a second, non-default persona) with each container's own declared
# port and each persona's own agent-data mount.
#
# REGISTRY MODE: reuses the existing single-project fixture/registry
# infrastructure (`repo`, `stub_image`) rather than duplicating it,
# adding just a persona catalog with a default persona. Both of the fixture's
# users resolve to that default persona, so `osprey up` must pull the SAME
# un-suffixed stub tag it always did (introducing a persona catalog must not
# re-image the default persona) and must never build anything.
#
# Same container-ops safety guardrail as the rest of this file: every removal
# below names one exact container, one exact volume, one exact image tag, or
# is a project-scoped `compose -p <project> down` (never `-v`, never a
# prune/wildcard/glob).

HETERO_PROJECT_NAME = "osprey-e2e-mus-p4-hetero"
HETERO_PREFIX = "e2ehet"
HETERO_USERS = ("alice", "bob", "carol")

# Disjoint from every other range in this file, including the two-project
# isolation test's 19180-20201.
HETERO_PORTS = {"nginx": 20280, "web": 20300, "artifact": 20400, "ariel": 20500, "lattice": 20600}

HETERO_DEFAULT_PERSONA = "assistant"
HETERO_ALT_PERSONA = "alt"
# The default persona's catalog `project` is deliberately chosen to equal
# `<prefix>-assistant` -- the same string resolve_personas() PINS
# container_project_dir to for the default persona regardless of the catalog
# entry's own `project` field, so this fixture's Dockerfile-created directory
# and resolve_personas()'s resolved mount path agree by construction.
HETERO_DEFAULT_PROJECT = f"{HETERO_PREFIX}-assistant"
HETERO_ALT_PROJECT = f"{HETERO_PREFIX}-alt"
HETERO_DEFAULT_TAG = f"{HETERO_DEFAULT_PROJECT}-{HETERO_DEFAULT_PERSONA}:local"
HETERO_ALT_TAG = f"{HETERO_ALT_PROJECT}-{HETERO_ALT_PERSONA}:local"

# The .env fixture content: one var the generated .env.users MUST carry
# (the LLM key, copied unconditionally), and three it must NEVER carry
# regardless of what's in .env -- the registry token, an external-registry-
# project token, and a service token OSPREY mints for its own containers
# (EVENT_DISPATCHER_TOKEN). None of the three are ever read by
# _build_env_production_subset, so this is a regression guard: if a future
# change ever taught that function to read one of them, the assertion below
# would catch the leak. The service token is the sharpest of the three,
# because the enabling module IS on in this fixture -- an enabled module is
# not, on its own, a licence to hand a web terminal the token its sibling
# containers authenticate with.
_HETERO_ENV_CONTENT = (
    "ANTHROPIC_API_KEY=fake-llm-key-value\n"
    "EVENT_DISPATCHER_TOKEN=fake-dispatcher-token-value\n"
    "REGISTRY_TOKEN=fake-registry-token-value\n"
    "EXTERNAL_PROJECT_TOKEN=fake-external-token-value\n"
)


def _write_persona_project(root: Path, project_name: str, container_project_dir: str) -> Path:
    """A minimal local-mode persona project: ``Dockerfile`` + ``config.yml``,
    in both copies a real build produces (the flat render and its
    ``.image/<name>`` sibling).

    Mirrors ``Dockerfile.web_terminal_stub``'s shape (a throwaway alpine
    stand-in that stays running and satisfies what seeding.py's container-side
    reconcile needs: a ``dispatch`` user and the two mount-point directories
    the compose template declares volumes onto) but parametrized per persona,
    since each persona has its own ``container_project_dir``
    (:func:`osprey.deployment.web_terminals.personas.resolve_personas`'s contract).
    ``config.yml`` only needs ``project_name`` -- nothing in the local-mode
    build path (:func:`osprey.deployment.web_terminals.provision.build_persona_images`)
    reads anything else from a persona project's own config.

    The agent-data directory is pre-created at ``var/agent_data``, which is what
    ``agent_data.base_dir`` defaults to and therefore where the render mounts
    the volume (the shipped ``templates/project/Dockerfile.j2`` pre-creates the
    same path for the same reason). Pre-creating it as ``dispatch`` is what
    makes the mount usable: a volume landing on a directory this image never
    made would be root-owned, and the container-side reconcile runs non-root.
    """
    root.mkdir(parents=True)
    config_text = f"project_name: {project_name}\n"
    dockerfile_text = (
        "FROM alpine:3.20\n"
        "RUN adduser -D dispatch \\\n"
        "    && mkdir -p /data/claude-config \\\n"
        f"    && mkdir -p {container_project_dir}/var/agent_data \\\n"
        f"    && mkdir -p {container_project_dir}/.claude/skills \\\n"
        f"    && chown -R dispatch:dispatch /data/claude-config {container_project_dir}\n"
        'CMD ["tail", "-f", "/dev/null"]\n'
    )
    (root / "config.yml").write_text(config_text, encoding="utf-8")
    (root / "Dockerfile").write_text(dockerfile_text, encoding="utf-8")
    # The container copy the persona image is actually built from: `docker build
    # -f <context>/build/Dockerfile <context>` with the context at the
    # `.image/<name>` sibling (`_persona_image_context`). The stub's recipe
    # holds no host paths, so the container copy IS the host copy.
    image_build = root.parent / ".image" / root.name / "build"
    image_build.mkdir(parents=True)
    (image_build / "config.yml").write_text(config_text, encoding="utf-8")
    (image_build / "Dockerfile").write_text(dockerfile_text, encoding="utf-8")
    return root


def _hetero_config_dict(default_persona_path: Path, alt_persona_path: Path) -> dict:
    """The heterogeneous facility config: local-mode image_source, a 2-entry
    persona catalog, and a roster where alice/bob inherit the default persona
    and carol references the non-default one explicitly.
    """
    return {
        "project_name": HETERO_PROJECT_NAME,
        "container_runtime": "docker",
        "facility": {
            "name": "E2E Heterogeneous Persona Fixture",
            "prefix": HETERO_PREFIX,
            "timezone": "UTC",
        },
        "llm": {"api_key_env_var": "ANTHROPIC_API_KEY"},
        "registry": {
            "url": "unused-registry.example.invalid",
            "token_env_var": "REGISTRY_TOKEN",
            "external_projects": [
                {
                    "name": "sibling",
                    "url": "unused-registry.example.invalid/sibling",
                    "image": "sibling:latest",
                    "token_env_var": "EXTERNAL_PROJECT_TOKEN",
                }
            ],
        },
        "deploy": {"fqdn": "localhost"},
        "deployed_services": [],
        "modules": {
            "event_dispatcher": {"enabled": True},
            "web_terminals": {
                "enabled": True,
                "image_source": "local",
                "default_persona": HETERO_DEFAULT_PERSONA,
                "personas": {
                    HETERO_DEFAULT_PERSONA: {
                        "project": HETERO_DEFAULT_PROJECT,
                        "project_path": str(default_persona_path),
                    },
                    HETERO_ALT_PERSONA: {
                        "project": HETERO_ALT_PROJECT,
                        "project_path": str(alt_persona_path),
                    },
                },
                "nginx_port": HETERO_PORTS["nginx"],
                "web_base_port": HETERO_PORTS["web"],
                "artifact_base_port": HETERO_PORTS["artifact"],
                "ariel_base_port": HETERO_PORTS["ariel"],
                "lattice_base_port": HETERO_PORTS["lattice"],
                "users": [
                    "alice",
                    "bob",
                    {"name": "carol", "index": 2, "persona": HETERO_ALT_PERSONA},
                ],
            },
        },
    }


def _make_hetero_repo(tmp_path: Path) -> Path:
    """Build the throwaway checkout: two persona project dirs + the deployment
    repo that references them, with ONLY a ``.env`` (no ``.env.users``) --
    the exact "checkout with only a .env" shape the local-mode acceptance
    scenario needs.
    """
    default_persona_path = _write_persona_project(
        tmp_path / "persona-assistant",
        HETERO_DEFAULT_PROJECT,
        f"/app/{HETERO_DEFAULT_PROJECT}",
    )
    alt_persona_path = _write_persona_project(
        tmp_path / "persona-alt", HETERO_ALT_PROJECT, f"/app/{HETERO_ALT_PROJECT}"
    )

    return _seed_repo(
        tmp_path / "project",
        yaml.safe_dump(
            _hetero_config_dict(default_persona_path, alt_persona_path), sort_keys=False
        ),
        name=HETERO_PROJECT_NAME,
        env_text=_HETERO_ENV_CONTENT,
        write_users_env=False,
    )


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


def _container_image(name: str) -> str | None:
    result = _runtime_cli(
        "inspect", "--type", "container", "-f", "{{.Config.Image}}", name, timeout=15
    )
    if result.returncode != 0:
        return None
    # Podman fully-qualifies a registry-less locally-built tag with a
    # `localhost/` prefix in Config.Image; docker keeps the bare `<name>:<tag>`.
    # Strip only that synthetic prefix so image-identity assertions read the
    # same on both runtimes. An explicit registry ref (`localhost:19081/...`,
    # `host/...`) starts with `localhost:` / `<host>/` and is left untouched.
    return result.stdout.strip().removeprefix("localhost/")


def _container_mounts(name: str) -> list[dict]:
    result = _runtime_cli(
        "inspect", "--type", "container", "-f", "{{json .Mounts}}", name, timeout=15
    )
    if result.returncode != 0:
        return []
    return json.loads(result.stdout)


def _mount_destination(name: str, volume_name: str) -> str | None:
    for mount in _container_mounts(name):
        if mount.get("Name") == volume_name:
            return mount.get("Destination")
    return None


def test_deploy_lifecycle_heterogeneous_local_mode_up(tmp_path: Path, stub_image: str) -> None:
    """The phase-4 flagship: a heterogeneous 2-persona LOCAL-MODE deploy from a
    checkout with only a ``.env`` reaches healthy.

    Drives a single ``osprey up`` against a deployment repo whose roster spans
    two personas (alice+bob on the default, carol on a second, non-default
    persona), then targeted-recreates carol's container to prove her
    persona's agent-data volume (mounted at HER OWN project dir, not the
    default persona's) survives.
    """
    osprey_bin = _find_osprey_console_script()
    repo = _make_hetero_repo(tmp_path)
    users_env_path = repo / ".env.users"

    alice_c = f"{HETERO_PREFIX}-web-alice"
    bob_c = f"{HETERO_PREFIX}-web-bob"
    carol_c = f"{HETERO_PREFIX}-web-carol"
    nginx_c = f"{HETERO_PREFIX}-nginx"

    try:
        # Precondition: exactly the "checkout with only a .env" shape.
        assert (repo / ".env").is_file()
        assert not users_env_path.exists()

        # --------------------------------------------------------------
        # up: local build of both referenced persona images, .env.users
        # generation, and bring-up of a heterogeneous roster. A local-only,
        # never-pushed tag makes `compose pull` hard-fail -- a green return
        # here is itself proof the local-mode pull-guard held (no `pull` ran).
        # --------------------------------------------------------------
        up1 = _run_osprey(osprey_bin, ["up"], repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up1.returncode == 0, _fmt("osprey up (heterogeneous local mode)", up1)

        for name in (alice_c, bob_c, carol_c, nginx_c):
            assert _container_id(name) is not None, f"{name} not created by 'osprey up'"

        # --------------------------------------------------------------
        # .env.users: generated, 0600, module-conditional subset --
        # carries the LLM key, never the registry/external-project tokens or
        # OSPREY's own minted service tokens, all of which are in .env.
        # --------------------------------------------------------------
        assert users_env_path.is_file(), ".env.users was not generated from .env"
        mode = users_env_path.stat().st_mode & 0o777
        assert mode == 0o600, f".env.users mode {oct(mode)} != 0600"
        content = users_env_path.read_text(encoding="utf-8")
        assert "ANTHROPIC_API_KEY=fake-llm-key-value" in content
        for excluded in ("REGISTRY_TOKEN", "EVENT_DISPATCHER_TOKEN", "EXTERNAL_PROJECT_TOKEN"):
            assert excluded not in content, (
                f"{excluded} leaked into generated .env.users:\n{content}"
            )

        # --------------------------------------------------------------
        # Two distinct :local images, each labeled with THIS deployment's
        # project (not either persona's own project) -- what a later `reset`
        # verifies before removing a persona image tag.
        # --------------------------------------------------------------
        assert _image_exists(HETERO_DEFAULT_TAG), f"{HETERO_DEFAULT_TAG} was not built"
        assert _image_exists(HETERO_ALT_TAG), f"{HETERO_ALT_TAG} was not built"
        assert _image_label(HETERO_DEFAULT_TAG, "com.osprey.project") == HETERO_PROJECT_NAME
        assert _image_label(HETERO_ALT_TAG, "com.osprey.project") == HETERO_PROJECT_NAME

        # alice & bob share the DEFAULT persona's image; carol runs the alt one.
        assert _container_image(alice_c) == HETERO_DEFAULT_TAG
        assert _container_image(bob_c) == HETERO_DEFAULT_TAG
        assert _container_image(carol_c) == HETERO_ALT_TAG

        # --------------------------------------------------------------
        # Two same-persona users on DISTINCT declared ports.
        # --------------------------------------------------------------
        assert _container_env_var(alice_c, "OSPREY_TERMINAL_WEB_PORT") == str(
            HETERO_PORTS["web"] + 0
        )
        assert _container_env_var(bob_c, "OSPREY_TERMINAL_WEB_PORT") == str(HETERO_PORTS["web"] + 1)
        assert _container_env_var(carol_c, "OSPREY_TERMINAL_WEB_PORT") == str(
            HETERO_PORTS["web"] + 2
        )

        # --------------------------------------------------------------
        # carol's (non-default persona) agent-data volume mounts at HER OWN
        # project dir, not the default persona's /app/<prefix>-assistant.
        # --------------------------------------------------------------
        carol_agent_volume = f"{HETERO_PROJECT_NAME}_carol-agent-data"
        assert (
            _mount_destination(carol_c, carol_agent_volume)
            == f"/app/{HETERO_ALT_PROJECT}/var/agent_data"
        )

        # --------------------------------------------------------------
        # Content written into that volume survives a container recreate.
        # Recreate ONLY carol's service via a targeted `compose ...
        # --force-recreate web-carol` (never rebuilding any image or
        # touching alice/bob/nginx), so this proves volume persistence
        # without confounding it with a second full `osprey up` reconcile.
        # --------------------------------------------------------------
        marker_path = f"/app/{HETERO_ALT_PROJECT}/var/agent_data/marker.txt"
        touch = _runtime_cli("exec", carol_c, "sh", "-c", f"touch {marker_path}", timeout=15)
        assert touch.returncode == 0, _fmt("touch marker in carol's agent-data volume", touch)

        alice_id_before = _container_id(alice_c)
        bob_id_before = _container_id(bob_c)
        nginx_id_before = _container_id(nginx_c)
        carol_id_before = _container_id(carol_c)

        # The web stack's compose file is build output, and every OSPREY
        # compose invocation pins the repo root as the project directory --
        # which is what lets a file under build/ name .env.users at the
        # root. This targeted recreate has to pin it the same way.
        compose_web_file = repo / BUILD_DIRNAME / "docker-compose.web.yml"
        assert compose_web_file.is_file()
        recreate = _runtime_cli(
            "compose",
            "-p",
            HETERO_PROJECT_NAME,
            "--project-directory",
            str(repo),
            "-f",
            str(compose_web_file),
            "up",
            "-d",
            "--force-recreate",
            "web-carol",
            timeout=60,
        )
        assert recreate.returncode == 0, _fmt("targeted recreate of web-carol", recreate)

        assert _container_id(alice_c) == alice_id_before, "alice's container was touched"
        assert _container_id(bob_c) == bob_id_before, "bob's container was touched"
        assert _container_id(nginx_c) == nginx_id_before, "nginx's container was touched"
        assert _container_id(carol_c) != carol_id_before, "carol's container was not recreated"

        survived = _runtime_cli("exec", carol_c, "test", "-f", marker_path, timeout=15)
        assert survived.returncode == 0, "marker did not survive carol's container recreate"

    finally:
        for name in (alice_c, bob_c, carol_c):
            _runtime_cli("rm", "-f", name)
        _runtime_cli("rm", "-f", nginx_c)
        _runtime_cli("compose", "-p", HETERO_PROJECT_NAME, "down", timeout=60)
        for user in HETERO_USERS:
            for volume in resolve_user_volume_names({"project_name": HETERO_PROJECT_NAME}, user):
                _runtime_cli("volume", "rm", volume)
        for tag in (HETERO_DEFAULT_TAG, HETERO_ALT_TAG):
            _runtime_cli("rmi", "-f", tag)


def test_deploy_lifecycle_heterogeneous_registry_mode_up(repo: Path, stub_image: str) -> None:
    """Registry mode with a persona catalog: the default persona still pulls
    the unchanged, un-suffixed stub tag, and no local build is ever invoked.

    Reuses the single-project fixture's ``repo``/``stub_image``
    infrastructure (its committed ``[alice, bob]`` roster, its registry
    pointed at the throwaway local registry the ``stub_image`` fixture
    pushed into) rather than duplicating it -- only a persona catalog with a
    default persona is added on top.
    """
    osprey_bin = _find_osprey_console_script()
    config_path = repo / BUILD_DIRNAME / "config.yml"

    config_writer.config_update_fields(
        config_path,
        {
            "modules.web_terminals.image_source": "registry",
            "modules.web_terminals.default_persona": "assistant",
            "modules.web_terminals.personas": {
                "assistant": {"project": f"{FACILITY_PREFIX}-assistant"}
            },
        },
    )

    # The tag a LOCAL build would have produced for this persona -- must
    # never exist, since registry mode builds nothing.
    never_built_tag = f"{FACILITY_PREFIX}-assistant-assistant:local"

    try:
        up = _run_osprey(osprey_bin, ["up"], repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up.returncode == 0, _fmt("osprey up (registry-mode personas)", up)

        for user in ("alice", "bob"):
            container = _web_container(user)
            assert _container_id(container) is not None, f"{container} not created"
            image = _container_image(container)
            assert image == stub_image, (
                f"{user}'s default-persona container did not pull the unsuffixed "
                f"registry tag: got {image!r}, expected {stub_image!r}"
            )

        assert not _image_exists(never_built_tag), (
            f"{never_built_tag} exists locally -- registry-mode 'osprey up' must never build"
        )

    finally:
        _teardown_all_project_resources()
        _runtime_cli("rmi", "-f", never_built_tag)


# =============================================================================
# Shared facility-knowledge bundle + qmd search sidecar
# =============================================================================
#
# The tests above prove web-terminal lifecycle mechanics. This one proves the
# CROSS-CONTAINER SHARING property the bundle mount exists for: one host
# directory bound read-write into every entitled `web-<user>` container and
# read-only into the qmd sidecar, such that a file written from one terminal is
# readable in another and reaches the sidecar's index.
#
# WHAT MAKES THIS A REAL TEST, AND NOT A TAUTOLOGY
# ------------------------------------------------
# Sharing has two independent halves and only one of them is filesystem state:
#
#   * setgid on the host directory decides which group OWNS a newly written
#     file (compose_generator.SHARED_CORPUS_DIR_MODE), and
#   * `group_add: ["<gid>"]` on each entitled service decides which groups the
#     container's processes are MEMBERS of (render.render_web_terminals).
#
# A write-then-read check alone exercises neither: every persona image in this
# module runs the same uid, so the read would succeed by uid coincidence with
# `group_add` deleted. So the membership half is asserted directly, three times
# over: in the RENDERED overlay (the gid the deploy read off the provisioned
# directory), in the container's RUNTIME CONFIG (`HostConfig.GroupAdd` — proof
# compose passed it down), and INSIDE a running process (`id -G`). Removing the
# renderer's `group_add` emission fails all three, regardless of what the
# filesystem would have allowed.
#
# The `id -G` probe runs as the image's unprivileged `dispatch` user and not as
# root, and that is a correctness requirement rather than a nicety: alpine's
# root already belongs to gids 0-27, so on a host whose shared-corpus group
# falls in that range — macOS `staff` is gid 20 — a root probe reports
# membership the IMAGE granted and passes with `group_add` deleted. Measured on
# this branch, not assumed.
#
# HOST CAVEAT, measured on this branch rather than assumed: Docker Desktop for
# macOS remaps bind-mount ownership -- a host directory owned by 501:20 with
# mode 2770 appears inside the container as root:root, and a container process
# of any uid may write it. The filesystem leg below therefore proves the MOUNT
# TOPOLOGY (one directory, three containers, `:ro` where it must be) on every
# host, and proves permission enforcement only on a Linux runtime. The two
# membership assertions are host-independent, which is why they carry the
# property.
#
# WHAT IS NOT CROSS-CONTAINER WRITABLE, by design: setgid fixes group
# ownership, not permission BITS, which still come from the writer's umask
# (022 -> rw-r--r--). Another container can READ and the sidecar can INDEX a
# drafted file; it cannot overwrite it in place. The assertions below stop
# exactly there.
#
# THE SIDECAR IMAGE IS NEVER BUILT HERE. It is a ~5 GB image whose build bakes
# 2.1 GB of GGUF models; this test consumes whatever `OSPREY_QMD_IMAGE` names
# (the same variable the rendered compose fragment overrides `image:` with, and
# the one the CI lane sets), defaulting to the locally verified tag. A missing
# image is a loud failure naming the tag -- never a skip, which would report
# success having proved nothing.
#
# Because that refusal is a hard failure rather than a skip, the test carries
# its own `qmd` marker: two CI lanes run this module wholesale without building
# or loading a sidecar image, and they deselect it with `-m 'not qmd'`. It is
# reached instead from the lane that builds the image. Adding another
# image-dependent test here means marking it the same way.
#
# Same container-ops safety guardrail as the rest of this file: the sidecar is
# exact-named `osprey-e2e-mus-p3-qmd`, removed by that name, and its stack goes
# down with the project-scoped `compose -p <project> down` the module already
# uses. No prune, no `-a`, no wildcard, no `-v`.

#: Host port the sidecar publishes. Disjoint from every other port this file
#: uses (19080/19081 nginx + registry, 19100-19400 the web tiers, 19180-20201
#: the two-project isolation test, 20280-20600 the heterogeneous tests).
QMD_PORT = 19085

#: Exact container name the sidecar fragment declares -- `<project>-qmd`, the
#: same namespacing rule the shipped service template applies.
QMD_CONTAINER = f"{PROJECT_NAME}-qmd"

#: The sidecar image, honouring `OSPREY_QMD_IMAGE`. Never built by this file.
QMD_IMAGE = os.environ.get("OSPREY_QMD_IMAGE") or "osprey-qmd:local-validate"

#: `facility_knowledge.bundle_path` for this scenario, relative to the repo
#: root exactly as a facility config spells it.
BUNDLE_REL = "data/facility_knowledge"

#: Where that same directory lands inside a web-terminal container. Derived by
#: render._container_bundle_dir as `<container_project_dir>/<bundle_path>`, and
#: `container_project_dir` for this fixture's persona-less roster is
#: `/app/<facility_prefix>-assistant` (resolve_personas' zero-migration path) --
#: the directory Dockerfile.web_terminal_stub already creates.
CONTAINER_BUNDLE_DIR = f"/app/{FACILITY_PREFIX}-assistant/{BUNDLE_REL}"

#: Where the sidecar mounts it, and the collection it is indexed under. Both
#: come from the framework's own constants rather than being re-spelled, since
#: the collection name is a contract with every OKF query.
SIDECAR_CORPUS_TARGET = f"{QMD_CORPUS_ROOT}/{QMD_OKF_COLLECTION}"

#: Marker file a corpus writer touches to ask the sidecar to re-index without
#: waiting out the fallback sweep (the sidecar entrypoint's `MARKER_NAME`).
QMD_TOUCH_MARKER = ".qmd-touch"

#: The seed corpus. The sidecar's entrypoint REFUSES to open its port on an
#: empty index (a healthy-looking sidecar that can find nothing is worse than
#: one that will not start), so the bundle must hold documents before `up`.
#: Two short documents, deliberately: this is a mount/sharing test, and a
#: larger corpus buys nothing but embedding time.
BUNDLE_SEED_DIR = FIXTURES_DIR / "facility_bundle"

#: How long the sidecar gets to build its first index and open the port. Two
#: documents index in seconds, but the embedder loads on CPU on first use.
QMD_READY_TIMEOUT_SEC = 300.0

#: How long a marker-triggered re-index gets to make a new document findable.
#: The sidecar polls markers on its own interval and then re-runs `qmd update`
#: + `qmd embed`; on macOS the bind-mount stat() also has to cross the VM.
QMD_REINDEX_TIMEOUT_SEC = 240.0

#: Marker poll interval the sidecar runs with here -- below the entrypoint's
#: own 5 s default (`POLL_INTERVAL`), because the freshness assertion waits on
#: it and the MECHANISM is what is under test, not the production tuning. Not
#: to be confused with the fallback SWEEP, a separate knob
#: (`OSPREY_QMD_UPDATE_INTERVAL`, shipped default 30 s) that this fixture pins
#: to 300 s precisely so a re-index inside the assertion window can only be
#: explained by the marker.
QMD_MARKER_POLL_SEC = 2


def _write_qmd_service_render(repo: Path) -> None:
    """Place the sidecar's rendered service artifacts under ``build/``.

    Hand-authored for the same reason ``build/config.yml`` is (see
    :func:`_seed_repo`): these fixtures drive lifecycle mechanics against a
    committed render rather than one a profile produced, and ``osprey up``
    reads ``build/`` without re-rendering anything.

    It is a deliberately NARROWED copy of the shipped fragment
    (``osprey/templates/services/qmd/docker-compose.yml.j2``). It diverges from
    it in five places, and each one is a choice rather than an omission:

    * **No ``build:`` block.** The shipped fragment carries one so a real
      deployment can produce the image on first run. Here the image is a
      prebuilt ~5 GB tag, and a fragment that *could* fall back to building it
      is a fragment that eventually will.
    * **No named index volume.** The shipped fragment persists the index
      because rebuilding it costs ~41 minutes at facility scale. Two fixture
      documents rebuild in seconds, and a per-run index means a stale volume
      can never make this test pass on a previous run's corpus.
    * **No ``labels:`` block and no network stanza or ``TZ``.** Those carry the
      deployment's identity and topology, which nothing here asserts on; the
      test finds the container by its exact name and reaches it over the
      published loopback port.
    * **No ``healthcheck:``.** The shipped probe exists so an orchestrator can
      report readiness; this test waits on ``GET /health`` itself, with its own
      timeout and its own failure message carrying the container's logs.
    * **``OSPREY_QMD_UPDATE_INTERVAL: "300"`` and an added
      ``OSPREY_QMD_MARKER_POLL_INTERVAL``.** A rendered deploy emits the sweep
      interval from ``qmd_service.DEFAULT_INTERVAL_SECONDS`` (30 s) and emits no
      marker-poll value at all. Both are retuned here so the freshness
      assertion has one explanation: the sweep is pushed far outside the
      assertion window and the marker poll is pulled well inside it, which is
      what makes a re-index attributable to the marker rather than to a sweep
      that happened to land.

    What the test asserts on is spelled exactly as the template spells it: the
    published loopback port, the ``:ro`` corpus mount at
    ``/corpus/<collection>``, the collection name, and the entrypoint's
    ``OSPREY_QMD_*`` contract. The rendered template itself is covered
    independently by ``tests/deployment/test_qmd_compose_fragment.py``, so this
    copy is not the only thing standing behind that fragment.
    """
    service_dir = repo / BUILD_DIRNAME / "services" / "qmd"
    service_dir.mkdir(parents=True, exist_ok=True)

    # One collection per mounted corpus, at the mount target below. The
    # entrypoint copies this into its state directory on every start.
    (service_dir / "index.yml").write_text(
        f"collections:\n"
        f"  {QMD_OKF_COLLECTION}:\n"
        f"    path: {SIDECAR_CORPUS_TARGET}\n"
        f'    pattern: "**/*.md"\n',
        encoding="utf-8",
    )

    # Every relative path resolves against the pinned compose project
    # directory, which is the repo root -- never this file's own subdir.
    (service_dir / "docker-compose.yml").write_text(
        f"""services:
  qmd:
    image: {QMD_IMAGE}
    container_name: {QMD_CONTAINER}
    restart: unless-stopped
    ports:
      - "127.0.0.1:{QMD_PORT}:{QMD_PORT}"
    environment:
      OSPREY_QMD_PORT: "{QMD_PORT}"
      OSPREY_QMD_UPDATE_INTERVAL: "300"
      OSPREY_QMD_MARKER_POLL_INTERVAL: "{QMD_MARKER_POLL_SEC}"
      OSPREY_QMD_STATE_DIR: "/var/lib/qmd"
      OSPREY_QMD_INDEX_CONFIG: "/etc/qmd/index.yml"
    volumes:
      - ./{BUILD_DIRNAME}/services/qmd/index.yml:/etc/qmd/index.yml:ro
      # collection `{QMD_OKF_COLLECTION}` -- READ-ONLY on purpose: the sidecar
      # indexes this tree, and every process that writes it (the web terminals)
      # is outside this container.
      - ./{BUNDLE_REL}:{SIDECAR_CORPUS_TARGET}:ro
""",
        encoding="utf-8",
    )


def _image_user_gid(image: str, user: str) -> int | None:
    """Primary gid of *user* inside *image*, or ``None`` when it cannot be read.

    One ephemeral, self-removing container off a local tag — never the registry
    ref, which the ``stub_image`` fixture deliberately untags so that
    ``compose pull`` is a genuine pull.
    """
    result = _runtime_cli("run", "--rm", "-u", user, image, "id", "-g", timeout=60)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return int(value) if value.isdigit() else None


def _retarget_bundle_group_if_vacuous(bundle_dir: Path, probe_gid: int) -> None:
    """Make sure the shared corpus group is one the image does not already grant.

    The ``id -G`` probe below is only evidence when the gid it looks for is a
    gid the container could not have had anyway. The unprivileged ``dispatch``
    user the stub image creates takes the first free gid, 1000 — and a Linux
    deploying user whose own primary group is 1000 (the shape GitHub-hosted
    runners have, where ``runner``'s primary group is ``docker``) hands the
    bundle directory that same gid. The probe would then pass with ``group_add``
    deleted, on exactly the hosts CI uses.

    So on that collision the directory is chgrp'd to another group the
    deploying user belongs to. This is not a workaround bolted onto the test:
    it is the documented retargeting path (``ensure_shared_corpus_dir`` inherits
    the group and never invents one, and the compose template says to retarget
    by chgrp'ing the bundle), exercised here for the reason it exists.

    Deliberately conditional. On a host with no collision the directory keeps
    the group a real deploy would have given it, which is the shape worth
    testing; only the pathological host takes the other branch.
    """
    if bundle_dir.stat().st_gid != probe_gid:
        return
    alternatives = [gid for gid in os.getgroups() if gid != probe_gid]
    assert alternatives, (
        f"the deploying user belongs to no group other than {probe_gid}, which is also "
        "the stub image's own 'dispatch' group. The container-side membership probe "
        "cannot be made non-vacuous on this host."
    )
    os.chown(bundle_dir, -1, alternatives[0])


def _wait_for_qmd_health(timeout: float) -> None:
    """Block until the sidecar opens its port, or fail naming what went wrong.

    The entrypoint runs its whole startup index pass BEFORE it opens the port,
    so this covers indexing the seed corpus, not just process start. A sidecar
    that never opens the port has almost always died in that pass; the last
    lines of its log say why, so they are carried into the failure.
    """
    deadline = time.monotonic() + timeout
    last_err = "(no attempt yet)"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(  # noqa: S310 - loopback only
                f"http://127.0.0.1:{QMD_PORT}/health", timeout=5.0
            ) as resp:
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = str(exc)
        time.sleep(2.0)
    logs = _runtime_cli("logs", "--tail", "40", QMD_CONTAINER, timeout=30)
    raise AssertionError(
        f"qmd sidecar did not answer /health on :{QMD_PORT} within {timeout:.0f}s "
        f"(last: {last_err})\n--- {QMD_CONTAINER} logs (tail) ---\n"
        f"{logs.stdout}\n{logs.stderr}"
    )


def _wait_for_indexed(client: QMDClient, token: str, filename: str, timeout: float) -> list:
    """Poll the sidecar until *filename* is among the hits for *token*.

    The wait condition is that ONE named document is findable, never that the
    query returned anything: the semantic half of a hybrid search returns its
    nearest neighbours for any input, so a query for a token that is nowhere in
    the corpus still comes back with the corpus's other documents. Polling on
    "non-empty" would therefore return instantly, before the re-index it is
    supposed to be waiting for, and the freshness assertion would prove
    nothing.

    Returns:
        The last hit list observed — the matching one when the wait succeeded,
        and whatever the sidecar was answering with when it timed out, which is
        what the caller reports.
    """
    deadline = time.monotonic() + timeout
    hits: list = []
    while time.monotonic() < deadline:
        # rerank=False deliberately: the reranker is a 610 MB model that loads
        # on CPU on first use and blows the client's timeout against a cold
        # sidecar. Nothing here is a ranking-quality assertion.
        hits = client.query(QMD_OKF_COLLECTION, token, limit=10, rerank=False)
        if any(hit.file == filename for hit in hits):
            return hits
        time.sleep(3.0)
    return hits


@pytest.mark.qmd
def test_deploy_lifecycle_shared_bundle_and_qmd_sidecar(repo: Path, stub_image: str) -> None:
    """One shared knowledge bundle across two web terminals and the sidecar.

    Reuses the single-project ``repo``/``stub_image`` infrastructure -- its
    committed ``[alice, bob]`` roster and the throwaway local registry the stub
    image was pushed into -- and adds only what this scenario needs: a
    ``facility_knowledge.bundle_path``, a seeded corpus at that path, and a
    ``services.qmd`` sidecar deployed alongside the web tier.

    Asserted, in order:

    1. the deploy provisioned the bundle directory setgid + group-writable;
    2. the RENDERED web overlay binds that one directory into both users'
       containers and puts both in the directory's group (``group_add``);
    3. a running container is actually IN that group (``id -G``) -- the half
       setgid cannot supply, and the assertion that makes this test fail if the
       renderer stops emitting ``group_add``;
    4. the sidecar holds the same directory read-only, and cannot write it;
    5. a file written from alice's container is readable from bob's; and
    6. touching the corpus marker makes that file findable through the
       sidecar's endpoint -- the whole path, host to index to query.
    """
    osprey_bin = _find_osprey_console_script()
    config_path = repo / BUILD_DIRNAME / "config.yml"

    # Seed the corpus BEFORE `up`: the sidecar's entrypoint refuses to open its
    # port on an empty index, so an unseeded bundle would fail as a health
    # timeout rather than as the thing it is.
    bundle_dir = repo / BUNDLE_REL
    shutil.copytree(BUNDLE_SEED_DIR, bundle_dir)

    # The gid the container-side membership probe must NOT be looking for, read
    # off the image rather than assumed. See _retarget_bundle_group_if_vacuous.
    dispatch_primary_gid = _image_user_gid(LOCAL_BUILD_TAG, "dispatch")
    assert dispatch_primary_gid is not None, (
        f"could not read dispatch's primary gid from {LOCAL_BUILD_TAG}; the "
        "container-side membership probe cannot be shown to be non-vacuous"
    )
    _retarget_bundle_group_if_vacuous(bundle_dir, dispatch_primary_gid)

    config_writer.config_update_fields(
        config_path,
        {
            # The entitlement AND the location, in one key: a project that
            # names a bundle path gets the mount
            # (personas.config_needs_facility_bundle), and this fixture's
            # roster carries no persona references, so both users resolve
            # their entitlement from here.
            "facility_knowledge.bundle_path": BUNDLE_REL,
            # `path` is what find_service_config resolves the rendered compose
            # fragment under build/ from; `port` is what the client and the
            # published binding agree on (deployment.qmd_service).
            "services.qmd": {"path": "services/qmd", "port": QMD_PORT},
            "deployed_services": ["qmd"],
        },
    )
    _write_qmd_service_render(repo)

    # A missing image is a loud failure, never a skip: nothing here builds one,
    # and a lane that skipped on an absent image would report success having
    # proved nothing.
    assert _image_exists(QMD_IMAGE), (
        f"qmd sidecar image {QMD_IMAGE!r} is not present locally. This test never "
        "builds it (it is a ~5 GB image). Build or load it, or point "
        "OSPREY_QMD_IMAGE at a tag that exists."
    )

    try:
        up = _run_osprey(osprey_bin, ["up"], repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up.returncode == 0, _fmt("osprey up (shared bundle + qmd sidecar)", up)

        # --------------------------------------------------------------
        # 1 — the deploy provisioned the shared directory itself
        # --------------------------------------------------------------
        bundle_stat = bundle_dir.stat()
        bundle_gid = bundle_stat.st_gid
        assert bundle_stat.st_mode & stat.S_ISGID, (
            f"{bundle_dir} is not setgid after 'osprey up' "
            f"(mode {stat.S_IMODE(bundle_stat.st_mode):04o}); files written from one "
            "container would not take the directory's group"
        )
        assert bundle_stat.st_mode & stat.S_IWGRP, (
            f"{bundle_dir} is not group-writable after 'osprey up' "
            f"(mode {stat.S_IMODE(bundle_stat.st_mode):04o})"
        )

        # --------------------------------------------------------------
        # 2 — the RENDERED overlay: one source directory for both users,
        # and the group that makes its shared bits reach them.
        # --------------------------------------------------------------
        web_compose = yaml.safe_load(
            (repo / BUILD_DIRNAME / "docker-compose.web.yml").read_text(encoding="utf-8")
        )
        expected_mount = f"./{BUNDLE_REL}:{CONTAINER_BUNDLE_DIR}"
        for user in ("alice", "bob"):
            service = web_compose["services"][f"web-{user}"]
            assert expected_mount in service["volumes"], (
                f"web-{user} does not bind the deployment's bundle: "
                f"expected {expected_mount!r} in {service['volumes']!r}"
            )
            assert service.get("group_add") == [str(bundle_gid)], (
                f"web-{user} was rendered without the shared corpus group. Expected "
                f"group_add == ['{bundle_gid}'] (the bundle directory's own gid), got "
                f"{service.get('group_add')!r}. setgid gives a new file its group; only "
                "group_add makes the container a MEMBER of that group, and without it "
                "cross-container sharing holds only where the deploying user's uid "
                "happens to equal the image's."
            )

        # --------------------------------------------------------------
        # 3 — and the RUNNING container is genuinely in that group. This is
        # the half a naive write-then-read test cannot reach: every persona
        # image in this module runs the same uid, so a read would succeed by
        # coincidence with group_add deleted.
        #
        # Two probes, because they fail for different reasons. The runtime
        # config says compose passed the group down at all; `id -G` says a
        # process inside actually carries it.
        # --------------------------------------------------------------
        host_group_add = _runtime_cli(
            "inspect",
            "--type",
            "container",
            "-f",
            "{{json .HostConfig.GroupAdd}}",
            _web_container("alice"),
            timeout=20,
        )
        assert host_group_add.returncode == 0, _fmt("inspect alice's GroupAdd", host_group_add)
        assert str(bundle_gid) in (json.loads(host_group_add.stdout) or []), (
            f"alice's container was created without the shared corpus group {bundle_gid}: "
            f"HostConfig.GroupAdd is {host_group_add.stdout.strip()}. The rendered "
            "group_add never reached the container runtime."
        )

        # Probed as `dispatch`, NOT as the image's default user, and the choice
        # is load-bearing rather than stylistic: alpine's root already carries
        # gids 0-27 (20 among them), so on a host whose bundle group happens to
        # fall in that range a root probe would report membership the IMAGE
        # granted and pass with group_add deleted. `dispatch` -- the
        # unprivileged user Dockerfile.web_terminal_stub creates, and the one a
        # real persona image runs its agent as -- carries only its own primary
        # group, so a gid that shows up beside it was granted by group_add.
        #
        # That reasoning has one hole, and it is checked rather than assumed:
        # `adduser -D dispatch` takes the first free gid, which is 1000, and a
        # Linux deploying user whose primary group is also 1000 -- the common
        # shape on the runners CI uses -- makes the bundle directory gid 1000
        # too. The probe would then be a tautology. Asserting the two differ
        # turns that into a loud failure naming the collision, instead of a leg
        # that quietly stops testing anything. The remedy is to give the bundle
        # directory a different group, not to delete the check.
        dispatch_gid = _runtime_cli(
            "exec", "-u", "dispatch", _web_container("alice"), "id", "-g", timeout=20
        )
        assert dispatch_gid.returncode == 0, _fmt("id -g as dispatch", dispatch_gid)
        assert str(bundle_gid) != dispatch_gid.stdout.strip(), (
            f"this probe would be vacuous: the shared corpus gid ({bundle_gid}) is "
            f"dispatch's own primary group inside the image, so 'id -G' would report it "
            "with or without group_add. chgrp the bundle directory to a group the image "
            "does not already grant, then re-run."
        )

        groups = _runtime_cli("exec", "-u", "dispatch", _web_container("alice"), "id", "-G")
        assert groups.returncode == 0, _fmt("id -G as dispatch in alice's container", groups)
        assert str(bundle_gid) in groups.stdout.split(), (
            f"alice's container is not a member of the shared corpus group {bundle_gid}: "
            f"'id -G' as dispatch reported {groups.stdout.strip()!r}. The rendered "
            "group_add did not reach the running process, so the bundle's group bits "
            "grant it nothing."
        )

        # --------------------------------------------------------------
        # 4 — the sidecar holds the SAME directory, read-only
        # --------------------------------------------------------------
        assert _container_id(QMD_CONTAINER) is not None, (
            f"{QMD_CONTAINER} not created by 'osprey up' -- the co-deployed services "
            "stack did not start"
        )
        corpus_mounts = [
            mount
            for mount in _container_mounts(QMD_CONTAINER)
            if mount.get("Destination") == SIDECAR_CORPUS_TARGET
        ]
        assert corpus_mounts, (
            f"{QMD_CONTAINER} has no mount at {SIDECAR_CORPUS_TARGET}; "
            f"mounts were {_container_mounts(QMD_CONTAINER)!r}"
        )
        assert Path(corpus_mounts[0].get("Source", "")).resolve() == bundle_dir.resolve(), (
            "the sidecar indexes a different directory than the web terminals write: "
            f"{corpus_mounts[0].get('Source')!r} vs {bundle_dir}"
        )
        assert corpus_mounts[0].get("RW") is False, (
            f"{QMD_CONTAINER} holds the corpus read-WRITE; the sidecar indexes this "
            "tree and must never be able to modify it"
        )
        sidecar_write = _runtime_cli(
            "exec", QMD_CONTAINER, "sh", "-c", f"touch {SIDECAR_CORPUS_TARGET}/nope", timeout=20
        )
        assert sidecar_write.returncode != 0, (
            f"the sidecar was able to write into {SIDECAR_CORPUS_TARGET}; the ':ro' "
            "mount is not in force"
        )

        # --------------------------------------------------------------
        # 5 — a file written from alice's container is readable from bob's
        # --------------------------------------------------------------
        # Hyphens only, never underscores: qmd normalises `_` to `-` in every
        # path it reports, so an underscored name would come back from the
        # query below spelled differently than it is on disk.
        token = f"osprey{uuid.uuid4().hex[:10]}"
        draft_name = f"alice-draft-{token}.md"
        draft_body = (
            f"# Sextupole chromaticity survey {token}\n\n"
            f"The {token} survey records the chromaticity measured after each "
            f"sextupole family change. Operators reference {token} when a tune "
            f"footprint moves without a corresponding lattice edit.\n"
        )
        write = _runtime_cli(
            "exec",
            _web_container("alice"),
            "sh",
            "-c",
            f"cat > {CONTAINER_BUNDLE_DIR}/{draft_name} <<'DRAFTEOF'\n{draft_body}DRAFTEOF",
            timeout=30,
        )
        assert write.returncode == 0, _fmt("write a draft from alice's container", write)

        read_back = _runtime_cli(
            "exec", _web_container("bob"), "cat", f"{CONTAINER_BUNDLE_DIR}/{draft_name}", timeout=20
        )
        assert read_back.returncode == 0, _fmt("read alice's draft from bob's container", read_back)
        assert token in read_back.stdout, (
            f"bob's container read the draft but not its content: {read_back.stdout!r}"
        )

        # The umask limit, asserted rather than assumed: setgid fixes group
        # OWNERSHIP, not the permission bits, so the draft is group-readable
        # and not group-writable. This test claims read+index, never overwrite.
        draft_mode = stat.S_IMODE((bundle_dir / draft_name).stat().st_mode)
        assert draft_mode & stat.S_IRGRP, (
            f"{draft_name} landed without group read ({draft_mode:04o}); neither the "
            "other terminal nor the sidecar could see it"
        )

        # --------------------------------------------------------------
        # 6 — and the sidecar indexes it, end to end over its endpoint
        # --------------------------------------------------------------
        _wait_for_qmd_health(QMD_READY_TIMEOUT_SEC)
        client = QMDClient(QMDServiceConfig(port=QMD_PORT))
        assert client.is_available(), (
            "qmd sidecar answered /health but QMDClient reports it unavailable at "
            f"{client.base_url}"
        )

        # Touch the corpus marker, exactly as a bundle writer does, so the
        # sidecar re-indexes within a poll instead of waiting out its sweep.
        marker = _runtime_cli(
            "exec",
            _web_container("alice"),
            "touch",
            f"{CONTAINER_BUNDLE_DIR}/{QMD_TOUCH_MARKER}",
            timeout=20,
        )
        assert marker.returncode == 0, _fmt("touch the corpus marker from alice", marker)

        hits = _wait_for_indexed(client, token, draft_name, QMD_REINDEX_TIMEOUT_SEC)
        logs = _runtime_cli("logs", "--tail", "30", QMD_CONTAINER, timeout=30)
        assert any(hit.file == draft_name for hit in hits), (
            f"the sidecar never indexed {draft_name}, written from alice's container "
            f"into the shared bundle. A query for {token!r} in collection "
            f"{QMD_OKF_COLLECTION!r} was still answering with "
            f"{[hit.file for hit in hits]!r} after {QMD_REINDEX_TIMEOUT_SEC:.0f}s.\n"
            f"--- {QMD_CONTAINER} logs (tail) ---\n{logs.stdout}\n{logs.stderr}"
        )
        assert all(hit.collection == QMD_OKF_COLLECTION for hit in hits), (
            "a hit came back under an unexpected collection: "
            f"{[(hit.collection, hit.file) for hit in hits]!r}"
        )

    finally:
        # Exact-named, same guardrail as every other teardown in this file: one
        # named container for the sidecar, then the module's own sweep (which
        # is exact names plus the project-scoped `compose down`).
        _runtime_cli("rm", "-f", QMD_CONTAINER)
        _teardown_all_project_resources()
