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
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from osprey.deployment.compose_generator import resolve_user_volume_names
from osprey.deployment.staleness import BUILD_DIRNAME
from osprey.deployment.web_terminals.personas import normalize_users
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
