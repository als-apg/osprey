"""E2E: the container privilege split, over both shipped topologies.

Every assertion here is made against a **real image built from a real
``osprey build`` render**, because the split is a filesystem fact and nothing
short of the built image can report it. A unit test can prove the Dockerfile
template emits a ``chown`` line; only ``stat`` inside the container proves the
line landed on the file it names, in the layer order that makes it stick.

Two renders of the ``control-assistant`` preset are built (one ``osprey init``
+ ``osprey build``, which renders the deployment's own project AND one per
persona delta):

* the **base render** — the single-user, restricted tier. Its permission render
  denies ``setup_patch``, so the Dockerfile's ``is_setup_patch_capable`` branch
  is absent and ``build/config.yml`` stays root-owned.
* the **admin render** (``<repo>-admin``) — the one tier that lifts the floor.
  Its ``build/config.yml`` is chowned to the agent's user, which is what keeps
  the Config panel's write alive without opening the rest of the render zone.

What each group pins:

``TestReadOnlyRenderZone`` — the render zone is the agent's ceiling. The
config the agent runs under is root-owned and unwritable by ``osprey``; the
state zone (``var/audit``) it must write is writable; the server the image
actually starts runs as uid 1000; and the entrypoint demonstrably ran and
dropped privileges before it did.

``TestAdminTierConfigWrites`` — the admin tier's config write across the
split: the agent's user really can rewrite the one file this tier hands it
(through the framework's own writer, not ``touch``), and a container restart
is what re-renders ``.claude/settings.json`` from it — as root, in the
entrypoint, into a tree the serving process cannot write.

``TestScaffoldRestoreReservedGate`` — a poisoned ownership record on the
agent-writable claude-config volume must not become a file in the render zone.
One test pins the gate itself; the other pins the shipped entrypoint path.

**This suite found two real defects in the built image, and now pins their
fixes.** Both were carried here as strict xfails until they were fixed:

1. ``PATCH /api/config`` 500'd in the shipped admin image
   (``Permission denied: '/app/<project>/build/var'``), so the admin Config
   panel could not save at all. The relocated backup was anchored on the RENDER
   dir while ``agent_data.base_dir`` is rendered relative to the repo ROOT,
   which is where the Dockerfile creates and chowns the zone. The backup scheme
   now derives its anchor from the config file itself
   (``config_writer.config_backup_path`` → ``repo_root_for_config``), so no
   caller can hand it the render.
2. The entrypoint's scaffold restore was a no-op, because running it as root
   flipped ``resolve_ownership`` from ``VOLUME`` (what the app sees as uid
   1000) to ``PROFILE`` — the durable store was never read, and the
   reserved-path gate never ran on the root path it was written to guard.
   ``resolve_ownership`` now skips the baked profile outright on a container
   render (``OSPREY_RENDER_ZONE_READONLY=1``), so root and the server resolve
   the same surface.

Neither is xfailed any more; ``test_config_patch_saves_and_backs_up_into_the_state_zone``
and ``test_the_entrypoint_restore_refuses_a_reserved_record`` are the pins.

``TestMultiUserSeeding`` — the per-user seeding target and its owners: skills
land in the render zone at ``<container_project_dir>/build/.claude/skills``
(the only scope the launcher's ``--setting-sources project`` actually loads),
root-owned so a session cannot edit the skills the next session reads, while
the claude-config volume is handed to the runtime uid.

Cost discipline: TWO images, built from one shared ``osprey init`` + ``osprey
build``, as module-scoped fixtures. Everything above the deployment ``COPY``
is byte-identical between the two renders, so the second build reuses the
first's layer cache and costs a ``COPY`` plus two ``RUN``s.

The images are built for the HOST platform, unlike ``test_dockerfile_e2e.py``
which pins linux/amd64 on arm64 hosts. That pin is for a dependency-wheel
availability problem; these tests assert file ownership and uids, which are
architecture-independent, and a native build is what ``osprey up`` produces on
the same host — emulating a second architecture would only add a cold layer
cache to a suite that already builds two images.

Skipped entirely when docker is unavailable.
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
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.deployment.web_terminals.naming import web_container_name
from osprey.deployment.web_terminals.seeding import seed_user_containers
from osprey.deployment.wheel_build import _copy_local_framework_for_override
from osprey.utils.workspace import container_image_context


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.dockerbuild,
    pytest.mark.skipif(not _docker_available(), reason="docker binary or daemon not available"),
]

PRESET = "control-assistant"
REPO_NAME = "psplit"
BUILD_TIMEOUT = 3600  # a cold build compiles the EPICS toolchain from source
RUN_TIMEOUT = 300
HEALTH_TIMEOUT = 420  # osprey web boots the full MCP/ARIEL stack before binding

#: Tag prefix for every image and container this module creates, so a stray
#: left by a killed run is recognizable (and greppable) rather than anonymous.
TAG_PREFIX = "osprey-e2e-privsplit"

#: The claude-config volume mount point, and the env var whose presence IS the
#: mount as far as ``ownership._volume_store`` is concerned.
CLAUDE_CONFIG_DIR = "/data/claude-config"

#: A reserved path that is NOT in ``ownership.generated_project_paths()`` — so
#: it passes the store's own path filter and reaches the reserved-set gate,
#: which is the gate under test. A local settings overlay is the sharpest case:
#: Claude Code reads it, so a restored one silently widens the permissions the
#: build rendered.
RESERVED_OUTPUT_PATH = ".claude/settings.local.json"

#: The canonical NAME the poisoned record is filed under. ``rehydrate`` returns
#: names, not output paths (``restored.append(name)``), so an assertion phrased
#: against :data:`RESERVED_OUTPUT_PATH` would hold no matter what the restore
#: did — the list can never contain a path. Spelled once and used at both the
#: plant and the assertions so the two cannot drift into that tautology again.
RESERVED_NAME = "settings-local"

#: A NON-reserved, ownable artifact used to prove the positive half of the
#: entrypoint restore: a claimed body has to come back into the render zone. The
#: data-visualizer agent, deliberately not the channel-finder one, because
#: :data:`DRIFT_KEY` disables channel-finder and the restart's regen therefore
#: reshapes that artifact — a restore probe has to sit outside what the regen
#: in front of it is rewriting.
RESTORED_NAME = "agents/data-visualizer"
RESTORED_OUTPUT_PATH = ".claude/agents/data-visualizer.md"
RESTORED_BODY = "---\nname: data-visualizer\n---\n\nEdited by the operator.\n"

#: An unprotected config key that provably re-renders ``.claude/settings.json``:
#: disabling an agent drops its ``Task(...)`` entry from the rendered permission
#: surface. It has to be BOTH — a protected key is refused 403 before any write,
#: and a key that touches no artifact would leave the restart with nothing to
#: regenerate and the test asserting on a no-op.
DRIFT_KEY = "claude_code.agents.channel-finder.enabled"


# ── docker helpers ───────────────────────────────────────────────────────────


def _docker(*args: str, timeout: int = RUN_TIMEOUT, stdin: bytes | None = None):
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=stdin is None,
        input=stdin,
        timeout=timeout,
    )


def _exec(cid: str, *cmd: str, user: str = "0", env: dict[str, str] | None = None):
    """``docker exec`` as *user* (root by default — the image has no USER)."""
    env_args: list[str] = []
    for key, value in (env or {}).items():
        env_args += ["-e", f"{key}={value}"]
    return _docker("exec", "-u", user, *env_args, cid, *cmd)


def _exec_python(cid: str, script: str, user: str = "0", env: dict[str, str] | None = None):
    """Run *script* through the image's own python, inside the container.

    The framework's modules are the source of truth for the store format and
    the ownership rules, so the probes below drive them rather than
    reimplementing a record layout a refactor could silently move.
    """
    env_args: list[str] = []
    for key, value in (env or {}).items():
        env_args += ["-e", f"{key}={value}"]
    return subprocess.run(
        ["docker", "exec", "-i", "-u", user, *env_args, cid, "python", "-"],
        capture_output=True,
        text=True,
        input=script,
        timeout=RUN_TIMEOUT,
    )


def _logs(cid: str) -> str:
    out = _docker("logs", cid, timeout=30)
    return (out.stdout or "") + (out.stderr or "")


def _is_running(cid: str) -> bool:
    return _docker("inspect", "-f", "{{.State.Running}}", cid, timeout=30).stdout.strip() == "true"


def _host_port(cid: str) -> int:
    out = _docker("port", cid, "8087/tcp", timeout=30)
    assert out.returncode == 0, f"docker port failed: {out.stderr}"
    return int(out.stdout.strip().splitlines()[0].rsplit(":", 1)[1])


def _wait_for_health(base_url: str, cid: str, timeout: float = HEALTH_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    last = "no response"
    while time.monotonic() < deadline:
        if not _is_running(cid):
            pytest.fail(f"container exited before becoming healthy:\n{_logs(cid)[-4000:]}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200 and json.loads(resp.read()).get("status") == "healthy":
                    return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = str(exc)
        time.sleep(1.0)
    pytest.fail(f"server never became healthy ({last}):\n{_logs(cid)[-4000:]}")


# The launcher prints its one-time login URL once at startup; the container
# mints its OWN operator secret, so the token exists nowhere but its log.
_OPEN_URL_RE = re.compile(r"Open:\s*(http://\S+\?token=\S+)")


def _login_opener(base_url: str, cid: str) -> urllib.request.OpenerDirector:
    """Exchange the container's printed ``?token=`` URL for a session cookie.

    Mirrors ``test_dockerfile_e2e._login_opener``: every ``/api/*`` route is
    behind the web-auth gate, and a browser gets in by following the announced
    URL, which sets an HttpOnly session cookie.
    """
    match = _OPEN_URL_RE.search(_logs(cid))
    assert match, f"container never printed an 'Open: …?token=…' URL:\n{_logs(cid)[-4000:]}"
    query = urllib.parse.urlparse(match.group(1)).query
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    with opener.open(f"{base_url}/?{query}", timeout=15) as resp:
        assert resp.status == 200, f"token exchange returned {resp.status}"
    return opener


def _api_patch(base_url: str, opener, path: str, body: dict, timeout: float = 60):
    """PATCH a JSON body to *path*; return ``(status, parsed_body)``.

    The same-origin ``Origin`` header is required, not decorative: ``/api/*``
    mutations are cookie-authenticated, so the gate refuses one without it 403.
    """
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Origin": base_url},
        method="PATCH",
    )
    try:
        with opener.open(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()[:2000]


# ── the shared render + the two images ───────────────────────────────────────


@pytest.fixture(scope="module")
def preset_repo(tmp_path_factory) -> Path:
    """One ``osprey init`` + ``osprey build`` of the preset, shared by both images.

    ``--no-git`` because nothing reads history and the tmp tree may sit inside
    another repository; ``--skip-deps``/``--skip-lifecycle`` because the images
    install their own dependencies and no service is started here.
    """
    repo = tmp_path_factory.mktemp("privilege-split") / REPO_NAME
    runner = CliRunner()
    created = runner.invoke(cli, ["init", str(repo), "--preset", PRESET, "--no-git"])
    assert created.exit_code == 0, created.output
    built = runner.invoke(cli, ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert built.exit_code == 0, built.output
    return repo


@pytest.fixture(scope="module")
def rendered_config(preset_repo: Path) -> dict:
    return yaml.safe_load((preset_repo / "build" / "config.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def admin_project(rendered_config: dict) -> str:
    """The admin persona's project name, read from the render's own catalog.

    Derived rather than spelled: the catalog is what ``osprey build`` renders a
    project per, so a hand-written ``<repo>-admin`` here would be a second
    source of truth that a naming change could silently desynchronize.
    """
    catalog = rendered_config["modules"]["web_terminals"]["personas"]
    return catalog["admin"]["project"]


def _build_image(context: Path, tag: str) -> None:
    """Stage the local dev wheel into *context* and build it.

    The wheel is not optional. The Dockerfile's default ``OSPREY_PIP_SPEC`` is
    the PyPI release, and everything under test here — the reserved-path gate,
    the config-backup relocation, the Config-panel gate, the entrypoint itself
    — is unreleased. Without the staged wheel these tests would build an image
    that does not contain the code they claim to be testing and report green.
    Staged through the production helper (``osprey up --dev``'s own path), so
    the framework AND the connectors workspace member both land.
    """
    staged = _copy_local_framework_for_override(str(context))
    assert staged, f"could not stage the local dev wheel into {context}"
    build = subprocess.run(
        [
            "docker",
            "build",
            "-f",
            str(context / "build" / "Dockerfile"),
            "-t",
            tag,
            "--build-arg",
            "OSPREY_DEV=1",
            "--label",
            f"com.osprey.project={context.name}",
            ".",
        ],
        cwd=context,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT,
        env={**os.environ, "DOCKER_BUILDKIT": "1"},
    )
    assert build.returncode == 0, (
        f"docker build failed for {tag}:\n--- stdout ---\n{build.stdout[-4000:]}"
        f"\n--- stderr ---\n{build.stderr[-4000:]}"
    )


@pytest.fixture(scope="module")
def ro_image(preset_repo: Path, rendered_config: dict):
    """The base (restricted) render's image. Yields ``(tag, project_name)``."""
    project = rendered_config["project_name"]
    tag = f"{TAG_PREFIX}-ro:{uuid.uuid4().hex[:8]}"
    try:
        _build_image(container_image_context(preset_repo, project), tag)
        yield tag, project
    finally:
        _docker("rmi", "-f", tag, timeout=120)


@pytest.fixture(scope="module")
def admin_image(preset_repo: Path, admin_project: str):
    """The admin render's image. Yields ``(tag, project_name)``."""
    tag = f"{TAG_PREFIX}-admin:{uuid.uuid4().hex[:8]}"
    try:
        _build_image(container_image_context(preset_repo, admin_project), tag)
        yield tag, admin_project
    finally:
        _docker("rmi", "-f", tag, timeout=120)


def _run_web(tag: str, *extra: str) -> str:
    """Start a container on the image's own CMD, port published on loopback."""
    name = f"{TAG_PREFIX}-{uuid.uuid4().hex[:8]}"
    run = _docker("run", "-d", "--name", name, "-p", "127.0.0.1:0:8087", *extra, tag, timeout=120)
    assert run.returncode == 0, f"docker run failed: {run.stderr}"
    return name


# ── the read-only render zone (single-user, restricted tier) ─────────────────


@pytest.fixture(scope="class")
def ro_container(ro_image):
    """The base image, running its real CMD (``osprey web``), healthy.

    The real CMD and not ``sleep infinity``: the uid assertion below is about
    the process that serves requests, and a sleeper would prove the entrypoint
    drops privileges for *something* while saying nothing about the server.

    Class-scoped, not module-scoped, so it is gone before the admin tier's own
    server starts. Each of these containers boots the deployment's full MCP and
    ARIEL stack; two of them alive at once on one machine starve each other,
    and the second one's health wait is where that shows up.
    """
    tag, project = ro_image
    cid = _run_web(tag)
    try:
        _wait_for_health(f"http://127.0.0.1:{_host_port(cid)}", cid)
        yield SimpleNamespace(cid=cid, project=project, root=f"/app/{project}")
    finally:
        _docker("rm", "-f", cid, timeout=120)


class TestReadOnlyRenderZone:
    """The restricted tier: the agent cannot rewrite what decides its limits."""

    def test_render_zone_config_is_root_owned(self, ro_container):
        """``config.yml`` is the file that says what the agent may do, and this
        tier's Dockerfile deliberately omits the chown that would hand it over."""
        stat = _exec(ro_container.cid, "stat", "-c", "%U", f"{ro_container.root}/build/config.yml")
        assert stat.returncode == 0, stat.stderr
        assert stat.stdout.strip() == "root"

    def test_agent_user_cannot_write_the_render_zone_config(self, ro_container):
        """Ownership is only the mechanism; refusal is the property. Asked as
        the agent's own user, which is what the running server actually is."""
        touch = _exec(
            ro_container.cid, "gosu", "osprey", "touch", f"{ro_container.root}/build/config.yml"
        )
        assert touch.returncode != 0, "the agent's user could write the render zone's config.yml"
        assert "Permission denied" in touch.stderr

    def test_agent_user_can_write_the_audit_zone(self, ro_container):
        """The other half, and the one a too-broad chown would break: a refusal
        nobody can record is a refusal only the agent's own transcript saw."""
        probe = f"{ro_container.root}/var/audit/e2e-privsplit-probe"
        touch = _exec(ro_container.cid, "gosu", "osprey", "touch", probe)
        assert touch.returncode == 0, (
            f"the agent's user cannot write the audit zone:\n{touch.stderr}"
        )

    def test_the_served_process_runs_as_the_runtime_uid(self, ro_container):
        """``osprey web`` — the image's CMD — runs as uid 1000, not as root.

        The uid is asserted numerically because that is the contract the
        outside world agrees with: ``OSPREY_RUNTIME_UID`` states the pair and
        the host-side volume seeding chowns to what it says.
        """
        ps = _exec(ro_container.cid, "ps", "-o", "uid=,args=", "-e")
        assert ps.returncode == 0, ps.stderr
        served = [line for line in ps.stdout.splitlines() if "osprey web" in line]
        assert served, f"no `osprey web` process in the container:\n{ps.stdout}"
        assert all(line.split()[0] == "1000" for line in served), (
            f"the served process is not the osprey uid:\n{ps.stdout}"
        )

    def test_the_entrypoint_ran_and_dropped_privileges(self, ro_container):
        """There is no ``USER`` line in this image — the drop is the
        entrypoint's doing, and its absence would be silent without this."""
        logs = _logs(ro_container.cid)
        assert "[osprey-entrypoint] starting:" in logs, logs[-4000:]
        assert "dropping privileges to the osprey user" in logs, logs[-4000:]


# ── the admin tier: config writes survive the split ──────────────────────────


@pytest.fixture(scope="module")
def admin_state(admin_image):
    """Drive the admin tier's whole config-write lifecycle once, and snapshot it.

    One container, one restart, four assertions — because the restart is the
    expensive step and every assertion below is about a different property of
    the *same* transition. Each test reads one field of the snapshot, so a
    failure still names exactly which property broke.

    The claude-config volume is mounted (and ``CLAUDE_CONFIG_DIR`` set) the way
    the multi-user compose service mounts it, because the reserved-record probe
    below needs the durable store that only exists when it is there.
    """
    tag, project = admin_image
    root = f"/app/{project}"
    render = f"{root}/build"
    settings = f"{render}/.claude/settings.json"
    volume = f"{TAG_PREFIX}-cfg-{uuid.uuid4().hex[:8]}"

    cid = _run_web(
        tag,
        "-v",
        f"{volume}:{CLAUDE_CONFIG_DIR}",
        "-e",
        f"CLAUDE_CONFIG_DIR={CLAUDE_CONFIG_DIR}",
    )
    try:
        base_url = f"http://127.0.0.1:{_host_port(cid)}"
        _wait_for_health(base_url, cid)

        before = _exec(cid, "cat", settings)
        assert before.returncode == 0, before.stderr

        opener = _login_opener(base_url, cid)
        status, body = _api_patch(base_url, opener, "/api/config", {"updates": {DRIFT_KEY: False}})
        backup = f"{root}/var/agent_data/config-backups/config.yml.bak"
        backup_owner = _exec(cid, "stat", "-c", "%U", backup)

        # The same write, driven the way the tier's OTHER shipped writer does
        # it: the framework's own config writer, as the agent's user, against
        # the config.yml this tier's image handed to that user. Done
        # unconditionally so the restart-regen assertion below is testing the
        # entrypoint rather than the HTTP surface — they are separate
        # properties and one of them was, for a while, broken on its own.
        wrote = _exec_python(
            cid,
            "from pathlib import Path\n"
            "from osprey.utils.config_writer import config_update_fields\n"
            f"config_update_fields(Path({root!r}) / 'build' / 'config.yml',"
            f" {{{DRIFT_KEY!r}: False}})\n"
            "print('WROTE')\n",
            user="1000",
            env={"HOME": "/home/osprey"},
        )

        # The claude-config volume is root-owned until its first chown; the real
        # deployment's per-user seeding hands it to the runtime uid before the
        # agent ever touches it (see TestMultiUserSeeding). Do the same here, or
        # the store below is written by root and its 0600 index is unreadable to
        # the user that actually owns it — a test artifact that would look
        # exactly like the gate never firing.
        chowned = _exec(cid, "chown", "-R", "1000:1000", CLAUDE_CONFIG_DIR)
        assert chowned.returncode == 0, chowned.stderr

        # Plant a poisoned record on the volume BEFORE the restart, driving the
        # store's own API so the record is exactly what the gallery would write,
        # as the user the gallery runs as.
        # TWO records, because the restore has two properties and one of them
        # can be satisfied by doing nothing at all: a poisoned record that must
        # be skipped, and an ordinary claimed body that must be installed. A
        # restore that resolved the wrong ownership surface — the defect this
        # pins — passes the first on its own and fails only the second.
        plant = _exec_python(
            cid,
            "from pathlib import Path\n"
            "from osprey.interfaces.web_terminal.ownership import OwnershipStore\n"
            f"store = OwnershipStore(root=Path({CLAUDE_CONFIG_DIR!r}) / 'osprey' / 'scaffold')\n"
            f"store.claim({RESERVED_NAME!r}, {RESERVED_OUTPUT_PATH!r},"
            ' \'{"permissions": {"allow": ["Bash(*)"]}}\\n\')\n'
            f"store.claim({RESTORED_NAME!r}, {RESTORED_OUTPUT_PATH!r}, {RESTORED_BODY!r})\n"
            "print(sorted(store.read()))\n",
            user="1000",
            env={"CLAUDE_CONFIG_DIR": CLAUDE_CONFIG_DIR, "HOME": "/home/osprey"},
        )
        assert plant.returncode == 0, f"{plant.stdout}\n{plant.stderr}"

        restart = _docker("restart", cid, timeout=180)
        assert restart.returncode == 0, restart.stderr
        # A restart re-publishes the ephemeral port: `-p 127.0.0.1:0:8087` is
        # resolved per start, so the pre-restart URL is refused forever.
        base_url = f"http://127.0.0.1:{_host_port(cid)}"
        _wait_for_health(base_url, cid)

        after = _exec(cid, "cat", settings)
        # The root maintenance phase runs under the writer's ``maintenance``
        # marker, so everything the entrypoint's restore records lands in
        # ``var/audit/<identity>/maintenance.jsonl`` -- a file the app user
        # never appends to, which is how the ledger keeps one uid per file. The
        # identity directory is globbed rather than named: it is whatever the
        # container's identity resolves to.
        audit = _exec(cid, "sh", "-c", f"cat {root}/var/audit/*/maintenance.jsonl")
        installed = _exec(cid, "test", "-e", f"{render}/{RESERVED_OUTPUT_PATH}")
        audit_owner = _exec(cid, "sh", "-c", f"stat -c '%U %G %a' {root}/var/audit/*/ | head -1")
        restored_body = _exec(cid, "cat", f"{render}/{RESTORED_OUTPUT_PATH}")
        restored_owner = _exec(cid, "stat", "-c", "%U", f"{render}/{RESTORED_OUTPUT_PATH}")

        yield SimpleNamespace(
            cid=cid,
            project=project,
            root=root,
            patch_status=status,
            patch_body=body,
            wrote=wrote,
            backup_path=backup,
            backup_owner=backup_owner,
            settings_before=before.stdout,
            settings_after=after.stdout,
            plant=plant,
            audit=audit,
            reserved_installed=installed.returncode == 0,
            audit_owner=audit_owner,
            restored_body=restored_body,
            restored_owner=restored_owner,
            logs=_logs(cid),
        )
    finally:
        _docker("rm", "-f", cid, timeout=120)
        _docker("volume", "rm", "-f", volume, timeout=60)


class TestAdminTierConfigWrites:
    """The tier that keeps the setup capability, and what the split costs it."""

    def test_admin_image_hands_config_to_the_agent_user(self, admin_image):
        """The conditional chown, observed on the file rather than in the
        template — and paired with the write that is its whole purpose, since
        an owner without a successful write proves only half of it.

        Run through the image's real ENTRYPOINT with no ``--user`` and no
        ``gosu``: the entrypoint has already dropped to ``osprey`` by the time
        the command runs, so this is the agent's own identity writing its own
        config, which is exactly the capability the tier is defined by.
        """
        tag, project = admin_image
        config = f"/app/{project}/build/config.yml"
        run = _docker(
            "run",
            "--rm",
            tag,
            "sh",
            "-c",
            f'echo "USER=$(id -un)"; echo "OWNER=$(stat -c %U {config})"; '
            f"touch {config} && echo WROTE",
            timeout=RUN_TIMEOUT,
        )
        assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
        assert "USER=osprey" in run.stdout, run.stdout
        assert "OWNER=osprey" in run.stdout, (
            f"the admin render's config.yml is not owned by the agent's user:\n{run.stdout}"
        )
        assert "WROTE" in run.stdout, (
            f"the agent's user could not write the admin render's config.yml:\n{run.stdout}"
        )

    def test_the_agent_user_can_write_its_own_render_config(self, admin_state):
        """The capability the tier is defined by, exercised through the
        framework's own writer rather than ``touch``: ``config_update_fields``
        is what ``setup_patch`` and the Config panel both reach disk through,
        and it has to work against a config.yml that sits in a directory the
        agent's user does NOT own."""
        assert admin_state.wrote.returncode == 0, (
            f"{admin_state.wrote.stdout}\n{admin_state.wrote.stderr}"
        )
        assert "WROTE" in admin_state.wrote.stdout

    def test_config_patch_saves_and_backs_up_into_the_state_zone(self, admin_state):
        """The admin panel's save, end to end — 200, backup, and regen report.

        Asserted as one test because they are one behaviour: the backup is
        taken BEFORE the mutation, so a backup directory the agent's user
        cannot write fails the save outright, and the reported regen skip is
        what tells the operator the write needs a restart to take effect.
        """
        assert admin_state.patch_status == 200, admin_state.patch_body
        assert admin_state.patch_body["status"] == "ok"
        assert admin_state.patch_body["fields_updated"] == 1

        assert admin_state.backup_owner.returncode == 0, (
            f"no config backup at {admin_state.backup_path}: {admin_state.backup_owner.stderr}"
        )
        assert admin_state.backup_owner.stdout.strip() == "osprey"

        # `regenerated: []` alone is also what a render with nothing to do
        # returns, so the reported reason is the load-bearing half: without it
        # an operator who just changed a render-shaping key reads "already in
        # effect".
        assert admin_state.patch_body["regenerated"] == []
        assert admin_state.patch_body.get("detail"), admin_state.patch_body

    def test_restart_re_renders_settings_json_from_the_written_config(self, admin_state):
        """The other end of that skip: the entrypoint does it, as root, once.

        This is the property the whole split rests on — the render zone stays
        unwritable by the agent AND still tracks config.yml — so it is asserted
        on the file's bytes, plus the entrypoint's own report of what it
        re-rendered.
        """
        assert admin_state.settings_after != admin_state.settings_before, (
            "settings.json did not change across the restart; the entrypoint "
            f"regen did not pick up the config write:\n{admin_state.logs[-4000:]}"
        )
        assert "regenerated" in admin_state.logs and ".claude/settings.json" in admin_state.logs, (
            f"the entrypoint did not report re-rendering settings.json:\n{admin_state.logs[-4000:]}"
        )
        before_allow = json.loads(admin_state.settings_before)["permissions"]["allow"]
        after_allow = json.loads(admin_state.settings_after)["permissions"]["allow"]
        assert "Task(channel-finder)" in before_allow, (
            "the drift key did not shape the image's rendered permission "
            "surface, so this test would pass on a render that never changed"
        )
        assert "Task(channel-finder)" not in after_allow, (
            "the disabled agent is still in the rendered permission surface"
        )


# ── the reserved-path gate on the scaffold restore ───────────────────────────


class TestScaffoldRestoreReservedGate:
    """A record on the agent-writable volume must not become a render-zone file."""

    def test_a_reserved_record_is_refused_and_audited(self, admin_state):
        """The gate itself, exercised in the real image as the agent's user.

        Run as uid 1000 because that is the identity the store belongs to and
        the one that resolves ``VOLUME`` ownership. The refusal is asserted in
        three places at once — nothing installed, nothing returned, and a
        durable audit line — because a gate that refuses silently is a gate an
        operator learns about only from the file that is missing.

        The audit assertion is a DELTA, taken inside the probe around its own
        restore call. The log is append-only and the root entrypoint already
        wrote a matching record into it at container start, so "a matching line
        exists in the file" would pass here even if this call audited nothing at
        all. Only the lines this restore appended are evidence about it.
        """
        probe = _exec_python(
            admin_state.cid,
            "import json\n"
            "from pathlib import Path\n"
            "from osprey.interfaces.web_terminal.scaffold_gallery_service import (\n"
            "    restore_scaffold_bodies,\n"
            ")\n"
            "from osprey.interfaces.web_terminal.ownership import resolve_ownership\n"
            f"render = Path({admin_state.root!r}) / 'build'\n"
            "from osprey.audit.protected import SURFACE_SCAFFOLD_RESTORE\n"
            "from osprey.audit.writer import ledger_path\n"
            "from osprey.utils.identity import acting_identity\n"
            f"audit = (Path({admin_state.root!r}) / 'var' / 'audit' / acting_identity()\n"
            "         / f'{SURFACE_SCAFFOLD_RESTORE}.jsonl')\n"
            "read = lambda: audit.read_text().splitlines() if audit.exists() else []\n"
            "before = len(read())\n"
            "DIAG = str(ledger_path(SURFACE_SCAFFOLD_RESTORE))\n"
            "mode = resolve_ownership(render).mode.value\n"
            "restored = restore_scaffold_bodies(render)\n"
            "print(json.dumps({\n"
            "    'mode': mode,\n"
            "    'restored': restored,\n"
            f"    'installed': (render / {RESERVED_OUTPUT_PATH!r}).exists(),\n"
            "    'appended': [json.loads(line) for line in read()[before:] if line.strip()],\n"
            "    'diag_log': DIAG,\n"
            "}))\n",
            user="1000",
            env={"CLAUDE_CONFIG_DIR": CLAUDE_CONFIG_DIR, "HOME": "/home/osprey"},
        )
        assert probe.returncode == 0, f"{probe.stdout}\n{probe.stderr}"
        result = json.loads(probe.stdout.strip().splitlines()[-1])

        assert result["mode"] == "volume", (
            "the durable store was not the ownership surface, so the record "
            f"under test was never consulted: {result}"
        )
        assert result["installed"] is False, "a reserved path was installed from the store"
        assert RESERVED_NAME not in result["restored"], result

        matching = [
            r
            for r in result["appended"]
            if r.get("surface") == "scaffold_restore" and r.get("subject") == RESERVED_OUTPUT_PATH
        ]
        assert matching, (
            f"this restore appended no scaffold_restore refusal: {result['appended']}; "
            f"the process writes its log to {result['diag_log']}"
        )
        assert matching[-1]["reason"] == "reserved path in ownership store"
        assert "channel=" in matching[-1]["detail"], "the refusal recorded no owning channel"

    def test_the_entrypoint_leaves_the_audit_zone_writable_by_the_runtime_user(self, admin_state):
        """Whoever creates the identity directory first decides who can file in it.

        The entrypoint's restore runs as root and is the FIRST writer into
        ``var/audit/<identity>/`` on a fresh deployment. Its own records go to
        ``maintenance.jsonl`` — root and the app user never share a file, which
        is what the writer's maintenance marker is for — but they share the
        DIRECTORY, and a root-owned 0755 directory would leave the server, as
        uid 1000, unable to create ``scaffold_restore.jsonl`` at all. The
        recorder never raises, so every later refusal would be dropped in
        silence: the exact failure the audit trail exists to prevent.

        Two shapes pass, and both are the invariant rather than a patch: the
        directory is handed back to the agent user (the state zone is theirs by
        construction — the image chowns ``var/`` wholesale), or it is the
        group-shared setgid directory a bind-mounted audit zone is provisioned
        as, which the hand-back deliberately prunes and the entrypoint joins by
        group instead.
        """
        assert admin_state.audit_owner.returncode == 0, (
            f"no audit identity directory after the restart: {admin_state.audit_owner.stderr}"
        )
        owner, _group, mode = admin_state.audit_owner.stdout.split()[:3]
        group_writable = int(mode[-2]) & 0o2 != 0
        assert owner == "osprey" or group_writable, (
            "the audit directory is root-owned with no group write, so the server "
            "cannot file a record in it and every refusal it makes is silently "
            f"lost: {admin_state.audit_owner.stdout!r}"
        )

    def test_a_body_that_symlinks_out_of_the_store_is_refused_and_audited(self, admin_state):
        """The SOURCE side of the same walk, in the real image, split by identity.

        The store's content directory is the per-user volume — the agent's own
        ``$HOME`` — so a body planted as a symlink onto a root-only file would
        have the root entrypoint read that file and write its bytes into the
        render zone, where the agent can read them.

        The two halves run as the two identities that actually do them, and the
        split is what makes the assertions mean anything. Planting as uid 1000
        is what makes it a real capability rather than a contrived one. Reading
        as ROOT is what makes ``installed``/``restored`` discriminate: as uid
        1000 the pre-fix ``read_text`` on ``/etc/shadow`` would raise
        ``PermissionError`` and return None anyway, so the test would pass
        against the unfixed code and prove nothing. Root is the identity that
        can actually complete the leak, and it is the identity the shipped
        entrypoint restore runs as.
        """
        secret = "/etc/shadow"
        plant = _exec_python(
            admin_state.cid,
            "from pathlib import Path\n"
            "from osprey.interfaces.web_terminal.ownership import OwnershipStore\n"
            f"store = OwnershipStore(root=Path({CLAUDE_CONFIG_DIR!r}) / 'osprey' / 'scaffold')\n"
            "store.claim('agents/pwn', '.claude/agents/pwn.md', 'placeholder\\n')\n"
            "body = store.content_dir / '.claude' / 'agents' / 'pwn.md'\n"
            "body.unlink()\n"
            f"body.symlink_to({secret!r})\n"
            "print('PLANTED')\n",
            user="1000",
            env={"CLAUDE_CONFIG_DIR": CLAUDE_CONFIG_DIR, "HOME": "/home/osprey"},
        )
        assert plant.returncode == 0, f"{plant.stdout}\n{plant.stderr}"
        assert "PLANTED" in plant.stdout, plant.stdout

        # Root really can read the target — otherwise the refusal below would be
        # indistinguishable from a permission error doing the work for us.
        readable = _exec(admin_state.cid, "head", "-c", "1", secret)
        assert readable.returncode == 0, (
            f"root cannot read {secret} in this image, so this test cannot "
            f"discriminate: {readable.stderr}"
        )

        probe = _exec_python(
            admin_state.cid,
            "import json\n"
            "from pathlib import Path\n"
            "from osprey.interfaces.web_terminal.scaffold_gallery_service import (\n"
            "    restore_scaffold_bodies,\n"
            ")\n"
            f"render = Path({admin_state.root!r}) / 'build'\n"
            "from osprey.audit.protected import SURFACE_SCAFFOLD_RESTORE\n"
            "from osprey.utils.identity import acting_identity\n"
            f"audit = (Path({admin_state.root!r}) / 'var' / 'audit' / acting_identity()\n"
            "         / f'{SURFACE_SCAFFOLD_RESTORE}.jsonl')\n"
            "read = lambda: audit.read_text().splitlines() if audit.exists() else []\n"
            "before = len(read())\n"
            "restored = restore_scaffold_bodies(render)\n"
            "target = render / '.claude' / 'agents' / 'pwn.md'\n"
            "print(json.dumps({\n"
            "    'restored': restored,\n"
            "    'installed': target.exists(),\n"
            "    'body': target.read_text() if target.is_file() else None,\n"
            "    'appended': [json.loads(line) for line in read()[before:] if line.strip()],\n"
            "}))\n",
            user="0",
            env={"CLAUDE_CONFIG_DIR": CLAUDE_CONFIG_DIR},
        )
        assert probe.returncode == 0, f"{probe.stdout}\n{probe.stderr}"
        result = json.loads(probe.stdout.strip().splitlines()[-1])

        assert result["installed"] is False, (
            f"root copied {secret} into the render zone, where the agent can read it: {result}"
        )
        assert "agents/pwn" not in result["restored"], result
        escapes = [
            r
            for r in result["appended"]
            if r.get("reason") == "ownership store body escapes the store"
        ]
        assert escapes, f"the refusal was not audited: {result['appended']}"
        assert escapes[-1]["surface"] == "scaffold_restore"

    def test_the_entrypoint_restore_refuses_a_reserved_record(self, admin_state):
        """The SHIPPED path: the same record, across a real container restart.

        The plant happened before the restart in the fixture, so what is read
        here is what the root entrypoint actually did with it.
        """
        assert RESERVED_NAME in admin_state.plant.stdout, (
            f"the record was never planted: {admin_state.plant.stdout}\n{admin_state.plant.stderr}"
        )
        assert admin_state.reserved_installed is False, (
            "the entrypoint installed a reserved path from the ownership store"
        )
        assert admin_state.audit.returncode == 0, (
            "the entrypoint's restore left no protected-write audit record: "
            f"{admin_state.audit.stderr}"
        )
        records = [
            json.loads(line) for line in admin_state.audit.stdout.splitlines() if line.strip()
        ]
        assert any(
            r.get("surface") == "scaffold_restore" and r.get("subject") == RESERVED_OUTPUT_PATH
            for r in records
        ), f"no scaffold_restore refusal from the entrypoint: {records}"

    def test_the_entrypoint_restore_installs_a_user_owned_body_as_root(self, admin_state):
        """The positive half, on the shipped path: the restore is not a no-op.

        For the whole of the container split's life before this, the entrypoint
        resolved its ownership surface by asking whether it could write the
        profile root — and as root, inside an image whose ``profile.yml`` really
        is there, the answer was yes. It took the PROFILE branch, never read the
        volume, and reported ``no user-owned artifact bodies to restore`` while
        the store held them. The app never made up the difference: it skips the
        restore entirely under ``OSPREY_RENDER_ZONE_READONLY=1``. So a claimed
        body was silently lost on every container recreation, and the refusal
        test above passed the entire time — nothing was installed because
        nothing was read.

        The restored file is ROOT-owned, and that is the intended posture, not
        an oversight. It lands in the render zone, whose defining property is
        that the agent's user cannot write it: an artifact the operator claimed
        must not become the one file in that tree the agent can rewrite. The
        write happens here, once, before the privilege drop, precisely so that
        the process serving requests never has the privilege to repeat it.
        """
        assert admin_state.restored_body.returncode == 0, (
            "the entrypoint restored no body for a claimed, non-reserved record; "
            f"the store held it: {admin_state.plant.stdout}\n"
            f"{admin_state.restored_body.stderr}\n{admin_state.logs[-4000:]}"
        )
        assert admin_state.restored_body.stdout == RESTORED_BODY, (
            "the render holds something other than the claimed body:\n"
            f"{admin_state.restored_body.stdout!r}"
        )
        assert "restored 1 user-owned artifact(s)" in admin_state.logs, (
            f"the entrypoint did not report the restore:\n{admin_state.logs[-4000:]}"
        )
        assert admin_state.restored_owner.stdout.strip() == "root", (
            "a restored artifact in the render zone must stay root-owned: "
            f"{admin_state.restored_owner.stdout!r}"
        )


# ── multi-user seeding: render-zone target and owners ────────────────────────


@pytest.fixture(scope="module")
def seeded_container(admin_image, preset_repo: Path, rendered_config: dict):
    """Seed one roster user's live container, exactly as ``osprey up`` does.

    The admin image is reused rather than a third one built: carol is the
    roster's admin-persona user, so her resolved ``container_project_dir`` IS
    this image's project by construction — which keeps the seeding assertions
    on the real catalog and the real roster instead of a hand-made config.

    The container is named the way the compose service names it, because
    ``seed_user_containers`` finds its target by that name and nothing else —
    so the name is made unique by giving the seeding a config whose
    ``facility.prefix`` is unique to this run. The preset pins a FIXED prefix
    (``ca``), which means the compose-dictated name ``ca-web-carol`` collides
    with any real deployment of this preset on the developer's own machine; a
    developer running one would otherwise silently lose this whole class to a
    skip. Overriding the prefix keeps both properties: the test never touches a
    container it did not create, and it always runs.

    Only carol's persona is seeded, and it names its own ``project``, so the
    prefix override cannot reach the project path — ``facility.prefix`` feeds
    ``resolve_personas`` solely as the default for a persona that names none.
    Asserted below rather than assumed.
    """
    tag, project = admin_image
    user = "carol"
    rendered_config = copy.deepcopy(rendered_config)
    web_terminals = rendered_config["modules"]["web_terminals"]
    persona = next(u for u in web_terminals["users"] if u["name"] == user)["persona"]
    assert web_terminals["personas"][persona]["project"] == project, (
        "this fixture reuses the admin image for the seeding target; the "
        "roster no longer maps carol onto it"
    )

    prefix = f"{TAG_PREFIX}-{uuid.uuid4().hex[:8]}"
    rendered_config.setdefault("facility", {})["prefix"] = prefix
    assert web_terminals["personas"][persona]["project"] == project, (
        "the prefix override changed the persona's project; it is only supposed "
        "to be the default for a persona that names none"
    )
    name = web_container_name(prefix, user)
    skill = "e2e-privsplit-probe"

    context_dir = preset_repo / "build" / "docker" / "web-terminal-context"
    assert (context_dir / "base.md").is_file(), f"no rendered overlay tree at {context_dir}"
    skill_dir = context_dir / user / "skills" / skill
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: e2e-privsplit-probe\ndescription: seeding probe\n---\n", encoding="utf-8"
    )

    volume = f"{TAG_PREFIX}-seed-{uuid.uuid4().hex[:8]}"
    # Still refuse to reclaim a name that is somehow taken -- `docker rm -f`
    # here would delete somebody's running terminal to make room for a test --
    # but with a run-unique prefix this is now a genuine "should never happen",
    # so it fails rather than skipping. A skip here used to hide the entire
    # class on any host running this preset.
    existing = _docker("ps", "-aq", "--filter", f"name=^{name}$", timeout=60)
    assert not existing.stdout.strip(), (
        f"a container named {name} already exists on this host; the name carries "
        "a per-run unique prefix, so this is a collision that should not be "
        "possible, and this test will not remove a container it did not create"
    )
    run = _docker(
        "run",
        "-d",
        "--name",
        name,
        "-v",
        f"{volume}:{CLAUDE_CONFIG_DIR}",
        "-e",
        f"CLAUDE_CONFIG_DIR={CLAUDE_CONFIG_DIR}",
        tag,
        "sleep",
        "infinity",
        timeout=120,
    )
    assert run.returncode == 0, f"docker run failed: {run.stderr}"
    try:
        # Wait for the entrypoint's startup maintenance to finish and hand over.
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if "dropping privileges" in _logs(name):
                break
            time.sleep(1.0)
        else:
            pytest.fail(f"seeding container never reached the privilege drop:\n{_logs(name)}")

        seed_user_containers(
            rendered_config,
            user=user,
            env=dict(os.environ),
            config_path=preset_repo / "build" / "config.yml",
        )
        yield SimpleNamespace(
            cid=name,
            project=project,
            root=f"/app/{project}",
            skills_dir=f"/app/{project}/build/.claude/skills",
            skill=skill,
        )
    finally:
        _docker("rm", "-f", name, timeout=120)
        _docker("volume", "rm", "-f", volume, timeout=60)


class TestMultiUserSeeding:
    """Where a seeded skill lands, and who owns what it lands in."""

    def test_seeded_skills_land_in_the_render_zone_root_owned(self, seeded_container):
        """The skills target is inside the render zone, so it is root-owned:
        a render-zone file the runtime user could rewrite would let one session
        edit the skills the next session loads."""
        stat = _exec(seeded_container.cid, "stat", "-c", "%U", seeded_container.skills_dir)
        assert stat.returncode == 0, f"no seeded skills dir: {stat.stderr}"
        assert stat.stdout.strip() == "root"

        owners = _exec(
            seeded_container.cid,
            "find",
            f"{seeded_container.skills_dir}/{seeded_container.skill}",
            "!",
            "-user",
            "root",
        )
        assert owners.returncode == 0, owners.stderr
        assert not owners.stdout.strip(), f"non-root paths under the seeded skill:\n{owners.stdout}"

    def test_the_claude_config_volume_is_owned_by_the_runtime_uid(self, seeded_container):
        """The other side of the same seed: the per-user volume is root-owned
        until its first chown, and the agent's user has to own it to use it.
        Asserted numerically — the seeding reads ``OSPREY_RUNTIME_UID`` out of
        the image rather than guessing, and 1000 is what that variable says."""
        stat = _exec(seeded_container.cid, "stat", "-c", "%u:%g", CLAUDE_CONFIG_DIR)
        assert stat.returncode == 0, stat.stderr
        assert stat.stdout.strip() == "1000:1000"

        claude_md = _exec(
            seeded_container.cid, "stat", "-c", "%u", f"{CLAUDE_CONFIG_DIR}/CLAUDE.md"
        )
        assert claude_md.returncode == 0, f"no seeded CLAUDE.md: {claude_md.stderr}"
        assert claude_md.stdout.strip() == "1000"

    def test_a_seeded_skill_is_visible_from_the_agents_project_cwd(self, seeded_container):
        """The regression guard, asked the way the CLI asks it.

        The launcher runs Claude Code with ``--setting-sources project`` after
        chdir-ing into ``<container_project_dir>/build``, so a skill anywhere
        else — ``$CLAUDE_CONFIG_DIR/skills``, or the repo root's ``.claude/`` —
        is loaded by nothing. Resolved from that cwd, as that user, relative:
        an absolute path would pass for a skill the agent cannot see.
        """
        seen = _docker(
            "exec",
            "-u",
            "1000",
            "-w",
            f"{seeded_container.root}/build",
            seeded_container.cid,
            "test",
            "-f",
            f".claude/skills/{seeded_container.skill}/SKILL.md",
            timeout=60,
        )
        assert seen.returncode == 0, (
            "the seeded skill is not visible from the agent's project cwd — it "
            "landed outside the scope --setting-sources project loads"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
