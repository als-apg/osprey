"""Advisory host-side hooks around the web-terminal compose reconcile.

Everything here is best-effort and non-fatal by design: rootless-podman
``loginctl`` linger, the post-up ``verify.sh`` smoke check, the nginx config
hot-reload, and the post-up reachability probe. A failure warns and returns;
it never fails the deploy. Called from
:func:`osprey.deployment.web_terminals.provision.deploy_up_web_terminals`.
"""

import getpass
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from osprey.deployment.runtime_helper import get_runtime_command
from osprey.utils.logger import get_logger

logger = get_logger("deployment.lifecycle")


def enable_linger(config: dict, run_env: dict[str, str]) -> None:
    """Enable rootless-podman linger so web-terminal containers survive logout.

    Rootless podman runs containers under the deploy user's ``systemd --user``
    session, which systemd-logind tears down (along with everything under it)
    the moment that user's last login session ends. ``loginctl enable-linger
    <user>`` asks logind to keep the session alive across logout and reboot
    instead, which is what makes a rootless-podman web-terminal deploy survive
    the operator closing their SSH session. Docker containers run under the
    docker daemon rather than a per-user systemd session, so there is nothing
    to enable there.

    This is a best-effort persistence step, not a deploy precondition: every
    way it can fail to apply (wrong runtime, no ``loginctl`` on ``PATH``, no
    systemd, no permission) is logged and swallowed rather than raised, so a
    host that can't support linger still completes its deploy.

    :param config: Raw deploy config, used only to detect podman vs. docker
        via :func:`get_runtime_command`.
    :param run_env: The ``COMPOSE_PROJECT_NAME``-pinned environment the caller
        already built via :func:`runtime_helper.runtime_env`; reused here so
        the ``loginctl`` subprocess sees the same ``PATH`` as the compose
        calls around it.
    """
    if get_runtime_command(config)[0] != "podman":
        return  # linger is a rootless-podman/systemd concept; docker has no analog

    if shutil.which("loginctl") is None:
        logger.warning("loginctl not found on PATH — skipping podman linger enable")
        return

    try:
        deploy_user = getpass.getuser()
    except (KeyError, OSError) as exc:
        # getuser() falls back to pwd.getpwuid(os.getuid()) when USER/LOGNAME
        # etc. are all unset, which raises KeyError (3.12 and earlier) or
        # OSError (3.13+) for a uid with no passwd entry -- e.g. an LDAP/NSS
        # user under a stripped-env systemd/cron context. Best-effort means
        # best-effort: give up on linger rather than aborting the deploy.
        logger.warning(f"Could not determine deploy user for linger: {exc}")
        return

    try:
        status = subprocess.run(
            ["loginctl", "show-user", deploy_user, "--property=Linger"],
            env=run_env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if status.returncode == 0 and status.stdout.strip() == "Linger=yes":
            logger.debug(f"Linger already enabled for {deploy_user} — nothing to do")
            return
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"Could not check linger status for {deploy_user}: {exc}")
        # Fall through -- a failed status check doesn't mean enabling would fail.

    try:
        enable = subprocess.run(
            ["loginctl", "enable-linger", deploy_user],
            env=run_env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if enable.returncode == 0:
            logger.info(f"Enabled systemd linger for {deploy_user} (podman persistence)")
        else:
            logger.warning(
                f"loginctl enable-linger {deploy_user} failed (exit {enable.returncode}): "
                f"{enable.stderr.strip()}"
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"Could not enable linger for {deploy_user}: {exc}")


def run_verify_script(project_root: str, run_env: dict[str, str]) -> None:
    """Best-effort, advisory post-up smoke check via the scaffolded ``scripts/verify.sh``.

    An ``osprey-build-deploy``-scaffolded project ships ``scripts/verify.sh``
    (see the ``osprey-build-deploy`` skill's ``templates/core/scripts/
    verify.sh``): a health-check script parameterized per-facility with a
    probe for each enabled module. Historically it was operator-run-by-hand
    only; this makes ``osprey deploy up`` run it automatically as the last
    step of the post-up hook, once ``compose up -d`` has already succeeded
    and containers are running, so an operator gets an immediate health
    signal without a separate manual step.

    Silently skipped (no log line at all) when ``<project_root>/scripts/
    verify.sh`` doesn't exist — an older project scaffolded before this file
    existed, or a non-``osprey-build-deploy`` project, must deploy exactly as
    before.

    The script's own convention (see its header) is to ALWAYS exit 0 —
    verification is advisory, never deploy-blocking — but this runs it via
    ``bash`` (rather than executing the path directly) and ignores whatever
    exit code it reports either way, so a site-customized copy that doesn't
    honor that convention still can never fail ``osprey deploy up``: this
    step runs after compose already reported success, so a nonzero exit is a
    signal to look closer, not evidence the deploy failed. Output streams
    straight to the operator's terminal (stdout/stderr are inherited, not
    captured) exactly like every other compose subprocess call in this
    module, so the health report appears live rather than being buffered and
    dumped at the end.

    :param project_root: The project root whose ``scripts/verify.sh`` (if
        any) to run; also the script's working directory, so its own
        ``./scripts/...``-relative assumptions resolve the same as when an
        operator runs it by hand from the project root.
    :param run_env: Environment for the subprocess — the same
        ``COMPOSE_PROJECT_NAME``-pinned env the compose calls in this module
        use, so any ``${COMPOSE_PROJECT_NAME}``-derived container name the
        script probes matches what compose actually named.
    """
    verify_path = Path(project_root) / "scripts" / "verify.sh"
    if not verify_path.is_file():
        return

    logger.key_info("Running post-up smoke check: %s", verify_path)
    try:
        result = subprocess.run(["bash", str(verify_path)], cwd=project_root, env=run_env)
    except OSError as exc:
        logger.warning("Could not run %s: %s", verify_path, exc)
        return

    if result.returncode == 0:
        logger.key_info("%s completed (exit 0)", verify_path)
    else:
        logger.warning(
            "%s exited %s -- advisory only, this does NOT fail the deploy. "
            "Review the output above.",
            verify_path,
            result.returncode,
        )


def reload_nginx_config(web_cmd: list[str], run_env: dict[str, str]) -> None:
    """Advisory nginx config hot-reload after the web stack's ``up -d``.

    Scoped to the web compose invocation (``exec -T nginx``) so no container
    name is guessed. Advisory like :func:`run_verify_script`: nginx validates
    the new config before applying it and keeps serving the old one on
    failure, and a reload that cannot run at all (container still starting)
    warns rather than failing a deploy that did reconcile.
    """
    reload_cmd = web_cmd + ["exec", "-T", "nginx", "nginx", "-s", "reload"]
    logger.info(f"Running command:\n    {' '.join(reload_cmd)}")
    result = subprocess.run(reload_cmd, env=run_env, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        logger.warning(
            "nginx config reload failed (rendered nginx.conf changes may not be "
            f"live until the nginx container restarts): {detail}"
        )


def warn_if_web_stack_unreachable(config: dict, attempts: int = 5, delay: float = 2.0) -> None:
    """Advisory post-up probe: is nginx actually reachable from THIS host?

    The whole web tier runs ``network_mode: host`` with loopback-only
    upstreams (the security baseline). On a Linux host that binds
    ``nginx_port`` on the machine itself — but on Docker Desktop
    (macOS/Windows) "host" is the hypervisor's Linux VM, and unless Docker
    Desktop's opt-in host-networking setting is enabled, nothing ever
    listens on the real host: ``compose up`` succeeds, every healthcheck
    passes (they probe from *inside* the VM), and the landing page is
    unreachable in any browser. This probe is the only signal that
    distinguishes that state from a working deploy.

    Advisory like :func:`run_verify_script`: the containers themselves are
    healthy, so an unreachable host port warns loudly (with the Docker
    Desktop remedy where that's the likely cause) but never fails a deploy
    that did, in fact, reconcile.
    """
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    nginx_port = web_terminals.get("nginx_port")
    if not isinstance(nginx_port, int):
        return
    url = f"http://127.0.0.1:{nginx_port}/"
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=3):
                return
        except urllib.error.HTTPError:
            return  # any HTTP response at all proves host-side reachability
        except OSError:
            if attempt + 1 < attempts:
                time.sleep(delay)
    hint = ""
    if sys.platform in ("darwin", "win32") and get_runtime_command(config)[0] == "docker":
        hint = (
            " On Docker Desktop the web stack's network_mode: host binds "
            "inside the Docker Linux VM, not on this machine, unless host "
            "networking is enabled: Docker Desktop -> Settings -> Resources "
            "-> Network -> 'Enable host networking', then Apply & Restart "
            "and re-run `osprey deploy up`."
        )
    logger.warning(
        f"Web-terminal containers are up, but {url} is not reachable from "
        f"this host after {attempts} probes -- the landing page will not "
        f"load in a browser.{hint}"
    )
