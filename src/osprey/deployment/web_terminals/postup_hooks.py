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
import time
import urllib.error
import urllib.request
from pathlib import Path

from osprey.cli.output import report_fact, warn_fact
from osprey.cli.phase_reporter import report_step as _report_step
from osprey.deployment.compose_generator import resolve_repo_root
from osprey.deployment.docker_desktop import (
    HOST_NETWORKING_REMEDY,
    host_networking_enabled,
    on_docker_desktop,
)
from osprey.deployment.runtime_helper import get_runtime_command
from osprey.deployment.subprocess_capture import run_captured
from osprey.deployment.web_terminals.ports import resolve_nginx_port
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
        logger.warning("loginctl not found on PATH. Skipping the podman linger enable.")
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
            logger.debug(f"Linger already enabled for {deploy_user}. There is nothing to do.")
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
            report_fact(logger, f"Enabled systemd linger for {deploy_user} (podman persistence)")
        else:
            logger.warning(
                f"loginctl enable-linger {deploy_user} failed (exit {enable.returncode}): "
                f"{enable.stderr.strip()}"
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"Could not enable linger for {deploy_user}: {exc}")


def run_verify_script(project_root: str, run_env: dict[str, str]) -> None:
    """Best-effort, advisory post-up smoke check via the scaffolded ``scripts/verify.sh``.

    A project built from a profile carrying ``project/scripts/verify.sh``
    ships that file as ``<project_root>/scripts/verify.sh`` (the profile's
    ``project/`` mirror copies it verbatim): a health-check script
    parameterized per-facility with a probe for each enabled module.
    ``osprey up`` runs it automatically as the last step of the post-up hook,
    once ``compose up -d`` has already succeeded and containers are running, so
    an operator gets an immediate health signal without a separate manual step.

    Silently skipped (no log line at all) when ``<project_root>/scripts/
    verify.sh`` doesn't exist — a profile that carries no such script must
    deploy without any mention of one.

    The script's own convention (see its header) is to ALWAYS exit 0 —
    verification is advisory, never deploy-blocking — but this runs it via
    ``bash`` (rather than executing the path directly) and ignores whatever
    exit code it reports either way, so a site-customized copy that doesn't
    honor that convention still can never fail ``osprey up``: this
    step runs after compose already reported success, so a nonzero exit is a
    signal to look closer, not evidence the deploy failed. The script's output
    is spooled rather than streamed, exactly like every other child on this
    path: the deploy reports it as one sub-step, and a non-zero exit names the
    spool file holding the whole health report. Under ``--verbose`` it streams
    to the terminal instead, like every other captured run.

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

    # No announcement before the run: the step line below reports the same
    # script WITH its exit code, so an "about to run it" line adds only latency.
    try:
        # check=False: the exit code is advisory (see above), so a site-
        # customized script that exits non-zero must not raise from here.
        # cwd is the project root and repo_root is where the output spools —
        # the same directory here, but they answer different questions.
        result = run_captured(
            ["bash", str(verify_path)],
            env=run_env,
            cwd=project_root,
            spool_name="verify-script",
            repo_root=project_root,
            check=False,
        )
    except OSError as exc:
        # Covers both halves of the call: the script itself failing to launch,
        # and the spool file it would have been captured into failing to open.
        logger.warning("Could not run %s or capture its output: %s", verify_path, exc)
        return

    _report_step(f"smoke check {verify_path.name}: exit {result.returncode}")
    if result.returncode != 0:
        # Read off the result, not the reporter: this hook has callers that run
        # it with no phase open, and they need the path just as much.
        # None only under --verbose, where the output already streamed past.
        spool = result.spool_path
        logger.warning(
            "%s exited %s -- advisory only, this does NOT fail the deploy. %s",
            verify_path,
            result.returncode,
            f"Its output: {spool}" if spool else "Review the output above.",
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
    logger.debug(f"Running command:\n    {' '.join(reload_cmd)}")
    result = subprocess.run(reload_cmd, env=run_env, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        logger.warning(
            "nginx config reload failed (rendered nginx.conf changes may not be "
            f"live until the nginx container restarts): {detail}"
        )


def _host_port_answers(url: str, attempts: int, delay: float) -> bool:
    """Poll ``url`` from this host; ``True`` as soon as anything answers.

    Any HTTP status counts — a 502 from nginx still proves the host can reach
    the listening socket, which is the only thing being tested here.
    """
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=3):
                return True
        except urllib.error.HTTPError:
            return True  # any HTTP response at all proves host-side reachability
        except OSError:
            if attempt + 1 < attempts:
                time.sleep(delay)
    return False


def warn_if_web_stack_unreachable(
    config: dict,
    attempts: int = 5,
    delay: float = 2.0,
    *,
    web_cmd: list[str] | None = None,
    run_env: dict[str, str] | None = None,
) -> None:
    """Advisory post-up probe: is nginx actually reachable from THIS host?

    The whole web tier runs ``network_mode: host`` with loopback-only
    upstreams (the security baseline). On a Linux host that binds
    ``nginx_port`` on the machine itself — but on Docker Desktop
    (macOS/Windows) "host" is the hypervisor's Linux VM, and the port reaches
    the real machine only through Docker Desktop's host-network forwarder:
    ``compose up`` succeeds, every healthcheck passes (they probe from
    *inside* the VM), and the landing page is unreachable in any browser.
    This probe is the only signal that distinguishes that state from a
    working deploy.

    TWO CAUSES, told apart by asking Docker Desktop. A dark host port on
    Docker Desktop is either a forwarder that is switched off, or a socket the
    running forwarder never saw. The second is real and common: the forwarder
    registers a VM-side socket by *watching* for the listen event, so a
    container that came back up under ``restart: unless-stopped`` while Docker
    Desktop was still starting (after an update, a reboot, or the Apply &
    restart that enabling the setting itself performs) opens its socket
    unobserved and stays invisible to the host indefinitely. Nothing about its
    compose definition changed, so every subsequent ``up -d`` leaves it
    ``Running``, reconciles nothing, and reports the same failure forever.
    Bouncing the container re-opens the socket while the forwarder is watching,
    which fixes it.

    :func:`~osprey.deployment.docker_desktop.host_networking_enabled` is what
    separates the two, so the bounce is no longer attempted blind: a definite
    "off" skips it (no restart can conjure a forwarder that is not running) and
    the warning names the checkbox as the cause. An unreadable setting keeps the
    old order, bouncing first and then naming the setting as a thing to check,
    because on a host that cannot be asked the repairable cause is still the
    likelier one. All of it is gated to Docker Desktop: on Linux the port is
    bound on the machine directly, so an unreachable one means something else
    entirely and a blind restart would be noise.

    Advisory like :func:`run_verify_script`: the containers themselves are
    healthy, so an unreachable host port warns loudly (with the Docker
    Desktop remedy where that's the likely cause) but never fails a deploy
    that did, in fact, reconcile. The warning goes out through
    :func:`~osprey.cli.output.warn_fact` rather than ``logger.warning``, which
    is not a style choice: the altitude gate drops raw WARNING records while a
    lifecycle reporter owns the terminal (see :mod:`osprey.cli.altitude`), and
    the root logger carries no other handler, so a warning emitted the raw way
    here reaches nobody at all.

    :param web_cmd: The web stack's compose argv (from
        :func:`provision.web_stack_compose_cmd`), used for the self-heal
        restart. ``None`` — the lifecycle callers that never had one — keeps
        the warn-only behaviour.
    :param run_env: Environment for that restart, the same
        ``COMPOSE_PROJECT_NAME``-pinned env the caller's other compose calls
        use.
    """
    # "This deployment has web terminals" is `enabled`, read exactly as
    # :func:`~osprey.deployment.container_lifecycle._web_terminals_enabled`
    # reads it — that function decides whether the web stack is deployed at
    # all, so anything probing that stack has to agree with it about whether
    # it exists. A rendered nginx port used to stand in for this and can no
    # longer: an unset `nginx_port` is a legal config whose nginx listens on
    # the gateway slot of the deployment's block all the same, so a gate keyed
    # on the literal would fall silent on exactly the deployments that lean on
    # the layout. A port key that is not a port has no address to probe.
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    if not web_terminals.get("enabled"):
        return
    try:
        nginx_port = resolve_nginx_port(config)
    except ValueError:
        return
    url = f"http://127.0.0.1:{nginx_port}/"
    if _host_port_answers(url, attempts, delay):
        return

    try:
        desktop = on_docker_desktop(config)
    except RuntimeError:
        # No runtime resolved on this host. An advisory probe must not sink
        # the deploy it is advising on, and without a runtime there is no
        # Docker Desktop to blame, so the generic warning below is the most
        # this host can honestly be told.
        desktop = False
    # Asked once, up front, because it decides two separate things below: whether
    # the self-heal bounce is worth attempting at all, and how confidently the
    # warning at the end is allowed to speak. ``None`` means this host could not
    # be asked, which is not the same as enabled and is not reported as either.
    forwarder = host_networking_enabled() if desktop else None

    # A bounce re-registers a socket the forwarder missed. It cannot conjure a
    # forwarder that is switched off, so a definite ``False`` skips it: the
    # restart would cost an operator ten seconds and then tell them the same
    # thing. ``None`` keeps the bounce, because an unknown host is exactly the
    # one where the far more common stale registration is still worth repairing.
    if desktop and web_cmd and forwarder is not False:
        restart_cmd = web_cmd + ["restart"]
        report_fact(
            logger,
            f"{url} is not reachable from this host yet. On Docker Desktop this is "
            "usually a stale host-network port registration; bouncing the web "
            f"stack to re-register it:\n    {' '.join(restart_cmd)}",
        )
        try:
            # Spooled, not inherited: the restart's compose chatter would bury
            # the warning above, which is the part the operator has to act on.
            # check=False keeps this advisory: a failed restart falls through
            # to the warning below, exactly as before.
            run_captured(
                restart_cmd,
                env=run_env,
                spool_name="compose-web-restart",
                repo_root=resolve_repo_root(config),
                check=False,
            )
        except OSError as exc:
            logger.warning("Could not restart the web stack: %s", exc)
        else:
            _report_step("bounced the web stack for host-port re-registration")
            if _host_port_answers(url, attempts, delay):
                # The bounce's payoff. Without it the step above is the last
                # word and never says whether the remediation worked.
                _report_step("web endpoint reachable")
                return

    summary = "the web terminals are running but not reachable from this host"
    if desktop and forwarder is False:
        # Docker Desktop was asked and said no. Nothing here is a guess, so the
        # copy states the cause and names the checkbox instead of hedging.
        detail = (
            f"every container is healthy and nginx is listening on port {nginx_port}, but it "
            "is listening inside the Docker Desktop Linux VM. Host networking is turned off "
            f"in Docker Desktop, so {url} never reaches this machine and the landing page "
            "will not load in a browser."
        )
        remedy = f"{HOST_NETWORKING_REMEDY}, and re-run `osprey up`"
    elif desktop:
        # Docker Desktop, but the setting could not be read. Same suspect, stated
        # as a thing to check rather than as a finding.
        bounced = " Restarting the stack did not re-register it." if web_cmd else ""
        detail = (
            f"{url} did not answer after {attempts} probes, so the landing page will not "
            "load in a browser. On Docker Desktop the web stack binds its port inside the "
            "Docker Linux VM and reaches this machine only through the host-network "
            f"forwarder, which is off unless host networking is enabled.{bounced}"
        )
        remedy = f"check that host networking is on: {HOST_NETWORKING_REMEDY}"
    else:
        # Not Docker Desktop, so the port is bound on the machine itself and this
        # is some other problem. Saying which one would be a guess.
        detail = (
            f"{url} did not answer after {attempts} probes, so the landing page will not "
            f"load in a browser, even though nginx reports itself healthy on port {nginx_port}."
        )
        remedy = None
    warn_fact(logger, summary, detail, remedy)
