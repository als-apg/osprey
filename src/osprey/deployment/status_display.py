"""Read-only inspection of a deployment repo: ``osprey status`` and ``osprey logs``.

Two verbs live here and neither changes anything.
:func:`show_repo_status` answers "what is this deployment doing, and does it
still match its source"; :func:`follow_logs` hands a ``compose logs``
invocation to the container runtime and gets out of the way. Both read
``build/`` — the config and compose files a build left there — and render
nothing, which is what makes them safe to run against a live stack.

Container state comes from the runtime's own ``ps``, never from compose. A
container whose service has since been deleted from the config is still
running, and compose — which only knows what the current files declare — is the
wrong witness for that.

Which containers belong to *this* deployment is decided by the
``com.osprey.repo-id`` label the render bakes in, so two checkouts of the same
repo on one host are told apart rather than merged. The label has an honest
limit that the display states rather than hides: it is applied when a container
is CREATED, so a stack whose containers were created by an OSPREY that did not
stamp it carries none, and this module can only match it by project name.

:func:`show_status` is the legacy project-scoped display behind
``osprey status``. It shares the query and table helpers below with the
repo-scoped verb, and goes away with the ``deploy`` group.
"""

import json
import subprocess
from pathlib import Path

from rich.table import Table

from osprey.deployment.compose_generator import (
    REPO_ID_LABEL,
    repo_identity,
    resolve_project_name,
    resolve_user_volume_names,
)
from osprey.deployment.runtime_helper import get_ps_command, get_runtime_command
from osprey.deployment.staleness import BUILD_DIRNAME, staleness_reasons
from osprey.deployment.web_terminals.naming import web_container_name
from osprey.utils.config import load_project_config
from osprey.utils.logger import get_logger

logger = get_logger("deployment.status")

#: Label naming the compose project a container belongs to. Spelled here as
#: every service template spells it (``osprey.project.name:`` in each
#: ``docker-compose.yml.j2``) because there is no shared constant to import —
#: this module only ever reads it, and a divergence would show up immediately as
#: containers filed under the wrong project.
PROJECT_LABEL = "osprey.project.name"


class _DefaultStyles:
    """Fallback styles when cli.styles is not available."""

    BOLD_PRIMARY = "bold"
    ACCENT = "cyan"
    SUCCESS = "green"
    PRIMARY = "bold"
    INFO = "cyan"
    DIM = "dim"
    ERROR = "red"
    WARNING = "yellow"


def _format_state(state, styles):
    """Format a container ``State`` value as a colored Rich status label.

    Shared by the project/other container tables (``_add_container_to_table``)
    and the per-user web terminal table (``_format_container_health``) so the
    state->label mapping can't drift between the two.

    :param state: Raw ``State`` value from the container runtime's ps output
    :type state: str
    :param styles: Style constants object with SUCCESS/ERROR/WARNING/DIM attributes
    :type styles: object
    :return: Rich markup string, e.g. ``"[green]● Running[/green]"``
    :rtype: str
    """
    if state == "running":
        return f"[{styles.SUCCESS}]● Running[/{styles.SUCCESS}]"
    elif state == "exited":
        return f"[{styles.ERROR}]● Stopped[/{styles.ERROR}]"
    elif state == "restarting":
        return f"[{styles.WARNING}]● Restarting[/{styles.WARNING}]"
    else:
        return f"[{styles.DIM}]● {state}[/{styles.DIM}]"


def _extract_web_terminal_user_names(users_raw):
    """Extract bare usernames from ``modules.web_terminals.users``.

    Entries may be bare strings (``"alice"``) or explicit object form
    (``{"name": "alice", "index": 0}``); both are handled defensively here so
    this read-only display never fails on a malformed roster.

    :param users_raw: Raw value of ``modules.web_terminals.users``
    :type users_raw: object
    :return: Bare usernames in declaration order; malformed entries are dropped
    :rtype: list[str]
    """
    if not isinstance(users_raw, list):
        return []
    names = []
    for entry in users_raw:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


# ---------------------------------------------------------------------------
# Reading the runtime
# ---------------------------------------------------------------------------


def _container_label(container, key):
    """Read one label off a runtime ``ps --format json`` record.

    Two shapes, because two runtimes: podman emits ``Labels`` as an object,
    docker as a comma-joined ``k=v`` string. ``None`` when the label is absent,
    and that absence is a real answer here rather than a parse failure — a
    container created by an OSPREY that did not stamp :data:`REPO_ID_LABEL`
    carries none.

    :param container: One decoded ``ps`` record
    :param key: Label key to read
    :return: The label's value, or ``None``
    """
    labels = container.get("Labels", {})
    if isinstance(labels, dict):
        value = labels.get(key)
        return value if isinstance(value, str) else None
    if isinstance(labels, str):
        for label in labels.split(","):
            if "=" in label:
                name, value = label.split("=", 1)
                if name.strip() == key:
                    return value.strip()
    return None


def _container_names(container):
    """Every name a ``ps`` record carries, without docker's leading ``/``."""
    names = container.get("Names", [])
    candidates = names if isinstance(names, list) else [names]
    return [str(name).lstrip("/") for name in candidates if name]


def _container_display_name(container):
    """The one name to show for a container; ``"unknown"`` when it has none."""
    names = _container_names(container)
    return names[0] if names else "unknown"


def _query_containers(config, console):
    """Every container on this host, or ``None`` when the runtime could not say.

    ``ps -a``, so a stopped container is part of the answer: "the database
    exited" and "the database was never started" are different problems with
    different fixes, and a listing of running containers alone cannot tell them
    apart.

    Failures are reported here and reported as failures — never as an empty
    list. A runtime that is not running would otherwise render as a tidy
    "nothing is deployed", which is the one wrong answer that looks right.

    :param config: Rendered config, for runtime selection; may be ``None``
    :param console: Where to report a failed query
    :return: Decoded ``ps`` records, or ``None`` when the query failed
    """
    try:
        result = subprocess.run(
            get_ps_command(config, all_containers=True), capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            console.print("\n[red]Error: Could not query container status[/red]")
            console.print(f"[dim]Command failed with return code {result.returncode}[/dim]\n")
            return None

        containers = []
        if result.stdout.strip():
            try:
                # JSON array (podman) first; docker emits one JSON object per line.
                containers = json.loads(result.stdout)
            except json.JSONDecodeError:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        containers.append(json.loads(line))
        return containers
    except subprocess.TimeoutExpired:
        console.print("\n[red]Error: Container query timed out[/red]\n")
        return None
    except json.JSONDecodeError as e:
        console.print("\n[red]Error: Could not parse container data[/red]")
        console.print(f"[dim]{e}[/dim]\n")
        return None
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]\n")
        return None


def _existing_volume_names(config, console, styles):
    """Names of every volume the runtime has, or ``None`` when it could not say.

    Read-only: volumes are listed, never created or removed.
    """
    try:
        runtime_bin = get_runtime_command(config)[0]
        result = subprocess.run(
            [runtime_bin, "volume", "ls", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return {line.strip() for line in result.stdout.splitlines() if line.strip()}
        console.print(f"[{styles.DIM}]Warning: could not query volumes for user status[/]")
    except subprocess.TimeoutExpired:
        console.print(f"[{styles.DIM}]Warning: volume query timed out[/]")
    except Exception as e:
        console.print(f"[{styles.DIM}]Warning: could not query volumes ({e})[/]")
    return None


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _create_status_table(styles):
    """Create a status table with consistent styling."""
    table = Table(show_header=True, header_style=styles.BOLD_PRIMARY)
    table.add_column("Service", style=styles.ACCENT, no_wrap=True)
    table.add_column("Project", style=styles.SUCCESS, no_wrap=True)
    table.add_column("Status", style=styles.PRIMARY)
    table.add_column("Ports", style=styles.INFO)
    table.add_column("Image", style=styles.DIM)
    return table


def _format_ports(container):
    """``host→container`` for every published port, or ``"-"``."""
    ports_raw = container.get("Ports", [])
    port_list = []
    if ports_raw:
        for port in ports_raw:
            if isinstance(port, dict):
                # podman ps: host_port/container_port. compose ps:
                # PublishedPort/TargetPort. docker ps --format json: published/target.
                published = (
                    port.get("host_port") or port.get("PublishedPort") or port.get("published", "")
                )
                target = (
                    port.get("container_port") or port.get("TargetPort") or port.get("target", "")
                )
                if published and target:
                    port_list.append(f"{published}→{target}")
    return ", ".join(port_list) if port_list else "-"


def _add_container_to_table(table, container, styles):
    """Add a container as a row in the status table."""
    project_name = _container_label(container, PROJECT_LABEL) or "unknown"
    if len(project_name) > 12:
        project_name = project_name[:9] + "..."

    image = container.get("Image", "unknown")
    if len(image) > 40:
        image = "..." + image[-37:]

    table.add_row(
        _container_display_name(container),
        project_name,
        _format_state(container.get("State", "unknown"), styles),
        _format_ports(container),
        image,
    )


def _container_table(containers, styles):
    """A populated status table for *containers*."""
    table = _create_status_table(styles)
    for container in containers:
        _add_container_to_table(table, container, styles)
    return table


def _show_web_terminal_users(config, all_containers, console, styles):
    """Print the per-user web terminal table, when this deployment has one.

    One row per rostered user: whether their container exists and what state it
    is in, and whether each of their two named volumes is present. A missing
    volume is the interesting case — it is where that user's work lives.

    Silently does nothing when web terminals are off or the roster is empty:
    there is no table to draw, and an empty one would suggest otherwise.
    """
    modules_cfg = config.get("modules", {})
    web_terminals_cfg = (
        modules_cfg.get("web_terminals", {}) if isinstance(modules_cfg, dict) else {}
    )
    if not isinstance(web_terminals_cfg, dict):
        web_terminals_cfg = {}
    user_names = _extract_web_terminal_user_names(web_terminals_cfg.get("users"))
    if not (web_terminals_cfg.get("enabled") and user_names):
        return

    facility_cfg = config.get("facility", {})
    if not isinstance(facility_cfg, dict):
        facility_cfg = {}
    facility_prefix = facility_cfg.get("prefix") or ""

    by_name = {}
    for container in all_containers:
        for name in _container_names(container):
            by_name.setdefault(name, container)

    existing_volumes = _existing_volume_names(config, console, styles)

    def _format_volume_status(volume_name):
        if existing_volumes is None:
            return f"[{styles.DIM}]? {volume_name} (unknown)[/{styles.DIM}]"
        if volume_name in existing_volumes:
            return f"[{styles.SUCCESS}]✓ {volume_name}[/{styles.SUCCESS}]"
        return f"[{styles.ERROR}]✗ {volume_name} (missing)[/{styles.ERROR}]"

    table = Table(show_header=True, header_style=styles.BOLD_PRIMARY)
    table.add_column("User", style=styles.ACCENT, no_wrap=True)
    table.add_column("Container", style=styles.PRIMARY)
    table.add_column("Claude Config Volume", style=styles.INFO)
    table.add_column("Agent Data Volume", style=styles.INFO)

    console.print("\n[bold]Web Terminal Users:[/bold]")
    for user in user_names:
        container = by_name.get(web_container_name(facility_prefix, user))
        claude_config_volume, agent_data_volume = resolve_user_volume_names(config, user)
        table.add_row(
            user,
            f"[{styles.DIM}]● Not created[/{styles.DIM}]"
            if container is None
            else _format_state(container.get("State", "unknown"), styles),
            _format_volume_status(claude_config_volume),
            _format_volume_status(agent_data_volume),
        )
    console.print(table)


def show_status(config_path, *, console=None, styles=None):
    """Show detailed status of services with formatted output.

    Uses direct container runtime ps to show actual container state, independent of compose files.
    Displays containers for this project separately from other running containers.

    The legacy project-scoped display, reached through ``osprey status``.
    :func:`show_repo_status` is the repo-scoped verb that replaces it, and the
    two share every helper above.

    :param config_path: Path to the configuration file
    :type config_path: str
    :param console: Rich Console instance for output (default: creates new Console)
    :type console: rich.console.Console, optional
    :param styles: Style constants object with attributes like SUCCESS, ERROR, etc.
        (default: uses _DefaultStyles with standard Rich markup)
    :type styles: object, optional
    """
    if console is None:
        from rich.console import Console

        console = Console()
    if styles is None:
        styles = _DefaultStyles

    config = load_project_config(config_path, wrap_errors=True)

    # Advisory render-provenance note: a stale project deploys an out-of-date
    # service set that looks perfectly healthy here, so status is the other
    # place (besides osprey up) the drift must be visible.
    for reason in staleness_reasons(Path(config_path).resolve().parent):
        console.print(f"[yellow]⚠ Project render is stale: {reason} — re-run osprey build[/yellow]")

    deployed_services = config.get("deployed_services", [])
    deployed_service_names = (
        [str(service) for service in deployed_services] if deployed_services else []
    )

    # The same resolver the deploy path pins as COMPOSE_PROJECT_NAME and stamps
    # into container labels. Status compares that label against this value, so
    # anything less than the shared resolver mis-classifies a project whose name
    # compose normalized (e.g. "Demo_Project!" is labelled "demo_project"), and
    # its own containers show up under "other Osprey containers".
    current_project = resolve_project_name(config)

    all_containers = _query_containers(config, console)
    if all_containers is None:
        return

    project_containers = []
    other_containers = []
    for container in all_containers:
        container_project = _container_label(container, PROJECT_LABEL) or "unknown"
        names_str = " ".join(_container_names(container)).lower()
        matches_service = any(
            service.split(".")[-1].lower() in names_str for service in deployed_service_names
        )
        if container_project == current_project or matches_service:
            project_containers.append(container)
        elif container_project != "unknown":
            other_containers.append(container)

    console.print("\n[bold]Service Status:[/bold]")

    if project_containers:
        console.print(_container_table(project_containers, styles))
    else:
        console.print(
            f"\n[warning]ℹ️  No services running for project '{current_project}'[/warning]"
        )
        if deployed_service_names:
            console.print(f"[dim]Configured services: {', '.join(deployed_service_names)}[/dim]")
        console.print("\n[info]Start services with:[/info]")
        console.print("  • [command]osprey up[/command]")

    if other_containers:
        console.print("\n[bold]Other Osprey Containers:[/bold]")
        console.print(_container_table(other_containers, styles))

    _show_web_terminal_users(config, all_containers, console, styles)

    console.print()


# ---------------------------------------------------------------------------
# osprey status — the repo-scoped report
# ---------------------------------------------------------------------------


def _print_build_section(repo_root, console, styles):
    """Print what ``build/`` is, and return the drift report the rest reads.

    Three facts, in the order an operator needs them: whether ``build/`` still
    matches ``profile.yml``, which framework version rendered it, and — when
    they disagree — what a start verb would do about it. The verdict comes from
    :func:`~osprey.deployment.staleness.check_drift`, the same call ``up`` and
    ``restart`` refuse on, so status and the start verbs can never disagree
    about whether this repo has drifted.

    The version line is printed on every run, not only on skew: "rendered by the
    version you are running" is worth stating, and a build that records no
    version at all is worth stating too.

    :return: The drift report, so callers can key later sections off its state
    """
    from osprey.cli.templates.manifest import get_framework_release_version
    from osprey.deployment.staleness import DriftState, build_manifest_path, check_drift

    report = check_drift(repo_root)

    state_style = {
        DriftState.CLEAN: styles.SUCCESS,
        DriftState.DRIFT: styles.WARNING,
        DriftState.UNRESOLVABLE: styles.WARNING,
        DriftState.NO_BUILD: styles.ERROR,
    }[report.state]
    console.print("\n[bold]Build:[/bold]")
    console.print(f"  [{state_style}]{report.status_line}[/{state_style}]")
    if report.refuses:
        console.print(
            f"  [{styles.DIM}]`osprey up` and `osprey restart` refuse until this is "
            f"resolved: {report.remedy}[/{styles.DIM}]"
        )

    installed = get_framework_release_version()
    if report.version_skew:
        console.print(f"  [{styles.WARNING}]{report.version_skew.message}[/{styles.WARNING}]")
    else:
        # The stamp itself, read from the same manifest `check_drift` read.
        # `version_skew` is only populated when the two versions DIFFER, so a
        # report without one cannot say whether they matched or whether the
        # build recorded no version to compare — and those are different facts.
        stamped = _stamped_version(build_manifest_path(repo_root))
        if stamped == installed:
            console.print(f"  [{styles.DIM}]rendered by osprey {installed} (installed)[/]")
        elif stamped:
            console.print(f"  [{styles.DIM}]rendered by osprey {stamped}[/]")
        else:
            console.print(
                f"  [{styles.DIM}]osprey {installed} installed; build/ records no "
                f"version that rendered it[/]"
            )
    return report


def _stamped_version(manifest_path):
    """The framework version recorded in a build manifest, or ``None``.

    Fails open on every error — an unreadable or version-less manifest is
    reported as "not recorded" rather than as a version, because status must
    never invent provenance.
    """
    try:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    creation = data.get("creation") if isinstance(data, dict) else None
    version = creation.get("osprey_version") if isinstance(creation, dict) else None
    return version if isinstance(version, str) and version else None


def _partition_by_checkout(containers, identity, project_name):
    """Sort every container on the host by which deployment it belongs to.

    Four buckets, and the split is by LABEL rather than by name because two
    checkouts of one repo on a single host share a project name and a volume
    namespace — the repo-id label is the only thing that tells them apart.

    * ``mine`` — :data:`REPO_ID_LABEL` equals this checkout's identity. The
      label wins outright, over a project name that disagrees: it is what
      ``osprey down``'s build-less fallback selects on, and a status that
      answered differently would describe a different set of containers than
      the verb acting on them.
    * ``unlabelled`` — the project name matches but there is no repo-id label at
      all — created by an OSPREY that did not stamp it. Shown as part of the
      deployment and flagged, because the label-driven paths cannot find them.
    * ``foreign`` — the project name matches and the repo-id does not. Another
      checkout of the same deployment, running on this host.
    * ``others`` — some other OSPREY project entirely.

    A container that is not labelled for this checkout AND carries no project
    label appears nowhere: it is either not OSPREY's or unattributable, and
    both are somebody else's business.

    :param containers: Decoded ``ps`` records
    :param identity: This checkout's :func:`repo_identity`
    :param project_name: This deployment's resolved compose project name
    :return: ``(mine, unlabelled, foreign, others)``
    """
    mine, unlabelled, foreign, others = [], [], [], []
    for container in containers:
        repo_id = _container_label(container, REPO_ID_LABEL)
        project = _container_label(container, PROJECT_LABEL)
        if repo_id == identity:
            mine.append(container)
        elif project is None:
            continue
        elif project != project_name:
            others.append(container)
        elif repo_id is None:
            unlabelled.append(container)
        else:
            foreign.append(container)
    return mine, unlabelled, foreign, others


def _print_containers_section(repo_root, config, console, styles):
    """Print what is running for this deployment, and what else is on the host.

    Returns every container the runtime reported, so the caller can answer the
    per-user web-terminal questions from the same single ``ps`` rather than
    querying twice.

    Without a build there is no rendered config to read the project name out of,
    so it falls back to the directory name — which is what compose itself would
    derive, and is stated as a derivation rather than presented as the recorded
    name. The label match does not depend on it either way.

    :return: All containers, or ``None`` when the runtime could not be queried
    """
    identity = repo_identity(repo_root)
    project_name = resolve_project_name(config) if config else Path(repo_root).name
    provenance = "project" if config else "project, from the directory name"

    console.print(f"\n[bold]Containers[/bold] [{styles.DIM}]({provenance} {project_name})[/]")
    all_containers = _query_containers(config, console)
    if all_containers is None:
        return None

    mine, unlabelled, foreign, others = _partition_by_checkout(
        all_containers, identity, project_name
    )

    if mine or unlabelled:
        console.print(_container_table([*mine, *unlabelled], styles))
    else:
        console.print(f"  [{styles.DIM}]Nothing is running for this deployment.[/]")
        console.print(f"  [{styles.DIM}]Start it with `osprey up -d`.[/]")

    if unlabelled:
        # Not a footnote for tidiness: these are exactly the containers
        # `osprey down`'s label fallback cannot see, so an operator who deletes
        # build/ and runs `down` would be told nothing was found while these
        # kept running.
        console.print(
            f"  [{styles.WARNING}]{len(unlabelled)} of the above matched by project name "
            f"only ({', '.join(_container_display_name(c) for c in unlabelled)}): they "
            f"carry no {REPO_ID_LABEL} label, so they were created before this repo "
            f"labelled its containers. `osprey down` finds them through build/'s compose "
            f"files, but not through the label fallback that stands in when build/ is "
            f"gone.[/{styles.WARNING}]"
        )

    if foreign:
        # NOT "this repo's verbs leave them alone". They carry the same compose
        # project name as this repo's own containers, because the name is
        # derived from the deployment rather than from the directory — so the
        # compose `down` this repo runs selects on that shared name and stops
        # them too. The label is what distinguishes them HERE; claiming it
        # isolates them would be exactly backwards for the operator who then
        # runs `osprey down`.
        console.print(
            f"\n  [{styles.WARNING}]{len(foreign)} container(s) named for project "
            f"'{project_name}' belong to a DIFFERENT checkout of it "
            f"({', '.join(sorted({_container_label(c, REPO_ID_LABEL) or '?' for c in foreign}))}); "
            f"this repo's identity is {identity}. They share this deployment's compose "
            f"project name, so `osprey down` run here stops them as well. Only the "
            f"{REPO_ID_LABEL} label tells the two checkouts apart.[/{styles.WARNING}]"
        )
        console.print(_container_table(foreign, styles))

    if others:
        console.print("\n[bold]Other OSPREY containers on this host:[/bold]")
        console.print(_container_table(others, styles))

    return all_containers


def _absolute_compose_files(compose_files, repo_root):
    """Anchor a build's compose-file list on the repo, for a reader that opens them.

    :func:`~osprey.deployment.container_lifecycle.as_built_compose_files` answers
    in paths relative to the repo root, which
    :func:`~osprey.deployment.compose_generator.compose_base_cmd` resolves for
    every ``-f`` argument it builds. Anything that opens those files ITSELF —
    the endpoint summary here — has no such anchor, and resolving them against
    the working directory instead reads nothing and reports a deployment with no
    endpoints at all.
    """
    root = Path(repo_root)
    return [
        str(path if path.is_absolute() else root / path)
        for path in (Path(name) for name in compose_files)
    ]


def _print_endpoints_section(config, compose_files, console, styles):
    """Print where this deployment's services are reachable, as ``build/`` declares it.

    Derived from the published ports in the rendered compose files — the same
    source the post-deploy summary uses — so what status prints and what ``up``
    printed cannot diverge. It says where a service *would* answer, not that it
    is answering: whether it is running is the table above.
    """
    from osprey.deployment.deploy_summary import format_endpoint_summary

    console.print()
    try:
        summary = format_endpoint_summary(config, compose_files)
    except Exception as exc:
        console.print(f"  [{styles.DIM}]Endpoints unavailable: {exc}[/]")
        return
    console.print(f"[bold]{summary.splitlines()[0]}[/bold]")
    for line in summary.splitlines()[1:]:
        console.print(line)
    console.print(f"  [{styles.DIM}](declared by build/ — not a check that anything answers)[/]")


def _auth_availability(secret_env, repo_root):
    """Where this deployment's provider credential can be found, if anywhere.

    Both places count, and which one it came from matters. The exported shell
    environment is what ``osprey chat`` authenticates from; the repo's ``.env``
    is the deployment's own secret store and what every container reads. A
    check against ``os.environ`` alone would report "NOT FOUND" for a
    perfectly-configured deployment whose key lives only in ``.env``.

    :return: ``(available, where)`` — *where* is a phrase, empty when not found
    """
    import os

    from osprey.utils.dotenv import parse_dotenv_file

    if os.environ.get(secret_env):
        return True, "exported in this shell"
    env_path = Path(repo_root) / ".env"
    if env_path.is_file():
        try:
            if parse_dotenv_file(env_path).get(secret_env):
                return True, f"set in {env_path.name}"
        except OSError:
            pass
    return False, ""


def _artifact_drift(repo_root, build_dir, config):
    """Whether the agent artifacts in ``build/`` match what a build would render.

    A dry run, which renders into a temporary directory and compares checksums:
    it writes nothing into the repo, which is what lets a read-only verb ask the
    question at all.

    The two override arguments are not optional decoration — they mirror the
    ones ``osprey build`` passes on its own three-zone regen. ``project_root``
    is the repo root (or, for a ``--runtime-root`` build, the path the render
    will run at inside a container), and the venv the artifacts launch from
    lives at ``build/.venv`` rather than beside the tree being compared. A dry
    run that omitted them would render a different interpreter path and a
    different project root into ``.mcp.json``, and report drift on a build that
    has none — the report would be wrong in the direction that sends an
    operator rebuilding for nothing.

    Which of the two applies is read back off the rendered config: its
    ``project_root`` is exactly what the build recorded, so a container-targeted
    render is recognised by that path not being this repo.

    :return: The dry-run result dict, or ``None`` when it could not be computed
    """
    import contextlib
    import io

    from osprey.cli.templates.manager import TemplateManager

    configured_root = str((config or {}).get("project_root") or repo_root)
    rendered_for_this_host = (
        Path(configured_root).expanduser().resolve() == Path(repo_root).expanduser().resolve()
    )
    # The renderer announces what it wrote ("✓ Created 38 Claude Code
    # integration file(s)") on its way through, and during a dry run it wrote
    # every one of them into a temporary directory that is deleted immediately
    # afterwards. Printed inside a status report that sentence claims this
    # command just rewrote the operator's repo, which is the opposite of what a
    # read-only verb did. Captured and kept for `-v` rather than discarded.
    chatter = io.StringIO()
    try:
        with contextlib.redirect_stdout(chatter):
            return TemplateManager().regenerate_claude_code(
                build_dir,
                dry_run=True,
                project_root_override=configured_root,
                runtime_venv_dir=build_dir if rendered_for_this_host else None,
            )
    except Exception as exc:
        # Broad on purpose: this is one section of a report, and a template that
        # cannot be rendered must not take the container table down with it.
        logger.debug("Artifact drift check failed: %s", exc)
        return None
    finally:
        if chatter.getvalue().strip():
            logger.debug("Artifact dry run said: %s", chatter.getvalue().strip())


def _print_agent_section(repo_root, build_dir, config, console, styles, *, show_agents):
    """Print how the agent in this deployment is configured, and whether it is in sync.

    Reports the provider, the environment block its settings carry, whether the
    credential those need can actually be found, the tier-to-model mapping, and
    whether the rendered agent artifacts still match the config they came from.

    The per-agent model assignments are behind *show_agents* rather than
    printed always: there are a dozen of them, and a status report whose longest
    section is a model table buries the two lines an operator came for.
    """
    import os

    from osprey.build.claude_code_resolver import AGENT_DEFAULT_TIERS, load_provider_spec

    console.print("\n[bold]Agent:[/bold]")

    claude_code = (config or {}).get("claude_code") or {}
    provider_name = claude_code.get("provider")
    if not provider_name:
        console.print(f"  [{styles.DIM}]provider: not configured[/]")
        console.print(
            f"  [{styles.DIM}]Set claude_code.provider in profile.yml and rebuild to "
            f"resolve models and provider environment automatically.[/]"
        )
    else:
        try:
            # The spec is read from the render, but its ``${VAR}`` placeholders
            # are expanded against the REPO root's ``.env`` — secrets are
            # durable state and never live in the disposable build zone, so a
            # custom provider's ``base_url: ${ARGO_PROD_URL}`` would otherwise
            # be reported as the literal placeholder. Same pairing as
            # ``_auth_availability`` below, which already looks for the
            # credential in ``repo_root/.env``.
            spec = load_provider_spec(build_dir, env_dir=repo_root)
        except Exception as exc:
            spec = None
            console.print(f"  [{styles.ERROR}]provider {provider_name}: {exc}[/{styles.ERROR}]")
        else:
            if spec is None:
                # The config names a provider and the resolver declined to
                # resolve one. Reporting the name alone would present a
                # configured provider where there is none in effect.
                console.print(
                    f"  [{styles.WARNING}]provider {provider_name}: named in the build, but "
                    f"could not be resolved from it[/{styles.WARNING}]"
                )

        if spec is not None:
            console.print(f"  provider: {spec.provider}")

            available, where = _auth_availability(spec.auth_secret_env, repo_root)
            if available:
                console.print(
                    f"  auth:     [{styles.SUCCESS}]✓[/{styles.SUCCESS}] ${spec.auth_secret_env} "
                    f"[{styles.DIM}]({where})[/]"
                )
            else:
                console.print(
                    f"  auth:     [{styles.ERROR}]✗[/{styles.ERROR}] ${spec.auth_secret_env} not "
                    f"found in this shell or in {repo_root / '.env'}"
                )

            if spec.env_block:
                console.print(f"  [{styles.DIM}]environment (rendered into settings.json):[/]")
                for key, value in spec.env_block.items():
                    console.print(f"    {key} = [{styles.DIM}]{value}[/]")

            console.print(f"  [{styles.DIM}]model tiers:[/]")
            model_overrides = claude_code.get("models") or {}
            for tier in ("haiku", "sonnet", "opus"):
                suffix = f" [{styles.DIM}](override)[/]" if tier in model_overrides else ""
                console.print(f"    {tier:8s} → {spec.tier_to_model.get(tier, '?')}{suffix}")

            if show_agents:
                console.print(f"  [{styles.DIM}]agent models:[/]")
                agent_overrides = claude_code.get("agent_models") or {}
                for agent_name, default_tier in sorted(AGENT_DEFAULT_TIERS.items()):
                    if agent_name in agent_overrides:
                        note = f"(override: {agent_overrides[agent_name]})"
                    else:
                        note = f"({default_tier})"
                    console.print(
                        f"    {agent_name:28s} → {spec.agent_model(agent_name)} "
                        f"[{styles.DIM}]{note}[/]"
                    )

            conflicts = spec.detect_env_conflicts(dict(os.environ))
            if conflicts:
                console.print(
                    f"  [{styles.WARNING}]⚠ this shell sets provider variables that "
                    f"disagree with the build:[/{styles.WARNING}]"
                )
                for var, (shell_val, settings_val) in sorted(conflicts.items()):
                    console.print(f"    {var}: shell {shell_val} / build {settings_val}")
                console.print(
                    f"  [{styles.DIM}]`osprey chat` resolves these in favour of the build.[/]"
                )

    result = _artifact_drift(repo_root, build_dir, config)
    if result is None:
        console.print(f"  [{styles.DIM}]artifacts: could not be checked[/]")
        return
    changed = result.get("changed") or []
    unchanged = result.get("unchanged") or []
    if changed:
        console.print(
            f"  [{styles.WARNING}]artifacts: {len(changed)} out of sync with build/config.yml — "
            f"run `osprey build`[/{styles.WARNING}]"
        )
        for path in changed:
            console.print(f"    [{styles.WARNING}]~[/{styles.WARNING}] {path}")
    else:
        console.print(
            f"  artifacts: [{styles.SUCCESS}]in sync[/{styles.SUCCESS}] "
            f"[{styles.DIM}]({len(unchanged)} files)[/]"
        )


def show_repo_status(repo_root, *, console=None, styles=None, show_agents=False):
    """Report what a deployment repo is doing and whether it still matches its source.

    The read-only counterpart to the start verbs, and deliberately the one place
    that answers all four questions at once:

    * **Build** — does ``build/`` still match ``profile.yml``, and which
      framework version rendered it. The same
      :func:`~osprey.deployment.staleness.check_drift` verdict ``up`` refuses
      on, so a refusal is never a surprise.
    * **Containers** — what the container runtime says is running, split by the
      ``com.osprey.repo-id`` label so another checkout's containers are named as
      another checkout's.
    * **Endpoints** — where this deployment's services are declared to answer.
    * **Agent** — the provider, whether its credential is findable, and whether
      the rendered agent artifacts still match the config.

    Nothing here starts, stops, renders or writes: every runtime call is a
    listing, and the artifact check is a dry run into a temporary directory. It
    is safe against a live stack, which is the point of a status verb.

    A repo with no build still reports: the build section says so and names the
    remedy, and the container table is still drawn — a stack whose ``build/``
    was deleted is exactly the case an operator most needs to see.

    :param repo_root: The deployment repo — the directory holding ``profile.yml``
    :param console: Rich Console to print to (default: a fresh one)
    :param styles: Style constants object (default: :class:`_DefaultStyles`)
    :param show_agents: Also print the per-agent model assignments
    """
    from osprey.deployment.container_lifecycle import as_built_compose_files, as_built_config_path

    if console is None:
        from rich.console import Console

        console = Console()
    if styles is None:
        styles = _DefaultStyles

    repo_root = Path(repo_root)
    build_dir = repo_root / BUILD_DIRNAME
    config_path = as_built_config_path(repo_root)
    config = (
        load_project_config(str(config_path), wrap_errors=True) if config_path.is_file() else None
    )

    console.print(f"\n[bold]{repo_root}[/bold]")

    _print_build_section(repo_root, console, styles)
    all_containers = _print_containers_section(repo_root, config, console, styles)

    if config is None:
        # Both remaining sections are read out of build/config.yml. Saying that
        # once beats letting two whole sections silently not appear, which reads
        # as a deployment that has no endpoints and no agent.
        console.print(
            f"\n[{styles.DIM}]Endpoints and agent are read from build/config.yml, "
            f"which this repo has none of yet.[/]"
        )
    else:
        compose_files = _absolute_compose_files(
            as_built_compose_files(config, repo_root), repo_root
        )
        if compose_files:
            _print_endpoints_section(config, compose_files, console, styles)
        if all_containers is not None:
            _show_web_terminal_users(config, all_containers, console, styles)
        _print_agent_section(repo_root, build_dir, config, console, styles, show_agents=show_agents)

    console.print()


# ---------------------------------------------------------------------------
# osprey logs — the compose wrapper
# ---------------------------------------------------------------------------


class NoComposeFilesError(RuntimeError):
    """Raised when a repo has no rendered compose files to read logs from."""


def logs_command(repo_root, *, service=None, follow=False, tail=None):
    """Build the ``compose logs`` argv and the environment to run it in.

    Split out from :func:`follow_logs` because everything interesting about this
    verb is in the argv, and the call that runs it replaces the process — which
    a test cannot observe.

    Both compose files go in the ``-f`` list: the services stack and, when the
    deployment has one, the web-terminal stack. The deploy path keeps those two
    invocations apart because their command *sequences* differ (the web stack
    pulls, the services stack may build), a constraint ``logs`` does not have —
    they are one compose project, and reading a project's logs from one
    invocation is what makes ``osprey logs`` show the whole deployment.

    ``COMPOSE_PROJECT_NAME`` is pinned through
    :func:`~osprey.deployment.runtime_helper.runtime_env`, the same way every
    other invocation pins it. Without it compose would derive a project name
    from the directory and address a project that has no containers, printing
    nothing at all and looking like a deployment that logs nothing.

    :param repo_root: The deployment repo
    :param service: One compose service name, or ``None`` for all of them
    :param follow: Pass ``-f``, so the runtime streams until interrupted
    :param tail: Pass ``--tail N``; ``None`` leaves compose's own default alone
    :return: ``(argv, env)``
    :raises NoComposeFilesError: When ``build/`` holds no compose files to read
    """
    import os

    from osprey.deployment.compose_generator import (
        COMPOSE_ENV_FILENAME,
        compose_base_cmd,
        compose_env_file_args,
    )
    from osprey.deployment.container_lifecycle import as_built_compose_files, as_built_config_path
    from osprey.deployment.runtime_helper import runtime_env, with_plain_progress
    from osprey.deployment.web_terminals.artifacts import web_compose_file

    repo_root = Path(repo_root)
    config_path = as_built_config_path(repo_root)
    if not config_path.is_file():
        raise NoComposeFilesError(
            f"No build found at {config_path.parent}. Run `osprey build` — `osprey logs` "
            "reads the compose files a build rendered."
        )

    config = load_project_config(str(config_path), wrap_errors=True)
    compose_files = list(as_built_compose_files(config, repo_root))
    web_compose = web_compose_file(repo_root)
    if web_compose.is_file():
        compose_files.append(str(web_compose))
    if not compose_files:
        raise NoComposeFilesError(
            f"No compose files in {config_path.parent}. Run `osprey build` to render them."
        )

    # The shared resolver answers `[]` for a missing `.env` and warns that
    # "services will start with default/empty environment variables" on its way
    # out — true for every OTHER caller, and false here, where nothing starts.
    # Short-circuiting to the same `[]` keeps the resolver's rule as the only
    # spelling of it and drops a sentence about starting services from a verb
    # that only reads them.
    env_file_args = (
        compose_env_file_args(repo_root) if (repo_root / COMPOSE_ENV_FILENAME).exists() else []
    )
    cmd = compose_base_cmd(
        with_plain_progress(get_runtime_command(config)),
        compose_files,
        repo_root,
        env_file_args,
    )
    cmd.append("logs")
    if follow:
        cmd.append("--follow")
    if tail is not None:
        cmd.extend(("--tail", str(tail)))
    if service:
        cmd.append(service)
    return cmd, runtime_env(config, os.environ.copy())


def follow_logs(repo_root, *, service=None, follow=False, tail=None):
    """Show this deployment's container logs, by handing over to the runtime.

    The process is *replaced* by the compose invocation rather than wrapping it
    in a subprocess, so what an operator gets is the runtime's own behaviour
    with nothing in the middle: Ctrl-C on a ``--follow`` reaches compose
    directly, ``osprey logs | head`` terminates the way any pipe does, and the
    exit code is compose's.

    Because of that this function does not return.

    :param repo_root: The deployment repo
    :param service: One compose service name, or ``None`` for all of them
    :param follow: Stream new output until interrupted
    :param tail: Show only the last N lines; ``None`` means compose's default
    :raises NoComposeFilesError: When ``build/`` holds no compose files to read
    """
    import os

    cmd, env = logs_command(repo_root, service=service, follow=follow, tail=tail)
    logger.debug("Running command:\n    %s", " ".join(cmd))
    os.execvpe(cmd[0], cmd, env)
