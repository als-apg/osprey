"""Per-user web-terminal lifecycle verbs: decommission, prune, nuke.

This module owns the destructive side of multi-user web-terminal deployments —
removing a user's container/volumes and the roster entry, artifacts, and routing
that referenced them. It is deliberately built from a small set of exact-named,
composable primitives (:func:`remove_container`, :func:`remove_volume`,
:func:`remove_image`, :func:`archive_volume`, :func:`confirm_destroy`) so that
every runtime-mutating call in this file is auditable at a glance: no
``prune``, no ``-a``/``--all``, no label glob, no bare ``down -v`` — every
removal argv names exactly one resource, or (for :func:`nuke_stack`'s container
teardown) is an explicit ``compose -p <project> down`` naming exactly one
project. The one place this module *reads* runtime state in bulk is
:func:`prune_users`'s orphan discovery (``ps -a`` / ``volume ls``), which is
read-only and never itself a removal argv; :func:`nuke_stack`'s pre-removal
``image inspect`` calls are the same kind of read-only check.

:func:`decommission_user`, :func:`prune_users`, and :func:`nuke_stack` are the
three verbs this module implements.

Volume-scoping boundary (applies to :func:`prune_users`'s discovery and to
:func:`nuke_stack`'s per-user volume teardown alike): orphan discovery
(:func:`_discover_orphan_containers`, :func:`_discover_orphan_volumes`) filters
the runtime listing by the compose-assigned owning-project label
(``com.docker.compose.project=<project>``, matching exactly what
:func:`osprey.deployment.runtime_helper.runtime_env`'s ``COMPOSE_PROJECT_NAME``
pins compose to stamp on every container/volume it creates) *before* any
name-based parsing — so two OSPREY deployments on the same host, even ones
whose project names happen to share a prefix or contain an underscore, can
never cross-match each other's containers or volumes. The printed plan + typed
confirmation every destructive verb here requires remains the operator's
backstop regardless: nothing is removed without the operator seeing the exact
resource names first.

Image-scoping boundary (applies only to :func:`nuke_stack`'s persona-image
teardown): unlike containers and volumes, image tags are host-global — two
OSPREY deployments that happen to use identically-named personas would build
identically-named ``<persona.project>-<persona>:local`` tags. There is no
label-filtered *listing* equivalent to ``ps -a``/``volume ls`` for this (image
tags aren't compose-project-scoped), so each candidate tag is instead
individually verified with ``image inspect`` against its own
``com.osprey.project`` label (stamped by the local-mode image build,
:func:`osprey.deployment.container_lifecycle.build_persona_images`) before it
is ever considered for removal — a missing or mismatched label means the tag
is skipped with a warning, not removed, since it may belong to a sibling
deployment.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from osprey.deployment.compose_generator import resolve_project_name, resolve_user_volume_names
from osprey.deployment.facility_config import normalize_facility_config
from osprey.deployment.runtime_helper import get_runtime_command, runtime_env
from osprey.deployment.web_terminals.artifacts import write_web_terminal_artifacts
from osprey.deployment.web_terminals.ports import normalize_users, resolve_personas
from osprey.utils.config import ConfigBuilder
from osprey.utils.config_writer import config_replace_list

_USERS_KEY_PATH = ["modules", "web_terminals", "users"]

# Directory (relative to the project cwd, i.e. wherever ``osprey deploy`` has
# already chdir'd) that --archive tarballs are written into.
_ARCHIVE_DIR_NAME = "web_terminal_archives"


def decommission_user(
    config_path: str | Path,
    user: str,
    *,
    archive: bool = False,
    purge: bool = False,
    assume_yes: bool = False,
) -> None:
    """Remove one user's web-terminal workspace.

    Order of operations, each a prerequisite for the next:

    1. Resolve the user against the roster; unknown user -> ``ValueError``, nothing
       touched.
    2. If ``archive``/``purge`` was requested, gate on a typed confirmation
       *before touching anything* — a decline (or a blank/non-matching response
       without ``assume_yes``) must be a true no-op: roster, ``config.yml``,
       container, and volumes are all left exactly as they were. This ordering
       matters because roster/artifact/container changes are only cheaply
       recoverable (a redeploy re-adds the user) while volume removal is not —
       declining must not leave the roster edited but the confirmation refused.
    3. Migrate the roster to explicit ``{name, index}`` entries (freezing current
       positional indices) and drop the target user, leaving their index as a gap
       so survivors' ports never shift. Write the result back to ``config.yml``,
       comment-preserving.
    4. Re-render the web-terminal artifacts from the updated config, so the
       deployed nginx route/compose service/landing card for the user disappear.
    5. Force-remove the user's exact-named container.
    6. Per ``archive``/``purge``, handle the user's two named volumes: retain
       (default, no confirmation needed), archive-then-remove, or purge (remove
       without archiving).

    Args:
        config_path: Path to the facility ``config.yml``.
        user: Web-terminal username to decommission.
        archive: Stream each of the user's volumes to a tarball before removing
            it. Mutually exclusive with ``purge`` (enforced by the CLI).
        purge: Remove each of the user's volumes without archiving. Mutually
            exclusive with ``archive`` (enforced by the CLI).
        assume_yes: Skip the typed confirmation gate for volume destruction.

    Raises:
        ValueError: If ``user`` is not present in ``modules.web_terminals.users``.
        RuntimeError: If volume destruction was requested but not confirmed.
    """
    config_path = Path(config_path)
    config = normalize_facility_config(ConfigBuilder(str(config_path)).raw_config)

    web_terminals = _as_dict(_as_dict(config.get("modules")).get("web_terminals"))
    migrated = normalize_users(web_terminals.get("users"))

    if not any(entry["name"] == user for entry in migrated):
        raise ValueError(
            f"User {user!r} is not present in modules.web_terminals.users; "
            "nothing was decommissioned."
        )

    remaining = [entry for entry in migrated if entry["name"] != user]

    # Confirm BEFORE any mutation for the destructive flags: declining must
    # leave the roster, config.yml, container, and volumes untouched.
    volumes: list[str] = []
    if archive or purge:
        claude_config_volume, agent_data_volume = resolve_user_volume_names(config, user)
        volumes = [claude_config_volume, agent_data_volume]
        prompt = (
            f"This will permanently destroy web-terminal volumes for user {user!r}: "
            f"{', '.join(volumes)}. Type '{user}' to confirm: "
        )
        if not confirm_destroy(prompt, assume_yes, expected=user):
            raise RuntimeError(f"Decommission of {user!r} aborted: confirmation did not match.")

    # Roster edit + artifact re-render happen before container/volume removal:
    # they are recoverable by re-running `osprey deploy up`, unlike volume removal.
    config_replace_list(config_path, _USERS_KEY_PATH, remaining)
    updated_config = normalize_facility_config(ConfigBuilder(str(config_path)).raw_config)
    write_web_terminal_artifacts(updated_config)

    runtime = get_runtime_command(config)[0]
    env = runtime_env(config)
    facility_prefix = _as_dict(config.get("facility")).get("prefix") or ""
    remove_container(runtime, f"{facility_prefix}-web-{user}", env=env)

    _apply_volume_policy(runtime, volumes, archive=archive, purge=purge, env=env)


# =============================================================================
# prune: remove resources for users no longer on the roster
# =============================================================================


def prune_users(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    archive: bool = False,
    purge: bool = False,
    assume_yes: bool = False,
) -> None:
    """Remove web-terminal resources for users no longer on the roster.

    Unlike :func:`decommission_user`, which targets one named user already known
    to the caller, ``prune`` targets *orphans*: containers/volumes that exist in
    the runtime but whose user is no longer in ``modules.web_terminals.users``
    (e.g. because the roster entry was hand-edited out of ``config.yml`` without
    running ``decommission`` first). Orphans are discovered by read-only listing
    (``ps -a`` / ``volume ls``) against the runtime, never assumed from config —
    the roster only tells us who is *not* an orphan.

    Order of operations:

    1. Load the config and resolve the current roster's user names.
    2. List all containers and volumes belonging to this project (via the
       compose-assigned ``com.docker.compose.project`` label) and split them
       into on-roster (left alone) and off-roster (orphans). This step never
       removes anything.
    3. Print a plan: which containers/volumes would be removed, and the disposal
       (retain/archive/purge) for each orphan's volumes. If ``dry_run``, stop
       here — nothing is touched.
    4. Otherwise, gate on a typed confirmation (unless ``assume_yes``) — pruning
       removes containers unconditionally (retain only protects volumes), so it
       is destructive even in the default retain mode.
    5. Force-remove each orphan's exact-named container, then apply the
       retain/archive/purge policy to each orphan's exact-named volumes.

    Args:
        config_path: Path to the facility ``config.yml``.
        dry_run: Print the plan and return without removing anything.
        archive: Stream each orphan's volumes to a tarball before removing them.
            Mutually exclusive with ``purge`` (enforced by the CLI).
        purge: Remove each orphan's volumes without archiving. Mutually
            exclusive with ``archive`` (enforced by the CLI).
        assume_yes: Skip the typed confirmation gate.

    Raises:
        RuntimeError: If pruning was requested but not confirmed.
    """
    config_path = Path(config_path)
    config = normalize_facility_config(ConfigBuilder(str(config_path)).raw_config)

    web_terminals = _as_dict(_as_dict(config.get("modules")).get("web_terminals"))
    roster_names = {entry["name"] for entry in normalize_users(web_terminals.get("users"))}

    runtime = get_runtime_command(config)[0]
    env = runtime_env(config)
    facility_prefix = _as_dict(config.get("facility")).get("prefix") or ""
    project = resolve_project_name(config)

    orphan_containers = _discover_orphan_containers(
        runtime, project, facility_prefix, roster_names, env=env
    )
    orphan_volumes = _discover_orphan_volumes(runtime, project, roster_names, env=env)
    orphan_users = sorted(set(orphan_containers) | set(orphan_volumes))

    if not orphan_users:
        print("prune: no off-roster web-terminal resources found; nothing to do.")
        return

    policy = "archive" if archive else "purge" if purge else "retain"
    print(f"prune: found {len(orphan_users)} off-roster user(s) (volume policy: {policy}):")
    for user in orphan_users:
        container = orphan_containers.get(user)
        if container:
            print(f"  - {user}: remove container {container!r}")
        for volume in orphan_volumes.get(user, []):
            print(f"      volume {volume!r}: {policy}")

    if dry_run:
        print("prune: dry-run — no resources were removed.")
        return

    prompt = (
        f"This will remove web-terminal resources for {len(orphan_users)} off-roster "
        f"user(s): {', '.join(orphan_users)}. Type 'prune' to confirm: "
    )
    if not confirm_destroy(prompt, assume_yes, expected="prune"):
        raise RuntimeError("Prune aborted: confirmation did not match.")

    for user in orphan_users:
        container = orphan_containers.get(user)
        if container:
            remove_container(runtime, container, env=env)
        _apply_volume_policy(
            runtime, orphan_volumes.get(user, []), archive=archive, purge=purge, env=env
        )


# =============================================================================
# nuke: full project teardown
# =============================================================================


def nuke_stack(config_path: str | Path, *, assume_yes: bool = False) -> None:
    """Tear down this project's entire web-terminal + service stack.

    The most destructive verb in this module: unlike :func:`decommission_user`
    (one named user) and :func:`prune_users` (only off-roster orphans), ``nuke``
    removes *everything* this project owns — every web-terminal, nginx, and base
    service container, every named volume belonging to this project's web
    terminals (roster *and* off-roster/orphaned alike), and every roster
    persona's locally-built image — with no retain option.

    Containers are torn down with a single project-scoped ``compose -p <project>
    down`` rather than by enumerating container names one at a time: the base
    services (postgres, dispatch worker, event dispatcher, etc.) have no
    config-derived name list the way web-terminal users do, but every container
    this project ever started shares the same compose project (pinned by
    :func:`osprey.deployment.runtime_helper.runtime_env`'s
    ``COMPOSE_PROJECT_NAME``), so an explicit ``-p <project>`` reaches exactly
    (and only) this project's containers without a wildcard, ``-a``/``--all``,
    or a container-name list that could drift out of sync with what is actually
    deployed. This is also why the volume set must include off-roster users:
    ``compose down`` removes an orphaned user's container right along with
    everyone else's (it doesn't consult the roster), so a volume set built from
    the roster alone would leave that user's two volumes behind — contradicting
    "tear down everything this project owns." The ``down`` never carries
    ``--volumes``/``-v``: volumes are removed
    afterwards, one exact name at a time, so the destroy argv for data is
    always provably exact-named (see the module docstring for the volume-
    scoping boundary this relies on).

    Image removal is deliberately the least trusting step: image tags
    (``<persona.project>-<persona>:local``) are host-global, not scoped to this
    project the way ``com.docker.compose.project``-labeled containers/volumes
    are, so a same-named tag could belong to an entirely different deployment.
    Every candidate tag is therefore ``image inspect``-verified against its own
    ``com.osprey.project`` label before removal (see the module docstring's
    image-scoping boundary); a missing tag is tolerated silently (nothing to
    remove), and a missing/mismatched label is skipped with a warning rather
    than removed.

    Order of operations:

    1. Load the config; resolve the project name and the roster's user names.
    2. Compute the volume teardown set: each roster user's two exact volume
       names (:func:`osprey.deployment.compose_generator.resolve_user_volume_names`),
       unioned with any off-roster/orphaned project volumes the runtime
       actually has (:func:`_discover_orphan_volumes`, the same read-only
       discovery :func:`prune_users` uses) — so a user who was decommissioned
       with the default retain policy, or hand-edited out of ``config.yml``,
       still gets their volumes torn down by ``nuke``.
    3. Compute the image teardown set: resolve the roster's personas leniently
       (:func:`osprey.deployment.web_terminals.ports.resolve_personas` with
       ``strict=False`` — a stale/bad persona reference never blocks ``nuke``),
       collect the distinct ``:local``-suffixed image tags (registry-mode
       images and lenient-degraded entries never resolve to a ``:local`` tag,
       so they are never candidates), then ``image inspect`` each candidate and
       keep only the ones whose ``com.osprey.project`` label matches this
       deployment's project name.
    4. Print a plan (the project-scoped container teardown, the full exact
       volume list — roster and orphan, and the label-verified image list, plus
       a warning line for any candidate image skipped on label mismatch), then
       gate on a typed confirmation (unless ``assume_yes``) *before touching
       anything* — same ordering lesson as :func:`decommission_user`: a decline
       must be a true no-op, so nothing runs until confirmation succeeds.
    5. Run the project-scoped ``compose down`` (containers only). A non-zero
       exit aborts immediately, before any volume or image is touched —
       proceeding would just fail again downstream (volume "in use") while
       masking the real error, and the CLI would report success on a failed
       teardown.
    6. Remove each exact-named volume from the teardown set.
    7. Remove each label-verified exact-named image tag from the teardown set.

    Args:
        config_path: Path to the facility ``config.yml``.
        assume_yes: Skip the typed confirmation gate — the non-interactive/
            scripted path.

    Raises:
        RuntimeError: If the teardown was not confirmed, or if ``compose down``
            exits non-zero.
    """
    config_path = Path(config_path)
    config = normalize_facility_config(ConfigBuilder(str(config_path)).raw_config)

    web_terminals = _as_dict(_as_dict(config.get("modules")).get("web_terminals"))
    roster_names = [entry["name"] for entry in normalize_users(web_terminals.get("users"))]

    runtime_cmd = get_runtime_command(config)
    runtime = runtime_cmd[0]
    env = runtime_env(config)
    project = resolve_project_name(config)
    facility_prefix = _as_dict(config.get("facility")).get("prefix") or ""

    volumes: list[str] = []
    for user in roster_names:
        volumes.extend(resolve_user_volume_names(config, user))

    # Off-roster volumes still belong to this project and must be swept too —
    # `compose down` doesn't distinguish roster from orphaned containers, so
    # neither should the volume teardown set.
    orphan_volumes = _discover_orphan_volumes(runtime, project, set(roster_names), env=env)
    for user_volumes in orphan_volumes.values():
        volumes.extend(user_volumes)

    # Roster personas' locally-built images. Lenient resolution (strict=False)
    # means a stale/bad persona reference degrades to the zero-migration
    # (non-":local") image rather than blocking nuke — see resolve_personas.
    registry_cfg = _as_dict(config.get("registry"))
    personas_resolved = resolve_personas(web_terminals, registry_cfg, facility_prefix, strict=False)
    candidate_images: list[str] = []
    for entry in personas_resolved:
        image = entry["image"]
        if image.endswith(":local") and image not in candidate_images:
            candidate_images.append(image)

    # Each candidate is individually label-verified against THIS deployment's
    # project name before it is ever considered for removal — image tags are
    # host-global, so a sibling deployment's identically-named tag must survive
    # even if it happens to match this roster's persona naming.
    images_to_remove: list[str] = []
    skipped_images: list[tuple[str, str | None]] = []
    for image in candidate_images:
        exists, label_value = _inspect_image_project_label(runtime, image, env=env)
        if not exists:
            continue  # absent image: tolerated, not part of the plan, not an error
        if label_value != project:
            skipped_images.append((image, label_value))
            continue
        images_to_remove.append(image)

    print(
        f"nuke: this will tear down project {project!r}'s entire web-terminal + "
        f"service stack — every container (project-scoped), {len(volumes)} "
        f"volume(s) ({len(roster_names)} roster user(s), "
        f"{len(orphan_volumes)} off-roster user(s)), and {len(images_to_remove)} "
        "persona image(s):"
    )
    print(f"  - containers: {' '.join(runtime_cmd)} -p {project} down")
    for volume in volumes:
        print(f"  - volume {volume!r}: removed permanently (no retain/archive)")
    for image in images_to_remove:
        print(f"  - image {image!r}: removed permanently (com.osprey.project verified)")
    for image, label_value in skipped_images:
        print(
            f"  - image {image!r}: SKIPPED — com.osprey.project label {label_value!r} "
            f"does not match this deployment's project {project!r}"
        )

    prompt = (
        f"This will PERMANENTLY tear down the entire web-terminal stack for "
        f"project {project!r}, including {len(volumes)} volume(s) and "
        f"{len(images_to_remove)} persona image(s). Type 'nuke' to confirm: "
    )
    if not confirm_destroy(prompt, assume_yes, expected="nuke"):
        raise RuntimeError("Nuke aborted: confirmation did not match.")

    result = _compose_down_project(runtime_cmd, project, env=env)
    if result.returncode != 0:
        print(f"nuke: 'compose down' failed (exit {result.returncode}): {result.stderr.strip()}")
        raise RuntimeError(
            f"Nuke aborted: 'compose down' for project {project!r} failed (exit "
            f"{result.returncode}); no volumes were removed."
        )

    for volume in volumes:
        remove_volume(runtime, volume, env=env)

    for image in images_to_remove:
        remove_image(runtime, image, env=env)


def _compose_down_project(
    runtime_cmd: list[str], project: str, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Project-scoped ``compose down`` — containers/networks only, never volumes.

    ``-p <project>`` is the only resource selector: compose resolves it against
    the ``com.docker.compose.project`` label every container this project ever
    started carries, so this reaches exactly this project's containers with no
    name enumeration and no wildcard. Never passed ``--volumes``/``-v`` — volume
    destruction is the caller's responsibility, one exact name at a time (see
    :func:`nuke_stack`).

    Does not raise on a non-zero exit and does not itself decide what a failure
    means — :func:`nuke_stack` is the caller that must inspect
    ``result.returncode`` and abort *before* touching any volume, since a volume
    removed out from under a container ``down`` failed to stop would just fail
    again downstream while masking the real error.

    Args:
        runtime_cmd: Full runtime+compose command, e.g. ``["docker", "compose"]``
            (the return value of
            :func:`osprey.deployment.runtime_helper.get_runtime_command`).
        project: Exact compose project name to tear down. Never a glob.
        env: Environment for the subprocess call.

    Returns:
        The completed subprocess. The caller must check ``returncode`` — this
        function does not raise on failure.
    """
    return subprocess.run(
        [*runtime_cmd, "-p", project, "down"], capture_output=True, text=True, env=env
    )


def _discover_orphan_containers(
    runtime: str,
    project: str,
    facility_prefix: str,
    roster_names: set[str],
    *,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Read-only: list containers and return ``{user: container_name}`` for orphans.

    Uses ``ps -a --filter label=com.docker.compose.project=<project> --format
    {{.Names}}`` — a listing, never a removal — so the ``-a`` here is not
    subject to the "no ``-a``/``--all``" removal-argv guardrail. The label
    filter scopes the listing to exactly this deployment's compose project
    (the same project name :func:`osprey.deployment.runtime_helper.runtime_env`
    pins as ``COMPOSE_PROJECT_NAME``), so a sibling OSPREY deployment on the
    same host — even one whose project name shares a prefix with this one —
    can never contribute a false match. Within that project-scoped listing,
    only containers matching ``<facility_prefix>-web-<user>`` are web-terminal
    containers; anything else the project owns (nginx, base services, ...) is
    ignored here.

    Args:
        runtime: Runtime binary, e.g. ``"docker"`` or ``"podman"``.
        project: This deployment's compose project name (the label value to
            filter on).
        facility_prefix: This facility's container-name prefix.
        roster_names: Current roster user names; matches are excluded.
        env: Environment for the subprocess call.

    Returns:
        Mapping of orphaned user name to their exact container name.
    """
    result = subprocess.run(
        [
            runtime,
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    prefix = f"{facility_prefix}-web-"
    orphans: dict[str, str] = {}
    for line in result.stdout.splitlines():
        name = line.strip()
        if not name.startswith(prefix):
            continue
        user = name[len(prefix) :]
        if user and user not in roster_names:
            orphans[user] = name
    return orphans


def _discover_orphan_volumes(
    runtime: str,
    project: str,
    roster_names: set[str],
    *,
    env: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Read-only: list volumes and return ``{user: [volume_names]}`` for orphans.

    Uses ``volume ls --filter label=com.docker.compose.project=<project>
    --format {{.Name}}`` — a listing, never a removal. The label filter scopes
    the listing to exactly this deployment's compose project (the same value
    :func:`osprey.deployment.runtime_helper.runtime_env` pins as
    ``COMPOSE_PROJECT_NAME``), so a sibling OSPREY deployment's volumes — even
    under a project name that shares a prefix or contains an underscore — can
    never be mistaken for this project's. Within that project-scoped listing,
    only volumes matching ``<project>_<user>-claude-config`` or
    ``<project>_<user>-agent-data`` (the same names
    :func:`osprey.deployment.compose_generator.resolve_user_volume_names`
    derives) are considered; anything else the project owns is ignored here.

    Args:
        runtime: Runtime binary, e.g. ``"docker"`` or ``"podman"``.
        project: This deployment's compose project name (the label value to
            filter on, and the volume namespace prefix).
        roster_names: Current roster user names; matches are excluded.
        env: Environment for the subprocess call.

    Returns:
        Mapping of orphaned user name to their exact volume names (0, 1, or 2
        entries depending on which volumes actually exist).
    """
    result = subprocess.run(
        [
            runtime,
            "volume",
            "ls",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.Name}}",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    prefix = f"{project}_"
    suffixes = ("-claude-config", "-agent-data")
    orphans: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        name = line.strip()
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        for suffix in suffixes:
            if rest.endswith(suffix):
                user = rest[: -len(suffix)]
                if user and user not in roster_names:
                    orphans.setdefault(user, []).append(name)
                break
    return orphans


def _inspect_image_project_label(
    runtime: str, image: str, *, env: dict[str, str] | None = None
) -> tuple[bool, str | None]:
    """Read-only: check whether *image* exists and, if so, its owning project.

    Uses ``image inspect <image> --format {{json .Config.Labels}}`` — a read,
    never a removal — and parses the label map itself in Python rather than
    indexing the label directly in the Go template, since ``index`` on a
    missing/nil label map is fragile across docker/podman versions; parsing a
    JSON object (or ``null``) defensively is not.

    Unlike :func:`_discover_orphan_containers`/:func:`_discover_orphan_volumes`,
    this has no compose-project-label-filtered *listing* to draw on: image tags
    are host-global, not scoped to a compose project, so each candidate tag
    :func:`nuke_stack` considers is checked individually rather than discovered
    in bulk.

    Args:
        runtime: Runtime binary, e.g. ``"docker"`` or ``"podman"``.
        image: Exact image tag to inspect. Never a glob or label selector.
        env: Environment for the subprocess call.

    Returns:
        ``(exists, label_value)``. ``exists`` is ``False`` when the tag isn't
        present on this host at all (a non-zero ``image inspect`` exit) —
        tolerated by the caller, not an error. When ``exists`` is ``True``,
        ``label_value`` is the image's ``com.osprey.project`` label value, or
        ``None`` if the image has no such label (or an unparseable/non-object
        labels blob).
    """
    result = subprocess.run(
        [runtime, "image", "inspect", image, "--format", "{{json .Config.Labels}}"],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        return False, None
    try:
        labels = json.loads(result.stdout.strip() or "null")
    except json.JSONDecodeError:
        labels = None
    if not isinstance(labels, dict):
        return True, None
    label_value = labels.get("com.osprey.project")
    return True, label_value if isinstance(label_value, str) else None


# =============================================================================
# Reusable primitives (shared with the prune/nuke verbs)
# =============================================================================


def remove_container(
    runtime: str, name: str, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Force-remove one exact-named container.

    Best-effort: a container that is already stopped or was never created is not
    an error (``rm -f`` already tolerates a missing container), since
    decommissioning a user without a running workspace should still succeed.

    Args:
        runtime: Runtime binary, e.g. ``"docker"`` or ``"podman"`` (the first
            element of :func:`osprey.deployment.runtime_helper.get_runtime_command`).
        name: Exact container name. Never a glob, label selector, or ``-a``.
        env: Environment for the subprocess call, e.g.
            :func:`osprey.deployment.runtime_helper.runtime_env`. Defaults to
            inheriting the parent process environment.

    Returns:
        The completed subprocess, for callers that want to inspect the outcome.
    """
    return subprocess.run([runtime, "rm", "-f", name], capture_output=True, text=True, env=env)


def remove_volume(
    runtime: str, name: str, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Remove one exact-named volume.

    Args:
        runtime: Runtime binary, e.g. ``"docker"`` or ``"podman"``.
        name: Exact volume name. Never a glob, label selector, or ``-a``.
        env: Environment for the subprocess call. Defaults to inheriting the
            parent process environment.

    Returns:
        The completed subprocess, for callers that want to inspect the outcome.
    """
    return subprocess.run([runtime, "volume", "rm", name], capture_output=True, text=True, env=env)


def remove_image(
    runtime: str, tag: str, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Remove one exact-named, label-verified image tag.

    Only ever called by :func:`nuke_stack`, and only for tags that
    :func:`_inspect_image_project_label` already confirmed both exist and carry
    a matching ``com.osprey.project`` label — this function itself does not
    re-check either condition, it just issues the single-tag removal. Never
    ``image prune``, never a label-selector or wildcard removal, and never more
    than one tag per invocation.

    Args:
        runtime: Runtime binary, e.g. ``"docker"`` or ``"podman"``.
        tag: Exact image tag, e.g. ``"<persona.project>-<persona>:local"``.
            Never a glob or label selector.
        env: Environment for the subprocess call. Defaults to inheriting the
            parent process environment.

    Returns:
        The completed subprocess, for callers that want to inspect the outcome.
    """
    return subprocess.run([runtime, "image", "rm", tag], capture_output=True, text=True, env=env)


def archive_volume(
    runtime: str, name: str, dest_dir: str | Path, *, env: dict[str, str] | None = None
) -> Path:
    """Stream one exact-named volume's contents to a gzip tarball.

    Runs a throwaway ``alpine`` container that mounts *name* read-only at
    ``/from`` and the host archive directory at ``/to``, and tars ``/from`` into
    ``/to/<name>.tar.gz``. The bind-mount-and-tar approach works identically for
    docker and podman, unlike (e.g.) ``podman volume export``, which has no
    docker equivalent.

    Uses the long-form ``--mount type=...,source=...,destination=...`` syntax
    rather than ``-v host:container[:opts]``: the short form's colon-separated
    fields break if *dest_dir* itself contains a colon, since a bare colon in a
    ``-v`` value is ambiguous with the host/container path separator. ``--mount``
    takes each field as a distinct ``key=value`` pair, so an embedded colon in a
    path is never misparsed.

    Raises rather than degrading on failure (``check=True``): callers that chain
    this with :func:`remove_volume` must not remove a volume whose archive did
    not actually succeed.

    Note:
        Under rootful docker, the ``alpine`` container runs as root, so the
        written tarball is root-owned on the host — the operator may need
        ``sudo`` to read or remove it later. Rootless podman does not have this
        problem (the container's root maps to the invoking user). This module
        does not chown the tarball; it is left as an operational note.

    Args:
        runtime: Runtime binary, e.g. ``"docker"`` or ``"podman"``.
        name: Exact volume name to archive. Never a glob or label selector.
        dest_dir: Host directory the tarball is written into; created if it
            doesn't already exist.
        env: Environment for the subprocess call. Defaults to inheriting the
            parent process environment.

    Returns:
        Path to the written tarball, ``<dest_dir>/<name>.tar.gz``.

    Raises:
        subprocess.CalledProcessError: If the archive container exits non-zero.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    tarball = dest / f"{name}.tar.gz"
    subprocess.run(
        [
            runtime,
            "run",
            "--rm",
            "--mount",
            f"type=volume,source={name},destination=/from,readonly",
            "--mount",
            f"type=bind,source={dest},destination=/to",
            "alpine",
            "tar",
            "czf",
            f"/to/{name}.tar.gz",
            "-C",
            "/from",
            ".",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return tarball


def _apply_volume_policy(
    runtime: str,
    volumes: list[str],
    *,
    archive: bool,
    purge: bool,
    env: dict[str, str] | None = None,
) -> None:
    """Apply the retain(default)/archive/purge policy to exact-named volumes.

    Shared by :func:`decommission_user` and :func:`prune_users` so both verbs
    agree on volume-destruction semantics. Confirmation is the caller's
    responsibility — by the time this runs, destruction (if any) is authorized.

    Args:
        runtime: Runtime binary, e.g. ``"docker"`` or ``"podman"``.
        volumes: Exact volume names to apply the policy to.
        archive: Stream each volume to a tarball (under
            ``<cwd>/<_ARCHIVE_DIR_NAME>``) before removing it. Mutually
            exclusive with ``purge``.
        purge: Remove each volume without archiving. Mutually exclusive with
            ``archive``.
        env: Environment for the subprocess calls.
    """
    if not (archive or purge):
        return  # retain (default): volumes are left in place

    if archive:
        archive_dir = Path.cwd() / _ARCHIVE_DIR_NAME
        for volume in volumes:
            archive_volume(runtime, volume, archive_dir, env=env)
            remove_volume(runtime, volume, env=env)
    else:  # purge
        for volume in volumes:
            remove_volume(runtime, volume, env=env)


def confirm_destroy(prompt: str, assume_yes: bool, *, expected: str) -> bool:
    """Typed-confirmation gate for irreversible data destruction.

    Requires the operator to type *expected* (e.g. the username) exactly — no
    generic "yes" alternative. For a two-volume irreversible destroy, an
    always-accepted "yes" invites muscle-memory confirmation and defeats the
    point of requiring the operator to type something specific to what's about
    to be destroyed.

    Args:
        prompt: Message shown to the operator before reading input.
        assume_yes: If True, skip the prompt entirely and confirm immediately —
            the non-interactive path used by scripted/CI callers.
        expected: The exact string (e.g. the username) required to confirm.

    Returns:
        True if destruction is confirmed; False if the operator declined, gave a
        blank response, or typed anything other than exactly *expected*.
    """
    if assume_yes:
        return True
    try:
        response = input(prompt).strip()
    except EOFError:
        response = ""
    return response == expected


def _as_dict(value: Any) -> dict[str, Any]:
    """Read a config section defensively: anything not a dict becomes empty."""
    return value if isinstance(value, dict) else {}
